#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
#------------------------------------------------------------------------------
# Name:  geo_utils.py
#
#   General purpose:
#       This is a small collection of useful functions for geo-processing in Python
#       For further information read the docs provided in the modules.
#
# Author:   Harald Kristen <haraldkristen at posteo dot at>
# Date: 22.09.2017
#-------------------------------------------------------------------------------
"""

from osgeo import gdal

def array2geotiff(e_min, e_max, n_min, n_max, e_res, n_res, epsg, fname, array, image_type=gdal.GDT_Float32):
    from osgeo import gdal, osr
    import numpy as np
    
    xsize = (e_max-e_min)/e_res
    ysize = (n_max-n_min)/n_res
    driver = gdal.GetDriverByName('GTiff')
    raster_path = fname
    outRaster = driver.Create(raster_path, int(xsize) ,int(ysize) , 1, image_type)
    outRaster.SetGeoTransform((e_min, e_res, 0, n_max, 0, -n_res))
    outband = outRaster.GetRasterBand(1)
    #outband.WriteArray(np.rot90(array[::-1], k = 3))
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(int(epsg))
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()
    outRaster = None
    outRasterSRS = None

def array2geotiff2(geo_transform, epsg, fname, array, image_type=gdal.GDT_Float32):
    from osgeo import gdal, osr
    import numpy as np
    cols, rows = array.shape
    driver = gdal.GetDriverByName('GTiff')
    raster_path = fname
    outRaster = driver.Create(raster_path, rows, cols , 1, image_type)
    outRaster.SetGeoTransform(geo_transform)
    outband = outRaster.GetRasterBand(1)
    #outband.WriteArray(np.rot90(array[::-1], k = 3))
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(int(epsg))
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()
    outRaster = None
    outRasterSRS = None

def write_geotiff(fname, data, geo_transform, projection, image_type=gdal.GDT_Float32):
    """
    Write a n-D NumpyArray as GeoTiff to the current workdirectory

    Args:
        fname (str): Name of the output file 
        data (ndarray): A n - D Numpy array -> multiple bands possible
        geo_transform (tuple): Returned value of gdal.Dataset.GetGeoTransform (coefficients for transforming between pixel/line (P,L) raster space, and projection coordinates (Xp,Yp) space. 
        projection (str): Projection definition string (Returned by gdal.Dataset.GetProjectionRef)  

    Returns:
         fname.tiff (GTIFF): The n-D NumpyArray written to the current work directory as Tiff
    """

    from osgeo import gdal
    driver = gdal.GetDriverByName('GTiff')
    options = [ 'INTERLEAVE=PIXEL' ]
    if len(data.shape) > 2:
        rows, cols, bands = data.shape        
        dataset = driver.Create(fname, cols, rows, bands, image_type, options)
        dataset.SetGeoTransform(geo_transform)
        dataset.SetProjection(projection)
        for i in range(bands):
            dataset.GetRasterBand( i + 1).WriteArray( data[:,:,i])
        # flush data to disk & close file
        dataset.FlushCache()
        dataset = None

    else:
        rows, cols = data.shape
        bands = 1
        dataset = driver.Create(fname, cols, rows, bands, image_type, options)
        dataset.SetGeoTransform(geo_transform)
        dataset.SetProjection(projection)
        band = dataset.GetRasterBand(1)
        band.WriteArray(data)
        # flush data to disk,  & set 0  = NODATA
        band.SetNoDataValue(0)
        band.FlushCache()        
        dataset = None  # Close the file

def clip_raster(vector_path, raster_path):
    """
    This functions clips a raster image with a shapefile as mask

    Args:
        vector_path (str): Path to a shapefile used as mask layer. It should contain a single polygon somewhere IN the raster extent
        raster_path (str): Path to the raster file to be clipped 

    Returns:
        clipped.tiff (GTiff): The clipped raster file saved to the current work directory

    Source: 
        https://mapbox.github.io/rasterio/topics/masking-by-shapefile.html?highlight=mask

    Example: 
        clip_raster("bbox_training_epsg3035.shp", "L4_FTY_020_061015\L4_fty_eur_20m_full01_100_fin01.tif")
    """
    import fiona
    import rasterio
    import rasterio.mask

    with fiona.open(vector_path, "r") as shapefile:
        features = [feature["geometry"] for feature in shapefile]

    with rasterio.open(raster_path) as src:
        out_image, out_transform = rasterio.mask.mask(src, features,
                                                      crop=True)
        out_meta = src.meta.copy()

    # Applying the features in the shapefile as a mask on the raster sets all pixels outside of the features to be zero.
    # Since crop=True in this example, the extent of the raster is also set to be the extent of the features in the shapefile.
    # We can then use the updated spatial transform and raster height and width to write the masked raster to a new file.

    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})
    with rasterio.open("clipped.tif", "w", **out_meta) as dest:
        dest.write(out_image)

    return

def create_raster_from_vector(vector_path, raster_path, raster_attribute, data_type = gdal.GDT_Int16, mask_path = None, rasterized_name = 'rasterized.tif'):
    """
    Rasterizes a shapefile with the same properties (pixel size, extent, projection) as a given raster file     

    Args:
        vector_path (str): Path to classification layer in ESRI Shapfile format 
        raster_path (str): Path to raster file (TIFF) to be classified with scikit-learn
        raster_attribute (str): Shapefile column/attribute that should be used for classification  

    Returns:
         rasterized.gtif(GTIFF): The rasterized shapefile written to the current work directory
    """

    from osgeo import gdal
    from osgeo import ogr
    import numpy as np

    # First we will open our raster image, to understand how we will want to rasterize our vector
    raster_ds = gdal.Open(raster_path)
    # Fetch number of rows and columns
    ncol = raster_ds.RasterXSize
    nrow = raster_ds.RasterYSize
    # Fetch projection and extent
    proj = raster_ds.GetProjectionRef()
    ext = raster_ds.GetGeoTransform()
    raster_ds = None
    # Create the raster dataset
    print(ncol,'x',nrow)
    print(ext)
    print(str(ext[0]) + ',' + str(ext[3] + ext[5]*nrow) + ',' + str(ext[0]+ext[1]*ncol) + ',' + str(ext[3]))
    print(proj)
    #gdal.UseException()
    #try:
     # load the vector layer
    vector = ogr.Open(vector_path)
    layer = vector.GetLayer(0)
    print(layer.GetExtent())
    memory_driver = gdal.GetDriverByName('GTiff')
    out_raster_ds = memory_driver.Create(rasterized_name, ncol, nrow, 1, data_type)
    # Set the ROI image's projection and extent to our input raster's projection and extent
    out_raster_ds.SetProjection(proj)
    out_raster_ds.SetGeoTransform(ext)
    # Fill our output band with the 0 blank, no class label, value
    b = out_raster_ds.GetRasterBand(1)
    b.Fill(0)
    # Rasterize the shapefile layer to our new dataset
    #print('yeah')
    status = gdal.RasterizeLayer(out_raster_ds,  # output to our new dataset
                                 [1],  # output to our new dataset's first band
                                 layer,  # the layer to be rasterized
                                 None, None,  # don't worry about transformations since we're in same projection
                                 [0],  # burn value 0
                                 ['ALL_TOUCHED=FALSE',  # rasterize all pixels touched by polygons
                                  'ATTRIBUTE=' + raster_attribute#, 
#                                  'INITVALUES=[0]',
#                                  'NODATA=0', 'XRES=' + str(ext[1]), 'YRES='+ str(abs(ext[5])),
#                                  'OUTPUTTYPE=GDT_Int16',
#                                  'OUTPUTBOUNDS=['+ str(ext[0]) + ',' + str(ext[3] + ext[5]*nrow) + ',' + str(ext[0]+ext[1]*ncol) + ',' + #str(ext[3]) +']'
                                 ]
                                 # put raster values according to the 'id' field values
                                )
    #print('yeah')
    if mask_path != None:
        mask_ds = gdal.Open(mask_path)
        b = out_raster_ds.GetRasterBand(1)
        mb = mask_ds.GetRasterBand(1)
        bData = b.ReadAsArray()
        mData = mb.ReadAsArray()
        b.WriteArray(np.multiply(bData, mData))
        b.SetNoDataValue(0)
        b.FlushCache()
    #print('yeah')
    # Close dataset
    out_raster_ds = None
    if status != 0:
        print("I don't think it worked...")
    else:
        print("Success")
    #except RuntimeError:
    #    print("RUN TIME ERROR OCCURED")
    #    pass
    #gdal.DontUseExceptions()
    
    return

def stack_geotiff(tifs, outtif='stacked.tif', options=['INTERLEAVE=PIXEL'], remove=True):
    """
    Merges multiple TIFFs to one Multilayer TIFF, all files need to have the same extent, projection, pixel size    

    Args:
        remove (boolean): Define if the result should be removed after the operation
        tifs (list): A list with the path to the TIFF files

    Returns:
        stacked.tif (TIFF): Writes a Multiband TIFF to the current work directory + returns it as GDAL Dataset
    """
    import os
    from osgeo import gdal
    outvrt = '/vsimem/stacked.vrt'  # /vsimem is special in-memory virtual "directory"
    bv_options = gdal.BuildVRTOptions(options, separate=True)
    outds = gdal.BuildVRT(outvrt, tifs, options=bv_options)
    tr_options = gdal.TranslateOptions(creationOptions=options)
    outds = gdal.Translate(outtif, outds, options=tr_options)
    if remove:
        for i in tifs:
            os.remove(i)
    print('\n The stacked tiff can be found in the working directory under ' + outtif)
    return

def mask(mask_path, keep, replacement, array = 'foo', data_path = 'bar'):
    """
    Replaces values with desired replacement in data at index of mask.
    
    Args:
        mask_path: path to mask tiff
        replacement: Value for Replacement
        keep: Value from mask where values should be kept. Only one value.
        data_path: data in which values shuold be replaced, Tiff file
        array: data in which values shuold be replaced, numpy array
    
    Returns:
        data: Array with replaced values
    """    
    import numpy as np
    from osgeo import gdal
    mask_set = gdal.Open(mask_path)
    mask_array = mask_set.ReadAsArray()

    if data_path == True:
        data_set = gdal.Open(data_path)
        data = data_set.ReadAsArray()
    else:
        data = array
    
    itemindex = np.where(mask_array != keep)
    for (index_row, index_column) in zip(itemindex[0], itemindex[1]):
        data[index_row, index_column] = replacement
    
    return data