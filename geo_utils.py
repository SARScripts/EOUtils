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

def write_geotiff(fname, data, geo_transform, projection):
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
    if len(data.shape) > 2:
        rows, cols, bands = data.shape
        dataset = driver.Create(fname, cols, rows, bands, gdal.GDT_Float32)
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
        dataset = driver.Create(fname, cols, rows, bands, gdal.GDT_Float32)
        dataset.SetGeoTransform(geo_transform)
        dataset.SetProjection(projection)
        band = dataset.GetRasterBand(1)
        band.WriteArray(data)
        # flush data to disk,  & set 0  = NODATA
        band.FlushCache()
        band.SetNoDataValue(0)
        dataset = None  # Close the file

def average_tiff_file(inFile, outFile, remove=False):
    
    from osgeo import gdal
    import numpy as np
    import os
    
    
    raster_dataset = gdal.Open(inFile, gdal.GA_ReadOnly)
        
    nrows = raster_dataset.RasterYSize
    ncols = raster_dataset.RasterXSize
    
    avgband = np.zeros([nrows,ncols],dtype=np.float32)
    for b in range(1, raster_dataset.RasterCount + 1):
        band = raster_dataset.GetRasterBand(b)
        avgband = avgband + band.ReadAsArray()

    print(b)
    avgband = avgband / b
    memory_driver = gdal.GetDriverByName('GTiff')
    
    out_raster_ds = memory_driver.Create(outFile, ncols, nrows, 1, gdal.GDT_Float32)
    out_raster_ds.SetProjection(raster_dataset.GetProjection())
    out_raster_ds.SetGeoTransform(raster_dataset.GetGeoTransform())
    
    oband = out_raster_ds.GetRasterBand(1)
    oband.WriteArray(avgband)
    oband.FlushCache()
    oband.SetNoDataValue(0)

    memory_driver  = None  # Close the file
    raster_dataset = None        
    
    if remove:
        os.remove(inFile)
        
def rotate_tiff_file(inFile, outFile):
    
    from osgeo import gdal
    import numpy as np
    
    raster = gdal.Open(inFile)
    band = np.flipud(np.rot90(raster.GetRasterBand(1).ReadAsArray()))
        
    nrows = raster.RasterYSize
    ncols = raster.RasterXSize

    memory_driver = gdal.GetDriverByName('GTiff')
    
    out_raster_ds = memory_driver.Create(outFile, ncols, nrows, 1, gdal.GDT_Float32)
    out_raster_ds.SetProjection(raster.GetProjection())
    out_raster_ds.SetGeoTransform(raster.GetGeoTransform())
    
    oband = out_raster_ds.GetRasterBand(1)
    oband.WriteArray(band)
    oband.FlushCache()
    oband.SetNoDataValue(0)

    memory_driver = None  # Close the file
    raster = None        
        
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

def create_raster_from_vector(vector_path, raster_path, raster_attribute, fname):
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
    memory_driver = gdal.GetDriverByName('GTiff')
    out_raster_ds = memory_driver.Create(fname, ncol, nrow, 1, gdal.GDT_Float32)
    # Set the ROI image's projection and extent to our input raster's projection and extent
    out_raster_ds.SetProjection(proj)
    out_raster_ds.SetGeoTransform(ext)
    # Fill our output band with the 0 blank, no class label, value
    b = out_raster_ds.GetRasterBand(1)
    b.Fill(0)
    # load the vector layer
    vector = ogr.Open(vector_path)
    layer = vector.GetLayer(0)
    # Rasterize the shapefile layer to our new dataset
    status = gdal.RasterizeLayer(out_raster_ds,  # output to our new dataset
                                 [1],  # output to our new dataset's first band
                                 layer,  # the layer to be rasterized
                                 None, None,  # don't worry about transformations since we're in same projection
                                 [0],  # burn value 0
                                 ['ALL_TOUCHED=TRUE',  # rasterize all pixels touched by polygons
                                  'ATTRIBUTE=' + raster_attribute]
                                 # put raster values according to the 'id' field values
                                 )
    # Close dataset
    out_raster_ds = None
    if status != 0:
        print("I don't think it worked...")
    else:
        print("Success")

    return


def create_raster_from_vector2(vector_path, ncol, nrow, proj, ext, raster_attribute):
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
    import os
    
    # Create the raster dataset
    memory_driver = gdal.GetDriverByName('GTiff')
    out_raster_ds = memory_driver.Create('rasterized.gtif', ncol, nrow, 1, gdal.GDT_Float32)
    # Set the ROI image's projection and extent to our input raster's projection and extent
    out_raster_ds.SetProjection(proj)
    out_raster_ds.SetGeoTransform(ext)
    # Fill our output band with the 0 blank, no class label, value
    b = out_raster_ds.GetRasterBand(1)
    b.Fill(0)
    
    # load the vector layer
    #cwd = os.getcwd()
    #os.chdir(os.path.dirname(vector_path))
    #print(os.path.basename(vector_path))
    vector = ogr.Open(vector_path)
    #os.chdir(cwd)
    layer = vector.GetLayer(0)
    # Rasterize the shapefile layer to our new dataset
    status = gdal.RasterizeLayer(out_raster_ds,  # output to our new dataset
                                 [1],  # output to our new dataset's first band
                                 layer,  # the layer to be rasterized
                                 None, None,  # don't worry about transformations since we're in same projection
                                 [0],  # burn value 0
                                 ['ALL_TOUCHED=TRUE',  # rasterize all pixels touched by polygons
                                  'ATTRIBUTE=' + raster_attribute]
                                 # put raster values according to the 'id' field values
                                 )
    # Close dataset
    out_raster_ds = None
    if status != 0:
        print("I don't think it worked...")
    else:
        print("Success")

    return

def create_raster_from_vector_clip(ivector_path, oraster_path, ncol, nrow, ext, proj, attribute):
    
    import os
    from osgeo import gdal, ogr

    vector = ogr.GetDriverByName('ESRI Shapefile').Open(ivector_path,0)
    layer = vector.GetLayer(0)
    extv = layer.GetExtent()

    out_raster_ds = gdal.GetDriverByName('GTiff').Create('output.tif', ncol, nrow, 1, gdal.GDT_Float32)
    # Set the ROI image's projection and extent to our input raster's projection and extent
    out_raster_ds.SetProjection(proj)
    out_raster_ds.SetGeoTransform([extv[0],ext[1],0,extv[3],0,ext[5]])
    # Fill our output band with the 0 blank, no class label, value
    b = out_raster_ds.GetRasterBand(1)
    b.Fill(0)
    status = gdal.RasterizeLayer(out_raster_ds,             # output to our new dataset
                                 [1],                       # output to our new dataset's first band
                                 layer,                     # the layer to be rasterized
                                 None, None,                # don't worry about transformations since we're in same projection
                                 [0],                       # burn value 0
                                 ['ALL_TOUCHED=TRUE',       # rasterize all pixels touched by polygons
                                  'ATTRIBUTE=' + attribute] # put raster values according to the 'id' field values
                                 )
    
    print(status)
    # Close dataset
    out_raster_ds = None
    
    x0 = ext[0]
    xn = x0 + ext[1] * (ncol)
    yn = ext[3]
    y0 = yn + ext[5] * (nrow)
    
    bbox = [x0,y0,xn,yn]
    ds = gdal.Warp(oraster_path, 'output.tif', format = 'GTiff', outputBounds = bbox)
    #print(ds)
    ds = None

    os.remove('output.tif')

def stack_geotiff(tifs, outtif='stacked.tif', remove=True):
    """
    Merges multiple TIFFs to one Multilayer TIFF, all files need to have the same extent, projection, pixel size    

    Args:
        remove (boolean): Define if the result should be removed after the operation
        tifs (list): A list with the path to the TIFF files
        outtif (file): file containing the stacked output

    Returns:
        stacked.tif (TIFF): Writes a Multiband TIFF to the current work directory + returns it as GDAL Dataset (compatibility)
    """
    import os
    from osgeo import gdal
    outvrt = '/vsimem/stacked.vrt'  # /vsimem is special in-memory virtual "directory"
    #outtif = 'stacked.tif'

    outds = gdal.BuildVRT(outvrt, tifs, separate=True)
    outds = gdal.Translate(outtif, outds)
    if remove:
        for i in tifs:
            os.remove(i)
    print('\n The stacked tiff can be found in ' + outtif)
    return outtif

    
    
    

