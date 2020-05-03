"""
#------------------------------------------------------------------------------
# Name:  eig_wcs.py
#
#   General purpose:
#       Collection of input/output + eigencomputation exploting rasdaman server
#		 queries to the WCS coverages
#
# Author:   Fernando Vicente <fvicente at dares dot tech>
# Date: 27.09.2017
#-------------------------------------------------------------------------------
"""



from wcps_rasdaman import *
import datetime
from geo_utils import stack_geotiff,write_geotiff
import numpy as np
from numpy import linalg as LA
import os, shutil

# import library for WCS request handling
from owslib.wcs import WebCoverageService


def print_coverages():
    # import library for WCS request handling
    from owslib.wcs import WebCoverageService

    # Select WCS server and service version
    wcs = WebCoverageService('http://saocompute.eurac.edu/sincohmap/rasdaman/ows?', version='2.0.1')
    print('Coverages at saocompute.eurac.edu/sincohmap/rasdaman')
    print('')
    # show all data sets available on the WCS server
    for coverage_name in wcs.contents.keys():
            print('\t' + coverage_name)
            
    print('')
    print('')
    
def loadIntensityStack(mSet,pol='',iniDate=0,endDate=0, verbose_query=True):
    
    # Select polarimetric channel to query
    
    if pol == 'VH':
        pQueryTag = '.Coherence_VH'
    elif pol == 'HV':
        pQueryTag = '.Coherence_HV'
    elif pol == 'HH':
        pQueryTag = '.Coherence_HH'
    elif pol == 'VV':
        pQueryTag = '.Coherence_VV'
    else:
        pQueryTag = ''
    
    # Select WCS server and service version
    wcs = WebCoverageService('http://saocompute.eurac.edu/sincohmap/rasdaman/ows?', version='2.0.1')
    mContents = wcs.contents[mSet]
    
    indxN  = getIndexLabelFromWCScontents('N',mContents) 
    indxE  = getIndexLabelFromWCScontents('E',mContents) 
    mDates = getTimeAxis(mSet, mContents, indxN, indxE)
    
    if iniDate == 0:
        iniDate = mDates[0]
    if endDate == 0:
        iniDate = mDates[-1]
    
    fdates = [dd for dd in mDates if (dd>=iniDate and dd<=endDate)]
    #print(fdates)

    if len(fdates) == 0:
        print('No images found within the supplied temporal range [' + str(iniDate) + ', ' + str(endDate) + ']')
    else:
        print(str(len(fdates)) + ' images found within the supplied temporal range [' + str(iniDate) + ', ' + str(endDate) + ']')
    

    files = []
    k = 0
    for mm in fdates:
        date0M = mm.strftime('%Y-%m-%d')
        
        k = k + 1
        
        print(date0M)
            
        subset = 'date(\"' + date0M +'\")'
        query = 'for c in ( ' + mSet + ' ) return encode (c'+pQueryTag+'[' + subset + '], "tiff")'
    
        if verbose_query: print(query)
    
        subset_intensity = wcps_rasdaman(query, ip = 'saocompute.eurac.edu/sincohmap',verbose=verbose_query)
        files.append(subset_intensity)
        
    raster_path = stack_geotiff(files,remove=True) #the raster image to be classififed with scikit-learn
    return(raster_path)
    
    
def loadDiagonal(mSet,verbose_query=True):
    
    # Select WCS server and service version
    wcs = WebCoverageService('http://saocompute.eurac.edu/sincohmap/rasdaman/ows?', version='2.0.1')
    mContents = wcs.contents[mSet]
    
    indxMaster = getIndexLabelFromWCScontents('master_date',mContents)
    indxSlave  = getIndexLabelFromWCScontents('slave_date',mContents) 
    indxN = getIndexLabelFromWCScontents('N',mContents) 
    indxE  = getIndexLabelFromWCScontents('E',mContents) 
    
    masterDates = getMasterTimeAxis(mSet, mContents,indxSlave, indxN, indxE)
    slaveDates  = getSlaveTimeAxis(mSet, mContents,indxMaster, indxN, indxE)
    
    files = []
    k = 0
    for mm in masterDates:
        date0M = mm.strftime('%Y-%m-%d')
        date0S = slaveDates[k].strftime('%Y-%m-%d')
        k = k + 1
        
        print(date0M)
        print(date0S)
    
        subset = 'master_date(\"' + date0M +'\"), slave_date(\"' + date0S  +'\")'
        query = 'for c in ( ' + mSet + ' ) return encode (c[' + subset + '], "tiff")'
    
        if verbose_query: print(query)
    
        subset_coherence = wcps_rasdaman(query, ip = 'saocompute.eurac.edu/sincohmap',verbose=verbose_query)
        files.append(subset_coherence)
        
    raster_path = stack_geotiff(files) #the raster image to be classififed with scikit-learn
    return(raster_path)
        
def compute_eigenvalues(coh):
    
    [mm,ss]       = coh[:,:,0,0].shape
    [nrows,ncols] = coh[0,0].shape

    eigenValues = np.ndarray([nrows,ncols,mm],dtype=float)
    
    for ii in range(nrows):

        if not ii % 100:
            print('Row ' + str(ii) + ' / ' + str(nrows))

        for jj in range(ncols):
            
            cc  = coh[:,:,ii,jj]
            w,v = LA.eig(cc)
            eigenValues[ii,jj,:] = np.flipud(np.sort(np.abs(w)))
        
    return eigenValues

def compute_eigenvalues_dual(coh_vv,coh_vh):
    
    [mm,ss]       = coh_vv[:,:,0,0].shape
    [nrows,ncols] = coh_vv[0,0].shape

    eigenValues_vv = np.ndarray([nrows,ncols,mm],dtype=float)
    eigenValues_vh = np.ndarray([nrows,ncols,mm],dtype=float)
    
    for ii in range(nrows):

        if not ii % 100:
            print('Row ' + str(ii) + ' / ' + str(nrows))

        for jj in range(ncols):
            
            cc  = coh_vv[:,:,ii,jj]
            w,v = LA.eig(cc)
            eigenValues_vv[ii,jj,:] = np.flipud(np.sort(np.abs(w)))
            
            cc  = coh_vh[:,:,ii,jj]
            w,v = LA.eig(cc)
            eigenValues_vh[ii,jj,:] = np.flipud(np.sort(np.abs(w)))

        
    return {'eigVV':eigenValues_vv,'eigVH':eigenValues_vh}


def loadSqData(mSet, date0, dateN,verbose_query=True):

    # Select WCS server and service version
    wcs = WebCoverageService('http://saocompute.eurac.edu/sincohmap/rasdaman/ows?', version='2.0.1')
    mContents = wcs.contents[mSet]
    
    indxMaster = getIndexLabelFromWCScontents('master_date',mContents)
    indxSlave  = getIndexLabelFromWCScontents('slave_date',mContents) 
    indxN = getIndexLabelFromWCScontents('N',mContents) 
    indxE  = getIndexLabelFromWCScontents('E',mContents) 
    
    masterDates = getMasterTimeAxis(mSet, mContents,indxSlave, indxN, indxE)
    slaveDates  = getSlaveTimeAxis(mSet, mContents,indxMaster, indxN, indxE)
    
    date0 = datetime.datetime.strptime(date0, '%Y-%m-%d')
    dateN = datetime.datetime.strptime(dateN, '%Y-%m-%d')
    
    date0M = [dd for dd in masterDates if dd >= date0]
    date0MSTR = date0M[0]
    date0M = date0M[0].strftime('%Y-%m-%d')
    print('Master initial date: ' + date0M)
    dateNM = [dd for dd in masterDates if dd <= dateN] 
    dateNM = dateNM[-1].strftime('%Y-%m-%d')
    print('Master last date:    ' + dateNM)
    
    date0S = [dd for dd in slaveDates if dd > date0MSTR] 
    date0S = date0S[0].strftime('%Y-%m-%d')
    print('Slave initial date: ' + date0S)
    dateNS = [dd for dd in slaveDates if dd <= dateN]
    dateNS = dateNS[-1].strftime('%Y-%m-%d')
    print('Slave last date:    ' + dateNS)
    
    
    subset = 'master_date(\"' + date0M + '\":\"' + dateNM +'\"), slave_date(\"' + date0S + '\":\"' + dateNS +'\")'
    query = 'for c in ( ' + mSet + ' ) return encode (c[' + subset + '], "netcdf")'
    
    print('loadSqData query:')
    print(query)
    
    ncdata = wcps_rasdaman(query, ip = 'saocompute.eurac.edu/sincohmap',verbose=verbose_query)
    
    return ncdata

def eigenSequence(mSet, odir, iniDate, endDate,rotateOutput = 0, verbose_query=True,x0=0,xN=0,y0=0,yN=0):

    dateTAG = iniDate + '_' + endDate
    loadCoh = loadSqData_sequential(mSet,iniDate,endDate, x0=x0, xN=xN, y0=y0, yN=yN)

    geoInfo = getGeoInfo(mSet,verbose_query=False, x0=x0, xN=xN, y0=y0, yN=yN)
    geo_transf = geoInfo['GeoTransform']
    geo_proj   = geoInfo['Projection']


    # Compute eigenvalues
    eig = compute_eigenvalues_dual(loadCoh['COH_VV'],loadCoh['COH_VH'])

    eigenValues_vv = eig['eigVV']
    eigenValues_vh = eig['eigVH']

    eigenValuesVV_file = os.path.join(odir,mSet+'_'+ dateTAG +'_eigValuesVV.tiff')
    eigenValuesVH_file = os.path.join(odir,mSet+'_'+ dateTAG +'_eigValuesVH.tiff')

    if rotateOutput:
        write_geotiff(eigenValuesVV_file, np.flipud(np.rot90(eigenValues_vv)), geo_transf, geo_proj)
        write_geotiff(eigenValuesVH_file, np.flipud(np.rot90(eigenValues_vh)), geo_transf, geo_proj)
    else:
        write_geotiff(eigenValuesVV_file, eigenValues_vv, geo_transf, geo_proj)
        write_geotiff(eigenValuesVH_file, eigenValues_vh, geo_transf, geo_proj)

def loadSqData_sequential(mSet, date0, dateN,verbose_query=True,x0=0,xN=0,y0=0,yN=0):

    if x0!=0 and xN!=0 and y0!=0 and yN!=0:
        DO_CROP = 1
    else:
        DO_CROP = 0
    
    # Select WCS server and service version
    wcs = WebCoverageService('http://saocompute.eurac.edu/sincohmap/rasdaman/ows?', version='2.0.1')
    mContents = wcs.contents[mSet]
    
    indxMaster = getIndexLabelFromWCScontents('master_date',mContents)
    indxSlave  = getIndexLabelFromWCScontents('slave_date',mContents) 
    indxN = getIndexLabelFromWCScontents('N',mContents) 
    indxE  = getIndexLabelFromWCScontents('E',mContents) 
    
    masterDates = getMasterTimeAxis(mSet, mContents,indxSlave, indxN, indxE)
    slaveDates  = getSlaveTimeAxis(mSet, mContents,indxMaster, indxN, indxE)
    
    date0 = datetime.datetime.strptime(date0, '%Y-%m-%d')
    dateN = datetime.datetime.strptime(dateN, '%Y-%m-%d')
    
    date0M = [dd for dd in masterDates if dd >= date0]
    date0MSTR = date0M[0]
    date0M = date0M[0].strftime('%Y-%m-%d')
    print('Master initial date: ' + date0M)
    dateNM = [dd for dd in masterDates if dd <= dateN] 
    dateNM = dateNM[-1].strftime('%Y-%m-%d')
    print('Master last date:    ' + dateNM)
    
    vMdates = [dd for dd in masterDates if dd >= date0 and dd <= dateN]
    
    print(vMdates)
    
    date0S = [dd for dd in slaveDates if dd > date0MSTR] 
    date0S = date0S[0].strftime('%Y-%m-%d')
    print('Slave initial date: ' + date0S)
    dateNS = [dd for dd in slaveDates if dd <= dateN]
    dateNS = dateNS[-1].strftime('%Y-%m-%d')
    print('Slave last date:    ' + dateNS)
    
    vSdates = [dd for dd in masterDates if dd > date0MSTR and dd <= dateN]
    
    print(vSdates)
    
    nMdates = len(vMdates)
    nSdates = len(vSdates)
    
    refGeoInfo = getGeoInfo(mSet,x0=x0,xN=xN,y0=y0,yN=yN)
    
    ncols = refGeoInfo['RasterYSize'] # inversion due to the way netcdf allocates data. ¿?
    nrows = refGeoInfo['RasterXSize']
    
    print(ncols)
    print(nrows)
    
    COH_VV = np.ndarray([nMdates, nMdates, nrows, ncols],dtype=float)
    #COH_VV = np.ndarray([nMdates, nMdates, ncols, nrows],dtype=float)
    COH_VH = np.ndarray([nMdates, nMdates, nrows, ncols],dtype=float)
    #COH_VH = np.ndarray([nMdates, nMdates, ncols, nrows],dtype=float)
    
    for mm,mDate in enumerate(vMdates):
        
        mdIndex = mm
        
        COH_VV[mdIndex,mdIndex] = 1.0
        COH_VH[mdIndex,mdIndex] = 1.0
        mDate = mDate.strftime('%Y-%m-%d')
        
        if mm >= nMdates:
            continue
            
        vSdates = vMdates[mm+1:]
        for ss,sDate in enumerate(vSdates):
            
            sdIndex = ss + mm + 1
            
            sDate = sDate.strftime('%Y-%m-%d')
    
            #subset = 'master_date(\"' + mDate +'\"), slave_date(\"' + sDate + '\")'
            if DO_CROP:
                subset = 'E(' + str(x0) + ':' + str(xN) + '), N(' + str(y0) + ':' + str(yN) + \
                            '), master_date(\"' + mDate +'\"), slave_date(\"' + sDate  +'\")'
            else:
                subset = 'master_date(\"' + mDate +'\"), slave_date(\"' + sDate  +'\")'
        
            query = 'for c in ( ' + mSet + ' ) return encode (c[' + subset + '], "netcdf")'
            
            print('loadSqData_sequential query:')
            print(query)
    
            ncdata = wcps_rasdaman(query, ip = 'saocompute.eurac.edu/sincohmap',verbose=verbose_query)
            
            vv = ncdata['Coherence_VV'].values.astype(np.float)
            vh = ncdata['Coherence_VH'].values.astype(np.float)
            
            COH_VV[mdIndex,sdIndex,:,:] = vv
            COH_VH[mdIndex,sdIndex,:,:] = vh
            COH_VV[sdIndex,mdIndex,:,:] = vv
            COH_VH[sdIndex,mdIndex,:,:] = vh
            
    return {'COH_VV':COH_VV,'COH_VH':COH_VH}
    
def months_between(date1,date2):
    if date1>date2:
        date1,date2=date2,date1
    m1=date1.year*12+date1.month
    m2=date2.year*12+date2.month
    months=m2-m1
    if date1.day>date2.day:
        months-=1
    elif date1.day==date2.day:
        seconds1=date1.hour*3600+date1.minute+date1.second
        seconds2=date2.hour*3600+date2.minute+date2.second
        if seconds1>seconds2:
            months-=1
    return months

def getDatesImStack(mSet):
    wcs = WebCoverageService('http://saocompute.eurac.edu/sincohmap/rasdaman/ows?', version='2.0.1')
    mContents = wcs.contents[mSet]

    indxN = getIndexLabelFromWCScontents('N',mContents)
    indxE = getIndexLabelFromWCScontents('E',mContents) 

    dates = getTimeAxis(mSet, mContents,indxN,indxE,verbose_query=False)
    
    return(dates)

def getTimeAxis(mSet, mContents,indxN,indxE,verbose_query=True):
    
    subset = 'E(' + str(mContents.grid.origin[indxE]) + '), N(' + str(mContents.grid.origin[indxN]) + ')'
    query = 'for c in ( ' + mSet + ' ) return encode (c[' + subset + '], "netcdf")'
    
    if verbose_query: print(query)
    
    ncdata = wcps_rasdaman(query, ip = 'saocompute.eurac.edu/sincohmap', verbose=verbose_query)
    dates = []
    for dd in ncdata.coords['date'].values:
        dates.append(datetime.datetime.fromtimestamp(int(dd)))#.strftime('%Y-%m-%d'))
    return dates

def getMasterTimeAxis(mSet, mContents,indx,indxN,indxE,verbose_query=True):
    
    subset = 'E(' + str(mContents.grid.origin[indxE]) + '), N(' + str(mContents.grid.origin[indxN]) + '), slave_date(' +  mContents.grid.origin[indx] +':' + mContents.grid.origin[indx] + ')'
    query = 'for c in ( ' + mSet + ' ) return encode (c[' + subset + '], "netcdf")'
    
    if verbose_query: print(query)
    
    ncdata = wcps_rasdaman(query, ip = 'saocompute.eurac.edu/sincohmap',verbose=verbose_query)
    dates = []
    for dd in ncdata.coords['master_date'].values:
        dates.append(datetime.datetime.fromtimestamp(int(dd)))#.strftime('%Y-%m-%d'))
    return dates
    
def getSlaveTimeAxis(mSet, mContents,indx,indxN,indxE,verbose_query=True):
    
    subset = 'E(' + str(mContents.grid.origin[indxE]) + '), N(' + str(mContents.grid.origin[indxN]) + '), master_date(' +  mContents.grid.origin[indx] +':' + mContents.grid.origin[indx] + ')'
    query = 'for c in ( ' + mSet + ' ) return encode (c[' + subset + '], "netcdf")'
    
    if verbose_query: print(query)
    
    ncdata = wcps_rasdaman(query, ip = 'saocompute.eurac.edu/sincohmap',verbose=verbose_query)
    dates = []
    for dd in ncdata.coords['slave_date'].values:
        dates.append(datetime.datetime.fromtimestamp(int(dd)))#.strftime('%Y-%m-%d'))
    return dates

def getIndexLabelFromWCScontents(label,contents):
    k = 0
    for s in contents.grid.axislabels:
        if label in s:
            return(k)
        k = k + 1
        
def getGeoInfo(mSet,verbose_query=True,x0=0,xN=0,y0=0,yN=0):
    
    from osgeo import gdal
    import os
    
    if x0!=0 and xN!=0 and y0!=0 and yN!=0:
        DO_CROP = 1
    else:
        DO_CROP = 0 
    
    wcs = WebCoverageService('http://saocompute.eurac.edu/sincohmap/rasdaman/ows?', version='2.0.1')
    mContents = wcs.contents[mSet]
    
    indxMaster = getIndexLabelFromWCScontents('master_date',mContents)
    indxSlave  = getIndexLabelFromWCScontents('slave_date',mContents)
    
    if DO_CROP:
        subset = 'E(' + str(x0) + ':' + str(xN) + '), N(' + str(y0) + ':' + str(yN) + \
                    '), master_date(' + mContents.grid.origin[indxMaster] +'), slave_date(' + mContents.grid.origin[indxSlave]  +')'
        #subset = 'master_date(' +  mContents.grid.origin[indxMaster] + '), slave_date(' +  mContents.grid.origin[indxSlave] + ')'
    else:
        subset = 'master_date(' +  mContents.grid.origin[indxMaster] + '), slave_date(' +  mContents.grid.origin[indxSlave] + ')'
        
    query = 'for c in ( ' + mSet + ' ) return encode (c[' + subset + '], "tiff", "nodata=-999")'
    
    if verbose_query: print(query)
        
    raster_path = wcps_rasdaman(query, ip = 'saocompute.eurac.edu/sincohmap',verbose=verbose_query)
    raster = gdal.Open(raster_path)
    
    remove = 0
    if remove:
        os.remove(raster_path)
        if verbose_query: print('Temporary file has been deleted')
    
    return {'Projection':raster.GetProjectionRef(), 'GeoTransform':raster.GetGeoTransform(),'RasterXSize':raster.RasterXSize, 'RasterYSize':raster.RasterYSize}
        
def getProjection(mSet,verbose_query=True):
    
    from osgeo import gdal
    import os
    
    wcs = WebCoverageService('http://saocompute.eurac.edu/sincohmap/rasdaman/ows?', version='2.0.1')
    mContents = wcs.contents[mSet]
    
    indxMaster = getIndexLabelFromWCScontents('master_date',mContents)
    indxSlave  = getIndexLabelFromWCScontents('slave_date',mContents)
    
    subset = 'master_date(' +  mContents.grid.origin[indxMaster] + '), slave_date(' +  mContents.grid.origin[indxSlave] + ')'
    query = 'for c in ( ' + mSet + ' ) return encode (c[' + subset + '], "tiff", "nodata=-999")'
    
    if verbose_query: print(query)
        
    raster_path = wcps_rasdaman(query, ip = 'saocompute.eurac.edu/sincohmap',verbose=verbose_query)
    
    raster = gdal.Open(raster_path)
    
    projj = raster.GetProjectionRef()
    
    remove = 0
    if remove:
        os.remove(raster_path)
        if verbose_query: print('Temporary file has been deleted')    
            
    return (projj)

def getProjectionImStack(mSet,verbose_query=True):
    
    from osgeo import gdal
    import os
    
    wcs = WebCoverageService('http://saocompute.eurac.edu/sincohmap/rasdaman/ows?', version='2.0.1')
    mContents = wcs.contents[mSet]
    
    indxMaster = getIndexLabelFromWCScontents('date',mContents)
    
    
    subset = 'date(' +  mContents.grid.origin[indxMaster] + ')'
    query = 'for c in ( ' + mSet + ' ) return encode (c[' + subset + '], "tiff", "nodata=-999")'
    
    if verbose_query: print(query)  
        
    raster_path = wcps_rasdaman(query, ip = 'saocompute.eurac.edu/sincohmap',verbose=verbose_query)
    
    raster = gdal.Open(raster_path)
    
    remove = 1
    if remove:
        os.remove(raster_path)
        if verbose_query: print('Temporary file has been deleted')
            
    return (raster.GetProjectionRef())

def getGeoTransform(mSet,verbose_query=True):
    
    from osgeo import gdal
    import os
    
    wcs = WebCoverageService('http://saocompute.eurac.edu/sincohmap/rasdaman/ows?', version='2.0.1')
    mContents = wcs.contents[mSet]
    
    indxMaster = getIndexLabelFromWCScontents('master_date',mContents)
    indxSlave  = getIndexLabelFromWCScontents('slave_date',mContents)
       
    subset = 'master_date(' +  mContents.grid.origin[indxMaster] + '), slave_date(' +  mContents.grid.origin[indxSlave] + ')'
    query = 'for c in ( ' + mSet + ' ) return encode (c[' + subset + '], "tiff", "nodata=-999")'
    
    if verbose_query: print(query)  
        
    raster_path = wcps_rasdaman(query, ip = 'saocompute.eurac.edu/sincohmap',verbose=verbose_query)
    
    raster = gdal.Open(raster_path)
    
    remove = 1
    if remove:
        os.remove(raster_path)
        if verbose_query: print('Temporary file has been deleted')
    
    return (raster.GetGeoTransform())

def getGeoTransformImStack(mSet,verbose_query=True, remove=True):
    
    from osgeo import gdal
    import os
    
    wcs = WebCoverageService('http://saocompute.eurac.edu/sincohmap/rasdaman/ows?', version='2.0.1')
    mContents = wcs.contents[mSet]
    
    indxDate = getIndexLabelFromWCScontents('date',mContents)
    
    subset = 'date(' +  mContents.grid.origin[indxDate] + ')'
    query = 'for c in ( ' + mSet + ' ) return encode (c[' + subset + '], "tiff", "nodata=-999")'
    
    if verbose_query: print(query)
        
    raster_path = wcps_rasdaman(query, ip = 'saocompute.eurac.edu/sincohmap',verbose=verbose_query)
    
    raster = gdal.Open(raster_path)
    
    if remove:
        os.remove(raster_path)
        if verbose_query: print('Temporary file has been deleted')
    
    return (raster.GetGeoTransform())

def getReferenceImage(mSet,verbose_query=True, rotate=False,x0=0,xN=0,y0=0,yN=0):
    
    if x0!=0 and xN!=0 and y0!=0 and yN!=0:
        DO_CROP = 1
    else:
        DO_CROP = 0
        
    print(DO_CROP)
    
    # Select WCS server and service version
    wcs = WebCoverageService('http://saocompute.eurac.edu/sincohmap/rasdaman/ows?', version='2.0.1')
    mContents = wcs.contents[mSet]
    
    indxMaster = getIndexLabelFromWCScontents('master_date',mContents)
    indxSlave  = getIndexLabelFromWCScontents('slave_date',mContents) 
    indxN = getIndexLabelFromWCScontents('N',mContents) 
    indxE  = getIndexLabelFromWCScontents('E',mContents) 
    
    masterDates = getMasterTimeAxis(mSet, mContents,indxSlave, indxN, indxE)
    slaveDates  = getSlaveTimeAxis(mSet, mContents,indxMaster, indxN, indxE)
        
    date0M = masterDates[0].strftime('%Y-%m-%d')
    date0S = slaveDates[0].strftime('%Y-%m-%d')
    
    print(date0M)
    print(date0S)
    
    #subset = 'master_date(\"' + date0M +'\"), slave_date(\"' + date0S  +'\")'
    if DO_CROP:
        subset = 'E(' + str(x0) + ':' + str(xN) + '), N(' + str(y0) + ':' + str(yN) + \
                    '), master_date(\"' + date0M +'\"), slave_date(\"' + date0S  +'\")'
    else:
        subset = 'master_date(\"' + date0M +'\"), slave_date(\"' + date0S  +'\")'
        
    query = 'for c in ( ' + mSet + ' ) return encode (c[' + subset + '], "tiff")'
    subset_coherence = wcps_rasdaman(query, ip = 'saocompute.eurac.edu/sincohmap',verbose=verbose_query)
    
    if rotate:
        refPath = mSet + '_reference_image_rotated.tiff'
        
        print(subset_coherence)
        print(refPath)
        
        rotate_tiff_file(subset_coherence,refPath)
        os.remove(subset_coherence)
    else:
        refPath = mSet + '_reference_image.tiff'
        shutil.move(subset_coherence, refPath)
        
        
    return(refPath)


def getReferenceImageImStack(mSet,verbose_query=True, rotate=False,x0=0,xN=0,y0=0,yN=0):
    
    if x0!=0 and xN!=0 and y0!=0 and yN!=0:
        DO_CROP = 1
    else:
        DO_CROP = 0
        
    print(DO_CROP)
    
    # Select WCS server and service version
    wcs       = WebCoverageService('http://saocompute.eurac.edu/sincohmap/rasdaman/ows?', version='2.0.1')
    mContents = wcs.contents[mSet]
   
    indxN     = getIndexLabelFromWCScontents('N',mContents) 
    indxE     = getIndexLabelFromWCScontents('E',mContents) 
    
    mDates    = getTimeAxis(mSet, mContents, indxN, indxE)
    date0     = mDates[0].strftime('%Y-%m-%d')
    print(date0)
        
    #subset = 'master_date(\"' + date0M +'\"), slave_date(\"' + date0S  +'\")'
    if DO_CROP:
        subset = 'E(' + str(x0) + ':' + str(xN) + '), N(' + str(y0) + ':' + str(yN) + \
                    '), date(\"' + date0 +'\")'
    else:
        subset = 'date(\"' + date0 +'\")'
        
    query = 'for c in ( ' + mSet + ' ) return encode (c[' + subset + '], "tiff")'
    subset_coherence = wcps_rasdaman(query, ip = '10.8.246.69',verbose=verbose_query)
    
    if rotate:
        refPath = mSet + '_reference_image_rotated.tiff'
        
        print(subset_coherence)
        print(refPath)
        
        rotate_tiff_file(subset_coherence,refPath)
        os.remove(subset_coherence)
    else:
        refPath = mSet + '_reference_image.tiff'
        shutil.move(subset_coherence, refPath)
        
        
    return(refPath)
    
        
    
 
    