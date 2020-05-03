def ctab(a, b):
    """
    Generate a Confusion-Matrix based on classified and reference image

    Args:
        a: n-D NumpyArray derived from classified image raster
        b: n-D NumpyArray derived from reference image raster
    Returns:
         result (n-D NumpyArray): Confusion Matrix
    """
    import numpy as np
    import pandas as pd
    #print(np.version.version)
    a = a.flatten().astype(int)
    b = b.flatten().astype(int)
    
    name = np.union1d(a, b)
    
    colnames = name
    rownames = name
    result = np.zeros((len(colnames), len(rownames)))
    
            
    for c in range(0, len(rownames)):
        for d in range(0, len(rownames)):
            # get all indices of classified image labelled as specified class
            c_classified=np.where(a==name[c])
            # get all indices of reference image labelled as specified class
            c_reference=np.where(b==name[d])
            # find all indices that are present in both classified and reference image
            c_agreement=np.in1d(c_classified,c_reference, assume_unique=True)
            # count number of positive matches for all indices present in both images
            result[c,d]=np.count_nonzero(c_agreement)
        
    #print(result)
    #print(len(result))

    # implement mask 
    #mask = None
    #for i in range(0, len(mask)):
    #    result = result[result[index] != mask[i],:]
    #    if len(result.as_matrix)==1 or len(result.as_matrix[0])==1:
    #        raise SystemExit('Calculation of contingency table requires at least two categories')
    #    result = result[:,result.index != mask[i]]
    #
    #if len(result.as_matrix)==1 or len(result.as_matrix[0])==1:
    #        raise SystemExit('Calculation of contingency table requires at least two categories')   
    
    return result

def kstat(classified, reference, perCategory = False):
    """
    Calculate Statistics based on Confusion Matrix. 
    Indizes for Accuracy and Validation of Image Classifications taken from:
    Pontius, R.G., Jr., and Millones, M. (2011): Death to Kappa: Birth of Quantity Disagreement and Allocation Disagreement for Accuracy Assessment. International Journal of Remote Sensing, 32: 4407-4429.
    
    Args:
        classified: path to your GTIFF containing the classified image
        reference: path to your GTIFF containing the reference image or rasterized Training Data
    Returns:
         resultDF (pandas DataFrame): DataFrame with Overall Accuracy, Kappa Index, Kappa of location, Kappa of histogram, chance agreement, quantity agreement, allocation agreement, allocation disagreement, quantity disagreement
    """
    import numpy as np
    import pandas as pd
    from osgeo import gdal
    ##################################################
    #Function: Calculate kappa indices and disagreements
    def calculateKappa(ct):
    
        ct = ct/np.sum(ct)#percent of pixels 
        cmax = len(ct)#number of categories
        #Fraction of Agreement:
        PA = 0
        for i in range(0, cmax):
            PA +=  ct[i][i]

        #Expected Fraction of Agreement subject to the observed distribution:
        PE = sum(np.sum(ct, axis = 1)*np.sum(ct, axis = 0))

        #Maximum  Fraction  of  Agreement  subject  to  the  observed  distribution:
        PMax = 0
        a1 = np.sum(ct, axis = 1)
        a2 = np.sum(ct, axis = 0)
        for i in range(0, cmax):
            PMax += min(a1[i], a2[i])
        #Kappa Index:
        K = (PA-PE)/(1-PE)
        #Kappa of location:
        Kloc = (PA-PE)/(PMax-PE)
        #Kappa of histogram:
        Khisto = (PMax-PE)/(1-PE)
        #chance agreement:
        CA = 100*min((1/cmax),PA,PE)
        #quantity agreement:
        if min((1/cmax),PE,PA)==(1/cmax):
            QA = 100*min((PE-1/cmax),PA-1/cmax)
        else:
            QA = 0
        #allocation agreement:
        AA = 100*max(PA-PE,0)
        #allocation disagreement:
        AD = 100*(PMax-PA)
        #quantity disagreement:
        QD = 100*(1-PMax)
        KappaResult = (PA*100,K,Kloc,Khisto,CA,QA,AA,AD,QD)
        return KappaResult
    #########################################################################
    
    
    #read tif as array and check dimensions/extent
    ds_cl = gdal.Open(classified)
    ds_ref = gdal.Open(reference)
    
    if ds_cl.RasterXSize != ds_ref.RasterXSize or ds_cl.RasterYSize != ds_ref.RasterYSize:
        raise ValueError('Warning: Classified Image and Reference Image do not have the same Extent!')
    
    a = ds_cl.GetRasterBand(1).ReadAsArray()
    b = ds_ref.GetRasterBand(1).ReadAsArray()
    
    if 0 in np.unique(b):
        b_nonzero = np.nonzero(b)
        b = b[b_nonzero]
        a = a[b_nonzero]
    ###########################################################################
    
    # generate confusion matrix
    ct = ctab(a,b)
    ct2 = ctab(a,b)
    #reclass to calculate kappa per category
    result = [] 
    cttmp = ct
    if len(cttmp) <= 2 or perCategory == False:
        result.append(calculateKappa(ct))
        if perCategory == True:
            print('Warning: Kappa per category requires at least 3 categories')
    
    if perCategory == True and len(cttmp[0]) > 2:
        for ca in range(0, len(cttmp[0])+1):
            if ca == len(cttmp[0]):
                ct = ct2
            else:
                ct = cttmp
                ctNew = ct[0:2,0:2]
                ctNew[0,0] = ct[ca, ca]
                ctNew[0,1] = sum(ct[ca])-ct[ca,ca]
                ctNew[1,0] = sum(ct[0:-1,ca])-ct[ca,ca]
                ctNew[1,1] = sum(sum(ct))-sum(ct[ca])-sum(ct[0:-1, ca])+ctNew[0,0]
                ct = ctNew
            result.append(calculateKappa(ct))
    ################################################################
    
    #arrange results
    if len(result) > 1:
        name = list(np.union1d(a, b))
        name.extend(['Overall'])
        resultDF = pd.DataFrame(result, columns = ['PA','K','Kloc','Khisto','CA','QA','AA','AD','QD'], index = name)
    else:
        resultDF = pd.DataFrame(result, columns = ['PA','K','Kloc','Khisto','CA','QA','AA','AD','QD'], index = ['Overall'])
    return resultDF
