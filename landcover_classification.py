#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
#------------------------------------------------------------------------------
# Name:  landcover_classification.py
#
#   General purpose:
#       A set of modules for landcover classifcation using RandomForests & Support Vector Machine.
#       As input any kind of satellite imagery (e.g Senintel-1/2, Landast) can
#       be used. As long as the input images are stacked into a single TIFF which all have the same spatial extent
#       and resolution. Also classification validation and plotting are supported.
#       For further information read the docs provided in the modules.
#
# Author:   Harald Kristen <haraldkristen at posteo dot at>
#                   Alexander Jacob <alexander dot jacob 
# Date: 04.05.2020
##
#-------------------------------------------------------------------------------
# Copyright (C) 2020 Harald Kristen, Alexander Jacob
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies of this Software or works derived from this Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#-------------------------------------------------------------------------------
"""

def prepare_training_data(training_path, raster_path, column, nr_points, sampling_methodology = 'proportional', vector_path=None, mask_path=None, sample_path=None):
    """
    Prepare training data from CORINE 2012 & LISS 2013 for Landcover Classification with scikit-learn

    Args:
        vector_path (str): Path to classification layer in ESRI Shapfile format
        raster_path (str): Path to raster file (TIFF) to be classified with scikit-learn
        column (str): Shapefile column/attribute that should be used for classification
        nr_points (int): Number of random sampling points
        sampling_methodology (str): Either choose 'proportional, 'equal', 'random' (default 'proportional')

    Return:
        training_labels (1D ndarray): Training labels as ndarray with (shape = rows*cols) 
        training_labels.tiff (GTiff): Training labels saved to current work directory as GTIFF 
        training_samples (2D ndarray): Training sample of the input raster dataset with shape = (rows*cols, bands)
        bands_data (3D ndarray): The input raster dataset as ndarray with shape = (rows, cols, bands)
        projection (str): Projection definition string (Returned by gdal.Dataset.GetProjectionRef)
        geo_transform (tuple): Returned value of gdal.Dataset.GetGeoTransform (coefficients for transforming between 
                                 pixel/line (P,L) raster space, and projection coordinates (Xp,Yp) space.
        test_labels (1D ndarray): Test labels as ndarray with (shape = rows*cols)
        test_samples (2D ndarray): Test sample of the input raster dataset with shape = (rows*cols, bands)

    Sources: https://github.com/ceholden/open-geo-tutorial
            https://www.machinalis.com/blog/python-for-geospatial-data-processing/
    """

    import numpy as np
    import os
    from osgeo import gdal
    from osgeo import ogr
    from geo_utils import write_geotiff, create_raster_from_vector
    import pickle
    def general_info(dataset):
        print('''
        ################################
        General info about the shapefile
        ################################
        ''')
        ### Let's get the driver from this file
        driver = dataset.GetDriver()
        print('Dataset driver is: {n}\n'.format(n=driver.name))

        ### How many layers are contained in this Shapefile?
        layer_count = dataset.GetLayerCount()
        print('The shapefile has {n} layer(s)\n'.format(n=layer_count))

        ### What is the name of the 1 layer?
        layer = dataset.GetLayerByIndex(0)
        print('The layer is named: {n}\n'.format(n=layer.GetName()))

        ### What is the layer's geometry? is it a point? a polyline? a polygon?
        # First read in the geometry - but this is the enumerated type's value
        geometry = layer.GetGeomType()

        # So we need to translate it to the name of the enum
        geometry_name = ogr.GeometryTypeToName(geometry)
        print("The layer's geometry is: {geom}\n".format(geom=geometry_name))

        ### What is the layer's projection?
        # Get the spatial reference
        spatial_ref = layer.GetSpatialRef()

        # Export this spatial reference to something we can read... like the Proj4
        proj4 = spatial_ref.ExportToProj4()
        print('Layer projection is: {proj4}\n'.format(proj4=proj4))

        ### How many features are in the layer?
        feature_count = layer.GetFeatureCount()
        print('Layer has {n} features\n'.format(n=feature_count))

        ### How many fields are in the shapefile, and what are their names?
        # First we need to capture the layer definition
        defn = layer.GetLayerDefn()

        # How many fields
        field_count = defn.GetFieldCount()
        print('Layer has {n} fields'.format(n=field_count))

        # What are their names?
        print('Their names are: ')
        for i in range(field_count):
            field_defn = defn.GetFieldDefn(i)
            print('\t{name} - {datatype}'.format(name=field_defn.GetName(),
                                                 datatype=field_defn.GetTypeName()))
    def random_sampling(roi, nr_points):
        """
        Produce a completely random sample of a 2D image
        Args:
            roi (2D ndarray): A 2D raster image for sampling
            nr_points (int): The number of pixels to be sampled

        Returns:
            samples (2D ndarray): A random sample of the input image, empty pixels=0
        """
        # create empty np_array with all cell values = 0
        samples = np.zeros(roi.shape, dtype=float)
        # Random sampling
        for i in range(0,nr_points):
            if i == 0 or i == -999:
                pass
            else:
                coord = np.random.randint(extent, size=2)
                x = coord[0]
                y = coord[1]
                samples[x,y] = roi[x,y]
        return samples
    
    def random_sampling_equal(roi, training_labels, nr_points, window=1):
        """
        Produce a random sample of a 2D image, where every class has the same amount of sampling pixels
        Args:
            roi (2D ndarray): A 2D raster image for sampling
            training_labels (ndarray): A list with the names of the classes (=unique pixel values)
            nr_points (int): The number of pixels to be sampled

        Returns:
            sample_raster (2D ndarray): A random equal sample of the input image, empty pixels = 0
        """
        total_pixel = roi.size
        #nr_points = total_pixel * percentage
        # create empty np_array with all cell values = 0
        sample_raster = np.zeros(roi.shape, dtype=float)
        # equally distribute the number of points to all classes
        nr_classes = training_labels.size
        if 0 in training_labels:
            nr_points_per_class = round(nr_points / (nr_classes-1)) 
        else:
            nr_points_per_class = round(nr_points / nr_classes)
        print('no of classes ', nr_classes, ' no of points per class: ', nr_points_per_class)
        for i in training_labels:           
            # if the class has a Nodata value like 0 or -999 pass
            maxCount = (roi == i).sum()
            print ("class: ", i)
            
            # avoid 0 class and error values.
            if i == 0 or i == -999:
                pass
            else:
                # subset only one class of the ROI
                roi_select = roi * (roi == i)
                count = 0
                # loop through the subset &
                try_count = 0
                
                # search for samples until you have found enough, 
                # for small classes make sure that you never select more than half of the available pixels
                while True:                    
                    try_count =  try_count + 1
                    # select random position in selected class
                    coord = np.random.randint(roi_select.shape[0], size=2)
                    x = coord[0]
                    y = coord[1]
                    skip_pixel = False
                    
                    # only select not yet assigned pixels with class label
                    if (i == roi_select[x,y]) and (sample_raster[x,y] == 0):
                        
                        # check if value in direct neighborhood is already set
                        # and avoid selecting pixels next to each other that way
                        for j in range(x-window, x+window):
                            for k in range(y-window, y +window):
                                try:
                                    if ( j != x and k != y) and (sample_raster[j][k] == roi_select[x][y]):
                                        skip_pixel = True
                                        continue
                                except IndexError:
                                    pass
                            if skip_pixel: continue
                        if skip_pixel: continue
                        
                        # avoid pixels being on the border to another class
#                         for j in range(x-1, x+1):
#                             for k in range(y-1, y+1):
#                                 try:
#                                     if ( j != x and k != y) and (roi_select[j][k] != roi_select[x][y]):
#                                         skip_pixel = True
#                                         continue
#                                 except IndexError:
#                                     pass
#                             if skip_pixel: continue
#                         if skip_pixel: continue
                        
                        # if no problems occured assign class value to selected sample
                        sample_raster[x,y] = roi_select[x, y]
                        count = count + 1
                        
                    if try_count > 10000000: 
                        print("too many trials")
                        break
                    if count >= nr_points_per_class:
                        print("found enough samples ")
                        break
                    if count >= maxCount/2:
                        print("found half of class already ")
                        break
                        
        return sample_raster
    
    def random_sampling_proportional(roi, training_labels, nr_points):
        """
        Produce a random sample of a 2D image, where the number of samples in one class is proportional to the total
        number of pixels in this class. The minimum number of pixels per class is 1% of all input pixels.
        Args:
            roi (2D ndarray): A 2D raster image for sampling
            training_labels (ndarray): A list with the names of the classes (=unique pixel values)
            nr_points (int): The number of pixels to be sampled

        Returns:
            sample_raster (2D ndarray): A random proportional sample of the input image, empty pixels = 0
        """
        # create empty np_array with all cell values = 0
        sample_raster = np.zeros(roi.shape, dtype=float)
        # number of points proportional to class size
        total_pixel = roi.size
        for i in training_labels:
            maxCount = (roi == i).sum()
            # proportianlly distribute the number of points to all classes in respect to their class size (e.g. nr of pixels)
            class_size = np.count_nonzero(roi == i)
            nr_points_per_class = round(nr_points * (class_size / total_pixel))
            # make sure that there are at least a few sampling points in every class
            # --> minimum nr_points_per_class = 1%
            if nr_points_per_class < (nr_points * 0.01):
                nr_points_per_class = nr_points * 0.01
            # if the class has a Nodata value like 0 or -999 pass
            if i == 0 or i == -999:
                pass
            else:
                # subset only one class of the ROI
                roi_select = roi * (roi == i)
                count = 0
                while count <= nr_points_per_class and count <= maxCount/2:
                    coord = np.random.randint(roi_select.shape[0], size=2)
                    x = coord[0]
                    y = coord[1]
                    if (i == roi_select[x,y]) and (sample_raster[x,y] == 0):
                        sample_raster[x,y] = roi_select[x,y]
                        count = count + 1
        return sample_raster
    
    if vector_path != None:
        ##############################
        # Rasterize the vector layer #
        ##############################
        # Open the dataset from the file
        print(vector_path)
        vector = ogr.Open(vector_path)
        # Print some general info about the shapefile
        general_info(vector)
        # Tie-in vector dataset with Raster dataset ( = rasterize vector)
        if mask_path != None:
            create_raster_from_vector(vector_path, raster_path, column, gdal.GDT_Int16, mask_path=mask_path, rasterized_name=training_path)
        else:
            create_raster_from_vector(vector_path, raster_path, column, gdal.GDT_Int16, rasterized_name=training_path)
        roi_ds = gdal.Open(training_path, gdal.GA_ReadOnly)
    else:
        roi_ds = gdal.Open(training_path, gdal.GA_ReadOnly)
    ############################
    # Random sample generation #
    ############################

    # Check the rasterized layer
    roi = roi_ds.GetRasterBand(1).ReadAsArray(buf_type = gdal.GDT_Int16)

    # How many pixels are in each class?
    training_labels = np.unique(roi)
    # Iterate over all class labels in the ROI image, printing out some information
    for c in training_labels:
                print('Class {c} contains {n} pixels'.format(c=c, n=(roi == c).sum()))

    extent = roi.shape[0] #extent of the array
    print('extent: ', extent)
    roi = np.array(roi).astype(np.int) #convert array to numpy_array in FLOAT
    if sampling_methodology == 'proportional':
        training_pixels = random_sampling_proportional(roi, training_labels, nr_points)
        #test_pixels = random_sampling_proportional(roi, training_labels, nr_points)
        #test_pixels = [None]
    elif sampling_methodology == 'equal':
        training_pixels = random_sampling_equal(roi, training_labels, nr_points)
        #test_pixels = random_sampling_equal(roi, training_labels, nr_points)
        #test_pixels = [None]
    elif sampling_methodology == 'random':
        training_pixels = random_sampling(roi, nr_points)
        #test_pixels = random_sampling(roi, nr_points)
        #test_pixels = [None]
    else:
        print('Error: Choose a implemented sampling strategy! -> proportional OR equal OR random')

    # write training_labels to disk as TIFF
    #write_geotiff('training_labels.tiff', training_pixels, roi_ds.GetGeoTransform(), roi_ds.GetProjectionRef())

    #############################
    # Training labels & samples #
    #############################

    raster_dataset = gdal.Open(raster_path, gdal.GA_ReadOnly)
    geo_transform = raster_dataset.GetGeoTransform()
    projection = raster_dataset.GetProjectionRef()
    bands_data = []
    for b in range(1, raster_dataset.RasterCount + 1):
        band = raster_dataset.GetRasterBand(b)
        bands_data.append(np.nan_to_num(band.ReadAsArray()))

    bands_data = np.dstack(bands_data)

    #training dataset
    is_train = np.nonzero(training_pixels)
    training_labels = training_pixels[is_train]
    training_samples = bands_data[is_train]

    #test dataset
    #is_test = np.nonzero(test_pixels)
    #test_labels = test_pixels[is_test]
    #test_samples = bands_data[is_test]

    # clean up
    roi_ds = None  # close file again
    #os.remove('rasterized.tif')
    
    # save your training aray to disk
    if sample_path != None:        
        pickle.dump( np.nonzero(training_pixels), open(sample_path , "wb" ) )

    return training_samples, training_labels, bands_data, projection, geo_transform, training_pixels

def load_training_data(raster_path, training_path, sample_path):
    
    import numpy as np
    from osgeo import gdal
    import pickle
    
    raster_dataset = gdal.Open(raster_path, gdal.GA_ReadOnly)
    reference_dataset = gdal.Open(training_path, gdal.GA_ReadOnly)
    reference_raster = reference_dataset.GetRasterBand(1).ReadAsArray()
    geo_transform = raster_dataset.GetGeoTransform()
    projection = raster_dataset.GetProjectionRef()
    bands_data = []
    for b in range(1, raster_dataset.RasterCount + 1):
        band = raster_dataset.GetRasterBand(b)
        bands_data.append(np.nan_to_num(band.ReadAsArray()))

    bands_data = np.dstack(bands_data)
    is_train = pickle.load( open(sample_path, "rb" ) )
    training_samples = bands_data[is_train]
    training_pixels = np.zeros(reference_raster.shape, dtype=int)
    training_pixels[is_train] = reference_raster[is_train]
    training_labels = training_pixels[is_train]

    # Iterate over all class labels in the ROI image, printing out some information
    for c in np.unique(reference_raster):
        print('Class {c} contains {n} pixels'.format(c=c, n=(training_pixels == c).sum()))
    
    return training_samples, training_labels, bands_data, projection, geo_transform, training_pixels

def classification(training_samples, training_labels, test_labels, test_samples, bands_data, projection, geo_transform, classifier='rf', gridsearch = True, mask_path = '', class_path='classified_image.tiff', **kwargs):
    from geo_utils import write_geotiff
    from osgeo import gdal
    import numpy as np

    if classifier == 'rf':
        from sklearn.ensemble import RandomForestClassifier

        print('\nStarting Random Forest classification, lean back and wait for the magic to happen :) ')

        # Define the classifier with aditional KWARGs
        classifier = RandomForestClassifier(oob_score=True, n_jobs=-1, **kwargs)

        if gridsearch:
            # Search for the best paramter combination with an Exhaustive Grid Search

            nr_features = training_samples.shape[1]
            from sklearn.model_selection import GridSearchCV
            from sklearn.metrics import accuracy_score, make_scorer

            param_grid = {'n_estimators': [50, 100, 150, 200, 250, 300], 'max_features': range(2, nr_features, 1)}
            classifier = GridSearchCV(classifier, param_grid, cv=5, scoring=make_scorer(accuracy_score))
            classifier.fit(training_samples, training_labels)

            # Fit the model again with the ideal parameters
            classifier = classifier.best_estimator_
            classifier.fit(training_samples, training_labels)
            print('GridSearchCV choose the best parameter combination for the classification as following:\n' + str(
                classifier) + '\n')

        classifier.fit(training_samples, training_labels)

    elif classifier == 'svm':
        from sklearn import svm
        print('\nStarting SVM classification, lean back and wait for the magic to happen :) ')

        #TODO: Clean up SVM code
        #Parameters as suggested in (Abdikan, Sanli, Ustuner, & Calò, 2016) -> Produces only one class, 35% accuracy
        #classifier = svm.SVC(gamma=0.333, C=100, kernel='rbf', cache_size=20000, **kwargs)

        #this parameter set one produces better results (e.g ~50% accuracy for Level 3)
        classifier = svm.SVC(gamma=0.000001, C=100, kernel='rbf', cache_size=20000, **kwargs)

        if gridsearch:
            # Perform a Randomized Parameter Optimization as shown in:
            # http://scikit-learn.org/stable/modules/grid_search.html

            from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedShuffleSplit
            import numpy as np
            import math
            from scipy import stats

            classifier_baseline = svm.SVC(kernel='rbf',cache_size=20000)
            
            gamma = range(-5, 5, 1)
            gamma_exp = np.zeros(len(gamma))
            cmargin = range(0, 10, 1)
            cmargin_exp = np.zeros(len(cmargin))
            
            count = 0
            for g in gamma:
                gamma_exp[count] = math.pow(10,g)
                count += 1
                
            count = 0            
            for c in cmargin:
                cmargin_exp[count] = math.pow(10,c)
                count += 1
                
            #param_grid = {'gamma': [10**-8,10**-7,10**-6,10**-5,10**-4,10**-3,10**-2,10**-1,10,100], 'C': range(1,100,10)}
            # use a random parameter grid as shown in: http://scikit-learn.org/stable/modules/grid_search.html
            param_grid = {'C': cmargin_exp, 'gamma': gamma_exp}
            print("param grid: ", param_grid)

            ## tune the hyperparameters via a randomized search (100 iterations & computation on all cores)
            #grid = RandomizedSearchCV(classifier_baseline, param_grid, n_iter=100, n_jobs=-1)
            cv = StratifiedShuffleSplit(n_splits=5, test_size=0.5)
            grid = GridSearchCV(classifier_baseline, param_grid, cv=cv, n_jobs=-1)            

            grid.fit(training_samples, training_labels)
            # evaluate the best randomized searched model on the testing
            # data
            print(grid.cv_results_)

            acc = grid.score(test_samples, test_labels)
            print("[INFO] grid search accuracy: {:.2f}%".format(acc * 100))
            print("[INFO] randomized search best parameters: {}".format(
                grid.best_params_))
            classifier = grid

        classifier.fit(training_samples, training_labels)
    else:
        print('\nThe selected classifier is not implemented, try "svm" or "rf"')

    # reshape array
    rows, cols, n_bands = bands_data.shape
    n_samples = rows * cols

    flat_pixels = bands_data.reshape((n_samples, n_bands))
    result = classifier.predict(flat_pixels)
    classified_image = result.reshape((rows, cols)).astype(int)
    if mask_path != '':
        mask_ds = gdal.Open(mask_path)
        mb = mask_ds.GetRasterBand(1)
        mData = mb.ReadAsArray(buf_type = gdal.GDT_Int16)
        classified_image = np.multiply(classified_image, mData)

    # write TIFF with labels to disk
    write_geotiff(class_path, classified_image, geo_transform, projection, image_type=gdal.GDT_UInt16)

    return classified_image, classifier

def validation(classified_image, is_test, test_labels, training_samples, classifier):
    """
    Calculates basic statistics to assess the classification result    
    Args:
        classified_image (2D ndarray): The reshaped 2D output of the classifier
        is_test (1D ndarray): A array with the pixels from the random sampling for testing
        test_labels (1D ndarray): Test labels as ndarray with (shape = rows*cols) 
        training_samples (2D ndarray): Training sample of the input raster dataset with shape = (rows*cols, bands)
        classifier (object): Scikit-learn classifier object

    Returns:
        Confusion matrix, Overall/user/producer accuracy, Kappa score, Mc Nemars´test
        Additionally if Random Forest is used: Feature ranking/importance
    """

    from sklearn import metrics
    import numpy as np
    from matplotlib import pyplot as plt
    import seaborn as sns

    # Select the predicted pixels + classes from the classified image
    predicted_labels = classified_image[is_test]
    classes = np.unique(test_labels).astype(int)

    ### Confusion matrix with seaborn
    sns.set()
    mat = metrics.confusion_matrix(test_labels, predicted_labels)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    sns.plt.show()

    ### Cassification report
    # precision = producer accuracy
    # recall = user accuracy
    target_names = ['Class %s' % s for s in classes]
    print("\nClassification report: \nprecision = producer accuracy \nrecall = user accuracy \n%s" %
          metrics.classification_report(test_labels, predicted_labels, target_names=target_names))

    # The next two statistics are only available in RandomForest
    if str(type(classifier)) == "<class 'sklearn.ensemble.forest.RandomForestClassifier'>":

        ### Feature importance
        importances = classifier.feature_importances_
        std = np.std([tree.feature_importances_ for tree in classifier.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]
        # Print the feature ranking
        print("Feature ranking:")
        for f in range(training_samples.shape[1]):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(training_samples.shape[1]), importances[indices],
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(training_samples.shape[1]), indices, rotation=-45)
        plt.xlim([-1, training_samples.shape[1]])
        plt.show()

        ### OOB prediction
       # print('\nThe OOB prediction of accuracy is: {oob}%'.format(n_estimators=classifier.n_estimators, oob=classifier.oob_score_ * 100))

    ### Classification accuracy
    print("\nOverall Classification accuracy: %f" %
          metrics.accuracy_score(test_labels, predicted_labels))

    ### Kappa score
    print("\nKappa score: %f" %
          metrics.cohen_kappa_score(test_labels, predicted_labels))

    ### Mc Nemars test
    def mcnemar(x, y=None, exact=True, correction=True):
        '''
        McNemars test

        Parameters
        ----------
        x, y : array_like
            two paired data samples. If y is None, then x can be a 2 by 2
            contingency table. x and y can have more than one dimension, then
            the results are calculated under the assumption that axis zero
            contains the observation for the samples.
        exact : bool
            If exact is true, then the binomial distribution will be used.
            If exact is false, then the chisquare distribution will be used, which
            is the approximation to the distribution of the test statistic for
            large sample sizes.
        correction : bool
            If true, then a continuity correction is used for the chisquare
            distribution (if exact is false.)

        Returns
        -------
        stat : float or int, array
            The test statistic is the chisquare statistic if exact is false. If the
            exact binomial distribution is used, then this contains the min(n1, n2),
            where n1, n2 are cases that are zero in one sample but one in the other
            sample.

        pvalue : float or array
            p-value of the null hypothesis of equal effects.

        Notes
        -----
        This is a special case of Cochran's Q test. The results when the chisquare
        distribution is used are identical, except for continuity correction.

        Source
        ------
        http://www.statsmodels.org/stable/_modules/statsmodels/sandbox/stats/runs.html#mcnemar

        '''

        import numpy as np
        from scipy import stats
        import warnings

        x = np.asarray(x)
        if y is None and x.shape[0] == x.shape[1]:
            if x.shape[0] != 2:
                raise ValueError('table needs to be 2 by 2')
            n1, n2 = x[1, 0], x[0, 1]
        else:
            # I'm not checking here whether x and y are binary,
            # isn't this also paired sign test
            n1 = np.sum(x < y, 0)
            n2 = np.sum(x > y, 0)

        if exact:
            stat = np.minimum(n1, n2)
            # binom is symmetric with p=0.5
            pval = stats.binom.cdf(stat, n1 + n2, 0.5) * 2
            pval = np.minimum(pval, 1)  # limit to 1 if n1==n2
        else:
            corr = int(correction)  # convert bool to 0 or 1
            stat = (np.abs(n1 - n2) - corr) ** 2 / (1. * (n1 + n2))
            df = 1
            pval = stats.chi2.sf(stat, df)
        return stat, pval
    stat, pval = mcnemar(test_labels, predicted_labels)
    print("\nMc Nemars test\nChi-square %f P-value %f" % (stat, pval))

    #TODO: Save classififed image + validation results to seperate folder for every run -> user can specify the name
    return print('\nEverything worked out just fine, congrats :)')

def plot_classified_image(class_path='classified_image.tiff', plot_map_info=False, plot_map_legend=False, plot_title='Classified Image'):
    
    """

    Args:
        class_path: path to image to plot
        plot_map_info: Optional boolean to print map info on plot
        plot_map_legend: Optional boolean to print map legend on plot
        plot_title: Provice a custom title for plot

    Returns:

    """
    
    import numpy as np
    from matplotlib import pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import NoNorm
    from matplotlib_scalebar.scalebar import ScaleBar
    import numpy as np
    from osgeo import gdal
    
    classified_image_ds = gdal.Open(class_path)
    classified_band = classified_image_ds.GetRasterBand(1)
    classified_image = classified_band.ReadAsArray(buf_type = gdal.GDT_Int16)

    corine_cmap, corine_norm, handles = get_corine_color_map(classified_image)
    
    plt.matplotlib.cm.register_cmap(name='corine', cmap=corine_cmap);
    
    dpi = 80
    height, width = classified_image.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    
    # Hide spines, ticks, etc.
    ax.axis('on')
    
    #plot map info elements
    if plot_map_info:
        ext = classified_image_ds.GetGeoTransform()
        ncol = classified_image_ds.RasterXSize
        nrow = classified_image_ds.RasterYSize
        x_min = ext[0]
        x_max = ext[0] + ext[1] * ncol
        y_min = ext[3] + ext[5] * nrow
        y_max = ext[3]
        plt.xticks(np.arange(x_min, x_max+5000, 5000))
        plt.yticks(np.arange(y_min, y_max+5000, 5000))
        scalebar = ScaleBar(1, location='lower left', box_alpha=0.5)
        plt.gca().add_artist(scalebar)
        plt.arrow(x_min+1000,y_max-2000,0,900,fc="k", ec="k", linewidth = 4, head_width=200, head_length=500)
        plt.text(x_min+950, y_max-500, 'N')
        # Display the image.
        ax.imshow(classified_image, cmap='corine', norm=corine_norm, extent=[x_min, x_max, y_min, y_max])
    else:
        # Display the image.
        ax.imshow(classified_image, cmap='corine', norm=corine_norm)
        
    # Plot legend
    if plot_map_legend:
        plt.legend(frameon=1, shadow=1, framealpha=0.5, handles=handles, title='LC classes', facecolor='white')
    
    plt.title(plot_title)
    
def get_corine_color_map(classified_image):
    
    """

    Args:
        classified_image: 2d numpy array containing classified image to be plotted

    Returns:
        corine_cmap: Color map for all corine classes present in classified_image
        corine_norm: Norm for color map to print all present colors correctly
        handles: List of handles containing correct label for each class present.

    """
    
    import numpy as np
    from matplotlib import pyplot as plt
    import matplotlib.patches as mpatches
    
    corine_colors = dict((
                (0, (0, 0, 0, 255)),
                (100, (255, 0, 0, 255)), # Urban
                (110, (255, 11, 27, 255)), 
                (111, (255, 0, 0, 255)),     # Continous Urban Fabric 
                (112, (255, 112, 112, 255)), # Discontinous Urban Fabric
                (120, (67, 62, 67, 255)),
                (121, (255, 19, 82, 255)),
                (122, (112, 112, 112, 255)), # Roads & Railways
                (123, (148, 0, 148, 255)),
                (124, (203, 0, 203, 255)),
                (130, (127, 109, 37, 255)),
                (131, (255, 0, 255, 255)),
                (132, (255, 0, 255, 255)),
                (133, (81, 9, 19, 255)),
                (140, (255, 213, 214, 255)),
                (141, (255, 200, 207, 255)),
                (142, (255, 200, 211, 255)),
                (200, (255, 246, 118, 255)),
                (210, (250, 198, 3, 255)),   #
                (211, (185, 255, 79, 255)),
                (212, (195, 255, 28, 255)),
                (213, (205, 255, 21, 255)),
                (220, (160, 255, 16, 255)),
                (221, (146, 211, 84, 255)),  #
                (222, (154, 228, 26, 255)),  #
                (223, (92, 186, 33, 255)),
                (230, (80, 255, 57, 255)),
                (231, (134, 251, 105, 255)), #
                (240, (180, 255, 68, 255)),
                (241, (183, 255, 74,255)),
                (242, (43, 255, 60,255)),
                (243, (55, 255, 52,255)),
                (244, (112, 255, 96,255)),
                (300, (26, 182, 23, 255)),
                (310, (15, 130, 11, 255)),   #
                (311, (31, 209, 0, 255)),
                (312, (0, 81, 0, 255)),      #
                (313, (0, 193, 0, 255)),     #
                (320, (60, 255, 34, 255)),
                (321, (93, 242, 73, 255)),   #
                (322, (23,255,124,255)),
                (323, (65,255,103,255)),
                (324, (86, 160, 63, 255)),   #
                (330, (179, 255, 57,255)),
                (331, (246, 255, 173, 255)),
                (332, (238, 255, 174, 255)), #
                (333, (201, 246, 176, 255)), #
                (334, (46, 61, 23, 255)),
                (335, (255, 255, 255, 255)), #
                (400, (81, 194, 180, 255)), # Wetlands
                (411, (55, 255, 158, 255)),
                (422, (145, 255, 187, 255)),
                (500, (117, 249, 233, 255)),  # Water bodies
                (511, (117, 249, 233, 255)),
                (512, (15, 175, 255, 255))
    ))
    
    corine_labels = dict((
                (0, ('Zero Class')),
                (100, ('Artificial')),
                (110, ('Urban fabric')),
                (111, ('Cont. urban fabric')),
                (112, ('Disc. urban fabric')),
                (120, ('Industrial/Commercial/Transport units')),
                (121, ('Industrial/commercial units')),
                (122, ('Road/rail networks, associated land')),
                (123, ('Port areas')),
                (124, ('Airport')),
                (130, ('Mine/dump/construction sites')),
                (131, ('Mineral extraction sites')),
                (132, ('Dump sites')),
                (133, ('Construction sites')),
                (140, ('Artificial/non-agricultural vegetated areas')),
                (141, ('Green urban areas')),
                (142, ('Sport/Leisure facilities')),
                (200, ('Agricultural')),
                (210, ('Arable land')),
                (211, ('Non-irrigated arable land')),
                (212, ('Permanently irrigated land')),
                (213, ('Rice fields')),
                (220, ('Permanent crops')),
                (221, ('Vineyards')),
                (222, ('Fruit trees/berry plantations')),
                (223, ('olive groves')),
                (230, ('Pastures')),
                (231, ('Pastures')),
                (240, ('Heterogenous agricultural areas')),
                (241, ('Annual crops/Permanent crops')),
                (242, ('Complex cultivation patterns')),
                (243, ('agricultur/significant areas of natural veg.')),
                (244, ('Agro-forestry areas')),
                (300, ('Forest/semi natural')),
                (310, ('Forest')),
                (311, ('Broad-leaved forest')),
                (312, ('Coniferous forest')),
                (313, ('Mixed forest')),
                (320, ('Scrub/herbaceous veg.')),
                (321, ('Natural grasslands')),
                (322, ('Moors/heathland')),
                (323, ('Sclerophyllous veg.')),
                (324, ('Transitional woodland-shrub')),
                (330, ('Open spaces w/ little veg.')),
                (331, ('Beaches/Dunes/Sands')),
                (332, ('Bare rocks')),
                (333, ('Sparsely vegetated areas')),
                (334, ('Burnt areas')),
                (335, ('Glaciers and perpetual snow')),
                (400, ('Wetlands')),
                (410, ('Inland wetlands')),
                (411, ('Inland marshes')),
                (412, ('Peat bogs')),
                (420, ('Maritime wetlands')),
                (421, ('Salt marshes')),
                (422, ('Salines')),
                (423, ('Intertidal flats')),
                (500, ('Inland waters')),
                (510, ('Water courses')),
                (511, ('Water bodies')),
                (512, ('Marine waters')),
                (520, ('Coastal lagoons')),
                (521, ('Estuaries')),
                (522, ('Sea/ocean'))
    ))

    # Normalize the color values
    for k in corine_colors:
        v = corine_colors[k]
        _v = [_v / 255.0 for _v in v]
        corine_colors[k] = _v
        
    keys = np.unique(classified_image)
    index_colors = [None]*len(keys)
    
    for i in range (0, len(keys)):
        key = keys[i]
        if key in corine_colors:
            index_colors[i] = corine_colors[keys[i]]
        else:
            index_colors[i] = (255, 255, 255, 0) 
            print('the following label has no defined color: ', key)
    
    # Create cmap object and discrete norm with exact number of classes/colors present in the current classification result
    corine_cmap = plt.matplotlib.colors.ListedColormap(index_colors, name='corine', N=len(keys))
    corine_norm = plt.matplotlib.colors.BoundaryNorm(keys, ncolors=corine_cmap.N)
    
    handles = []
    for key in keys:
        patch = mpatches.Patch(color=corine_colors[key], label=corine_labels[key])
        handles.append(patch)
            
    return corine_cmap, corine_norm, handles

def spatial_autocorrelation():
    print('work in progress')
    # TODO: Implement a statistical valid test on spatial autocorrelation for the random sampling
    ###########################
    # Spatial autocorrelation #
    ###########################
    # Sources:
    # https://github.com/pysal/notebooks/blob/master/notebooks/PySAL_esda.ipynb
    # http://pysal.readthedocs.io/en/latest/users/tutorials/autocorrelation.html#moran-s-i
    # https://2015.foss4g-na.org/sites/default/files/slides/Intro%20to%20Spatial%20Data%20Analysis%20in%20Python%20-%20FOSS4G%20NA%202015.pdf
    #
    # 1. Global - quantifies clustering/dispersion across a region
    #    a. values ~ 1.0: highly clustered
    #    b. values ~0.0: no spatial autocorrelation
    #    c. values ~ -1.0: highly dispersed
    #
    #
    # import pysal as ps
    # w = ps.lat2W(training_samples.shape[0], training_samples.shape[1])
    # mr = ps.Moran(training_samples, w)
    # print(mr.I)
    #
    ##Global spatial autocorrelation:
    ##random: mr.I = -0.00042470680516889457
    ##equal: mr.I = 0.0831439481594
    ##proportional: mr.I = -0.000509175447953
    #
    #
    ##2. Local - identifies clusters (hot-spots) within the region
    # import pysal as ps
    # w = ps.lat2W(training_samples.shape[0], training_samples.shape[1])
    # mr = ps.Moran_Local(training_samples, w)
