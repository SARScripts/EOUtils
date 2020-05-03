#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
#------------------------------------------------------------------------------
# Name:  landcover_classification.py
#
#   General purpose:
#       A set of modules for landcover classifcation with machine learning alghorithms. Right now RandomForests &
#       Support Vector Machineare implemented. As input any kind of satellite imagery (e.g Senintel-1/2, Landast) can
#       be used. As long as the input images are stacked into a single TIFF which all have the same spatial extent
#       and resolution. Also classification validation and plotting are supported.
#
# Author:   Harald Kristen <haraldkristen at posteo dot at>
# Date: 27.09.2017
#-------------------------------------------------------------------------------
"""

def prepare_training_data(vector_path, raster_path, column, nr_points, sampling_methodology = 'proportional'):
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
    import random
    from osgeo import gdal
    from osgeo import ogr
    from geo_utils import write_geotiff, create_raster_from_vector_clip

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
            coord = np.random.randint(extent, size=2)
            x = coord[0]
            y = coord[1]
            samples[x,y] = roi[x,y]
        return samples
    def random_sampling_equal(roi, training_labels, nr_points):
        """
        Produce a random sample of a 2D image, where every class has the same amount of sampling pixels
        Args:
            roi (2D ndarray): A 2D raster image for sampling
            training_labels (ndarray): A list with the names of the classes (=unique pixel values)
            nr_points (int): The number of pixels to be sampled

        Returns:
            samples (2D ndarray): A random equal sample of the input image, empty pixels = 0
        """
        # create empty np_array with all cell values = 0
        samples = np.zeros(roi.shape, dtype=float)
        for i in training_labels:
            # equally distribute the number of points to all classes
            nr_classes = training_labels.size
            nr_points_per_class = round(nr_points / nr_classes)
            # if the class has a Nodata value like 0 or -999 pass
            if i == 0 or i == -999:
                pass
            else:
                # subset only one class of the ROI
                roi_select = roi * (roi == i)
                count = 0
                # loop through the subset &
                while count != nr_points_per_class:
                    coord = np.random.randint(roi_select.shape[0], size=2)
                    x = coord[0]
                    y = coord[1]
                    samples[x, y] = roi_select[x, y]
                    count = count + 1
                    
                    
                    
        return samples
    def random_sampling_proportional(roi, training_labels, nr_points):
        """
        Produce a random sample of a 2D image, where the number of samples in one class is proportional to the total
        number of pixels in this class. The minimum number of pixels per class is 1% of all input pixels.
        Args:
            roi (2D ndarray): A 2D raster image for sampling
            training_labels (ndarray): A list with the names of the classes (=unique pixel values)
            nr_points (int): The number of pixels to be sampled

        Returns:
            samples (2D ndarray): A random proportional sample of the input image, empty pixels = 0
        """
        # create empty np_array with all cell values = 0
        samples = np.zeros(roi.shape, dtype=float)
        # number of points proportional to class size
        total_pixel = roi.size
        for i in training_labels:
            # proportianlly distribute the number of points to all classes in respect to their class size (e.g. nr of pixels)
            class_size = np.count_nonzero(roi == i)
            nr_points_per_class = round(nr_points * (class_size / total_pixel))
            # make sure that there are at least a few sampling points in every class
            # --> minimum nr_points_per_class = 1%
            if nr_points_per_class < (nr_points * 0.01):
                nr_points_per_class = int(nr_points * 0.01)
                if nr_points_per_class > class_size:
                    nr_points_per_class = class_size
            # if the class has a Nodata value like 0 or -999 pass
            if i == 0 or i == -999:
                pass
            else:
                # subset only one class of the ROI
                roi_select = roi * (roi == i)
                roi_select_index = np.nonzero(roi_select)
                
                rg = range(0,class_size)
                rSamp = random.sample(rg,nr_points_per_class)
                
                rSelIdx = [roi_select_index[0][rSamp],roi_select_index[1][rSamp]]
                samples[rSelIdx] = roi_select[rSelIdx]
        return samples

    ##############################
    # Rasterize the vector layer #
    ##############################
    # Open the dataset from the file
    
    cwd = os.getcwd()
    os.chdir(os.path.dirname(vector_path))
    print(os.getcwd())
    vector = ogr.Open(vector_path)
    # Print some general info about the shapefile
    general_info(vector)
    os.chdir(cwd)
    
    # Tie-in vector dataset with Raster dataset ( = rasterize vector)
    print(raster_path)
    raster_ds = gdal.Open(raster_path)
    # Fetch number of rows and columns
    ncol = raster_ds.RasterXSize
    nrow = raster_ds.RasterYSize
    # Fetch projection and extent
    proj = raster_ds.GetProjectionRef()
    ext = raster_ds.GetGeoTransform()
    raster_ds = None
    #create_raster_from_vector(vector_path, ncol, nrow, proj, ext, column)
    gtraster_path = os.path.splitext(vector_path)[0] + '.tif'
    create_raster_from_vector_clip(vector_path, gtraster_path, ncol, nrow, ext, proj, column)
    
    ############################
    # Random sample generation #
    ############################create_raster_from

    # Check the rasterized layer
    roi_ds = gdal.Open(gtraster_path, gdal.GA_ReadOnly)
    roi = roi_ds.GetRasterBand(1).ReadAsArray()

    # How many pixels are in each class?
    training_labels = np.unique(roi)
    # Iterate over all class labels in the ROI image, printing out some information
    for c in training_labels:
                print('Class {c} contains {n} pixels'.format(c=c, n=(roi == c).sum()))

    extent = roi.shape[0] #extent of the array
    roi = np.array(roi).astype(np.float) #convert array to numpy_array in FLOAT

    if sampling_methodology == 'proportional':
        training_pixels = random_sampling_proportional(roi, training_labels, nr_points)
        test_pixels = random_sampling_proportional(roi, training_labels, nr_points)
    elif sampling_methodology == 'equal':
        training_pixels = random_sampling_equal(roi, training_labels, nr_points)
        test_pixels = random_sampling_equal(roi, training_labels, nr_points)
    elif sampling_methodology == 'random':
        training_pixels = random_sampling(roi, nr_points)
        test_pixels = random_sampling(roi, nr_points)
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
        bands_data.append(band.ReadAsArray())

    bands_data = np.dstack(bands_data)

    #training dataset
    is_train = np.nonzero(training_pixels)
    training_labels = training_pixels[is_train]
    training_samples = bands_data[is_train]

    #test dataset
    is_test = np.nonzero(test_pixels)
    test_labels = test_pixels[is_test]
    test_samples = bands_data[is_test]

    # clean up
    roi_ds = None  # close file again
    #os.remove('rasterized.gtif')

    return training_samples, training_labels, bands_data, projection, geo_transform, is_test, test_labels, test_samples

def classification(training_samples, training_labels, test_labels, test_samples, bands_data, projection, geo_transform, classifier='rf', gridsearch = True, **kwargs):
    """
    This module does landcover classification using the scikit-learn machine learning library. Right now Random Forests
    & Support Vector Machine algorithms are implemented.

    Args:
        training_samples (2D ndarray): Training sample of the input raster dataset with shape = (rows*cols, bands)
        training_labels (1D ndarray): Training labels as ndarray with (shape = rows*cols)
        test_labels (1D ndarray): Test labels as ndarray with (shape = rows*cols)
        test_samples (2D ndarray): Test sample of the input raster dataset with shape = (rows*cols, bands)
        bands_data (3D ndarray): The input raster dataset as ndarray with shape = (rows, cols, bands)
        projection (str): Projection definition string (Returned by gdal.Dataset.GetProjectionRef)
        geo_transform (tuple): Returned value of gdal.Dataset.GetGeoTransform (coefficients for transforming between
                                 pixel/line (P,L) raster space, and projection coordinates (Xp,Yp) space.
        classifier (str): The classification method to be used.
                          Either 'rf'=Random Forests or 'svm'=Support Vector Machine
        gridsearch (bool): Either 'True' or 'False' -> Use a Gridsearch of a grid already defined in the code.
        **kwargs:
    Returns:
        classifier: The scikit-learn classifier object
        classified image (TIFF): The classififed image, written to disk as georeferenced TIFF.
        classified image (2D ndarray): The classified image as Numpy Array to directly use in Python.
    """

    from geo_utils import write_geotiff

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

            from sklearn.model_selection import RandomizedSearchCV
            import numpy as np
            from scipy import stats

            classifier_baseline = svm.SVC(kernel='rbf',cache_size=20000)

            #param_grid = {'gamma': [10**-8,10**-7,10**-6,10**-5,10**-4,10**-3,10**-2,10**-1,10,100], 'C': range(1,100,10)}
            # use a random parameter grid as shown in: http://scikit-learn.org/stable/modules/grid_search.html
            param_grid = {'C': stats.expon(scale=100), 'gamma': stats.expon(scale=.01)}

            ## tune the hyperparameters via a randomized search (100 iterations & computation on all cores)
            grid = RandomizedSearchCV(classifier_baseline, param_grid, n_iter=100, n_jobs=-1)

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
    classified_image = result.reshape((rows, cols))

    # write TIFF with labels to disk
    write_geotiff('classified_image.tiff', classified_image, geo_transform, projection)

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
    classes = np.unique(test_labels)

    ### Confusion matrix with seaborn
    sns.set()
    labels = list(map(int, classes))
    mat = metrics.confusion_matrix(test_labels, predicted_labels)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=labels, yticklabels=labels)
    plt.xlabel('True label')
    plt.ylabel('Predicted label')
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

def plot_classified_image(classified_image, level):
    """
    Plots the classified image for CORINE LEVEL 1
    Args:
        classified_image (2D ndarray): The classified image from the classification module.
        level (int): The CORINE class level, right now only level 1 is implemented.

    Returns:
        Plot of the classified image
    """
    from matplotlib import pyplot as plt
    import matplotlib.patches as mpatches

    if level == 1:
    # Setup a colormap for LC classification (Level 1)
        colors = dict((
            (100, (255, 0, 0, 255)),  # Urban
            (200, (0, 255, 0, 255)),  # Agricultural
            (300, (0, 150, 0, 255)),  # Forest
            (400, (160, 82, 45, 255)),  # Wetlands
            (500, (0, 0, 255, 255))  # Water bodies
        ))

        # Normalize the color values
        for k in colors:
            v = colors[k]
            _v = [_v / 255.0 for _v in v]
            colors[k] = _v

        index_colors = [colors[key] if key in colors else
                        (255, 255, 255, 0) for key in [100, 200, 300, 400, 500]]

        # Create cmap object
        cmap = plt.matplotlib.colors.ListedColormap(index_colors, 'Classification')

        # Plot the classified image
        plt.style.use('classic')
        plt.figure(figsize=(10, 10))
        plt.imshow(classified_image, cmap)

        # Plot legend
        handles = []
        for key in colors:
            patch = mpatches.Patch(color=colors[key], label=str(key))
            handles.append(patch)
        plt.legend(handles=handles, title='LC classes', facecolor='white')

        plt.title('Classified Image')
    else:
        print('Right now you can only visualize LEVEL 1 ')

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
