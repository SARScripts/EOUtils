def wcps_rasdaman(query, ip='saocompute.eurac.edu/sincohmap', file_name='',verbose=False):
    """
    Sends a WCPS query to a Rasdaman server and wraps the response for further use in Python depending on the 
    the response format chosen in the query.
    
    Args: 
        query (str) -- WCPS query you want to send to the Rasdaman server
        ip (str) -- IP of Rasdaman server (default saocompute.eurac.edu)
    
    Returns:
        Either one of the following
        - Numpy array for JSON/CSV formatted response
        - Xarray Dataset for a netCDF formatted response
        - Filepath to a TIFF/JPEG/JP2/PNG file saved to disk, in respect to the response image type
        - The response object, when the response could not be processed
    
    Sources:
        http://xarray.pydata.org/en/stable/io.html#netcdf
        http://xarray.pydata.org/en/stable/data-structures.html
        
    Author: Harald Kristen, Alexander Jacob
    Date: 2019-05-29
    """

    import requests
    import json
    import werkzeug
    import numpy as np
    import io
    import xarray as xr
    import uuid
    import os
    import xml.etree.ElementTree as ET
    import base64

    #set the work directory
    work_directory = ''#os.getcwd()
    
    #print('WCPS init')
    
    if ip == 'saocompute.eurac.edu/sincohmap' or ip == 'saocompute.eurac.edu':
        url = 'http://' + ip + '/rasdaman/ows?SERVICE=WCS&VERSION=2.0.1&REQUEST=ProcessCoverages'  
    else:
        url = 'http://' + ip + ':8080/rasdaman/ows?SERVICE=WCS&VERSION=2.0.1&REQUEST=ProcessCoverages'
    
    #Fix the special characters used the input query like ' ', $ and so on
    query = werkzeug.url_fix(query)
    print('This is the URL, used for the request:\n' + url + '&query=' + query)
    url = url + '&query=' + query

    try:
        #Send the request to Rasdaman and save the response in the variable "r"
        r = requests.get(url, stream=True)
    except Exception as ex:
        print(tpye(ex))
        print(ex.args)
        print(ex)

    #If there is an error, plot the error message that comes from Rasdaman & exit script
    if r.status_code != requests.codes.ok:
        print('HTTP Error ' + str(r.status_code))
        root = ET.fromstring(r.text)
        for element in root.iter():
            if element.tag == '{http://www.opengis.net/ows/2.0}ExceptionText':
                print(element.text)

    print('currently receiving content of type: ' + r.headers['Content-Type'])

    #print('This is the URL, used for the request \n' + r.url)

    if r.headers['Content-Type'] == 'text/plain':
        #print('return type is text')
        # Convert CSV or json to NumpyArray
        response_text = r.text()
        output = response_text
        # The JSON version also works for 2D arrays.
        if response_text.startswith("{"):
            loaded = r.json()
            output = np.array(loaded)
        else:
            output = np.fromstring(response_text[1:-1], dtype = float, sep = ',')

    elif r.headers['Content-Type'] == 'application/json':
        # Convert JSON to NumpyArray
        loaded = r.json()
        output = np.array(loaded)

    elif r.headers['Content-Type'] == 'application/netcdf':
        print(r.headers)
        # create x array dataset from input stream
        if file_name == '':
            file_name = 'wcps_' + str(uuid.uuid4()) + '.nc'
        #print('the following file has been saved locally: ' + file_name)
        with io.open(file_name, 'wb') as outfile:
            outfile.write(r.content)        
        output_open = xr.open_dataset(file_name)
        # Xarray is normally lazy loading netCDF files
        # As we want to perform intense computation, we load the file directly in the main memory with Dataset.load()
        output = xr.Dataset.load(output_open)
        os.remove(file_name)

    elif r.headers['Content-Type'] in ['image/tiff', 'image/png', 'image/jp2', 'image/jpeg']:
        # Write response in choosen image format to disk and print filepath
        image_type = r.headers['Content-Type']
        if file_name == '':
            file_ending = image_type[6:]
            # write TIFF to disk and print filepath
            tf = 'wcps_' + str(uuid.uuid4())
            file_name = tf + '.' + file_ending
        with io.open(file_name, 'wb') as outfile:
            outfile.write(r.content)
        print('the following file has been saved locally: ' + file_name)
        output = file_name

    else:
        output = r
        output_type = r.headers['Content-Type']
        print('The response could not be processed, as it is a ' + output_type)

    return output
