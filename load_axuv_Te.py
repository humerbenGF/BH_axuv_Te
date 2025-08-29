import os
import bson
import numpy as np

import GF_data_tools as gdt

from BH_axuv_Te.axuv_Te_function_with_uncertainty_gftools_noplot_v2_202408 import axuv_Te_with_error    # type: ignore

def load_axuv_Te_data(shot_number):
    try:
        filename = os.environ['AURORA_REPOS'] + "/BH_axuv_Te/axuv_Te_data/" + str(int(shot_number / 1000)*1000) + "/" + "axuv_Te_data_" + str(shot_number) + ".bson"
        
        if not os.path.isdir(os.environ['AURORA_REPOS'] + "/BH_axuv_Te/axuv_Te_data/"):
            os.mkdir(os.environ['AURORA_REPOS'] + "/BH_axuv_Te/axuv_Te_data/")
        if not os.path.isdir(os.environ['AURORA_REPOS'] + "/BH_axuv_Te/axuv_Te_data/" + str(int(shot_number / 1000)*1000) + "/"):
            os.mkdir(os.environ['AURORA_REPOS'] + "/BH_axuv_Te/axuv_Te_data/" + str(int(shot_number / 1000)*1000) + "/")
        
        # Read the BSON data from the file
        if os.path.isfile(filename):
            with open(filename, 'rb') as f:
                bson_data = f.read()


        # Decode the BSON data to retrieve the dictionary
        data = bson.decode_all(bson_data)[0]
        print(f"Retrieved AXUV Te for Shot {shot_number}")
        
    except:
        print(f"Calculating AXUV Te for Shot {shot_number}")
        
        data = axuv_Te_with_error(shot_number)
        print(data)
        
        for k in data.keys():
            if type(data[k]) == type(np.array([])):
                data[k] = list(data[k])

        # Serialize the data to BSON format and save to a file
        if type(data) == type(dict()):
            with open(filename, "wb") as file:
                file.write(bson.BSON.encode(data))
    
    for k in data.keys():
        if type(data[k]) == type(list()):
            data[k] = np.array(data[k])
    
    return data