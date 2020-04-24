# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 12:53:15 2020

@author: bav
"""

import matplotlib.pyplot as plt
import bav_lib as bl
import numpy as np
import h5py
import pandas as pd
from pandas import HDFStore

filename = "data/FirnCoverData_2.0_2019_11_20.h5"

f =  h5py.File(filename, "r")
# List all groups
print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[0]
print("Keys: %s" % f[a_group_key].keys())
# Get the data
dset = f[a_group_key]

Air_Temp_Hourly = pd.DataFrame(dset['Air_Temp_Hourly'][:])
Air_Temp_Hourly['sitename'] =[x.decode('utf-8') for x in Air_Temp_Hourly['sitename'].to_list()]

Compaction_Daily = pd.DataFrame(dset['Compaction_Daily'][:])
Compaction_Daily['sitename'] =[x.decode('utf-8') for x in Compaction_Daily['sitename'].to_list()]

Compaction_Instrument_Metadata = pd.DataFrame(dset['Compaction_Instrument_Metadata'][:])
Compaction_Instrument_Metadata['sitename'] =[x.decode('utf-8') for x in Compaction_Instrument_Metadata['sitename'].to_list()]

Firn_Temp_Daily = pd.DataFrame(dset['Firn_Temp_Daily'][:])
Firn_Temp_Daily['sitename'] =[x.decode('utf-8') for x in Firn_Temp_Daily['sitename'].to_list()]

Meteorological_Daily = pd.DataFrame(dset['Meteorological_Daily'][:])
Meteorological_Daily['sitename'] =[x.decode('utf-8') for x in Meteorological_Daily['sitename'].to_list()]

Station_Visit_Notes = pd.DataFrame(dset['Station_Visit_Notes'][:])
Station_Visit_Notes['sitename'] =[x.decode('utf-8') for x in Station_Visit_Notes['sitename'].to_list()]

Station_Metadata = pd.DataFrame(dset['Station_Metadata'][:])
Station_Metadata['sitename'] =[x.decode('utf-8') for x in Station_Metadata['sitename'].to_list()]


    
