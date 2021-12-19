# -*- coding: utf-8 -*-
"""
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""
import numpy as np
import pandas as pd
import tables as tb
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates


df_dens = pd.read_csv('data/sumup_greenland.csv')


df_dens = df_dens.loc[np.isin(df_dens.Citation, [45, 47])]
coreid = df_dens.coreid.unique()

fig, ax = plt.subplots(1,6)
for ind in coreid:
    df = df_dens.loc[df_dens.coreid==ind,:]
    break
    
    
