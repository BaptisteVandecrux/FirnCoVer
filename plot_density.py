#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2019 Max Stevens <maxstev@uw.edu>
#
# Distributed under terms of the MIT license.

"""
Created on %(date)s

@author: %(username)s

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""

import netCDF4 as nc
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import sys


'''
This script processes the sumup netCDF-formatted database of firn cores and puts
them into pandas dataframes, saved in pickles (.pkl files). 

It puts the data into two pandas dataframes (one for Antarctica and one for 
Greenland). The data is saved in a pickle for easy reloading (the processing is 
slow because the script needs to find unique values.) 

This may be of use in other scripts as an easy way to pull in data from the
sumup database.

The changes that I make from the original sumup database are:
- There is a core from the 1950's (likely a Benson core from Greeland) that has
the lat/lon listed as -9999. I set the coordinates of the core to 75N, 60W.
- For cores that do not have a specific date, Lynn put 00 for the month and day.
I use January 1 for all of those cores so that I can create a datetime object.

Optionally, you can choose to write metadata to a csv file that can be imported
into google earth to see the locations of each core.

I have not done this, but it should be easy to do a csv write with only cores 
e.g. deeper than some depth.
'''

### Load the data.
su = nc.Dataset('data/sumup_density_2020.nc','r+')
df = pd.DataFrame()

df['latitude']=su['Latitude'][:]
df['longitude']=su['Longitude'][:]
df['elevation']=su['Elevation'][:]
date=su['Date'][:].astype(int).astype(str)
for kk,dd in enumerate(date):
    yr=dd[0:4]
    mo=dd[4:6]
    dy=dd[6:]
    if mo=='00':
        mo='01'
    elif mo=='90':
        mo = '01'
    elif mo=='96':
        mo = '06'
    if dy == '00':
        dy='01'
    elif dy == '32':
        dy='31'
    date[kk]=yr+mo+dy
df['date'] = pd.to_datetime(date)

df['density']=su['Density'][:]
df['depth_top']=su['Start_Depth'][:]
df['depth_bot']=su['Stop_Depth'][:]
df['depth_mid']=su['Midpoint'][:]
df['error']=su['Error'][:]
df['citation_num']=su['Citation'][:]
df[df==-9999]=np.nan

ind= df.latitude<-100
df.loc[ind,'longitude']=-60
df.loc[ind,'latitude']=75

ind = np.isnan(df.depth_mid)
df.loc[ind,'depth_mid'] = df.loc[ind,'depth_top'] + (df.loc[ind,'depth_bot'] - df.loc[ind,'depth_top'])/2

df = df.loc[df.latitude>0]
df=df.reset_index()
df = df.loc[np.isin(df.citation_num, [45, 47, 48])]
df=df.loc[df['index'] != 838078, :]
df['core_id'] = ((df.depth_mid.diff()<-0.2)+0).cumsum()

#%%
plt.figure()
df.depth_top.plot()
df.depth_bot.plot()
df.depth_mid.plot()
df.depth_mid.diff().plot()
((df.depth_mid.diff()<-0.2)+0).cumsum().plot(secondary_y=True)

df_save = df.copy()
#%% 
df = df_save.copy()

#  Giving core name
sumup_citations = pd.read_csv('data/sumup_citations.txt', sep=';.-',header=None,engine='python',encoding='ANSI')  

df['citation'] = [sumup_citations.values[int(i)-1][0] for i in df['citation_num']]
df['name'] = [c.split(' ')[0].split(',')[0] +'_'+str(d.year) for c,d in zip(df['citation'],df['date'])]

for source in df.citation.unique():
    core_id_in_source = df.loc[df.citation==source].core_id
    if len(core_id_in_source.unique())>1:
        df.loc[df.citation==source, 'name'] = [n1 + '_' + str(n2 - core_id_in_source.min() + 1) for n1,n2 in zip(df.loc[df.citation==source, 'name'] , core_id_in_source)]
        
df['maxdepth'] = np.nan
for core in df.core_id.unique():
    maxdepth = np.max((df.loc[df.core_id==core].depth_bot.values[-1],
                       df.loc[df.core_id==core].depth_mid.values[-1]))
    df.loc[df.core_id==core,'maxdepth']=maxdepth
df = df.set_index('core_id')
# df = df.drop(columns=('level_0','index'))

site_list = pd.read_csv('data/FirnCover_location.csv')

df['site'] = ''
for core in df.index.get_level_values(0).unique():
    lon_core = df.loc[core].longitude.values[0]
    lat_core = df.loc[core].latitude.values[0]
    dist = (site_list.longitude-lon_core)**2 + (site_list.latitude-lat_core)**2
    if np.min(dist)*111*1000<200:
        closest = site_list.loc[dist == np.min(dist), 'site'].values
    else:
        closest = []
    if len(closest)>0:
        df.loc[core,'site'] = closest[0]
    print(core, np.floor(np.min(dist)*111*1000), df.loc[core,'site'].iloc[0])

tmp, ind = np.unique(df.name,return_index=True)
core_list = df.iloc[ind,:].reset_index()
core_list.to_csv('data/sumup_firncover.csv')

df['name'] = [site +' '+str(d.year) for site, d in zip(df['site'],df['date'])]
df = df.loc[df.site!='',:]

# %% Plotting

fig, ax = plt.subplots(2,4,figsize=(9,6), sharex=True, sharey=True)
plt.subplots_adjust(left=0.08, right=0.95, top=0.95, 
                    bottom = 0.1, wspace=0.15, hspace=0.2)
ax = ax.flatten()

cores = [0, 5, 8, 13, 22, 27, 37, 30]
# cores = core_list.loc[core_list.site == 'KAN_U','core_id']
for count, core in enumerate(cores):
    ax[count].step(df.loc[core].density*1000, -df.loc[core].depth_mid,   where='mid')
    ax[count].set_title(df.loc[core].name.unique()[0])
    ax[count].set_xlim(200,950)
    ax[count].set_ylim(-15, 0)
    ax[count].grid()
fig.text(0.5, 0.02, 'Density (kg m$^{-3}$)', ha='center', size = 12)
fig.text(0.02, 0.5, 'Depth (m)', va='center', rotation='vertical', size = 12)
plt.savefig('core_all.png')

        

