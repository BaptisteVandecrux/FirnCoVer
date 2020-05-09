# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 12:53:15 2020

@author: bav
"""
# This notebook opens the Greenland FirnCover data and puts it into pandas dataframes.
# The core data comes from core_data_df.pkl, which is created by running firncover_core_data_df.py
# (The dataframe is created in that script.)

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys
import tables as tb
import datetime
import pandas as pd
import scipy.io as spio
from datetime import datetime, timedelta
import pickle
import h5py as h5
from dateutil import parser
import time
import seaborn as sns
from FirnCover_lib import *

# Default Settings for figures
sb=True
if sb:
    sns.set()
    sns.set_context('paper', font_scale=1.5, rc={"lines.linewidth": 1.5})
#     sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    sns.set_style("whitegrid",{"xtick.top":True, "xtick.bottom":True,"ytick.left":True, "ytick.right":True})
# plt.style.use('seaborn-notebook')

# pd.set_option('display.height', 500)
pd.set_option('display.max_rows', 500)
mpl.rcParams['axes.formatter.useoffset'] = False
mpl.rcParams.update({'figure.autolayout': True})
fontsz = 20
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
plt.rcParams['axes.titlesize'] = 22
plt.rcParams["axes.labelsize"] = 22
plt.rcParams['figure.figsize'] = [10, 8]
palette = sns.color_palette()

# Things to change
figex = '_200504.png' #extension for figures (name plus type)
pathtodata = './data'
date1 = '2020_04_27' # Date on the hdf5 file

# %matplotlib inline

# with open('/Users/maxstev/Documents/Grad_School/Research/FIRN/GREENLAND_CVN/Data/CVN_DATA/core_data_dict.pkl','rb') as f:
#     d=pickle.load(f)
# core_df = pd.DataFrame.from_dict(d,orient='index')
# core_df.index.name='coreid'

sites=['Summit','KAN-U','NASA-SE','Crawford','EKT','Saddle','EastGrip','DYE-2']

### Import the core depth/density data.
# with open(os.path.join(pathtodata,'core_data_df.pkl'),'rb') as f:
#     core_df=pickle.load(f)

### Import the FirnCover data tables. 
filename='FirnCoverData_2.0_' + date1 + '.h5'
filepath=os.path.join(pathtodata,filename)
CVNfile=tb.open_file(filepath, mode='r', driver="H5FD_CORE")
datatable=CVNfile.root.FirnCover
epoch =np.datetime64('1970-01-01')

inst_meta_df = pd.DataFrame.from_records(datatable.Compaction_Instrument_Metadata[:])
inst_meta_df.sitename=inst_meta_df.sitename.str.decode("utf-8")
inst_meta_df.set_index('instrument_ID',inplace=True)
# inst_meta_df.loc[7,'borehole_bottom_from_surface_m']

datatable

airtemp_df = pd.DataFrame.from_records(datatable.Air_Temp_Hourly[:])
airtemp_df.sitename = airtemp_df.sitename.str.decode("utf-8")
airtemp_df['date'] = pd.to_datetime(airtemp_df.daynumber_YYYYMMDD,format='%Y%m%d')+pd.to_timedelta(airtemp_df.hournumber_HH,unit='h')
airtemp_df.set_index(['sitename','date'],inplace=True)

airtemp_df2 = airtemp_df['air_temp_C']

airtemp_df2.loc['KAN-U'].resample('a').mean()

# compaction_df.head(3)
# inst_meta_df.head(3)

### First set up the compaction data frame, basic.

compaction_df=pd.DataFrame.from_records(datatable.Compaction_Daily[:])
compaction_df.sitename=compaction_df.sitename.str.decode("utf-8")
compaction_df['date']=pd.to_datetime(compaction_df.daynumber_YYYYMMDD.values,format='%Y%m%d')
compaction_df.set_index(['instrument_id','date'],inplace=True)
compaction_df.drop(columns=["compaction_ratio","compaction_wire_correction_ratio","compaction_cable_distance_m"],inplace=True)
compaction_df.sort_index(inplace=True)

uids = compaction_df.index.get_level_values(0).unique()
compaction_df['hole_init_length']=0*compaction_df['compaction_borehole_length_m']
compaction_df['hole_botfromsurf']=0*compaction_df['compaction_borehole_length_m']
compaction_df['hole_topfromsurf']=0*compaction_df['compaction_borehole_length_m']

for ii in uids:
    compaction_df.loc[ii,'hole_init_length'] = -1*inst_meta_df.loc[ii,'borehole_initial_length_m']*np.ones_like(compaction_df.loc[ii,"compaction_borehole_length_m"].values)
    compaction_df.loc[ii,'hole_botfromsurf'] = -1*inst_meta_df.loc[ii,'borehole_bottom_from_surface_m']*np.ones_like(compaction_df.loc[ii,"compaction_borehole_length_m"].values)
    compaction_df.loc[ii,'hole_topfromsurf'] = -1*inst_meta_df.loc[ii,'borehole_top_from_surface_m']*np.ones_like(compaction_df.loc[ii,"compaction_borehole_length_m"].values)
#Filter saddle data.
compaction_df.drop(compaction_df[((compaction_df.sitename=='Saddle')&(compaction_df.index.get_level_values("date")<'2016-01-01'))].index,inplace=True)    

#Put the 'virtual' holes into the frame (i.e. the differential compaction between holes), Summit, EGRIP only for now.

### Summmit
cd2 = compaction_df.loc[31].copy()
cd2.hole_init_length = compaction_df.loc[30,'hole_init_length']-compaction_df.loc[31,'hole_init_length']
cd2.hole_botfromsurf = pd.concat([compaction_df.loc[30,'hole_botfromsurf'],compaction_df.loc[31,'hole_botfromsurf']],axis=1).max(axis=1)
cd2.hole_topfromsurf = pd.concat([compaction_df.loc[30,'hole_botfromsurf'],compaction_df.loc[31,'hole_botfromsurf']],axis=1).min(axis=1)
cd2.compaction_borehole_length_m = compaction_df.loc[30,'compaction_borehole_length_m']-compaction_df.loc[31,'compaction_borehole_length_m']
cd2['Compaction_Instrument_ID'] = int(101)
cd2.set_index(['Compaction_Instrument_ID',cd2.index],inplace=True)
compaction_df = pd.concat([compaction_df,cd2])
idx=compaction_df.index
compaction_df.index=compaction_df.index.set_levels([idx.levels[0].astype(int),idx.levels[1]])

cd2 = compaction_df.loc[31].copy()
cd2.hole_init_length = compaction_df.loc[30,'hole_init_length']-compaction_df.loc[32,'hole_init_length']
cd2.hole_botfromsurf = pd.concat([compaction_df.loc[30,'hole_botfromsurf'],compaction_df.loc[32,'hole_botfromsurf']],axis=1).max(axis=1)
cd2.hole_topfromsurf = pd.concat([compaction_df.loc[30,'hole_botfromsurf'],compaction_df.loc[32,'hole_botfromsurf']],axis=1).min(axis=1)
cd2.compaction_borehole_length_m = compaction_df.loc[30,'compaction_borehole_length_m']-compaction_df.loc[32,'compaction_borehole_length_m']
cd2['Compaction_Instrument_ID'] = int(102)
cd2.set_index(['Compaction_Instrument_ID',cd2.index],inplace=True)
compaction_df = pd.concat([compaction_df,cd2])
idx=compaction_df.index
compaction_df.index=compaction_df.index.set_levels([idx.levels[0].astype(int),idx.levels[1]])

cd2 = compaction_df.loc[31].copy()
cd2.hole_init_length = compaction_df.loc[32,'hole_init_length']-compaction_df.loc[31,'hole_init_length']
cd2.hole_botfromsurf = pd.concat([compaction_df.loc[32,'hole_botfromsurf'],compaction_df.loc[31,'hole_botfromsurf']],axis=1).max(axis=1)
cd2.hole_topfromsurf = pd.concat([compaction_df.loc[32,'hole_botfromsurf'],compaction_df.loc[31,'hole_botfromsurf']],axis=1).min(axis=1)
cd2.compaction_borehole_length_m = compaction_df.loc[32,'compaction_borehole_length_m']-compaction_df.loc[31,'compaction_borehole_length_m']
cd2['Compaction_Instrument_ID'] = int(103)
cd2.set_index(['Compaction_Instrument_ID',cd2.index],inplace=True)
compaction_df = pd.concat([compaction_df,cd2])
idx=compaction_df.index
compaction_df.index=compaction_df.index.set_levels([idx.levels[0].astype(int),idx.levels[1]])
#####

#### EastGRIP

# compaction_df.loc[28,'2015-05-28']=compaction_df.loc[28,'2015-05-29']
# compaction_df.loc[29,'2015-05-28']=compaction_df.loc[29,'2015-05-29']
compaction_df.drop((27,'2015-05-28'),inplace=True)
compaction_df.drop((26,'2015-05-28'),inplace=True)

cd2 = compaction_df.loc[26].copy()
cd2.hole_init_length = compaction_df.loc[26,'hole_init_length']-compaction_df.loc[28,'hole_init_length']
cd2.hole_botfromsurf = pd.concat([compaction_df.loc[26,'hole_botfromsurf'],compaction_df.loc[28,'hole_botfromsurf']],axis=1).max(axis=1)
cd2.hole_topfromsurf = pd.concat([compaction_df.loc[26,'hole_botfromsurf'],compaction_df.loc[28,'hole_botfromsurf']],axis=1).min(axis=1)
cd2.compaction_borehole_length_m = compaction_df.loc[26,'compaction_borehole_length_m']-compaction_df.loc[28,'compaction_borehole_length_m']
cd2['Compaction_Instrument_ID'] = int(104)
cd2.set_index(['Compaction_Instrument_ID',cd2.index],inplace=True)
compaction_df = pd.concat([compaction_df,cd2])
idx=compaction_df.index
compaction_df.index=compaction_df.index.set_levels([idx.levels[0].astype(int),idx.levels[1]])

cd2 = compaction_df.loc[26].copy()
cd2.hole_init_length = compaction_df.loc[26,'hole_init_length']-compaction_df.loc[27,'hole_init_length']
cd2.hole_botfromsurf = pd.concat([compaction_df.loc[26,'hole_botfromsurf'],compaction_df.loc[27,'hole_botfromsurf']],axis=1).max(axis=1)
cd2.hole_topfromsurf = pd.concat([compaction_df.loc[26,'hole_botfromsurf'],compaction_df.loc[27,'hole_botfromsurf']],axis=1).min(axis=1)
cd2.compaction_borehole_length_m = compaction_df.loc[26,'compaction_borehole_length_m']-compaction_df.loc[27,'compaction_borehole_length_m']
cd2['Compaction_Instrument_ID'] = int(105)
cd2.set_index(['Compaction_Instrument_ID',cd2.index],inplace=True)
compaction_df = pd.concat([compaction_df,cd2])
idx=compaction_df.index
compaction_df.index=compaction_df.index.set_levels([idx.levels[0].astype(int),idx.levels[1]])

cd2 = compaction_df.loc[28].copy()
cd2.hole_init_length = compaction_df.loc[28,'hole_init_length']-compaction_df.loc[27,'hole_init_length']
cd2.hole_botfromsurf = pd.concat([compaction_df.loc[28,'hole_botfromsurf'],compaction_df.loc[27,'hole_botfromsurf']],axis=1).max(axis=1)
cd2.hole_topfromsurf = pd.concat([compaction_df.loc[28,'hole_botfromsurf'],compaction_df.loc[27,'hole_botfromsurf']],axis=1).min(axis=1)
cd2.compaction_borehole_length_m = compaction_df.loc[28,'compaction_borehole_length_m']-compaction_df.loc[27,'compaction_borehole_length_m']
cd2['Compaction_Instrument_ID'] = int(106)
cd2.set_index(['Compaction_Instrument_ID',cd2.index],inplace=True)
compaction_df = pd.concat([compaction_df,cd2])
idx=compaction_df.index
compaction_df.index=compaction_df.index.set_levels([idx.levels[0].astype(int),idx.levels[1]])

### NASA-SE
cd2 = compaction_df.loc[13].copy()
cd2.hole_init_length = compaction_df.loc[13,'hole_init_length']-compaction_df.loc[14,'hole_init_length']
cd2.hole_botfromsurf = pd.concat([compaction_df.loc[13,'hole_botfromsurf'],compaction_df.loc[14,'hole_botfromsurf']],axis=1).max(axis=1)
cd2.hole_topfromsurf = pd.concat([compaction_df.loc[13,'hole_botfromsurf'],compaction_df.loc[14,'hole_botfromsurf']],axis=1).min(axis=1)
cd2.compaction_borehole_length_m = compaction_df.loc[13,'compaction_borehole_length_m']-compaction_df.loc[14,'compaction_borehole_length_m']
cd2['Compaction_Instrument_ID'] = int(107)
cd2.set_index(['Compaction_Instrument_ID',cd2.index],inplace=True)
compaction_df = pd.concat([compaction_df,cd2])
idx=compaction_df.index
compaction_df.index=compaction_df.index.set_levels([idx.levels[0].astype(int),idx.levels[1]])

cd2 = compaction_df.loc[13].copy()
cd2.hole_init_length = compaction_df.loc[13,'hole_init_length']-compaction_df.loc[15,'hole_init_length']
cd2.hole_botfromsurf = pd.concat([compaction_df.loc[13,'hole_botfromsurf'],compaction_df.loc[15,'hole_botfromsurf']],axis=1).max(axis=1)
cd2.hole_topfromsurf = pd.concat([compaction_df.loc[13,'hole_botfromsurf'],compaction_df.loc[15,'hole_botfromsurf']],axis=1).min(axis=1)
cd2.compaction_borehole_length_m = compaction_df.loc[13,'compaction_borehole_length_m']-compaction_df.loc[15,'compaction_borehole_length_m']
cd2['Compaction_Instrument_ID'] = int(108)
cd2.set_index(['Compaction_Instrument_ID',cd2.index],inplace=True)
compaction_df = pd.concat([compaction_df,cd2])
idx=compaction_df.index
compaction_df.index=compaction_df.index.set_levels([idx.levels[0].astype(int),idx.levels[1]])

cd2 = compaction_df.loc[13].copy()
cd2.hole_init_length = compaction_df.loc[15,'hole_init_length']-compaction_df.loc[14,'hole_init_length']
cd2.hole_botfromsurf = pd.concat([compaction_df.loc[15,'hole_botfromsurf'],compaction_df.loc[14,'hole_botfromsurf']],axis=1).max(axis=1)
cd2.hole_topfromsurf = pd.concat([compaction_df.loc[15,'hole_botfromsurf'],compaction_df.loc[14,'hole_botfromsurf']],axis=1).min(axis=1)
cd2.compaction_borehole_length_m = compaction_df.loc[15,'compaction_borehole_length_m']-compaction_df.loc[14,'compaction_borehole_length_m']
cd2['Compaction_Instrument_ID'] = int(109)
cd2.set_index(['Compaction_Instrument_ID',cd2.index],inplace=True)
compaction_df = pd.concat([compaction_df,cd2])
idx=compaction_df.index
compaction_df.index=compaction_df.index.set_levels([idx.levels[0].astype(int),idx.levels[1]])
#######

### Crawford
# 22-24
cd2 = compaction_df.loc[22].copy()
cd2.hole_init_length = compaction_df.loc[22,'hole_init_length']-compaction_df.loc[24,'hole_init_length']
cd2.hole_botfromsurf = pd.concat([compaction_df.loc[22,'hole_botfromsurf'],compaction_df.loc[24,'hole_botfromsurf']],axis=1).max(axis=1)
cd2.hole_topfromsurf = pd.concat([compaction_df.loc[22,'hole_botfromsurf'],compaction_df.loc[24,'hole_botfromsurf']],axis=1).min(axis=1)
cd2.compaction_borehole_length_m = compaction_df.loc[22,'compaction_borehole_length_m']-compaction_df.loc[24,'compaction_borehole_length_m']
cd2['Compaction_Instrument_ID'] = int(110)
cd2.set_index(['Compaction_Instrument_ID',cd2.index],inplace=True)
compaction_df = pd.concat([compaction_df,cd2])
idx=compaction_df.index
compaction_df.index=compaction_df.index.set_levels([idx.levels[0].astype(int),idx.levels[1]])

# 22-25
cd2 = compaction_df.loc[22].copy()
cd2.hole_init_length = compaction_df.loc[22,'hole_init_length']-compaction_df.loc[25,'hole_init_length']
cd2.hole_botfromsurf = pd.concat([compaction_df.loc[22,'hole_botfromsurf'],compaction_df.loc[25,'hole_botfromsurf']],axis=1).max(axis=1)
cd2.hole_topfromsurf = pd.concat([compaction_df.loc[22,'hole_botfromsurf'],compaction_df.loc[25,'hole_botfromsurf']],axis=1).min(axis=1)
cd2.compaction_borehole_length_m = compaction_df.loc[22,'compaction_borehole_length_m']-compaction_df.loc[25,'compaction_borehole_length_m']
cd2['Compaction_Instrument_ID'] = int(111)
cd2.set_index(['Compaction_Instrument_ID',cd2.index],inplace=True)
compaction_df = pd.concat([compaction_df,cd2])
idx=compaction_df.index
compaction_df.index=compaction_df.index.set_levels([idx.levels[0].astype(int),idx.levels[1]])

# 22-23
cd2 = compaction_df.loc[22].copy()
cd2.hole_init_length = compaction_df.loc[22,'hole_init_length']-compaction_df.loc[23,'hole_init_length']
cd2.hole_botfromsurf = pd.concat([compaction_df.loc[22,'hole_botfromsurf'],compaction_df.loc[23,'hole_botfromsurf']],axis=1).max(axis=1)
cd2.hole_topfromsurf = pd.concat([compaction_df.loc[22,'hole_botfromsurf'],compaction_df.loc[23,'hole_botfromsurf']],axis=1).min(axis=1)
cd2.compaction_borehole_length_m = compaction_df.loc[22,'compaction_borehole_length_m']-compaction_df.loc[23,'compaction_borehole_length_m']
cd2['Compaction_Instrument_ID'] = int(112)
cd2.set_index(['Compaction_Instrument_ID',cd2.index],inplace=True)
compaction_df = pd.concat([compaction_df,cd2])
idx=compaction_df.index
compaction_df.index=compaction_df.index.set_levels([idx.levels[0].astype(int),idx.levels[1]])

# 24-25
cd2 = compaction_df.loc[22].copy()
cd2.hole_init_length = compaction_df.loc[24,'hole_init_length']-compaction_df.loc[25,'hole_init_length']
cd2.hole_botfromsurf = pd.concat([compaction_df.loc[24,'hole_botfromsurf'],compaction_df.loc[25,'hole_botfromsurf']],axis=1).max(axis=1)
cd2.hole_topfromsurf = pd.concat([compaction_df.loc[24,'hole_botfromsurf'],compaction_df.loc[25,'hole_botfromsurf']],axis=1).min(axis=1)
cd2.compaction_borehole_length_m = compaction_df.loc[24,'compaction_borehole_length_m']-compaction_df.loc[25,'compaction_borehole_length_m']
cd2['Compaction_Instrument_ID'] = int(113)
cd2.set_index(['Compaction_Instrument_ID',cd2.index],inplace=True)
compaction_df = pd.concat([compaction_df,cd2])
idx=compaction_df.index
compaction_df.index=compaction_df.index.set_levels([idx.levels[0].astype(int),idx.levels[1]])

# 24-23
cd2 = compaction_df.loc[24].copy()
cd2.hole_init_length = compaction_df.loc[24,'hole_init_length']-compaction_df.loc[23,'hole_init_length']
cd2.hole_botfromsurf = pd.concat([compaction_df.loc[24,'hole_botfromsurf'],compaction_df.loc[23,'hole_botfromsurf']],axis=1).max(axis=1)
cd2.hole_topfromsurf = pd.concat([compaction_df.loc[24,'hole_botfromsurf'],compaction_df.loc[23,'hole_botfromsurf']],axis=1).min(axis=1)
cd2.compaction_borehole_length_m = compaction_df.loc[24,'compaction_borehole_length_m']-compaction_df.loc[23,'compaction_borehole_length_m']
cd2['Compaction_Instrument_ID'] = int(114)
cd2.set_index(['Compaction_Instrument_ID',cd2.index],inplace=True)
compaction_df = pd.concat([compaction_df,cd2])
idx=compaction_df.index
compaction_df.index=compaction_df.index.set_levels([idx.levels[0].astype(int),idx.levels[1]])

#25-23
cd2 = compaction_df.loc[24].copy()
cd2.hole_init_length = compaction_df.loc[25,'hole_init_length']-compaction_df.loc[23,'hole_init_length']
cd2.hole_botfromsurf = pd.concat([compaction_df.loc[25,'hole_botfromsurf'],compaction_df.loc[23,'hole_botfromsurf']],axis=1).max(axis=1)
cd2.hole_topfromsurf = pd.concat([compaction_df.loc[25,'hole_botfromsurf'],compaction_df.loc[23,'hole_botfromsurf']],axis=1).min(axis=1)
cd2.compaction_borehole_length_m = compaction_df.loc[25,'compaction_borehole_length_m']-compaction_df.loc[23,'compaction_borehole_length_m']
cd2['Compaction_Instrument_ID'] = int(115)
cd2.set_index(['Compaction_Instrument_ID',cd2.index],inplace=True)
compaction_df = pd.concat([compaction_df,cd2])
idx=compaction_df.index
compaction_df.index=compaction_df.index.set_levels([idx.levels[0].astype(int),idx.levels[1]])


# Now do figure out the compaction rates and strain on all the holes.
compaction_df['compdiff']=compaction_df.groupby(level=0)['compaction_borehole_length_m'].diff()
compaction_df['totcomp']=compaction_df.groupby(level=0)['compaction_borehole_length_m'].apply(lambda x: x[0]-x)
msk = compaction_df['compdiff']>0
compaction_df.loc[msk,'compdiff']=0
compaction_df['compdiff'].fillna(value=0,inplace=True)
compaction_df['strain']=compaction_df.groupby(level=0)['compaction_borehole_length_m'].apply(strainfun)
compaction_df['ss1970']=(compaction_df.index.get_level_values(1)-epoch).total_seconds()
compaction_df['filtstrain']=0*compaction_df['strain']
compaction_df['strainrate']=0*compaction_df['strain']
compaction_df['filtstrainrate']=0*compaction_df['strain']
compaction_df['comprate']=0*compaction_df['strain']
compaction_df['filttotcomp']=0*compaction_df['strain']
compaction_df['filtcomprate']=0*compaction_df['strain']

uids = compaction_df.index.get_level_values(0).unique()
for ii in uids:
    grad = np.gradient(compaction_df['strain'][ii],compaction_df['ss1970'][ii])
    compaction_df.loc[ii,'strainrate'] = grad # units: per second
    comprate = np.gradient(compaction_df['totcomp'][ii],compaction_df['ss1970'][ii])
    compaction_df.loc[ii,'comprate'] = comprate # units: per second
    try:
        filtstr = smooth(compaction_df['strain'][ii])
        filtgrad = np.gradient(filtstr,compaction_df['ss1970'][ii])
        compaction_df.loc[ii,'filtstrain'] = filtstr # units: per second
        compaction_df.loc[ii,'filtstrainrate'] = filtgrad
        
        filtcomp = smooth(compaction_df['totcomp'][ii])
        filtcomprate = np.gradient(filtcomp,compaction_df['ss1970'][ii])
        compaction_df.loc[ii,'filttotcomp'] = filtcomp # units: per second
        compaction_df.loc[ii,'filtcomprate'] = filtcomprate
    except:
        pass

fcr,acr=plt.subplots()
(compaction_df.loc[31,'filtcomprate']*spy).plot(ax=acr)
acr.grid(True)
acr.set_xlabel('Date')
acr.set_ylabel('Compaction Rate (m a$^{-1}$)')

f2,a2=plt.subplots(figsize=(10,8))
(compaction_df.loc[11].compaction_borehole_length_m).plot(ax=a2)

f1,a1 = plt.subplots(figsize=(8,4))
a1.plot(compaction_df.loc[30,'filtstrainrate'])
a1.plot(compaction_df.loc[31,'filtstrainrate'])
a1.plot(compaction_df.loc[32,'filtstrainrate'])
a1.plot(compaction_df.loc[33,'filtstrainrate'])
a1.plot(compaction_df.loc[41,'filtstrainrate'])
a1.grid(True)
a1.set_xlabel('Date')
a1.set_ylabel('Strain rate')

f1,a1 = plt.subplots(figsize=(8,4))
a1.plot(compaction_df.loc[26,'filtstrain'])
a1.plot(compaction_df.loc[31,'filtstrain'])
a1.plot(compaction_df.loc[32,'filtstrain'])
a1.plot(compaction_df.loc[33,'filtstrain'])
a1.plot(compaction_df.loc[41,'filtstrain'])
a1.grid(True)
a1.set_xlabel('Date')
a1.set_ylabel('Total Compaction')

f1,a1 = plt.subplots(figsize=(8,4))
a1.plot(compaction_df.loc[26,'totcomp'])
a1.plot(compaction_df.loc[27,'totcomp'])
a1.plot(compaction_df.loc[28,'totcomp'])
a1.plot(compaction_df.loc[29,'totcomp'])
# a1.plot(compaction_df.loc[41,'totcomp'])
a1.grid(True)
a1.set_xlabel('Date')
a1.set_ylabel('Total Compaction')

compaction_df.head()

df=compaction_df.loc[41].copy()
df['tvalue'] = df.index
df['delta'] = (df['tvalue']-df['tvalue'].shift()).fillna(0)

statmeta_df=pd.DataFrame.from_records(datatable.Station_Metadata[:].tolist(),columns=datatable.Station_Metadata.colnames)
statmeta_df.sitename=statmeta_df.sitename.str.decode("utf-8")
statmeta_df.iridium_URL=statmeta_df.iridium_URL.str.decode("utf-8")
# pd.to_datetime(compaction_df.daynumber_YYYYMMDD.values,format='%Y%m%d')
statmeta_df['install_date']=pd.to_datetime(statmeta_df.installation_daynumer_YYYYMMDD.values,format='%Y%m%d')
statmeta_df['rtd_date']=pd.to_datetime(statmeta_df.RTD_installation_daynumber_YYYYMMDD.values,format='%Y%m%d')
zz=[]
for ii in range(len(statmeta_df.RTD_depths_at_installation_m[0])):
    st = 'rtd%s' %ii
    zz.append(st)
    
statmeta_df[zz]=pd.DataFrame(statmeta_df.RTD_depths_at_installation_m.values.tolist(),index=statmeta_df.index)  
statmeta_df.set_index('sitename',inplace=True)
statmeta_df.loc['Crawford','rtd_date']=statmeta_df.loc['Crawford','install_date']
statmeta_df.loc['NASA-SE','rtd_date']=statmeta_df.loc['NASA-SE','install_date']-pd.Timedelta(days=1)

airtemp_df=pd.DataFrame.from_records(datatable.Air_Temp_Hourly[:])
airtemp_df.sitename=airtemp_df.sitename.str.decode("utf-8")
# pd.to_datetime(compaction_df.daynumber_YYYYMMDD.values,format='%Y%m%d')
airtemp_df['date']=pd.to_datetime(airtemp_df.daynumber_YYYYMMDD.values*100+airtemp_df.hournumber_HH.values.astype('uint16'),format='%Y%m%d%H')
airtemp_df

metdata_df=pd.DataFrame.from_records(datatable.Meteorological_Daily[:])
metdata_df.sitename=metdata_df.sitename.str.decode("utf-8")
# pd.to_datetime(compaction_df.daynumber_YYYYMMDD.values,format='%Y%m%d')
metdata_df['date']=pd.to_datetime(metdata_df.daynumber_YYYYMMDD.values,format='%Y%m%d')
for site in sites:
    msk=(metdata_df['sitename']==site)&(metdata_df['date']<statmeta_df.loc[site,'rtd_date'])
    metdata_df.drop(metdata_df[msk].index,inplace=True)
    if site=='NASA-SE':
        # NASA-SE had a new tower section in 5/17; distance raised is ??, use 1.7 m for now. 
        m2 = (metdata_df['sitename']==site)&(metdata_df['date']>'2017-05-10')
        metdata_df.loc[m2,'sonic_range_dist_corrected_m']=metdata_df.loc[m2,'sonic_range_dist_corrected_m']-1.7
    elif site=='Crawford':
        # Crawford has bad sonic data for 11/3/17 to 2/16/18
        m2 = (metdata_df['sitename']==site)&(metdata_df['date']>'2017-11-03')&(metdata_df['date']<'2018-02-16')
        metdata_df.loc[m2,'sonic_range_dist_corrected_m']=np.nan
    elif site=='EKT':
        # EKT had a new tower section in 5/17; distance raised is 0.86 m. 
        m2 = (metdata_df['sitename']==site)&(metdata_df['date']>'2017-05-05')
        metdata_df.loc[m2,'sonic_range_dist_corrected_m']=metdata_df.loc[m2,'sonic_range_dist_corrected_m']-0.86
    elif site=='Saddle':
        # Saddle had a new tower section in 5/17; distance raised is 1.715 m. 
        m2 = (metdata_df['sitename']==site)&(metdata_df['date']>'2017-05-07')
        metdata_df.loc[m2,'sonic_range_dist_corrected_m']=metdata_df.loc[m2,'sonic_range_dist_corrected_m']-1.715
    elif site=='EastGrip':
        # Eastgrip has bad sonic data for 11/7/17 onward 
        m2 = (metdata_df['sitename']==site)&(metdata_df['date']>'2017-11-17')
        metdata_df.loc[m2,'sonic_range_dist_corrected_m']=np.nan
        m3 = (metdata_df['sitename']==site)&(metdata_df['date']>'2015-10-01')&(metdata_df['date']<'2016-04-01')
        metdata_df.loc[m3,'sonic_range_dist_corrected_m']=np.nan
        m4 = (metdata_df['sitename']==site)&(metdata_df['date']>'2016-12-07')&(metdata_df['date']<'2017-03-01')
        metdata_df.loc[m4,'sonic_range_dist_corrected_m']=np.nan
    elif site=='DYE-2':
        # 
        m3 = (metdata_df['sitename']==site)&(metdata_df['date']>'2015-12-24')&(metdata_df['date']<'2016-05-01')
        metdata_df.loc[m3,'sonic_range_dist_corrected_m']=np.nan
#         m4 = (metdata_df['sitename']==site)&(metdata_df['date']>'2016-12-07')&(metdata_df['date']<'2017-03-01')
#         metdata_df.loc[m4,'sonic_range_dist_corrected_m']=np.nan
        
metdata_df.reset_index(drop=True)

sonic_df = metdata_df[['sitename','date','sonic_range_dist_corrected_m']].set_index(['sitename','date'])
sonic_df.columns = ['sonic_m']
sonic_df.sonic_m[sonic_df.sonic_m<-100]=np.nan
sonic_df.loc['Saddle','2015-05-16']=sonic_df.loc['Saddle','2015-05-17']

for site in sites:
    if site=='Summit':
        sonic_df.loc['Summit','sonic_m']=sonic_df.loc['Summit'].interpolate()
        sonic_df.loc['Summit','sonic_m']=smooth(sonic_df.loc['Summit','sonic_m'].values)
    elif site=='KAN-U':
        gradthresh = 0.1
        vals = sonic_df.loc['KAN-U','sonic_m'].values
        vals[np.isnan(vals)]=-9999
        msk = np.where(np.abs(np.gradient(vals))>=gradthresh)[0]
        vals[msk] = np.nan
        vals[msk-1] = np.nan
        vals[msk+1] = np.nan
        vals[vals==-9999]=np.nan
        sonic_df.loc['KAN-U','sonic_m']=vals
        sonic_df.loc['KAN-U','sonic_m']=sonic_df.loc['KAN-U'].interpolate(method='linear')
        sonic_df.loc['KAN-U','sonic_m']=smooth(sonic_df.loc['KAN-U','sonic_m'].values)
    elif site=='NASA-SE':
        sonic_df.loc['NASA-SE','sonic_m']=sonic_df.loc['NASA-SE'].interpolate()
        sonic_df.loc['NASA-SE','sonic_m']=smooth(sonic_df.loc['NASA-SE','sonic_m'].values)
    elif site=='Crawford':
        gradthresh = 0.1
        vals = sonic_df.loc['Crawford','sonic_m'].values
        vals[np.isnan(vals)]=-9999
        msk = np.where(np.abs(np.gradient(vals))>=gradthresh)[0]
        vals[msk] = np.nan
        vals[msk-1] = np.nan
        vals[msk+1] = np.nan
        vals[vals==-9999]=np.nan
        sonic_df.loc['Crawford','sonic_m']=vals
        sonic_df.loc['Crawford','sonic_m']=sonic_df.loc['Crawford'].interpolate(method='linear')
        sonic_df.loc['Crawford','sonic_m']=smooth(sonic_df.loc['Crawford','sonic_m'].values)
    elif site=='EKT':
        gradthresh = 0.1
        vals = sonic_df.loc['EKT','sonic_m'].values
        vals[np.isnan(vals)]=-9999
        msk = np.where(np.abs(np.gradient(vals))>=gradthresh)[0]
        vals[msk] = np.nan
        vals[msk-1] = np.nan
        vals[msk+1] = np.nan
        vals[vals==-9999]=np.nan
        sonic_df.loc['EKT','sonic_m']=vals
        sonic_df.loc['EKT','sonic_m']=sonic_df.loc['EKT'].interpolate(method='linear')
        sonic_df.loc['EKT','sonic_m']=smooth(sonic_df.loc['EKT','sonic_m'].values)
    elif site=='Saddle':
        gradthresh = 0.1
        vals = sonic_df.loc['Saddle','sonic_m'].values
        vals[np.isnan(vals)]=-9999
        msk = np.where(np.abs(np.gradient(vals))>=gradthresh)[0]
        vals[msk] = np.nan
        vals[msk-1] = np.nan
        vals[msk+1] = np.nan
        vals[vals==-9999]=np.nan
        sonic_df.loc['Saddle','sonic_m']=vals
        sonic_df.loc['Saddle','sonic_m']=sonic_df.loc['Saddle'].interpolate(method='linear')
        sonic_df.loc['Saddle','sonic_m']=smooth(sonic_df.loc['Saddle','sonic_m'].values)
    elif site=='EastGrip':
        gradthresh = 0.1
        vals = sonic_df.loc['EastGrip','sonic_m'].values
        vals[np.isnan(vals)]=-9999
        msk = np.where(np.abs(np.gradient(vals))>=gradthresh)[0]
        vals[msk] = np.nan
        vals[msk-1] = np.nan
        vals[msk+1] = np.nan
        vals[vals==-9999]=np.nan
        sonic_df.loc['EastGrip','sonic_m']=vals
        sonic_df.loc['EastGrip','sonic_m']=sonic_df.loc['EastGrip'].interpolate(method='linear')
        sonic_df.loc['EastGrip','sonic_m']=smooth(sonic_df.loc['EastGrip','sonic_m'].values)
    elif site=='DYE-2':
        gradthresh = 0.1
        vals = sonic_df.loc['DYE-2','sonic_m'].values
        vals[np.isnan(vals)]=-9999
        msk = np.where(np.abs(np.gradient(vals))>=gradthresh)[0]
        vals[msk] = np.nan
        vals[msk-1] = np.nan
        vals[msk+1] = np.nan
        vals[vals==-9999]=np.nan
        sonic_df.loc['DYE-2','sonic_m']=vals
        sonic_df.loc['DYE-2','sonic_m']=sonic_df.loc['DYE-2'].interpolate(method='linear')
        sonic_df.loc['DYE-2','sonic_m']=smooth(sonic_df.loc['DYE-2','sonic_m'].values)
        
for ss in sonic_df.index.unique(level='sitename'):
    dd = statmeta_df.loc[ss]['rtd_date']
    if ss=='Saddle':
        dd = dd + pd.Timedelta('1D')
    sonic_df.loc[ss,'delta']=sonic_df.loc[[ss]].sonic_m-sonic_df.loc[(ss,dd)].sonic_m

for site in sites:
    sonic_df.loc[site].plot()
    plt.title(site)
    plt.grid(True)

rtd_depth_df=statmeta_df[zz].copy()
xx=statmeta_df.RTD_top_usable_RTD_num
for site in sites:
    vv=rtd_depth_df.loc[site].values
    ri = np.arange(xx.loc[site],24)
    vv[ri]=np.nan
    rtd_depth_df.loc[site]=vv
rtd_d = sonic_df.join(rtd_depth_df, how='inner')
rtd_dc = rtd_d.copy()
rtd_dep = rtd_dc[zz].add(rtd_dc['delta'],axis='rows')

rtd_df=pd.DataFrame.from_records(datatable.Firn_Temp_Daily[:].tolist(),columns=datatable.Firn_Temp_Daily.colnames)
rtd_df.sitename=rtd_df.sitename.str.decode("utf-8")
rtd_df['date']=pd.to_datetime(rtd_df.daynumber_YYYYMMDD.values,format='%Y%m%d')
rtd_df.set_index(['sitename','date'])

rtd_df.head(3)

rtd_trun = rtd_df[['sitename','date','RTD_temp_avg_corrected_C']].copy().set_index(['sitename','date'])
rtd_trun.columns = ['T_avg']
rtd_trun[zz]=pd.DataFrame(rtd_trun.T_avg.values.tolist(),index=rtd_trun.index)
rtd_trun.drop('T_avg',axis=1,inplace=True)
rtd_trun.replace(-100.0,np.nan,inplace=True)

site='Saddle'
sitetemp=rtd_trun.loc[site]
sitedep = rtd_dep.loc[site]
n_grid = np.linspace(-15,3,61)
time=sitetemp.index.values
temps = sitetemp.values
depths = sitedep.values
rtd_trun.loc[site]

### Use the separate script to plot these.

# for site in sites:
#     print(site)
#     sitetemp=rtd_trun.loc[site]
#     sitedep = rtd_dep.loc[site]
#     n_grid = np.linspace(-15,3,61)
#     time=sitetemp.index.values
#     temps = sitetemp.values
#     depths = sitedep.values
#     ro,co=np.shape(temps)
#     t_interp=np.zeros((ro,len(n_grid)))
#     for kk in range(ro):
#             tif = sp.interpolate.interp1d(depths[kk,:],temps[kk,:],bounds_error=False)
#             t_interp[kk,:]= tif(n_grid)
            
#     f1, ax1 = plt.subplots()
#     f1.set_size_inches(12,6)
#     cax1 = ax1.contourf(time,n_grid,t_interp.T,256, extend='both', vmin=max(-50,np.nanmin(t_interp)),vmax=min(0,np.nanmax(t_interp)))
#     cax1.cmap.set_over('cyan')
#     cax1.cmap.set_under('black')
#     ax1.axhline(y=0,color='k')
#     ax1.set_ylabel('depth')
#     ax1.set_xlabel('date')
#     cb1 = f1.colorbar(cax1)
#     cb1.set_label('Temperature (C)')
#     ax1.set_title(site)
# #     f1.savefig('figures/RTD_temp_%s.eps' %site)
#     plt.close(f1)

# rtd_trun.to_pickle('firncover_rtd_temp.pkl')
# rtd_dep.to_pickle('firncover_rtd_depth.pkl')
compaction_df.to_pickle('firncover_compaction.pkl')

# Now we want to find the viscosity, so we need a mean temperature and density for each measurement.

t_interp[4:7,:]

site = 'Summit'
sitetemp=rtd_trun.loc[site]
sitedep = rtd_dep.loc[site]
n_grid = np.linspace(-22,0,61)
time=sitetemp.index.values
temps = sitetemp.values
depths = sitedep.values
ro,co=np.shape(temps)
t_interp=np.zeros((ro,len(n_grid)))
for kk in range(ro):
        tif = sp.interpolate.interp1d(depths[kk,:],temps[kk,:],kind='linear',bounds_error=False,fill_value=np.nan)
        try:
            t_interp[kk,:]= tif(n_grid)

            ind = np.where(~np.isnan(t_interp[kk,:]))[0]
            first, last = ind[0], ind[-1]
            t_interp[kk,:first] = t_interp[kk,first]
            t_interp[kk,last + 1:] = t_interp[kk,last]
        except:
            continue
ti_df=pd.DataFrame(data=t_interp,index=rtd_trun.loc['Summit'].index)

site = 'EastGrip'
sitetemp=rtd_trun.loc[site]
sitedep = rtd_dep.loc[site]
n_grid = np.linspace(-22,0,61)
time=sitetemp.index.values
temps = sitetemp.values
depths = sitedep.values
ro,co=np.shape(temps)
t_interp=np.zeros((ro,len(n_grid)))
for kk in range(ro):
        tif = sp.interpolate.interp1d(depths[kk,:],temps[kk,:],kind='linear',bounds_error=False,fill_value=np.nan)
        try:
            t_interp[kk,:]= tif(n_grid)

            ind = np.where(~np.isnan(t_interp[kk,:]))[0]
            first, last = ind[0], ind[-1]
            t_interp[kk,:first] = t_interp[kk,first]
            t_interp[kk,last + 1:] = t_interp[kk,last]
        except:
            continue
ti_df_egr=pd.DataFrame(data=t_interp,index=rtd_trun.loc['EastGrip'].index)

site = 'NASA-SE'
sitetemp=rtd_trun.loc[site]
sitedep = rtd_dep.loc[site]
n_grid = np.linspace(-22,0,61)
time=sitetemp.index.values
temps = sitetemp.values
depths = sitedep.values
ro,co=np.shape(temps)
t_interp=np.zeros((ro,len(n_grid)))
for kk in range(ro):
        tif = sp.interpolate.interp1d(depths[kk,:],temps[kk,:],kind='linear',bounds_error=False,fill_value=np.nan)
        t_interp[kk,:]= tif(n_grid)

        ind = np.where(~np.isnan(t_interp[kk,:]))[0]
        first, last = ind[0], ind[-1]
        t_interp[kk,:first] = t_interp[kk,first]
        t_interp[kk,last + 1:] = t_interp[kk,last]

ti_df_nasa=pd.DataFrame(data=t_interp,index=rtd_trun.loc['NASA-SE'].index)

site = 'Saddle'
sitetemp=rtd_trun.loc[site]
sitedep = rtd_dep.loc[site]
n_grid = np.linspace(-22,0,61)
time=sitetemp.index.values
temps = sitetemp.values
depths = sitedep.values
ro,co=np.shape(temps)
t_interp=np.zeros((ro,len(n_grid)))
for kk in range(ro):
        tif = sp.interpolate.interp1d(depths[kk,:],temps[kk,:],kind='linear',bounds_error=False,fill_value=np.nan)
        t_interp[kk,:]= tif(n_grid)
        try:
            ind = np.where(~np.isnan(t_interp[kk,:]))[0]
            first, last = ind[0], ind[-1]
            t_interp[kk,:first] = t_interp[kk,first]
            t_interp[kk,last + 1:] = t_interp[kk,last]
        except:
            t_interp[kk,:]=np.nan

ti_df_sad=pd.DataFrame(data=t_interp,index=rtd_trun.loc['Saddle'].index)

deparr = np.arange(0,25,0.1)
sumagehl,sumdenhl = hl_analytic(300,np.arange(0,25,0.1),243,0.23*0.917)
egragehl,egrdenhl = hl_analytic(350,np.arange(0,25,0.1),244,0.14*0.917)
nasaagehl,nasadenhl = hl_analytic(350,np.arange(0,25,0.1),253,0.68*0.917)
sadagehl,saddenhl = hl_analytic(350,np.arange(0,25,0.1),253,0.47*0.917)
crawagehl,crawdenhl = hl_analytic(350,np.arange(0,25,0.1),255,0.5*0.917)
sumageif = sp.interpolate.interp1d(deparr,sumagehl)
egrageif = sp.interpolate.interp1d(deparr,egragehl)
nasaageif = sp.interpolate.interp1d(deparr,nasaagehl)
sadageif = sp.interpolate.interp1d(deparr,nasaagehl)
crawageif = sp.interpolate.interp1d(deparr,nasaagehl)

core_df['age']=np.zeros_like(core_df.depth)
core_df['lnrho']=np.zeros_like(core_df.depth)
core_df['slope']=np.zeros_like(core_df.depth)

for core in core_df.xs('Summit',level='site').index.get_level_values(level=0).unique():
    core_df.loc[core,'age']=sumageif(core_df.loc[core,'depth'])
    core_df.loc[core,'lnrho']= -1*np.log(917.0-core_df.loc[core,'density'].values)
    core_df.loc[core,'slope'] = np.polyfit(core_df.loc[core,'age'].values,core_df.loc[core,'lnrho'].values,1)[0]

summitdf = compaction_df[compaction_df.sitename=='Summit']
sdf2=summitdf.copy()
sdf2['stress']=np.zeros_like(sdf2.hole_init_length)
sdf2['rho']=np.zeros_like(sdf2.hole_init_length)
sdf2['temperature']=np.zeros_like(sdf2.hole_init_length)
sdf2['age']=sumageif(sdf2['hole_botfromsurf'])

for ii in sdf2.index.unique(level='instrument_id'):
    htop = sdf2.loc[ii,'hole_topfromsurf'][0] * -1.0
    hbot = sdf2.loc[ii,'hole_botfromsurf'][0] * -1.0
    qq = ((n_grid<htop) & (n_grid>hbot))
    rr = pd.DataFrame(ti_df.iloc[:,qq].mean(axis=1),columns=['temperature'])
    inter = rr.index.intersection(sdf2.loc[ii].index)
    sdf2.loc[ii,'temperature'] = rr.loc[inter].values.T[0]
    if ii<100:
        sdf2.loc[ii,'stress']=core_df.xs(ii,level='instrument').stress_cu[-1]
        sdf2.loc[ii,'rho']=core_df.xs(ii,level='instrument').density.mean()
    elif ii==101:
        mxx = core_df.xs(31,level='instrument').depth.max()
        sdf2.loc[ii,'stress']=core_df.xs(30,level='instrument').stress_cu[-1]
        sdf2.loc[ii,'rho']=core_df.xs(30,level='instrument')[core_df.xs(30,level='instrument').depth>mxx].density.mean()
    elif ii==102:
        mxx = core_df.xs(32,level='instrument').depth.max()
        sdf2.loc[ii,'stress']=core_df.xs(30,level='instrument').stress_cu[-1]
        sdf2.loc[ii,'rho']=core_df.xs(30,level='instrument')[core_df.xs(30,level='instrument').depth>mxx].density.mean()
    elif ii==103:
        mxx = core_df.xs(31,level='instrument').depth.max()
        sdf2.loc[ii,'stress']=core_df.xs(32,level='instrument').stress_cu[-1]
        sdf2.loc[ii,'rho']=core_df.xs(32,level='instrument')[core_df.xs(32,level='instrument').depth>mxx].density.mean()        

egripdf = compaction_df[compaction_df.sitename=='EastGrip']
egdf2=egripdf.copy()
egdf2['stress']=np.zeros_like(egdf2.hole_init_length)
egdf2['rho']=np.zeros_like(egdf2.hole_init_length)
egdf2['temperature']=np.zeros_like(egdf2.hole_init_length)
egdf2['age']=egrageif(egdf2['hole_botfromsurf'])

for ii in egdf2.index.unique(level='instrument_id'):
    htop = egdf2.loc[ii,'hole_topfromsurf'][0] * -1.0
    hbot = egdf2.loc[ii,'hole_botfromsurf'][0] * -1.0
    qq = ((n_grid<htop) & (n_grid>hbot))
    rr = pd.DataFrame(ti_df_egr.iloc[:,qq].mean(axis=1),columns=['temperature'])
    inter = rr.index.intersection(egdf2.loc[ii].index)
    egdf2.loc[ii,'temperature'] = rr.loc[inter].values.T[0]
    if ii<100:
        egdf2.loc[ii,'stress']=core_df.xs(ii,level='instrument').stress_cu[-2]
        egdf2.loc[ii,'rho']=core_df.xs(ii,level='instrument').density.mean()
    elif ii==104:
        mxx = core_df.xs(28,level='instrument').depth.max()
        egdf2.loc[ii,'stress']=core_df.xs(26,level='instrument').stress_cu[-2]
        egdf2.loc[ii,'rho']=core_df.xs(26,level='instrument')[core_df.xs(26,level='instrument').depth>mxx].density.mean()
    elif ii==105:
        mxx = core_df.xs(27,level='instrument').depth.max()
        egdf2.loc[ii,'stress']=core_df.xs(26,level='instrument').stress_cu[-2]
        egdf2.loc[ii,'rho']=core_df.xs(26,level='instrument')[core_df.xs(26,level='instrument').depth>mxx].density.mean()
    elif ii==106:
        mxx = core_df.xs(27,level='instrument').depth.max()
        egdf2.loc[ii,'stress']=core_df.xs(28,level='instrument').stress_cu[-2]
        egdf2.loc[ii,'rho']=core_df.xs(28,level='instrument')[core_df.xs(28,level='instrument').depth>mxx].density.mean() 
                        
nasadf = compaction_df[compaction_df.sitename=='NASA-SE']
nadf2=nasadf.copy()
nadf2['stress']=np.zeros_like(nadf2.hole_init_length)
nadf2['rho']=np.zeros_like(nadf2.hole_init_length)
nadf2['temperature']=np.zeros_like(nadf2.hole_init_length)
nadf2['age']=nasaageif(nadf2['hole_botfromsurf'])

for ii in nadf2.index.unique(level='instrument_id'):
    htop = nadf2.loc[ii,'hole_topfromsurf'][0] * -1.0
    hbot = nadf2.loc[ii,'hole_botfromsurf'][0] * -1.0
    qq = ((n_grid<htop) & (n_grid>hbot))
    rr = pd.DataFrame(ti_df_nasa.iloc[:,qq].mean(axis=1),columns=['temperature'])
    inter = rr.index.intersection(nadf2.loc[ii].index)
    nadf2.loc[ii,'temperature'] = rr.loc[inter].values.T[0]
    if ii<100:
        nadf2.loc[ii,'stress']=core_df.xs(ii,level='instrument').stress_cu[-2]
        nadf2.loc[ii,'rho']=core_df.xs(ii,level='instrument').density.mean()
    elif ii==107:
        mxx = core_df.xs(14,level='instrument').depth.max()
        nadf2.loc[ii,'stress']=core_df.xs(13,level='instrument').stress_cu[-2]
        nadf2.loc[ii,'rho']=core_df.xs(13,level='instrument')[core_df.xs(13,level='instrument').depth>mxx].density.mean()
    elif ii==108:
        mxx = core_df.xs(15,level='instrument').depth.max()
        nadf2.loc[ii,'stress']=core_df.xs(13,level='instrument').stress_cu[-2]
        nadf2.loc[ii,'rho']=core_df.xs(13,level='instrument')[core_df.xs(13,level='instrument').depth>mxx].density.mean()
    elif ii==109:
        mxx = core_df.xs(14,level='instrument').depth.max()
        nadf2.loc[ii,'stress']=core_df.xs(15,level='instrument').stress_cu[-2]
        nadf2.loc[ii,'rho']=core_df.xs(15,level='instrument')[core_df.xs(15,level='instrument').depth>mxx].density.mean()          
        
saddledf = compaction_df[compaction_df.sitename=='Saddle']
saddf2=saddledf.copy()
saddf2 = saddf2[((saddf2.index.get_level_values("date")>'2016-01-01')&(saddf2.index.get_level_values("date")<'2017-08-21'))]
saddf2['stress']=np.zeros_like(saddf2.hole_init_length)
saddf2['rho']=np.zeros_like(saddf2.hole_init_length)
saddf2['temperature']=np.zeros_like(saddf2.hole_init_length)
saddf2['age']=sadageif(saddf2['hole_botfromsurf'])

for ii in saddf2.index.unique(level='instrument_id'):
    htop = saddf2.loc[ii,'hole_topfromsurf'][0] * -1.0
    hbot = saddf2.loc[ii,'hole_botfromsurf'][0] * -1.0
    qq = ((n_grid<htop) & (n_grid>hbot))
    rr = pd.DataFrame(ti_df_sad.iloc[:,qq].mean(axis=1),columns=['temperature'])
    rr = rr.reindex(saddf2.loc[ii].index,method='nearest')
    inter = rr.index.intersection(saddf2.loc[ii].index)
    saddf2.loc[ii,'temperature'] = rr.loc[inter].values.T[0]
    if ii<100:
        saddf2.loc[ii,'stress']=core_df.xs(ii,level='instrument').stress_cu[-2]
        saddf2.loc[ii,'rho']=core_df.xs(ii,level='instrument').density.mean()

# crawforddf = compaction_df[compaction_df.sitename=='Crawford']
# crawdf2=crawforddf.copy()
# crawdf2 = crawdf2[((crawdf2.index.get_level_values("date")>'2016-01-01')&(crawdf2.index.get_level_values("date")<'2017-08-21'))]
# crawdf2['stress']=np.zeros_like(crawdf2.hole_init_length)
# crawdf2['rho']=np.zeros_like(crawdf2.hole_init_length)
# crawdf2['temperature']=np.zeros_like(crawdf2.hole_init_length)
# crawdf2['age']=crawageif(crawdf2['hole_botfromsurf'])

# for ii in crawdf2.index.unique(level='compaction_instrument_id'):
#     htop = crawdf2.loc[ii,'hole_topfromsurf'][0] * -1.0
#     hbot = crawdf2.loc[ii,'hole_botfromsurf'][0] * -1.0
#     qq = ((n_grid<htop) & (n_grid>hbot))
#     rr = pd.DataFrame(ti_df_craw.iloc[:,qq].mean(axis=1),columns=['temperature'])
#     rr = rr.reindex(crawdf2.loc[ii].index,method='nearest')
#     inter = rr.index.intersection(crawdf2.loc[ii].index)
#     crawdf2.loc[ii,'temperature'] = rr.loc[inter].values.T[0]
#     if ii<100:
#         crawdf2.loc[ii,'stress']=core_df.xs(ii,level='instrument').stress_cu[-2]
#         crawdf2.loc[ii,'rho']=core_df.xs(ii,level='instrument').density.mean()

dc=dict()
sints_s = [30,31,32,33,101,102,103]
# sints_s = [31]
sints_e = [26,27,28,104,105,106]
sints_n = [13,14,15,107,108,109]
sints_sad = [17,18,19,20]
sints_all = [30,31,32,33,41,101,102,103,26,27,28,40,104,105,106,13,14,15,37,107,108,109]
sints_all = [30,31,32,33,41,101,102,103,26,27,28,40,104,105,106]#,104,108]
rhoi = 917.0
qguess = 60000.0
R = 8.314
for bb in sints_n:
    if bb in sints_s:
        s4=sdf2.loc[bb][['compaction_borehole_length_m','hole_init_length','compdiff','totcomp','stress','rho','temperature','age']]
    elif bb in sints_e:
        s4=egdf2.loc[bb][['compaction_borehole_length_m','hole_init_length','compdiff','totcomp','stress','rho','temperature','age']]
    elif bb in sints_n:
        s4=nadf2.loc[bb][['compaction_borehole_length_m','hole_init_length','compdiff','totcomp','stress','rho','temperature','age']]
    elif bb in sints_sad:
        s4=saddf2.loc[bb][['compaction_borehole_length_m','hole_init_length','compdiff','totcomp','stress','rho','temperature','age']]
    s5=s4.resample('2W').agg({'compaction_borehole_length_m':np.min, 'hole_init_length':np.mean, 'compdiff': np.sum, 'totcomp': np.max, 'stress': np.mean, 'rho': np.mean, 'temperature':np.mean, 'age':np.mean})
    
    # Now do figure out the compaction rates and strain on all the holes.
    # s5['strain']=s5['compaction_borehole_length_m'].apply(strainfun)
    s5['strain'] = -1*np.log(s5['compaction_borehole_length_m']/s5['compaction_borehole_length_m'][0])
    s5['ss1970']=(s5.index-epoch).total_seconds()
    s5['filtstrain']=0*s5['strain']
    s5['strainrate']=0*s5['strain']
    s5['filtstrainrate']=0*s5['strain']
    s5['comprate']=0*s5['strain']
    s5['filttotcomp']=0*s5['strain']
    s5['filtcomprate']=0*s5['strain']

    # uids = s5.index.get_level_values(0).unique()
    # for ii in uids:
    grad = np.gradient(s5['strain'],s5['ss1970'])
    s5['strainrate'] = grad # units: per second
    comprate = np.gradient(s5['totcomp'],s5['ss1970'])
    s5['comprate'] = comprate # units: per second
#     try:
    filtstr = smooth(s5['strain'])
    filtgrad = np.gradient(filtstr,s5['ss1970'])
    s5['filtstrain'] = filtstr # units: per second
    s5['filtstrainrate'] = filtgrad    
    filtcomp = smooth(s5['totcomp'])
    filtcomprate = np.gradient(filtcomp,s5['ss1970'])
    s5['filttotcomp'] = filtcomp # units: per second
    s5['filtcomprate'] = filtcomprate
    s5['srsratio'] = s5.rho * s5.filtstrainrate / s5.stress
    s5['kguess'] = s5.srsratio / ((rhoi-s5.rho) * np.exp(-qguess/(R*(s5.temperature+273.15))))
    s5 = s5[pd.notnull(s5['filtstrainrate'])]
#     except:
#         pass
    dstr = 'df'+str(bb)
    dc[dstr]=s5


# kg=np.linspace(1.0e-6,3.0e-1,400) # for if we remove the age.
# kg=np.linspace(1e13,1e15,600)
kg=np.linspace(5e9,1e10,600)
q=np.linspace(56000,65000,400)
qt=np.tile(q,(len(kg),1)).T
kt=np.tile(kg,(len(q),1))
Qpref=60000
rmsdict = dict()
rmsexpdict = dict()
krmsd = dict()

blen = sum(len(dc[dct].srsratio) for dct in dc.keys())
minsmat = np.empty((blen,2))
ksmat = np.empty((blen,1))
ll = 0
for jj,key in enumerate(dc.keys()):
#     minsmat = np.empty((len(dc[key].srsratio),2))
#     for ii, val in enumerate(dc[key].srsratio): # option a
    for ii, val in enumerate(dc[key].filtstrainrate): # option b: rms error in m/s
        T = dc[key].temperature[ii]+273.15
#         kqmat = (rhoi-dc[key].rho[ii]) * kt * np.exp(-1*qt/(8.314*T)) #option a
        kqmat = (rhoi-dc[key].rho[ii]) * kt * np.exp(-1*qt/(8.314*T)) * dc[key].stress[ii] / dc[key].rho[ii] / dc[key].age[ii] #option b
        kvec_s = (rhoi-dc[key].rho[ii]) * kg * np.exp(-1*Qpref/(8.314*T)) * dc[key].stress[ii] / dc[key].rho[ii] / dc[key].age[ii] #option b
#         kvec_s = (rhoi-dc[key].rho[ii]) * kg * np.exp(-1*Qpref/(8.314*T)) * dc[key].stress[ii]**2 / dc[key].rho[ii] / dc[key].age[ii]**2 #option b
        rmsdict[ll] = (kqmat - val)**2
        krmsd[ll] = (kvec_s - val)**2
        rmsexpdict[ll] = np.exp(-1*q/(8.314*T))
        if np.isnan(rmsdict[ll]).any():
            print(key)
            print(val)
            print(ll)
        ind = np.unravel_index(np.argmin(rmsdict[ll], axis=None), rmsdict[ll].shape)
        kmin = kt[ind]
        qmin = qt[ind]
        minsmat[ll,0]=kmin
        minsmat[ll,1]=qmin
        
        ind2 = np.argmin(krmsd[ll])
        kmin = kg[ind2]
        ksmat[ll] = kmin
        
        ll+=1
        
v = rmsdict.values()
ke = rmsdict.keys()
rmse = np.sqrt(sum(v)/len(ke))

f4,a4=plt.subplots()
n, bins, patches=a4.hist(ksmat,20)

f1,a1 = plt.subplots()
a1.plot(minsmat[:,0],minsmat[:,1],'.')


cra = np.arange(0,len(minsmat[:,1]))
f3,a3 = plt.subplots()
a3.scatter(minsmat[:,0],minsmat[:,1],c=cra)
# a3.set_xlim(0,1e-3)

yy=rmse.min()
ex = np.floor(np.log10(np.abs(yy)))
cmax = ((np.ceil(yy/1.0*10**(ex+1)))*1.0*10**(ex+1))/1.1
cmin = ((np.ceil(yy/1.0*10**(ex)))*1.0*10**(ex))*7.5
rmi = rmse.min()
f2,a2 = plt.subplots()
lels = np.linspace(cmin,cmax,41)
cax = a2.contourf(kt,qt,rmse,levels = lels,extend = 'max')
cax.cmap.set_over('white')
a2.set_xlabel('k')
a2.set_ylabel('Q')
plt.colorbar(cax)

lels=np.linspace(1e-27,1e-20,256)
f10,a10=plt.subplots()
cax = a10.contourf(kg,q,rmsdict[10],levels=lels,extend='max')
plt.colorbar(cax)
a10.set_xlabel('k')
a10.set_ylabel('Q')

ks = np.array([9.9e8,7.2e9,8.5e9])
acs = np.array([0.68,0.23,0.14])
f8,a8 = plt.subplots()
a8.scatter(acs,ks)

np.polyfit(acs,ks,1)

f7,a7=plt.subplots()
a7.plot(sonic_df.loc['NASA-SE'].delta.values)

summitdf.loc[30]['2016']

compaction_df[compaction_df.sitename=='Summit'].head()

f1,a1=plt.subplots()
compaction_df.loc[30].comprate.plot(ax=a1)

compaction_df.loc[30,'2016-01-01'].totcomp-compaction_df.loc[30,'2016-12-31'].totcomp

compaction_df.loc[30].resample('a').mean()*spy

    
