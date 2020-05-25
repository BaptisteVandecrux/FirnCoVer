# -*- coding: utf-8 -*-
"""
Created on Sun May 10 11:05:10 2020

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
from FirnCover_lib import R, S_PER_YEAR, spy, RHO_1, RHO_2, RHO_I, RHO_I_MGM, RHO_1_MGM, GRAVITY, K_TO_C, BDOT_TO_A, RHO_W_KGM, P_0, epoch
import FirnCover_lib as fcl

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

#%% Loading data
# Things to change
figex = '_200504.png' #extension for figures (name plus type)
pathtodata = './data'
date1 = '2020_04_27' # Date on the hdf5 file

# %matplotlib inline
# %matplotlib qt

sites=['Summit','KAN-U','NASA-SE','Crawford','EKT','Saddle','EastGrip','DYE-2']

### Import the core depth/density data.
with open(os.path.join(pathtodata,'core_data_df.pkl'),'rb') as f:
    core_df=pd.read_pickle(f)

### Import the FirnCover data tables. 
filename='FirnCoverData_2.0_' + date1 + '.h5'
filepath=os.path.join(pathtodata,filename)
compaction_df, airtemp_df, inst_meta_df = fcl.import_firncover_dataset(filepath)

#%%   Now do figure out the compaction rates and strain on all the holes.
compaction_df['compdiff']=compaction_df.groupby(level=0)['compaction_borehole_length_m'].diff()
compaction_df['totcomp']=compaction_df.groupby(level=0)['compaction_borehole_length_m'].apply(lambda x: x[0]-x)

msk = compaction_df['compdiff']>0
compaction_df.loc[msk,'compdiff']=0
compaction_df['compdiff'].fillna(value=0,inplace=True)

compaction_df['strain']=compaction_df.groupby(level=0)['compaction_borehole_length_m'].apply(fcl.strainfun)
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
        filtstr = fcl.fcl.smooth(compaction_df['strain'][ii])
        filtgrad = np.gradient(filtstr,compaction_df['ss1970'][ii])
        compaction_df.loc[ii,'filtstrain'] = filtstr # units: per second
        compaction_df.loc[ii,'filtstrainrate'] = filtgrad
        
        filtcomp = fcl.fcl.smooth(compaction_df['totcomp'][ii])
        filtcomprate = np.gradient(filtcomp,compaction_df['ss1970'][ii])
        compaction_df.loc[ii,'filttotcomp'] = filtcomp # units: per second
        compaction_df.loc[ii,'filtcomprate'] = filtcomprate
    except:
        pass

#%% Loading side data
statmeta_df, airtemp_df, sonic_df, rtd_df, rtd_trun, rtd_dep = fcl.load_metadata(compaction_df,filepath,sites)

#%% saving to pkl
# rtd_trun.to_pickle('firncover_rtd_temp.pkl')
# rtd_dep.to_pickle('firncover_rtd_depth.pkl')
compaction_df.to_pickle('firncover_compaction.pkl')

#%% Now we want to find the viscosity, so we need a mean temperature and density for each measurement.

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
sumagehl,sumdenhl = fcl.hl_analytic(300,np.arange(0,25,0.1),243,0.23*0.917)
egragehl,egrdenhl = fcl.hl_analytic(350,np.arange(0,25,0.1),244,0.14*0.917)
nasaagehl,nasadenhl = fcl.hl_analytic(350,np.arange(0,25,0.1),253,0.68*0.917)
sadagehl,saddenhl = fcl.hl_analytic(350,np.arange(0,25,0.1),253,0.47*0.917)
crawagehl,crawdenhl = fcl.hl_analytic(350,np.arange(0,25,0.1),255,0.5*0.917)
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
    filtstr = fcl.smooth(s5['strain'])
    filtgrad = np.gradient(filtstr,s5['ss1970'])
    s5['filtstrain'] = filtstr # units: per second
    s5['filtstrainrate'] = filtgrad    
    filtcomp = fcl.smooth(s5['totcomp'])
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

#%% Plotting something
f4,a4=plt.subplots()
n, bins, patches=a4.hist(ksmat,20)

f1,a1 = plt.subplots()
a1.plot(minsmat[:,0],minsmat[:,1],'.')


#%% Plotting something
cra = np.arange(0,len(minsmat[:,1]))
f3,a3 = plt.subplots()
a3.scatter(minsmat[:,0],minsmat[:,1],c=cra)
# a3.set_xlim(0,1e-3)

#%% Plotting something
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

#%% Plotting something
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

#%% Plotting something
f7,a7=plt.subplots()
a7.plot(sonic_df.loc['NASA-SE'].delta.values)

summitdf.loc[30]['2016']

compaction_df[compaction_df.sitename=='Summit'].head()

#%% Plotting something
f1,a1=plt.subplots()
compaction_df.loc[30].comprate.plot(ax=a1)
compaction_df.loc[30,'2016-01-01'].totcomp-compaction_df.loc[30,'2016-12-31'].totcomp
compaction_df.loc[30].resample('a').mean()*spy

    #%%
def hl_analytic(rhos0, h, THL, AHL):
    '''
    Model steady-state firn density and age profiles and bubble close-off, uses m w.e. a^-1

    :param rhos0: surface density
    :param h: depth
    :param THL: 
    :param AHL:

    :return age: age vector of firn column with steady-state dynamics
    :return rho: density vector of firn column with steady state dynamics
    '''

    hSize = np.size(h)
    rhos = rhos0 / 1000.0

    A = AHL * RHO_I_MGM
    k0 = 11.0 * np.exp(-10160 / (R * THL))
    k1 = 575.0 * np.exp(-21400 / (R * THL))

    # depth of critical density, eqn 8 from Herron and Langway
    h0_55 = 1 / (RHO_I_MGM * k0) * (np.log(RHO_1_MGM / (RHO_I_MGM - RHO_1_MGM)) - np.log(rhos / (RHO_I_MGM - rhos)))
    Z0 = np.exp(RHO_I_MGM * k0 * h + np.log(rhos / (RHO_I_MGM - rhos)))

    # The boundary from zone 1 to zone 2 = t0_55
    t0_55 = 1 / (k0 * A) * np.log((RHO_I_MGM - rhos) / (RHO_I_MGM - RHO_1_MGM))
    rho_h0 = (RHO_I_MGM * Z0) / (1 + Z0)
    if np.max(rho_h0) >= RHO_I_MGM:
        t0 = np.zeros(hSize)
        for jj in range(hSize):
            if rho_h0[jj] <= RHO_I_MGM - 0.001:
                t0[jj] = (1 / (k0 * A) * np.log((RHO_I_MGM - rhos) / (RHO_I_MGM - rho_h0[jj])))
                jj_max = jj
            else:
                t0[jj] = (t0[jj_max])
    else:
        t0 = 1 / (k0 * A) * np.log((RHO_I_MGM - rhos) / (RHO_I_MGM - rho_h0))

    Z1 = np.exp(RHO_I_MGM * k1 * (h - h0_55) / np.sqrt(A) + np.log(RHO_1_MGM / (RHO_I_MGM - RHO_1_MGM)))
    Z = np.concatenate((Z0[h < h0_55], Z1[h > h0_55]))
    rho_h = (RHO_I_MGM * Z) / (1 + Z)
    tp = np.ones(hSize)
    for j in range(hSize):
        if rho_h[j] < RHO_I_MGM - 0.01:
            tp[j] = 1 / (k1 * np.sqrt(A)) * np.log((RHO_I_MGM - RHO_1_MGM) / (RHO_I_MGM - rho_h[j])) + t0_55
            jMax = j
        else:
            tp[j] = tp[jMax]

    # Zone 1 and Zone 2 repsectively
    age = np.concatenate((t0[h < h0_55], tp[h > h0_55])) * S_PER_YEAR
    rho = rho_h * 1000

    return age, rho
#%%
def toYearFraction(date):
    def sinceEpoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())
    s = sinceEpoch

    year = date.year
    startOfThisYear = datetime(year=year, month=1, day=1)
    startOfNextYear = datetime(year=year+1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration

    return date.year + fraction




#%% 
def butter_lowpass_filter(dd, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = sps.butter(order, normal_cutoff, btype='low', analog=False)
    y = sps.filtfilt(b, a, dd)
    return y

#%% 
def strainfun(arr):
    strain = -1*np.log(arr/arr[0])
    return strain