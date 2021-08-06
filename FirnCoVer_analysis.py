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
import datetime
import pandas as pd
import time
import seaborn as sns

from FirnCover_lib import R, BDOT_TO_A, RHO_W_KGM, P_0, epoch
import FirnCover_lib as fcl

import matplotlib.dates as mdates
years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

np.seterr(invalid='ignore')
# %matplotlib inline
# %matplotlib qt

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
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

mpl.rcParams['axes.formatter.useoffset'] = False
mpl.rcParams.update({'figure.autolayout': False})
fontsz = 20
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams['axes.titlesize'] = 22
plt.rcParams["axes.labelsize"] = 22
plt.rcParams['figure.figsize'] = [10, 8]
palette = sns.color_palette()

#%% Loading data
# Things to change
figex = '_200504.png' #extension for figures (name plus type)
pathtodata = './data'
date1 = '2021_07_30' # Date on the hdf5 file
sites=['Summit','KAN-U','NASA-SE','Crawford','EKT','Saddle','EastGrip','DYE-2']

### Import the FirnCover data tables. 
filename='FirnCoverData_2.0_' + date1 + '.h5'
filepath=os.path.join(pathtodata,filename)
compaction_df, airtemp_df, inst_meta_df = fcl.import_firncover_dataset(filepath)

# Loading side data
statmeta_df, sonic_df, rtd_df, rtd_trun, rtd_dep,metdata_df = fcl.load_metadata(compaction_df,filepath,sites)

# %% Plotting erroneous periods removed from analysis

erroneous_periods = [[13,'2018-02-20','end'] , 
[10,'2019-07-29','end'] ,
[42,'start','2017-10-14'] ,
[48,'start','2017-10-18'] , 
[48,'2018-05-27','2018-07-19'] ,
[1,'start','2013-12-01'] ,
[35,'start','2016-09-15'] ,
[43,'2018-07-16','end'] ] 

for i in range(len(erroneous_periods)):
    print(erroneous_periods[i])

    if erroneous_periods[i][1] == 'start':
        tmp = compaction_df.loc[erroneous_periods[i][0],
                          'compaction_borehole_length_m'].loc[:erroneous_periods[i][2]]
    elif erroneous_periods[i][2] == 'end':
        tmp = compaction_df.loc[erroneous_periods[i][0],
                                'compaction_borehole_length_m'].loc[erroneous_periods[i][1]:]
    else: 
        tmp = compaction_df.loc[erroneous_periods[i][0],
                          'compaction_borehole_length_m'].loc[erroneous_periods[i][1]:erroneous_periods[i][2]]
    
    if len(tmp.loc[tmp.notnull()])==0:
        print('already removed')
        continue
    plt.figure()
    compaction_df.loc[erroneous_periods[i][0],'compaction_borehole_length_m'].plot(marker='o')
    tmp.plot(marker='o')
    plt.title('Instrument '+str(erroneous_periods[i][0]))
 
# %% Removing erroneous periods from the analysis
compaction_df.loc[13,'compaction_borehole_length_m'].loc['2018-02-20':] = np.nan
compaction_df.loc[10,'compaction_borehole_length_m'].loc['2019-07-29':] = np.nan
compaction_df.loc[42,'compaction_borehole_length_m'].loc[:'2017-10-14'] = np.nan
compaction_df.loc[48,'compaction_borehole_length_m'].loc[:'2017-10-18'] = np.nan
compaction_df.loc[48,'compaction_borehole_length_m'].loc['2018-05-27':'2018-07-19'] = np.nan
compaction_df.loc[1,'compaction_borehole_length_m'].loc[:'2013-12-01'] = np.nan
compaction_df.loc[35,'compaction_borehole_length_m'].loc[:'2016-09-01'] = np.nan
compaction_df.loc[43,'compaction_borehole_length_m'].loc['2018-07-16':] = np.nan    

#%% calculating borehole shortening
compaction_df = compaction_df.assign(borehole_length_m_smoothed = 0*compaction_df['compaction_borehole_length_m'])
compaction_df = compaction_df.assign( borehole_shortening_m = 0*compaction_df['compaction_borehole_length_m'])
compaction_df = compaction_df.assign( delta_L_m_smoothed = 0*compaction_df['compaction_borehole_length_m'])

ind_start = 60
count = -1
for site in sites:
    count = count+1
    ind_instr = inst_meta_df.loc[inst_meta_df['sitename'] == site].index
    
    for instr_nr in ind_instr:
        if  np.isin(instr_nr, np.unique(compaction_df.index.get_level_values(0))):
            if compaction_df.loc[instr_nr,'compaction_borehole_length_m'].shape[0]>ind_start:
                
                
                compaction_df.loc[instr_nr,'borehole_length_m_smoothed'] =   compaction_df.loc[instr_nr,'compaction_borehole_length_m'].rolling(60,center=True, win_type='gaussian',min_periods=30).mean(std=5).values
                
                time_start =  compaction_df.loc[instr_nr,'compaction_borehole_length_m'][ind_start:].first_valid_index()
                compaction_df.loc[instr_nr,'borehole_shortening_m'] = - compaction_df.loc[instr_nr,'compaction_borehole_length_m'].loc[time_start] + compaction_df.loc[instr_nr,'compaction_borehole_length_m'].values

                                
                compaction_df.loc[instr_nr,'delta_L_m_smoothed'] = - compaction_df.loc[instr_nr,'borehole_length_m_smoothed'].loc[time_start]  + compaction_df.loc[instr_nr,'borehole_length_m_smoothed'].values
                
msk = compaction_df['borehole_shortening_m']>0
compaction_df.loc[msk,'borehole_shortening_m'] = np.nan
msk = compaction_df['delta_L_m_smoothed']>0
compaction_df.loc[msk,'delta_L_m_smoothed'] = np.nan
                
msk = compaction_df['compaction_borehole_length_m'].isna()
compaction_df.loc[msk,'borehole_length_m_smoothed'] = np.nan
compaction_df.loc[msk,'borehole_shortening_m'] = np.nan
compaction_df.loc[msk,'delta_L_m_smoothed'] = np.nan

#% plotting borehole length
fcl.multi_plot(inst_meta_df, compaction_df,
           var = 'compaction_borehole_length_m',
           sites = sites, sp1 = 4, sp2 = 2,
           title =  'Borehole length (m)', 
           filename_out ='compaction_borehole_length_m')

f, ax = fcl.multi_plot(inst_meta_df, compaction_df,
           var = 'borehole_shortening_m',
           sites = sites, sp1 = 4, sp2 = 2,
           title =  'Borehole total length change (m)',
           filename_out ='borehole_shortening_m')
for k in range(np.size(ax)):
    i,j = np.unravel_index(k, ax.shape)
    ax[i,j].set_ylim((-1.25,0))
f.savefig('figures/borehole_shortening_m.png')


#%% daily compaction
compaction_df = compaction_df.assign(daily_compaction_md = np.nan*compaction_df['compaction_borehole_length_m'])

compaction_df['daily_compaction_md'] = compaction_df.groupby(level=0)['compaction_borehole_length_m'].diff()*1000

compaction_df = compaction_df.assign(daily_compaction_md_smoothed = np.nan*compaction_df['borehole_length_m_smoothed'])

compaction_df['daily_compaction_md_smoothed'] = - compaction_df.groupby(level=0)['borehole_length_m_smoothed'].diff()*1000
# fcl.hampel(compaction_df.groupby(level=0)['borehole_length_m_smoothed'].diff(), k=30, t0=3)

msk = compaction_df['borehole_length_m_smoothed'].isna()
compaction_df.loc[msk,'daily_compaction_md'] = np.nan
compaction_df.loc[msk,'daily_compaction_md_smoothed'] = np.nan

# msk = np.logical_or(compaction_df['daily_compaction_md']>0, compaction_df['daily_compaction_md']<-0.006)
# compaction_df.loc[msk,'daily_compaction_md']=np.nan

# msk = np.logical_or(compaction_df['daily_compaction_md_smoothed']>0, compaction_df['daily_compaction_md_smoothed']<-0.006)
# compaction_df.loc[msk,'daily_compaction_md_smoothed']=np.nan


f, ax = fcl.multi_plot(inst_meta_df, compaction_df,
           var = 'daily_compaction_md_smoothed',
           sites = sites, sp1 = 4, sp2 = 2,
           title =  'Daily compaction rate (mm d$^{-1}$)',  
           filename_out ='daily_compaction_md_smoothed')
# for k in range(np.size(ax)):
#     i,j = np.unravel_index(k, ax.shape)
ax[1,1].set_ylim((0, 2))
ax[3,1].set_ylim((0,2))
f.savefig('figures/daily_compaction_md_smoothed.png')

#%% print period where instruments where tower was not working
for instr_nr in np.array([32, 7, 11,17]):
    site = inst_meta_df.loc[inst_meta_df.index == instr_nr, 'sitename'].values
    print(site)
    df = compaction_df.loc[instr_nr,'compaction_borehole_length_m']
    print(df.first_valid_index())
    mask = df.isna()
    d = df.index.to_series()[mask].groupby((~mask).cumsum()[mask]).agg(['first', 'last'])
    d.rename(columns=dict(size='num of contig null', first='Start_Date')).reset_index(drop=True)
    print(d)

#%% Air temperature and surface height
airtemp_df = airtemp_df.sort_index()
airtemp_df.loc[pd.IndexSlice['KAN-U', '2013-09-01':'2013-11-15'], 'air_temp_C'] = np.nan
airtemp_df.loc[pd.IndexSlice['KAN-U', '2019-01-01':], 'air_temp_C'] = np.nan 

airtemp_df.loc[pd.IndexSlice['DYE-2', '2013-09-01':'2013-11-15'], 'air_temp_C'] = np.nan 

airtemp_df.loc[pd.IndexSlice['DYE-2' ,'2014-04-01':'2015-10-01'], 'air_temp_C'] = np.nan 
airtemp_df.loc[pd.IndexSlice['DYE-2' ,'2019-06-15':], 'air_temp_C'] = np.nan 

sonic_df = sonic_df.sort_index()
sonic_df.loc[pd.IndexSlice['Summit', '2017-06-01':'2017-09-01'], 'delta'] = np.nan
sonic_df.loc[pd.IndexSlice['EastGrip', '2017-11-01':], 'delta'] = np.nan
sonic_df.loc[pd.IndexSlice['KAN-U', '2017-11-01':'2018-04-01'], 'delta'] = np.nan
sonic_df.loc[pd.IndexSlice['EKT', '2017-11-01':'2018-04-01'], 'delta'] = np.nan

f1, ax = plt.subplots(4 ,2,figsize=(15, 13))
f1.subplots_adjust(hspace=0.2, wspace=0.2,
                   left = 0.08 , right = 0.9 ,
                   bottom = 0.2 , top = 0.9)
count = -1
for site in sites:
    print(site)
    count = count+1
    i,j = np.unravel_index(count, ax.shape)
         
    if np.isin(site, airtemp_df.index.get_level_values(0).unique()):
        color1 = 'tab:red'
        ax[i,j].plot(airtemp_df.loc[site,'air_temp_C'].resample('D').mean(),color=color1)
        ax[i,j].set_ylabel('', color=color1)
        ax[i,j].tick_params(axis='y', labelcolor=color1)
    
        ax2 = ax[i,j].twinx()  
        color2 = 'tab:blue'
        ax2.set_ylabel('', color=color2)  
        ax2.plot(-sonic_df.loc[site,'delta'], color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)

        ax[i,j].set_title(site)
        if site == 'Crawford':
            ax[i,j].set_title('Crawford Point')

        ax[i,j].set_xlim([datetime.date(2012, 5, 1), datetime.date(2019, 10, 1)])    
        ax[i,j].xaxis.set_major_locator(years)
        ax[i,j].xaxis.set_major_formatter(years_fmt)
        ax[i,j].xaxis.set_minor_locator(months)
        ax[i,j].set_xlabel("")
        for k in range(2013,2020):
            ax[i,j].axvspan(*mdates.datestr2num([str(k)+'-06-01', str(k)+'-09-01']), color='orange', alpha=0.1)
        
    if count<len(sites)-2:
        ax[i,j].set_xticklabels("")
f1.text(0.5, 0.1, 'Year', ha='center', size = 20)
f1.text(0.02, 0.5, 'Daily air temperature ($^o$C)', va='center', rotation='vertical', size = 20, color = color1)
f1.text(0.95, 0.5, 'Surface height (m)', va='center', rotation='vertical', size = 20, color = color2)
f1.savefig('figures/Ta_HS.png')
    
    
#%% firn temperature

sites2 = sites.copy()
sites2.remove('EastGrip')
sites2.remove('NASA-SE')

f1, ax = plt.subplots(3,2,figsize=(20, 15))
f1.subplots_adjust(hspace=0.2, wspace=0.1,
                   left = 0.08 , right = 0.85 ,
                   bottom = 0.2 , top = 0.9)
count = -1
for site in sites2:
    print(site)
    count = count+1
    i,j = np.unravel_index(count, ax.shape)
    
    sitetemp=rtd_trun.loc[site]
    sitedep = rtd_dep.loc[site]
    n_grid = np.linspace(0,12,15)
    time=sitetemp.index.values
    temps = sitetemp.values
    if site == 'Crawford':
        temps[temps>-0.5]=np.nan    
        print(np.nanmax(temps))
        
    surface_height = sonic_df.loc[site,'delta'].interpolate(limit = 24*7)
    time_surf_height = surface_height.index.get_level_values(0).values
    surface_height = surface_height.loc[ np.isin(time_surf_height, time)]
    temps = temps[np.isin(time, time_surf_height), :]
    time = time[np.isin(time, time_surf_height)]

    time_sitedep = sitedep.index.get_level_values(0).values
    sitedep = sitedep.loc[np.isin(time_sitedep, time)]
    time_sitedep = sitedep.index.get_level_values(0).values
    depths = sitedep.values
    depths = (sitedep.values.T - surface_height.to_numpy()).T
    
    t_interp=np.zeros((depths.shape[0],len(n_grid)))
    for kk in range(depths.shape[0]):
            tif = sp.interpolate.interp1d(depths[kk,:], temps[kk,:], bounds_error=False)
            t_interp[kk,:]= tif(n_grid)
    for kk in range(t_interp.shape[1]):
        t_interp[:,kk] = pd.DataFrame(t_interp[:,kk], time).interpolate(limit = 7).values.reshape(1,-1)
        
    t_interp[t_interp>0]=0

    cax1 = ax[i,j].contourf(time,n_grid,t_interp.T, 50, extend='neither',
                            vmin=-50, vmax=0, zorder=0)
    surface_height.plot(ax=ax[i,j],linewidth=3,rot=0)

    ax[i,j].set_title(site)
    if site == 'Crawford':
        ax[i,j].set_title('Crawford Point')

    ax[i,j].set_ylim( 10, sonic_df.loc[site,'delta'].min()*1.2)
    ax[i,j].set_xlim("2015-05-21", '2019-09-04')
    
    # ax[i,j].set_xticklabels(ax[i,j].xaxis.get_majorticklabels(),rotation=0)
    ax[i,j].xaxis.set_major_locator(years)
    ax[i,j].xaxis.set_major_formatter(years_fmt)
    ax[i,j].xaxis.set_minor_locator(months)
    ax[i,j].set_xlabel("")
    if count<len(sites2)-2:
        ax[i,j].set_xticklabels("")
cbar_ax = f1.add_axes([0.9, 0.2, 0.02, 0.7])
cb1 = f1.colorbar(cax1, cax=cbar_ax)
cb1.set_label('Firn temperature ($^o$C)')
cb1.set_ticks([-20, -15, -10,-5,0])  # horizontal colorbar
f1.text(0.5, 0.1, 'Year', ha='center', size = 20)
f1.text(0.02, 0.5, 'Depth (m)', va='center', rotation='vertical', size = 20)
f1.savefig('figures/RTD_temp.png')
    # plt.close(f1)   


#%% Average temp, average daily compaction

for site in sites:
    print(site)
     
    if np.isin(site, airtemp_df.index.get_level_values(0).unique()):
        ind_instr = inst_meta_df.loc[inst_meta_df['sitename'] == site].index
        df = pd.DataFrame()
        df['Ta_count'] = airtemp_df.loc[site,'air_temp_C'].resample('Y').count()
        df['Ta_mean'] = airtemp_df.loc[site,'air_temp_C'].resample('Y').mean()
        df.loc[df['Ta_count']<365*24*0.9, 'Ta_mean'] = np.nan
        df.drop(columns='Ta_count')
        for instr_nr in ind_instr:
            if np.isin(instr_nr, np.unique(compaction_df.index.get_level_values(0))):
                df[instr_nr] = compaction_df.loc[instr_nr,'daily_compaction_md_smoothed'].resample('Y').mean()
        df.loc['mean'] = df.mean()
        print(df)

#%% 10 m firn temp
f1, ax = plt.subplots(1,1,figsize=(10, 10))

count = -1
for site in sites2:
    print(site)
    count = count+1
    
    sitetemp=rtd_trun.loc[site]
    sitedep = rtd_dep.loc[site]
    n_grid = np.linspace(0,15,15)
    time=sitetemp.index.values
    temps = sitetemp.values
    
    surface_height = sonic_df.loc[site,'delta']
    time_surf_height = surface_height.index.get_level_values(0).values
    surface_height = surface_height.loc[ np.isin(time_surf_height, time)]

    temps = temps[np.isin(time, time_surf_height), :]
    time = time[np.isin(time, time_surf_height)]

    time_sitedep = sitedep.index.get_level_values(0).values
    sitedep = sitedep.loc[np.isin(time_sitedep, time)]
    time_sitedep = sitedep.index.get_level_values(0).values
    depths = sitedep.values

    depths = (sitedep.values.T - surface_height.to_numpy()).T
    depths_from_surface = (depths.T - surface_height.to_numpy()).T
    T_10m = np.nan*depths_from_surface[:,0]

    for i in range(temps.shape[0]):
        ind_nonan = np.argwhere(np.logical_and(~np.isnan(temps[i,:]), ~np.isnan(depths_from_surface[i,:])))
        if (len(depths_from_surface[i,ind_nonan][:,0])>2 and \
                          len(temps[i,ind_nonan][:,0])>2) and np.max(depths_from_surface[i,ind_nonan][:,0])>10:
            T_10m[i] = sp.interpolate.interp1d(depths_from_surface[i,ind_nonan][:,0],\
                                 temps[i,ind_nonan][:,0], kind='linear',\
                                         assume_sorted=False)(10)
    T_10m[np.argwhere(T_10m>-8)] = np.nan
    if np.sum(np.isnan(T_10m))>1:
        T_10m[np.argwhere(np.isnan(T_10m))]= \
            sp.interpolate.interp1d(np.argwhere(~np.isnan(T_10m))[:,0],\
                                    T_10m[np.argwhere(~np.isnan(T_10m))][:,0], \
                                        fill_value = 'extrapolate') (np.argwhere(np.isnan(T_10m)))
    
    T_10m[np.where(time<np.datetime64('2015-06-01'))] = np.nan
    
    if site == 'Summit':
        ind = np.where(np.logical_and(
            time>np.datetime64('2017-05-15'),
            time<np.datetime64('2017-09-10')))
    if site == 'Saddle':
        ind = np.where( time>np.datetime64('2017-05-10') )
    if site == 'EKT':
        ind = np.where(np.logical_and(
            time>np.datetime64('2017-05-01'),
            time<np.datetime64('2018-06-01')))
    if site == 'Crawford':
        ind1 = np.logical_and(
            time>np.datetime64('2017-03-01'),
            time<np.datetime64('2017-07-01'))
        ind2 = np.logical_and(
            time>np.datetime64('2018-05-01'),
            time<np.datetime64('2017-06-01'))
        ind = np.where(np.logical_or(ind1,ind2))
    if site == 'KAN-U':
        ind = np.where(np.logical_and(
            time>np.datetime64('2017-11-01'),
            time<np.datetime64('2018-05-01')))
    T_10m[ind] = np.nan
    print('Average T10: ' + str(np.nanmean(T_10m)))

    ax.plot(time, T_10m, label=site,linewidth=2)
ax.legend()
ax.set_xlim([datetime.date(2015, 5, 1), datetime.date(2019, 10, 1)])
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_minor_locator(months)
ax.set_xlabel("Year")
ax.set_ylabel('10 m firn temperature (C)')
f1.savefig('figures/T10.png')
    # plt.close(f1)   

