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
fontsz = 15
plt.rc('xtick',labelsize=fontsz)
plt.rc('ytick',labelsize=fontsz)
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams['axes.titlesize'] = fontsz
plt.rcParams["axes.labelsize"] = fontsz
plt.rcParams['figure.figsize'] = [10, 8]
palette = sns.color_palette()

#% Loading data
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

# % Plotting erroneous periods removed from analysis
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
    fig = plt.figure()
    compaction_df.loc[erroneous_periods[i][0],'compaction_borehole_length_m'].plot(marker='o')
    tmp.plot(marker='o')
    plt.title('Instrument '+str(erroneous_periods[i][0]))
    fig.savefig('figures/err_instr_'+str(erroneous_periods[i][0])+'.png')
# % Removing erroneous periods from the analysis
compaction_df.loc[13,'compaction_borehole_length_m'].loc['2018-02-20':] = np.nan
compaction_df.loc[10,'compaction_borehole_length_m'].loc['2019-07-29':] = np.nan
compaction_df.loc[42,'compaction_borehole_length_m'].loc[:'2017-10-14'] = np.nan
compaction_df.loc[48,'compaction_borehole_length_m'].loc[:'2017-10-18'] = np.nan
compaction_df.loc[48,'compaction_borehole_length_m'].loc['2018-05-27':'2018-07-19'] = np.nan
compaction_df.loc[1,'compaction_borehole_length_m'].loc[:'2013-12-01'] = np.nan
compaction_df.loc[35,'compaction_borehole_length_m'].loc[:'2016-09-01'] = np.nan
compaction_df.loc[43,'compaction_borehole_length_m'].loc['2018-07-16':] = np.nan    

compaction_df = compaction_df.loc[compaction_df.index.get_level_values(0)<100,:]
# % loading CEN-COM data

df_CEN = pd.read_csv('data/CEN-COM/FIRNCO_DATA_2020.01.20.csv')

df_CEN['date'] = pd.to_datetime(df_CEN.TRANSMISSION_TIMESTAMP)
df_CEN_1 = df_CEN[['date', 'INST_1_LENGTH_CORRECTED_M']].copy()
df_CEN_2 = df_CEN[['date', 'INST_2_LENGTH_CORRECTED_M']].copy()
df_CEN_3 = df_CEN[['date', 'INST_3_LENGTH_CORRECTED_M']].copy()
df_CEN_1.columns = ['date', 'compaction_borehole_length_m']
df_CEN_2.columns = ['date', 'compaction_borehole_length_m']
df_CEN_3.columns = ['date', 'compaction_borehole_length_m']
df_CEN_1a = df_CEN_1.loc[df_CEN_1.date<'2019-05-15T12:00:00',:].copy()
df_CEN_1b = df_CEN_1.loc[df_CEN_1.date>'2019-05-16T12:00:00',:].copy()
df_CEN_2 = df_CEN_2.loc[df_CEN_2.date<'2019-05-15T12:00:00',:]
df_CEN_3 = df_CEN_3.loc[df_CEN_3.date<'2019-05-15T12:00:00',:]

# df_CEN_1a.set_index('date').plot()
# df_CEN_1b.set_index('date').plot()
# df_CEN_2.set_index('date').plot()
# df_CEN_3.set_index('date').plot()

df_CEN_1a['instrument_id'] = compaction_df.index.get_level_values(0).max()+1
df_CEN_2['instrument_id'] = compaction_df.index.get_level_values(0).max()+2
df_CEN_3['instrument_id'] = compaction_df.index.get_level_values(0).max()+3
df_CEN_1b['instrument_id'] = compaction_df.index.get_level_values(0).max()+4

df_CEN = df_CEN_1a.append(df_CEN_1b).append(df_CEN_2).append(df_CEN_3)
df_CEN['sitename'] = 'CEN'

compaction_df = compaction_df.reset_index()[['instrument_id', 'date', 'sitename', 'compaction_borehole_length_m']].append(df_CEN[['instrument_id', 'date', 'sitename', 'compaction_borehole_length_m']]).set_index(['instrument_id', 'date'])

# % Loading FA data
df_FA_1 = pd.read_csv('data/AquiferCompactionDataFINAL/CompactionSEGreenland_L2.txt',sep='\t',header=None)
df_FA_1['instrument_id'] = compaction_df.index.get_level_values(0).max()+1
df_FA_2 = pd.read_csv('data/AquiferCompactionDataFINAL/CompactionSEGreenland_L3.txt',sep='\t',header=None)
df_FA_2['instrument_id'] = compaction_df.index.get_level_values(0).max()+2
df_FA = df_FA_1.append(df_FA_2)
df_FA.columns = ['year','month','day','hour','compaction_borehole_length_m','instrument_id']
df_FA['date'] = pd.to_datetime(df_FA[['year','month','day','hour']])
df_FA = df_FA.reset_index().set_index('date').groupby('instrument_id').resample('D').mean().reset_index('date')
df_FA['sitename'] = 'FA'

# df_FA.loc[df_FA.instrument_id==120,:].set_index('date').compaction_borehole_length_m.plot()
# df_FA.loc[df_FA.instrument_id==121,:].set_index('date').compaction_borehole_length_m.plot()
compaction_df = compaction_df.reset_index()[['instrument_id', 'date', 'sitename', 'compaction_borehole_length_m']].append(df_FA[['instrument_id', 'date', 'sitename', 'compaction_borehole_length_m']]).set_index(['instrument_id', 'date'])
plt.close('all')

compaction_df = compaction_df.loc[compaction_df.sitename.notnull(),:]
compaction_df = compaction_df.loc[compaction_df.index.get_level_values(0).notnull(),:]

# % Updating metadata table
inst_meta_df = inst_meta_df.reset_index()[['instrument_ID', 'sitename', 'installation_daynumber_YYYYMMDD', 'borehole_top_from_surface_m', 'borehole_bottom_from_surface_m', 'borehole_initial_length_m']].append(pd.read_excel('data/meta_CEN_FA.xlsx')).set_index('instrument_ID')

#% plotting borehole length
sites = compaction_df.sitename.unique()

f, ax = fcl.multi_plot(inst_meta_df, compaction_df,
           var = 'compaction_borehole_length_m',
           sites = sites, sp1 = 5, sp2 = 2,
           title =  'Borehole total length change (m)',
           filename_out ='compaction_borehole_length_m')
for k in range(np.size(ax)):
    i,j = np.unravel_index(k, ax.shape)
    # ax[i,j].set_ylim((-1.25,0))
    ax[i,j].legend(loc='lower left', fontsize = 10)
    ax[i,j].set_title(ax[i,j].title.get_text(), fontsize = 13)

# % Filtering
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import interp1d

compaction_df['borehole_length_smoothed_m'] = np.nan
for instr in compaction_df.reset_index().instrument_id.unique():
    print(instr)
    if instr in [42, 46, 48, 56]:
        frac = 0.2
    else:
        frac = 0.1

    out = lowess(compaction_df.loc[instr,'compaction_borehole_length_m'].values,
                                  compaction_df.loc[instr,'compaction_borehole_length_m'].index.values, 
                                  is_sorted=True, frac=frac, it=0)
    compaction_df.loc[instr,'borehole_length_smoothed_m'] = interp1d(out[:,0], out[:,1], bounds_error=False)(compaction_df.loc[instr,'compaction_borehole_length_m'].index.values.astype('float64'))
    
    
    plt.figure()
    compaction_df.loc[instr,'compaction_borehole_length_m'].plot(label = 'After smoothing',marker='o')
    compaction_df.loc[instr,'borehole_length_smoothed_m'].plot(label = 'After smoothing')
    plt.legend()
    plt.title(compaction_df.loc[instr,'sitename'].unique()[0]+', instr '+str(int(instr)))
compaction_df.loc[compaction_df.compaction_borehole_length_m.isnull(),'borehole_length_smoothed_m'] = np.nan

#% plotting borehole length
sites = compaction_df.sitename.unique()

f, ax = fcl.multi_plot(inst_meta_df, compaction_df,
           var = 'borehole_length_smoothed_m',
           sites = sites, sp1 = 5, sp2 = 2,
           title =  'Borehole total length change (m)',
           filename_out ='borehole_length_smoothed_m')
for k in range(np.size(ax)):
    i,j = np.unravel_index(k, ax.shape)
    # ax[i,j].set_ylim((-1.25,0))
    ax[i,j].legend(loc='lower left', fontsize = 10)
    ax[i,j].set_title(ax[i,j].title.get_text(), fontsize = 13)


#% resampling to daily values
plt.close('all')

tmp = compaction_df.reset_index().set_index('date').copy()
compaction_df = tmp.groupby('instrument_id').resample('D').first().copy()
compaction_df = compaction_df.drop(columns='instrument_id')
for instr in compaction_df.index.get_level_values(0).unique():
    compaction_df.loc[instr, 'sitename'] = compaction_df.loc[compaction_df.sitename.notnull()].loc[instr, 'sitename'].unique()[0]
compaction_df['borehole_shortening_m'] = 0

#%%
ind_start = 0
count = -1
for site in sites:
    count = count+1
    ind_instr = inst_meta_df.loc[inst_meta_df['sitename'] == site].index
    
    for instr_nr in ind_instr:
        if  np.isin(instr_nr, np.unique(compaction_df.index.get_level_values(0))):
            if compaction_df.loc[instr_nr,'compaction_borehole_length_m'].shape[0]>ind_start:
                
                # discarding first 60 days
                tmp =  compaction_df.loc[instr_nr,'compaction_borehole_length_m'].values
                tmp[:ind_start] = np.nan
                compaction_df.loc[instr_nr,'compaction_borehole_length_m'] = tmp
                
                # time of first valid measurement
                time_start =  compaction_df.loc[instr_nr,'compaction_borehole_length_m'][ind_start:].first_valid_index()
                compaction_df.loc[instr_nr,'borehole_shortening_m'] = - compaction_df.loc[instr_nr,'borehole_length_smoothed_m'].loc[time_start] + compaction_df.loc[instr_nr,'borehole_length_smoothed_m'].values
                
msk = compaction_df['borehole_shortening_m']>0
compaction_df.loc[msk,'borehole_shortening_m'] = np.nan
                
msk = compaction_df['compaction_borehole_length_m'].isna()
compaction_df.loc[msk,'borehole_shortening_m'] = np.nan

#% plotting borehole length
f, ax = fcl.multi_plot(inst_meta_df, compaction_df,
           var = 'borehole_shortening_m',
           sites = sites, sp1 = 5, sp2 = 2,
           title =  'Borehole total length change (m)',
           filename_out ='borehole_shortening_m')
for k in range(np.size(ax)):
    i,j = np.unravel_index(k, ax.shape)
    # ax[i,j].set_ylim((-1.25,0))
    ax[i,j].legend(loc='lower left', fontsize = 10)
    ax[i,j].set_title(ax[i,j].title.get_text(), fontsize = 13)
f.savefig('figures/borehole_shortening_m.tiff', dpi=600, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
# f.savefig('figures/fig4_new_borehole_shortening_m.pdf')


#% daily compaction
compaction_df['daily_compaction_md'] = -compaction_df.groupby(level=0)['borehole_length_smoothed_m'].diff().values*1000
from scipy.ndimage import binary_dilation
msk = binary_dilation(compaction_df['compaction_borehole_length_m'].isna().values)
compaction_df.loc[msk,'daily_compaction_md'] = np.nan

f, ax = fcl.multi_plot(inst_meta_df, compaction_df,
           var = 'daily_compaction_md',
           sites = sites, sp1 = 5, sp2 = 2,
           title =  'Daily compaction rate (mm d$^{-1}$)',  
           filename_out ='daily_compaction_md')
for k in range(np.size(ax)):
    i,j = np.unravel_index(k, ax.shape)
    ax[i,j].legend(loc='lower left', fontsize = 10)
    ax[i,j].set_title(ax[i,j].title.get_text(), fontsize = 13)
    ax[i,j].set_ylim((-0.1,2.5))
f.savefig('figures/daily_compaction_md.tiff', dpi=600, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
# f.savefig('figures/fig5_daily_compaction_md_smoothed.pdf')


# %% output text file
compaction_df.to_csv('data/FirnCover_all_station.csv')
inst_meta_df.to_csv('data/FirnCover_metadata_all_station.csv')
