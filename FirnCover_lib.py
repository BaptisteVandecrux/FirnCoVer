# -*- coding: utf-8 -*-
"""
Created on Sat May  9 16:04:39 2020

@author: bav
"""
import numpy as np
import pandas as pd
import tables as tb
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates

#% Constants
R          = 8.314                          # gas constant used to calculate Arrhenius's term
S_PER_YEAR = 31557600.0                     # number of seconds in a year
spy = S_PER_YEAR
RHO_1      = 550.0                          # cut off density for the first zone densification (kg/m^3)
RHO_2      = 815.0                          # cut off density for the second zone densification (kg/m^3)
RHO_I      = 917.0                          # density of ice (kg/m^3)
RHO_I_MGM  = 0.917                          # density of ice (g/m^3)
RHO_1_MGM  = 0.550                          # cut off density for the first zone densification (g/m^3)
GRAVITY    = 9.8                            # acceleration due to gravity on Earth
K_TO_C     = 273.15                         # conversion from Kelvin to Celsius
BDOT_TO_A  = S_PER_YEAR * RHO_I_MGM         # conversion for accumulation rate
RHO_W_KGM  = 1000.                          # density of water
P_0 = 1.01325e5
epoch =np.datetime64('1970-01-01')


#% Put the 'virtual' holes into the frame (i.e. the differential compaction between holes), Summit, EGRIP only for now.
def differential_compaction(compaction_df): 
    #### Removing bad time step at EastGRIP
    # compaction_df.loc[28,'2015-05-28']=compaction_df.loc[28,'2015-05-29']
    # compaction_df.loc[29,'2015-05-28']=compaction_df.loc[29,'2015-05-29']
    compaction_df.drop((27,'2015-05-28'),inplace=True)
    compaction_df.drop((26,'2015-05-28'),inplace=True)
    
    # Summmit subtracting 31 and 30, saves into 101
    # Summmit subtracting 32 and 30, saves into 102
    # Summmit subtracting 31 and 32, saves into 103
    # EastGRIP subtracting 28 and 26, saves into 104
    # EastGRIP subtracting 27 and 26, saves into 105
    # EastGRIP subtracting 27 and 28, saves into 106
    # NASA-SE subtracting 14 and 13, saves into 107
    # NASA-SE subtracting 15 and 13, saves into 108
    # NASA-SE subtracting 14 and 15, saves into 109
    # Crawford subtracting 24 and 22, saves into 110
    # Crawford subtracting 25 and 22, saves into 111
    # Crawford subtracting 23 and 22, saves into 112
    # Crawford subtracting 25 and 24, saves into 113
    # Crawford subtracting 23 and 24, saves into 114
    # Crawford subtracting 23 and 25, saves into 115
   
    ind =  np.array([[31, 30, 101], [32, 30, 102],
    [31, 32, 103],  [28, 26, 104],   [27, 26, 105],
    [27, 28, 106],  [14, 13, 107],
    [15, 13, 108], [14, 15, 109], [ 24, 22, 110], [25, 22, 111], [23, 22, 112],
    [25, 24, 113],[23, 24, 114],[23, 25, 115]])
    for i in range(ind.shape[0]):
        cd2 = compaction_df.loc[ind[i,1]].copy()
        cd2.hole_init_length = compaction_df.loc[ind[i,1],'hole_init_length'] 
        -compaction_df.loc[ind[i,0],'hole_init_length']
        cd2.hole_botfromsurf = pd.concat([compaction_df.loc[ind[i,1],'hole_botfromsurf'], 
                                          compaction_df.loc[ind[i,0],'hole_botfromsurf']],
                                         axis=1).max(axis=1)
        cd2.hole_topfromsurf = pd.concat([compaction_df.loc[ind[i,1],'hole_botfromsurf'],
                                          compaction_df.loc[ind[i,0],'hole_botfromsurf']],
                                         axis=1).min(axis=1)
        cd2.compaction_borehole_length_m =  compaction_df.loc[ind[i,1],'compaction_borehole_length_m']
        -compaction_df.loc[ind[i,0],'compaction_borehole_length_m']
        
        cd2['Compaction_Instrument_ID'] = int(ind[i,2])
        
        cd2.set_index(['Compaction_Instrument_ID',cd2.index],inplace=True)
        compaction_df = pd.concat([compaction_df,cd2])
        idx=compaction_df.index
        compaction_df.index = compaction_df.index.set_levels([idx.levels[0].astype(int),idx.levels[1]])
    
    return compaction_df

#% Loading function
def import_firncover_dataset(filepath):
    CVNfile=tb.open_file(filepath, mode='r', driver="H5FD_CORE")
    datatable=CVNfile.root.FirnCover
    
    print('Reading '+filepath)
    print('contains:')
    for key in datatable._v_children:
        print('   '+key)
        for child in datatable[key].colnames:
            print('       '+child)
    
    # metadata to pandas
    print('loading Compaction_Instrument_Metadata to inst_meta_df')
    inst_meta_df = pd.DataFrame.from_records(datatable.Compaction_Instrument_Metadata[:])
    inst_meta_df.sitename=inst_meta_df.sitename.str.decode("utf-8")
    inst_meta_df.set_index('instrument_ID',inplace=True)
    # inst_meta_df.loc[7,'borehole_bottom_from_surface_m']
    
    # airtemp to pandas
    print('loading Air_Temp_Hourly to airtemp_df')
    airtemp_df = pd.DataFrame.from_records(datatable.Air_Temp_Hourly[:])
    airtemp_df.sitename = airtemp_df.sitename.str.decode("utf-8")
    airtemp_df['date'] = pd.to_datetime(airtemp_df.daynumber_YYYYMMDD,format='%Y%m%d')+pd.to_timedelta(airtemp_df.hournumber_HH,unit='h')
    airtemp_df.set_index(['sitename','date'],inplace=True)
    
    
    # comapction table to pandas
    print('loading Compaction_Daily to compaction_df')
    compaction_df=pd.DataFrame.from_records(datatable.Compaction_Daily[:])
    compaction_df.sitename = compaction_df.sitename.str.decode("utf-8")
    compaction_df['date']= pd.to_datetime(compaction_df.daynumber_YYYYMMDD.values,format='%Y%m%d')
    compaction_df.set_index(['instrument_id', 'date'],inplace=True)
    compaction_df.drop(columns=["compaction_ratio", "compaction_wire_correction_ratio", "compaction_cable_distance_m"], inplace=True)
    compaction_df.sort_index(inplace=True)
       
    uids = compaction_df.index.get_level_values(0).unique()
    for ii in uids:
        tmp = compaction_df.loc[ii].resample('D').asfreq()
        tmp.insert(0, "instrument_id", ii* np.ones_like(tmp['compaction_borehole_length_m'].resample('D').asfreq()).astype(int), allow_duplicates = True) 
        if ii == 1:
            compaction_rs_df = tmp
        else:
            compaction_rs_df = compaction_rs_df.append(tmp)

    compaction_rs_df.set_index(['instrument_id'],inplace=True,append=True)
    compaction_rs_df= compaction_rs_df.reorder_levels(['instrument_id', 'date'])
    compaction_rs_df.sort_index(inplace=True)

    compaction_df =  compaction_rs_df
    
    compaction_df['hole_init_length']=0*compaction_df['compaction_borehole_length_m']
    compaction_df['hole_botfromsurf']=0*compaction_df['compaction_borehole_length_m']
    compaction_df['hole_topfromsurf']=0*compaction_df['compaction_borehole_length_m']
    
    for ii in uids:
        compaction_df.loc[ii,'hole_init_length'] = -inst_meta_df.loc[ii,'borehole_initial_length_m'] * np.ones_like( compaction_df.loc[ii,"compaction_borehole_length_m"].values)
        
        compaction_df.loc[ii,'hole_botfromsurf'] = -1*inst_meta_df.loc[ii,'borehole_bottom_from_surface_m'] * np.ones_like( compaction_df.loc[ii,"compaction_borehole_length_m"].values)
        
        compaction_df.loc[ii,'hole_topfromsurf'] = -1*inst_meta_df.loc[ii,'borehole_top_from_surface_m'] * np.ones_like(compaction_df.loc[ii,"compaction_borehole_length_m"].values)
         
    
    # Put the 'virtual' holes into the frame (i.e. the differential compaction between holes), Summit, EGRIP only for now.
    compaction_df_out = differential_compaction(compaction_df)
    
    return compaction_df_out, airtemp_df, inst_meta_df

#% Loading metadata, RTD and sonic ranger
def load_metadata(compaction_df,filepath,sites):
    CVNfile=tb.open_file(filepath, mode='r', driver="H5FD_CORE")
    datatable=CVNfile.root.FirnCover
    
    df=compaction_df.loc[41].copy()
    df['tvalue'] = df.index
    df['delta'] = (df['tvalue']-df['tvalue'].shift()).fillna(pd.Timedelta(seconds=0))
    
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
    
    # Meteorological_Daily to pandas
    print('loading Meteorological_Daily to metdata_df')
    metdata_df=pd.DataFrame.from_records(datatable.Meteorological_Daily[:])
    metdata_df.sitename=metdata_df.sitename.str.decode("utf-8")
    # pd.to_datetime(compaction_df.daynumber_YYYYMMDD.values,format='%Y%m%d')
    metdata_df['date']=pd.to_datetime(metdata_df.daynumber_YYYYMMDD.values,format='%Y%m%d')
    
    for site in sites:
        msk=(metdata_df['sitename']==site) & (metdata_df['date'] < statmeta_df.loc[site,'rtd_date'])
        metdata_df.drop(metdata_df[msk].index,inplace=True)
        if site == "NASA-SE":
            # NASA-SE had a new tower section in 5/17; distance raised is ??, use 1.7 m for now.
            # m2 = (metdata_df["sitename"] == site) & (metdata_df["date"] > "2017-05-10")
            # metdata_df.loc[m2, "sonic_range_dist_corrected_m"] = (
            #     metdata_df.loc[m2, "sonic_range_dist_corrected_m"] - 1.7
            # )
            m3 = (
                (metdata_df["sitename"] == site)
                & (metdata_df["date"] > "2017-02-12")
                & (metdata_df["date"] < "2017-04-12")
            )
            metdata_df.loc[m3, "sonic_range_dist_corrected_m"] = np.nan
        # elif site == "Crawford":
            # Crawford has bad sonic data for 11/3/17 to 2/16/18
            # m2 = (
            #     (metdata_df["sitename"] == site)
            #     & (metdata_df["date"] > "2017-11-03")
            #     & (metdata_df["date"] < "2018-02-16")
            # )
            # metdata_df.loc[m2, "sonic_range_dist_corrected_m"] = np.nan
        if site == "EKT":
            # EKT had a new tower section in 5/17; distance raised is 0.86 m.
            # m2 = (metdata_df["sitename"] == site) & (metdata_df["date"] > "2017-05-05")
            # metdata_df.loc[m2, "sonic_range_dist_corrected_m"] = (
            #     metdata_df.loc[m2, "sonic_range_dist_corrected_m"] - 0.86
            # )
            m2 = (metdata_df["sitename"] == site) & (metdata_df["date"] > "2018-05-15")
            metdata_df.loc[m2, "sonic_range_dist_corrected_m"] = (
                metdata_df.loc[m2, "sonic_range_dist_corrected_m"] - 0.5
            )
        # elif site == "Saddle":
        #     # Saddle had a new tower section in 5/17; distance raised is 1.715 m.
        #     m2 = (metdata_df["sitename"] == site) & (metdata_df["date"] > "2017-05-07")
            # metdata_df.loc[m2,'sonic_range_dist_corrected_m']=metdata_df.loc[m2,'sonic_range_dist_corrected_m']-1.715
        # elif site == "EastGrip":
            # Eastgrip has bad sonic data for 11/7/17 onward
            # m2 = (metdata_df["sitename"] == site) & (metdata_df["date"] > "2017-11-17")
            # metdata_df.loc[m2, "sonic_range_dist_corrected_m"] = np.nan
            # m3 = (
            #     (metdata_df["sitename"] == site)
            #     & (metdata_df["date"] > "2015-10-01")
            #     & (metdata_df["date"] < "2016-04-01")
            # )
            # metdata_df.loc[m3, "sonic_range_dist_corrected_m"] = np.nan
            # m4 = (
            #     (metdata_df["sitename"] == site)
            #     & (metdata_df["date"] > "2016-12-07")
            #     & (metdata_df["date"] < "2017-03-01")
            # )
            # metdata_df.loc[m4, "sonic_range_dist_corrected_m"] = np.nan
        if site == "DYE-2":
            m2 = (metdata_df["sitename"] == site) & (metdata_df["date"] > "2016-04-29")
            metdata_df.loc[m2, "sonic_range_dist_corrected_m"] = (
                metdata_df.loc[m2, "sonic_range_dist_corrected_m"] - 0.3
            )
            # m3 = (
            #     (metdata_df["sitename"] == site)
            #     & (metdata_df["date"] > "2015-12-24")
            #     & (metdata_df["date"] < "2016-05-01")
            # )
            # metdata_df.loc[m3, "sonic_range_dist_corrected_m"] = np.nan
    #         m4 = (metdata_df['sitename']==site)&(metdata_df['date']>'2016-12-07')&(metdata_df['date']<'2017-03-01')
    #         metdata_df.loc[m4,'sonic_range_dist_corrected_m']=np.nan
            
    metdata_df.reset_index(drop=True)
    # metdata_df.set_index(['sitename','date'],inplace=True)
   
    sonic_df = metdata_df[['sitename', 'date', 'sonic_range_dist_corrected_m']].set_index(['sitename', 'date'])
    sonic_df.columns = ['sonic_m']
    sonic_df.sonic_m[sonic_df.sonic_m<-100]=np.nan
    sonic_df.loc['Saddle','2015-05-16']=sonic_df.loc['Saddle','2015-05-17']

# filtering
    gradthresh = 0.1

    for site in sonic_df.index.unique(level='sitename'):
        dd = statmeta_df.loc[site]['rtd_date']
        if site=='Saddle':
            dd = dd + pd.Timedelta('1D')
        sonic_df.loc[site,'delta']=sonic_df.loc[[site]].sonic_m-sonic_df.loc[(site,dd)].sonic_m
        
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
    
    rtd_df=pd.DataFrame.from_records(datatable.Firn_Temp_Daily[:].tolist(), columns=datatable.Firn_Temp_Daily.colnames)
    rtd_df.sitename=rtd_df.sitename.str.decode("utf-8")
    rtd_df['date']=pd.to_datetime(rtd_df.daynumber_YYYYMMDD.values,format='%Y%m%d')
    rtd_df.set_index(['sitename','date'])
    
    rtd_trun = rtd_df[['sitename','date','RTD_temp_avg_corrected_C']].copy().set_index(['sitename','date'])
    rtd_trun.columns = ['T_avg']
    rtd_trun[zz]=pd.DataFrame(rtd_trun.T_avg.values.tolist(),index=rtd_trun.index)
    rtd_trun.drop('T_avg',axis=1,inplace=True)
    rtd_trun.replace(-100.0,np.nan,inplace=True)
    
    return statmeta_df, sonic_df, rtd_df, rtd_trun, rtd_dep, metdata_df

#% multiplot
def multi_plot(inst_meta_df, compaction_df, var = 'daily_compaction_md',
               var2 = '',
               sites = 'Summit', sp1 = 4, sp2 = 2,
               title = '', filename_out = 'plot'):
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    years_fmt = mdates.DateFormatter('%Y')
    f1, ax = plt.subplots(sp1,sp2,figsize=(15, 12))
    f1.subplots_adjust(hspace=0.2, wspace=0.1,
                       left = 0.08 , right = 0.98 ,
                       bottom = 0.08 , top = 0.94)
    count = -1
    for site in sites:
        print(site)
        count = count+1
        i,j = np.unravel_index(count, ax.shape)
        ind_instr = inst_meta_df.loc[inst_meta_df['sitename'] == site].index
 
        for instr_nr in ind_instr:
            if np.isin(instr_nr, np.unique(compaction_df.index.get_level_values(0))):
                ini_depth = inst_meta_df.loc[inst_meta_df.index.values == instr_nr,
                                 'borehole_initial_length_m']
                # print(instr_nr,'\t', compaction_df.loc[instr_nr,var].index[0],'\t', compaction_df.loc[instr_nr,var].index[-1])
                ax[i,j].plot(compaction_df.loc[instr_nr,var], 
                             label='instr. '+str(instr_nr)+', ini. len.: %0.1f m'%abs(ini_depth))
                if var2:
                    ax[i,j].plot(compaction_df.loc[instr_nr,var2])
                
        if site in ['EKT', 'KAN-U', 'DYE-2']:
            ax[i,j].legend(loc='upper left',ncol=2, fontsize =12)
        else:
            ax[i,j].legend(loc='upper left',fontsize =12)
        # ax[i,j].legend(loc='lower left',ncol=1, fontsize =12)
        ax[i,j].grid(True)
        ax[i,j].set_title(site)
        if site == 'Crawford':
            ax[i,j].set_title('Crawford Point')
        ax[i,j].set_xlim([datetime.date(2012, 2, 13), datetime.date(2019, 10, 1)])    
        ax[i,j].xaxis.set_major_locator(years)
        ax[i,j].xaxis.set_major_formatter(years_fmt)
        ax[i,j].xaxis.set_minor_locator(months)
        ax[i,j].set_xlabel("")
        for k in range(2013,2020):
            ax[i,j].axvspan(*mdates.datestr2num([str(k)+'-06-01', str(k)+'-09-01']), color='orange', alpha=0.1)
            
        if count<len(sites)-2:
            ax[i,j].set_xticklabels("")
    f1.text(0.5, 0.02, 'Year', ha='center', size = 20)
    f1.text(0.02, 0.5, title, va='center', rotation='vertical', size = 20)
    # f1.savefig('figures/'+filename_out+'.png')
    return f1, ax

#% 
def export_gif(arr):
    import imageio
    import glob
    for file in glob.glob("figures/filtcomprate_*"):
        print(file)
        
    filenames =  np.array(glob.glob("figures/filtcomprate_*"))
    filenames = filenames[[0, 2, 3, 4, 1]]
    
     
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('filtcomprate.gif', images, duration=0.5)
#%%
def hampel(vals_orig, k=7, t0=3):
    '''
    vals: pandas series of values from which to remove outliers
    k: size of window (including the sample; 7 is equal to 3 on either side of value)
    '''
    #Make copy so original not edited
    vals=vals_orig.copy()    
    #Hampel Filter
    L= 1.4826
    rolling_median=vals.rolling(k).median()
    difference=np.abs(rolling_median-vals)
    median_abs_deviation=difference.rolling(k).median()
    threshold= t0 *L * median_abs_deviation
    outlier_idx=difference>threshold
    vals[outlier_idx]=np.nan
    return(vals)

#%% 
def interp_gap(data, gap_size):
    mask = data.copy()
    grp = ((mask.notnull() != mask.shift().notnull()).cumsum())
    grp['ones'] = 1
    for i in list('abcdefgh'):
        mask[i] = (grp.groupby(i)['ones'].transform('count') < gap_size) | data[i].notnull()
    return mask
    
#%%
def smooth(x,window_len=14,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')

    return y[int(window_len/2-1):-int(window_len/2)]
