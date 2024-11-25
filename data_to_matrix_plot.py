# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 14:16:47 2023

@author: 2310011
"""
import numpy as np
import json as js
import os
from netCDF4 import Dataset
import netCDF4
import gzip
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.markers import MarkerStyle
from mpl_toolkits.basemap import Basemap
import glob
import pandas as pd
import sys
import time
import requests
import io
import pdfplumber
import re
import datetime
import random

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import shapefile

import gettopocolor
import matplotlib.colors as colors
from matplotlib.colors import LightSource

import data_to_matrix

def read_shapefile(fname):
  shp = shapefile.Reader(fname)
  patches = []
  name = []

  for n in range(shp.numRecords):
    r = shp.record(n)
    s = shp.shape(n) 
    npart = len(s.parts)
    
    for i in range(npart):
      i0 = s.parts[i]
      i1 = s.parts[(i+1)%npart]-1
      if i1 == -1: i1 = None

      patches += [Polygon(s.points[i0:i1])]
      name  += [r[2]]
  
  name = np.array(name)
  patches = np.array(patches)

  return patches, name


county, countyname = read_shapefile('D:/2310011_Liao/refernce_old_code/mapdata202301070205/COUNTY_MOI_1090820.shp')
town, townname = read_shapefile('D:/2310011_Liao/refernce_old_code/administrative/twtownship/twtownship.shp')



TOPO = 'TW'
parameter__ = 'drought' # 'sea, rain', 'tem', 'drought'
ar = 'AR6'
fz = (5,5)

county_name = '南投縣'
datatype = 'json' # 'json', 'csv'

#path = 'D:/2310011_Liao/水資源/TCCIP/降雨/%s/'%(county_name)
path = 'D:/2310011_Liao/水資源/done_model/prediction_result/'
os.makedirs(path + 'plot/', exist_ok=True)

ddd = 20170203
# fff = glob.glob(path + '*__%s*.%s'%(ddd, datatype))
fff = glob.glob(path + 'waterAI_XGBRegressor_new_20201118.%s'%(datatype))


# cwb_my = cm.jet
# bounds_my= np.arange(0,6,1)
# norm_my = mpl.colors.BoundaryNorm(bounds_my, cwb_my.N)

colors_my = ["#009CFF", "#44C572", "#FFFF00", "#FF9000", "#EF2F2F"]
cmapmine = mpl.colors.LinearSegmentedColormap.from_list('mycmap', list(zip(np.linspace(0, 1, len(colors_my)), colors_my[::1])))
bounds_my = np.arange(0, 6, 1)
cwb_my = mpl.colors.ListedColormap(cmapmine(np.linspace(0, 1, bounds_my.size-1)))
cwb_my.set_over(cmapmine(0.99))
norm_my = mpl.colors.BoundaryNorm(bounds_my, cwb_my.N)


if TOPO == 'TW':
    lon_bb = [119.5, 122.5]#[120.6, 121.4]#[121.27, 122.02]#[119.99, 122.012]
    lat_bb = [21.5, 25.75]#[23.4, 24.3]#[24.66, 25.31]#[21.870001, 25.3258]
    diff = .5
    parallels = np.round(np.arange(lat_bb[0],lat_bb[1]+diff, diff), 2)
    meridians = np.round(np.arange(lon_bb[0],lon_bb[1]+diff, diff), 2) 
    def drawBasemap(ax, coast_color='k'):
        m = Basemap(llcrnrlon=lon_bb[0], llcrnrlat=lat_bb[0],\
                    urcrnrlon=lon_bb[1], urcrnrlat=lat_bb[1],\
                    lon_0=np.mean(lon_bb), lat_0=np.mean(lat_bb),
                    projection='cyl', resolution='l', ax=ax)
        m.drawparallels(parallels, labels=[1,0,0,0], fontsize=5, linewidth=0.3)
        m.drawmeridians(meridians, labels=[0,0,0,1], fontsize=5, linewidth=0.3) 
        #m.drawcoastlines(linewidth=1, zorder=0)
        return m

    # df_scatter = pd.read_csv('D:/2310011_Liao/水資源/報告會議/report_plot/sample.csv') 
    # df_lonlat = df_scatter.iloc[:, 1:3].sample(n=150)
    # scatter_lat = df_lonlat['lat']
    # scatter_lon = df_lonlat['lng']

    # df_lonlat_555 = df_lonlat.sample(n=5)
    # scatter_lat_555 = df_lonlat_555['lat']
    # scatter_lon_555 = df_lonlat_555['lng']
    # scatter_lat = [24.12694427, 25.04687178, 23.90877952, 23.9044863, 24.55229494, 24.1532264, 24.13983169, 24.16579026, 24.1714584, 24.14893358, 24.13091439, 24.15707046]
    # scatter_lon = [120.7154686, 121.2926956, 120.6867396, 120.6880083, 120.8160622, 120.6855474, 120.6826231, 120.6950719, 120.6766075, 120.6353976, 120.6845123, 120.6504929]
    
    # TW_topo_path = '/home/tcfdadm/dataPool/TOPO/TW/USGS_TW_topo.npz'
    # topo_ = np.load(TW_topo_path)['topo']
    # topo_xx = np.load(TW_topo_path)['xx']
    # topo_yy = np.load(TW_topo_path)['yy']

elif TOPO == 'SHai':
    lon_bb = [119.5, 122]#[118, 123]#[119.5, 122]#
    lat_bb = [30.6, 31.7]#[21, 27]#[30.5, 31.7]#
    diff = .5
    parallels = np.round(np.arange(lat_bb[0],lat_bb[1]+diff, diff), 2)
    meridians = np.round(np.arange(lon_bb[0],lon_bb[1]+diff, diff), 2) 
    def drawBasemap(ax, coast_color='k'):
        m = Basemap(llcrnrlon=lon_bb[0], llcrnrlat=lat_bb[0],\
                    urcrnrlon=lon_bb[1], urcrnrlat=lat_bb[1],\
                    lon_0=np.mean(lon_bb), lat_0=np.mean(lat_bb),
                    projection='cyl', resolution='l', ax=ax)
        m.drawparallels(parallels, labels=[1,0,0,0], fontsize=5, linewidth=0.5)
        m.drawmeridians(meridians, labels=[0,0,0,1], fontsize=5, linewidth=0.5) 
        m.drawcoastlines(linewidth=1, zorder=1)
        return m
        
    scatter_lat = [31.09513142688024, 31.466835368214955]
    scatter_lon = [121.15338570277622, 120.93901367367123]
    
    # SHai_topo_path = '/home/tcfdadm/dataPool/TOPO/SHai/shanhai_topo.npz'
    # topo_ = np.load(SHai_topo_path)['topo']
    # topo_xx = np.load(SHai_topo_path)['xx']
    # topo_yy = np.load(SHai_topo_path)['yy']

elif TOPO == 'CAN':
    lon_bb = [-124, -121]
    lat_bb = [36, 39]
    diff = 1
    parallels = np.round(np.arange(lat_bb[0],lat_bb[1]+diff, diff), 2)
    meridians = np.round(np.arange(lon_bb[0],lon_bb[1]+diff, diff), 2) 
    def drawBasemap(ax, coast_color='k'):
        m = Basemap(llcrnrlon=lon_bb[0], llcrnrlat=lat_bb[0],\
                    urcrnrlon=lon_bb[1], urcrnrlat=lat_bb[1],\
                    lon_0=np.mean(lon_bb), lat_0=np.mean(lat_bb),
                    projection='cyl', resolution='l', ax=ax)
        m.drawparallels(parallels, labels=[1,0,0,0], fontsize=5, linewidth=0.1)
        m.drawmeridians(meridians, labels=[0,0,0,1], fontsize=5, linewidth=0.1) 
        m.drawcoastlines(linewidth=1, zorder=1)
        return m

    CAN_topo_path = '/home/tcfdadm/dataPool/TOPO/US_CAN/upscale_CAN_topo_640.0.npz'
    topo_ = np.load(CAN_topo_path)['topo']
    topo_xx = np.load(CAN_topo_path)['xx']
    topo_yy = np.load(CAN_topo_path)['yy']

    # CAN_topo_path_org = '/home/tcfdadm/dataPool/TOPO/US_CAN/CAN_topo.npz'
    # topo_org = np.load(CAN_topo_path_org)['topo']
    # topo_xx_org = np.load(CAN_topo_path_org)['xx']
    # topo_yy_org = np.load(CAN_topo_path_org)['yy']

elif TOPO == 'CAS':
    lon_bb = [-119, -117]
    lat_bb = [33, 35]
    diff = 1
    parallels = np.round(np.arange(lat_bb[0],lat_bb[1]+diff, diff), 2)
    meridians = np.round(np.arange(lon_bb[0],lon_bb[1]+diff, diff), 2) 
    def drawBasemap(ax, coast_color='k'):
        m = Basemap(llcrnrlon=lon_bb[0], llcrnrlat=lat_bb[0],\
                    urcrnrlon=lon_bb[1], urcrnrlat=lat_bb[1],\
                    lon_0=np.mean(lon_bb), lat_0=np.mean(lat_bb),
                    projection='cyl', resolution='l', ax=ax)
        m.drawparallels(parallels, labels=[1,0,0,0], fontsize=5, linewidth=0.1)
        m.drawmeridians(meridians, labels=[0,0,0,1], fontsize=5, linewidth=0.1) 
        m.drawcoastlines(linewidth=1, zorder=1)
        return m




# =================== precipitation forcast ===================

url = 'https://www.cwa.gov.tw/Data/fcst_pdf/FW14.pdf'
response = requests.get(url)

if response.status_code == 200:
    # tranform to the text 
    pdf_file = io.BytesIO(response.content)
    date_report = []
    precip_data = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            text = page.extract_text()
            precip_data.append(tables)
            date_report.append(text)
else:
    print(f"Error: Unable to fetch the PDF. Status code: {response.status_code}")

need_precip_info = []
for di, d in enumerate(precip_data):
    for subi, sublist in enumerate(d):
        for inni, inner_list in enumerate(sublist):
            if '雨\n量\n北部\n中部\n南部\n東部' in inner_list:
                for rr in precip_data[di][subi][1:-1]:
                    precip_forcast = [float(num) for num in re.findall(r'\d+\.\d+', rr[-1])]
                    # print(rr[0], precip_forcast, np.nanmean(precip_forcast))
                    need_precip_info.append({
                        'region_space': rr[0],
                        'forcast_range': precip_forcast,
                        'forcast_mean': np.nanmean(precip_forcast)
                    })

need_time_info = []
for datei, datej in enumerate(np.array(date_report)[0].split()):
    if '發布日期' in datej:
        # print(datej)
        need_time_info.append(datej)
    if '有效期間' in datej:
        # print(datej)
        need_time_info.append(datej)


match_year = [int(my)+1911 for my in re.findall(r'(\d{1,3})年', need_time_info[1])]
match_date = re.findall(r'(\d{1,2})月(\d{1,2})日', need_time_info[1])
if match_date:
    d_111 = datetime.datetime(match_year[0], int(match_date[0][0]), int(match_date[0][1]))
    d_222 = datetime.datetime(match_year[0], int(match_date[1][0]), int(match_date[1][1]))
    middle_date = d_111 + (d_222 - d_111)/2

input_futuretime = [{'date': int(middle_date.strftime("%Y%m%d"))}]


# =========================================================================

for filename in fff:
    if datatype == 'json':
        # FOR tem & precip_risk 
        if parameter__ == 'tem' or parameter__ =='rain' or parameter__ =='drought':
            # ============= for json ==============
            with open(filename)as f: #, encoding='utf-8'
                df = js.load(f)
            
            lon____ = []
            lat____ = []
            par = []
            for dfdf in df:
                # lon____.append(dfdf['wgs84_lon'])
                # lat____.append(dfdf['wgs84_lat'])
                time___ = dfdf['date']
                lon____.append(dfdf['lon'])
                lat____.append(dfdf['lat'])
                
                if parameter__ == 'tem':
                    par.append(dfdf['average_tas_var'])
                elif parameter__ == 'rain':
                    par.append(dfdf['level'])
                elif parameter__ == 'drought':
                    par.append(dfdf['level'])
            
            lon____ = np.array(lon____)
            lat____ = np.array(lat____)
            par = np.array(par) 
            
            lon_metrix, lat_metrix, parameter = data_to_matrix.data_to_matrix(lon____, lat____, 0.01, par)
            parameter = np.where(parameter<0, np.nan, parameter)
            
            
            
            fig,ax = plt.subplots(figsize=fz)
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
            mapp = drawBasemap(ax)
            ax.set_facecolor("darkblue")
            ppp = plt.pcolormesh(lon_metrix, lat_metrix, parameter, cmap=cwb_my, norm=norm_my, zorder=0)
            CRB_ = plt.colorbar(ppp, shrink=0.9, ticks=[0.5, 1.5, 2.5, 3.5, 4.5])#, orientation='horizontal') #, ticks=np.arange(4, 7, 0.5)
            CRB_.ax.set_yticklabels(["水情正常", "水情稍緊", "一階限水", "二階限水", "三階限水"])
            county_draw = PatchCollection(county, fc='None', ec='white', lw=0.3)
            county_draw_spec_1 = PatchCollection(county[countyname=='宜蘭縣'], fc='gainsboro', ec='white', lw=0.3)
            county_draw_spec_2 = PatchCollection(county[countyname=='花蓮縣'], fc='gainsboro', ec='white', lw=0.3)
            county_draw_spec_3 = PatchCollection(county[countyname=='臺東縣'], fc='gainsboro', ec='white', lw=0.3)
            county_draw_spec_4 = PatchCollection(county[countyname=='澎湖縣'], fc='gainsboro', ec='white', lw=0.3)
            ax.add_collection(county_draw)
            ax.add_collection(county_draw_spec_1)
            ax.add_collection(county_draw_spec_2)
            ax.add_collection(county_draw_spec_3)
            ax.add_collection(county_draw_spec_4)
            plt.title('限水預警情況', loc='left', fontsize=16)
            #plt.title('解析度: 1 公里\n%s_%s'%(int(d_111.strftime("%Y%m%d")), int(d_222.strftime("%Y%m%d"))), loc='right', fontsize=6.5)
            plt.title('解析度: 1 公里\n時間: %s'%(ddd), loc='right', fontsize=6.5)
            plt.savefig(path + 'plot/' + filename.split('/')[-1].split('\\')[-1][:-5] + '.png', dpi=1000)
            # plt.show()
            plt.close('all')
    

        # FOR sea level
        elif parameter__ == 'sea':
            with open(filename)as f:
                df = js.load(f)
            
            lon____ = []
            lat____ = []
            par = []
            for llonat in df:
                lon____.append(llonat['wgs84_lon'])
                lat____.append(llonat['wgs84_lat'])
                par.append(llonat['flood_risk'])
            
            lon____ = np.array(lon____)
            lat____ = np.array(lat____)
            par = np.array(par)

            min_, max_ = 1, 6
            colors_my = ['b', 'deepskyblue', 'magenta', 'mediumvioletred', 'maroon']
            cmapmine = mpl.colors.LinearSegmentedColormap.from_list('mycmap', list(zip(np.linspace(0, 1, len(colors_my)), colors_my[::1])))
            bounds_my = np.arange(min_, max_+1, 1)
            # cwb_my = mpl.colors.ListedColormap(cmapmine(np.linspace(0, 0.9, bounds_my.size-1)))
            # cwb_my.set_over(cmapmine(0.99))
            cwb_my = mpl.colors.ListedColormap(colors_my)  # 使用原始顏色清單
            cwb_my.set_over(colors_my[-1])
            norm_my = mpl.colors.BoundaryNorm(bounds_my, cwb_my.N)
            


            fig,ax = plt.subplots(figsize=fz)
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
            mapp = drawBasemap(ax)
            #####===== topography =====#####
            ax.set_facecolor("darkblue")
            cxx, cyy = topo_xx_org, topo_yy_org
            ls = LightSource(azdeg=-45, altdeg=45)   
            hs = ls.hillshade(topo_org, vert_exag=1)
            cmapg = colors.LinearSegmentedColormap.from_list('', [(0,0,0,0.3), (0,0,0,0.05), (0,0,0,0)], N=512)
            c_ = plt.contourf(cxx, cyy, topo_org/1000, cmap=gettopocolor.TOPO(), levels=np.arange(0, 2.53, 0.02), extend='max', zorder=0)
            plt.contourf(cxx, cyy, hs, cmap=cmapg, levels=np.arange(0.01, 0.99, 0.05), antialiased=True, extend='both')
            cb = plt.colorbar(c_, label='Elevation (m)', ticks=np.arange(0, 2.6, 0.5), shrink=0.7)#, aspect=50

            ppp = mapp.scatter(lon____, lat____, s=0.2, c=par, cmap=cwb_my, vmin=min_, vmax=max_, marker=MarkerStyle(',', fillstyle='full'), linewidth=0.1) #
            cbar = plt.colorbar(ppp, label='Risk', ticks=[1.5, 2.5, 3.5, 4.5, 5.5], shrink=0.7) #
            cbar.ax.set_yticklabels(['1', '2', '3', '4', '5'])
            plt.title('Sea Level Rise %s %s'%(filename.split('/')[-1].split('_')[2], filename.split('/')[-1].split('_')[3]), loc='left', fontsize=10)
            plt.title('Resolution:\n[640m]', loc='right', fontsize=5) #filename.split('/')[-1].split('_')[-2]
            plt.savefig(path + 'plot/' + filename.split('/')[-1][:-5] + '.png', dpi=1000)
            plt.close('all')
            #sys.exit()
    
    
    elif datatype == 'csv':
        # ============= for CSV ==============
        with open(filename)as f:
            df = pd.read_csv(f)

        lon____ = np.array(df['LON'])
        lat____ = np.array(df['LAT'])
        par = np.array(df.iloc[:, 70])

        expand_ = 1
        if TOPO == 'TW':
            lon_bb = [lon____.min()-expand_, lon____.max()+expand_]
            lat_bb = [lat____.min()-expand_, lat____.max()+expand_]
            diff = .1
            parallels = np.round(np.arange(lat_bb[0],lat_bb[1]+diff, diff), 2)
            meridians = np.round(np.arange(lon_bb[0],lon_bb[1]+diff, diff), 2) 
            def drawBasemap(ax, coast_color='k'):
                m = Basemap(llcrnrlon=lon_bb[0], llcrnrlat=lat_bb[0],\
                            urcrnrlon=lon_bb[1], urcrnrlat=lat_bb[1],\
                            lon_0=np.mean(lon_bb), lat_0=np.mean(lat_bb),
                            projection='cyl', resolution='l', ax=ax)
                m.drawparallels(parallels, labels=[1,1,0,0], fontsize=5, linewidth=0.3)
                m.drawmeridians(meridians, labels=[0,0,1,1], fontsize=5, linewidth=0.3) 
                #m.drawcoastlines(linewidth=1, zorder=0)
                return m

            df_scatter = pd.read_csv('D:/2310011_Liao/水資源/報告/report_plot/sample.csv') 
            df_lonlat = df_scatter.iloc[:, 1:3].sample(n=150)
            scatter_lat = df_lonlat['lat']
            scatter_lon = df_lonlat['lng']

            df_lonlat_555 = df_lonlat.sample(n=5)
            scatter_lat_555 = df_lonlat_555['lat']
            scatter_lon_555 = df_lonlat_555['lng']
            # scatter_lat = [24.12694427, 25.04687178, 23.90877952, 23.9044863, 24.55229494, 24.1532264, 24.13983169, 24.16579026, 24.1714584, 24.14893358, 24.13091439, 24.15707046]
            # scatter_lon = [120.7154686, 121.2926956, 120.6867396, 120.6880083, 120.8160622, 120.6855474, 120.6826231, 120.6950719, 120.6766075, 120.6353976, 120.6845123, 120.6504929]
            
            # TW_topo_path = '/home/tcfdadm/dataPool/TOPO/TW/USGS_TW_topo.npz'
            # topo_ = np.load(TW_topo_path)['topo']
            # topo_xx = np.load(TW_topo_path)['xx']
            # topo_yy = np.load(TW_topo_path)['yy']

        elif TOPO == 'SHai':
            lon_bb = [119.5, 122]#[118, 123]#[119.5, 122]#
            lat_bb = [30.6, 31.7]#[21, 27]#[30.5, 31.7]#
            diff = .5
            parallels = np.round(np.arange(lat_bb[0],lat_bb[1]+diff, diff), 2)
            meridians = np.round(np.arange(lon_bb[0],lon_bb[1]+diff, diff), 2) 
            def drawBasemap(ax, coast_color='k'):
                m = Basemap(llcrnrlon=lon_bb[0], llcrnrlat=lat_bb[0],\
                            urcrnrlon=lon_bb[1], urcrnrlat=lat_bb[1],\
                            lon_0=np.mean(lon_bb), lat_0=np.mean(lat_bb),
                            projection='cyl', resolution='l', ax=ax)
                m.drawparallels(parallels, labels=[1,0,0,0], fontsize=5, linewidth=0.5)
                m.drawmeridians(meridians, labels=[0,0,0,1], fontsize=5, linewidth=0.5) 
                m.drawcoastlines(linewidth=1, zorder=1)
                return m
                
            scatter_lat = [31.09513142688024, 31.466835368214955]
            scatter_lon = [121.15338570277622, 120.93901367367123]
            
            # SHai_topo_path = '/home/tcfdadm/dataPool/TOPO/SHai/shanhai_topo.npz'
            # topo_ = np.load(SHai_topo_path)['topo']
            # topo_xx = np.load(SHai_topo_path)['xx']
            # topo_yy = np.load(SHai_topo_path)['yy']

        elif TOPO == 'CAN':
            lon_bb = [-124, -121]
            lat_bb = [36, 39]
            diff = 1
            parallels = np.round(np.arange(lat_bb[0],lat_bb[1]+diff, diff), 2)
            meridians = np.round(np.arange(lon_bb[0],lon_bb[1]+diff, diff), 2) 
            def drawBasemap(ax, coast_color='k'):
                m = Basemap(llcrnrlon=lon_bb[0], llcrnrlat=lat_bb[0],\
                            urcrnrlon=lon_bb[1], urcrnrlat=lat_bb[1],\
                            lon_0=np.mean(lon_bb), lat_0=np.mean(lat_bb),
                            projection='cyl', resolution='l', ax=ax)
                m.drawparallels(parallels, labels=[1,0,0,0], fontsize=5, linewidth=0.1)
                m.drawmeridians(meridians, labels=[0,0,0,1], fontsize=5, linewidth=0.1) 
                m.drawcoastlines(linewidth=1, zorder=1)
                return m

            CAN_topo_path = '/home/tcfdadm/dataPool/TOPO/US_CAN/upscale_CAN_topo_640.0.npz'
            topo_ = np.load(CAN_topo_path)['topo']
            topo_xx = np.load(CAN_topo_path)['xx']
            topo_yy = np.load(CAN_topo_path)['yy']

            # CAN_topo_path_org = '/home/tcfdadm/dataPool/TOPO/US_CAN/CAN_topo.npz'
            # topo_org = np.load(CAN_topo_path_org)['topo']
            # topo_xx_org = np.load(CAN_topo_path_org)['xx']
            # topo_yy_org = np.load(CAN_topo_path_org)['yy']

        elif TOPO == 'CAS':
            lon_bb = [-119, -117]
            lat_bb = [33, 35]
            diff = 1
            parallels = np.round(np.arange(lat_bb[0],lat_bb[1]+diff, diff), 2)
            meridians = np.round(np.arange(lon_bb[0],lon_bb[1]+diff, diff), 2) 
            def drawBasemap(ax, coast_color='k'):
                m = Basemap(llcrnrlon=lon_bb[0], llcrnrlat=lat_bb[0],\
                            urcrnrlon=lon_bb[1], urcrnrlat=lat_bb[1],\
                            lon_0=np.mean(lon_bb), lat_0=np.mean(lat_bb),
                            projection='cyl', resolution='l', ax=ax)
                m.drawparallels(parallels, labels=[1,0,0,0], fontsize=5, linewidth=0.1)
                m.drawmeridians(meridians, labels=[0,0,0,1], fontsize=5, linewidth=0.1) 
                m.drawcoastlines(linewidth=1, zorder=1)
                return m

        
        lon_metrix, lat_metrix, parameter = data_to_matrix.data_to_matrix(lon____, lat____, 0.01, par)
        np.savez(path + 'matrix_data_%s.npz'%(county_name.replace('/', '')), lon=lon_metrix, lat=lat_metrix, data=parameter)

        another = pd.read_csv(r'D:\2310011_Liao\水資源\TCCIP\降雨\基隆市\觀測_日資料_基隆市_降雨量_1960.csv')
        an_lon = another['LON']
        an_lat = another['LAT']
        
        cwb1 = cm.viridis
        bounds1 = np.arange(10, 210, 50) #
        norm1 = mpl.colors.BoundaryNorm(bounds1, cwb1.N)

        fig,ax = plt.subplots(figsize=fz)
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        mapp = drawBasemap(ax)
        ppp = plt.pcolormesh(lon_metrix, lat_metrix, parameter, cmap=cwb1, norm=norm1)
        # plt.scatter(an_lon, an_lat, s=10, c='red', marker=MarkerStyle(',', fillstyle='full'), alpha=0.4)#, linewidth=0.5, edgecolor='black')
        county_draw = PatchCollection(county, fc='None', ec='k', lw=0.6)
        # town_draw = PatchCollection(town, fc='None', ec='k', lw=0.2)
        # county_draw_spec_1 = PatchCollection(county[countyname=='南投縣'], fc='None', ec='r', lw=1)
        # county_draw_spec_2 = PatchCollection(county[countyname=='新北市'], fc='None', ec='r', lw=1)
        # county_draw_spec_3 = PatchCollection(county[countyname=='基隆市'], fc='None', ec='r', lw=1)
        ax.add_collection(county_draw)
        # ax.add_collection(town_draw)
        ax.add_collection(county_draw_spec_1)
        # ax.add_collection(county_draw_spec_2)
        # ax.add_collection(county_draw_spec_3)
        CBB = plt.colorbar(ppp, shrink=0.9, ticks=np.arange(bounds1.min(), bounds1.max()+50, 50), extend='both', orientation='vertical', pad=0.1)
        CBB.set_label('Precipitation [mm]', labelpad=5)
        ttext = np.arange(1, len(lon____)+1)
        # for i in range(len(lon____)):
        #     plt.scatter(lon____[i], lat____[i], c=ttext[i], s=0.2, cmap=cwb1, vmin=ttext.min(), vmax=ttext.max(), marker=MarkerStyle(',', fillstyle='full'))
        #     plt.text(lon____[i], lat____[i], str(ttext[i]), fontsize=1)
        
        plt.title('%s'%(county_name.replace('/', '')), pad=20)
        plt.savefig(path + 'plot/try_%s.png'%(county_name.replace('/', '')), dpi=1000)
        # plt.show()
        plt.close('all')