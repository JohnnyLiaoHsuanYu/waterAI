# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 14:08:12 2023

@author: 2310011
"""
import numpy as np


def data_to_matrix(lon, lat, interval, parameter):
    lat_formatrix = np.round(np.arange(lat.min(), lat.max()+interval, interval), 2)
    #np.arange(lat.min(), lat.max()+interval, interval)
    lon_formatrix = np.round(np.arange(lon.min(), lon.max()+interval, interval), 2)
    #np.arange(lon.min(), lon.max()+interval, interval)
    para_matrix = np.full([len(lat_formatrix), len(lon_formatrix)], np.nan)
    
    ## transformation ##
    for i in range(para_matrix.shape[0]):
      loc_ = np.where(lat==lat_formatrix[i])[0]
      if len(loc_)!=0:
        lonn = lon[loc_]
        pp = parameter[loc_]
        for j in range(len(lonn)):
          para_matrix[i, np.where(lon_formatrix==lonn[j])[0][0]] = pp[j]
    
    cx, cy = np.meshgrid(lon_formatrix, lat_formatrix)
    return lon_formatrix, lat_formatrix, para_matrix


def data_to_matrix_sea_level(lon, lat, interval, parameter):
    lat_formatrix = np.round(np.arange(lat.min(), lat.max()+(1/450)*interval, (1/450)*interval), 8)
    #np.arange(lat.min(), lat.max()+interval, interval)
    lon_formatrix = np.round(np.arange(lon.min(), lon.max()+(1/450)*interval, (1/450)*interval), 8)
    #np.arange(lon.min(), lon.max()+interval, interval)
    para_matrix = np.full([len(lat_formatrix), len(lon_formatrix)], np.nan)
    
    ## transformation ##
    for i in range(para_matrix.shape[0]):
      loc_ = np.where(np.round(lat, 4)==np.round(lat_formatrix[i], 4))[0]
      if len(loc_)!=0:
        lonn = lon[loc_]
        pp = parameter[loc_]
        for j in range(len(lonn)):
          para_matrix[i, np.where(abs(np.round(lon_formatrix, 4)-np.round(lonn[j], 4))<0.005)[0][0]] = pp[j]
    
    cx, cy = np.meshgrid(lon_formatrix, lat_formatrix)
    return lon_formatrix, lat_formatrix, para_matrix


def data_to_matrix_with_number(lon, lat, interval, parameter, index):
    lat_formatrix = np.round(np.arange(lat.min(), lat.max()+interval, interval), 2)
    #np.arange(lat.min(), lat.max()+interval, interval)
    lon_formatrix = np.round(np.arange(lon.min(), lon.max()+interval, interval), 2)
    #np.arange(lon.min(), lon.max()+interval, interval)
    para_matrix = np.full([len(lat_formatrix), len(lon_formatrix)], np.nan)
    para_index = np.full([len(lat_formatrix), len(lon_formatrix)], np.nan)
    
    ## transformation ##
    for i in range(para_matrix.shape[0]):
      loc_ = np.where(lat==lat_formatrix[i])[0]
      if len(loc_)!=0:
        lonn = lon[loc_]
        pp = parameter[loc_]
        iin = index[loc_]
        for j in range(len(lonn)):
          para_matrix[i, np.where(lon_formatrix==lonn[j])[0][0]] = pp[j]
          para_index[i, np.where(lon_formatrix==lonn[j])[0][0]] = iin[j]
    
    cx, cy = np.meshgrid(lon_formatrix, lat_formatrix)
    return lon_formatrix, lat_formatrix, para_matrix, para_index
    
    
