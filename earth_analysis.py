#!/usr/bin/python
# -*- coding: utf-8 -*-

import httplib
import pandas as pd
import math
import numpy as np
import pywt
import time
import Queue
import datetime
from bson.objectid import ObjectId
import pymongo as pm
import statsmodels.tsa.stattools as ts
from pymongo import MongoClient
import matplotlib.pyplot as plt

from pykalman import KalmanFilter
import seaborn as sns
from pandas.io.data import DataReader

from pandas.tools.plotting import scatter_matrix
from matplotlib import cm
import urllib, json
from mpl_toolkits.mplot3d import Axes3D

class Earth(object):
	def __init__(self,query):
		self.query = query
	def get_earth(self):
		'''
		get earthquake data from usgs.
		-----------
		default query: 
		significant_week
		4.5_week
		'''
		query = self.query
		url01="https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/"
		url01+=query
		url01+=".geojson"
		response = urllib.urlopen(url01)
		data = json.loads(response.read())
		return data

	def extract_earth(self):
		earth_dict = self.get_earth()
		dataDic={"time":pd.Series(),"longit":pd.Series(),"latit":pd.Series(),
				 "depth":pd.Series(),"mag":pd.Series(),
				 "sig":pd.Series(),"place":pd.Series()}
		n = len(earth_dict['features'])
		t0= datetime.datetime(1970, 1, 1, 0, 0, 0, 0)
		for i in np.arange(0,n,1):
			feature_dict= earth_dict['features'][i]['properties']
			#time = feature_dict['time']
			time= t0 + datetime.timedelta(0, 0, 1000*feature_dict['time'])
			coord = earth_dict['features'][i]['geometry']['coordinates']
			longit = coord[0]
			latit = coord[1]
			depth = coord[2]
			mag= feature_dict['mag']
			sig = feature_dict['sig']
			place = feature_dict['place']
			dataDic['time']=np.append(dataDic['time'],time)
			dataDic['longit']=np.append(dataDic['longit'],longit)
			dataDic['latit']=np.append(dataDic['latit'],latit)
			dataDic['depth']=np.append(dataDic['depth'],depth)
			dataDic['mag']=np.append(dataDic['mag'],mag)
			dataDic['sig']=np.append(dataDic['sig'],sig)
			dataDic['place']=np.append(dataDic['place'],place)
		df_earth=pd.DataFrame(dataDic,index=np.arange(0,n,1))
		#df_earth=pd.DataFrame(dataDic)
		return df_earth

	def scatter_feature(self,f1,f2):
		'''
		scatter plot for features
		default arguments:
		latit, longit, mag, depth
		'''
		df01=self.extract_earth()
		fig= plt.figure(figsize=(12,8))
		plt.scatter(df01[f1],df01[f2])
		plt.xlabel(f1)
		plt.ylabel(f2)
		plt.title('scatter_plot of '+f1+' and '+f2)
		plt.show()
	def scatter_matrix_feature(self):
		'''
		Default: 'longit','latit','depth','mag'
		'''
		df_lldm= self.extract_earth()[[f1,f2,f3,f4]].copy()
		scatter_matrix(df_lldm, alpha=1.0, figsize=(12, 12), diagonal='kde')

	def scatter_3d_feature(self,f1,f2,f3):
		'''
		Default: 'longit','latit','mag'
		'''
		fig = plt.figure(figsize=(12,10))
		ax = fig.add_subplot(111, projection='3d')
		df01 = self.extract_earth()
		x = df01[f1].as_matrix()
		y = df01[f2].as_matrix()
		z = df01[f3].as_matrix()
		#Axes3D.scatter(x, y, z, zdir='z', s=20, c=None, depthshade=True)
		ax.scatter(x, y, z, zdir='z',s=20, c=None, depthshade=True)
		ax.set_xlabel(f1)
		ax.set_ylabel(f2)
		ax.set_zlabel(f3)