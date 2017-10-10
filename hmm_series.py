from hmmlearn.hmm import GaussianHMM
from hmmlearn import hmm
from matplotlib.finance import quotes_historical_yahoo_ochl
import numpy as np
import pandas as pd
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


class model(object):
	def __init__(self,X):
		self.lis= X
	def fit_hmm(self,components,iteration):
		'''
		fit a 1D series X, with #components, #iteration
		return an array of the predicted labels
		'''
		X = self.lis
		X = np.column_stack([X])
		remodel= hmm.GaussianHMM(n_components=components, n_iter=iteration,
			covariance_type="diag",init_params="cm",params="cmt")
		remodel.fit(X)
		print("fitting to HMM and decoding ...")
		print("Transition matrix")
		print(remodel.transmat_)
		print()
		print("Means and vars of each hidden state")
		for i in range(remodel.n_components):
			print("{0}th hidden state".format(i))
			print("mean = ", remodel.means_[i])
			print("var = ", np.diag(remodel.covars_[i]))
			print()
		return remodel.predict(X)
	def model_hmm(self,components,iteration):
		X = self.lis
		X = np.column_stack([X])
		remodel= hmm.GaussianHMM(n_components=components, n_iter=iteration,
			covariance_type="diag",init_params="cm",params="cmt")
		remodel.fit(X)
		return remodel

	def plot_hmm(self,components,iteration):
		'''
		this function plot the original series with the HMM labels embedded.
		-----
		Parameters
		 #components, #iteration
		'''
		modelx = self.model_hmm(components,iteration)
		#z = self.fit_hmm(components,iteration)
		fig, axs = plt.subplots(modelx.n_components, sharex=True, sharey=True,figsize=(12,8))
		colours = cm.rainbow(np.linspace(0, 1, modelx.n_components))
		X = np.column_stack([self.lis])
		hidden_states = modelx.predict(X)
		for i, (ax, colour) in enumerate(zip(axs, colours)):
			# Use fancy indexing to plot data in each state.
			mask = hidden_states == i
			#ax.plot_date(dates[mask], close_v[mask], ".-", c=colour)
			length= len(self.lis)
			ax.plot(np.arange(0,length,1)[mask], self.lis[mask], ".-", c=colour)
			ax.set_title("{0}th hidden state".format(i))
			# Format the ticks.
			#ax.xaxis.set_major_locator(YearLocator()) #(this two lines don't work. Oct 2017)
			#ax.xaxis.set_minor_locator(MonthLocator())
			ax.grid(True)
		plt.show()


