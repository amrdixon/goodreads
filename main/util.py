from scipy import stats
import pandas
import matplotlib.pyplot as plt #plotting
import numpy as np

def guass_fit_dist(pd_series, hist_bins):
	actual_count, actual_bins, ignored = plt.hist(pd_series, bins = hist_bins)

	param = stats.norm.fit(pd_series)
	gprob = np.diff(stats.norm.cdf(actual_bins, loc=param[0], scale=param[1]))
	plt.plot(actual_bins[1:],gprob*pd_series.size, 'r-')
	#Perform a chi squared test to detemine goodness of fit of guassian on user's ratings
	nch, npval = stats.chisquare(actual_count, gprob*pd_series.size)
	
	return (actual_count, actual_bins, param[0], param[1], nch, npval)