from scipy import stats
import pandas
import matplotlib.pyplot as plt #plotting
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve #classifier calibration
from sklearn.metrics import brier_score_loss
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score


def guass_fit_dist(pd_series, hist_bins, ax_opt=None):
	if ax_opt != None:
		actual_count, actual_bins, ignored = ax_opt.hist(pd_series, bins = hist_bins)
	else:
		actual_count, actual_bins, ignored = plt.hist(pd_series, bins = hist_bins)

	param = stats.norm.fit(pd_series)
	gprob = np.diff(stats.norm.cdf(actual_bins, loc=param[0], scale=param[1]))
	if ax_opt != None:
		ax_opt.plot(actual_bins[1:],gprob*pd_series.size, 'r-')
		
	else:
		plt.plot(actual_bins[1:],gprob*pd_series.size, 'r-')
	#Perform a chi squared test to detemine goodness of fit of guassian on user's ratings
	nch, npval = stats.chisquare(actual_count, gprob*pd_series.size)
	
	return (actual_count, actual_bins, param[0], param[1], nch, npval)
	
	
	
def pro_class_calibration(model_og, x_train, y_train, x_test, y_test):
	
	#Find the P(y=1) according to the classifier
	prob_pos = model_og.predict_proba(x_test)[:, 1]
	#Find the brier score (a way of comparing probabilistic classifiers)
	orig_score = brier_score_loss(y_test, prob_pos)
	
	#Classifier with isotonic calibration
	model_og_isotonic = CalibratedClassifierCV(model_og, cv=2, method='isotonic')
	model_og_isotonic.fit(x_train, y_train)
	prob_pos_isotonic = model_og_isotonic.predict_proba(x_test)[:, 1]
	iso_score = brier_score_loss(y_test, prob_pos_isotonic)

	# Classifier with sigmoid calibration
	model_og_sigmoid = CalibratedClassifierCV(model_og, cv=2, method='sigmoid')
	model_og_sigmoid.fit(x_train, y_train)
	prob_pos_sigmoid = model_og_sigmoid.predict_proba(x_test)[:, 1]
	sig_score = brier_score_loss(y_test, prob_pos_sigmoid)


	#Find the calibration curve values
	fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)
	fraction_of_positives_iso, mean_predicted_value_iso = calibration_curve(y_test, prob_pos_isotonic, n_bins=10)
	fraction_of_positives_sig, mean_predicted_value_sig = calibration_curve(y_test, prob_pos_sigmoid, n_bins=10)


	#Plot the calibration curves
	plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s (%1.3f)" % ('No calibration', orig_score))
	plt.plot(mean_predicted_value_iso, fraction_of_positives_iso, "s-", label="%s (%1.3f)" % ('Isotonic Calibration', iso_score))
	plt.plot(mean_predicted_value_sig, fraction_of_positives_sig, "s-", label="%s (%1.3f)" % ('Sigmoid Calibration', sig_score))
	#Plot the ideal line
	plt.plot([0, 1], [0, 1], "k", label = 'Ideal')
	#Show legend, axes, etc formatting
	plt.legend(loc="best")
	plt.xlabel('Mean Predicted Value')
	plt.ylabel('Fraction of Positives')
	plt.show()
	return(model_og_isotonic, model_og_sigmoid)
	
def print_classifier_metrics(model, x_test, y_test):
	y_hat = model.predict(x_test)
	print('Accuracy of model: {:.2}'.format(model.score(x_test, y_test)))
	print('Recall Score: {:.2}'.format(recall_score(y_test,y_hat)))
	print('Precision Score: {:.2}'.format(precision_score(y_test,y_hat)))
	return()

def print_kfold_classifier_metrics(model, cv_builder, x, y):
	
	scoring_metrics = ['accuracy_score', 'precision_score', 'recall_score', 'f1_score']
	
	results={}
	for i in scoring_metrics:
		results[i] = cross_val_score(estimator=model, X=x, y=y, cv=cv_builder, scoring=make_scorer(eval(i)))
	
	print('Accuracy of model: mean= {:.2}, std. dev. = {:.2}'.format(np.mean(results['accuracy_score']), np.std(results['accuracy_score'])))
	print('Precision of model: mean= {:.2}, std. dev. = {:.2}'.format(np.mean(results['precision_score']), np.std(results['precision_score'])))
	print('Recall of model: mean= {:.2}, std. dev. = {:.2}'.format(np.mean(results['recall_score']), np.std(results['recall_score'])))
	
	return()