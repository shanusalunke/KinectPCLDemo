import scipy
import numpy as np
import pymc
from matplotlib import pyplot
from matplotlib.mlab import normpdf

def get_data():
	box1_1 = np.array([   0.     ,    0.     ,    0.     ,    0.88767,   12.527  ,
	      45.2769 ,   37.8034 ,    3.50499,    0.     ,    0.     ,
	       0.     ,    0.     ,    0.     ,    1.53539,    8.10268,
	      15.3949 ,   34.6364 ,   26.6042 ,    8.16714,    4.82106,
	       0.712  ,    0.02619,    0.12343,    3.37853,   22.3174 ,
	      32.854  ,   12.6823 ,    1.9913 ,    3.03218,   15.8104 ,
	       6.46365,    1.29959,    0.0473 ])
	box1_2 = np.array([   0.     ,    0.     ,    0.     ,    0.     ,    1.8142 ,
	      75.8037 ,   22.3821 ,    0.     ,    0.     ,    0.     ,
	       0.     ,    0.     ,    0.     ,    6.20211,   41.216  ,
	      20.0739 ,   19.2503 ,   13.2578 ,    0.     ,    0.     ,
	       0.     ,    0.     ,    0.     ,    3.83999,    2.75477,
	      28.1541 ,   25.8983 ,   14.0535 ,   19.9512 ,    2.91603,
	       2.36043,    0.06773,    0.00383])
	box1_3 = np.array([   0.     ,    0.     ,    0.     ,    0.     ,    0.     ,
	      21.3456 ,   66.5776 ,   12.0486 ,    0.02813,    0.     ,
	       0.     ,    0.     ,    0.1018 ,    0.93984,   12.4819 ,
	      17.892  ,   34.4132 ,   26.9439 ,    6.43302,    0.75821,
	       0.0362 ,    0.     ,    0.01674,    1.94928,   39.5243 ,
	      37.8137 ,   17.7929 ,    1.81125,    0.66189,    0.2237 ,
	       0.2063 ,    0.     ,    0.     ])
	box1_4 = np.array([   0.     ,    0.     ,    0.     ,    0.     ,    0.02065,
	      36.3163 ,   49.323  ,   14.1136 ,    0.22654,    0.     ,
	       0.     ,    0.     ,    0.5695 ,    3.79238,   14.3359 ,
	      21.7236 ,   29.701  ,   18.3981 ,    7.63832,    3.35859,
	       0.48262,    0.     ,    0.02328,    8.32146,   35.4589 ,
	      26.4164 ,   18.9233 ,    6.30369,    4.14119,    0.3149 ,
	       0.09684,    0.     ,    0.     ])
	chair1_1 = np.array([   0.     ,    0.     ,    0.     ,    0.     ,    0.     ,
	     100.     ,    0.     ,    0.     ,    0.     ,    0.     ,
	       0.     ,    0.     ,    0.     ,    0.     ,    0.     ,
	       9.69301,   83.4912 ,    6.81577,    0.     ,    0.     ,
	       0.     ,    0.     ,    0.14339,    0.14116,    0.1238 ,
	       1.63297,    6.80354,   57.4656 ,   32.2591 ,    1.20897,
	       0.15163,    0.     ,    0.06985])
	chair1_2 = np.array([   0.     ,    0.     ,    0.     ,    0.     ,    0.     ,
	      95.197  ,    4.80297,    0.     ,    0.     ,    0.     ,
	       0.     ,    0.     ,    0.     ,    0.     ,    0.17281,
	      28.8277 ,   58.4992 ,   12.2869 ,    0.21337,    0.     ,
	       0.     ,    0.     ,    0.     ,    0.04904,    0.00448,
	       1.98569,   41.2433 ,   30.1197 ,   25.2127 ,    0.16221,
	       1.22298,    0.     ,    0.     ])
	chair1_3 = np.array([   0.     ,    0.     ,    0.     ,    0.     ,    0.     ,
	      32.666  ,   67.3341 ,    0.     ,    0.     ,    0.     ,
	       0.     ,    0.     ,    0.     ,    0.     ,    0.00597,
	       4.86658,   81.3563 ,   13.6884 ,    0.08274,    0.     ,
	       0.     ,    0.     ,    0.0191 ,    0.0576 ,    1.01824,
	      45.8162 ,   49.673  ,    3.41579,    0.     ,    0.     ,
	       0.     ,    0.     ,    0.     ])
	chair1_4 = np.array([   0.     ,    0.     ,    0.     ,    0.     ,    0.     ,
	      99.9983 ,    0.00166,    0.     ,    0.     ,    0.     ,
	       0.     ,    0.     ,    0.     ,    0.     ,    0.     ,
	       0.     ,   99.7715 ,    0.22847,    0.     ,    0.     ,
	       0.     ,    0.     ,    0.98023,    0.     ,    0.0018 ,
	       0.00648,    2.89593,   95.5836 ,    0.52479,    0.00437,
	       0.00287,    0.     ,    0.     ])
	chair1_5 = np.array([   0.     ,    0.     ,    0.     ,    0.     ,    0.     ,
	      49.0464 ,   50.9536 ,    0.     ,    0.     ,    0.     ,
	       0.     ,    0.     ,    0.     ,    0.     ,    0.11764,
	      18.3074 ,   58.843  ,   22.6155 ,    0.11647,    0.     ,
	       0.     ,    0.     ,    0.     ,    0.     ,    0.35326,
	      23.0919 ,   68.9529 ,    7.60193,    0.     ,    0.     ,
	       0.     ,    0.     ,    0.     ])
	chair1_6 = np.array([   0.     ,    0.     ,    0.     ,    0.     ,    0.     ,
	      33.1548 ,   66.8452 ,    0.     ,    0.     ,    0.     ,
	       0.     ,    0.     ,    0.     ,    0.     ,    0.00145,
	       4.62795,   92.2517 ,    3.11894,    0.     ,    0.     ,
	       0.     ,    0.     ,    0.     ,    0.01951,    0.03054,
	      25.4797 ,   70.3472 ,    4.1231 ,    0.     ,    0.     ,
	       0.     ,    0.     ,    0.     ])
	chair2_1 = np.array([   0.     ,    0.     ,    0.     ,    0.     ,    0.     ,
	     100.     ,    0.     ,    0.     ,    0.     ,    0.     ,
	       0.     ,    0.     ,    0.     ,    0.     ,    0.     ,
	       0.     ,  100.     ,    0.     ,    0.     ,    0.     ,
	       0.     ,    0.     ,    0.00733,    0.     ,    0.42527,
	       0.71798,   19.9753 ,   78.3903 ,    0.10371,    0.     ,
	       0.01122,    0.01976,    0.34919])
	chair2_2 = np.array([   0.     ,    0.     ,    0.     ,    0.     ,    0.     ,
	      64.1336 ,   35.8664 ,    0.     ,    0.     ,    0.     ,
	       0.     ,    0.     ,    0.     ,    0.     ,    1.27338,
	      21.9851 ,   59.5296 ,   17.2119 ,    0.     ,    0.     ,
	       0.     ,    0.     ,    0.     ,    0.08301,    2.49736,
	      24.4921 ,   63.7271 ,    9.13432,    0.06609,    0.     ,
	       0.     ,    0.     ,    0.     ])
	chair2_3 = np.array([   0.     ,    0.     ,    0.     ,    0.     ,    0.     ,
	      38.5065 ,   60.534  ,    0.95953,    0.     ,    0.     ,
	       0.     ,    0.     ,    0.     ,    0.01843,    0.85226,
	      18.3474 ,   55.8883 ,   23.1736 ,    1.72003,    0.     ,
	       0.     ,    0.     ,    0.04002,    2.61294,   13.6776 ,
	      41.9586 ,   38.6228 ,    3.0109 ,    0.0612 ,    0.     ,
	       0.01591,    0.     ,    0.     ])
	chair2_4 = np.array([   0.     ,    0.     ,    0.     ,    0.     ,    0.     ,
	     100.     ,    0.     ,    0.     ,    0.     ,    0.     ,
	       0.     ,    0.     ,    0.     ,    0.     ,    0.     ,
	       0.     ,   99.9034 ,    0.0966 ,    0.     ,    0.     ,
	       0.     ,    0.     ,    0.     ,    3.03857,    0.55783,
	       6.78539,   36.6    ,   41.4019 ,    8.70449,    0.26626,
	       2.64558,    0.     ,    0.     ])
	table1_1 = np.array([   0.     ,    0.     ,    0.     ,    0.     ,    0.     ,
	     100.     ,    0.     ,    0.     ,    0.     ,    0.     ,
	       0.     ,    0.     ,    0.     ,    0.     ,    0.     ,
	       0.07686,   99.9231 ,    0.     ,    0.     ,    0.     ,
	       0.     ,    0.     ,    0.     ,    0.01051,    0.39117,
	       0.51669,    3.4938 ,   95.5472 ,    0.03468,    0.00597,
	       0.     ,    0.     ,    0.     ])
	table1_2 = np.array([   0.     ,    0.     ,    0.     ,    0.     ,    0.     ,
	      79.8292 ,   20.1708 ,    0.     ,    0.     ,    0.     ,
	       0.     ,    0.     ,    0.     ,    0.     ,    0.75287,
	      23.8224 ,   49.5054 ,   24.8429 ,    1.07648,    0.     ,
	       0.     ,    0.     ,    0.     ,    0.90549,    3.85613,
	      24.7709 ,   51.1268 ,   19.1436 ,    0.13294,    0.02157,
	       0.     ,    0.0426 ,    0.     ])
	table1_3 = np.array([   0.     ,    0.     ,    0.     ,    0.     ,    0.     ,
	      78.7734 ,   21.2266 ,    0.     ,    0.     ,    0.     ,
	       0.     ,    0.     ,    0.     ,    0.2007 ,    4.56878,
	      26.1286 ,   42.4671 ,   22.7159 ,    3.8995 ,    0.01952,
	       0.     ,    0.     ,    0.23228,    0.91323,    3.28183,
	      20.6447 ,   51.0631 ,   20.6968 ,    2.69925,    0.4028 ,
	       0.02909,    0.00639,    0.03044])


	data_box1 = np.array([box1_1, box1_2, box1_3, box1_4]).T
	mu_box1 = np.mean(data_box1, axis=1)
	cov_box1 = np.cov(data_box1)
	# cov_box1 = np.eye(len(mu_box1)) * 0.0001
	box1_sample = np.random.multivariate_normal(mu_box1, cov_box1)

	data_chair1 = np.array([chair1_1, chair1_2, chair1_3, chair1_4, chair1_5, chair1_6]).T
	mu_chair1 = np.mean(data_chair1, axis=1)
	cov_chair1 = np.cov(data_chair1)
	# cov_chair1 = np.eye(len(mu_chair1)) * 0.0001
	chair1_sample = np.random.multivariate_normal(mu_chair1, cov_chair1)

	data_chair2 = np.array([chair2_1, chair2_2, chair2_3, chair2_4]).T
	mu_chair2 = np.mean(data_chair2, axis=1)
	cov_chair2 = np.cov(data_chair2)
	chair2_sample = np.random.multivariate_normal(mu_chair2, cov_chair2)

	data_table1 = np.array([table1_1, table1_2, table1_3]).T
	mu_table1 = np.mean(data_table1, axis=1)
	cov_table1 = np.cov(data_table1)
	table1_sample = np.random.multivariate_normal(mu_table1, cov_table1)

	# sanity check:
	print "should be small: ", [np.linalg.norm(box1_sample - bd) for bd in [box1_1, box1_2, box1_3, box1_4]]
	print "should be big: ", [np.linalg.norm(chair1_sample - bd) for bd in [box1_1, box1_2, box1_3, box1_4]]
	
	print "should be small: ", [np.linalg.norm(chair1_sample - bd) for bd in [chair1_1, chair1_2, chair1_3, chair1_4]]
	print "should be big: ", [np.linalg.norm(box1_sample - bd) for bd in [chair1_1, chair1_2, chair1_3, chair1_4]]

	# import ipdb;ipdb.set_trace()
	return {'box1': data_box1, 'chair1': data_chair1, 'chair2': data_chair2, 'table1': data_table1}

if __name__ == '__main__':
	get_data()

