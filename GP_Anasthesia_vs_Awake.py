import numpy as np
import matplotlib
# matplotlib.rcParams['figure.figsize']=(4,3)
# %matplotlib inline
from matplotlib_inline.config import InlineBackend
from matplotlib import pyplot as plt
plt.style.use('dark_background')
import GPy

# Import datasets as pandas
# from scipy.io import loadmat
import pandas as pd
# data_set = loadmat("Mark_4sec_CA1PSD_ISO_freqamp_020619.mat")
CA1_df = pd.read_csv("Mark_4sec_CA1PSD_ISO_freqamp_020619.csv")
print(CA1_df)
print(CA1_df.describe())

# Select the Gamma  freq. stimulation
print(CA1_df.iloc[1, 133:152])
# calculate mean psd for Xk0 and Xk1
Y = np.mean(CA1_df.iloc[:, 133:152],axis=1)
print(np.shape(Y))
# Y.reset_index(drop=True, inplace=True)
Y_reshape =Y[:, np.newaxis]
print(Y)

plt.plot(np.arange(0,384), np.array(Y))
plt.show()

X = CA1_df.iloc[:, 0:2]
print(X)

# define kernel
ker = GPy.kern.Matern52(2,ARD=True) + GPy.kern.White(2)

# create simple GP model
m = GPy.models.GPRegression(X,Y_reshape,ker)
print(m)

GPy.plotting.change_plotting_library("matplotlib")
# GPy.plotting.change_plotting_library('plotly')
fig1 = m.plot(legend=False, xlabel='Stim. Freq.', ylabel='Stim. Amp.',
              label="Mark_4sec_CA1PSD_ISO_freqamp_020619");
print(m);
ax = plt.gca()
PCM = ax.get_children()[0]
plt.colorbar(PCM, ax=ax)
plt.show()

# optimize and plot
m.optimize(messages=True,max_f_eval = 1000);
figure = m.plot(legend=False, xlabel='Stim. Freq.' , ylabel='Stim. Amp.');
plt.show()
print(m)
ax = plt.gca()
PCM = ax.get_children()[0]
plt.colorbar(PCM, ax=ax)
