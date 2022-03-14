import numpy as np
import matplotlib
# matplotlib.rcParams['figure.figsize']=(4,3)
# from matplotlib_inline.config import InlineBackend
from matplotlib import pyplot as plt
plt.style.use('dark_background')
import GPy
import pandas as pd

def run_GP_model(X,Y, ker, cond):

    # create simple GP model
    m = GPy.models.GPRegression(X,Y,ker)
    print(m)

    GPy.plotting.change_plotting_library("matplotlib")
    # GPy.plotting.change_plotting_library('plotly')
    fig1 = m.plot(legend=False, xlabel='Stim. Freq.', ylabel='Stim. Amp.', label=None);
    print(m);
    ax = plt.gca()
    PCM = ax.get_children()[0]
    plt.colorbar(PCM, ax=ax)
    plt.title("before optimization " + cond)
    # plt.show()

    # optimize and plot
    m.optimize(messages=True,max_f_eval = 1000);
    figure = m.plot(legend=False, xlabel='Stim. Freq.' , ylabel='Stim. Amp.', label=None);
    plt.title("After optimization " + cond)
    print(m)
    ax2 = plt.gca()
    PCM2 = ax2.get_children()[0]
    plt.colorbar(PCM2, ax=ax2)
    # plt.show()
    return m

def get_model_inputs(dataset_path, condition_rows):

    # Import datasets as pandas
    CA1_df = pd.read_csv(dataset_path)
    print(CA1_df)
    print(CA1_df.describe())

    # Select the Gamma  freq. stimulation
    print(CA1_df.iloc[1, 133:152])
    # calculate mean psd for Xk0 and Xk1
    Y = np.mean(CA1_df.iloc[condition_rows, 133:152],axis=1)
    print(np.shape(Y))
    # Y.reset_index(drop=True, inplace=True)
    Y_reshape =Y[:, np.newaxis]
    print("The gamma freq and corresponding columns to be selected: ")
    print(Y)

    # plt.plot(condition_rows, np.array(Y))
    # plt.ylabel("gamma power")
    # plt.xlabel("trials")
    # plt.show()

    X = CA1_df.iloc[condition_rows, 0:2]
    print(X)

    # define kernel
    ker = GPy.kern.Matern52(2,ARD=True) + GPy.kern.White(2)

    return X,Y_reshape, ker


def main():
    dataset_path = "Mark_4sec_CA1PSD_ISO_freqamp_020619.csv";
    condition = ['Anasthesia', 'Awake'] #,
    for idx in range(len(condition)):
        val =  condition[idx]
        print("the val is: ", val, '\n')
        if val == "Anasthesia":
            condition_rows = np.arange(0,120);
            print("--------------------------The Anastheisa model is running: --------------------------")
        else:
            condition_rows = np.arange(121, 384);
            print("--------------------------The Awake model is running: --------------------------")

        X, Y, ker = get_model_inputs(dataset_path, condition_rows);
        model = run_GP_model(X,Y,ker, condition[idx]);
        print("-------------------Sampling from optimized model-------------------")

        testX = np.array([[32, 37,42, 47, 52], [0, 20, 30, 40, 50]])
        testX = np.transpose(testX)
        posteriorTestY = model.posterior_samples_f(testX, full_cov=True, size=3)
        simY, simMse = model.predict(testX)
        print("for sampling these test values for freq and amplitude: ", testX)
        print("we get these model predictions: ", simY, simMse, '\n')

    plt.show()

    return model

M = main()