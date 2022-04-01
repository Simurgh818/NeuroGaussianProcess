import numpy as np
import matplotlib
matplotlib.rcParams['figure.figsize']=(5,3)
# from matplotlib_inline.config import InlineBackend
from matplotlib import pyplot as plt
plt.style.use('dark_background')
import GPy
import pandas as pd

def sample_the_model(m, testX):
    posteriorTestY = np.empty((1,2, 100))
    simY = np.empty((1,2, 100))
    simMse = np.empty((1,2, 100))
    
    for s in range(100):
        posteriorTestY[:,:,s] = m.posterior_samples_f(testX, full_cov=True, size=1)
        # print(np.shape(m.predict(testX)))
        simY[:,:, s], simMse[:,:,s] = m.predict(testX)
#     posteriorTestY = m.posterior_samples_f(testX, full_cov=True, size=100)
#     simY, simMse = m.predict(testX)
        
    # print("for sampling these test values for freq and amplitude: ", testX)
    # print("we get these model predictions: ", simY, simMse, '\n')


    # plt.plot(posteriorTestY[0,1,:])
    # plt.plot(simY[0,1,:] - 3 * simMse[0,1,:] ** 0.5, '--g')
    # plt.plot(simY[0,1,:] + 3 * simMse[0,1,:] ** 0.5, '--g')
    # plt.show()
    n, bins, patches = plt.hist(posteriorTestY[0,1,:], bins=10)
    max_n = int(max(n))
    plt.plot(simY[0,0,0:max_n], np.arange(0,max_n),'r--')
    plt.plot(simY[0,0,0:max_n] - 3 * simMse[0,0,0:max_n] ** 0.5, np.arange(0,max_n),'--g')
    plt.plot(simY[0,0,0:max_n] + 3 * simMse[0,0,0:max_n] ** 0.5, np.arange(0,max_n), '--g')
    plt.ylabel("count")
    plt.xlabel("gamma power")
    plt.legend(["mean","3 Std. Dev."])
    plt.show()
    
    return simY, simMse, posteriorTestY

def run_GP_model(X,Y, ker, cond):

    # create simple GP model
    m = GPy.models.GPRegression(X,Y,ker)
    print(m)

    # GPy.plotting.change_plotting_library("matplotlib",label=None, linewidth=1)
    # GPy.plotting.change_plotting_library('plotly')
    fig1 = m.plot(legend=False, xlabel='Stim. Freq.', ylabel='Stim. Amp.', label=None, linewidth=2);
    print(m);
    ax = plt.gca()
    PCM = ax.get_children()[0]
    plt.colorbar(PCM, ax=ax)
    plt.title("before optimization " + cond)
    # plt.show()

    # optimize and plot
    m.optimize(messages=True,max_f_eval = 1000);
    figure = m.plot(legend=False, xlabel='Stim. Freq.' , ylabel='Stim. Amp.', label=None, linewidth=2);
    plt.title("After optimization " + cond)
    print(m)
    ax2 = plt.gca()
    PCM2 = ax2.get_children()[0]
    plt.colorbar(PCM2, ax=ax2)
    
    return m

def get_model_inputs(dataset_path, condition_rows):

    # Import datasets as pandas
    CA1_df = pd.read_csv(dataset_path)
    # print(CA1_df)
    # print(CA1_df.describe())

    # Select the Gamma  freq. stimulation
    # print(CA1_df.iloc[1, 133:152])
    # calculate mean psd for Xk0 and Xk1
    Y = np.array(np.mean(CA1_df.iloc[condition_rows, 133:152],axis=1))
    print(np.shape(Y))
    # Y.reset_index(drop=True, inplace=True)
    # Y_reshape =np.array(Y[:, np.newaxis])
    Y_reshape =Y[:, np.newaxis]
    # print("The gamma freq and corresponding columns to be selected: ")
    # print(np.shape(Y))

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
    dataset_path = "Mark_4sec_CA3PSD_ISO_freqamp_020619.csv";
    print(dataset_path)
    condition = ['Anasthesia', 'Awake'] #,
    for idx in range(len(condition)):
        val =  condition[idx]
        print("the val is: ", val, '\n')
        if val == "Anasthesia":
            condition_rows = np.arange(0,100);
            print("--------------------------The Anastheisa model is running: --------------------------")
        else:
            condition_rows = np.arange(101, 384);
            print("--------------------------The Awake model is running: --------------------------")

        X, Y, ker = get_model_inputs(dataset_path, condition_rows);
        model = run_GP_model(X,Y,ker, condition[idx]);
        plt.show()
        print("-------------------Sampling from optimized model-------------------")
        testX = np.array([[47], [40]]) #52 , 50
        testX = np.transpose(testX)
        
        simY, simMse, posteriorTestY = sample_the_model(model, testX)
        plt.show()
      
       

    # plt.show()

    return model, testX, simY, simMse, posteriorTestY

model, testX, simY, simMse, posteriorTestY = main()