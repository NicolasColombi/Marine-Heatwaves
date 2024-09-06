
import numpy as np 
import pandas as pd  
import time
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, mean_squared_error


def produce_ets_score(y_true: np.ndarray, dhw: np.ndarray)->tuple:

    """ Compute ETS score for a discretized interval of dhw values and dhw threshold. Returns
        the dhw_threshold that maximise the ETS for every dhw value of the interval.
        
        Parameters:
        -----------
        y_true : nd.array
            Observed bleaching percentage
        dhw : nd.array
            array of dhw values from the glorys DataFrame 

        Returns:
        -----------
        t_dhw : list
                list of the thresholds that maximise the ETS 
        max_ets : list
                list of the maximal ETS for every dhw value corresponding to the above threshold
        max_hits : list
                list of the maximal number of hits for every dhw value corresponding to the above threshold
        max_misses : list
                list of the maximal number of misses for every dhw value corresponding to the above threshold
        max_cm : list
                list of the maximal number of correct miss for every dhw value corresponding to the above threshold
        max_fa : list
                list of the maximal number of false alarms for every dhw value corresponding to the above threshold
        
    """
     
    max_dhw = np.nanmax(dhw) + 5
    t_dhw, max_ets, max_hits, max_misses, max_cm, max_fa = [], [], [], [], [], []
    
    for value in range(1,100,1):

        
        bleaching = y_true.copy()
        bleaching[np.where(bleaching < value)] = 0
        bleaching[np.where(bleaching >= value)] = 1

        thresholds = np.arange(0, max_dhw + 0.05, 0.05)
        n = dhw.shape[0]

        l, h, m, cm, fa = [], [], [], [], []

        for threshold in thresholds:
            # calc prediction of bleaching or not 
            mat = dhw - threshold
            mat[np.where(mat >= 0)] = 2
            mat[np.where(mat < 0)] = 0
            # results of hits, misses, false alarms and correct misses
            results = mat - bleaching
            H = np.count_nonzero(results == 1) # hits
            M = np.count_nonzero(results == -1) # miss
            FA = np.count_nonzero(results == 2) # false alarms
            H_random = (H + FA) * (H + M )/n
            correct_miss = np.count_nonzero(results == 0) 

            # calc ets score

            if (H + FA + M - H_random)== 0:
                ets = 0
            else:
                ets = (H - H_random)/(H + FA + M - H_random)
                # calc threat score
                # ets = H/(H + FA + M)

            l.append(ets)
            h.append(H/n)
            m.append(M/n)
            cm.append(correct_miss/n)
            fa.append(FA/n)

        
        max_index = np.argmax(l)
        dhw_t = thresholds[max_index]
        t_dhw.append(dhw_t)
        max_ets.append(max(l))
        max_hits.append(h[max_index])
        max_misses.append(m[max_index])
        max_cm.append(cm[max_index])
        max_fa.append(fa[max_index])

    return t_dhw, max_ets, max_hits, max_misses, max_cm, max_fa


def fit_impact_function(t_dhw: list, max_ets: list, type: str = 'GLORYS'):

    """ Fit a sigmoid function to the dhw thresholds and bleaching percentage array between 0 and 100.
        
        Parameters:
        -----------

        t_dhw : list
                list of the thresholds that maximise the ETS 
        max_ets : list
                list of the max ETS that maximise the ETS 

        Returns:
        -----------
        fit_params: np.array
                    parameters of the sigmoid function. array([center, slope, height])
    
    """

    # Define sigmoid function that we will fit 
    def sigmoid(x, center, slope):
        return 100 / (1 + np.exp(-slope * (x - center)))

    # Given data
    x_data = t_dhw
    y_data = np.arange(1, 100, 1)  # Update y_data range

    # Fit the sigmoid function to the data
    fit_params, cov_matrix = curve_fit(sigmoid, x_data, y_data)
    param_std = np.sqrt(np.diag(cov_matrix))

    # Generate the x values for the fitted sigmoid curve
    x_fit = np.linspace(0, 60, 100)

    # Calculate the y values using the fitted parameters
    y_fit = sigmoid(x_fit, *fit_params)

    # Calculate the confidence interval for the fitted curve
    confidence_interval = 1.96 * param_std[1]  # Assuming 95% confidence level

    print(confidence_interval)
    # Set the desired figure size
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the original data, the fitted sigmoid curve, and the confidence interval
    scatter = ax.scatter(t_dhw, np.arange(1, 100, 1), 
                        c=max_ets, cmap='nipy_spectral', 
                        label='Optimal DHW-BBT')
    scatter.set_clim(0, 0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('ETS')

    ax.plot(x_fit, y_fit, 'r-', label='Sigmoid fit')

    # Plot the confidence interval
    ax.fill_between(x_fit, y_fit - confidence_interval, y_fit + confidence_interval, color='gray', alpha=1)

    # Calculate goodness of fit
    y_pred = sigmoid(x_data, *fit_params)
    mae = mean_absolute_error(y_data, y_pred)

    # Set labels for the axes
    ax.set_xlabel('DHW [°C-Weeks]', fontfamily='Times New Roman', fontsize=12)
    ax.set_ylabel('Coral bleaching [%]', fontfamily='Times New Roman', fontsize=13)
    ax.set_title(f'ciao {type} Impact function \n param_std: {param_std}. MAE: {np.round(mae, 3)}')

    # Set y-axis limits
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 40)

    # Set y-axis ticks and label
    ax.set_yticks(np.arange(0, 101, 20))

    # Display the plot
    plt.legend(prop='Times New Roman', fontsize=16)
    plt.show()

    return fit_params


def DHW(weeks:int, 
        bleaching_treshold:int, 
        temperature_path:str, 
        MMM_path:str,
        destination_path:str,
        type):
    
    """ Compute the degree heating week.
        
        Parameters:
        -----------

        weeks : int
                duration of the accumulation windows in weeks 
        bleaching_treshold : int
        temperature_path: str
                path to the temeprature data
        MMM_path: str
                path to the maximum monthly mean 
        destination_path: str
                path where you wish to save the resulting dhw
                

        Returns:
        -----------
        D : np.array
            Degree heating week matrix
            
    """

    #record time
    start = time.time()
    # load data
    T = np.load(temperature_path)
    MMM = np.load(MMM_path)
    # calc anomalies
    threshold = MMM + bleaching_treshold
    A = T - threshold   
    # put to zeros negative anomalies 
    A[A < 0] = 0
    # days
    days = weeks * 7
    # add a NaN chunk to the matrix
    D = np.zeros((T.shape[0], days))
    D[:,:] = np.nan

    for i in tqdm.tqdm(list(range(days, T.shape[1]))):
        # extract the column and row 
        dhw = np.nansum(A[:,i-days:i], axis=1)/7  
        dhw = dhw.reshape(len(dhw), 1)
        D = np.concatenate((D, dhw), axis=1)

    # save 
    np.save(f'/{destination_path}/DHW_{weeks}_weeks_{bleaching_treshold}_{type}.npy', D)
    end = time.time()
    print('time:', end-start, 's')

    return D











