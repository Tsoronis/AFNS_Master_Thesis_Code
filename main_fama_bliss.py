# Import dependcies
import pandas as pd # The best dataframe module
import numpy as np # The best module for numerical calculations
from scipy import linalg # nice linear algebra features
from scipy.optimize import minimize # scipy minimize might be one of the best library for optimization
from scipy import stats # making statistics easier (not in use)
from sklearn.metrics import mean_squared_error # calculated mse (we do it manually, but might come in handy)
from numdifftools import Jacobian, Hessian, Gradient # Calculate standard errors (diagonal inverse of hessian)
from matplotlib import pyplot as plt # just to update matplotlib cache 
import time # to check runtime on optimization
import importlib # just to update modules if needed
# Builded modules from scratch
import yield_adj_term as yat
import factor_loadings as fl
import constrained_guesser as cg
import Chart as ch


# model and notation inspired by
# Christensen, J., Diebold, F. & Rudebusch, G., (2011):
# "The affine arbitrage-free class of Nelson–Siegel term structure models", 
# Journal of Econometrics 164(2011), 4-20.

# ----- Data ----- #
# Pandas Dataframe collecting zero rates (data from crsp)
df = pd.read_excel(
    r'C:\Users\abt\OneDrive - BankInvest\speciale\python_code\Old_fama_bliss_unsmoothed.xlsx','Sheet1') # same as Christensen et al (2011)
df2 = pd.read_excel(
    r'C:\Users\abt\OneDrive - BankInvest\speciale\python_code\Old_fama_bliss_unsmoothed - Copy.xlsx','Sheet1') # More tenors (empty rates) for smoothing


# AFNS wrappers
# independent AFNS
def  independent_AFNS(par_guess, zcb_yield):
    """
    Kalman filter for log likelihood estimation and a wrapper for factor loadings and yield adjustment tern´m
    ----------
    par_guess: array
        initial guesses (constant) requires the length of the model parameters 10 +
        errors with the lenght of N maturities
    zcb_yield: pandas dataframe
        zero coupon yield, do not include date column
    Returns
    -------
    loglik: value
        log likelihood value
    """
    # create a 
    t = list(zcb_yield.columns) # get maturities
    delta_t = 1/12 # for months
    obs = len(zcb_yield) # get number of observations
    zcb_yield=np.matrix(zcb_yield) #yield dataframe to numpy matrix (easier math)

    par_List = [] # empty paramter list
    par_List[:] = par_guess #[:] to make sure that input do not mutate
    par_List[10:10+len(t)] = np.power(par_List[10:10+len(t)], 2)  # use parameter restriction on errors
   
    # parameters
    l_value = par_List[9]
    k_matrix = np.array([[-par_List[0],0,0],
                        [0,-par_List[1],0],
                        [0,0,-par_List[2]]])
    S_matrix = np.array([[par_List[3],0,0],
                        [0,par_List[4],0],
                        [0,0,par_List[5]]])
    theta_vec = np.array([par_List[6],par_List[7],par_List[8]])
    errors = np.diag(par_List[10:10+len(t)])
    S_matrix = abs(S_matrix)
    k_mat_matrix = abs(k_matrix)
    
    # calculate using the analytical solutions provided in Fisher and Gilles (1996)
    # Eigenvalue and Eigenvector of kappa
    eigen_val,eigen_vec = np.linalg.eig(k_mat_matrix)
    idx = eigen_val.argsort()[::-1] # sort eigen values and vector (not auto in python)
    eigen_val = eigen_val[idx]
    eigen_vec = eigen_vec[:,idx]
    
    # Solution to matrix equation 
    eig_sol = np.linalg.solve(eigen_vec,np.diag((1,1,1)))

    # S_bar to calculate cov matrix
    S_bar = eig_sol.dot(S_matrix).dot(np.matrix.transpose(S_matrix)).\
        dot(np.matrix.transpose(eig_sol))

    # Declaring empty arrays
    arraysize = len(eigen_val) # minus 1 since python range start at 0 
    V_con = np.array([])
    V_uncon = np.array([])
    
    # Ugly nested for loop but it works
    for i in range(0,arraysize): 
        append_con = [] 
        append_uncon = [] 
        for j in range(0,arraysize):
            # for conditional covariance
            V_con_n_k = S_bar[i,j]*(1-np.exp(-((eigen_val[i]+eigen_val[j])*delta_t)))/\
                                                    (eigen_val[i]+eigen_val[j])
            append_con = np.append(append_con, V_con_n_k)
            V_con_n_k = []
            # for unconditional covariance
            V_uncon_n_k = S_bar[i,j]/(eigen_val[i]+eigen_val[j])
            append_uncon = np.append(append_uncon, V_uncon_n_k)
            V_uncon_n_k = []
        V_con = np.append(V_con, append_con)
        V_uncon = np.append(V_uncon, append_uncon)
    

    # reshape to 3x3 matrix (np array)
    V_con = V_con.reshape((3,3))
    V_uncon = V_uncon.reshape((3,3))

    #  one-month analytical conditional & unconditional covariance matrix 
    # as the eigenvector matrix * eigenvalue matrix * eigenvetor matrix transposed
    global Covar
    Covar  = eigen_vec.dot(V_con).dot(np.matrix.transpose(eigen_vec)) 
    Covar_un = eigen_vec.dot(V_uncon).dot(np.matrix.transpose(eigen_vec))
    
    # ini Right part of the conditional mean equation
    Phi_1 = abs(-linalg.expm((k_matrix.dot(delta_t))))
    
    # ini Predict state
    Phi_0 = (np.diag((1,1,1))-Phi_1).dot(theta_vec).reshape(3,1)

    # ini Predict covariance
    Covar_hat = Phi_1.dot(Covar_un).dot(np.matrix.transpose(Phi_1))+Covar

    # initialize analytical yield adj. 
    yield_adj = yat.AFNS_yieldadjterm(
        s11=par_List[3], s12=0, s13=0,
        s21=0, s22=par_List[4], s23=0, 
        s31=0, s32=0, s33=par_List[5],
        l=l_value,t=t
        )

    # initialize factor loading
    factor_loadings = fl.NS_FL(l=l_value, t=t)

    # prep kalman filter and make some values global
    Updat_state = theta_vec
    Updat_cov = Covar_un
    global factor_e
    global yieldcurvemean
    global yieldcurve 
    global state_vec
    global y_error_array
    global QQ
    factor_e = np.array([]) # empty list of factor estimation
    yieldcurvemean = np.array([]) # empty list of yield mean estimation
    state_vec = np.array([]) # empty list of yield mean estimation
    y_error_array = np.array([])
    loglikelihood = 0
    loglik = 0
    global y_error_array_F
    global yieldcurvemean_F
    y_error_array_F = np.array([]) 
    yieldcurvemean_F = np.array([]) 
    for k in range(0, obs):
  
        # define global variable to call outside function
        X_hat = Phi_0+(Phi_1.dot(Updat_state)).reshape(3,1) # state updates for every time step
        Covar_hat = Phi_1.dot(Updat_cov).dot(np.matrix.transpose(Phi_1))+Covar # predicted covariance updates for every time step
            
        # observed and implied zero coupon yield for all maturities, predicted yield and error  
        yield_o = zcb_yield[k]/100 # observed
        yield_i = factor_loadings.dot(Updat_state.reshape(3,1)) + np.transpose(yield_adj).reshape((len(t), 1)) # model ex implied yield
        y_error = (yield_o).reshape((len(t), 1))-yield_i #error term
        
        # calculate kalman gain...
        eig_v = factor_loadings.dot(Covar_hat).dot(np.transpose(factor_loadings))+errors
        eig_v_sol = np.linalg.solve(eig_v,np.diag([1]*np.size(yield_o)))
        kalman_gain = Covar_hat.dot(np.transpose(factor_loadings)).dot(eig_v_sol)

        # now we need to update state and covariance
        Updat_state = X_hat+kalman_gain.dot(y_error)
        Updat_cov = Covar_hat-kalman_gain.dot(factor_loadings).dot(Covar_hat)

        # Evaluate the Gaussian log likelihood
        loglikelihood = - 0.5*len(t)*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(eig_v)).item()-0.5*np.transpose(y_error).dot(eig_v_sol).dot(y_error).item()
        # append to make arrays of output
        y_error_array = np.append(y_error_array, y_error)
        loglik = np.append(loglik,loglikelihood)
        state_vec = np.append(state_vec, X_hat)
        factor_e = np.append(factor_e, Updat_state)
        yieldcurvemean = np.append(yieldcurvemean, yield_i)
        # Rolling 1 month ahead 2m MA forecast
        if k > obs*0.66:
            yield_o_F = zcb_yield[k]/100 # observed   
            yield_i_F = factor_loadings.dot(((factor_e[[-6,-5,-4]]+factor_e[[-9,-8,-7]])/2).reshape(3,1)) + np.transpose(yield_adj).reshape((len(t), 1)) # model ex implied yield
            y_error_F = (yield_o_F).reshape((len(t), 1))-yield_i #error term
            y_error_array_F = np.append(y_error_array_F, y_error_F)
            yieldcurvemean_F = np.append(yieldcurvemean_F, yield_i_F)
    yieldcurve = np.array([]) # empty list of yield estimation
    yieldcurve = np.append(yieldcurve, yield_i)
    QQ = np.array([]) 
    QQ = np.append(QQ, Covar_hat)
    return -np.sum(loglik)


# smoothed yield curve (function to calculate yield for tenors with no data)
def print_yield_for_mat(df, parameters, states, t_new):
    # initialize analytical yield adj. 
    yield_adj = yat.AFNS_yieldadjterm(
        s11=parameters[3],s12=0,s13=0,
        s21=0, s22=parameters[4], s23=0, 
        s31=0, s32=0, s33=parameters[5],
        l=parameters[9],t=t_new
        )

    # initialize factor loading
    factor_loadings = fl.NS_FL(l=parameters[9], t=t_new)
    
    state_vec_frame = states.reshape((len(df),3))
    yield_pred_out = np.array([]) 
    for i in range(0, len(df)):
        yield_pred = factor_loadings.dot(state_vec_frame[i].reshape(3,1)) + np.transpose(yield_adj).reshape((len(t_new), 1))
        yield_pred_out = np.append(yield_pred_out, yield_pred)
    return yield_pred_out



# ---- data cleaning & create initial guess ---- #
df_no_date = df.drop(['Date'], axis=1) # fama bliss
t0 = list(df_no_date.columns)

# create random guess from the constrained guesser for loop
ini_guess_famabliss = cg.random_constrained_geuss(t=t0)

# ---- AFNS results famablisss data ----#
in_AFNS_famabliss = independent_AFNS(par_guess = par_ini_guess, zcb_yield = df_no_date)


# ---- maximize log likelihood ---- #

# Fama Bliss from Christensen et al. 
start_time = time.time() # see time of optim
max_log_like = minimize(independent_AFNS, x0=ini_guess_famabliss,args=(df_no_date),method='Nelder-Mead', 
                                                                options = {'disp':True,'maxiter':100000})
print("--- %s seconds ---" % (time.time() - start_time))
max_log_like.x # optimal para
max_log_like.fun # max log lik
x = max_log_like.x[:] # as array
x = list(x) # as list
independent_AFNS(par_guess=x, zcb_yield=df_no_date) # get results!


#------PLOT DATA-----#
#Plot latent factors: level, slope & curve
plot_data_factor_e = factor_e.reshape((len(df_no_date),3))
plot_data_factor_e = pd.DataFrame(plot_data_factor_e, columns=['L','S', 'C'])
plot_data_factor_e['index1'] = plot_data_factor_e.index
ch.graph.lines(x1='index1', y1='L', x2='index1', y2='S', x3='index1', y3='C',
    input=plot_data_factor_e, Fodnote='', y1label='L', y2label='S', y3label='C',
     path=r'C:\Users\abt\OneDrive - BankInvest\speciale\python_code\plots', save_name='factor_plot_fama.png')

importlib.reload(ch)

#Plot latest yield curve
yieldcurve_plot = pd.DataFrame(yieldcurve*100,columns=['yield'], index = t0)
yieldcurve_plot['index1'] = yieldcurve_plot.index
observed_yieldcurve = df_no_date[len(df_no_date)-1:].T
observed_yieldcurve.rename(columns = {'2':'yield_o'}, inplace = True)
observed_yieldcurve['index1'] = yieldcurve_plot.index
observed_yieldcurve.index = yieldcurve_plot.index
dfcombined = pd.merge(yieldcurve_plot,observed_yieldcurve,on='index1')
dfcombined.columns = ['yield_i', 'index', 'yield_o']

yieldcurve_plot_curve = pd.DataFrame(yieldcurve*100,columns=['yield'], index = t0)
yieldcurve_plot_curve['index1'] = yieldcurve_plot_curve.index
dfcombined = pd.merge(dfcombined,yieldcurve_plot_curve,on='index1')
dfcombined.columns = ['yield_i', 'index', 'yield_o']

ch.graph.lines(x1='index', y1='yield_i', x2='index', y2='yield_o',
    input=dfcombined, Fodnote='', title='', y1label='AFNS fitted yield', y2label='observed yield')

#Plot mean yield
yieldcurvemean_df = yieldcurvemean.reshape(len(df_no_date),len(t0))
yieldcurvemean_df = pd.DataFrame(yieldcurvemean_df,columns=df_no_date.columns)
yieldcurvemean_df = yieldcurvemean_df.iloc[1:]
yieldcurvemean_df = yieldcurvemean_df.mean()
yieldcurvemean_df = pd.DataFrame(yieldcurvemean_df, columns=['yield_i_mean'])
yieldcurvemean_df['index1'] = yieldcurvemean_df.index

obs_yield_mean = df_no_date.mean()
obs_yield_mean = pd.DataFrame(obs_yield_mean/100, columns=['obs_yield_mean'])
obs_yield_mean['index1'] = obs_yield_mean.index

mean_combined = pd.merge(yieldcurvemean_df, obs_yield_mean,on='index1')

plt.figure().clear()
importlib.reload(ch)
ch.graph.lines(x1='index1', y1='yield_i_mean', x2='index1', y2='obs_yield_mean', 
    input=mean_combined, Fodnote='', title='',
     y1label='Independent - AFNS fitted yield', y2label='observed yield',
     path=r'C:\Users\abt\OneDrive - BankInvest\speciale\python_code\plots', save_name='mean_yield_fama.png')



#smoothed yield curve
df_no_date2 = df2.drop(['Date'], axis=1) # fama bliss
t_new = list(df_no_date2.columns)
new_yieldcurvemean = print_yield_for_mat(df_no_date2, x,factor_e, t_new)

new_yieldcurvemean_df = new_yieldcurvemean.reshape(len(df_no_date2),len(t_new))
new_yieldcurvemean_df = pd.DataFrame(new_yieldcurvemean_df,columns=df_no_date2.columns)
new_yieldcurvemean_df = new_yieldcurvemean_df.mean()
new_yieldcurvemean_df = pd.DataFrame(new_yieldcurvemean_df, columns=['yield_i_mean'])
new_yieldcurvemean_df['index1'] = new_yieldcurvemean_df.index


new_obs_yield_mean = df_no_date2.mean()/100
new_obs_yield_mean = pd.DataFrame(new_obs_yield_mean, columns=['obs_yield_mean'])
new_obs_yield_mean['index1'] = new_obs_yield_mean.index

new_mean_combined = pd.merge(new_yieldcurvemean_df,new_obs_yield_mean,on='index1')
new_mean_combined = new_mean_combined.replace(0, np.nan)

linspace = np.linspace(0.0, 30, num=121, endpoint=True)
linspace = pd.DataFrame(linspace, columns=['index1'])
new_mean_combined = linspace.merge(new_mean_combined, how='left', on='index1')

new_mean_combined = new_mean_combined.interpolate()


plt.figure().clear()
importlib.reload(ch)
ch.graph.lines(x1='index1', y1='yield_i_mean',  x2='index1', y2='obs_yield_mean', 
    input=new_mean_combined[1:], Fodnote='', title='', pct ='Yes', y1label='AFNS', y2label='Observed Yields',
    path=r'C:\Users\abt\OneDrive - BankInvest\speciale\python_code\plots', save_name='mean_yield_smooth_fama.png')


#Get forecast results
yieldcurvemean_df_F = yieldcurvemean_F.reshape(len(df_no_date)-127,len(t0))
yieldcurvemean_df_F = pd.DataFrame(yieldcurvemean_df_F,columns=df_no_date.columns)
f_date = df['Date'].iloc[127-len(df_no_date):]
f_date = f_date.reset_index()
frames = [yieldcurvemean_df_F,f_date['Date']]
yieldcurvemean_df_F = pd.merge(yieldcurvemean_df_F, f_date['Date'], left_index=True, right_index=True)

yieldcurvemean_df_actual = yieldcurvemean.reshape(len(df_no_date),len(t0))
yieldcurvemean_df_actual = pd.DataFrame(yieldcurvemean_df_actual,columns=df_no_date.columns)
yieldcurvemean_df_actual = yieldcurvemean_df_actual.iloc[127-len(df_no_date):].reset_index()
yieldcurvemean_df_F_np = yieldcurvemean_df_F.drop(columns=['Date']).to_numpy()

yield_curve_F_vs_act = pd.merge(yieldcurvemean_df_F, yieldcurvemean_df_actual, left_index=True, right_index=True)

yield_curve_F_vs_act = yield_curve_F_vs_act.rename(columns = {'2_x' : '2Y_Forecast', '5_x' : '5Y_Forecast', '10_x': '10Y_Forecast',
                                                               '2_y' : '2Y_Actual', '5_y' : '5Y_Actual', '10_y': '10Y_Actual'})

plt.figure().clear()
importlib.reload(ch)
ch.graph.lines(x1='Date', y1='2Y_Actual', x3='Date', y3='2Y_Forecast',  x4='Date', y4='10Y_Actual',
    x8='Date', y8='10Y_Forecast',
    input=yield_curve_F_vs_act, Fodnote='', title='', y3style='--', y2style='--', pct='Yes',
    path=r'C:\Users\abt\OneDrive - BankInvest\speciale\python_code\plots', save_name='fama_forecast.png'
    )


# Table for in sample fit
rmse_table = (np.sqrt(np.power(y_error_array*100, 2)))
res_table = (y_error_array*100).reshape(len(df_no_date),len(t0))
rmse_table = rmse_table.reshape(len(df_no_date),len(t0))
res_table = pd.DataFrame(res_table, columns=df_no_date.columns)
rmse_table = pd.DataFrame(rmse_table, columns=df_no_date.columns)
res_table = res_table.mean()
rmse_table = rmse_table.mean()

# Table for out of sample fit
rmsfe_table = (np.sqrt(np.power(y_error_array_F*100, 2)))
resf_table = (y_error_array_F*100).reshape(len(df_no_date)-127,len(t0))
rmsfe_table = rmsfe_table.reshape(len(df_no_date)-127,len(t0))
resf_table = pd.DataFrame(resf_table, columns=df_no_date.columns)
rmsfe_table = pd.DataFrame(rmsfe_table, columns=df_no_date.columns)
resf_table = resf_table.mean()
rmsfe_table = rmsfe_table.mean()

# Testing statistical inference

# Jacobian gives dimensions 
def fun_der(par_guess, zcb_yield):
    return Jacobian(lambda par_guess: independent_AFNS(par_guess, zcb_yield))(par_guess).ravel()

degrees_free = len(df_no_date)-len(ini_guess_famabliss)
grad = fun_der(x, df_no_date)
grad = np.array(grad).reshape(26,1)
hess = np.inv(grad.dot(np.transpose(grad)))
se = np.sqrt(np.diag(hess))
se

#kappa matrix
np.exp(x[0]/12)
np.exp(-x[1]/12)
np.exp(x[2]/12)

# one month conditional covar
Covar

# optimal parameters
x

# ---- maximize log likelihood convergence test ---- #

#independent optim
log_max_array = []
for i in range(0,40):
    ini_guess_famabliss = cg.random_constrained_geuss(t=t0)
    max_log_like_test = minimize(independent_AFNS, x0=ini_guess_famabliss,args=(df_no_date),method='Nelder-Mead', 
                                                                options = {'disp':True,'maxiter':50000})
    log_max = max_log_like.fun
    log_max_array = np.append(log_max_array,log_max)
    
convergence = pd.DataFrame(log_max_array)
file_name = 'convergence_fama.xlsx'
convergence.to_excel(file_name)
