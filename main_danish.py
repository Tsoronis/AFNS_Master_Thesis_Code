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
import QuantLib as ql # Best library for Finance
import importlib # just to update modules if needed
# Builded modules from scratch
import yield_adj_term as yat # We build a function for the yield term adjustments
import factor_loadings as fl # We build a function for factor loadings
import constrained_guesser as cg # To make good guesses 
import Chart as ch # a class to make plotting beautiful


# model and notation inspired by
# Christensen, J., Diebold, F. & Rudebuscha, G., (2011):
# "The affine arbitrage-free class of Nelson–Siegel term structure models", 
# Journal of Econometrics 164(2011), 4-20.

# ----- Data ----- #
# Pandas Dataframe collecting raw dep and swaps (downloaded data from bloomberg)
df3 = pd.read_excel(
    r'C:\Users\abt\OneDrive - BankInvest\speciale\python_code\DanishDepFraSwapRates.xlsx','Sheet1') # raw deposit and swap data
df5 = pd.read_excel(
    r'C:\Users\abt\OneDrive - BankInvest\speciale\python_code\DanishDepFraSwapRates.xlsx','Sheet3') #  More tenors (empty rates) raw deposit and swap data

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

# -----zero rates from cibor and vanilla danish interest rate swaps ----- #
# Create zeros from danish deposits and swaps
DanishDepSwapRates = df3
DanishDepSwapRates['Date'] = DanishDepSwapRates['Date'].dt.strftime("%Y-%m-%d")
DanishDepSwapRates = DanishDepSwapRates.to_dict('records')
DanishDepSwapRates = list(DanishDepSwapRates)

calendar = ql.Germany() # germany? yes danish swaps follow this calaender
zeros = [] # empty list
deposits = ['3M', '6M'] # tenors
swaps = ['1Y','2Y','3Y', '4Y', '5Y', '6Y', '7Y', '8Y', '9Y', '10Y', '15Y', '20Y', '30Y'] # tenors
spread = ql.QuoteHandle(ql.SimpleQuote(0.005)) # handle float 6m
for row in DanishDepSwapRates:
    curve_date = ql.Date(row['Date'][:30], '%Y-%m-%d') # get curve date
    ql.Settings.instance().evaluationDate = curve_date 
    spot_date = calendar.advance(curve_date, 0, ql.Days) # important zero days in advance for danish swaps
    helpers = ql.RateHelperVector()
    for tenor in deposits:
        index = ql.IborIndex('DK dep', ql.Period(tenor), 0, ql.DKKCurrency(), #creating own ibor index
                  calendar, ql.ModifiedFollowing, True,  ql.Thirty360()) # we use modified following both
        helpers.append(                                                  # on fixed and floating leg
            ql.DepositRateHelper(row[tenor] / 100, index)
        )
    for tenor in swaps:
        swapIndex = ql.SwapIndex('DK Swap', ql.Period(tenor), 0, ql.DKKCurrency(), ql.Germany(), ql.Period('6M'), 
        ql.ModifiedFollowing, ql.Thirty360(), ql.Euribor(ql.Period('6M'))) #Euribor similar structure as Cibor, calendar etc-
        discountCurve = ql.YieldTermStructureHandle(ql.FlatForward(0, calendar, 0.05, ql.Thirty360(), ql.Compounded))
        helpers.append(
            ql.SwapRateHelper(row[tenor] / 100, swapIndex, spread, ql.Period())
        )
    curve = ql.PiecewiseSplineCubicDiscount(curve_date, helpers, ql.Actual360()) # curve fit

    for tenor in deposits + swaps:
        date = calendar.advance(spot_date, ql.Period(tenor)) # spot date and tenors
        rate = curve.zeroRate(date, ql.Actual360(), ql.Compounded, ql.Monthly).rate() #make use of zeroRate which calculate them based on fitted curve
        zeros.append({ 'curve_date': curve_date.to_date(), 'tenor': tenor, 'zero_rate': rate})

# reconstruting the dataframe
zero_rates = pd.DataFrame(zeros)
zero_rates['curve_date'] = pd.to_datetime(zero_rates['curve_date'])
zero_rates = zero_rates.pivot(index='curve_date', columns='tenor')['zero_rate']
zero_rates = zero_rates.rename_axis(columns = None).reset_index()
zero_rates = zero_rates[["curve_date","3M", "6M", "1Y","2Y","3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "15Y", "20Y", "30Y"]]

# plotting the danish zero rate evolution
plt.figure().clear()
importlib.reload(ch)
ch.graph.lines(x1='curve_date', y1="3M", x2='curve_date', y2="6M", x3='curve_date', y3="1Y", x4='curve_date', y4="2Y", 
    x5='curve_date', y5="3Y", x6='curve_date', y6="4Y", x7='curve_date', y7="5Y", x8='curve_date', y8="6Y", 
    x9='curve_date', y9="7Y", x10='curve_date', y10="8Y", x11='curve_date', y11="9Y", x12='curve_date', y12="10Y", 
    x13='curve_date', y13="15Y", x14='curve_date', y14="20Y", x15='curve_date', y15="30Y",
    input=zero_rates, title='', pct='Yes')


# ---- data cleaning & create initial guess ---- #
zero_rates = zero_rates.rename(columns = {'curve_date' : 'Date','3M' : 0.25, '6M' : 0.5, '1Y':1,'2Y':2,'3Y':3, '4Y':4, 
                    '5Y':5, '6Y':6, '7Y':7, '8Y':8, '9Y':9, '10Y':10, '15Y':15, '20Y':20, '30Y':30})
zero_rates_no_date = zero_rates.drop(['Date'], axis=1) # own calculated zeros
zero_rates_no_date.columns = zero_rates_no_date.columns.map(float)
zero_rates_no_date = zero_rates_no_date.multiply(100)

t0 = list(zero_rates_no_date.columns)


# create random guess from the constrained guesser for loop
ini_guess = cg.random_constrained_geuss(t=t0)


# ---- AFNS results danish data ----#
in_AFNS_res independent_AFNS(par_guess = ini_guess, zcb_yield = zero_rates_no_date)


# ---- maximize log likelihood ---- #
start_time = time.time() # see time of optim
max_log_like = minimize(independent_AFNS, x0=ini_guess,args=(zero_rates_no_date),method='Nelder-Mead', 
                                                                options = {'disp':True,'maxiter':100000})
print("--- %s seconds ---" % (time.time() - start_time))
max_log_like.x # optimal para
max_log_like.fun # max log lik
x = max_log_like.x[:] # as array
x = list(x) # as list
independent_AFNS(par_guess=x, zcb_yield=df_no_date) # get results!


#------PLOT DATA-----#
#Plot latent factors: level, slope & curve
plot_data_factor_e = factor_e.reshape((len(zero_rates_no_date),3))
plot_data_factor_e = pd.DataFrame(plot_data_factor_e, columns=['L','S', 'C'])
plot_data_factor_e['index1'] = plot_data_factor_e.index
ch.graph.lines(x1='index1', y1='L', x2='index1', y2='S', x3='index1', y3='C',
    input=plot_data_factor_e, Fodnote='', y1label='L', y2label='S', y3label='C',
     path=r'C:\Users\abt\OneDrive - BankInvest\speciale\python_code\plots', save_name='factor_plot_danish.png')


#Plot mean yield
yieldcurvemean_df = yieldcurvemean.reshape(len(zero_rates_no_date),len(t0))
yieldcurvemean_df = pd.DataFrame(yieldcurvemean_df,columns=zero_rates_no_date.columns)
yieldcurvemean_df = yieldcurvemean_df.iloc[1:]
yieldcurvemean_df = yieldcurvemean_df.mean()
yieldcurvemean_df = pd.DataFrame(yieldcurvemean_df, columns=['yield_i_mean'])
yieldcurvemean_df['index1'] = yieldcurvemean_df.index

obs_yield_mean = zero_rates_no_date.mean()
obs_yield_mean = pd.DataFrame(obs_yield_mean/100, columns=['obs_yield_mean'])
obs_yield_mean['index1'] = obs_yield_mean.index

mean_combined = pd.merge(yieldcurvemean_df, obs_yield_mean,on='index1')

plt.figure().clear()
importlib.reload(ch)
ch.graph.lines(x1='index1', y1='yield_i_mean', x2='index1', y2='obs_yield_mean', 
    input=mean_combined, Fodnote='', title='',
     y1label='Independent - AFNS fitted yield', y2label='observed yield',
     path=r'C:\Users\abt\OneDrive - BankInvest\speciale\python_code\plots', save_name='mean_yield_danish.png')



#Get forecast results
yieldcurvemean_df_F = yieldcurvemean_F.reshape(len(zero_rates_no_date)-137,len(t0))
yieldcurvemean_df_F = pd.DataFrame(yieldcurvemean_df_F,columns=zero_rates_no_date.columns)
f_date = df3['Date'].iloc[137-len(zero_rates_no_date):]
f_date = f_date.reset_index()
frames = [yieldcurvemean_df_F,f_date['Date']]
yieldcurvemean_df_F = pd.merge(yieldcurvemean_df_F, f_date['Date'], left_index=True, right_index=True)

yieldcurvemean_df_actual = yieldcurvemean.reshape(len(zero_rates_no_date),len(t0))
yieldcurvemean_df_actual = pd.DataFrame(yieldcurvemean_df_actual,columns=zero_rates_no_date.columns)
yieldcurvemean_df_actual = yieldcurvemean_df_actual.iloc[137-len(zero_rates_no_date):].reset_index()

yield_curve_F_vs_act = pd.merge(yieldcurvemean_df_F, yieldcurvemean_df_actual, left_index=True, right_index=True)

yield_curve_F_vs_act = yield_curve_F_vs_act.rename(columns = {'2.0_x' : '2Y_Forecast', '5.0_x' : '5Y_Forecast', '10.0_x': '10Y_Forecast',
                                                               '2.0_y' : '2Y_Actual', '5.0_y' : '5Y_Actual', '10.0_y': '10Y_Actual'})

yield_curve_F_vs_act['Date'] = pd.to_datetime(yield_curve_F_vs_act['Date'])

plt.figure().clear()
importlib.reload(ch)
ch.graph.lines(x1='Date', y1='2Y_Actual', x3='Date', y3='2Y_Forecast',  x4='Date', y4='10Y_Actual',
    x8='Date', y8='10Y_Forecast',
    input=yield_curve_F_vs_act, Fodnote='', title='', y3style='--', y2style='--', pct='Yes',
    path=r'C:\Users\abt\OneDrive - BankInvest\speciale\python_code\plots', save_name='danish_forecast.png'
    )



# Table for in sample fit
rmse_table = (np.sqrt(np.power(y_error_array*100, 2)))
res_table = (y_error_array*100).reshape(len(zero_rates_no_date),len(t0))
rmse_table = rmse_table.reshape(len(zero_rates_no_date),len(t0))
res_table = pd.DataFrame(res_table, columns=zero_rates_no_date.columns)
rmse_table = pd.DataFrame(rmse_table, columns=zero_rates_no_date.columns)
res_table = res_table.mean()
rmse_table = rmse_table.mean()

# Table for out of sample fit
rmsfe_table = (np.sqrt(np.power(y_error_array_F*100, 2)))
resf_table = (y_error_array_F*100).reshape(len(zero_rates_no_date)-137,len(t0))
rmsfe_table = rmsfe_table.reshape(len(zero_rates_no_date)-137,len(t0))
resf_table = pd.DataFrame(resf_table, columns=zero_rates_no_date.columns)
rmsfe_table = pd.DataFrame(rmsfe_table, columns=zero_rates_no_date.columns)
resf_table = resf_table.mean()
rmsfe_table = rmsfe_table.mean()

# Testing statistical inference

# Jacobian gives dimensions 
def fun_der(par_guess, zcb_yield):
    return Jacobian(lambda par_guess: independent_AFNS(par_guess, zcb_yield))(par_guess).ravel()

degrees_free = len(zero_rates_no_date)-len(ini_guess)
grad = fun_der(x, zero_rates_no_date)
grad = np.array(grad).reshape(25,1)
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
    ini_guess = cg.random_constrained_geuss(t=t0)
    max_log_like_test = minimize(independent_AFNS, x0=ini_guess,args=(zero_rates_no_date),method='Nelder-Mead', 
                                                                options = {'disp':True,'maxiter':50000})
    log_max = max_log_like.fun
    log_max_array = np.append(log_max_array,log_max)
    
convergence = pd.DataFrame(log_max_array)
file_name = 'convergence.xlsx'
convergence.to_excel(file_name)

#smoothed yield curve
# create new maturities 0.75, 1.5, 12,14,17,18,22,24,27,29
zero_rates_no_date_smooth = []
zero_rates_no_date_smooth = zero_rates_no_date
zero_rates_no_date_smooth[0.75] = 0
zero_rates_no_date_smooth[1.5] = 0
zero_rates_no_date_smooth[12] = 0
zero_rates_no_date_smooth[14] = 0
zero_rates_no_date_smooth[17] = 0
zero_rates_no_date_smooth[18] = 0
zero_rates_no_date_smooth[22] = 0
zero_rates_no_date_smooth[24] = 0
zero_rates_no_date_smooth[27] = 0
zero_rates_no_date_smooth[29] = 0


zero_rates_no_date_smooth = zero_rates_no_date_smooth[[0.25,0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 17, 18, 20, 22, 24, 27, 29, 30]]


t_new = list(zero_rates_no_date_smooth.columns)
new_yieldcurvemean = print_yield_for_mat(zero_rates_no_date_smooth, x,factor_e, t_new)

new_yieldcurvemean_df = new_yieldcurvemean.reshape(len(zero_rates_no_date_smooth),len(t_new))
new_yieldcurvemean_df = pd.DataFrame(new_yieldcurvemean_df,columns=zero_rates_no_date_smooth.columns)
new_yieldcurvemean_df = new_yieldcurvemean_df.mean()
new_yieldcurvemean_df = pd.DataFrame(new_yieldcurvemean_df, columns=['yield_i_mean'])
new_yieldcurvemean_df['index1'] = new_yieldcurvemean_df.index


new_obs_yield_mean = zero_rates_no_date_smooth.mean()/100
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
    path=r'C:\Users\abt\OneDrive - BankInvest\speciale\python_code\plots', save_name='mean_yield_smooth_danish.png')


