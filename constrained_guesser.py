import random

# Use random lib to make uniform random guess for each parameter
# Multiple for loops since different groups of paramters have different
# Economic relevance

# create random geuss for independent AFNS
def random_constrained_geuss(t):
    random_float_list = []
    for i in range(0,3):
        x = round(random.uniform(0.01, 1.0), 5) # Kappa matrix diagonal values
        random_float_list.append(x) 
    for i  in range(0,3):
        x = round(random.uniform(0.001, 0.005), 5)  # Sigma matrix diagonal values
        random_float_list.append(x)
    for i  in range(0,1):
        x = round(random.uniform(0.01, 0.05), 5) # Theta_1 
        random_float_list.append(x)
    for i  in range(0,2):
        x = round(random.uniform(-0.05, -0.01), 5) # Theta_2 and Theta_3
        random_float_list.append(x)
    for i  in range(0,1):
        x = round(random.uniform(0.1, 0.9), 5) # Lambda
        random_float_list.append(x)  
    for i  in range(0,len(t)):                  
        x = round(random.uniform(0.0001, 0.0009), 5) # Random guees for the range..
        random_float_list.append(x) 
    return random_float_list    

  
