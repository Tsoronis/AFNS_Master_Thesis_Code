import random

# Use random lib to make uniform random guess for each parameter
# Multiple for loops since different groups of paramters have different
# economic relevance

# create random geuss for independent AFNS
def random_constrained_geuss(t):
    random_float_list = []
    for i in range(0,3):
        x = round(random.uniform(0.08, 1.0), 5)
        random_float_list.append(x) 
    for i  in range(0,3):
        x = round(random.uniform(0.001, 0.005), 5)
        random_float_list.append(x)
    for i  in range(0,3):
        x = round(random.uniform(-0.05, 0.08), 5)
        random_float_list.append(x)
    for i  in range(0,1):
        x = round(random.uniform(0.5, 0.6), 5)
        random_float_list.append(x)  
    for i  in range(0,len(t)):                  
        x = round(random.uniform(0.0001, 0.0009), 5)
        random_float_list.append(x) 
    return random_float_list    

  