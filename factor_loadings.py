import numpy as np

# factor loading function
def NS_FL(l, t):
    """
    Dynamic Nelson-Siegel factor loading structure
    ----------
    l: float
        lambda (constant) "mean-reversion rate of the
        curvature"
    t: array
        tau (array of month to maturities)
    Returns
    -------
    combine_facL: array
        Factor loading matrix
    """

    facL1 = np.repeat(1,len(t)) # level, vector of ones with the len of tau 
    facL2 = (np.subtract(1,np.exp(np.multiply(t,-l))))/(np.multiply(t,l))  # slope
    facL3 = facL2 - np.exp(np.multiply(t,-l))  # curvature
    combine_facL = np.stack([facL1, facL2, facL3]).transpose() # combine the factors
    return combine_facL
