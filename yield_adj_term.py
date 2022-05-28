import numpy as np

def AFNS_yieldadjterm(s11,s12,s13,s21,s22,s23,s31,s32,s33,l,t):
    """
    Yield Adjustment Term 
    ----------
    s11,..,s33: floats
        variances from the volatility matrix
    l: float
        lambda (constant) "mean-reversion rate of the
        curvature"
    t: array
        tau (array of month to maturities)
    Returns
    -------
    adjterm: array
        yield adjustment term
    """

    Atilde = s11**2+s12**2+s13**2
    Btilde = s21**2+s22**2+s23**2
    Ctilde = s31**2+s32**2+s33**2
    Dtilde = s11*s21+s12*s22+s13*s23
    Etilde = s11*s31+s12*s32+s13*s33
    Ftilde = s21*s31+s22*s32+s23*s33
    # since the sigma matrix is restricted to only have values in the diagonal in the independent factor AFNS we only use these
    term1 = np.multiply(np.power(t,2)/6,Atilde) 
    term2 = Btilde*(1/(2*l**2)-(1-np.exp(np.multiply(t,-l)))/ \
        (np.multiply(t,np.power(l,3)))+(1-np.exp(-2*np.multiply(t,l)))/ \
        (4*np.multiply(t,np.power(l,3))))
    term3 = Ctilde*(1/(2*l**2)+np.exp(np.multiply(t,-l))/(l**2)- \
        np.multiply(t,np.exp(-2*np.multiply(t,l)))/(4*l)- \
        3*np.exp(-2*np.multiply(t,l))/(4*l**2)-2*(1-np.exp(np.multiply(t,-l)))/ \
        (np.multiply(t,np.power(l,3)))+5*(1-np.exp(-2*np.multiply(t,l)))/ \
        (8*np.multiply(t,np.power(l,3))))
    # but we also apply these such that the model can be expanded to a correlated factor AFNS
    term4 = Dtilde*(np.divide(t,2*l)+np.exp(np.multiply(t,-l)) \
        /(l**2)-(1-np.exp(np.multiply(t,-l)))/ \
        (np.multiply(t,np.power(l,3))))
    term5 = Etilde*((3*np.exp(np.multiply(t,-l)))/(l**2)+np.divide(t,2*l)+ \
        np.multiply(t,np.exp(np.multiply(t,-l)))/(l) \
        -3*(1-np.exp(np.multiply(t,-l)))/(np.multiply(t,l**3)))
    term6 = Ftilde*((1/(l**2))+np.exp(np.multiply(t,-l))/(l**2) \
        -np.exp(-2*np.multiply(t,l))/(2*l**2)-3*(1-np.exp(np.multiply(t,-l))) \
        /(np.multiply(t,l**3))+3*(1-np.exp(-2*np.multiply(t,l))) \
        /(4*np.multiply(t,l**3)))
    adjterm = term1+term2+term3+term4+term5+term6
    return -adjterm
