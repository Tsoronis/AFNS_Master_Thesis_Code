# Master Thesis Code, provided by Jakob Meldgaard & Alexandros Bøgeskov-Tsoronis

Our Master Thesis is titled **A Technical Analysis of the Arbitrage-Free
Nelson-Siegel Model**, a hands-on implementation of the Arbitrage Free Nelson Siegel (AFNS).
We explain the model in depth, implement it with the use of python and test it in a Danish setting.
The code can replicate the results from the original paper using the same data set fairly accurate (see reference below).

**main_fama_bliss & main_danish**:

These modules are as the name indicates, the main python files to execute the model.
They load datasets CRSP & Bloomberg respectively. Both are licensed data sources, thus
one needs access to both in order to replicate the data (we provide precise explanations, index names and date of data in the master thesis)

**Dependencies:** 

Apart from a standard Anaconda Python 3 installation, the project requires the following installations:

``conda install -c conda-forge numdifftools``
``pip install QuantLib``

**Other dependencies can be downloaded from this repository like yield_adj_term.py, Chart,py etc.**
**Anaconda Python 3 installation includes NumPy, pandas, scipy and sklearn which are used in the above modules**

**Possible improvements**:

The main caveat is, that it is difficult to find convergence in the maximum likelihood estimation and
the selected Nelder-Mead algorithm is time-consuming. Thus we suggest trying other algorithms and 
testing whether more restrictions to the model could help the robustness of estimation.
There is also room for improvement in terms of error handlers, especially regarding estimating the standard errors.

**Model Reference**:

Christensen, J. H., Diebold, F. X., & Rudebusch, G. D. (2011). The affine arbitrage-
free class of nelson–siegel term structure models. Journal of Econometrics, 164 (1),
4–20.


