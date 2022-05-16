# Master Thesis Code, provided by Jakob Meldgaard & Alexandros Bøgeskov-Tsoronis

Our Master Thesis is titled **Are Arbitrage-Free conditions on the Popular
Nelson-Siegel Model Practical?** and is about the implementation of the Arbitrage Free Nelson Siegel (AFNS).
We explain the model in depth, implement it with the use of python & test it on a practical Danish setting.

**main_fama_bliss & main_danish**:
These modules are as the name indicate, the main python files to execute the model.
They load datasets CRSP & Bloomberg respectivly. Both are licensed data sources, thus
one need access to both in order to replicate the data (we provide precise explanations, index names and date of data in the master thesis)

**Dependencies:** Apart from a standard Anaconda Python 3 installation, the project requires the following installations:

``conda install -c conda-forge numdifftools``
``pip install QuantLib``

**Other dependencies can be downloaded from this repository like yield_adj_term.py, Chart,py etc.**
**Anaconda Pyhton 3 installation include numpy, pandas, scipy and sklearn which is used in above modules**

