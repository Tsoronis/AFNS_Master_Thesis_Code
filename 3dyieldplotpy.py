import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import axes3d
import matplotlib.dates as dates
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

# Get yield data
df = pd.read_excel(
    r'C:\Users\abt\OneDrive - BankInvest\speciale\python_code\Old_fama_bliss_unsmoothed.xlsx','Sheet1') # same as Christensen et al (2011)

# Set index to Date col
df = df.set_index('Date')

# Records for pyplot format
df_record = df.to_records()

# Create headers
header = df.drop(['Date'], axis=1) # fama bliss
header = list(header.columns)

# Format x,y,z data
x_data = []; y_data = []; z_data = []
for dt in df_record.Date:
    dt_num = dates.date2num(dt)
    x_data.append([dt_num for i in range(len(df_record.dtype.names)-1)])
print ('x_data: ', x_data[1:5])

# append header
for row in df_record:
    y_data.append(header)
    z_data.append(list(row.tolist()[1:]))
print ('y_data: ', y_data[1:5])
print ('z_data: ', z_data[1:5])

# data to np array with dtype 'f' see matplotlib documentation
x = np.array(x_data, dtype='f'); y = np.array(y_data, dtype='f'); z = np.array(z_data, dtype='f')

# plotting
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, rstride=2, cstride=1, cmap='viridis', vmin=np.nanmin(z), vmax=np.nanmax(z))
ax.set_title('')
ax.set_ylabel('Maturity (\u03C4)')
ax.set_zlabel('Yield')

def format_date(x, pos=None):
     return dates.num2date(x).strftime('%Y')

ax.w_xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
for tl in ax.w_xaxis.get_ticklabels():
    tl.set_ha('right')
    tl.set_rotation(40)

plt.show()