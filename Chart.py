from ctypes.wintypes import DWORD
from tkinter import Y
from click import style
from matplotlib.axes import Axes
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from cycler import cycler
import os 

# This is our chart class for constructing beautiful charts

# https://matplotlib.org/3.1.0/gallery/color/named_colors.html    for colors
# https://stackoverflow.com/questions/14432557/matplotlib-scatter-plot-with-different-text-at-each-data-point   for datapoints

class graph:
    PrimaryColor =  '#002B5C'
    SecondaryColor = '#5CBEAF' 
    TertiaryColor = '#30B2E7'
    FourthColor = '#014F59'
    FifthColor = '#EDECE6'
    SixthColor = '#DA5E4F'
    SeventhColor = '#59B370'
    EightColor = '#FFFF00'
    NineColor = '#948B54' 
    TenColor ='#925494'
    EllevenColoer = '#002B5C'
    TwelveColor = '#282C2C'
    ThirteenColor = '#015A69'
    Fourteen = '#F62D13'
    FithteenColor = '#F68D2E'
    SixteenColor = '#282C2C'

    def style():
        
        plt.rcParams['axes.edgecolor'] = 'gray'#cycler(color='bgrcmyk')
        plt.rcParams['axes.linewidth'] = 2.5
        plt.rcParams['figure.figsize'] = (16,10)         #Figure size
        plt.rcParams['axes.titlelocation'] = 'left'      #Location of the title
        plt.rcParams['font.family'] = 'Century Gothic'   #Font 
        plt.rcParams['font.size'] = 16                   #Font Size
        plt.rcParams['axes.spines.top'] = False         #Turns off top axis
        #plt.rcParams['axes.spines.right'] = False       #Turns off right axis
        plt.rcParams['legend.frameon'] = False         #No Frame around label
        plt.rcParams['legend.fontsize'] = 14    
        plt.rcParams['ytick.labelsize'] = 25            #Size of xtick labelsize         
        plt.rcParams['xtick.labelsize'] = 25            #Size of xtick labelsize

        plt.rcParams['grid.linewidth'] = 0.8            #Size of gridlines
        plt.rcParams['lines.linewidth'] = 6         #Size of line 
        #plt.rcParams['']
        #plt.rcParams.keys() #To see differenct rcParams

    def lines(x1, y1, input, title='', x2=[], y2=[], x3=[], y3=[],
            xlabel='None', ylabel='None', Fodnote='',
            mio = 'Yes or No', mia = 'Yes or No', pct = 'Yes or No',
            y1label='', y2label='', y3label='', path='', save_name = '', y1style='-', y2style='-',y3style='-', 
            x4=[], y4=[], x5=[], y5=[],x6=[], y6=[], x7=[], y7=[], x8=[], y8=[], x9=[], y9=[],x10=[], y10=[], x11=[], y11=[], x12=[], y12=[], 
            x13=[], y13=[], x14=[], y14=[], x15=[], y15=[],x16=[], y16=[]):

        """Primary lines function. Works Good with timeseries data from a dataframe. 
        Notice when using this function, you can only pass a maximum of 3 different lines. If there is need for more, contact AFY or ABT. 
        Notice that the path to where the chart should be. 
        Line styles: (dashed line: '--'), (dotted line: ':'), (circle marker: 'o'), (Triangle: '^')
        """
        graph.style()

        ######Optional Units: Millions, Bilions. or percent
        if mio == 'Yes':
            mia = 'No'
            graph.format_mio(input=input, y1=y1, y2=y2, y3=y3)
            ylabel='mio.'
        if mia == 'Yes':
            graph.format_mia(input=input, y1=y1, y2=y2, y3=y3)
            ylabel='mia.'
        if pct is 'Yes':
            graph.format_pct(input=input, y1=y1, y2=y2, y3=y3, y4=y4, y5=y5, y6=y6, y7=y7, y8=y8, 
            y9=y9, y10=y10, y11=y11, y12=y12, y13=y13, y14=y14, y15=y15, y16=y16)
            ylabel='%'

        ######plotting the different timeseries
        plt.plot(x1, y1, data=input, color=graph.PrimaryColor)
        plt.plot(x2, y2, data=input, color=graph.SecondaryColor)
        plt.plot(x3, y3, y3style, data=input, color=graph.TertiaryColor)
        plt.plot(x4, y4,  data=input, color=graph.FourthColor)
        plt.plot(x5, y5,  data=input, color=graph.FifthColor)
        plt.plot(x6, y6, y1style, data=input, color=graph.SixthColor)
        plt.plot(x7, y7,  data=input, color=graph.SeventhColor)
        plt.plot(x8, y8, y2style, data=input, color=graph.EightColor)
        plt.plot(x9, y9,  data=input, color=graph.NineColor)
        plt.plot(x10, y10,  data=input, color=graph.TenColor)
        plt.plot(x11, y11,  data=input, color=graph.EllevenColoer)
        plt.plot(x12, y12,  data=input, color=graph.TwelveColor)
        plt.plot(x13, y13,  data=input, color=graph.ThirteenColor)
        plt.plot(x14, y14,  data=input, color=graph.Fourteen)
        plt.plot(x15, y15,  data=input, color=graph.FithteenColor)
        plt.plot(x16, y16,  data=input, color=graph.SixteenColor)

        sns.set_style("ticks", {"xtick.major.size": 24, "ytick.major.size": 24})            

        plt.figtext(0.75, -0.02, Fodnote, fontsize=20, font='Century Gothic')                #Fodnote
        
        plt.title(title,font='Century Gothic', fontsize=25, y=1.04, x=-0.025, loc='left')
        
        #y-label
        if ylabel=='None': 
            plt.ylabel('', fontsize=22)
        else: 
            plt.ylabel(ylabel, fontsize=22, y=1, x=-0.1, rotation=0)
        #X-label        
        if xlabel=='None': 
            plt.xlabel('', fontsize=16)
        else: 
            plt.xlabel(xlabel, fontsize=16, y=1)
        ###Legend
        if y3label != '':
            plt.legend(labels=[y1label, y2label,y3label], fontsize=14, prop='Century Gothic')
        elif y2label != '':
            plt.legend(labels=[y1label, y2label], fontsize=14, prop='Century Gothic')
        elif y1label != '':
            plt.legend(labels=[y1label], fontsize=14, prop='Century Gothic')
        else:
            plt.legend(fontsize=14, prop='Century Gothic')

        plt.tick_params(labelright=True)
        plt.margins(x=0.025, y=0.025)

        plt.ticklabel_format(axis='y', style='plain')
        plt.grid(axis='y')

        #FORMATTING BACK#
        if mio == "Yes": 
            graph.format_back_mio(input=input, y1=y1, y2=y2, y3=y3)
        if mia == "Yes": 
            graph.format_back_mia(input=input, y1=y1, y2=y2, y3=y3)
        if pct is 'Yes': 
            graph.format_back_pct(input=input, y1=y1, y2=y2, y3=y3, y4=y4, y5=y5, y6=y6, y7=y7, y8=y8, 
            y9=y9, y10=y10, y11=y11, y12=y12, y13=y13, y14=y14, y15=y15, y16=y16)

        #SHOW OR SAVE#
        if save_name == '':
            plt.show()
        elif save_name != '':
            if path == '':
                plt.savefig(save_name, bbox_inches='tight') 
            elif path != '':
                PATH = r'{}\{}'.format(path, save_name)
                plt.savefig(PATH, bbox_inches='tight') 
        


    def format_mio(input, y1=[], y2=[], y3=[]):
        input[y1] = input[y1] / 1000000
        input[y2] = input[y2] / 1000000
        input[y3] = input[y3] / 1000000
    def format_back_mio(input, y1=[], y2=[], y3=[]):
        input[y1] = input[y1] * 1000000
        input[y2] = input[y2] * 1000000
        input[y3] = input[y3] * 1000000
    def format_mia(input, y1=[], y2=[], y3=[]):
        input[y1] = input[y1] / 1000000000
        input[y2] = input[y2] / 1000000000
        input[y3] = input[y3] / 1000000000
    def format_back_mia(input, y1=[], y2=[], y3=[]):
        input[y1] = input[y1] * 1000000000
        input[y2] = input[y2] * 1000000000
        input[y3] = input[y3] * 1000000000
    def format_pct(input, y1=[], y2=[], y3=[], y4=[], y5=[], y6=[], y7=[], y8=[], 
            y9=[], y10=[], y11=[], y12=[], y13=[], y14=[], y15=[], y16=[]):
        input[y1] = input[y1] * 100
        input[y2] = input[y2] * 100
        input[y3] = input[y3] * 100
        input[y4] = input[y4] * 100
        input[y5] = input[y5] * 100
        input[y6] = input[y6] * 100
        input[y7] = input[y7] * 100
        input[y8] = input[y8] * 100
        input[y9] = input[y9] * 100
        input[y10] = input[y10] * 100
        input[y11] = input[y11] * 100
        input[y12] = input[y12] * 100
        input[y13] = input[y13] * 100
        input[y14] = input[y14] * 100
        input[y15] = input[y15] * 100
        input[y16] = input[y16] * 100
    def format_back_pct(input, y1=[], y2=[], y3=[], y4=[], y5=[], y6=[], y7=[], y8=[], 
            y9=[], y10=[], y11=[], y12=[], y13=[], y14=[], y15=[], y16=[]):
        input[y1] = input[y1] / 100
        input[y2] = input[y2] / 100
        input[y3] = input[y3] / 100
        input[y4] = input[y4] / 100
        input[y5] = input[y5] / 100
        input[y6] = input[y6] / 100
        input[y7] = input[y7] / 100
        input[y8] = input[y8] / 100
        input[y9] = input[y9] / 100
        input[y10] = input[y10] / 100
        input[y11] = input[y11] / 100
        input[y12] = input[y12] / 100
        input[y13] = input[y13] / 100
        input[y14] = input[y14] / 100
        input[y15] = input[y15] / 100
        input[y16] = input[y16] / 100
