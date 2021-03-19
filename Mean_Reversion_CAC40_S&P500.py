# -*- coding: utf-8 -*-
"""MeanReversionCAC40-S&P500

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1k_hRZf9-073SyNAOBDHrAe8e90kGfjCE
"""

pip install yfinance --upgrade --no-cache-dir

from pandas_datareader import data as pdr
import yfinance as yf 
yf.pdr_override

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class data():
  def __init__(self, tickers,period,interval):
    self.tickers = tickers
    self.period = period
    self.interval = interval

  def get_data(self):
    self.data = yf.download(self.tickers,period=self.period,interval=self.interval)
   
  def compute_returns(self):
    self.data = self.data['Close'].pct_change(1)
    self.data.dropna(inplace=True)

  def compute_spread(self):
    self.data['spread'] = self.data.iloc[:,0] - self.data.iloc[:,1]
    self.data['position'] = self.data['spread'].apply(lambda x : -1 if x >= 0 else 1)
    self.data.position = self.data.position.shift(1)
    self.data.dropna(inplace=True)

  def trade(self):
    self.data['P/L'] = 1 + self.data.position*self.data.spread
  
  def plot_profit(self):
    plt.plot(self.data['P/L'].cumprod(),label='strategy')
    plt.plot((1+self.data.iloc[:,0]).cumprod(),label='CAC40')
    plt.plot((1+self.data.iloc[:,1]).cumprod(),label='S&P500')
    plt.legend()
    plt.grid(True)
    plt.show()
    print('Annualized return is:',(self.data['P/L'].prod()**(1/int(self.period[0])))-1)
    

  def strategy_analysis(self):
    win=0 
    for i in range(len(self.data['P/L'])):
      if self.data.iloc[i,-1]>1:
        win += 1 
    accuracy = (win/len(self.data['P/L']))*100
    mean = ((self.data['P/L']-1).mean()*100)
    std = ((self.data['P/L']-1).std()*100)
    print('Accuracy = {} % '.format(accuracy))
    print('Mean = {} %'.format(mean))
    print('Stdev = {} %'.format(std))
    plt.hist((self.data['P/L']-1),bins=200,range=(-0.05,0.06))

df = data('^FCHI ^GSPC','2y','1d')
df.get_data()
df.compute_returns()
df.compute_spread()
df.trade()
df.plot_profit()
