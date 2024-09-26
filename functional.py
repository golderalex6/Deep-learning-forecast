import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
import os
import json
plt.rc('figure',titleweight='bold',titlesize='large',figsize=(15,6))
plt.rc('axes',titleweight='bold',titlesize='large',labelweight='bold',labelsize='large',grid=True)


class funtional():
    def __init__(self):
        with open(os.path.join(Path().cwd(),'parameter.json'),'r') as f:
            params=json.load(f)
            self.symbol=params['symbol']
            self.timeframe=params['timeframe']
            self.test_size=params['test_size']
            self.ma_length=params['ma_length']
            self.price_type=params['price_type']
            self.seq_length=params['seq_length']

        self.df=pd.read_csv(os.path.join(Path().cwd(),f'data/{self.symbol}/{self.symbol}_{self.timeframe}.csv'),index_col=0,parse_dates=True)
        self.df['price']=self.df['Close']
        self.df['log']=np.log(self.df['Close'])
        self.df['moving_average']=self.df['price'].rolling(self.ma_length).mean()
        self.df['diff']=(self.df['price']-self.df['moving_average']).dropna()
        self.df.dropna(inplace=True)

        self.x,self.y=[],[]
        for i in range(len(self.df[self.price_type])-self.seq_length-1):
            self.x.append(self.df[self.price_type].iloc[i:i+self.seq_length].tolist())
            # self.x.append([u for u in range(i,i+self.seq_length)])
            self.y.append(self.df[self.price_type].iloc[i+self.seq_length]) 
        self.x,self.y=np.array(self.x),np.array(self.y)

        index=int(self.x.shape[0]*(1-self.test_size))
        self.x_train=self.x[:index]
        self.y_train=self.y[:index]
        self.x_test=self.x[index:]
        self.y_test=self.y[index:]
    def plot_summary(self):
        self.draw=pd.DataFrame({'True values':self.y,'Predict values':self.pred},index=self.df.index[self.seq_length+1:])
        fg=plt.figure()
        ax_1=fg.add_subplot(2,1,1)
        ax_2=fg.add_subplot(2,1,2)
        self.draw.plot(ax=ax_1)
        ax_1.set_title(f'Data-{self.timeframe}')

        ax_2.plot(self.history.history['loss'],label='loss')
        ax_2.plot(self.history.history['val_loss'],label='val_loss')
        ax_2.legend()
        ax_2.set_title('loss_curve')
        plt.tight_layout()
        plt.show()