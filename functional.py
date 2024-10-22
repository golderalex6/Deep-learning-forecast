import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
import os
import json
plt.rc('figure',titleweight='bold',titlesize='large',figsize=(15,6))
plt.rc('axes',titleweight='bold',titlesize='large',labelweight='bold',labelsize='large',grid=True)


class plot_training(tf.keras.callbacks.Callback):
    def __init__(self,data,labels,plot_label=True):
        super().__init__()
        self.plot_label=plot_label
        self.data = data
        self.labels = labels
        self.fg=plt.figure()
        self.ax=self.fg.add_subplot()
    def on_epoch_end(self,epoch,logs=None):
        predictions = self.model.predict(self.data)
        self.ax.cla()
        self.ax.plot(predictions,label='Predicted Values',color='C3')
        self.ax.legend(loc='upper right')
        if self.plot_label:
            self.ax.plot(self.labels,label='True Values')
            self.ax.set_ylim([np.min(self.labels)-np.abs(np.min(self.labels))*0.1,np.max(self.labels)+np.abs(np.max(self.labels))*0.1])
        plt.pause(0.1)
        if len(plt.get_fignums())==0:
            raise 'Turn off training'

class funtional():
    def __init__(self):
        with open(os.path.join(Path(__file__).parent,'parameter.json'),'r') as f:
            params=json.load(f)
            self.symbol=params['symbol']
            self.timeframe=params['timeframe']
            self.test_size=params['test_size']
            self.ma_length=params['ma_length']
            self.price_type=params['price_type']
            self.seq_length=params['seq_length']

        self.df=pd.read_csv(os.path.join(Path(__file__).parent,f'data/{self.symbol}/{self.symbol}_{self.timeframe}.csv'),index_col=0,parse_dates=True)
        self.df['price']=self.df['Close']
        self.df['log']=np.log(self.df['Close'])
        self.df['moving_average']=self.df['price'].rolling(self.ma_length).mean()
        self.df['diff']=(self.df['price']-self.df['moving_average']).dropna()
        self.df.dropna(inplace=True)

        self.x,self.y=[],[]
        for i in range(len(self.df[self.price_type])-self.seq_length-1):
            self.x.append(self.df[self.price_type].iloc[i:i+self.seq_length].tolist())
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

    def train(self,epochs=10,optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='mae',batch_size=32,plot_label=True):
        self.model=self.build()
        self.model.compile(optimizer=optimizer,loss=loss)
        best_lost=tf.keras.callbacks.ModelCheckpoint(os.path.join(Path(__file__).parent,f'models/{self.model_name}_{self.price_type}_{self.symbol}_{self.timeframe}.weights.h5'),save_weights_only=True,monitor='val_loss',mode='min',save_best_only=True)
        draw=plot_training(self.x,self.y,plot_label)

        self.history=self.model.fit(self.x_train,self.y_train,epochs=epochs,batch_size=batch_size,callbacks=[best_lost,draw],validation_data=(self.x_test,self.y_test))
        self.pred=self.model.predict(self.x).reshape(-1)