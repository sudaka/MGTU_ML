import rivercommon
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt
import seaborn as sns
import os.path

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import acf,pacf,plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARMA, ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.iolib.table import SimpleTable

import tensorflow as tf

from scipy.fft import fft, fftfreq, ifft

class MainParams():
    def __init__(self, basefname = '00000'):
        self.basefname = basefname
        self.trainpart = 0.55
        self.prognozecount = 365
        self.unusualcolumns = []
        self.droppercent = 0.01
        self.curmean = 0
        self.curstd = 1
        self.pasthistory = self.prognozecount * 3
        self.traincount = self.prognozecount * 4
        self.testcount = self.prognozecount * 4
        self.buffersize = self.prognozecount * 50
        self.batchsize = self.prognozecount * 4

    def loadfromdf(self, df = pd.DataFrame()):
        self.traincount = int(self.trainpart * df.shape[0])
        self.testcount = df.shape[0] - self.traincount

    def save(self):
        with open(self.basefname+'.json', 'w') as outfile:
            json.dump(self.__dict__, outfile)

    def load(self):
        curfname = self.basefname + '.json'
        if not os.path.exists(curfname):
            return False
        curset = dict()
        with open(curfname) as infile:
            curset = json.load(infile)
        for curattr, curval in curset.items():
            self.__setattr__(curattr, curval)
        return True
        

def showdfinfo(cinfo = rivercommon.RiverInfo(), postreport = rivercommon.ReportHTML()):
    postreport.addsubheader('Структура данных:')
    print(cinfo.data.head())
    postreport.addtext(cinfo.data.head().to_html())
    postreport.addsubheader('Размер данных:')
    print('='*40,' Размер данных ', '='*40)
    print(cinfo.data.shape)
    postreport.addtext(cinfo.data.shape)
    print(cinfo.data.info())
    print('='*40,' Описательные статистики ', '='*40)
    postreport.addsubheader('Описательные статистики:')
    print(cinfo.data.describe().T)
    postreport.addtext(cinfo.data.describe().T.to_html())
    print('='*40,' Кол-во уникальных значений ', '='*40)
    print(cinfo.data.nunique())
    print(cinfo.data.isna())
    print(cinfo.data['Уровень'].isna().sum())

def dropunusualcolumns(cinfo = rivercommon.RiverInfo(), params = MainParams()):
    #Удаляем флаги, которые ни разу не изменялись за весь период наблюдений
    if len(params.unusualcolumns) == 0:
        print('=' * 40, ' Создание списка неиспользуемых флагов ', '=' * 40)
        fullcount = cinfo.data.shape[0]
        unusecolstxt = ''
        for colname in cinfo.data.columns:
            if colname != 'Уровень':
                difcount = cinfo.data[colname].value_counts().to_dict()
                for val in difcount.values():
                    if (val / fullcount) < params.droppercent or val == fullcount:
                        params.unusualcolumns.append(colname)
        params.save()
    postreport.addsubheader(f'Флаги, количество которых за весь период наблюдений менее {params.droppercent}:')
    unusecolstxt = ''
    for colname in params.unusualcolumns:
        unusecolstxt += f', {colname}'
    postreport.addtext(unusecolstxt+', количество удаленных флагов:'+str(len(params.unusualcolumns)))            
    cinfo.data = cinfo.data.drop(params.unusualcolumns, axis=1)

def findreplacenan(cinfo = rivercommon.RiverInfo()):
    #Находим последовательные промежутки, содержащие nan в столбце 'Уровень'
    postreport.addsubheader('Даты с пропущенными уровнями:')
    nanindexlist = cinfo.data[cinfo.data['Уровень'].isna()].index.to_list()
    datestxt = ''
    for curindex in nanindexlist:
        #if cinfo.data.iloc[curindex:curindex, 'прсх']:
        #    cinfo.data.iloc[curindex:curindex, 'Уровень'] = 0
        #else:
            cinfo.data[curindex:curindex] = cinfo.data[curindex-pd.Timedelta(days=1):curindex-pd.Timedelta(days=1)]
            datestxt += f'{curindex.day}/{curindex.month}/{curindex.year}, '
    if len(datestxt) > 1:
        postreport.addtext(datestxt[:-2])
    else:
        postreport.addtext('Пропущенных наблюдений нет.')

def showgist(data = pd.DataFrame()):

    spec = GridSpec(ncols=3, nrows=10, height_ratios=[1, 4]*5, hspace=0.55)
    fig = plt.figure(figsize=(18, 60))

    for i in range(5):
        for j in range(3):
            ax = fig.add_subplot(spec[6*i + j])
            sns.boxplot(data=data, x=data.columns[3*i + j], palette='Set2', ax=ax).set(xlabel=None)
            ax = fig.add_subplot(spec[6*i + j + 3])
            sns.histplot(data=data, x=data.columns[3*i + j], ax=ax, kde=True, stat='probability').set(ylabel=None)
    plt.show()

def showgistflags(data = pd.DataFrame(), picname = 'gist.jpg'):
    colnames = data.columns.to_list()
    gistcount = len(colnames)
    hbalance = [4] + [1]*(gistcount-1)
    spec = GridSpec(ncols=1, nrows=gistcount, height_ratios=hbalance, hspace=0.2)
    fig = plt.figure(figsize=(18, 80))
    for i in range(gistcount):
        ax = fig.add_subplot(spec[i])
        sns.lineplot(data=data, x=data.index, y=data.columns[i], ax=ax).set(xlabel=None)
    plt.savefig(picname, bbox_inches='tight',pad_inches = 0)
    postreport.addsubheader('Распределение уровня и флагов по времени:')
    postreport.addpic(picname)

def showpairdist(data = pd.DataFrame(), picname = 'gist.jpg'):
    #sns.pairplot(data, height=4, diag_kind='kde')
    postreport.addsubheader('Попарные графики зависимостей:')
    #plt.savefig(picname)
    postreport.addpic(picname, 200)

def normalizelevel(data = pd.DataFrame(), params = MainParams()):
    if params.curmean == 0:
        params.curmean = data['Уровень'].mean()
    if params.curstd == 1:
        params.curstd = data['Уровень'].max()/2
    params.save()
    data['Уровень'] = (data['Уровень'] - params.curmean) / params.curstd

def plot_train_history(history, title, picname):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()
    postreport.addsubheader('История обучения сети:')
    plt.savefig(picname)
    postreport.addpic(picname, 50)

def create_time_steps(length):
    return list(range(-length, 0))

def multi_step_plot(history, true_future, prediction, picname):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, 0]), label='Исторические данные')
    plt.plot(np.arange(num_out), np.array(true_future), 'bo',
             label='Действительные данные')
    if prediction.any():
      plt.plot(np.arange(num_out), np.array(prediction), 'ro',
               label='Спрогнозированные данные')
    plt.legend(loc='upper left')
    postreport.addsubheader('Предсказанный и реальный уровни:')
    plt.savefig(picname)
    postreport.addpic(picname, 50)

def multi_step_plot_dates(history, history_label, true_future, prediction, feature_label, picname):
    plt.figure(figsize=(12, 6))

    plt.plot(history_label, np.array(history[:, 0]), label='Исторические данные')
    plt.plot(feature_label, np.array(true_future), 'bo',
             label='Действительные данные')
    if prediction.any():
      plt.plot(feature_label, np.array(prediction), 'ro',
               label='Спрогнозированные данные')
    plt.legend(loc='upper left')
    postreport.addsubheader('Предсказанный и реальный уровни:')
    plt.savefig(picname)
    postreport.addpic(picname, 50)

def multivariate_data(dataset, target, mtype = 'val',  params = MainParams()):
    data = []
    labels = []
    start_index = params.pasthistory
    end_index = params.traincount
    if mtype == 'val':
        start_index = params.traincount + params.pasthistory
        end_index = len(dataset) - params.prognozecount
    for i in range(start_index, end_index):
        indices = list(range(i-params.pasthistory, i))
        data.append(dataset.iloc[indices])
        labels.append(target.iloc[i:i+params.prognozecount])
    return np.array(data), np.array(labels)

def one_test_data(dataset, target, start_index,  params = MainParams()):
    data = []
    labels = []
    indices = list(range(start_index-params.pasthistory, start_index))
    data.append(dataset.iloc[indices])
    labels.append(target.iloc[start_index:start_index+params.prognozecount])
    return np.array(data), np.array(labels)


def createltsmmodel(data = pd.DataFrame(), params = MainParams()):
    x_train_multi, y_train_multi = multivariate_data(data, data['Уровень'], 'train', params)
    x_val_multi, y_val_multi = multivariate_data(data, data['Уровень'], 'val', params)
    
    x_train_multi = np.asarray(x_train_multi).astype('float32')
    train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
    train_data_multi = train_data_multi.cache().shuffle(params.buffersize).batch(params.batchsize).repeat()
    
    x_val_multi = np.asarray(x_val_multi).astype('float32')
    val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
    val_data_multi = val_data_multi.batch(params.batchsize).repeat()
    
    multi_step_model = tf.keras.models.Sequential()
    multi_step_model.add(tf.keras.layers.LSTM(32, return_sequences=True, input_shape=x_train_multi.shape[-2:]))
    multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
    multi_step_model.add(tf.keras.layers.Dense(params.prognozecount))

    multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

    multi_step_history = multi_step_model.fit(train_data_multi, epochs=6, steps_per_epoch=365, validation_data=val_data_multi, validation_steps=365)
    plot_train_history(multi_step_history, 'Multi-Step Training and validation loss', params.basefname+'_multistep.jpg')
    multi_step_model.save(params.basefname + '_ltsm')

def loadsavedltsmmodel(data = pd.DataFrame(), params = MainParams()):
    x_val_multi, y_val_multi = multivariate_data(data, data['Уровень'], 'val', params)
    x_val_multi = np.asarray(x_val_multi).astype('float32')
    val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
    val_data_multi = val_data_multi.batch(params.batchsize).repeat()
    multi_step_model = tf.keras.models.load_model(params.basefname + '_ltsm')
    i = 0
    for x, y in val_data_multi.take(1):
        multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0], f'{params.basefname}_{i}_predictions.jpg')
        i += 1

def loadsavedltsmmodelone(data = pd.DataFrame(), params = MainParams(), sdate = '2013-03-15'):
    startindex = data.index.get_indexer([sdate])[0]
    x_val_multi, y_val_multi = one_test_data(data, data['Уровень'], startindex, params)
    x_val_multi = np.asarray(x_val_multi).astype('float32')
    val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
    val_data_multi = val_data_multi.batch(params.batchsize).repeat()
    multi_step_model = tf.keras.models.load_model(params.basefname + '_ltsm')
    for x, y in val_data_multi.take(1):
        unnorm_hist = x[0] * params.curstd + params.curmean
        y_unnorm_real = y[0] * params.curstd + params.curmean
        y_unnorm_predict = multi_step_model.predict(x)[0] * params.curstd + params.curmean
        hist_start = startindex - len(unnorm_hist)
        hist_lbl = data.iloc[hist_start:startindex].index.to_list()
        pred_end = startindex + len(y_unnorm_real)
        feature_lbl = data.iloc[startindex:pred_end].index.to_list()
        multi_step_plot_dates(unnorm_hist, hist_lbl, y_unnorm_real, y_unnorm_predict, feature_lbl, f'{params.basefname}_ltsm_0_prediction.jpg')

class CommonModel():
    def __init__(self, basefname = '00000'):
        self.params = MainParams(basefname)
        self.params.load()
        curfname = Path('postsinfo', self.params.basefname + '.csv')
        curinfo = rivercommon.RiverInfo()
        curinfo.loadfromfile(curfname)
        self.data = curinfo.data
        self.postreport = rivercommon.ReportHTML(self.params.basefname)

    def showdfinfo(self, isprint = False):
        print('='*40,' Создание описания датасета ', '='*40)
        dfhead = self.data.head()
        dfshape = self.data.shape
        dfinfo = self.data.info()
        dfdescgr = self.data.describe().T
        self.postreport.addsubheader('Структура данных:')
        self.postreport.addtext(dfhead.to_html())
        self.postreport.addsubheader('Размер данных:')
        self.postreport.addtext(dfshape)
        self.postreport.addsubheader('Описательные статистики:')
        self.postreport.addtext(dfdescgr.to_html())
        self.postreport.savereport()
        if isprint:
            print(dfhead)
            print('='*40,' Размер данных ', '='*40)
            print(dfshape)
            print(dfinfo)
            print('='*40,' Описательные статистики ', '='*40)
            print(dfdescgr)
            print('='*40,' Кол-во уникальных значений ', '='*40)
            print(self.data.nunique())
            print(self.data.isna())
            print(self.data['Уровень'].isna().sum())

    def normalizelevel(self):
        if self.params.curmean == 0:
            self.params.curmean = self.data['Уровень'].mean()
        if self.params.curstd == 1:
            self.params.curstd = self.data['Уровень'].max()/2
        self.params.save()
        self.data['Уровень'] = (self.data['Уровень'] - self.params.curmean) / self.params.curstd

class ArmaModel():
    def __init__(self, basefname = '00000'):
        super().__init__(basefname)
        self.postreport = rivercommon.ReportHTML(self.params.basefname+'_arma')
    
    def jbtest(self, curdata):
        row =  [u'JB', u'p-value', u'Перекос', u'Эксцесс']
        jb_test = sm.stats.stattools.jarque_bera(curdata)
        a = np.vstack([jb_test])
        itog = SimpleTable(a, row)
        print(itog)
        if jb_test[1] < 0.01 and abs(jb_test[3]-3) < 2:
            print('Данные нормальные')
        else:
            print('Данные не нормальные')
        print()
        
    def adf_test(self):
        adftest = adfuller(self.data['Уровень'])
        print('adf: ', adftest[0])
        print('p-value: ', adftest[1])
        print('Critical values: ', adftest[4])
        if adftest[0]> adftest[4]['5%']: 
            print('есть единичные корни, ряд не стационарен')
        else:
            print('единичных корней нет, ряд стационарен')

    def logdata(self):
        self.data['Уровень'] = np.log(self.data['Уровень'])

    def get_pdq(self):
        r,rac,Q = sm.tsa.acf(self.data['Уровень'], qstat=True)
        prac = pacf(self.data['Уровень'], method='ywmle')
        table_data = np.c_[range(1,len(r)), r[1:],rac,prac[1:len(rac)+1],Q]
        table = pd.DataFrame(table_data, columns=['lag', "AC","Q", "PAC", "Prob(>Q)"])

        print(table)


class FourierModel(CommonModel):
    def __init__(self, basefname = '00000'):
        super().__init__(basefname)
        self.postreport = rivercommon.ReportHTML(self.params.basefname+'_fourer')
    
    def cutconstlevel(self):
        #if self.params.constlevel == 0:
        #    self.params.constlevel = self.data['Уровень'].min()
        #self.params.save()
        minur = self.data['Уровень'].min()
        self.data['Уровень'] = self.data['Уровень'] - minur

    def checkdelta(self):
        self.cutconstlevel()
        period = 365
        dataarr = self.data['Уровень']
        basefreq = fft(dataarr[0:period])
        delta = []
        for i in range(10):
            newfreq = np.subtract(basefreq, fft(dataarr[i*period:(i+1)*period]))
            delta.append(newfreq)
        print(delta[9])

    def showtransform(self):
        #self.normalizelevel()
        self.cutconstlevel()
        N = self.params.prognozecount * 3
        #print(self.data.iloc[:N]['Уровень'].shape)
        yf = fft(self.data.iloc[:N]['Уровень'])
        xf = fftfreq(N, 1 / self.params.prognozecount)
        #print(yf[:100])
        #plt.plot(xf, np.abs(yf))
        new_sig = ifft(yf)
        #plt.plot(np.array(self.data.iloc[:1000]['Уровень']))
        #plt.plot(np.abs(new_sig[:1000]))
        #plt.show()

if __name__ == '__main__':
    #rivercommon.InitData()
    md = rivercommon.DataFile(fname=Path('rawinfo','Нижняя Тунгуска 2008.xls'))
    md.loadfile()
    curleg = md.legend.to_html()
    
    params = MainParams('09405')
    postreport = rivercommon.ReportHTML(params.basefname)
    postreport.addsubheader('Легенда файла данных:')
    postreport.addtext(curleg)
    curfname = Path('postsinfo', params.basefname + '.csv')
    curinfo = rivercommon.RiverInfo()
    curinfo.loadfromfile(curfname)
    if not params.load():
        params.loadfromdf(curinfo.data)
        params.save()
    findreplacenan(curinfo)
    dropunusualcolumns(curinfo, params)
    showdfinfo(curinfo, postreport)
    showgistflags(curinfo.data, params.basefname + '_level_flags.jpg')
    showpairdist(curinfo.data, params.basefname + '_pairs.jpg')
    normalizelevel(curinfo.data, params)
    #createltsmmodel(curinfo.data, params)
    loadsavedltsmmodel(curinfo.data, params)
    loadsavedltsmmodelone(curinfo.data, params, '2013-03-15')
    postreport.savereport()

    '''
    
    armamod = ArmaModel('09405')
    armamod.logdata()
    armamod.jbtest(armamod.data['Уровень'])
    #armamod.get_pdq()
    armamod.adf_test()
    '''
    '''ft = FourierModel('09405')
    ft.showtransform()
    ft.checkdelta()'''
