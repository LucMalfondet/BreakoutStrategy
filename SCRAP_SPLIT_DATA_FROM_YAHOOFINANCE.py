#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 00:31:43 2022

@author: lukas
"""

import yfinance as yf
import datetime as dt
from datetime import datetime
import numpy as np
import os
import shutil
import math
import pathlib
import pandas as pd
import tensorflow as tf
import datetime as dt
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as web
import os
import shutil
import pathlib
import PIL
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import plotly.graph_objects as go
from math import *
import decimal
from decimal import Decimal
import random

pathfichier='/Users/lukas/Documents/FICHIER_STRATGIE_TRADING_TEST10'
if os.path.exists(pathfichier)==False:
    os.mkdir(pathfichier)
else:
    shutil.rmtree(pathfichier)
    os.mkdir(pathfichier)
    
    
    
pathfichier2='/Users/lukas/Documents/FICHIER_STRATGIE_TRADING_TEST10/FUSION_DES_DONNEES'
path2haussier=pathfichier2+'/HAUSSIER'
path2baissier=pathfichier2+'/BAISSIER'

if os.path.exists(pathfichier2)==False:
    os.mkdir(pathfichier2)
    os.mkdir(path2haussier)
    os.mkdir(path2baissier)
            
else:
    shutil.rmtree(pathfichier2)
    os.mkdir(pathfichier2)
    os.mkdir(path2haussier)
    os.mkdir(path2baissier)
    
os.chdir(pathfichier2)
    
    
trainpath='/Users/lukas/Documents/FICHIER_STRATGIE_TRADING_TEST10/TRAIN'
trainpathhaussier='/Users/lukas/Documents/FICHIER_STRATGIE_TRADING_TEST10/TRAIN/HAUSSIER'
trainpathbaissier='/Users/lukas/Documents/FICHIER_STRATGIE_TRADING_TEST10/TRAIN/BAISSIER'

if os.path.exists(trainpath)==False:
    os.mkdir(trainpath)
    os.mkdir(trainpathhaussier)
    os.mkdir(trainpathbaissier)
            
else:
    shutil.rmtree(trainpath)
    os.mkdir(trainpath)
    os.mkdir(trainpathhaussier)
    os.mkdir(trainpathbaissier)
        
validpath='/Users/lukas/Documents/FICHIER_STRATGIE_TRADING_TEST10/VALID'
validpathhaussier='/Users/lukas/Documents/FICHIER_STRATGIE_TRADING_TEST10/VALID/HAUSSIER'
validpathbaissier='/Users/lukas/Documents/FICHIER_STRATGIE_TRADING_TEST10/VALID/BAISSIER'

if os.path.exists(validpath)==False:
    os.mkdir(validpath)
    os.mkdir(validpathhaussier)
    os.mkdir(validpathbaissier)
            
else:
    shutil.rmtree(validpath)
    os.mkdir(validpath)
    os.mkdir(validpathhaussier)
    os.mkdir(validpathbaissier)
    
    
testpath='/Users/lukas/Documents/FICHIER_STRATGIE_TRADING_TEST10/TEST'
testpathhaussier='/Users/lukas/Documents/FICHIER_STRATGIE_TRADING_TEST10/TEST/HAUSSIER'
testpathbaissier='/Users/lukas/Documents/FICHIER_STRATGIE_TRADING_TEST10/TEST/BAISSIER'    
    
if os.path.exists(testpath)==False:
    os.mkdir(testpath)
    os.mkdir(testpathhaussier)
    os.mkdir(testpathbaissier)
            
else:
    shutil.rmtree(testpath)
    os.mkdir(pathfichier2)
    os.mkdir(testpathhaussier)
    os.mkdir(testpathbaissier)
    
    
pathdossier_haussier='/Users/lukas/Documents/FICHIER_STRATGIE_TRADING_TEST10/FUSION_DES_DONNEES/HAUSSIER'
pathdossier_baissier='/Users/lukas/Documents/FICHIER_STRATGIE_TRADING_TEST10/FUSION_DES_DONNEES/BAISSIER'

trainpathhaussier='/Users/lukas/Documents/FICHIER_STRATGIE_TRADING_TEST10/TRAIN/HAUSSIER'
trainpathbaissier='/Users/lukas/Documents/FICHIER_STRATGIE_TRADING_TEST10/TRAIN/BAISSIER'

validpathhaussier='/Users/lukas/Documents/FICHIER_STRATGIE_TRADING_TEST10/VALID/HAUSSIER'
validpathbaissier='/Users/lukas/Documents/FICHIER_STRATGIE_TRADING_TEST10/VALID/BAISSIER'

testpathhaussier='/Users/lukas/Documents/FICHIER_STRATGIE_TRADING_TEST10/TEST/HAUSSIER'
testpathbaissier='/Users/lukas/Documents/FICHIER_STRATGIE_TRADING_TEST10/TEST/BAISSIER'




tickerlist=['^GDAXI','^FCHI','^STOXX50E','^IBEX','^IBEX','FTSEMIB.MI','^AEX','^SSMI','^N225','^IXIC','^SPX','^DJI','^NYA','^KS11','SNSX50.BO']
seuillist=[100,100,100,100,100,200,10,100,200,200,100,200,200,100,200]




enddate = dt.date.today()
td = dt.timedelta(days = 729)
startdate=enddate-td 		

roundlist=[]

for i in tickerlist:
    df=yf.download(tickers=i,start=startdate, end=enddate,interval='1h')
    arrond=round(df['Close'][0],2)
    roundlist.append(arrond)
    
    
print(roundlist)    

			
dc = pd.DataFrame({'ticker':tickerlist,
                   'first_round':roundlist,
                   'seuil':seuillist
                    })


def GetData(ticker):
    r=[]
    ticker=[ticker]
    df=yf.download(tickers=ticker,start=startdate, end=enddate,interval='1h')
    print(df.head())
    
    for i in range(len(df)): 
        r.append(i)
        
    df["index"]=r
    
    df.to_csv('/Users/lukas/Documents/FICHIER_STRATGIE_TRADING_TEST10/_{}.csv'.format(ticker))     
    return df




def PlotCandels(df_index,col_open,col_high,col_low,col_close,s):
            fig = go.Figure(data=[go.Candlestick(x=df_index,
                        open=col_open,
                        high=col_high,
                        low=col_low,
                        close=col_close, 
                        increasing_line_color='green',
                        decreasing_line_color='red'
                        )]
                        )
            fig.update_layout(xaxis_rangeslider_visible=False)
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=True)
            fig.show()
            
            
            fig.write_image(pathplot+'_{}.png'.format(s))  



def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier


def round_down(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier



def SaveData(df,ticker,first,seuil):
    end='_'+ticker
    pathplot=pathfichier+'/PLOTS_PAIRES'+end
    DataframeSave=pd.DataFrame()
    DataframeSave2=pd.DataFrame()
    
    
    list_i=[]
    history=[]
    up=1
    down=1
    init=first
    list_init=[]
    list_dif=[]
    list_close=[]
    list_initseuil=[]
    pre_inf=[]
    pre_sup=[]
    bornesup=[]
    borneinf=[]
    
    arrondi=[]
    
    s=0


    
    PATH_TIKKER=ticker
    PATH_TIKKER_HAUSSIER=PATH_TIKKER+'/HAUSSIER'
    PATH_TIKKER_BAISSIER=PATH_TIKKER+'/BAISSIER'
    
    # TICKER_PATH=ticker
    # TICKER_PATH_HAUSSIER=ticker+'/HAUSSIER'
    # TICKER_PATH_BAISSIER=ticker+'/BAISSIER'
    

    if os.path.exists(PATH_TIKKER)==False:
        os.mkdir(PATH_TIKKER)
        os.mkdir(PATH_TIKKER_HAUSSIER)
        os.mkdir(PATH_TIKKER_BAISSIER)
            
    else:
        shutil.rmtree(PATH_TIKKER)
        os.mkdir(PATH_TIKKER)
        os.mkdir(PATH_TIKKER_HAUSSIER)
        os.mkdir(PATH_TIKKER_BAISSIER)
    
    

    for i in range(len(df)):
        
        if df["Close"][i]>init+seuil:  
            
            sup=init+seuil
            inf=init-seuil
            pre_sup.append(sup)
            pre_inf.append(inf)
            
            list_init.append(init)
            dif=df["Close"][i]-init
            y=round_down(df["Close"][i]-init,2)
                 
            
            init=init+y
            sup=init+seuil
            inf=init-seuil
            bornesup.append(sup)
            borneinf.append(inf)
            arrondi.append(y)
            
            DataframeSave2=DataframeSave2.append(df.iloc[i])
           
            list_i.append(i)
            list_dif.append(dif)
            list_close.append(df["Close"][i])
            
            del DataframeSave
            DataframeSave=pd.DataFrame()
            s=s+1
            
            # bornesup.append(list_init+seuil)
            # borneinf.append(list_init-seuil)
               
    
            
        if df["Close"][i]<init-seuil: 
            
            sup=init+seuil
            inf=init-seuil
            pre_sup.append(sup)
            pre_inf.append(inf)
            
            
            list_init.append(init)
            dif=df["Close"][i]-init
            z=round_up(df["Close"][i]-init,2)
  
            
            arrondi.append(z)
            
            init=init+z
            sup=init+seuil
            inf=init-seuil
            
            bornesup.append(sup)
            borneinf.append(inf)
            
            DataframeSave2=DataframeSave2.append(df.iloc[i])
            
            list_i.append(i)
            list_dif.append(dif)
            list_close.append(df["Close"][i])


            del DataframeSave
            DataframeSave=pd.DataFrame()
            s=s+1
            
            
            
            
    DataframeSave2["index_i"] = list_i
    
    # DataframeSave["InitSeuil"] =list_initseuil
    
    DataframeSave2["df Close"] =list_close

   
    
   
    for i in range(len(DataframeSave2)-1):
            
        if DataframeSave2["df Close"][i]<DataframeSave2["df Close"][i+1]:
            history.append('HAUSSIER')
            
        if DataframeSave2["df Close"][i]>DataframeSave2["df Close"][i+1]:
            history.append('BAISSIER')
            
            
    print("longueur de history")        
    print(len(history))        
    print("longueuer dataframe")        
    print(len(DataframeSave))  

    history.append(np.NaN)      
    print("nouvelle longueur de history")        
    print(len(history)) 
            
    
    DataframeSave2["Tendance"] = history
    DataframeSave2["df Close"] =list_close
    DataframeSave2["PRE BorneSup"] =pre_sup
    DataframeSave2["PRE BorneInf"] =pre_inf
    DataframeSave2["POST BorneSup"] = bornesup
    DataframeSave2["POST BorneInf"] = borneinf
    DataframeSave2["Init"] = list_init
    DataframeSave2["Dif"] = list_dif
    DataframeSave2["Arrondie"] = arrondi
    DataframeSave2.to_csv('/Users/lukas/Documents/FICHIER_STRATGIE_TRADING_TEST10/export.csv')
            
    
    index=[]
    
    a=0
    
    print("DF INDEX")
    print(df.index)
    print("DF INDEX APRES RESET")
    df.reset_index(inplace=True, drop=True)
    print(df.index)
    
    
    indexlist=[]    
    indexlist2=[]
    for j in  DataframeSave2["index_i"]:
            indexlist.append(list(range(a,j+1,1)))
            print(index)
            a=j+1
            
    s=0 
    
    for i in indexlist:        
        
            if len(i)<4:
                continue
        
            MiniDataframeSave=pd.DataFrame()
            MiniDataframeSave["Open"]=df["Open"][i]
            MiniDataframeSave["High"]=df["High"][i]
            MiniDataframeSave["Low"]=df["Low"][i]
            MiniDataframeSave["Close"]=df["Close"][i]
            MiniDataframeSave["Index"]=i
            
            MiniDataframeSave=MiniDataframeSave.iloc[len(MiniDataframeSave)-4:len(MiniDataframeSave)]
           
            
            fig = go.Figure(data=[go.Candlestick(x=i,
                          open=MiniDataframeSave["Open"],
                          high=MiniDataframeSave["High"],
                          low=MiniDataframeSave["Low"],
                          close=MiniDataframeSave["Close"], 
                          increasing_line_color='green',
                          decreasing_line_color='red'
                          )]
                          )
            
            fig.update_layout(xaxis_rangeslider_visible=False)
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            
                   
       
           # LE PROBLEME EST QUE LE S EST SCOTCHER SUR 0 
            print("LA VALEUR DE S AVANT")
            print(s) 
       
            print(DataframeSave2["Tendance"][s])
            
            
            tick='{}'.format(ticker)
            print(type(tick))
            num=str('{}'.format(s))
            print(type(num))
            pathfichierpng='/Users/lukas/Documents/FICHIER_STRATGIE_TRADING_TEST10/FUSION_DES_DONNEES/HAUSSIER/'+tick+num
            print(pathfichierpng)
            
            
            
            
            
            
            
            if DataframeSave2["Tendance"][s]=='HAUSSIER' and s!=0 and s!=len(DataframeSave2):
                fig.write_image(PATH_TIKKER_HAUSSIER+'/'+tick+num+'.png')
                         
                
            if DataframeSave2["Tendance"][s]=='BAISSIER' and s!=0 and s!=len(DataframeSave2):
                fig.write_image(PATH_TIKKER_BAISSIER+'/'+tick+num+'.png')         
    
            s=s+1
       
            
      


def RedirectData(ticker):
    TICKER_PATH=ticker
    TICKER_PATH_HAUSSIER=ticker+'/HAUSSIER'
    TICKER_PATH_BAISSIER=ticker+'/BAISSIER'
    
    listfichehaussier=os.listdir(TICKER_PATH_HAUSSIER)
    listfichebaissier=os.listdir(TICKER_PATH_BAISSIER)


    len_haussier=len(os.listdir(TICKER_PATH_HAUSSIER))
    len_baissier=len(os.listdir(TICKER_PATH_BAISSIER))
    
    print("NOMBRE DE FICHIER HAUSSIER DE "+ticker+" EST DE " +str(len_haussier))
    print("  ")
    print("NOMBRE DE FICHIER BAISSIER DE "+ticker+" EST DE " +str(len_baissier))
    print("  ")
    
    if len_haussier>len_baissier:
        len_standart=len_baissier
    if len_haussier<len_baissier:
        len_standart=len_haussier
    if len_haussier==len_baissier:
        len_standart=len_baissier
    
      
            
    num_trainfile=ceil(len_standart*0.72)  
    num_validfile=ceil(len_standart*0.18) 
    num_testfile= ceil(len_standart*0.04) 
    

    subset_haussier=random.sample(listfichehaussier, num_trainfile)
    subset_baissier=random.sample(listfichebaissier, num_trainfile)
    
    
    for i in subset_haussier:
        file_oldname=TICKER_PATH_HAUSSIER+'/'+i
        file_newname=trainpathhaussier+'/'+i
        os.rename(file_oldname, file_newname)
            
    for i in subset_baissier:  
        file_oldname=TICKER_PATH_BAISSIER+'/'+i
        file_newname=trainpathbaissier+'/'+i
        os.rename(file_oldname, file_newname)
                
    listfichehaussier=os.listdir(TICKER_PATH_HAUSSIER)
    listfichebaissier=os.listdir(TICKER_PATH_BAISSIER)
    
    subset_haussier=random.sample(listfichehaussier, num_validfile)
    subset_baissier=random.sample(listfichebaissier, num_validfile)
    
    

    

    for i in subset_haussier:
        file_oldname=TICKER_PATH_HAUSSIER+'/'+i
        file_newname=validpathhaussier+'/'+i
        os.rename(file_oldname, file_newname)
        
    for i in subset_baissier:  
        file_oldname=TICKER_PATH_BAISSIER+'/'+i
        file_newname=validpathbaissier+'/'+i
        os.rename(file_oldname, file_newname)            
            
             
           
    listfichehaussier=os.listdir(TICKER_PATH_HAUSSIER)
    listfichebaissier=os.listdir(TICKER_PATH_BAISSIER)
            
    subset_haussier=random.sample(listfichehaussier, num_testfile)
    subset_baissier=random.sample(listfichebaissier, num_testfile)

    for i in subset_haussier:
        file_oldname=TICKER_PATH_HAUSSIER+'/'+i
        file_newname=testpathhaussier+'/'+i
        os.rename(file_oldname, file_newname)
        
    for i in subset_baissier:  
        file_oldname=TICKER_PATH_BAISSIER+'/'+i
        file_newname=testpathbaissier+'/'+i
        os.rename(file_oldname, file_newname)            
            


for row in dc.itertuples():
      #StoreData(row.ticker)
      df=GetData(row.ticker)
      first=df['Close'][0]

      print(first)
      first_round=round(first, 2)
      print(first_round)
      #pathfichier='myfile.csv'
      SaveData(df,row.ticker,first_round,row.seuil)
      RedirectData(row.ticker)
      
      
      
      
      
    