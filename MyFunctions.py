import pandas as pd
import base64
import requests
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import altair as alt
#Jaredstock code
#import SessionState
from typing import Tuple
import mplfinance as fplt
##from sklearn.ensemble import RandomForestClassifier
##from sklearn.datasets import make_classification
##from sklearn.model_selection import train_test_split
##from sklearn import metrics
##from sklearn import preprocessing
from datetime import datetime, timedelta
import itertools


start_date = (datetime.now() - timedelta(30))
fund_list = ['P/L', 'P/VP', 'PSR' ,'Div.Yield', 'ROIC', 'ROE', 'Cresc. Rec.5a', 'Mrg Ebit', 'Mrg. Líq.' ]

NADAQ_list = sorted([
         'ebitdaMargins',
         'profitMargins',
         'revenueGrowth',
         'operatingMargins',
         'returnOnAssets',
    #     'recommendationKey',
         'recommendationMean',
         'debtToEquity',
         'totalCashPerShare',
         'quickRatio',
         'enterpriseToEbitda',
         'enterpriseToRevenue',
         'bookValue',
         'earningsQuarterlyGrowth',
         'forwardPE'
         ])

#Un a random forest model
###
def random_forest_forecast(ticker):
    # get ticker
    tickerData = yf.Ticker(ticker)
    # take one year of historical data
    tickerDF = tickerData.history(period = '1d',
                              start = (datetime.now() - timedelta(365)).strftime('%Y-%m-%d'),
                              end = (datetime.now() + timedelta(3)).strftime('%Y-%m-%d'))

    # Creating lags for close and volume
    lista = ['Close', 'Volume']
    for c in range(1,4):
        a = str(c)
        #b = "Close_lag_"+ a
        for i in lista:
            d = str(i)
            b = d+"_lag_"+ a
            tickerDF[b] = tickerDF[i].shift(c)

    #creating shifted rolling mean
    tickerDF['rolling_avg'] = tickerDF['Close'].rolling(7).mean()
    tickerDF['rolling_avg'] = tickerDF['rolling_avg'].shift(1)

    #creating a target variable [0,1]. is weird but works that way
    tickerDF['target'] = tickerDF['Close'] - tickerDF['Close_lag_1']
    tickerDF['target'].where(tickerDF['target'] >= 0, 0, inplace=True)
    tickerDF['target'].where(tickerDF['target'] < 0.000000000000001, 1, inplace=True)

    # creat a list of all columns except target
    hs = list(tickerDF.columns)
    hs.remove('target')

    # take the trend out of the time series
    for i in hs:
        tickerDF[i] = tickerDF[i].diff(1)

    # droping na values
    tickerDF = tickerDF.dropna()

    # substitute the values for standarzided values
    min_max_scaler = preprocessing.MinMaxScaler()
    x = tickerDF.values #returns a numpy array
    x_scaled = min_max_scaler.fit_transform(x)
    df_x_scaled = pd.DataFrame(x_scaled)
    df_x_scaled.index = tickerDF.index
    z = 0
    for i in hs:
        tickerDF[i] = df_x_scaled[z]
        z = z+1

    # drop nonused columns
    tickerDF = tickerDF.drop(columns = ['Dividends', 'Stock Splits'])

    # remove from the list columsn that cant/wont be used for X
    hs.remove('Open')
    hs.remove('High')
    hs.remove('Low')
    hs.remove('Dividends')
    hs.remove('Stock Splits')
    hs.remove('Volume')
    hs.remove('Close')

    #Create a list of all possible combinations of X variables [x1], [x1, x2], [x1, x3]....
    lst = hs
    combs = []

    for i in range(1, len(lst)+1):
        els = [list(x) for x in itertools.combinations(lst, i)]
        combs.extend(els)

    # check which X combination makes the best model and chose it
    best_metrics = 0
    best_list = []
    y = tickerDF['target']

    count = len(combs)
    for i in combs:
        hs = list(i)
        X = tickerDF[hs]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70, shuffle = False)
        clf=RandomForestClassifier(n_estimators = 100, random_state=7)
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        count = count -1
        print(count)

        if metrics.accuracy_score(y_test, y_pred) > best_metrics:
            best_metrics = metrics.accuracy_score(y_test, y_pred)
            best_list = hs

    # build the model with the best X
    X = tickerDF[best_list]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70, shuffle = False)
    clf=RandomForestClassifier(n_estimators = 100, random_state=7)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)

    # take model metrics
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
    sensibilidade = tp/(tp + fp)
    especificidade = tn/(tn + fn)
    precisao = metrics.accuracy_score(y_test, y_pred)

    #Substitute the list found in order to update the values for future prediction
    abc_array = np.array(best_list)
    transdict = {'Close_lag_1': 'Close',
             'Close_lag_2': 'Close_lag_1',
             'Close_lag_3': 'Close_lag_2',
             'Close_lag_4': 'Close_lag_3',
             'Close_lag_5': 'Close_lag_4',
             'Close_lag_6': 'Close_lag_5',
             'Close_lag_7': 'Close_lag_6',
             'Volume_lag_1': 'Volume',
             'Volume_lag_2': 'Volume_lag_1',
             'Volume_lag_3': 'Volume_lag_2',
             'Volume_lag_4': 'Volume_lag_3',
             'Volume_lag_5': 'Volume_lag_4',
             'Volume_lag_6': 'Volume_lag_5',
             'Volume_lag_7': 'Volume_lag_6',
             'rolling_avg':'rolling_avg'}
    phoenetic = [transdict[letter] for letter in abc_array]


    #predict if the stock will go up or down
    new_pred = tickerDF.iloc[0:1][phoenetic]
    new_y_pred=clf.predict(new_pred)
    futuro = ''
    if new_y_pred[0] == 0:
        futuro = "Down"
    else:
        futuro = "Up"


    return sensibilidade, especificidade, precisao, futuro

# fucntion to make the data downloadable(dataprofessor)
def filedownload(df, df_name):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="{df_name}">Download CSV File</a>'
    return href

#function to transform number into date : not being used
def num_to_time(date):
    try:
        timestamp = datetime.fromtimestamp(date).strftime('%Y-%m-%d')

    except:
        timestamp = "No info"

    return timestamp

#create lists  for the fundamentals dataframe
def fundlist(lista, df_selected_sector):
    fund_list_z = []
    fund_list_s_z = []
    fund = []
    z_score = []
    s_zscore = []


    for i in lista:
        i = i + '_zscore'
        fund_list_z.append(i)


    for i in lista:
        i = i + '_s_zscore'
        fund_list_s_z.append(i)

    for i in lista:
            fund.append(df_selected_sector[i].unique()[0])

    for i in fund_list_z:
            z_score.append(df_selected_sector[i].unique()[0])

    for i in fund_list_s_z:
            s_zscore.append(df_selected_sector[i].unique()[0])



    return fund, z_score, s_zscore

#calculate de zscore of the fundamentals
def zscore(df, sector = 'Setor'):

    #change de 0 for NA so it doesnt calculate non existing values
    #for i in df.columns:
        #df[i].replace(0, np.nan, inplace=True)
    a = df.select_dtypes(include='number')
    colist = list(a.columns)

    #remove string columns
    #remov = ['Setor', 'Subsetor', 'Nome', 'Papel','Ticker', "Data_ultimo_balanco"]
    #for i in remov:
        #colist.remove(i)

    #calculate de overall zscore for every non string variable
    for col in colist:
        col_zscore = col + '_zscore'
        df[col_zscore] = (df[col] - df[col].mean())/df[col].std(ddof=0)

    #calculate the zscore for every sector
    for col in colist:
        col_s_zscore = col + '_s_zscore'
        df[col_s_zscore] = np.nan
        for setor in list(df[sector].unique()):
                mask = df[sector] == setor
                df.loc[mask, col_s_zscore] = (df.loc[mask, col] -  df.loc[mask, col].mean())/ df.loc[mask, col].std(ddof=0)

    return df

# create a candletic chart using mtfploty
def statick_candlestick(tickerDF):



        mc = fplt.make_marketcolors(
                                up='tab:green',down='tab:red',
                                wick={'up':'green','down':'red'},
                                volume='blue'
                               )



        s = fplt.make_mpf_style(base_mpl_style="seaborn",marketcolors=mc, mavcolors=["black"])



        fig = fplt.plot(
                    tickerDF,
                    type='candle',
                    style=s,
                    ylabel='Price',
                    volume=True,
                    ylabel_lower='Shares\nTraded',
                    figratio = (12,8),
                    #mav=2
                    )
        return fig

# Create a candlestick layers using altair chart : not being used
def candlestick(tickerDF: pd.DataFrame) -> Tuple[alt.Chart, alt.Chart, alt.Chart]:

    # altair chart needs a date colum, it seems not work with the index
    tickerDF['Date'] = tickerDF.index

    open_close_color = alt.condition("datum.Open < datum.Close",
                      alt.value('#06982d'),
                      alt.value('#ae1325'))

    base = alt.Chart(tickerDF, width = 530).encode(x = 'Date')

    rule = base.mark_rule().encode(
        y = alt.Y(
           'Low',
            scale = alt.Scale(zero = False),
            axis = alt.Axis(title = 'Price')
        ),
        y2 = alt.Y2('High'),
        color = open_close_color
    )

    bar = base.mark_bar().encode(
        y = 'Open',
        y2 = 'Close',
        color =open_close_color
    )

    volume: alt.chart = base.properties(height = 100).mark_bar().encode(
        y = alt.Y(
            'Volume',
            scale = alt.Scale(zero = False)
        )
    )

    return rule, bar, volume

# combining the layers to plot the chart: not being used
def int_chart(rule: alt.Chart, bar: alt.Chart, volume: alt.Chart, method = 0) -> alt.VConcatChart:
    if method == 0:
        combined =  ((rule+bar).interactive()&volume).resolve_scale(x = 'shared')
    else:
        scales = alt.selection_interval(bind='scales')
        cadlesticks = rule.add_selection(scales) + bar

        combined = (cadlesticks & volume).resolve_scale(x = 'shared')

        return combined

# read the first table of an URL
def load_data(url, thousands = ',', decimal = '.'):
    header = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36","X-Requested-With": "XMLHttpRequest"}
    url = url
    r = requests.get(url, headers = header)
    html = pd.read_html(r.text, thousands=thousands, decimal=decimal)
    df = html[0]
    return df

# Read the fundamentus website table in order to fech fundamentals data from BM&FBOVESPA
def load_fundamentus():

    lista = ['Div.Yield', 'ROIC', 'ROE', 'Cresc. Rec.5a', 'Mrg Ebit', 'Mrg. Líq.' ]
    fund = load_data('https://www.fundamentus.com.br/resultado.php', '.', ',')
    lista_nomes = pd.read_csv('df_b3.csv', encoding = 'UTF-8')[['Setor','Subsetor',"Nome",'Papel', 'Data_ultimo_balanco', 'Ticker']]
    df_b3 = pd.merge(lista_nomes, fund, on="Papel")
    for i in lista:
        df_b3[i] = df_b3.apply(lambda x: x[i].replace('.', ""),axis=1)
        df_b3[i] = df_b3.apply(lambda x: x[i].replace(',', "."),axis=1)
        df_b3[i] = df_b3.apply(lambda x: x[i].replace("%", ""),axis=1)
        df_b3[i] = df_b3[i].astype(float)
        df_b3[i] = df_b3[i]/100

    return df_b3


