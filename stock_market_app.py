#from sklearn import metrics
import streamlit as st
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
import MyFunctions
from MyFunctions import fund_list, random_forest_forecast
from MyFunctions import NADAQ_list
start_date = (datetime.now() - timedelta(30))

# dont show # warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# Set page configuration
st.set_page_config(
        page_title="Easy Stock Exchange Data",
        page_icon="📈",
        initial_sidebar_state="expanded")

#@st.cache(suppress_st_warning=True)

# Set Tittle
st.title('Stock Exchange Markets app (BM&FBOVESPA, Euronext, SIX, NASDAQ)')

# Inroduction
st.sidebar.title('Stock Exchange Markets app')
st.sidebar.markdown("""
This app retrieves the list of the papers from BM&FBOVESPA, Euronext, SIX and NASDAQ displaying them in a easy and downloadable way
* **Data source:** [Fundamentus](https://www.fundamentus.com.br/), [NASDAQ](https://www.nasdaq.com/), [Euronext](https://www.euronext.com/), [SIX](https://www.six-group.com/en/products-services/the-swiss-stock-exchange.html), [Yahoo finance](https://finance.yahoo.com/).
* **Created by:** [Thiago Borba](https://www.linkedin.com/in/thiago-da-silva-borba-407351123/)
""")

# create a diferente page for every stock market
if 'market' not in st.session_state:
    st.session_state.market = 'Euronext'

#st.sidebar.title('Stock Exchange Market')
# stock market slide box
st.session_state.market = st.sidebar.selectbox('Exchange Market', ['BM&FBOVESPA', 'Euronext', 'NASDAQ', 'SIX'])

if st.session_state.market == 'BM&FBOVESPA':

    # upload the b3 table
    df = MyFunctions.load_fundamentus()

    # put the data to be downloaded
    df_name = "BM&FBOVESPA.csv"
    st.sidebar.markdown(MyFunctions.filedownload(df, df_name), unsafe_allow_html=True)

    # take the list of Sector to choose
    sorted_sector_unique = sorted(df['Setor'].unique())
    
    # Sector side bar
    st.session_state.data_type = st.sidebar.multiselect("Sector:", sorted_sector_unique, "Agropecuária")

    #take the list of subsector to choose
    sorted_subsector_unique = sorted(df[ (df['Setor'].isin(st.session_state.data_type)) ]['Subsetor'].unique())
    st.session_state.data_type = st.sidebar.multiselect("Industry:", sorted_subsector_unique,sorted_subsector_unique)

    #take the list of companies to choose
    sorted_company_unique =  sorted(df[ (df['Subsetor'].isin(st.session_state.data_type)) ]['Nome'].unique())
        ##make a dataframe for further download
        ##selection_dataframe = df[ (df['Subsetor'].isin(st.session_state.data_type))]
    #companies list continuation
    st.session_state.data_type = st.sidebar.selectbox("Company:", sorted_company_unique)

    #radio boxes for the papers - make a list to work on the isin function(a bit of silvertape)
    lista = [st.session_state.data_type]
    papers = sorted(df[ (df['Nome'].isin(lista)) ]['Papel'].unique())
    st.session_state.data_type = st.sidebar.radio("Papel:", papers)

    # get the Ticker from the data frame
    lista = [st.session_state.data_type]
    df_selected_sector = df[ (df['Papel'].isin(lista)) ]
    tickerSymbol = df_selected_sector['Ticker'].unique()[0]

    #get data on this ticker
    tickerData = yf.Ticker(tickerSymbol)


    #taking pieces of information about the selected ticker
    nome = df_selected_sector['Nome'].unique()[0]
    papel = df_selected_sector['Papel'].unique()[0]
    setor = df_selected_sector['Setor'].unique()[0]
    subsetor = df_selected_sector['Subsetor'].unique()[0]
    data_b = df_selected_sector['Data_ultimo_balanco'].unique()[0]
    pl = df_selected_sector['P/L'].unique()[0]
    pl_zscore = df_selected_sector['P/L_zscore'].unique()[0]
    pl_s_zscore = df_selected_sector['P/L_s_zscore'].unique()[0]
    pv = df_selected_sector['P/L'].unique()[0]

    #building the lists for the fundamentals dataframe creation
    fund, zscore , s_zscore = MyFunctions.fundlist(fund_list)

    #creating the fundamentals dataframe
    hue = pd.DataFrame(columns = fund_list, index = ['value', 'zscore', 'Setorial_zscore'])
    hue.iloc[0] = fund
    hue.iloc[1] = zscore
    hue.iloc[2] = s_zscore

    st.write(f""" # {nome} - {papel} """)

    st.write(f"""
                Sector: {setor} \n
               Subsetor: {subsetor}   """)
    st.write(f"""## Fundamentals: {data_b} """ )
    st.write("""In statistics, the **z-score** is the number of standard deviations by which the value of a raw score
                    (i.e., an observed value or data point) is above or below the mean value of what is being observed or measured.
                    Raw scores above the mean have positive standard scores,
                    while those below the mean have negative standard scores. [Wikipedia](https://en.wikipedia.org/wiki/Standard_score)""")
    #display fundamentals dataframe
    hue


    st.write('## Historical data')

    # user input historical prices for this ticker
    start = st.date_input ( "Start date" , value=start_date , min_value=None , max_value=None , key=None )
    end = st.date_input( "End date" , value=None , min_value=None , max_value=None , key=None )

    # fetch historical prices for this ticker
    tickerDF = tickerData.history(period = '1d', start = start, end = end)

    # historical prices download
    df_name = f"{papel}.csv"
    st.markdown(MyFunctions.filedownload(tickerDF, df_name), unsafe_allow_html=True)

        ## interactive candlestick

            #tickerDF['Date'] = tickerDF.index

            #rule, bar, volume = candlestick(tickerDF)

            #combined =  ((rule+bar).interactive()&volume).resolve_scale(x = 'shared')

            #st.altair_chart(combined, use_container_width=True)##

    # Display static candlestick
    try:
        fig = MyFunctions.statick_candlestick(tickerDF)
        st.pyplot(fig = fig)
        #tickerDF
    except:
        st.write('Data not found, check the dates')

    if st.button('predict Up or Down'):
        try:
            sensibilidade, especificidade, precisao, futuro = random_forest_forecast(tickerSymbol)
            st.write(f"""## {futuro}
            * Accuracy: {precisao:.2f}
            * Specificity: {especificidade:.2f}
            * Sensibility: {sensibilidade:.2f}
            """)
        except:
            st.write('Not enough data')
    else:
        st.write('it may take several minutes')

elif st.session_state.market == 'NASDAQ':
    df = pd.read_csv('NASDAQ.csv')
    df = MyFunctions.zscore(df, "Industry")


    #adjusting the name of columns to fit the code
    df['Ticker'] = df['Symbol']
    df['Sector'] = df['Sector'].replace({"":"Not found"})
    df['Industry'] = df['Industry'].replace({"":"No info"})
    df['Sector'] = df['Sector'].replace({"":"Not found"})
    df["Industry"] = df.apply(lambda x: str(x["Industry"]),axis=1)
    df["Sector"] = df.apply(lambda x: str(x["Sector"]),axis=1)

    df = df.drop(columns = 'Unnamed: 0')
    df_name = "NASDAQ.csv"
    st.sidebar.markdown(MyFunctions.filedownload(df, df_name), unsafe_allow_html=True)


    sorted_sector_unique = sorted(df['Sector'].unique())
    st.session_state.data_type = st.sidebar.multiselect("Sector:", sorted_sector_unique, sorted_sector_unique[0])

    #take the list of subsector to choose
    sorted_subsector_unique = sorted(df[ (df['Sector'].isin(st.session_state.data_type)) ]['Industry'].unique())
    st.session_state.data_type = st.sidebar.multiselect("Industry:", sorted_subsector_unique, sorted_subsector_unique[0])

    #take the list of companies to choose
    sorted_company_unique =  sorted(df[ (df['Industry'].isin(st.session_state.data_type)) ]['longName'].unique())
     #$make a dataframe for further download
     ##selection_dataframe = df[ (df['Industry'].isin(st.session_state.data_type))]
    #companies list continuation
    st.session_state.data_type = st.sidebar.selectbox("Name:", sorted_company_unique)

    #radio boxes for the papers - make a list to work on the isin function(a bit of silvertape)
    lista = [st.session_state.data_type]
    papers = sorted(df[ (df['longName'].isin(lista)) ]['Symbol'].unique())
    st.session_state.data_type = st.sidebar.radio("Paper:", papers)

             # get the Ticker from the data frame
    lista = [st.session_state.data_type]
    df_selected_sector = df[ (df['Symbol'].isin(lista)) ]

    nome = df_selected_sector['longName'].unique()[0]
    papel = df_selected_sector['Symbol'].unique()[0]
    setor = df_selected_sector['Sector'].unique()[0]
    subsetor = df_selected_sector['Industry'].unique()[0]
    quarter = df_selected_sector['mostRecentQuarter'].unique()[0]


    tickerSymbol = df_selected_sector['Ticker'].unique()[0]
    tickerData = yf.Ticker(tickerSymbol)

    fund, zscore , s_zscore = MyFunctions.fundlist(NADAQ_list)
    hue = pd.DataFrame(columns = NADAQ_list, index = ['value', 'zscore', 'Industry_zscore'])
    hue.iloc[0] = fund
    hue.iloc[1] = zscore
    hue.iloc[2] = s_zscore


    st.write(f""" # {nome} - {papel} """)

    st.write(f"""
            Sector: {setor} \n
            Subsetor: {subsetor}   """)


    st.write(f"""## Fundamentals: {quarter} """ )
    st.write("""In statistics, the **z-score** is the number of standard deviations by which the value of a raw score
                    (i.e., an observed value or data point) is above or below the mean value of what is being observed or measured.
                    Raw scores above the mean have positive standard scores,
                    while those below the mean have negative standard scores. [Wikipedia](https://en.wikipedia.org/wiki/Standard_score)""")
    hue


    st.write('## Historical data')
    start = st.date_input ( "Start date" , value=start_date , min_value=None , max_value=None , key=None )
    end = st.date_input( "End date" , value=None , min_value=None , max_value=None , key=None )
    tickerDF = tickerData.history(period = '1d', start = start, end = end)

    df_name = f"{papel}.csv"
    st.markdown(MyFunctions.filedownload(tickerDF, df_name), unsafe_allow_html=True)

    try:
        fig = MyFunctions.statick_candlestick(tickerDF)
        st.pyplot(fig = fig)
        #tickerDF
    except:
        st.write('Data not found, check the dates')

    if st.button('predict Up or Down'):
        try:
            sensibilidade, especificidade, precisao, futuro = random_forest_forecast(tickerSymbol)
            st.write(f"""## {futuro}
            * Accuracy: {precisao:.2f}
            * Specificity: {especificidade:.2f}
            * Sensibility: {sensibilidade:.2f}
            """)
        except:
            st.write('Not enough data')
    else:
        st.write('it may take several minutes')

elif st.session_state.market == 'SIX':


    df = pd.read_csv('SIX.csv')
    df = MyFunctions.zscore(df, "sector")

    #adjusting columns names
    df = df.rename(columns={"industry": "Industry", "ticker": "Ticker", 'Company': 'Name', 'sector': 'Sector'})
    df = df.drop(columns = 'Unnamed: 0')

    #making data downloadable
    df_name = "SIX.csv"
    st.sidebar.markdown(MyFunctions.filedownload(df, df_name), unsafe_allow_html=True)

    #removing na and making sure that Sector and Industry are stings
    df['Sector'] = df['Sector'].replace({"":"Not found"})
    df['Industry'] = df['Industry'].replace({"":"Not found"})
    df["Industry"] = df.apply(lambda x: str(x["Industry"]),axis=1)
    df["Sector"] = df.apply(lambda x: str(x["Sector"]),axis=1)


    #take the list of sectors to choose
    sorted_sector_unique = sorted(df['Sector'].unique())
    st.session_state.data_type = st.sidebar.multiselect("Sector:", sorted_sector_unique, sorted_sector_unique[0])

     #take the list of industries to choose
    sorted_subsector_unique = sorted(df[ (df['Sector'].isin(st.session_state.data_type)) ]['Industry'].unique())
    st.session_state.data_type = st.sidebar.multiselect("Industry:", sorted_subsector_unique, sorted_subsector_unique[0])

     #take the list of companies to choose
    sorted_company_unique =  sorted(df[ (df['Industry'].isin(st.session_state.data_type)) ]['Name'].unique())
          ##make a dataframe for further download
          ##selection_dataframe = df[ (df['Industry'].isin(st.session_state.data_type))]
     #companies list continuation
    st.session_state.data_type = st.sidebar.selectbox("Name:", sorted_company_unique)

    #radio boxes for the papers - make a list to work on the isin function(a bit of silvertape)
    lista = [st.session_state.data_type]
    papers = sorted(df[ (df['Name'].isin(lista)) ]['Symbol'].unique())
    st.session_state.data_type = st.sidebar.radio("Paper:", papers)

     # get the Ticker from the data frame
    lista = [st.session_state.data_type]
    df_selected_sector = df[ (df['Symbol'].isin(lista)) ]


    nome = df_selected_sector['Name'].unique()[0]
    papel = df_selected_sector['Symbol'].unique()[0]
    setor = df_selected_sector['Sector'].unique()[0]
    country = df_selected_sector['Country'].unique()[0]
    class_share = df_selected_sector['Class of Share'].unique()[0]
    currency = df_selected_sector['Traded Currency'].unique()[0]
    quarter = df_selected_sector['mostRecentQuarter'].unique()[0]
    subsetor = df_selected_sector['Industry'].unique()[0]



    tickerSymbol = df_selected_sector['Ticker'].unique()[0]

    tickerData = yf.Ticker(tickerSymbol)

    fund, zscore , s_zscore = MyFunctions.fundlist(NADAQ_list)


    hue = pd.DataFrame(columns = NADAQ_list, index = ['value', 'zscore', 'sector_zscore'])
    hue.iloc[0] = fund
    hue.iloc[1] = zscore
    hue.iloc[2] = s_zscore


    st.write(f""" # {nome} - {papel} """)

    st.write(f"""
                Sector: {setor} \n
               Subsetor: {subsetor}\n
               Country: {country}\n
               Traded currency: {currency}\n
               Class of share: {class_share}\n """)
    st.write(f"""## Fundamentals: {quarter} """ )
    st.write("""In statistics, the **z-score** is the number of standard deviations by which the value of a raw score
                    (i.e., an observed value or data point) is above or below the mean value of what is being observed or measured.
                    Raw scores above the mean have positive standard scores,
                    while those below the mean have negative standard scores. [Wikipedia](https://en.wikipedia.org/wiki/Standard_score)""")
    hue


    st.write('## Historical data')


    start = st.date_input ( "Start date" , value=start_date , min_value=None , max_value=None , key=None )

    end = st.date_input( "End date" , value=None , min_value=None , max_value=None , key=None )

    tickerDF = tickerData.history(period = '1d', start = start, end = end)

    df_name = f"{papel}.csv"
    st.markdown(MyFunctions.filedownload(tickerDF, df_name), unsafe_allow_html=True)

    try:
        fig = MyFunctions.statick_candlestick(tickerDF)
        st.pyplot(fig = fig)
        #tickerDF
    except:
        st.write('Data not found, check the dates')

    if st.button('predict Up or Down'):
        try:
            sensibilidade, especificidade, precisao, futuro = random_forest_forecast(tickerSymbol)
            st.write(f"""## {futuro}
            * Accuracy: {precisao:.2f}
            * Specificity: {especificidade:.2f}
            * Sensibility: {sensibilidade:.2f}
            """)
        except:
            st.write('Not enough data')
    else:
        st.write('it may take several minutes')

elif st.session_state.market == 'Euronext':

    df = pd.read_csv('Euronext.csv', encoding = 'UTF-8' )

    # "translating" some letters to UTF-8
    for i in list(df.select_dtypes(include='object').columns):
        df[i] = df.apply(lambda x : str(x[i]).replace('?','ø').replace('??','Ø'), axis = 1)


    df = MyFunctions.zscore(df, "sector")

    #adjusting the columns name for the code
    df = df.rename(columns={"industry": "Industry", "ticker": "Ticker", 'Company': 'Name', 'sector': 'Sector'})
    df = df.drop(columns = 'Unnamed: 0')
    #make the data downloadable
    df_name = "Euronext.csv"
    st.sidebar.markdown(MyFunctions.filedownload(df, df_name), unsafe_allow_html=True)

    #replacing NA and making sure that industry and Sector are strings
    df['Sector'] = df['Sector'].replace({"":"Not found"})
    df['Industry'] = df['Industry'].replace({"":"Not found"})
    df["Industry"] = df.apply(lambda x: str(x["Industry"]),axis=1)
    df["Sector"] = df.apply(lambda x: str(x["Sector"]),axis=1)

    #sort by city
    sorted_sector_unique = sorted(df['City'].unique())
    st.session_state.data_type = st.sidebar.multiselect("City:", sorted_sector_unique, sorted_sector_unique)


    sorted_subsector_unique = sorted(df[ (df['City'].isin(st.session_state.data_type)) ]['Sector'].unique())
     
    # make sure that the filter only takes the choosen cities 
    df = df[ (df['City'].isin(st.session_state.data_type)) ]
         
    st.session_state.data_type = st.sidebar.multiselect("Sector:", sorted_subsector_unique, sorted_subsector_unique[0])


    #take the list of subsector to choose
    sorted_subsector_unique = sorted(df[ (df['Sector'].isin(st.session_state.data_type)) ]['Industry'].unique())
    st.session_state.data_type = st.sidebar.multiselect("Industry:", sorted_subsector_unique, sorted_subsector_unique)

    #take the list of companies to choose
    sorted_company_unique =  sorted(df[ (df['Industry'].isin(st.session_state.data_type)) ]['Name'].unique())
      ##selection_dataframe = df[ (df['Industry'].isin(st.session_state.data_type))]
    #companies list continuation
    st.session_state.data_type = st.sidebar.selectbox("Name:", sorted_company_unique)

    #radio boxes for the papers - make a list to work on the isin function(a bit of silvertape)
    lista = [st.session_state.data_type]
    papers = sorted(df[ (df['Name'].isin(lista)) ]['Symbol'].unique())
    st.session_state.data_type = st.sidebar.radio("Paper:", papers)

    # get the Ticker from the data frame
    lista = [st.session_state.data_type]
    df_selected_sector = df[ (df['Symbol'].isin(lista)) ]


    nome = df_selected_sector['Name'].unique()[0]
    papel = df_selected_sector['Symbol'].unique()[0]
    setor = df_selected_sector['Sector'].unique()[0]
    country = df_selected_sector['City'].unique()[0]
    class_share = df_selected_sector['Market'].unique()[0]
    currency = df_selected_sector['Trading Currency'].unique()[0]
    quarter = df_selected_sector['mostRecentQuarter'].unique()[0]
    subsetor = df_selected_sector['Industry'].unique()[0]


    tickerSymbol = df_selected_sector['Ticker'].unique()[0]

    tickerData = yf.Ticker(tickerSymbol)

    fund, zscore , s_zscore = MyFunctions.fundlist(NADAQ_list)


    hue = pd.DataFrame(columns = NADAQ_list, index = ['value', 'zscore', 'sector_zscore'])
    hue.iloc[0] = fund
    hue.iloc[1] = zscore
    hue.iloc[2] = s_zscore

    st.write(f""" # {nome} - {papel} """)

    st.write(f"""
                Sector: {setor} \n
               Subsetor: {subsetor}\n
               City: {country}\n
               Traded currency: {currency}\n
               Market: {class_share}\n """)
    st.write(f"""## Fundamentals: {quarter} """ )
    st.write("""In statistics, the **z-score** is the number of standard deviations by which the value of a raw score
                    (i.e., an observed value or data point) is above or below the mean value of what is being observed or measured.
                    Raw scores above the mean have positive standard scores,
                    while those below the mean have negative standard scores. [Wikipedia](https://en.wikipedia.org/wiki/Standard_score)""")
    hue


    st.write('## Historical data')
    start = st.date_input ( "Start date" , value=start_date , min_value=None , max_value=None , key=None )
    end = st.date_input( "End date" , value=None , min_value=None , max_value=None , key=None )
    tickerDF = tickerData.history(period = '1d', start = start, end = end)
    df_name = f"{papel}.csv"
    st.markdown(MyFunctions.filedownload(tickerDF, df_name), unsafe_allow_html=True)
    try:
        fig = MyFunctions.statick_candlestick(tickerDF)
        st.pyplot(fig = fig)
        #tickerDF
    except:
        st.write('Data not found, check the dates')


    if st.button('predict Up or Down'):
        try:
            sensibilidade, especificidade, precisao, futuro = random_forest_forecast(tickerSymbol)
            st.write(f"""## {futuro}
            * Accuracy: {precisao:.2f}
            * Specificity: {especificidade:.2f}
            * Sensibility: {sensibilidade:.2f}
            """)
        except:
            st.write('Not enough data')
    else:
        st.write('it may take several minutes')
