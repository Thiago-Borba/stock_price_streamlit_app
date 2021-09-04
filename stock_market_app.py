import streamlit as st
import pandas as pd
import base64
import requests
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import altair as alt
#Jaredstock code
import SessionState
from typing import Tuple
import mplfinance as fplt


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


# fucntion to make the data downloadable(dataprofessor)
def filedownload(df, df_name):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="{df_name}">Download CSV File</a>'
    return href




def fundlist(lista):
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


# dont show # warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

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


# Create a candlestick layers using altair chart
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

# combining the layers to plot the chart
def int_chart(rule: alt.Chart, bar: alt.Chart, volume: alt.Chart, method = 0) -> alt.VConcatChart:
    if method == 0:
        combined =  ((rule+bar).interactive()&volume).resolve_scale(x = 'shared')
    else:
        scales = alt.selection_interval(bind='scales')
        cadlesticks = rule.add_selection(scales) + bar

        combined = (cadlesticks & volume).resolve_scale(x = 'shared')

        return combined



@st.cache(suppress_st_warning=True)

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


# Set page configuration
st.set_page_config(
        page_title="Easy Stock Exchange Data",
        page_icon="🏠",
        initial_sidebar_state="expanded")
# Title
st.title('Stock Exchange Markets app (BM&FBOVESPA, Euronext, SIX, NASDAQ)')

# Inroduction
st.markdown("""
This app retrieves the list of the papers from BM&FBOVESPA, Euronext, SIX and NASDAQ displaying them in a easy and downloadable way
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn, Soup
* **Data source:** [Fundamentus](https://www.fundamentus.com.br/), [NASDAQ](https://www.nasdaq.com/), [Euronext](https://www.euronext.com/), [SIX](https://www.six-group.com/en/products-services/the-swiss-stock-exchange.html), [Yahoo finance](https://finance.yahoo.com/).
* **Created by:** [Thiago Borba](https://www.linkedin.com/in/thiago-da-silva-borba-407351123/)
""")

# create a diferente page for every stock market
session_state = SessionState.get(workflow='BM&FBOVESPA')

st.sidebar.title('Stock Exchange Market')
# stock market slide box
session_state.workflow = st.sidebar.selectbox('Exchange Market', ['BM&FBOVESPA', 'Euronext', 'NASDAQ', 'SIX'])

if session_state.workflow == 'BM&FBOVESPA':

    # upload the b3 table
    df = load_fundamentus()
    df = zscore(df)
    #setor = df.groupby('Setor')
    # take the list of Sector to choose

    df_name = "BM&FBOVESPA.csv"
    st.sidebar.markdown(filedownload(df, df_name), unsafe_allow_html=True)

    sorted_sector_unique = sorted(df['Setor'].unique())
    session_state.data_type = st.sidebar.multiselect("Sector:", sorted_sector_unique, "Agropecuária")

    #take the list of subsector to choose
    sorted_subsector_unique = sorted(df[ (df['Setor'].isin(session_state.data_type)) ]['Subsetor'].unique())
    session_state.data_type = st.sidebar.multiselect("Industry:", sorted_subsector_unique,sorted_subsector_unique)

    #take the list of companies to choose
    sorted_company_unique =  sorted(df[ (df['Subsetor'].isin(session_state.data_type)) ]['Nome'].unique())
    #make a dataframe for further download
    selection_dataframe = df[ (df['Subsetor'].isin(session_state.data_type))]
    #companies list continuation
    session_state.data_type = st.sidebar.selectbox("Company:", sorted_company_unique)

    #radio boxes for the papers - make a list to work on the isin function(a bit of silvertape)
    lista = [session_state.data_type]
    papers = sorted(df[ (df['Nome'].isin(lista)) ]['Papel'].unique())
    session_state.data_type = st.sidebar.radio("Papel:", papers)

    # get the Ticker from the data frame
    lista = [session_state.data_type]
    df_selected_sector = df[ (df['Papel'].isin(lista)) ]
    tickerSymbol = df_selected_sector['Ticker'].unique()[0]
    #get data on this ticker

    tickerData = yf.Ticker(tickerSymbol)
    # get historical prices for this ticker

    # Open High Low Close  Volume Dividens Stock Splits
    ##df_selected_sector
    nome = df_selected_sector['Nome'].unique()[0]
    papel = df_selected_sector['Papel'].unique()[0]
    setor = df_selected_sector['Setor'].unique()[0]
    subsetor = df_selected_sector['Subsetor'].unique()[0]
    data_b = df_selected_sector['Data_ultimo_balanco'].unique()[0]
    pl = df_selected_sector['P/L'].unique()[0]
    pl_zscore = df_selected_sector['P/L_zscore'].unique()[0]
    pl_s_zscore = df_selected_sector['P/L_s_zscore'].unique()[0]
    pv = df_selected_sector['P/L'].unique()[0]

    fund, zscore , s_zscore = fundlist(fund_list)

    hue = pd.DataFrame(columns = fund_list, index = ['value', 'zscore', 'Setorial_zscore'])
    hue.iloc[0] = fund
    hue.iloc[1] = zscore
    hue.iloc[2] = s_zscore

    #st.write(f""" ## Selection Table  """)
    #selection_dataframe[['Setor', 'Subsetor', 'Nome', 'Papel' ]]

    st.write(f""" # {nome} - {papel} """)

    st.write(f"""
                Sector: {setor} \n
               Subsetor: {subsetor}   """)
    st.write(f"""## Fundamentals: {data_b} """ )
    hue

    #df_name = "BM&FBOVESPA.csv"
    #st.sidebar.markdown(filedownload(df, df_name), unsafe_allow_html=True)

    st.write('## Historical data')



    #st.write(f""" ## Subsetor: {subsetor} """)

    start = st.date_input ( "Start date" , value=None , min_value=None , max_value=None , key=None )

    end = st.date_input( "End date" , value=None , min_value=None , max_value=None , key=None )

    tickerDF = tickerData.history(period = '1d', start = start, end = end)

    df_name = f"{papel}.csv"
    st.markdown(filedownload(tickerDF, df_name), unsafe_allow_html=True)

    ## interactive candlestick

        #tickerDF['Date'] = tickerDF.index

        #rule, bar, volume = candlestick(tickerDF)

        #combined =  ((rule+bar).interactive()&volume).resolve_scale(x = 'shared')

        #st.altair_chart(combined, use_container_width=True)


    try:
        fig = statick_candlestick(tickerDF)
        st.pyplot(fig = fig)
        tickerDF
    except:
        st.write('Data not found, check the dates')



        #st.write("""
        # Closing Price
        #""")
        #st.line_chart(tickerDF.Close)
        #st.write("""
        # Volume
        #""")
        #st.line_chart(tickerDF.Volume)
        #session_state.data_type = st.sidebar.selectbox("Papel:", ('klabim', 'petrobras'), index=0)

        # Filtering data
        ## df_selected_sector = df['Nome']['session_state.data_type']

elif session_state.workflow == 'NASDAQ':
    df = pd.read_csv('NASDAQ.csv')
    df = zscore(df, "Industry")



    df['Ticker'] = df['Symbol']
    df['Sector'] = df['Sector'].replace({"":"Not found"})
    df['Industry'] = df['Industry'].replace({"":"Not found"})
    df["Industry"] = df.apply(lambda x: str(x["Industry"]),axis=1)
    df["Sector"] = df.apply(lambda x: str(x["Sector"]),axis=1)


    df = df.drop(columns = 'Unnamed: 0')
    df_name = "NASDAQ.csv"
    st.sidebar.markdown(filedownload(df, df_name), unsafe_allow_html=True)

    sorted_sector_unique = sorted(df['Sector'].unique())
    session_state.data_type = st.sidebar.multiselect("Sector:", sorted_sector_unique, sorted_sector_unique[0])

             #take the list of subsector to choose
    sorted_subsector_unique = sorted(df[ (df['Sector'].isin(session_state.data_type)) ]['Industry'].unique())
    session_state.data_type = st.sidebar.multiselect("Industry:", sorted_subsector_unique, sorted_subsector_unique[0])

             #take the list of companies to choose
    sorted_company_unique =  sorted(df[ (df['Industry'].isin(session_state.data_type)) ]['Name'].unique())
             #make a dataframe for further download
    selection_dataframe = df[ (df['Industry'].isin(session_state.data_type))]
             #companies list continuation
    session_state.data_type = st.sidebar.selectbox("Name:", sorted_company_unique)

             #radio boxes for the papers - make a list to work on the isin function(a bit of silvertape)
    lista = [session_state.data_type]
    papers = sorted(df[ (df['Name'].isin(lista)) ]['Symbol'].unique())
    session_state.data_type = st.sidebar.radio("Paper:", papers)

             # get the Ticker from the data frame
    lista = [session_state.data_type]
    df_selected_sector = df[ (df['Symbol'].isin(lista)) ]

    nome = df_selected_sector['Name'].unique()[0]
    papel = df_selected_sector['Symbol'].unique()[0]
    setor = df_selected_sector['Sector'].unique()[0]
    subsetor = df_selected_sector['Industry'].unique()[0]
    data_b = '27/08/2021'


    tickerSymbol = df_selected_sector['Ticker'].unique()[0]

    tickerData = yf.Ticker(tickerSymbol)

    fund, zscore , s_zscore = fundlist(NADAQ_list)


    hue = pd.DataFrame(columns = NADAQ_list, index = ['value', 'zscore', 'Industry_zscore'])
    hue.iloc[0] = fund
    hue.iloc[1] = zscore
    hue.iloc[2] = s_zscore

    st.write(f""" # {nome} - {papel} """)

    st.write(f"""
                Sector: {setor} \n
               Subsetor: {subsetor}   """)
    st.write(f"""## Fundamentals: {data_b} """ )
    hue


    st.write('## Historical data')

    start = st.date_input ( "Start date" , value=None , min_value=None , max_value=None , key=None )

    end = st.date_input( "End date" , value=None , min_value=None , max_value=None , key=None )

    tickerDF = tickerData.history(period = '1d', start = start, end = end)

    df_name = f"{papel}.csv"
    st.markdown(filedownload(tickerDF, df_name), unsafe_allow_html=True)

    try:
        fig = statick_candlestick(tickerDF)
        st.pyplot(fig = fig)
        tickerDF
    except:
        st.write('Data not found, check the dates')

elif session_state.workflow == 'SIX':




    df = pd.read_csv('SIX.csv')
    df = zscore(df, "sector")

    df = df.rename(columns={"industry": "Industry", "ticker": "Ticker", 'Company': 'Name', 'sector': 'Sector'})
    df = df.drop(columns = 'Unnamed: 0')
    df_name = "SIX.csv"
    st.sidebar.markdown(filedownload(df, df_name), unsafe_allow_html=True)

    df['Sector'] = df['Sector'].replace({"":"Not found"})
    df['Industry'] = df['Industry'].replace({"":"Not found"})
    df["Industry"] = df.apply(lambda x: str(x["Industry"]),axis=1)
    df["Sector"] = df.apply(lambda x: str(x["Sector"]),axis=1)

    sorted_sector_unique = sorted(df['Sector'].unique())
    session_state.data_type = st.sidebar.multiselect("Sector:", sorted_sector_unique, sorted_sector_unique[0])

             #take the list of subsector to choose
    sorted_subsector_unique = sorted(df[ (df['Sector'].isin(session_state.data_type)) ]['Industry'].unique())
    session_state.data_type = st.sidebar.multiselect("Industry:", sorted_subsector_unique, sorted_subsector_unique[0])

             #take the list of companies to choose
    sorted_company_unique =  sorted(df[ (df['Industry'].isin(session_state.data_type)) ]['Name'].unique())
             #make a dataframe for further download
    selection_dataframe = df[ (df['Industry'].isin(session_state.data_type))]
             #companies list continuation
    session_state.data_type = st.sidebar.selectbox("Name:", sorted_company_unique)

             #radio boxes for the papers - make a list to work on the isin function(a bit of silvertape)
    lista = [session_state.data_type]
    papers = sorted(df[ (df['Name'].isin(lista)) ]['Symbol'].unique())
    session_state.data_type = st.sidebar.radio("Paper:", papers)

             # get the Ticker from the data frame
    lista = [session_state.data_type]
    df_selected_sector = df[ (df['Symbol'].isin(lista)) ]


    nome = df_selected_sector['Name'].unique()[0]
    papel = df_selected_sector['Symbol'].unique()[0]
    setor = df_selected_sector['Sector'].unique()[0]
    country = df_selected_sector['Country'].unique()[0]
    class_share = df_selected_sector['Class of Share'].unique()[0]
    currency = df_selected_sector['Traded Currency'].unique()[0]

    subsetor = df_selected_sector['Industry'].unique()[0]
    data_b = '27/08/2021'


    tickerSymbol = df_selected_sector['Ticker'].unique()[0]

    tickerData = yf.Ticker(tickerSymbol)

    fund, zscore , s_zscore = fundlist(NADAQ_list)


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
    st.write(f"""## Fundamentals: {data_b} """ )
    hue


    st.write('## Historical data')


    start = st.date_input ( "Start date" , value=None , min_value=None , max_value=None , key=None )

    end = st.date_input( "End date" , value=None , min_value=None , max_value=None , key=None )

    tickerDF = tickerData.history(period = '1d', start = start, end = end)

    df_name = f"{papel}.csv"
    st.markdown(filedownload(tickerDF, df_name), unsafe_allow_html=True)

    try:
        fig = statick_candlestick(tickerDF)
        st.pyplot(fig = fig)
        tickerDF
    except:
        st.write('Data not found, check the dates')



elif session_state.workflow == 'Euronext':

    df = pd.read_csv('Euronext.csv')
    df = zscore(df, "sector")

    df = df.rename(columns={"industry": "Industry", "ticker": "Ticker", 'Company': 'Name', 'sector': 'Sector'})
    df = df.drop(columns = 'Unnamed: 0')
    df_name = "Euronext.csv"
    st.sidebar.markdown(filedownload(df, df_name), unsafe_allow_html=True)

    df['Sector'] = df['Sector'].replace({"":"Not found"})
    df['Industry'] = df['Industry'].replace({"":"Not found"})
    df["Industry"] = df.apply(lambda x: str(x["Industry"]),axis=1)
    df["Sector"] = df.apply(lambda x: str(x["Sector"]),axis=1)

    sorted_sector_unique = sorted(df['City'].unique())
    session_state.data_type = st.sidebar.multiselect("City:", sorted_sector_unique, sorted_sector_unique)

    #take the list of subsector to choose
    sorted_subsector_unique = sorted(df[ (df['City'].isin(session_state.data_type)) ]['Sector'].unique())
    session_state.data_type = st.sidebar.multiselect("Sector:", sorted_subsector_unique, sorted_subsector_unique[0])


    #sorted_sector_unique = sorted(df['Sector'].unique())
    #session_state.data_type = st.sidebar.multiselect("Sector:", sorted_sector_unique, sorted_sector_unique[0])

    #take the list of subsector to choose
    sorted_subsector_unique = sorted(df[ (df['Sector'].isin(session_state.data_type)) ]['Industry'].unique())
    session_state.data_type = st.sidebar.multiselect("Industry:", sorted_subsector_unique, sorted_subsector_unique[0])

    #take the list of companies to choose
    sorted_company_unique =  sorted(df[ (df['Industry'].isin(session_state.data_type)) ]['Name'].unique())
    #make a dataframe for further download
    selection_dataframe = df[ (df['Industry'].isin(session_state.data_type))]
             #companies list continuation
    session_state.data_type = st.sidebar.selectbox("Name:", sorted_company_unique)

             #radio boxes for the papers - make a list to work on the isin function(a bit of silvertape)
    lista = [session_state.data_type]
    papers = sorted(df[ (df['Name'].isin(lista)) ]['Symbol'].unique())
    session_state.data_type = st.sidebar.radio("Paper:", papers)

             # get the Ticker from the data frame
    lista = [session_state.data_type]
    df_selected_sector = df[ (df['Symbol'].isin(lista)) ]


    nome = df_selected_sector['Name'].unique()[0]
    papel = df_selected_sector['Symbol'].unique()[0]
    setor = df_selected_sector['Sector'].unique()[0]
    country = df_selected_sector['City'].unique()[0]
    class_share = df_selected_sector['Market'].unique()[0]
    currency = df_selected_sector['Trading Currency'].unique()[0]

    subsetor = df_selected_sector['Industry'].unique()[0]
    data_b = '27/08/2021'


    tickerSymbol = df_selected_sector['Ticker'].unique()[0]

    tickerData = yf.Ticker(tickerSymbol)

    fund, zscore , s_zscore = fundlist(NADAQ_list)


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
    st.write(f"""## Fundamentals: {data_b} """ )
    hue


    st.write('## Historical data')


    start = st.date_input ( "Start date" , value=None , min_value=None , max_value=None , key=None )

    end = st.date_input( "End date" , value=None , min_value=None , max_value=None , key=None )

    tickerDF = tickerData.history(period = '1d', start = start, end = end)

    df_name = f"{papel}.csv"
    st.markdown(filedownload(tickerDF, df_name), unsafe_allow_html=True)

    try:
        fig = statick_candlestick(tickerDF)
        st.pyplot(fig = fig)
        tickerDF
    except:
        st.write('Data not found, check the dates')
