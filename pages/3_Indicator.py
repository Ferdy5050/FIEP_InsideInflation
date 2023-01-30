### Import Libraries
import pandas as pd
import pandas_datareader.data as pdr
import datetime as dt
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import statistics
import plotly.express as px
import plotly.graph_objects as go

### Streamlit Page
st.set_page_config(page_title = "Forecasting", page_icon = ":bar_chart:")
st.title('Inflation Indicator')

### Data Pull Requirements - Symbols and Names
countries = ["United States","Euro Area"]

### Create Selectors and prepare Data Pull
country = st.sidebar.selectbox(label="Select Country", options=countries, index=0)

if country == "United States":

    year_range = range(1960,2022,1)
    start_year = st.sidebar.selectbox(label="Select Start Year", options=year_range, index=year_range.index(st.session_state["curr_year"]))
    # start_year = st.sidebar.selectbox(label="Select Start Year", options=year_range, index=0)

    ### Import Data
    start = dt.datetime(start_year-1, 1, 1)
    end = dt.datetime.today()

    df = pdr.DataReader(["CPIAUCSL"], "fred", start, end)
    df_i = pdr.DataReader(["GPDI"], "fred", start, end)
    df_p = pdr.DataReader(["INDPRO"], "fred", start, end)
    df_c = pdr.DataReader(["PCE"], "fred", start, end)
    gdp = pdr.DataReader(["GDPC1"], "fred", start, end)
    
    ### Visualization for Rates
    df["Inflation Rate"] = round((df["CPIAUCSL"] / df["CPIAUCSL"].shift(12) - 1) * 100, 2)
    df_i["Investment Rate (% change)"] = round((df_i["GPDI"] / df_i["GPDI"].shift(4) - 1) * 100, 2)
    df_p["Production Rate (% Change)"] = round((df_p["INDPRO"] / df_p["INDPRO"].shift(12) - 1) * 100, 2)
    df_c["Consumption Rate (% Change)"] = round((df_c["PCE"] / df_c["PCE"].shift(12) - 1) * 100, 2)
    gdp["Real GDP Growth"] = round((gdp["GDPC1"] / gdp["GDPC1"].shift(4) - 1) * 100, 2)



else:
    # Import Data
    data_quarterly = pd.read_excel("euroarea.xlsx", sheet_name="Quarterly", index_col=0)
    data_monthly = pd.read_excel("euroarea.xlsx", sheet_name="Monthly", index_col=0)
    
    year_range = range(1996,2022,1)
    start_year = st.sidebar.selectbox(label="Select Start Year", options=year_range, index=0)
    
    # Select Data based on selected timeframe
    data_quarterly = data_quarterly[data_quarterly["Years"]>start_year-1]
    data_monthly = data_monthly[data_monthly["Years"]>start_year-1]
    
    df = data_monthly["HICP"]
    df_i = data_quarterly["Gross capital formation"]
    df_p = data_monthly["Industrial Production"]
    df_c = data_quarterly["Personal consumption expenditures"]
    gdp = data_quarterly["Real GDP"]

    ### Visualization for Rates
    df["Inflation Rate"] = round((df/ df.shift(12) - 1) * 100, 2)
    df_i["Investment Rate (% change)"] = round((df_i / df_i.shift(4) - 1) * 100, 2)
    df_p["Production Rate (% Change)"] = round((df_p / df_p.shift(12) - 1) * 100, 2)
    df_c["Consumption Rate (% Change)"] = round((df_c / df_c.shift(4) - 1) * 100, 2)
    gdp["Real GDP Growth"] = round((gdp / gdp.shift(4) - 1) * 100, 2)


### drop NaN Values
df_i.dropna(inplace=True)
df.dropna(inplace=True)
gdp.dropna(inplace=True)
df_p.dropna(inplace=True)
df_c.dropna(inplace=True)

### Rates today

cur_inflation = df["Inflation Rate"] [-1:][0]
cur_investment_growth = df_i["Investment Rate (% change)"] [-1:][0]
cur_REALGDP_growth = gdp["Real GDP Growth"] [-1:][0]
cur_prod_rate = df_p["Production Rate (% Change)"] [-2:][0]
cur_cons_rate = df_c["Consumption Rate (% Change)"] [-1:][0]

# Select Variable
selection = st.selectbox(label="Select Economic Indicator", options=["Inflation","Investment Rate","Real GDP growth","Consumption Rate","Production Rate"], index=0)


### colums
colum1, colum2= st.columns(2)
colum3, colum4= st.columns(2)
colum5, colum6= st.columns(2)
colum7, colum8= st.columns(2)
colum9, colum10= st.columns(2)

if selection == "Inflation":

    box_fig = go.Figure()
    box_fig.add_trace(go.Box(x=df["Inflation Rate"]))
    
    colum1.plotly_chart(box_fig , use_container_width=True)

    line_fig = go.Figure()
    line_fig.add_trace(go.Scatter(x=df.index, y=df["Inflation Rate"],
                        mode='lines',
                        name="Inflation Rate"))
    colum2.plotly_chart(line_fig, use_container_width=True)

if selection == "Investment Rate":

    box_fig = go.Figure()
    box_fig.add_trace(go.Box(x=df_i["Investment Rate (% change)"]))
    colum1.plotly_chart(box_fig , use_container_width=True)

    line_fig = go.Figure()
    line_fig.add_trace(go.Scatter(x=df_i.index, y=df_i["Investment Rate (% change)"],
                        mode='lines',
                        name="Investment Rate"))
                        
    colum2.plotly_chart(line_fig, use_container_width=True)

if selection == "Real GDP growth":

    box_fig = go.Figure()
    box_fig.add_trace(go.Box(x=gdp["Real GDP Growth"]))
    colum1.plotly_chart(box_fig, use_container_width=True)

    line_fig = go.Figure()
    line_fig.add_trace(go.Scatter(x=gdp.index, y=gdp["Real GDP Growth"],
                                  mode='lines',
                                  name="Real GDP growth"))

    colum2.plotly_chart(line_fig, use_container_width=True)

if selection == "Consumption Rate":

    box_fig = go.Figure()
    box_fig.add_trace(go.Box(x=df_c["Consumption Rate (% Change)"]))
    colum1.plotly_chart(box_fig, use_container_width=True)

    line_fig = go.Figure()
    line_fig.add_trace(go.Scatter(x=df_c.index, y=df_c["Consumption Rate (% Change)"],
                                  mode='lines',
                                  name="Consumption Rate"))

    colum2.plotly_chart(line_fig, use_container_width=True)

if selection == "Production Rate":

    box_fig = go.Figure()
    box_fig.add_trace(go.Box(x=df_p["Production Rate (% Change)"]))
    colum1.plotly_chart(box_fig, use_container_width=True)

    line_fig = go.Figure()
    line_fig.add_trace(go.Scatter(x=df_p.index, y=df_p["Production Rate (% Change)"],
                                  mode='lines',
                                  name="Production Rate"))

    colum2.plotly_chart(line_fig, use_container_width=True)

else:
    st.write("")

### upper and lower whisker, for score

### Inflation Rate
upper_quartile = df["Inflation Rate"].quantile(0.75)
lower_quartile = df["Inflation Rate"].quantile(0.25)

iqr = upper_quartile - lower_quartile
upper_whisker = df["Inflation Rate"][df["Inflation Rate"]<=upper_quartile+1.5*iqr].max()
lower_whisker = df["Inflation Rate"][df["Inflation Rate"]>=lower_quartile-1.5*iqr].min()

### Investment Rate
upper_quartile_i = df_i["Investment Rate (% change)"].quantile(0.75)
lower_quartile_i = df_i["Investment Rate (% change)"].quantile(0.25)

iqr_i = upper_quartile_i - lower_quartile_i
upper_whisker_i = df_i["Investment Rate (% change)"][df_i["Investment Rate (% change)"]<=upper_quartile_i+1.5*iqr_i].max()
lower_whisker_i = df_i["Investment Rate (% change)"][df_i["Investment Rate (% change)"]>=lower_quartile_i-1.5*iqr_i].min()

### Real GDP
upper_quartile_g = gdp["Real GDP Growth"].quantile(0.75)
lower_quartile_g = gdp["Real GDP Growth"].quantile(0.25)

iqr_g = upper_quartile_g - lower_quartile_g
upper_whisker_g = gdp["Real GDP Growth"][gdp["Real GDP Growth"]<=upper_quartile_g+1.5*iqr_g].max()
lower_whisker_g = gdp["Real GDP Growth"][gdp["Real GDP Growth"]>=lower_quartile_g-1.5*iqr_g].min()

### Produc. Rate
upper_quartile_p = df_p["Production Rate (% Change)"].quantile(0.75)
lower_quartile_p = df_p["Production Rate (% Change)"].quantile(0.25)

iqr_p = upper_quartile_p - lower_quartile_p
upper_whisker_p = df_p["Production Rate (% Change)"][df_p["Production Rate (% Change)"]<=upper_quartile_p+1.5*iqr_p].max()
lower_whisker_p = df_p["Production Rate (% Change)"][df_p["Production Rate (% Change)"]>=lower_quartile_p-1.5*iqr_p].min()

### consump. Rate
upper_quartile_c = df_c["Consumption Rate (% Change)"].quantile(0.75)
lower_quartile_c = df_c["Consumption Rate (% Change)"].quantile(0.25)

iqr_c = upper_quartile_c - lower_quartile_c
upper_whisker_c = df_c["Consumption Rate (% Change)"][df_c["Consumption Rate (% Change)"]<=upper_quartile_c+1.5*iqr_c].max()
lower_whisker_c = df_c["Consumption Rate (% Change)"][df_c["Consumption Rate (% Change)"]>=lower_quartile_c-1.5*iqr_c].min()


### Score Inflation Rate

infaltion_score = 0

if cur_inflation > upper_whisker:
    inflation_score = 1

elif cur_inflation > df["Inflation Rate"].quantile(0.75):
    inflation_score = 2

elif cur_inflation > df["Inflation Rate"].quantile(0.5):
    inflation_score = 3

elif cur_inflation > df["Inflation Rate"].quantile(0.25):
    inflation_score = 4

elif cur_inflation > lower_whisker:
    inflation_score = 5

else:
    inflation_score = 6

### Score Investment Rate

investment_score = 0

if cur_investment_growth < lower_whisker_i:
    investment_score = 1

elif cur_investment_growth < df_i["Investment Rate (% change)"].quantile(0.25):
    investment_score = 2

elif cur_investment_growth < df_i["Investment Rate (% change)"].quantile(0.5):
    investment_score = 3

elif cur_investment_growth < df_i["Investment Rate (% change)"].quantile(0.75):
    investment_score = 4

elif cur_investment_growth < upper_whisker_i:
    investment_score = 5

else:
    investment_score = 6

### Score For Real GDP Growth

GDP_score = 0

if cur_REALGDP_growth < lower_whisker_g:
    GDP_score = 1

elif cur_REALGDP_growth < gdp["Real GDP Growth"].quantile(0.25):
    GDP_score = 2

elif cur_REALGDP_growth < gdp["Real GDP Growth"].quantile(0.5):
    GDP_score = 3

elif cur_REALGDP_growth < gdp["Real GDP Growth"].quantile(0.75):
    GDP_score = 4

elif cur_REALGDP_growth < upper_whisker_g:
    GDP_score = 5

else:
    GDP_score = 6

### Industrial Production

prod_score = 0

if cur_prod_rate < lower_whisker_p:
    prod_score = 1

elif cur_prod_rate < df_p["Production Rate (% Change)"].quantile(0.25):
    prod_score = 2

elif cur_prod_rate < df_p["Production Rate (% Change)"].quantile(0.5):
    prod_score = 3

elif cur_prod_rate < df_p["Production Rate (% Change)"].quantile(0.75):
    prod_score = 4

elif cur_prod_rate < upper_whisker_p:
    prod_score = 5

else:
    prod_score = 6

### Real Consumer Spendings

consumer_score = 0

if cur_cons_rate < lower_whisker_c:
    consumer_score = 1

elif cur_cons_rate < df_c["Consumption Rate (% Change)"].quantile(0.25):
    consumer_score = 2

elif cur_cons_rate < df_c["Consumption Rate (% Change)"].quantile(0.5):
    consumer_score = 3

elif cur_cons_rate < df_c["Consumption Rate (% Change)"].quantile(0.75):
    consumer_score = 4

elif cur_cons_rate < upper_whisker_c:
    consumer_score = 5

else:
    consumer_score = 6



### create Visualization for Score Board

fig9, ax = plt.subplots()

long_df= [inflation_score, GDP_score, prod_score, investment_score, consumer_score]
fruits = ['Inflation','Real GDP', 'Production', 'Investment', 'Consumption']

fig9= go.Figure([go.Bar(x=fruits, y=long_df)], layout_yaxis_range=[0,6])

fig9.update_layout(
    title="Indicator Scoreboard",
    yaxis=dict(title_text="Score", titlefont=dict(size=12)))

colum9.plotly_chart(fig9 , use_container_width=True)


### Visualization of Todays Inflation

dif=cur_inflation-df["Inflation Rate"] [-2:][0]

st.metric(label="Inflation", value=cur_inflation, delta=round(dif,2),
    delta_color="inverse")


### Describing Text

st.write('Overall desription for Score values (except for Inflation Rate):')
st.write('Score 1: Current Rate is lower than the lower whisker from the Data Boxplot')
st.write('Score 2: Current Rate is lower than 0.25 Quantile')
st.write('Score 3: Current Rate is lower than the median')
st.write('Score 4: Current Rate is lower than 0.75 Quantile')
st.write('Score 5: Current Rate is lower than upper whisker')
st.write('Score 6: Current Rate is higher than upper whisker')