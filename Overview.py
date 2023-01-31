### Import Libraries
import streamlit as st
import pandas as pd
import pandas_datareader.data as pdr
import pandas_datareader as web
import numpy as np
import plotly.graph_objects as go
import datetime as dt

###########################################################
# This is the Overview part of our Inflation Dashboard App

### Set Up the Streamlit App
st.set_page_config(page_title = "InsideInflation", page_icon = ":bar_chart:", layout = "wide")
st.title(":bar_chart: InsideInflation")
st.markdown("##")

st.session_state["shared"] = True

### Data Pull Requirements - Symbols and Names
countries = ["United States","Euro Area"]

### Create Selectors and prepare Data Pull
country = st.sidebar.selectbox(label="Select Country", options=countries, index=0)
st.session_state["country"] = country

if country == "United States":
    
    @st.cache
    def get_data(start_year):
    
        # Import Data
        df = pd.read_excel("usa.xlsx", index_col=0)
        df = df[df.index.year > start_year-2] 
        
        return df

    years = list(range(1960,2022,1))
    start_year = st.sidebar.selectbox(label="Select Start Year", options=years, index=0)
    
    df = get_data(start_year)

    names = list(df.columns)

    rates = pd.DataFrame()

    for name in names:
        rates[name] = round((df[name] / df[name].shift(12) - 1) * 100,2)

    ### Set CPI as "Standard" Variable and define stable get data for set of explanatory variables (i.e., country-specific symbols)

    # Get CPI Data
    cpi = pd.read_excel("cpi.xlsx", index_col=0)
    cpi = cpi[cpi.index.year > start_year-1]
    
    #df.to_excel("usa.xlsx")
    #cpi.to_excel("cpi.xlsx")
    
    st.session_state["country"] = country
    st.session_state["curr_year"] = start_year
    st.session_state["names"] = names
    st.session_state["df"] = df
    st.session_state["rates"] = rates
    st.session_state["cpi"] = cpi
    
else:
    # Import Data from Excel
    df = pd.read_excel("euroarea.xlsx", sheet_name="Monthly", index_col=0)
    
    years = df["Years"].unique()
    start_year = st.sidebar.selectbox(label="Select Start Year", options=years, index=0)
    
    # Select Data for year
    df = df[df["Years"]>start_year-2]
    
    cpi = pd.DataFrame()
    cpi["CPI"] = df["HICP"]
    
    # Calculate Inflation Rate
    cpi["Inflation Rate"] = round((cpi["CPI"] / cpi["CPI"].shift(12) - 1) * 100,2)
    
    # Calculate Annual Rates of Change
    rates = pd.DataFrame()
    rates["CPI"] = round((df["HICP"] / df["HICP"].shift(12) - 1) * 100,2)
    rates["CPI Core"] = round((df["HICP Core"] / df["HICP Core"].shift(12) - 1) * 100,2)
    rates["CPI Energy"] = round((df["HICP Energy"] / df["HICP Energy"].shift(12) - 1) * 100,2)
    rates["CPI Food"] = round((df["HICP Food"] / df["HICP Food"].shift(12) - 1) * 100,2)
    rates["CPI Housing"] = round((df["HICP Housing"] / df["HICP Housing"].shift(12) - 1) * 100,2)
    rates["Unemployment Rate"] = round((df["Unemployment Rate"] / df["Unemployment Rate"].shift(12) - 1) * 100,2)
    rates["Industrial Production"] = round((df["Industrial Production"] / df["Industrial Production"].shift(12) - 1) *100,2)
    rates["M1"] = round((df["M1"] / df["M1"].shift(12) - 1) * 100,2)
    rates["M2"] = round((df["M2"] / df["M2"].shift(12) - 1) * 100,2)
    rates["M3"] = round((df["M3"] / df["M3"].shift(12) - 1) * 100,2)
    
    names = list(rates.columns)
    
    old_cols = list(df.columns[1:])
    
    for old_col, name in zip(old_cols, names):
        df = df.rename(columns={old_col:name})
 
### Metrics above the Plots
### Metrics

st.subheader("Latest Values of Key Inflation Components")
metric_col1, metric_col2, metric_col3, metric_col4, metric_col5, metric_col6 = st.columns(6)

metric_col1.metric(label="Headline", value=str(round(cpi["Inflation Rate"].iloc[-1],1))+" %")
metric_col2.metric(label="Historical Average", value=str(cpi["Inflation Rate"].mean().round(1))+" %")
metric_col3.metric(label="Core", value=str(round(rates["CPI Core"].iloc[-1],1))+" %")
metric_col4.metric(label="Energy", value=str(round(rates["CPI Energy"].iloc[-1],1))+" %")
metric_col5.metric(label="Food", value=str(round(rates["CPI Food"].iloc[-1],1))+" %")
metric_col6.metric(label="Housing", value=str(round(rates["CPI Housing"].iloc[-1],1))+" %")

st.markdown("##")
st.subheader("Inflation Overview")
### Show two plots: Development of the Inflation Rate over time and Heatmap

#checkcol1, checkcol2, checkcol3, checkcol4, checkcol5 = st.columns(5) 

#cpi_headline = checkcol1.checkbox("CPI", value=True)
#cpi_core = checkcol2.checkbox("CPI Core")
#cpi_energy = checkcol3.checkbox("CPI Energy")
#cpi_food = checkcol4.checkbox("CPI Food")
#cpi_housing = checkcol5.checkbox("CPI Housing")

st.sidebar.markdown("Adjust Inflation Rate Plot")
cpi_headline = st.sidebar.checkbox("CPI", value=True)
cpi_core = st.sidebar.checkbox("CPI Core")
cpi_energy = st.sidebar.checkbox("CPI Energy")
cpi_food = st.sidebar.checkbox("CPI Food")
cpi_housing = st.sidebar.checkbox("CPI Housing")

# Plot 1: Inflation Rate
##### Add traces if checkboxes are activated

fig1 = go.Figure()

if cpi_headline:

    fig1.add_trace(go.Scatter(x=cpi.index, y=cpi["Inflation Rate"],
                        mode='lines', 
                        name='Inflation Rate'))
   
if cpi_core:

    fig1.add_trace(go.Scatter(x=rates.index, y=rates["CPI Core"],
                        mode='lines', 
                        name='CPI Core'))                     
if cpi_energy:

    fig1.add_trace(go.Scatter(x=rates.index, y=rates["CPI Energy"],
                        mode='lines', 
                        name='CPI Energy'))
                        
if cpi_food:

    fig1.add_trace(go.Scatter(x=rates.index, y=rates["CPI Food"],
                        mode='lines', 
                        name='CPI Food'))

if cpi_housing:

    fig1.add_trace(go.Scatter(x=rates.index, y=rates["CPI Housing"],
                        mode='lines', 
                        name='CPI Housing'))

fig1.update_layout(
    title="Inflation Rate (Monthly, Year-over-Year)",
    yaxis=dict(title_text="Inflation Rate (%)", titlefont=dict(size=12)),
    xaxis=dict(title_text="Date", titlefont=dict(size=12)))
    
    
# Plot 2: Heatmap - Contemporaneous Correlation
z1=rates.dropna().corr().values.round(3)
x1=rates.dropna().corr().columns
y1=rates.dropna().corr().index

fig2 = go.Figure(data=go.Heatmap(
        z=z1,
        x=x1,
        y=y1,
        colorscale='Viridis'))

fig2.update_layout(
    title='Contemporaneous Correlation of Inflation-Related Variables',
    xaxis_nticks=36)   
    
fig2['layout']['yaxis']['autorange'] = "reversed"
fig2.update_traces(text=z1, texttemplate="%{text}")
                  
plot_col1, plot_col2 = st.columns(2)

plot_col1.plotly_chart(fig1, use_container_width=True)
plot_col2.plotly_chart(fig2, use_container_width=True)


########### Correlogramm

### Calculate Year-over-year percent changes and lags
df_corr = df.dropna()

changes = pd.DataFrame()
lags = list(range(1,25))

for name in names:
    # Calculate Changes
    changes[name] = round((df_corr[name] / df_corr[name].shift(12) - 1) * 100, 2)
   
### Create a DF, where for each variable the correlation and its lags are calculated lag by lag and not at once, so that I can create a new df with the variables as index and the lag as the columns
lags = list(range(25))

corr_lags = pd.DataFrame()

corr_col1, corr_col2 = st.columns(2)
cpi_comp = corr_col1.checkbox("Only CPI Components")
mon_agg = corr_col2.checkbox("Only Monetary Aggregrates")

if cpi_comp:
    names = ["CPI Core","CPI Energy","CPI Food","CPI Housing"]
    
    
if country == "United States":
    if mon_agg:
        names = ["M0","M1","M2","M3"]

if country == "Euro Area":
    if mon_agg:
        names = ["M1","M2","M3"]

for name in names:
        
    corr = []

    if name != "CPI":
        
        for lag in lags:
            selected_df = pd.DataFrame()
            selected_df["Inflation"] = changes["CPI"]
            selected_df[name+"_L"+str(lag)] = changes[name].shift(lag)

            curr_corr=selected_df.corr().round(3)["Inflation"].iloc[1]

            corr.append(curr_corr)

        corr_lags["Lags"] = lags
        corr_lags[name] = corr
        
corr_lags = corr_lags.set_index("Lags")


###Plot 3: Heatmap - Lagged Value Correlation
### Create Correlation Heatmap
z2=corr_lags.values
x2=corr_lags.columns
y2=corr_lags.index

fig3 = go.Figure(data=go.Heatmap(
       z=z2,
       x=x2,
       y=y2,
       colorscale='Viridis'))

fig3.update_layout(
   title='Correlation of Lagged Values with Current Inflation',
   xaxis_nticks=36)   
    
fig3['layout']['yaxis']['autorange'] = "reversed"
fig3.update_traces(text=z2, texttemplate="%{text}")


# Plot 4: Time Series of Lagged Values Correlation
# Plot 1: Inflation Rate
fig4 = go.Figure()

corr_vars = list(corr_lags.columns)

for i in range(len(corr_vars)):

    fig4.add_trace(go.Scatter(x=corr_lags.index, y=corr_lags[corr_lags.columns[i]],
                    mode='lines',
                    name=corr_lags.columns[i]))
                    
fig4.update_layout(
    title="At which time lag does the correlation with the current inflation peak?",
    yaxis=dict(title_text="Correlation", titlefont=dict(size=12)),
    xaxis=dict(title_text="Lag", titlefont=dict(size=12)))

plot_col3, plot_col4 = st.tabs(["Lagged Correlation Time Series","Lagged Correlation Heatmap"])

#plot_col3.plotly_chart(fig3, use_container_width=True)
#plot_col4.plotly_chart(fig4, use_container_width=True)

plot_col3.plotly_chart(fig4, use_container_width=True)
plot_col4.plotly_chart(fig3, use_container_width=True)
#######################################################
### Public Attention Measure

st.markdown("##")
st.subheader("Public Attention Measure of the Current Inflation")

cpi_checkbox = st.checkbox("Add Actual Inflation", value=True)

### Test Google Trends API
from pytrends.request import TrendReq
pytrend = TrendReq()

### Input

start = dt.datetime(start_year,1,1)
end = dt.datetime.today()
keyword = "inflation"

if country == "United States":

    if start_year < dt.datetime.today().year - 5:
        st.write("Data only available for the last 5 years.")
        
    else:
        geo = "US"
        
        # Format start and end to necessary format
        start = start.strftime("%Y-%m-%d")
        end = end.strftime("%Y-%m-%d")

        timeframe = start+" "+end

        ### Get Data
        data = pd.DataFrame()

        # Define Payload (i.e., the "query") 
        pytrend.build_payload(kw_list=[keyword], geo=geo, timeframe=timeframe)

        # Define Method. Here: Interest over time (default timeframe: last 5 years)
        geo_df = pytrend.interest_over_time()
            
        data[geo] = geo_df["inflation"]

        data = data.resample("M").mean()
        
        ### Get Data for Inflation Expectations (Michigan Survey)
        try:
            mich = pdr.DataReader("MICH","fred",start,end)      
            mich = mich.resample("M").mean()
            
        except:
            print("No Data available")
            
            
        from plotly.subplots import make_subplots

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces
        fig.add_trace(
            go.Scatter(x=data.index, y=data[geo], name="Google Trends Index"),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=mich.index, y=mich["MICH"], name="Inflation Expectations"),
            secondary_y=True,
        )

        if cpi_checkbox:
        
            fig.add_trace(go.Scatter(x=cpi.index, y=cpi["Inflation Rate"],
                    mode='lines', 
                    name='Actual Inflation'),
                    secondary_y=True)
                    
        # Add figure title
        fig.update_layout(
            title_text="Google Trends Search Index for the Keyword 'inflation', Inflation Expectations measured by the Michigan Survey (and if selected: actual inflation)",
            yaxis=dict(title_text="Index", titlefont=dict(size=12)),
            xaxis=dict(title_text="Date", titlefont=dict(size=12)),
            legend=dict(x=0.05, y=0.9))

        # Set x-axis title
        fig.update_xaxes(title_text="Date")

        # Set y-axes titles
        fig.update_yaxes(title_text="Index", secondary_y=False)
        fig.update_yaxes(title_text="Inflation Expectatuins (%)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
else:
    
    if start_year < dt.datetime.today().year - 5:
        st.write("Data only available for the last 5 years.")
        
    else:
    
        ### Create a "portfolio" for the euro area
        geo_countries = ["Germany","France","Italy","Spain","Netherlands","Belgium"]
        geos = ["DE","FR","IT","ES","NL","BE"]

        # GDP Shares in Euro Area for these countries
        gdp_countries = ["Euro Area","Germany","France","Italy","Spain","Netherlands","Belgium"]
        gdp_codes = ["CLVMNACSCAB1GQEA19","CLVMNACSCAB1GQDE","CLVMNACSCAB1GQFR","CLVMNACSCAB1GQIT","CLVMNACSCAB1GQES","CLVMNACSCAB1GQNL","CLVMNACSCAB1GQBE"]

        gdp = pd.DataFrame()
        for gdp_country, gdp_code in zip(gdp_countries, gdp_codes):
            gdp[gdp_country] = pdr.DataReader(gdp_code,"fred",start,end)
            
        gdp_shares = pd.DataFrame()

        for gdp_country in gdp_countries:
            gdp_shares[gdp_country] = gdp[gdp_country] / gdp["Euro Area"]
            
            
        ### Google Trends API
        from pytrends.request import TrendReq
        pytrend = TrendReq()

        # Input
        geos = ["DE","FR","IT","ES","NL","BE"]

        # Format start and end to necessary format
        start = start.strftime("%Y-%m-%d")
        end = end.strftime("%Y-%m-%d")

        timeframe = start+" "+end

        ### Loop over list
        data = pd.DataFrame()

        for geo in geos:
            # Define Payload (i.e., the "query") 
            pytrend.build_payload(kw_list=[keyword], geo=geo, timeframe=timeframe)

            # Define Method. Here: Interest over time (default timeframe: last 5 years)
            geo_df = pytrend.interest_over_time()
            
            data[geo] = geo_df["inflation"]

        data = data.resample("M").mean()
        
        ### Calculate Google Trends Search Index for the Euro Area out that
        
        coeff = pd.DataFrame()

        coeff["DE"] = data["DE"] * gdp_shares.iloc[-1][1:]["Germany"]
        coeff["FR"] = data["FR"] * gdp_shares.iloc[-1][1:]["France"]
        coeff["IT"] = data["IT"] * gdp_shares.iloc[-1][1:]["Italy"]
        coeff["ES"] = data["ES"] * gdp_shares.iloc[-1][1:]["Spain"]
        coeff["NL"] = data["NL"] * gdp_shares.iloc[-1][1:]["Netherlands"]
        coeff["BE"] = data["BE"] * gdp_shares.iloc[-1][1:]["Belgium"]

        coeff["EA"] = coeff.sum(axis=1)
        
        # Plot Data   

        from plotly.subplots import make_subplots  
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(go.Scatter(x=coeff.index, y=coeff["EA"],
                            mode='lines', name="Google Trends Search Index"), 
                            secondary_y=False)
                            
        if cpi_checkbox:
        
            fig.add_trace(go.Scatter(x=cpi.index, y=cpi["Inflation Rate"],
                    mode='lines', 
                    name='Actual Inflation'),
                    secondary_y=True)

        fig.update_layout(
        title="Google Trends Search Index for the Keyword 'inflation' (and if selected: actual inflation)",
        yaxis=dict(title_text="Index", titlefont=dict(size=12)),
        xaxis=dict(title_text="Date", titlefont=dict(size=12)),
        legend=dict(x=0.1, y=0.8))
        
        # Set x-axis title
        fig.update_xaxes(title_text="Date")

        # Set y-axes titles
        fig.update_yaxes(title_text="Index", secondary_y=False)
        fig.update_yaxes(title_text="Actual Inflation (%)", secondary_y=True)
       
        st.plotly_chart(fig, use_container_width=True)
