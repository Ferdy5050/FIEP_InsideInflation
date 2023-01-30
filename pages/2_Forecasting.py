### Import Libraries
import streamlit as st
import pandas as pd
import pandas_datareader.data as pdr
import pandas_datareader as web
import numpy as np
import plotly.graph_objects as go
import datetime as dt

### Set Up the App
st.set_page_config(page_title = "Forecasting", page_icon = ":bar_chart:", layout = "wide")
st.title(":bar_chart: Forecasting")
st.markdown("##")

### Regression Section
st.subheader("Regression Analysis & Forecasting")

### Data Pull Requirements - Symbols and Names
countries = ["United States","Euro Area"]
country = st.sidebar.selectbox(label="Select Country", options=countries, index=countries.index(st.session_state["country"]))

if country == "United States":

    years = list(range(1960,2022,1))
    start_year = st.sidebar.selectbox(label="Select Start Year", options=years, index=years.index(st.session_state["curr_year"]))

    ### Data Pull from previous page
    names = st.session_state["names"]
    
    
    df = st.session_state["df"]
    df = df[df.index.year > start_year-2]
    
    rates = st.session_state["rates"]
    rates = rates[rates.index.year > start_year-1]
    
    cpi = st.session_state["cpi"]
    cpi = cpi[cpi.index.year > start_year-1]
    
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

# Define Inputs
choices = st.sidebar.multiselect(label="Select exogenous variables for Regression (CPI Inflation is by default always the endogenous variable)", options=names, default=names[0])
max_lag = st.sidebar.number_input(label="Select number of lags (i.e., previous values for the selected variables, to forecast the inflation with)", min_value=0, max_value=24, value=3, step=1)
lags = list(range(1,max_lag+1))

# Define Regression Dataframe and calculate selected lags
reg_cols = list(choices)

reg_df = pd.DataFrame()

for choice in choices:

    for lag in lags:
        reg_df[choice+"_L"+str(lag)] = rates[choice].shift(lag)
        
reg_df["InflationRate"] = cpi["Inflation Rate"]

# Rename Reg_df (no spaces)
old_cols = list(reg_df.columns)
new_cols = [col.replace(" ", "") for col in old_cols]

for old_col,new_col in zip(old_cols,new_cols):
    reg_df = reg_df.rename(columns={old_col:new_col})

reg_df.dropna(inplace=True)

# Define Regression String
Y_string = "InflationRate"
X_list = list(reg_df.columns[:-1])
X_join = "+".join(X_list)

reg_string = Y_string+"~"+X_join

# Perform Regression and show results
import statsmodels.formula.api as smf

reg_output = smf.ols(reg_string,reg_df).fit()

### Forecasting Part

# Select Variables to Forecast with
pred_choices = list(reg_output.params.iloc[np.where(reg_output.pvalues < 0.05)].index)[1:]
pred_select = st.sidebar.multiselect(label="Here are automatically available variables with p<0.05", options=pred_choices, default=pred_choices)

# Define Prediction Dataframe
pred_df = reg_df[pred_select]
pred_df["InflationRate"] = reg_df["InflationRate"]

# Redo Regression for selected parameters
pred_join = "+".join(pred_select)
pred_string = "InflationRate~"+pred_join

pred_reg_output = smf.ols(pred_string,pred_df).fit()

# Define new prediction df
new_pred_df = pd.DataFrame()
new_pred_df["Inflation Rate (Actual)"] = pred_df["InflationRate"]
    
# Forecast Inflation
new_pred_df["Inflation Rate (Predicted)"] = pred_reg_output.predict(pred_df)

conf_int = pred_reg_output.conf_int()

# Calculate Forecasting Error (as RMSFE)
rmsfe = round(np.sqrt(np.mean((new_pred_df["Inflation Rate (Actual)"] - new_pred_df["Inflation Rate (Predicted)"]) ** 2)),1)

#new_pred_df.to_excel("pred_df.xlsx")

### Implement Step-Forecast: out-of-sample forecast, i.e.: 12 steps ahead

forecast_checkbox = st.sidebar.checkbox("Out-of-Sample Forecast")

# Calculate step-by-step forecast based on in-sample regression and use new predicted value as new predictor
steps = range(1,12)
forecasts = pd.DataFrame()

# Initial predictor-value = last in-sample prediction
Xnew = new_pred_df["Inflation Rate (Predicted)"].iloc[-1]
Xnew2 = new_pred_df["Inflation Rate (Predicted)"].iloc[-2]
Xnew3 = new_pred_df["Inflation Rate (Predicted)"].iloc[-3]

sig_vars = list(pred_reg_output.params.index)[1:]

for i in steps:
    # Create Dateindex for forecast, to append at correct location in pred_df
    date = dt.datetime(2023,0+i,1)
    
    # Create Forecasted Value
    if len(sig_vars) == 1:
        Ynew = pred_reg_output.params[sig_vars[0]] * Xnew
        
        ci_low = Xnew * conf_int.loc[sig_vars[0]][0]
        ci_up = Xnew * conf_int.loc[sig_vars[0]][1]
        
        Xnew = Ynew

    if len(sig_vars) == 2:
        Ynew = pred_reg_output.params[sig_vars[0]] * Xnew + pred_reg_output.params[sig_vars[1]] * Xnew2
        
        ci_low = Xnew * conf_int.loc[sig_vars[0]][0] + Xnew2 * conf_int.loc[sig_vars[1]][0]
        ci_up = Xnew * conf_int.loc[sig_vars[0]][1] + Xnew2 * conf_int.loc[sig_vars[1]][1]
        
        Xnew2 = Xnew
        Xnew = Ynew

    if len(sig_vars) == 3:
        Ynew = pred_reg_output.params[sig_vars[0]] + Xnew + pred_reg_output.params[sig_vars[1]] * Xnew2 + pred_reg_output.params[sig_vars[2]] * Xnew3
        
        ci_low = Xnew * conf_int.loc[sig_vars[0]][0] + Xnew2 * conf_int.loc[sig_vars[1]][0] + Xnew3 * conf_int.loc[sig_vars[2]][0]
        ci_up = Xnew * conf_int.loc[sig_vars[0]][1] + Xnew2 * conf_int.loc[sig_vars[1]][1] + Xnew3 * conf_int.loc[sig_vars[2]][1]
        
        Xnew3 = Xnew2
        Xnew2 = Xnew
        Xnew = Ynew
    
    # Append to Forecast DF
    new_row = {"Date": date, "Forecast":Ynew, "CI_low":ci_low, "CI_up":ci_up}   
    forecasts = forecasts.append(new_row, ignore_index=True)  
    
forecasts = forecasts.set_index(["Date"])


### Append new forecasts to pred_df
#new_pred_df = new_pred_df.append(forecast)

### Calculate average out-of-sample predicted inflation
avg_inflation_forecast = round(forecasts["Forecast"].mean(),1)

### Show Metrics
st.markdown("##")
metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

metric_col1.metric(label="Actual Value (latest)", value=str(round(new_pred_df["Inflation Rate (Actual)"].iloc[-1],1))+" %")
metric_col2.metric(label="Predicted Value (latest)", value=str(round(new_pred_df["Inflation Rate (Predicted)"].iloc[-1],1))+" %")
metric_col3.metric(label="Avg. Predicted 1 Year Ahead Inflation", value=str(avg_inflation_forecast)+" %")
metric_col4.metric(label="Adj. R^2", value=str(round(pred_reg_output.rsquared_adj * 100,1))+" %")
metric_col5.metric(label="RMSFE", value=str(rmsfe)+" %")

### Show results (table and plot)
pred_fig = go.Figure()

pred_fig.add_trace(go.Scatter(x=new_pred_df.index, y=new_pred_df["Inflation Rate (Actual)"],
                    mode='lines', 
                    name='Actual'))
                    
pred_fig.add_trace(go.Scatter(x=new_pred_df.index, y=new_pred_df["Inflation Rate (Predicted)"],
                    mode='lines', 
                    name='Predicted (In-Sample)'))


if forecast_checkbox:    
            
    pred_fig.add_trace(go.Scatter(x=forecasts.index, y=forecasts["Forecast"],
                        mode='lines', 
                        name='Predicted (Out-of-Sample)'))
                        

    pred_fig.add_traces([go.Scatter(x = forecasts.index, y = forecasts["CI_low"],
                           mode = 'lines', line_color = 'rgba(0,0,0,0)',
                           showlegend = False),
                go.Scatter(x = forecasts.index, y = forecasts["CI_up"],
                           mode = 'lines', line_color = 'rgba(0,0,0,0)',
                           name = '95% confidence interval',
                           fill='tonexty', fillcolor = 'rgba(255, 0, 0, 0.2)')])
                        
                    
pred_fig.update_layout(
    title="Actual vs. Predicted Inflation Rate",
    yaxis=dict(title_text="Inflation Rate (%)", titlefont=dict(size=12)),
    xaxis=dict(title_text="Date", titlefont=dict(size=12)),
    legend=dict(x=0.05, y=0.9))

st.plotly_chart(pred_fig, use_container_width=True)

pred_check = st.checkbox("See Prediction Regression Results")

if pred_check:
    st.write(pred_reg_output.summary2())
    
    
    
################# Statistical decomposition
st.markdown("##")
st.subheader("Statistical Decomposition")

from statsmodels.tsa.seasonal import seasonal_decompose

decomp_data = new_pred_df["Inflation Rate (Actual)"]

from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose

def plot_seasonal_decompose(result:DecomposeResult, dates:pd.Series=None, title:str="Seasonal Decomposition"):
    x_values = dates if dates is not None else np.arange(len(result.observed))
    return (
        make_subplots(
            rows=4,
            cols=1,
            subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"],
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.observed, mode="lines", name='Observed'),
            row=1,
            col=1,
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.trend, mode="lines", name='Trend'),
            row=2,
            col=1,
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.seasonal, mode="lines", name='Seasonal'),
            row=3,
            col=1,
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.resid, mode="lines", name='Residual'),
            row=4,
            col=1,
        )
        .update_layout(
            height=900, title=f'<b>{title}</b>', margin={'t':100}, title_x=0.5, showlegend=False
        )
    )
    
data = decomp_data
decomposition = seasonal_decompose(decomp_data, model='additive', period=12)
fig = plot_seasonal_decompose(decomposition, dates=decomp_data.index)

st.plotly_chart(fig, use_container_width=True)