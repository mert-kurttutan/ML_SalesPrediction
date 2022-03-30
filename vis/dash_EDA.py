# Import required libraries
from turtle import window_width
import pandas as pd
import dash
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

from dash import html, dcc
from dash.dependencies import Input, Output

pp=os.path.dirname(os.path.abspath(__file__))
pp = os.path.dirname(pp)

# This appends the path of parent folder to the global path of program
# Try not to use this anymore
sys.path.append(pp)       

from utils import generalized_hist_v2

# Read the sales data into pandas dataframe
df_items = pd.read_csv('../data/items.csv')
df_categories = pd.read_csv('../data/item_categories.csv')
df_shops = pd.read_csv('../data/shops.csv')
df_sales = pd.read_csv('../data/sales_train.csv')
df_sales_test = pd.read_csv('../data/test.csv')


# Add revenue info
df_sales['revenue'] = df_sales['item_price'] * df_sales['item_cnt_day']

# For convenience add category information to the sales data
df_sales['item_category_id'] = df_sales['item_id'].map(df_items['item_category_id'])


# Dictionary of functions to give appropriate title
site_to_title = {
                'date_block': lambda x: f'Total Number of {x} in each month',
                'item': lambda x: f'Total Number of {x} by Item',
                'category': lambda x: f'Total Number of {x} by Category',
                'shop': lambda x: f'Total Number of {x} by Shopping Store',
                'outlier': lambda x: f'Outliers in {"Price" if x=="Sales" else "Item_cnt_day"}',
                }

# Create a dash application
app = dash.Dash(__name__)

# Create an app layout
app.layout = html.Div(children=[html.H1('Daily Sales Data',
                                        style={'textAlign': 'center', 'color': '#3054D1',
                                               'font-size': 40}),
                                # Drop down menu to select the type of page
                                dcc.Dropdown(id='site-dropdown', 
                                                options=[
                                                        {'label': 'Date Blocks', 'value': 'date_block'},
                                                        {'label': 'Items', 'value': 'item'},
                                                        {'label': 'Categories', 'value': 'category'}, 
                                                        {'label': 'Shopping Stores', 'value': 'shop'},
                                                        {'label': 'Outliers', 'value': 'outlier'},],
                                                value='date_block',
                                                placeholder="Select a Transaction Feature",
                                                searchable=True),
                                html.Br(),
                                html.Div(dcc.Graph(id='num-transactions')),
                                html.Br(),

                                html.Div(dcc.Graph(id='num-sales')),
                                html.Br(),

                                html.Div(dcc.Graph(id='num-revenues')),
                                html.Br(),
                            ])       



# 
# Function decorator to specify function input and output
@app.callback(Output(component_id='num-transactions', component_property='figure'),
              Input(component_id='site-dropdown', component_property='value'))
def get_graph_transaction_num(entered_site):

    ''''
    Returns Graph for the amount of number of transaction numbers
    based on the entered_site
    '''
    filtered_df = df_sales
    title = site_to_title[entered_site]("Transactions")

    # Create figure object to put graph
    fig = go.Figure(layout_title_text=title)

    if entered_site == 'date_block':

        filtered_df = filtered_df[['date','date_block_num']].groupby(['date_block_num']).count().reset_index()

        # Add figures
        fig.add_trace(go.Bar(x=filtered_df["date_block_num"], y=filtered_df['date']))
        fig.add_trace(go.Scatter(x=filtered_df["date_block_num"], y=filtered_df["date"], mode='lines+markers'))

    elif entered_site == "item":

        # Adjust the width of bins based on bins_num for visual convenience
        bins_num = 10000
        count, division = np.histogram(df_sales['item_id'], bins=bins_num)

        width = 20*(division.max() - division.min()) / bins_num
        # Add figures
        fig.add_trace(go.Bar(x=division, y=count, marker_color="#C42200", opacity=1.0, width=width))

    elif entered_site == "category":
        filtered_df = filtered_df[['date','item_category_id']].groupby(['item_category_id']).count().reset_index()
        fig.add_trace(go.Bar(x=filtered_df["item_category_id"], y=filtered_df['date']))

    elif entered_site == "shop":
        filtered_df = filtered_df[['date','shop_id']].groupby(['shop_id']).count().reset_index()
        fig.add_trace(go.Bar(x=filtered_df["shop_id"], y=filtered_df['date']))

    else:
        filtered_df = filtered_df[['item_cnt_day']]
        # Adjust the width of bins based on bins_num for visual convenience
        bins_num = 100
        width = 1200 / bins_num
        count, division = np.histogram(filtered_df['item_cnt_day'], bins=bins_num)

        # Add figures
        fig.add_trace(go.Bar(x=division, y=count, marker_color="#C42200", opacity=1.0, width=width))
        fig.update_yaxes(title_text="y-axis (log scale)", type="log")

    # Set the gap between histogram bars
    fig.update_layout(bargap=0.2)

    
    return fig



# Function decorator to specify function input and output
@app.callback(Output(component_id='num-sales', component_property='figure'),
              Input(component_id='site-dropdown', component_property='value'))
def get_graph_sales_num(entered_site):

    ''''
    Returns Graph for the amount of number of sales numbers
    based on the entered_site
    '''
    filtered_df = df_sales
    title = site_to_title[entered_site]("Sales")
    fig = go.Figure(layout_title_text=title)
    fig.update_layout(bargap=0.2)

    if entered_site == 'date_block':

        filtered_df = filtered_df[['item_cnt_day','date_block_num']].groupby(['date_block_num']).sum().reset_index()

        fig.add_trace(go.Bar(x=filtered_df["date_block_num"], y=filtered_df['item_cnt_day']))
        fig.add_trace(go.Scatter(x=filtered_df["date_block_num"], y=filtered_df["item_cnt_day"], mode='lines+markers'))

    elif entered_site == "item":

        # Adjust the width of bins based bins_num for visual convenience
        bins_num = 1000
        filtered_df = filtered_df[['item_cnt_day','item_id']].groupby(['item_id']).sum().to_dict()['item_cnt_day']

        item_sales = df_items['item_id'].map(lambda x: filtered_df.get(x, 0)).reset_index()
        item_sales.columns = ['item_id', 'item_cnt']

        division, count = generalized_hist_v2(item_sales['item_id'], item_sales['item_cnt'], bins_num)
        width = 2*(division.max() - division.min()) / bins_num

        fig.add_trace(go.Bar(x=division, y=count, marker_color="#C42200", opacity=1.0, width=width))


    elif entered_site == "category":
        filtered_df = filtered_df[['item_cnt_day','item_category_id']].groupby(['item_category_id']).sum().reset_index()
        fig.add_trace(go.Bar(x=filtered_df["item_category_id"], y=filtered_df['item_cnt_day']))

    elif entered_site == "shop":
        filtered_df = filtered_df[['item_cnt_day','shop_id']].groupby(['shop_id']).sum().reset_index()
        fig.add_trace(go.Bar(x=filtered_df["shop_id"], y=filtered_df['item_cnt_day']))

    else:
        filtered_df = filtered_df[['item_price']]
        # Adjust the width of bins based on bins_num for visual convenience
        bins_num = 100
        width = 120000 / bins_num
        count, division = np.histogram(filtered_df['item_price'], bins=bins_num)

        # Add figures
        fig.add_trace(go.Bar(x=division, y=count, marker_color="#C42200", opacity=1.0, width=width))
        fig.update_yaxes(title_text="y-axis (log scale)", type="log")
    
    
    return fig




# Function decorator to specify function input and output
@app.callback(Output(component_id='num-revenues', component_property='figure'),
              Input(component_id='site-dropdown', component_property='value'))
def get_graph_revenue_num(entered_site):
    ''''
    Returns Graph for the amount of number of revenue numbers
    based on the entered_site
    '''

    filtered_df = df_sales
    title = site_to_title[entered_site]("Revenues")
    fig = go.Figure(layout_title_text=title)

    if entered_site == 'date_block':

        filtered_df = filtered_df[['revenue','date_block_num']].groupby(['date_block_num']).sum().reset_index()

        fig.add_trace(go.Bar(x=filtered_df["date_block_num"], y=filtered_df['revenue']))
        fig.add_trace(go.Scatter(x=filtered_df["date_block_num"], y=filtered_df["revenue"], mode='lines+markers'))

    elif entered_site == "item":

        # Adjust the width of bins based bins_num for visual convenience
        bins_num = 1000
        filtered_df = filtered_df[['revenue','item_id']].groupby(['item_id']).sum().to_dict()['revenue']

        item_sales = df_items['item_id'].map(lambda x: filtered_df.get(x, 0)).reset_index()
        item_sales.columns = ['item_id', 'item_rvn']

        division, count = generalized_hist_v2(item_sales['item_id'], item_sales['item_rvn'], bins_num)
        width = 2*(division.max() - division.min()) / bins_num
        fig.add_trace(go.Bar(x=division, y=count, marker_color="#C42200", opacity=1.0, width=width))


    elif entered_site == "category":
        filtered_df = filtered_df[['revenue','item_category_id']].groupby(['item_category_id']).sum().reset_index()
        fig.add_trace(go.Bar(x=filtered_df["item_category_id"], y=filtered_df['revenue']))

    elif entered_site == "shop":
        filtered_df = filtered_df[['revenue','shop_id']].groupby(['shop_id']).sum().reset_index()
        fig.add_trace(go.Bar(x=filtered_df["shop_id"], y=filtered_df['revenue']))

    
    return fig



# Call back function to hide the last plot when outlier is selected
@app.callback(Output('num-revenues', 'style'), [Input('site-dropdown','value')])
def hide_graph(input):
    if input=="outlier":
        return {'display':'none'}
    else:
        return {'display':'block'}


# Run the app
if __name__ == '__main__':
    app.run_server()
