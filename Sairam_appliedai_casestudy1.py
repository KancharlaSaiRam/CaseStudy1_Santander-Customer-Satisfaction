import dash
from dash import html, dcc
import pandas as pd
import pickle
from dash.dependencies import (
    Input, Output
)

import os
import random

'''
* make a text file
* Example input.txt
* load the text file using pandas.
* load your model and predict
'''

########################################
external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css'
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config['suppress_callback_exceptions'] = True
app.title = 'Customer Satisfaction'
server = app.server
########################################

# /home/ramchowdary/Desktop/APPLIED AI COURSE/casestudy-1/re-query/customer-dissatisfaction/deployment/required_columns.pkl
deployment_path = "/home/ramchowdary/Desktop/APPLIED AI COURSE/casestudy-1/re-query"

model_path = deployment_path+"/customer-dissatisfaction/deployment/best_model.pkl"
col_path = deployment_path+"/customer-dissatisfaction/deployment/required_columns.pkl"
# Frontend
app.layout = html.Div([
    html.H3('SANTANDER Customer Satisfaction Prediction'),
    # input section
    html.Div([
        # debounce=True allows to press enter and then the input is read
        # press enter when you provide the filepath in the input filed
        dcc.Input(
            id='input-filepath', type='text', value='', 
            placeholder='Path of the File', debounce=True
        )
    ]),

    # output section
    html.Div(id='output-prediction')

])


# Backend - callback
@app.callback(
    Output('output-prediction', 'children'),
    [Input('input-filepath', 'value')]
)
def show_prediction(file_path):
    if not os.path.isfile(path=file_path):
        return html.Div([
            html.P('File path either invalid or empty')
        ])
    
    else:
  
        data = pd.read_csv(file_path)
        # model prediction
        #output_val = sum(data.values[0])
        
        clf = pickle.load(open(model_path,"rb"))
        
        cols = pickle.load(open(col_path,"rb"))
        
        pred = data[cols]
        output_val = clf.predict(pred)
        
        if output_val[0] == 0:
            predict = "CUSTOMER IS SATISFIED"
        else:
            predict = "CUSTOMER IS UNSATISFIEDunsatisfied"

        return html.Div([
            html.P('The prediction is : {}'.format(predict))
        ])


if __name__ == '__main__':
    app.run_server(debug=True)