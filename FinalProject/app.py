import pandas as pd
import numpy as np
import subprocess
import os
import pickle

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

'''
COLUMN_NAMES :
        'image',
        'title',
        'author',
        'keywords',
        'features',
        'code',
        'pixel_embedding', 
        'represented_features_embeddings',
        'word2vec_embeddings'
'''
def collapsible_item(i):
    # we use this function to make the collapsible items to avoid code duplication
    return dbc.Card(
        [
            dbc.CardHeader(
                dbc.Button(
                    f"{i}",
                    color="link",
                    id=f"{i}_toggle",
                )
            ),
            dbc.Collapse(
                dbc.CardBody(f"{i} goes here..."),
                id=f"collapse_{i}",
            ),
        ]
    )

#begin visualization
def render_visualization():
    
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    app.layout =html.Div(
        children=[

            # row with title
            dbc.Row(
                dbc.Col(
                    html.Div(
                        children=html.H1("Our Final Project"),
                    )
                )
            ),

            dbc.Row([

                # column with sliders and filters
                dbc.Col(
                    html.Div(
                        children=[
                            html.Div(
                                children=dcc.Markdown('# sliders go here'),
                                style={
                                    'height' : '25em',
                                    'backgroundColor' : 'rgba(0,0,0,0.01)'
                                }
                            ),

                            html.Div(
                                children=dcc.Markdown('# filters go here'),
                                style={
                                    'height' : '25em',
                                    'backgroundColor' : 'rgba(255,255,255,0.5)'
                                }
                            )
                        ],
                        style={
                            'backgroundColor' : 'rgb(0,0,220)',
                            'height' : '50em'    
                        }
                    ),
                    width=2
                ),

                # column with scatterplot
                dbc.Col(
                    html.Div(
                        children=dcc.Markdown("# Scatterplot goes here"),
                        style={
                            'backgroundColor' : 'rgb(0,220,0)',
                            'height' : '50em'
                        }
                    ),
                    width=7
                ),

                # column with Data Coverage Board and Cluster Board
                dbc.Col(
                    html.Div(
                        children=html.Div(
                                    children=[
                                        collapsible_item("Data Coverage Board"),
                                        collapsible_item("Cluster Board")
                                    ],
                                    className="accordion"
                                ),
                        style={
                            'backgroundColor' : 'rgb(220,0,0)',
                            'height' : '50em'
                        }
                    ),
                    width=3
                )
            ])
        ]
    )

    @app.callback(
        [Output(f"collapse_Data Coverage Board", "is_open"),
        Output(f"collapse_Cluster Board", "is_open")],
        [Input(f"Data Coverage Board_toggle", "n_clicks"),
        Input(f"Cluster Board_toggle", "n_clicks")],
        [State(f"collapse_Data Coverage Board", "is_open"),
        State(f"collapse_Cluster Board", "is_open")],
    )
    def toggle_accordion(n1, n2, is_open1, is_open2):
        ctx = dash.callback_context

        if not ctx.triggered:
            return ""
        else:
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if button_id == "Data Coverage Board_toggle" and n1:
            return not is_open1, is_open2
        elif button_id == "Cluster Board_toggle" and n2:
            return is_open1, not is_open2
        return True, True

    app.run_server(debug=True)

if __name__ == "__main__":

    global df

    if not os.path.isfile("dataFrames/df_embeddings.pckl"):
        subprocess.call(["unzip", "df_embeddings.zip"], cwd=f"{os.path.dirname(os.path.abspath(__file__))}/dataFrames/")
    
    with open("dataFrames/df_embeddings.pckl", "rb") as f:
        df = pickle.load(f)

    render_visualization()
