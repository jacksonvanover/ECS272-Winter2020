import pandas as pd
import numpy as np
import subprocess
import os
import pickle
from sklearn.cluster import KMeans

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import plotly.express as px

from Embedding_concat import concat_embeddings

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
                                children=[
                                    dcc.Slider(
                                        id='cluster_slider', 
                                        min= 2, 
                                        max= 5, 
                                        value= 2, 
                                        marks={str(num): str(num) for num in range(2,6)}, 
                                        step=None
                                    ),
                                    dcc.Slider(
                                        id='wt1_slider', 
                                        min= 0, 
                                        max= 1, 
                                        value= 1, 
                                        marks={str(num): str(num) for num in range(0,2)}, 
                                        step=0.01
                                    ),
                                    dcc.Slider(
                                        id='wt2_slider', 
                                        min= 0, 
                                        max= 1, 
                                        value= 1, 
                                        marks={str(num): str(num) for num in range(0,2)}, 
                                        step=0.01
                                    ),
                                    dcc.Slider(
                                        id='wt3_slider', 
                                        min= 0, 
                                        max= 1, 
                                        value= 1, 
                                        marks={str(num): str(num) for num in range(0,2)}, 
                                        step=0.01
                                    ),
                                ],
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
                        children=dcc.Graph(figure=px.scatter(x=[0,1,2,3,4], y=[0,1,4,9,16]),id='scatter_plot'),
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
                                        dbc.Card(
                                            children=[
                                                        dbc.CardHeader(
                                                            dbc.Button(
                                                                "Data Coverage Board",
                                                                color="link",
                                                                id="data_coverage_board_toggle",
                                                            )
                                                        ),
                                                        dbc.Collapse(
                                                            dbc.CardBody("Data Coverage Board goes here..."),
                                                            id="collapse_data_coverage_board",
                                                        ),
                                                    ]
                                            ),
                                        dbc.Card(
                                            children=[
                                                        dbc.CardHeader(
                                                            dbc.Button(
                                                                "Cluster Board",
                                                                color="link",
                                                                id="cluster_board_toggle",
                                                            )
                                                        ),
                                                        dbc.Collapse(
                                                            id="collapse_cluster_board",
                                                        ),
                                                    ]
                                            )
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

    # callback to toggle the collapsible data coverage board and
    # cluster board
    @app.callback(
        [Output("collapse_data_coverage_board", "is_open"),
        Output("collapse_cluster_board", "is_open")],
        [Input("data_coverage_board_toggle", "n_clicks"),
        Input("cluster_board_toggle", "n_clicks")],
        [State("collapse_data_coverage_board", "is_open"),
        State("collapse_cluster_board", "is_open")],
    )
    def toggle_accordion(n1, n2, is_open1, is_open2):
        ctx = dash.callback_context

        if not ctx.triggered:
            return ""
        else:
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if button_id == "data_coverage_board_toggle" and n1:
            return not is_open1, is_open2
        elif button_id == "cluster_board_toggle" and n2:
            return is_open1, not is_open2
        return True, True


    # callback to display stats on the cluster selected from the scatterplot
    @app.callback(
        Output("collapse_cluster_board", "children"),
        [Input("scatter_plot", "clickData")]
    )
    def generate_cluster_board(click_data):
        if not click_data:
            return dbc.CardBody("Default Cluster Board stats"),
        else:
            return dbc.CardBody("Updated Cluster board stats")


    # callback to update dashboard with slider values
    @app.callback(
            [Output('scatter_plot', 'figure')],
            [Input('wt1_slider', 'value'),
            Input('wt2_slider', 'value'),
            Input('wt3_slider', 'value'),
            Input('cluster_slider', 'value')]
    )
    def update_chart(wt1, wt2, wt3, k):
        concat_embeddings(df, wt1, wt2, wt3)
        embedding_combo_array = np.array(df['concat_embedding'].tolist())
        km = KMeans(n_clusters=k, random_state=10).fit(embedding_combo_array)
        df['cluster'] = km.labels_ 
        
        #insert call to recalculate PCA#
        
        # fig = px.scatter(df, x="PC1", y="PC2", color="cluster",
        #         hover_data=['title'])

        # temporary scatterplot
        fig = px.scatter(x=[0,1,2,3,4], y=[0,1,4,9,16])

        return [fig]

    app.run_server(debug=True)

if __name__ == "__main__":

    global df

    if not os.path.isfile("dataFrames/df_embeddings.pckl"):
        subprocess.call(["unzip", "df_embeddings.zip"], cwd=f"{os.path.dirname(os.path.abspath(__file__))}/dataFrames/")
    
    with open("dataFrames/df_embeddings.pckl", "rb") as f:
        df = pickle.load(f)

    render_visualization()
