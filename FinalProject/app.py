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
import plotly.graph_objects as go

from Embedding_concat import concat_embeddings
from pca_reduction import pca_columns

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

DEFAULT_COLORS = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#d62728',  # brick red
    '#bcbd22',  # curry yellow-green
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#17becf'   # blue-teal
    '#2ca02c',  # cooked asparagus green
]

#begin visualization
def render_visualization():
    
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    app.layout =html.Div(
        children=[

            # row with title
            dbc.Row(
                dbc.Col(
                    html.Div(
                        children=html.H1("Chart Constellations", style={"textAlign" : "center"})
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
                                    dbc.Card(
                                        [
                                            dbc.CardHeader(dcc.Markdown("#### Number of Clusters", style={"textAlign" : "center"})),
                                            dbc.CardBody(                                    
                                                dcc.Slider(
                                                    id='cluster_slider', 
                                                    min= 2, 
                                                    max= 5, 
                                                    value= 4, 
                                                    marks={str(num): str(num) for num in range(2,6)}, 
                                                    step=None
                                                )
                                            )
                                        ]
                                    ),
                                    dbc.Card(
                                        [
                                            dbc.CardHeader(dcc.Markdown("#### Embedding Weights", style={"textAlign" : "center"})),
                                            dbc.CardBody(
                                                [
                                                    dcc.Markdown("##### Keywords", style={"textAlign" : "center"}),
                                                    dcc.Slider(
                                                        id='wt1_slider', 
                                                        min= 0, 
                                                        max= 1, 
                                                        value= 1, 
                                                        marks={str(num): str(num) for num in range(0,2)}, 
                                                        step=0.01
                                                    ),
                                                    dcc.Markdown("##### Represented Features", style={"textAlign" : "center"}),
                                                    dcc.Slider(
                                                        id='wt2_slider', 
                                                        min= 0, 
                                                        max= 1, 
                                                        value= 1, 
                                                        marks={str(num): str(num) for num in range(0,2)}, 
                                                        step=0.01
                                                    ),
                                                    dcc.Markdown("##### Local Binary Encoding", style={"textAlign" : "center"}),
                                                    dcc.Slider(
                                                        id='wt3_slider', 
                                                        min= 0, 
                                                        max= 1, 
                                                        value= 1, 
                                                        marks={str(num): str(num) for num in range(0,2)}, 
                                                        step=0.01
                                                    )
                                                ]
                                            )
                                        ]
                                    )
                                ],
                                style={
                                    'height' : '29em',
                                }
                            ),
                            html.Div(
                                dbc.Card(
                                    [
                                        dbc.CardHeader(dcc.Markdown("#### Filters", style={"textAlign" : "center"})),
                                        dbc.CardBody(
                                            [
                                                dcc.Markdown("##### Authors", style={"textAlign" : "center"}),
                                                dcc.Dropdown(
                                                    id = 'author_filter',
                                                    options=author_filter,
                                                    multi=True,
                                                    value=[]
                                                ),
                                                dcc.Markdown("##### Keywords", style={"textAlign" : "center"}),
                                                dcc.Dropdown(
                                                    id = 'keyword_filter',
                                                    options=keyword_filter,
                                                    multi=True,
                                                    value=[]
                                                )
                                            ]
                                        )
                                    ]
                                ),
                                style={
                                    'height' : '25em',
                                }
                            )
                        ],
                        style={
                            'height' : '50em'    
                        }
                    ),
                    width=2
                ),

                # column with scatterplot
                dbc.Col(
                    html.Div(
                        children=dcc.Graph(
                            figure=px.scatter(x=[0,1,2,3,4], y=[0,1,4,9,16]),
                            id='scatter_plot',
                            style={
                                'height' : '50em'
                            }
                        ),
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
                                                                dcc.Markdown("#### Data Coverage Board", style={"textAlign" : "center"}),
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
                                                                dcc.Markdown("#### Cluster Board", style={"textAlign" : "center"}),
                                                                color="link",
                                                                id="cluster_board_toggle",
                                                            )
                                                        ),
                                                        dbc.Collapse(
                                                            children=dcc.Checklist(options=[], value=None, id="selected_author"),
                                                            id="collapse_cluster_board",
                                                        ),
                                                    ]
                                            )
                                    ],
                                    className="accordion"
                                ),
                        style={
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
        if click_data:
            chosen_cluster = int(click_data["points"][0]['curveNumber'])
        else:
            return dbc.CardBody("Select a cluster from the scatterplot", id="selected_author")

        slice = df[df["cluster"] == chosen_cluster]

        author_stats = slice['author'].value_counts().to_dict()
        keyword_stats = pd.Series([j for i in list(df['keywords'].values) for j in i]).value_counts().to_dict()
        
        content = dbc.CardBody(
            [
                html.H4(f"Cluster {chosen_cluster}:", className="card-title"),
                html.H5("Authors:"),
                html.P(str(author_stats)),
            ]
        )

        return content


    # callback to update dashboard with slider values
    @app.callback(
            [Output('scatter_plot', 'figure')],
            [Input('wt1_slider', 'value'),
            Input('wt2_slider', 'value'),
            Input('wt3_slider', 'value'),
            Input('cluster_slider', 'value'),
            Input('author_filter', 'value'),
            Input('keyword_filter', 'value')]
    )
    def update_chart(wt1, wt2, wt3, k, authors_filter, keywords_filter):
        global df

        # concatenate embeddings based on weights
        concat_embeddings(df, wt1, wt2, wt3)

        # perform clustering
        embedding_combo_array = np.array(df['concat_embedding'].tolist())
        km = KMeans(n_clusters=k, random_state=10).fit(embedding_combo_array)
        df['cluster'] = km.labels_ 
        
        # dimensionality reduction
        df = pca_columns(df)

        # filtering
        if not authors_filter:
            authors_filter = author_set
        if not keywords_filter:
            keywords_filter = keyword_set

        temp_df = df[df['author'].isin(authors_filter)]
        filtered_df = {}
        for index, row in temp_df.iterrows():
            if any(x in keywords_filter for x in row["keywords"]):
                filtered_df[index] = row

        filtered_df = pd.DataFrame(filtered_df).T

        fig = go.Figure()

        for x in range(k):
            fig.add_trace(go.Scatter(x=filtered_df[filtered_df["cluster"] == x]["PC1"].tolist(),
                                        y=filtered_df[filtered_df["cluster"] == x]["PC2"].tolist(),
                                        mode="markers",
                                        opacity=0.65,
                                        marker=dict(
                                            color = DEFAULT_COLORS[x], 
                                            size = 23,
                                            line = dict(
                                                color = "rgb(0,0,0)",
                                                width = 1
                                            )
                                        ),
                                        name="Cluster {}".format(x)
                                    )
            )
        
        fig.update_layout(
            showlegend=True,
            xaxis={
                'ticks' : '',
                'showticklabels' : False
            },
            yaxis={
                'ticks' : '',
                'showticklabels' : False
            },
            plot_bgcolor="rgb(255,255,255)"
        )

        return [fig]

    app.run_server(debug=True)

if __name__ == "__main__":

    global df

    if not os.path.isfile("dataFrames/df_embeddings.pckl"):
        subprocess.call(["unzip", "df_embeddings.zip"], cwd=f"{os.path.dirname(os.path.abspath(__file__))}/dataFrames/")
    
    with open("dataFrames/df_embeddings.pckl", "rb") as f:
        df = pickle.load(f)

    author_filter = []
    authors = df['author'].tolist()
    author_set = list(set(authors))
    for author in author_set:
        author_filter.append({'label': author, 'value': author})
        
    keyword_set = []
    keyword_filter = []
    keyword_lol = df['keywords'].tolist()
    for keyword in keyword_lol:
        keyword_set.extend(keyword)  
    keyword_set = list(set(keyword_set))
    for keyword in keyword_set:
        keyword_filter.append({'label': keyword, 'value': keyword})

    render_visualization()
