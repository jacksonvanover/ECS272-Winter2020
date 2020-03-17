#!/usr/bin/env python
# coding: utf-8

# In[24]:


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

from io import BytesIO
import base64

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
TREEMAP_FEATURE = []

DEFAULT_COLORS = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#d62728',  # brick red 
    '#bcbd22',  # curry yellow-green
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#17becf',  # blue-teal
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
                                #style={
                                #    'height' : '29em',
                                #}
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
                                #style={
                                #    'height' : '15em',
                                #}
                            ),
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
                            figure=px.scatter(x=[0,1,2,3,4], y=[0,1,4,9,16]
                            #                  ,hover_data=[0,1,2,3,4]
                                             ),
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
                                        #chart image 
                                        #showing chosen chart
                                        dbc.Card([
                                            dbc.CardHeader(dbc.Button(
                                                dcc.Markdown("#### Chosen Chart", style={"textAlign" : "center"}),
                                                color="link",
                                                id="chosen_chart_board_toggle",
                                                
                                            )),
                                            dbc.Collapse(
                                                dbc.CardBody([
                                                    html.Img(
                                                        id="body-image",
                                                        src = "",
                                                        alt = "choose a point",
                                                        draggable = "True",
                                                        style={
                                                            'height' : '10em',
                                                            "textAlign" : "center"
                                                        }
                                                    ),
                                                ]),
                                                id="collapse_chosen_chart_board",
                                            )
                                        ]),
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
                                                            dbc.CardBody(
                                                                # dcc.Graph(
                                                                #     figure=
                                                                #         px.treemap(
                                                                #             df_fake,
                                                                #             path=['features'],
                                                                #             values='feature_frequency',
                                                                #             color=feature_frequency,
                                                                #         ),
                                                                #     id='treemap',
                                                                #     )
                                                                ),
                                                            id="collapse_data_coverage_board",
                                                        ),
                                                        dcc.Graph(
                                                                figure=
                                                                    px.treemap(
                                                                        df_fake,
                                                                        path=['features'],
                                                                        values='feature_frequency',
                                                                        color=feature_frequency,
                                                                    ),
                                                                id='treemap',
                                                                )
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
    
    #callback to show image
    @app.callback(
        Output("body-image","src"),
        [Input("scatter_plot", "clickData")]
    )
    def load_image(click_data):
        global df
        
        if not click_data:
            return ''
        
        #determine datapoint from click data
        pc1 = click_data["points"][0]['x']
        pc2 = click_data["points"][0]['y']
        pt_df = df[df['PC1'] == pc1]
        pt_df = pt_df[pt_df['PC2'] == pc2]
        
        #load image from dataframe
        i = pt_df.index[0]
        img = pt_df['image'][i]
        
        #remove alpha channel
        img = img.convert("RGB")
        
        #save image to bytes, jpg
        img_out = BytesIO()
        img.save(img_out, format = "JPEG")
        
        #encode to base64 byte string
        img_jpg_data = base64.b64encode(img_out.getvalue())
        if not isinstance(img_jpg_data, str):
            img_jpg_data = img_jpg_data.decode()
        
        #return byte string of image
        data_url = 'data:image/jpg;base64,' + img_jpg_data
        return data_url
    
    
    # callback to toggle the collapsible data coverage board and
    # cluster board and chosen chart board
    @app.callback(
        [Output("collapse_data_coverage_board", "is_open"),
        Output("collapse_cluster_board", "is_open"),
        Output("collapse_chosen_chart_board","is_open")],
        [Input("data_coverage_board_toggle", "n_clicks"),
        Input("cluster_board_toggle", "n_clicks"),
        Input("chosen_chart_board_toggle", "n_clicks")],
        [State("collapse_data_coverage_board", "is_open"),
        State("collapse_cluster_board", "is_open"),
        State("collapse_chosen_chart_board", "is_open")],
    )
    def toggle_accordion(n1, n2, n3, is_open1, is_open2, is_open3):
        ctx = dash.callback_context

        if not ctx.triggered:
            return ""
        else:
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if button_id == "data_coverage_board_toggle" and n1:
            return not is_open1, is_open2, is_open3
        elif button_id == "cluster_board_toggle" and n2:
            return is_open1, not is_open2, is_open3
        elif button_id == "chosen_chart_board_toggle" and n3:
            return is_open1, is_open2, not is_open3
        return True, True, True


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
            Input('keyword_filter', 'value'),
            Input("treemap", "clickData"),
            Input('scatter_plot', 'selectedData')]
    )
    def update_chart(wt1, wt2, wt3, k, authors_filter, keywords_filter, treemap_click_data, select_value):
        global df
        print(select_value)
        # concatenate embeddings based on weights
        concat_embeddings(df, wt1, wt2, wt3)

        # perform clustering
        embedding_combo_array = np.array(df['concat_embedding'].tolist())
        km = KMeans(n_clusters=k, random_state=10).fit(embedding_combo_array)
        df['cluster'] = km.labels_ 
        
        # dimensionality reduction
        df = pca_columns(df)

        global TREEMAP_FEATURE
        selected_feature = []
        if treemap_click_data is not None:
            selected_feature.append(treemap_click_data['points'][0]['label'])
        treemap_df = {}
        if not TREEMAP_FEATURE == selected_feature:
            TREEMAP_FEATURE = selected_feature
            for index, row in df.iterrows():
                if any(x in selected_feature for x in row["features"]):
                    treemap_df[index] = row
        treemap_df = pd.DataFrame(treemap_df).T

        # select points for highlighting
        selected_authors_indices = df[df['author'].isin(authors_filter)].index
        selected_keywords_indices = []
        for index, row in df.iterrows():
            if any(x in keywords_filter for x in row["keywords"]):
                selected_keywords_indices.append(index)

        if not selected_authors_indices.empty and selected_keywords_indices:
            selected_row_indices = list(set(selected_authors_indices) & set(selected_keywords_indices))
        elif not selected_authors_indices.empty:
            selected_row_indices = selected_authors_indices
        elif selected_keywords_indices:
            selected_row_indices = selected_keywords_indices
        else:
            selected_row_indices = []

        highlight_points_df = df.iloc[selected_row_indices]

        fig = go.Figure()

        for x in range(k):
            fig.add_trace(go.Scatter(x=df[df["cluster"] == x]["PC1"].tolist(),
                                        y=df[df["cluster"] == x]["PC2"].tolist(),
                                        mode="markers",
                                        opacity=0.65,
                                        #hovertext=[str(i) for i in range(len(df))],
                                        hovertext=[str(i) for i in df.index.tolist()],
                                        hoverinfo='text',
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

        # plot highlighted points 
        if not highlight_points_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=highlight_points_df["PC1"].tolist(),
                    y=highlight_points_df["PC2"].tolist(),
                    mode="markers",
                    hovertext=[str(i) for i in highlight_points_df.index.tolist()],
                    hoverinfo='text',
                    marker=dict(
                        color="rgba(0,0,0,0)",
                        size=23,
                        line = dict(
                            color = "rgb(0,0,0)",
                            width = 7
                        ),
                    ),
                name="Filtered Points"
                )
            )

        # plot treemap selection 
        if not treemap_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=treemap_df["PC1"].tolist(),
                    y=treemap_df["PC2"].tolist(),
                    mode="markers",
                    marker_symbol="asterisk",
                    hovertext=[str(i) for i in treemap_df.index.tolist()],
                    hoverinfo='text',
                    marker=dict(
                        color="rgb(0,0,0)",
                        size=23,
                        line = dict(
                            color = "rgb(0,0,0)",
                            width = 2
                        ),
                    ),
                name="Treemap Selection"
                )
            )

        fig.update_layout(
            showlegend=True,
            legend={'x' : 0.8, 'y' : 0.9, 'bordercolor': "Black", 'borderwidth': 1},
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

    app.run_server(debug=False)

if __name__ == "__main__":

    global df

    with open("df_embeddings.pckl", "rb") as f:
        df = pickle.load(f)
######treeMap###############
    matrix=np.zeros([55,33])
    sum_list=np.array((1,33))
    for i, value in enumerate(df.represented_features_embeddings):
        matrix[i,:]=np.array(value)
    feature_frequency=np.sum(matrix, axis=0).tolist()
        
    features=['school', 'sex', 'age', 'address','famsize','Pstatus', 'Medu','Fedu', 'Mjob', 
          'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup',
          'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet','romantic',
          'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1',
          'G2', 'G3']
    df_fake = pd.DataFrame(dict(features=features, feature_frequency=feature_frequency))
######treeMap###############
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

