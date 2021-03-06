import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE 

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.express as px
import plotly.graph_objects as go

import pickle

#import data frame
from cluster_comp import df, MAX_CLUSTERS

#globals
CURRENT_CLUSTER_NO = 3

# load a dict to translate short column headers to full survey questions    
with open("columns.csv") as f:
    x = pd.read_csv(f)
    headerDict = pd.Series(x.original.values, index=x.short).to_dict()

# create dict of categories for survey questions
question_categories = {
    "Music" : list(df.columns[0:19]),
    "Movies" : list(df.columns[19:31]),
    "Hobbies" : list(df.columns[31:63]),
    "Phobias" : list(df.columns[63:73]),
    "Health" : list(df.columns[73:76]),
    "Personality" : list(df.columns[76:133]),
    "Spending" : list(df.columns[133:140]),
    "Demographics" : list(df.columns[140:150])
}


default_colors = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#d62728',  # brick red
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
    '#2ca02c',  # cooked asparagus green
]

#construct cluster options
cluster_option_list = []
for val in range(1,MAX_CLUSTERS):
    val += 1
    cluster_option_list.append({'label' : str(val)+ ' Clusters', 'value' : val})

#begin visualization
def render_visualization():
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    app.layout = html.Div([

        html.Div([
            html.Div(html.H1("Young People Survey")),
        ],className="row", style={'width' : '100%'}),

        html.Div([

            html.Div(dcc.Markdown('''
            The **scatterplot** shows each survey respondent plotted by top 3 principle components and colored by K-means cluster.
            
            The **heatmap** shows correlation coefficients between survey responses for the selected cluster.
            
            The **violin plot** shows distributions of survey responses per cluster for the selected question categories.
            
            The **radar plot** shows survey question responses for any individual selected from the scatter plot.'''
            ), style={'width' : '30%'}, className='three columns'),

            html.Div([

                html.Div([
                    dcc.Markdown("**Question Categories:**"),

                    dcc.Checklist(
                        id = 'question-dropdown',
                        options=[
                            {'label' : 'Music', 'value' : 'Music'},
                            {'label' : 'Movies', 'value' : 'Movies'},
                            {'label' : 'Hobbies', 'value' : 'Hobbies'},
                            {'label' : 'Phobias', 'value' : 'Phobias'},
                            {'label' : 'Health', 'value' : 'Health'},
                            {'label' : 'Personality', 'value' : 'Personality'},
                            {'label' : 'Spending', 'value' : 'Spending'}],
                        value=['Music','Movies']
                    )],
                    className='six columns',
                    style={'width': '45%'}
                ),

                html.Div([
                    dcc.Markdown("**Clustering Options:**"),

                    dcc.RadioItems(
                        id = 'cluster-dropdown',
                        options=cluster_option_list,
                        value=3,
                    )],
                    className='six columns',
                    style={'width': '45%'}
                )],
                className="three columns",
                style={'backgroundColor' : 'rgb(220,220,220)', 'width' : '25%'})
        ],
        className="row",
        style={'width' : '100%'}),

        html.Div([
            html.Div(
                dcc.Graph(
                        id='3d-plot',
                        style={'height' : '50em'}
                ),
                className='six columns',
                style={'width': '50em', 'height': '50em', 'backgroundColor' : 'rgb(0,0,0)'}
            ),
            # html.Div([
            #     dcc.Graph(
            #         id='radar-chart',
            #         style={'width' : '30em'}),
            #     html.Div([
            #         dcc.Markdown("###### Selected Individual's Demographic Info",
            #                         style={'textAlign' : 'center', 'width' : '30em', }),
            #         dcc.Markdown(
            #             id='demographic-info',
            #             style={'textAlign' : 'center', 'width' : '30em'}
            #         )
            #     ], style={'backgroundColor' : 'rgb(220, 220,220)'}),
            #     ],
            #     style={'width' : '30em', 'height' : '50em'},
            #     className='six columns'
            # ),
            html.Div(
                dcc.Graph(
                    id='corr-heatmap',
                    style={'height' : '49em'}
                ),
                className='six columns',
                style={'width': '49em', 'height' : '49em'}
            ),
        ], className="row"),
        
        html.Div([
            html.Div([
                html.Div([
                    dcc.Markdown("###### Selected Individual's Demographic Info",
                                    style={'textAlign' : 'center', 'width' : '39em', }),
                    dcc.Markdown(
                        id='demographic-info',
                        style={'textAlign' : 'center', 'width' : '39em'}
                    )
                ], style={'backgroundColor' : 'rgb(220, 220,220)'}),
                dcc.Graph(
                    id='radar-chart',
                    style={'width' : '39em'}),
                ],
                style={'width' : '39em', 'height' : '50em'},
                className='six columns'
            ),

            # html.Div(
            #     dcc.Graph(
            #         id='corr-heatmap',
            #         style={'height' : '40em'}
            #     ),
            #     className='six columns',
            #     style={'width': '40em', 'height' : '40em'}
            # ),
            html.Div([
                html.Div(
                    dcc.Markdown("#### Distribution of Survey Responses Per Cluster",
                                    style={'textAlign': 'center', 'width' : '60em'}),
                    style={'width' : '60em'}, className='six-columns'
                ),

                html.Div(
                    dcc.Graph(
                        id='violins',
                        style={'width':'100%'}
                    ),
                style={'maxWidth' : '60em', 'overflowX' : 'scroll'}
                )], 
                className='six columns'
            )
        ], className='row')
    ])


    @app.callback(
        [Output('radar-chart', 'figure'),
        Output('demographic-info', 'children')],
        [Input('question-dropdown', 'value'),
        Input('3d-plot', 'clickData')]
    )
    def update_radar_chart(category_choices, click_data):
        if click_data:
            x = click_data['points'][0]['x']
            y = click_data['points'][0]['y']
            z = click_data['points'][0]['z']
        else:
            x = df.iloc[0]['xcoord']
            y = df.iloc[0]['ycoord']
            z = df.iloc[0]['zcoord']

        df_single = df[(df['xcoord'] == x ) & (df['ycoord'] == y) & (df['zcoord'] == z)]
        
        chosen_columns = []
        for choice in category_choices:
            chosen_columns = chosen_columns + question_categories[choice]

        individual = pd.DataFrame(dict(
            r = df_single[chosen_columns].values[0].tolist(),
            theta = chosen_columns
        ))

        demographic_info = '''**Age**: {}
        
**Height**: {} cm

**Weight**: {} kg

**Gender**: {}

**Number of Siblings** : {}
        '''.format(int(df_single['Age'].tolist()[0]),
                    df_single['Height'].tolist()[0],
                    df_single['Weight'].tolist()[0],
                    ["Male" if int(x) == 5 else "Female" for x in df_single['Gender'].tolist()][0],
                    int(df_single['Number of siblings'].tolist()[0]))

        fig = px.line_polar(individual,
                            r='r',
                            theta='theta',
                            line_close=True,
                            hover_data=['theta','r'])
        fig.update_traces(fill='toself', line_color = default_colors[int(df_single['clusterGrouping{}'.format(CURRENT_CLUSTER_NO)].tolist()[0])])

        return fig, demographic_info


    # updates the 3d scatterplot of survey response clusters
    @app.callback(
        Output('3d-plot', 'figure'),
        [Input('cluster-dropdown', 'value')]
    )
    def update_figure(clusterNo):

        global df
        global CURRENT_CLUSTER_NO
        
        #reset the current clusterNo
        CURRENT_CLUSTER_NO = clusterNo

        fig = go.Figure()

        for x in range(clusterNo):
            fig.add_trace(go.Scatter3d(x=df[df["clusterGrouping{}".format(clusterNo)] == str(x)]['xcoord'].tolist(),
                                        y=df[df["clusterGrouping{}".format(clusterNo)] == str(x)]['ycoord'].tolist(),
                                        z=df[df["clusterGrouping{}".format(clusterNo)] == str(x)]['zcoord'].tolist(),
                                        mode='markers',
                                        marker={'color' : default_colors[x]},
                                        name="Cluster {}".format(x)
                                    )
            )

        fig.update_layout(showlegend=True, legend={'x' : 0.8, 'y' : 0.9, 'bordercolor': "Black", 'borderwidth': 1},
                            # title={'text':"Plot by top 3 principal components, color by K-means cluster",
                            #         'y' : 0.9,
                            #         'yanchor' : 'middle',
                            # },
                            # titlefont={'size':25},
                            scene={
                                'xaxis_title':'PC1',
                                'yaxis_title':'PC2',
                                'zaxis_title':'PC3',
                            },
                            margin = {
                                't' : 5
                            }
        )

        return  fig


    # updates the heatmap of correlation coefficients
    @app.callback(
        Output('corr-heatmap', 'figure'),
        [Input('question-dropdown', 'value'),
        Input('3d-plot', 'clickData')]
    )
    def update_heatmap(category_choices, click_data):
        
        #choose cluster
        if click_data:
            chosen_cluster = str(click_data["points"][0]['curveNumber'])
        else:
            chosen_cluster = "0"
        
        df_selected = df[df['clusterGrouping' + str(CURRENT_CLUSTER_NO)] == chosen_cluster]
            
        #choose columns
        chosen_columns = []
        for choice in category_choices:
            chosen_columns = chosen_columns + question_categories[choice]

        df_selected = df_selected[chosen_columns]
        
        #matrix of correlated values
        #masked with triangular matrix (so don't have repeat values)
        matrix = df_selected.corr().values.tolist()
        mask = np.tri(len(df_selected.columns), k=0)
        matrix = np.ma.array(matrix, mask=mask)  #bug here, with categorical data

        fig = go.Figure(data=go.Heatmap(z=[np.flip(x) for x in matrix.filled(np.nan)],
                                        x=[x for x in np.flip(df_selected.columns)[:-1]],
                                        y=[x for x in df_selected.columns[:-1]],
                                        colorscale=px.colors.diverging.balance,
                                        hoverongaps=False,
                                        zmid=0,
                                        )
                        )

        fig.update_layout(title={'text' : "Correlation Coefficients in Cluster {}".format(chosen_cluster),
                                    'x' : 0.57,
                                    'xanchor' : 'center',
                                    'y' : 0.9,
                                    'yanchor' : 'middle'
                                },
                            titlefont={'size' : 25},
                            xaxis={'showgrid' : False},
                            yaxis={'showgrid' : False},
                            plot_bgcolor='rgb(255,255,255)'
                        )
        
        return fig


    # updates the violin plot of survey responses
    @app.callback(
        Output('violins', 'figure'),
        [Input('question-dropdown', 'value'),
        Input('cluster-dropdown', 'value')]
    )
    def update_violins(category_choices, number_of_clusters):

        chosen_columns = []
        for choice in category_choices:
            chosen_columns = chosen_columns + question_categories[choice]

        fig = go.Figure()

        for column in chosen_columns:
            for clusterNo in range(number_of_clusters):
                fig.add_trace(go.Violin(y=df[column][df["clusterGrouping{}".format(number_of_clusters)] == str(clusterNo)],
                                        name=column,
                                        legendgroup="Cluster {}".format(clusterNo),
                                        #box_visible=True,
                                        meanline_visible=True,
                                        line_color=default_colors[clusterNo],
                                        opacity=0.4)
                            )

        #for clusterNo in range(number_of_clusters):
        # clusterNo = 1

        # fig.add_trace(go.Violin(y=df[df["clusterGrouping{}".format(number_of_clusters)] == str(clusterNo)],
        #                         x=chosen_columns,
        #                         legendgroup="Cluster {}".format(clusterNo),
        #                         box_visible=True,
        #                         meanline_visible=True,
        #                         line_color=default_colors[clusterNo],
        #                         opacity=0.4,
        #                         points='all')
        #             )

        fig.update_traces(side='positive', points=False, width = 1.5, hoverinfo='skip')
        fig.update_layout(xaxis_showgrid=False,
                            xaxis_zeroline=False,
                            width=len(chosen_columns) * 100,
                            autosize=False,
                            height=750,
                            margin={
                                't' : 5,
                                'l' : 5
                            })

        return fig

    app.run_server(debug=True)

if __name__ == "__main__":
    render_visualization()
