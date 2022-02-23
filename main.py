# data manipulation
import pandas as pd
import numpy as np
from collections import OrderedDict

# plotly 
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# dashboards
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

# baseball package
from pybaseball import statcast

# data manipulation
df_list = []
for i in ['a', 'b', 'c', 'd', 'e', 'f', 'g']:
    df_list.append(pd.read_csv('pitch_data_2021_'+i+'.csv'))
data = pd.concat(df_list, axis = 0)

name_id = {' '.join(i.split(', ')[::-1]) : j for i, j in zip(data.player_name, data.pitcher)}
pitch_type_dict = {'All':'All', 'FF': 'Fastball', 'SL': 'Slider', 'CH':'Changeup', 'SI': 'Sinker', 'CU': 'Curveball', 'FC': 'Cutter', 'KC':'Knuckle-curve', 'FS': 'Splitter'}
pitch_type_dict.pop('All')
color_dict = {i:j for i, j in zip(pitch_type_dict.keys(), px.colors.qualitative.Set1[:8])}
description_color = {i:j for i, j in zip(data.description.unique()[:-1], px.colors.qualitative.Set3)}
pitch_type_dict = {'All':'All Pitch Type','FF': 'Fastball', 'SL': 'Slider', 'CH':'Changeup', 'SI': 'Sinker', 'CU': 'Curveball', 'FC': 'Cutter', 'KC':'Knuckle-curve', 'FS': 'Splitter'}






# app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUMEN])
server = app.server


app.layout = html.Div([
    html.H1('Pitch Performance in 2021'),
    html.A(children = 'Data Source: statcast from pybaseball', href = 'https://pypi.org/project/pybaseball/', target="_blank"),
    html.Br(),
    html.Br(),
    # Options
    html.Div([
        html.Div(children = [
            html.Label('1. Select Name ( >200 total pitches)'),
            html.Br(),
            dcc.Dropdown(
                id = 'name_list',
                options = [{'label': i, 'value': name_id[i]} for i in OrderedDict(sorted(name_id.items())).keys()],
                ),
            ], style = {'width': '45%', 'height':'100px', 'display': 'inline-block', 'text-align':'left'}
        ),
        html.Div(children = [
            html.Label(id = 'pt_pop'),
            html.Br(),
            dcc.Dropdown(
                id='pt_list',
                ),
            ], style = {'width': '45%', 'height':'100px', 'display': 'inline-block', 'text-align':'left'}
        ),
    ], style={'width': '100%', "display":"inline-block", 'text-align':'center'}),
    
    #Graphs
    dcc.Tabs([
            #tab1
            dcc.Tab(label = 'Fundamental Data', children = [
                html.Div([
                    html.Div([
                        # pie
                        dcc.Loading(dcc.Graph(id='pie'
                            )),
                        ], style={'width': '45%', "display":"inline-block", 'text-align':'left'}),
                    html.Div([
                        # score
                        dcc.Loading(dcc.Graph(id='score'
                            )),
                        ], style={'width': '45%', "display":"inline-block", 'text-align':'left'}),
                ], style={'width': '100%', "display":"inline-block", 'text-align':'center'})
            ]),


            #tab2
            dcc.Tab(label = 'Pitch Features', children = [
                html.Div([
                    html.Div([
                        # spin
                        dcc.Loading(dcc.Graph(id='spin'
                            )),
                        ], style={'width': '50%', "display":"inline-block", 'text-align':'left'}),
                    html.Div([
                        #  track
                        dcc.Loading(dcc.Graph(id='track'
                            )),
                        ], style={'width': '50%', "display":"inline-block", 'text-align':'left'}),
                ], style={'width': '100%', "display":"inline-block", 'text-align':'center'})
            ]),         


            #tab3
            dcc.Tab(label = 'Pitch Zone', children = [
                html.Div([
                    html.Div([
                        #  location
                        dcc.Loading(dcc.Graph(id='location'
                            )),
                        ], style={'width': '45%', "display":"inline-block", 'text-align':'left'}),
                    html.Div([
                        #  zone
                        dcc.Loading(dcc.Graph(id='zone'
                            )),
                        ], style={'width': '45%', "display":"inline-block", 'text-align':'left'}),
                ], style={'width': '100%', "display":"inline-block", 'text-align':'center'})
            ]),         
        ]),
    ])



# pt options producer
@app.callback(
    Output('pt_list', 'options'),        
    Output('pt_pop', 'children'),
    Input('name_list', 'value'))
def pt_option_producer(player_id):
    options = ['All']
    player_data = data[(data.pitcher == player_id) & (data['pitch_type'].isin(pitch_type_dict))]
    for i in player_data.groupby('pitch_type')['game_date'].count().sort_values(ascending = False).index:
        if player_data[(player_data.pitcher == player_id) & (player_data['pitch_type'].isin(pitch_type_dict))].groupby('pitch_type')['game_date'].count()[i] > 10:
            options.append(i)
    return [{'label': pitch_type_dict[i], 'value': i} for i in options],  '2. Select Pitch Type of the Pitcher ( >10 pitches )'



# pie function
@app.callback(
    Output('pie', 'figure'),
    Input('pt_list', 'value'),
   State('name_list', 'value'))        
def pie(pt, player_id):
    player_data = data[(data.pitcher == player_id) & (data['pitch_type'].isin(pitch_type_dict))]
    player_data = player_data[player_data.pitch_type.isin(player_data.groupby('pitch_type')['game_date'].\
                                            count()[player_data.groupby('pitch_type')['game_date'].count()>10].index)]
    pitch_data = player_data.groupby('pitch_type')['game_date'].count().sort_values(ascending = False).reset_index()
    pull_list = [0] * (pitch_data.shape[0])
    if pt != 'All':
        pull_list[pitch_data.index[pitch_data['pitch_type'] == pt][0]] = 0.3
    fig1 = go.Figure(go.Pie(
        name = "",
        values = pitch_data['game_date'],
        labels = pitch_data['pitch_type'],
        hovertemplate = "Pitch Type:%{label}: <br>Number of Pitch: %{value}",
        direction ='clockwise',
        pull = pull_list
    ))
    fig1.update_traces(marker=dict(colors=[color_dict[i] for i in pitch_data.pitch_type.to_list()]))
    fig1.update_layout(
        width=600,
        height=600,
        title = 'Distribution of Pitch Type for {}'.format(' '.join(player_data['player_name'].unique()[0].split(', ')[::-1])),
        title_x=0.5)
    return fig1



# score function
@app.callback(
    Output('score', 'figure'),
    Input('pt_list', 'value'),
   State('name_list', 'value'))        
def score(pt, player_id):
    player_data = data[(data.pitcher == player_id) & (data['pitch_type'].isin(pitch_type_dict))]
    player_data = player_data[player_data.pitch_type.isin(player_data.groupby('pitch_type')['game_date'].\
                                            count()[player_data.groupby('pitch_type')['game_date'].count()>10].index)]
    if pt == 'All':
        pitch_data = player_data
        all_pitch_data = data
    else:
        pitch_data = player_data[player_data['pitch_type'] == pt]
        all_pitch_data = data[data['pitch_type'] == pt]
    ave_speed = pitch_data['release_speed'].mean()
    ave_extension = pitch_data['release_extension'].mean()
    ave_woba = pitch_data['woba_value'].mean()
    strike_per = pitch_data.loc[pitch_data.type == 'S', 'type'].count()/pitch_data.type.count()
    swing_per = pitch_data.loc[pitch_data.description.isin(['swinging_strike', 'swinging_strike_blocked']) , 'description'].count()/pitch_data.description.count()
    data_strike = all_pitch_data.groupby('pitcher').apply(lambda x: x.loc[x.type == 'S', 'type'].count()/x.type.count())
    data_swing = all_pitch_data.groupby('pitcher').apply(lambda x: x.loc[x.description.isin(['swinging_strike', 'swinging_strike_blocked']) , 'description'].count()/x.description.count())
    speed_score = stats.percentileofscore(all_pitch_data['release_speed'].dropna(), ave_speed)
    extension_score = stats.percentileofscore(all_pitch_data['release_extension'].dropna(), ave_extension)
    woba_score = 100 - stats.percentileofscore(all_pitch_data['woba_value'].dropna(), ave_woba)
    strike_score = stats.percentileofscore(data_strike, strike_per)
    swing_score = stats.percentileofscore(data_swing, swing_per)
    overall_score = round((speed_score+extension_score+woba_score+strike_score+swing_score)/5, 1)
    score_df = pd.DataFrame(dict(r = [speed_score, extension_score, woba_score, strike_score, swing_score], theta = ['speed', 'extension', 'wOBA', 'strike', 'swing']))
    fig2 = px.line_polar(score_df, r='r', theta='theta', line_close=True, 
                    title = '{0} Status of {1} \t\t\t\t\t\t\t\tOverall : {2}'.format(pitch_type_dict[pt] ,' '.join(player_data['player_name'].unique()[0].split(', ')[::-1]), overall_score),)
    fig2.update_traces(fill='toself')
    fig2.update_traces(name=f'percentile score', showlegend = True, hovertemplate = None, hoverinfo = 'skip')
    fig2.update_layout(
        title_x=0.5,
        width=600,
        height=600,
      polar=dict(
        radialaxis=dict(
          range = [0, 100],
          visible=True,
        ),
      ),
      showlegend=True,
    )
    return fig2


# spin function
@app.callback(
    Output('spin', 'figure'),
    Input('pt_list', 'value'),
   State('name_list', 'value'))        
def spin(pt, player_id):
    player_data = data[(data.pitcher == player_id) & (data['pitch_type'].isin(pitch_type_dict))]
    player_data = player_data[player_data.pitch_type.isin(player_data.groupby('pitch_type')['game_date'].\
                                            count()[player_data.groupby('pitch_type')['game_date'].count()>10].index)]
    pitch_data = player_data.groupby('pitch_type')[['release_spin_rate', 'spin_axis']].mean().sort_values('pitch_type').reset_index()
    ave_spin_degree = pitch_data.spin_axis.to_list()
    ave_spin_rate = pitch_data.release_spin_rate.to_list()
    width_list = [0] * (pitch_data.shape[0])
    if pt != 'All':
        width_list[pitch_data.index[pitch_data['pitch_type'] == pt][0]] = 5
        text = 'spin rate: {0} RPM<br>spin axis: {1}'.format(round(ave_spin_rate[pitch_data.index[pitch_data['pitch_type'] == pt][0]]), round(ave_spin_degree[pitch_data.index[pitch_data['pitch_type'] == pt][0]]))
    else:
        text = ''
    fig3 = go.Figure(go.Barpolar(
        customdata = pitch_data['pitch_type'],
        r= ave_spin_rate,
        theta=list(np.array(ave_spin_degree) - 180),
        width=[15]*pitch_data.shape[0],
        marker_line_color="black",
        marker_line_width= width_list,
        opacity= 0.8,
        hovertemplate = 'Pitch Type:%{customdata}',
        name = ''
    ))
    fig3.update_traces(marker=dict(color=[color_dict[i] for i in pitch_data.pitch_type.to_list()]))
    fig3.update_layout(
        width = 700,
        height = 700,
        title = {'text': "{0} Spin rate and axis for {1} ({2}) <br> <br><sup>{3}</sup>".\
        format(pitch_type_dict[pt], ' '.join(player_data['player_name'].unique()[0].split(', ')[::-1]), player_data.p_throws.unique()[0], text),'x': 0.1},
        template=None,
        polar = dict(
            radialaxis = dict(range=[0, data['release_spin_rate'].max()], showticklabels=True, ticks=''),
            angularaxis = dict(showticklabels=False, ticks='', rotation = 90, direction = 'clockwise'),
        ),
    )
    return fig3



# location function
@app.callback(
    Output('location', 'figure'),
    Input('pt_list', 'value'),
    State('name_list', 'value'))        
def location(pt, player_id):
    player_data = data[(data.pitcher == player_id) & (data['pitch_type'].isin(pitch_type_dict))]
    player_data = player_data[player_data.pitch_type.isin(player_data.groupby('pitch_type')['game_date'].\
                                            count()[player_data.groupby('pitch_type')['game_date'].count()>10].index)]
    if pt == 'All':
        pitch_data = player_data
    else:
        pitch_data = player_data[player_data['pitch_type'] == pt]
    fig4 = px.scatter(pitch_data, x='plate_x', y = 'plate_z', color = 'description', color_discrete_map = description_color, title="{0} Pitch Location for {1}".format(pitch_type_dict[pt],' '.join(player_data['player_name'].unique()[0].split(', ')[::-1])))
    fig4.update_traces(hovertemplate=None, hoverinfo = 'skip' )
    fig4.update_layout(
        {'legend_title_text': ''},
        title_x=0.5,
        width=650,
        height=650)
    return fig4


# zone function
@app.callback(
    Output('zone', 'figure'),
    Input('pt_list', 'value'),
    State('name_list', 'value'))        
def zone(pt, player_id):
    df_list = []
    for i in range(1,10):
        zone_data = data.dropna(subset = ['zone'])
        df = [i, zone_data.loc[zone_data.zone ==i, 'plate_x'].min(), zone_data.loc[zone_data.zone ==i, 'plate_x'].max(), zone_data.loc[zone_data.zone ==i, 'plate_z'].min(), zone_data.loc[zone_data.zone ==i, 'plate_z'].max()]
        df_list.append(df)
    zone_metrix = pd.DataFrame(df_list, columns = ['zone', 'min_x', 'max_x', 'min_z', 'max_z']).set_index('zone')
    a, b, c, d, e, f = round((zone_metrix.iloc[0:3, 3].mean() - zone_metrix.iloc[6:9, 2].mean())*2/3 + zone_metrix.iloc[6:9, 2].mean(), 2),\
                               round((zone_metrix.iloc[0:3, 3].mean() - zone_metrix.iloc[6:9, 2].mean())*1/3 + zone_metrix.iloc[6:9, 2].mean(), 2),\
                               round(zone_metrix.iloc[6:9, 2].mean(), 2),\
                               round(zone_metrix.iloc[0:3, 3].mean(), 2),\
                               round((zone_metrix.iloc[0:3, 3].mean() - zone_metrix.iloc[6:9, 2].mean())*2/3 + zone_metrix.iloc[6:9, 2].mean(), 2),\
                               round((zone_metrix.iloc[0:3, 3].mean() - zone_metrix.iloc[6:9, 2].mean())*1/3 + zone_metrix.iloc[6:9, 2].mean(), 2)
    zone_metrix.iloc[0:3, 2] = a
    zone_metrix.iloc[3:6, 2] = b
    zone_metrix.iloc[6:9, 2] = c
    zone_metrix.iloc[0:3, 3] = d
    zone_metrix.iloc[3:6, 3] = e
    zone_metrix.iloc[6:9, 3] = f
    player_data = data[(data.pitcher == player_id) & (data['pitch_type'].isin(pitch_type_dict))]
    player_data = player_data[player_data.pitch_type.isin(player_data.groupby('pitch_type')['game_date'].\
                                            count()[player_data.groupby('pitch_type')['game_date'].count()>10].index)]
    if pt == 'All':
        pitch_data = player_data
    else:
        pitch_data = player_data[player_data.pitch_type == pt]
    zone_woba = pd.DataFrame(pitch_data.groupby('zone')['woba_value'].mean()).reset_index()
    for i in range(0, 9):
        zone_woba.loc[i, 'score'] = stats.percentileofscore(zone_woba.loc[zone_woba['zone'] < 10,'woba_value'], zone_woba.loc[i, 'woba_value'])
    zone_woba['red'] = [255 if i > 50 else 0 for i in zone_woba['score']]
    zone_woba['blue'] = [255 if i < 50 else 0 for i in zone_woba['score']]
    zone_woba['opacity'] = [(50 - i)*0.014 if i < 50 else (i - 50)*0.014 for i in zone_woba['score']]
    zone_woba['opacity'] = zone_woba['opacity'].fillna(0)
    fig5 = go.Figure()
    for i in range(0, 9):
        fig5.add_trace(go.Scatter(
                                        x=[zone_metrix.iloc[i, 0], zone_metrix.iloc[i, 0], zone_metrix.iloc[i, 1], zone_metrix.iloc[i, 1], [zone_metrix.iloc[i, 0]]],
                                        y=[zone_metrix.iloc[i, 2], zone_metrix.iloc[i, 3], zone_metrix.iloc[i, 3], zone_metrix.iloc[i, 2], zone_metrix.iloc[i, 2]],
                                        line=dict(color="white"),
                                        fill = 'toself', 
                                        fillcolor ='rgba({0}, 0, {1}, {2})'.format(zone_woba.iloc[i, 3], zone_woba.iloc[i, 4], zone_woba.iloc[i, 5]),
                                        hovertemplate = None,
                                        hoverinfo = 'skip'
                                        )
                                 )
        fig5.add_annotation(text = round(zone_woba.iloc[i, 1], 3),
                                          x = np.mean([zone_metrix.iloc[i, 0], zone_metrix.iloc[i, 1]]),
                                          y = np.mean([zone_metrix.iloc[i, 2], zone_metrix.iloc[i, 3]]),
                                          showarrow = False)
    fig5.update_layout(plot_bgcolor='white', 
                                    showlegend = False, 
                                    width=650,
                                    height=650,
                                    title = '{0} Strike Zone wOBA for {1}'.format(pitch_type_dict[pt], pitch_type_dict[pt] ,' '.join(player_data['player_name'].unique()[0].split(', ')[::-1])),
                                   title_x=0.5, 
                                   xaxis = go.XAxis(
                                                        title = '',
                                                        showticklabels=False
                                                        ),
                                    yaxis = go.YAxis(
                                                        title = '',
                                                        showticklabels=False
                                                        )
                     )

    return fig5



# track function
@app.callback(
    Output('track', 'figure'),
    Input('pt_list', 'value'),
    State('name_list', 'value'))        
def track(pt, player_id):
    player_data = data[(data.pitcher == player_id) & (data['pitch_type'].isin(pitch_type_dict))]
    player_data = player_data[player_data.pitch_type.isin(player_data.groupby('pitch_type')['game_date'].\
                                            count()[player_data.groupby('pitch_type')['game_date'].count()>10].index)]
    if pt == 'All':
        pitch_data = player_data.groupby('pitch_type').mean()
    else:
        pitch_data = player_data[player_data.pitch_type == pt].groupby('pitch_type').mean()
    motion_data = pitch_data[['release_speed', 'vx0', 'ax', 'vy0', 'ay', 'vz0', 'az', 'release_pos_x','release_pos_z', 'plate_x', 'plate_z']].rename(columns = {'release_pos_x': 'release_pos_y', 'plate_x':'plate_y'})
    motion_data['release_pos_x'] = 55
    motion_data = motion_data.reset_index()
    motion_data['pitch_type'] = pd.Categorical(motion_data['pitch_type'], categories = list(player_data.groupby('pitch_type')['game_date'].count().sort_values(ascending = False).index))
    motion_data = motion_data.sort_values(['pitch_type'])
    strike = player_data[player_data.description == 'called_strike'].groupby('batter').mean()
    right, left, top, bottom = strike.plate_x.max(), strike.plate_x.min(), strike.plate_z.max(), strike.plate_z.min()
    for j in motion_data.index:
            distance = np.sqrt((0-55)**2 + (motion_data.iloc[j, 8] - motion_data.iloc[j, 10])**2 + (motion_data.iloc[j, 9] - motion_data.iloc[j, 11])**2)
            time = distance/motion_data.iloc[j,1]*5280/3600
            x, y, z = [], [], []
            vx0 = motion_data.iloc[j,2]
            ax = motion_data.iloc[j, 3]
            vy0 = motion_data.iloc[j,4]
            ay = motion_data.iloc[j, 5]
            vz0 = motion_data.iloc[j,6]
            az = motion_data.iloc[j, 7]
            for i in np.arange(0, time, 0.005):
                y.append(motion_data.iloc[j,8] +  vx0*i + 0.5*ax*i**2)
                x.append(55 - (motion_data.iloc[j,-1] +  vy0*i + 0.5*ay*i**2))
                z.append(motion_data.iloc[j,9] +  vz0*i + 0.5*az*i**2)  
    if pt == 'All':
        fig6 = go.Figure()
        for j in motion_data.index:
                distance = np.sqrt((0-55)**2 + (motion_data.iloc[j, 8] - motion_data.iloc[j, 10])**2 + (motion_data.iloc[j, 9] - motion_data.iloc[j, 11])**2)
                time = distance/motion_data.iloc[j,1]*5280/3600
                x, y, z = [], [], []
                vx0 = motion_data.iloc[j,2]
                ax = motion_data.iloc[j, 3]
                vy0 = motion_data.iloc[j,4]
                ay = motion_data.iloc[j, 5]
                vz0 = motion_data.iloc[j,6]
                az = motion_data.iloc[j, 7]
                for i in np.arange(0, time, 0.005):
                    y.append(motion_data.iloc[j,8] +  vx0*i + 0.5*ax*i**2)
                    x.append(55 - (motion_data.iloc[j,-1] +  vy0*i + 0.5*ay*i**2))
                    z.append(motion_data.iloc[j,9] +  vz0*i + 0.5*az*i**2)
                fig6.add_trace(
                        go.Scatter3d(
                            x= x, y=y, z = z,
                            name = motion_data.iloc[j, 0],
                            mode = 'lines',
                            line=dict(color=color_dict[motion_data.iloc[j, 0]], width=4),
                            hovertext=[],
                            hoverinfo="text",
                        ))
        fig6.add_trace(go.Scatter3d(
            x= [55,55,55,55,55],
            y= [left, left, right, right, left],
            z= [bottom, top, top, bottom, bottom],
            mode = 'lines',
            name = 'strike',
            line=dict(color= 'black', width=2),
            hoverinfo = 'none',
            ))
        for trace in fig6['data']: 
            trace['hoverinfo'] = 'skip'
            if(trace['name'] == 'strike'): 
                trace['showlegend'] = False
        fig6.update_layout(
            plot_bgcolor = 'white',
            width = 700,
            height = 700,
            title = '{0} Trajectory for {1}'.format(pitch_type_dict[pt] ,' '.join(player_data['player_name'].unique()[0].split(', ')[::-1])),
            scene = dict(
                        aspectmode='manual', 
                        aspectratio=dict(x=55, y= 10, z= 7.5),
                        xaxis = dict(showticklabels=False, range=[0, 55],),
                        yaxis = dict(showticklabels=False, range=[-5,5],),
                        zaxis = dict(showticklabels=False, range=[0, 7.5],),
                        xaxis_title= '',
                        yaxis_title='',
                        zaxis_title='',
                        bgcolor='white',
                        camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=36, y=3, z=0.5)),
                    ),
        )
    else:
        fig6 = go.Figure(
                data = [go.Scatter3d(
                    x= [55,55,55,55,55],
                    y= [left, left, right, right, left],
                    z= [bottom, top, top, bottom, bottom],
                    mode = 'lines',
                    name = 'strike',
                    line=dict(color= 'black',width=2),
                    hovertext=[],
                    hoverinfo="text"
                    ),
                    go.Scatter3d(
                            x= x, y=y, z = z,
                            name = pt,
                            marker=dict(size=2, color= color_dict[motion_data.iloc[j, 0]]),
                            line=dict(color=color_dict[motion_data.iloc[j, 0]], width=2),
                            hovertext=[],
                            hoverinfo="text"
                        )
                ],
                layout = go.Layout(
                    plot_bgcolor = 'white',
                    width = 700,
                    height = 700,
                    showlegend = False,
                    updatemenus=[dict(type="buttons", buttons=[dict(label="Play", method="animate", args=[None, {"frame": {"duration": 1}}])])],
                    title = '{0} Trajectory for {1}'.format(pitch_type_dict[pt] ,' '.join(player_data['player_name'].unique()[0].split(', ')[::-1])),
                    scene = dict(
                                aspectmode='manual', 
                                aspectratio=dict(x=55, y= 10, z= 7.5),
                                xaxis = dict(showticklabels=False, range=[0, 55],),
                                yaxis = dict(showticklabels=False, range=[-5,5],),
                                zaxis = dict(showticklabels=False, range=[0, 7.5],),
                                xaxis_title= '',
                                yaxis_title='',
                                zaxis_title='',
                                bgcolor='white',
                                camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=36, y=0, z=0)),
                            ),
                        ),
                frames=[go.Frame(
                    data=[go.Scatter3d(
                        x=[x[k]],
                        y=[y[k]],
                        z=[z[k]],
                        name = pt,
                        mode = 'markers',
                        marker=dict(size=2, color= color_dict[motion_data.iloc[j, 0]]),
                        hovertext=[],
                        hoverinfo="text"
                        ),
                    go.Scatter3d(
                        x= list([[55,55,55,55,55]]*200)[k],
                        y= list([[left, left, right, right, left]]*200)[k],
                        z= list([[bottom, top, top, bottom, bottom]]*200)[k],
                        mode = 'lines',
                        name = 'strike',
                        line=dict(color= 'black', width=2),
                        hovertext=[],
                        hoverinfo="text"
                        ),
                     ],
                )
                    for k in range(198)]
        )
    return fig6



if __name__ == '__main__':
        app.run_server(debug=False, port=8899)
