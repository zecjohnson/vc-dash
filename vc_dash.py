import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import joblib
from google.cloud import bigquery
import db_dtypes
from scipy.stats import randint

#import algorithm libraries

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

#import sklearn libraries

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# import dash & bootstrap libraries

import dash
from dash import Dash, dcc, html, Output, Input, State, callback
import plotly.express as px
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# load the data

VC = bigquery.Client(project='vcbeach-x-zechariah-johnson')
table_ref = VC.dataset('Raw_Datasets').table('Database_Elite16_Challenge')

# Construct the SQL query to filter rows based on the desired condition

query = f"""
    SELECT *
    FROM `{table_ref.project}.{table_ref.dataset_id}.{table_ref.table_id}`
    WHERE Skill = 'Attack'
"""

# Execute the query and retrieve the filtered results
df_attack = VC.query(query).to_dataframe()

# add column RedZone

df_attack['RedZone'] = df_attack.apply(lambda x: 1 if ((x['Set'] == 1 or x['Set'] == 2) and 
                                         (x['OpponentScore'] >= 16 or x['TeamScore'] >= 16) and 
                                         (x['ScoreDifference'] <= 2)) else 0, axis=1)

# creating new feature called AttackOutcome

df_attack['AttackOutcome'] = df_attack.apply(lambda x: 1 if x['GradeAttack'] == 'Perfect'
                               else 0 if x['GradeAttack'] in ['Error', 'Incomplete']
                               else 0.25 if x['GradeAttack'] == 'Poor'
                               else 0.75 if x['GradeAttack'] == 'Positive'
                               else '', axis=1)


# Pre-Process the Data

df_pp = df_attack.copy()



df_pp = df_pp[['AttackBlockType', 'AttackLocation', 'AttackStyle', 'AttackType', 'Phase', 'SetType',
               'ReceiveSide', 'ServeType', 'Set', 'TournamentPhase', 'From_Zone_X', 'From_Zone_Y', 'Zone_X',
               'Zone_Y', 'To_Zone_X', 'To_Zone_Y', 'RedZone', 'Weather', 'SetStatus',
               'ScorePlusMinus', 'ScoreDifference', 'OpponentSetWon', 'TeamSetWon', 'Wind',
               'GameState', 'TeamScore', 'OpponentScore', 'Light', 'RallyNumber', 'New_Player',
               'ContactNumber', 'AttackOutcome', 'Gender']].copy()
'''
df_pp = df_pp[['Weather','TournamentPhase', 'SetType', 'SetStatus', 'Set', 'ServeType', 
               'ScorePlusMinus', 'ScoreDifference',
       'ReceiveSide', 'AttackBlockType', 'OpponentSetWon', 'TeamSetWon', 'Zone_X', 'AttackType', 'Wind', 
               'GameState', 'AttackLocation', 'From_Zone_X', 'Phase',
       'TeamScore', 'Zone_Y',
       'OpponentScore', 'Light', 'AttackStyle', 'RallyNumber', 'New_Player',
       'From_Zone_Y', 'ContactNumber', "RedZone", 'AttackOutcome', 'Gender', 'To_Zone_X', 'To_Zone_Y']].copy() '''

# creating copy of data to perform One Hot Encoding

df_ohe = df_pp.copy()

# creating dummy variables for all categorical variables except AttackOutcome and New_Player

cols_to_encode = [col for col in df_ohe.columns if col not in ["AttackOutcome", "New_Player", "Gender"]]
df_ohe_encoded = pd.get_dummies(df_ohe[cols_to_encode])
df_ohe = pd.concat([df_ohe_encoded, df_ohe[["AttackOutcome", "New_Player", "Gender"]]], axis=1)
encoded = list(df_ohe.columns)

# create copy of data that includes new_player and gender
df_ohe_full = df_ohe.copy()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

logo_color = '#eb0029'

# displaying options for dropdowns

all_columns = df_pp.columns.tolist()
all_columns_eo = df_ohe_full.columns.tolist()
all_players = df_pp['New_Player'].unique()

excluded_features_constant = ['Weather', 'ScorePlusMinus', 'ScoreDifference', 'OpponentSetWon', 'TeamSetWon', 'Wind', 'GameState', 'TeamScore', 'OpponentScore', 'Light', 'RallyNumber', 'Gender', 'New_Player', 'AttackOutcome']

excluded_features_variable = ['Weather', 'ScorePlusMinus', 'ScoreDifference', 'OpponentSetWon', 'TeamSetWon', 'Wind', 'GameState', 'TeamScore', 'OpponentScore', 'Light', 'RallyNumber', 'Gender', 'New_Player', 'Zone_X', 'Zone_Y', 'To_Zone_X', 'To_Zone_Y', 'From_Zone_X', 'From_Zone_Y', 'AttackOutcome']

gender_options = [
    {'label': 'Men', 'value': 'Men'},
    {'label': 'Women', 'value': 'Women'}
]

const_feat_options = [{'label': column, 'value': column} for column in all_columns if column not in excluded_features_constant]

var_feat_options = [{'label': column, 'value': column} for column in all_columns if column not in excluded_features_variable]

eo_const_feat_options = [{'label': column, 'value': column} for column in all_columns_eo]

player_options = [{'label': value, 'value': value} for value in all_players]



# function to generate dataframe

def generate_df(df, gender, constant_feature, constant_value, var1, var2):
    filtered_df = df[df['Gender'] == gender]
    filtered_df = filtered_df[filtered_df[constant_feature] == constant_value]
    grouped_df = filtered_df.groupby([var1, var2])
    average_outcome = grouped_df['AttackOutcome'].mean()
    num_instances = grouped_df.size().reset_index(name='num_instances')
    result_df = pd.DataFrame({'AverageAttackOutcome': average_outcome}).reset_index()
    result_df = pd.merge(result_df, num_instances, on=[var1, var2])
    return result_df

def generate_df_pp(df, gender, constant_feature, constant_value, var1, var2, player):
    filtered_df = df[df['Gender'] == gender]
    filtered_df = df[df['New_Player'] == player]
    filtered_df = filtered_df[filtered_df[constant_feature] == constant_value]
    grouped_df = filtered_df.groupby([var1, var2])
    average_outcome = grouped_df['AttackOutcome'].mean()
    num_instances = grouped_df.size().reset_index(name='num_instances')
    result_df = pd.DataFrame({'AverageAttackOutcome': average_outcome}).reset_index()
    result_df = pd.merge(result_df, num_instances, on=[var1, var2])
    return result_df

def generate_df_eo(df, gender, constant_feature, constant_value, player):
    filtered_df = df[df['Gender'] == gender]
    filtered_df = filtered_df[filtered_df['New_Player'] == player]
    filtered_df = filtered_df[filtered_df[constant_feature] == constant_value]
    columns_to_remove = ['Gender', 'New_Player', 'To_Zone_X', 'To_Zone_Y']
    filtered_df = filtered_df.drop(columns_to_remove, axis=1)


    average_outcome = filtered_df['AttackOutcome'].mean()
    df_without_attack_outcome = filtered_df.drop('AttackOutcome', axis=1)

    if gender == 'Men':
        model = joblib.load('mens_chal_e16.joblib')
    elif gender == 'Women':
        model = joblib.load('womens_chal_e16.joblib')
    else:
        raise ValueError("Invalid gender specified.")

    expected_outcome = model.predict(df_without_attack_outcome)
    average_expected_outcome = expected_outcome.mean()
    num_instances = len(expected_outcome)

    result_df = pd.DataFrame({
        'AverageAttackOutcome': [average_outcome],
        'num_instances': [num_instances],
        'ExpectedOutcome': [average_expected_outcome]
    })

    return result_df

# App Layout

app.layout = html.Div(children=[
    html.Div(
        style={
            'padding-top': 0,
            'padding-bottom': 30,
            'text-align': 'center',
            'backgroundColor': "black",
        },
        children=[
            html.Img(src="assets/VC_text.jpg", height="110px", style={'marginTop': 14}),
            html.Header(
                'Beach Volleyball Attacking',
                style={
                    'color': logo_color,
                    'font-weight': "bold",
                    'fontSize': 36
                }
            ),
            dbc.RadioItems(
                id="radios",
                className="btn-group",
                inputClassName="btn-check",
                labelClassName="btn btn-outline-danger",
                labelCheckedClassName="active",
                style={'marginTop': 14, 'width':'400px'},
                options=[
                    {"label": "International Trends", "value": 1},
                    {"label": "Player Analysis", "value": 2},
                    {"label": "Expected Outcomes", "value": 3}
                ],
                value=1,
            ),
        ]
    ), 
    dbc.Container(
    fluid=True,
    children=[
        dbc.Row(
            children=[
                dbc.Col(
                    width=2,
                    children=[
                        html.Div(
                            style={
                                'padding-top': 20,
                                'padding-bottom': 25,
                                'padding-left': 10
                            },
                            children=[
                                html.H5(children='', id='high-level-output', style={'marginBottom': 12}),
                                dcc.RadioItems(
                                    options = gender_options,
                                    value='men',
                                    id='gender',
                                    style={'marginTop': '64 !important', 'marginBottom': '36 !important'}
                                ),
                                html.H6("Constant Feature:", style={'marginTop': 12}),
                                dcc.Dropdown(
                                    id='constant-feature',
                                    style={'marginTop': 14, 'marginBottom': 14},
                                    optionHeight=55,
                                ),
                                html.H6('Constant Value:'),
                                dcc.Dropdown(
                                    id='constant-value',
                                    style={'marginTop': 14, 'marginBottom': 14},
                                    optionHeight=55,
                                ),
                                html.H6('Variable 1:', id='var1-text'),
                                dcc.Dropdown(
                                    id='var1',
                                    style={'marginTop': 14, 'marginBottom': 14},
                                    options=var_feat_options,
                                    optionHeight=55,
                                ),
                                html.Br(id='v1-break'),
                                html.H6('Variable 2:', id='var2-text'),
                                dcc.Dropdown(
                                    id='var2',
                                    style={'marginTop': 14, 'marginBottom': 14},
                                    options=var_feat_options,
                                    optionHeight=55,
                                ),
                                html.Br(id='v2-break'),
                                html.H6('Select a Player:', id='player-tag'),
                                dcc.Dropdown(
                                    id='player',
                                    options=player_options,
                                    placeholder='Select...',
                                    optionHeight=55,
                                    style={'display': 'none'}                                 
                                )
                            ]
                        ),
                    ],
                    style={'backgroundColor': 'oldlace', 'color': 'black'},
                ),
                dbc.Col(
                    children=[
                        html.H6(children='', id='chart-description',
                        style= {'text-align':'left', 'padding-top':25, 'padding-left':15}),
                        dcc.Graph(id='fig1'),
                        html.P(children='', id="analysis"),
                    ],
                    width={"size":10},
                    style={'color': 'black'}
                )
            ]
        )
    ]
    )
])

# CALLBACKSSSS

# callback to generate constant value selection

@app.callback(
    Output('constant-value', 'options'),
    Input('constant-feature', 'value'),
    Input('radios', 'value')
)
def update_constant_value_dropdown(constant_feature, radios):
    if constant_feature is None:
        return []
    if radios == 1 or radios == 2:
        constant_values = df_pp[df_pp[constant_feature].notna()][constant_feature].unique()
        constant_value_options = [{'label': value, 'value': value} for value in constant_values]
        return constant_value_options
    elif radios == 3:
        constant_values = df_ohe_full[df_ohe_full[constant_feature].notna()][constant_feature].unique()
        constant_value_options = [{'label': value, 'value': value} for value in constant_values]
        return constant_value_options

# callback to generate constant feature selection

@app.callback(
    Output('constant-feature', 'options'),
    Input('radios', 'value')
)
def update_dropdown_options(radios):
    if radios == 1 or radios == 2:
        return const_feat_options
    elif radios == 3:
        return eo_const_feat_options
    else:
        return []

# callbacks to hide gender options, H6 and V1, V2

@app.callback(
    Output('player-tag', 'style'),
    [Input('radios', 'value')]
)
def update_h6_visibility(radios):
    if radios == 1:
        return {'display': 'none'}
    else:
        return {'display': 'block'}

@app.callback(
    Output('gender', 'style'),
    Input('radios', 'value')
)
def update_gender_visibility(radios):
    if radios == 1 or radios == 3:
        return {'display': 'block'}
    else:
        return {'display': 'none'}

'''
@app.callback(
    [Output('player-tag', 'style'),
     Output('gender', 'style')],
    [Input('radios', 'value')]
)
def update_visibility(radios):
    player_tag_style = {'display': 'none'}
    gender_style = {'display': 'block'}

    if radios == 1 or radios == 3:
        player_tag_style = {'display': 'none'}
        gender_style = {'display': 'block'}
    else:
        player_tag_style = {'display': 'block'}
        gender_style = {'display': 'none'}

    return player_tag_style, gender_style '''

@app.callback(
    [Output('var1-text', 'style'),
     Output('var2-text', 'style'),
     Output('var1', 'style'),
     Output('v1-break', 'style'),
     Output('v2-break', 'style'),
     Output('var2', 'style')],
    [Input('radios', 'value')]
)
def update_vis_var(radios):
    if radios == 3:
        return ({'display': 'none'},
                {'display': 'none'},
                {'display': 'none'},
                {'display': 'none'},
                {'display': 'none'},
                {'display': 'none'})
    else:
        return ({'display': 'block'},
                {'display': 'block'},
                {'display': 'block'},
                {'display': 'block'},
                {'display': 'block'},
                {'display': 'block'})


# callback for High Level Buttons

@app.callback(
    Output('high-level-output', 'children'),
    Input('radios', 'value')
)
def update_text(value):
    if value == 1:
        return 'International Trends'
    elif value == 2:
        return 'Player Analysis'
    elif value == 3:
        return 'Expected Outcomes'

# callback to generate header for Graphs

@app.callback(
    Output('chart-description', 'children'),
    Input('constant-feature', 'value'),
    Input('constant-value', 'value')
)
def generate_header(constant_feature, constant_value):
    if constant_feature and constant_value != '':
        return f"{constant_feature}: {constant_value}"
    else:
        return ""

# callback to show or hide player dcc

@app.callback(
    Output('player', 'style'),
    Input('radios', 'value')
)
def update_component_visibility(value):
    if value == 2 or value == 3:
        return {'display': 'block'}
    else:
        return {'display': 'none'}

# callback to populate fig1

@app.callback(
    Output('fig1', 'figure'),
    Input('gender', 'value'),
    Input('constant-feature', 'value'),
    Input('constant-value', 'value'),
    Input('var1', 'value'),
    Input('var2', 'value'),
    Input('radios', 'value'),
    Input('player', 'value')
)
def generate_graph(gender, constant_feature, constant_value, var1, var2, radios, player):
    if radios == 1:
        if None in [gender, constant_feature, constant_value, var1, var2]:
            return {}
        else:
            df = generate_df(df_pp, gender, constant_feature, constant_value, var1, var2)
            unique_var1 = df[var1].astype(str).unique()
            unique_var2 = df[var2].astype(str).unique()
            num_var1 = len(unique_var1)
            num_var2 = len(unique_var2)

            fig = make_subplots(rows=num_var1, cols=3, subplot_titles=[f"{v1}" for v1 in unique_var1])

            row_num = 1
            col_num = 1

            for i, v1 in enumerate(unique_var1):
                subset_df = df[df[var1].astype(str) == v1]
                hover_text = [f"Volume: {num_instances}" for num_instances in zip(subset_df['num_instances'])]
                fig.add_trace(
                    go.Bar(x=subset_df[var2].astype(str),
                    y=subset_df['AverageAttackOutcome'],
                    hovertext=hover_text,
                    name=f"{var1}: {v1}"),
                    row=row_num, col=col_num
                )
                col_num += 1
                if col_num > 3:
                    col_num = 1
                    row_num += 1

            fig.update_layout(height=400 * num_var1, width=800, title_text=f"{var2} by {var1}")
            fig.update_yaxes(range=[0.4, 0.8])
            fig.update_layout(showlegend=True, autosize=False)
            fig.update_xaxes(tickangle=45)

            return fig
    elif radios == 2:
        if None in [gender, constant_feature, constant_value, var1, var2, player]:
            return {}
        else:
            df = generate_df_pp(df_pp, gender, constant_feature, constant_value, var1, var2, player)
            unique_var1 = df[var1].astype(str).unique()
            unique_var2 = df[var2].astype(str).unique()
            num_var1 = len(unique_var1)
            num_var2 = len(unique_var2)

            fig = make_subplots(rows=num_var1, cols=3, subplot_titles=[f"{v1}" for v1 in unique_var1])

            row_num = 1
            col_num = 1

            for i, v1 in enumerate(unique_var1):
                subset_df = df[df[var1].astype(str) == v1]
                hover_text = [f"Volume: {num_instances}" for num_instances in zip(subset_df['num_instances'])]
                fig.add_trace(
                    go.Bar(x=subset_df[var2].astype(str),
                    y=subset_df['AverageAttackOutcome'],
                    hovertext=hover_text,
                    name=f"{var1}: {v1}"),
                    row=row_num, col=col_num
                )
                col_num += 1
                if col_num > 3:
                    col_num = 1
                    row_num += 1

            fig.update_layout(height=400 * num_var1, width=800, title_text=f"{var2} by {var1}")
            fig.update_yaxes(range=[0.4, 0.8])
            fig.update_layout(showlegend=True, autosize=False)
            fig.update_xaxes(tickangle=45)

            return fig

    elif radios == 3:
        if None in [gender, constant_feature, constant_value, player]:
            return {}
        else:
            df_eo = generate_df_eo(df_ohe_full, gender, constant_feature, constant_value, player)

            hover_text = [f"Volume: {num_instances}" for num_instances in df_eo['num_instances']]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Actual Average Outcome', 'Expected Average Outcome'],
                y=[df_eo['AverageAttackOutcome'].values[0], df_eo['ExpectedOutcome'].values[0]],
                name='Average Attack Outcome',
                hovertext=hover_text,
                #marker_color=['steelblue', 'darkorange']
            ))
            fig.update_layout(
                title=f"Expected vs Actual Average Attack Outcome<br>({constant_feature}: {constant_value})",
                xaxis_title="",
                yaxis_title="Average Attack Outcome",
                showlegend=True,
                width=800
            )
            fig.update_yaxes(range=[0.4, 0.8])
            fig.update_xaxes(tickangle=45)

            return fig
    else:
        return {}            

'''

@app.callback(
    Output('fig1', 'figure'),
    Input('gender', 'value'),
    Input('constant-feature', 'value'),
    Input('constant-value', 'value'),
    Input('var1', 'value'),
    Input('var2', 'value'),
    Input('radios', 'value')
)
def generate_graph(gender, constant_feature, constant_value, var1, var2, radios):
    fig = None
    if radios == 1:
        if None in [gender, constant_feature, constant_value, var1, var2]:
            return {}
        else:
            df = generate_df(df_pp, gender, constant_feature, constant_value, var1, var2)
            fig = make_subplots(rows=2, cols=3)

            fig.add_trace(
            go.Bar(x=df[var1], y=df['AverageAttackOutcome'], base=df[var2]),
            row=1, col=1
            )

            fig.update_layout(height=600, width=800, title_text="Subplots")
            fig.update_yaxes(range=[0.4, 0.75])
            return fig
    else:
        return {}
'''

'''
# callback to populate fig1

@app.callback(
    Output('fig1', 'figure'),
    Input('radios', 'value'),
    Input('low-level-dropdown', 'value'),
    Input('low-low-level-dropdown', 'value')
)
def generate_graph(radios, lowlevelbutton, lowlowlevelbutton):
    fig = None 
    if radios == 3 and lowlevelbutton == "Cross Validation":
        fig = px.bar(
            df_cv_results.melt(id_vars='Model', var_name='Metric', value_name='Score'),
            x='Model',
            y='Score',
            color='Model',
            facet_col='Metric',
            barmode='group'
        )
        fig.update_layout(
            xaxis_title='Model',
            yaxis_title='Score'
        )
        return fig
    elif radios == 3 and lowlevelbutton == 'Hyperparameter Tuning' and lowlowlevelbutton == 'Full Feature Set':
        fig = px.bar(
            df_tuned_results.melt(id_vars='Model', var_name='Metric', value_name='Score'),
            x='Model',
            y='Score',
            color='Model',
            facet_col='Metric',
            barmode='group'
        )
        fig.update_layout(
            xaxis_title='Model',
            yaxis_title='Score'
        )
        return fig
    elif radios == 3 and lowlevelbutton == 'Hyperparameter Tuning' and lowlowlevelbutton == 'Average Feature Set':
        fig = px.bar(
            df_fs_avg_results.melt(id_vars='Model', var_name='Metric', value_name='Score'),
            x='Model',
            y='Score',
            color='Model',
            facet_col='Metric',
            barmode='group'
        )
        fig.update_layout(
            xaxis_title='Model',
            yaxis_title='Score'
        )
        return fig
    elif radios == 3 and lowlevelbutton == 'Hyperparameter Tuning' and lowlowlevelbutton == 'RF Feature Set':
        fig = px.bar(
            df_fs_rfr_results.melt(id_vars='Model', var_name='Metric', value_name='Score'),
            x='Model',
            y='Score',
            color='Model',
            facet_col='Metric',
            barmode='group'
        )
        fig.update_layout(
            xaxis_title='Model',
            yaxis_title='Score'
        )
        return fig
    elif radios == 1 and lowlevelbutton == "Efficiency by Round":
        fig = make_subplots(rows=2, cols=3,)
        
        fig.add_trace(go.Bar(x=df_quali_eff['Attack Style'], y=df_quali_eff['Attack Outcome'], name='Qualifier'), row=1, col=1)
        fig.add_trace(go.Bar(x=df_maindraw_eff['Attack Style'], y=df_maindraw_eff['Attack Outcome'], name='Main Draw'), row=1, col=2)
        fig.add_trace(go.Bar(x=df_quarters_eff['Attack Style'], y=df_quarters_eff['Attack Outcome'], name='Quarters'), row=1, col=3)
        fig.add_trace(go.Bar(x=df_bronze_eff['Attack Style'], y=df_bronze_eff['Attack Outcome'], name='Bronze'), row=2, col=1)
        fig.add_trace(go.Bar(x=df_finals_eff['Attack Style'], y=df_finals_eff['Attack Outcome'], name='Finals'), row=2, col=2)

        fig.update_yaxes(range=[.3, .75])
        fig.update_layout(title_text="Attack Efficiency in each round of the tournament")

        return fig
    elif radios == 1 and lowlevelbutton == "Attack Style":
        fig = make_subplots(rows=2, cols=2,)
        
        fig.add_trace(go.Bar(x=df_attack_eff['Attack Style'], y=df_attack_eff['AttackOutcome'], name='General Efficiency'), row=1, col=1)
        fig.add_trace(go.Bar(x=df_hard_eff['Attack Block Type'], y=df_hard_eff['Hard Efficiency'], name='Hard'), row=1, col=2)
        fig.add_trace(go.Bar(x=df_shot_eff['Attack Block Type'], y=df_shot_eff['Shot Efficiency'], name='Shot'), row=2, col=1)
        fig.add_trace(go.Bar(x=df_pokey_eff['Attack Block Type'], y=df_pokey_eff['Pokey Efficiency'], name='Pokey'), row=2, col=2)

        fig.update_yaxes(range=[.4, .8])
        fig.update_layout(title_text="Attack Style Efficiency against different Defensive Structures")

        return fig
    elif radios == 1 and lowlevelbutton == "In System":
        fig = make_subplots(rows=2, cols=2,)
        
        fig.add_trace(go.Bar(x=df_is_attackstyle_eff['Attack Style'], y=df_is_attackstyle_eff['AttackOutcome'], name='Attack Style'), row=1, col=1)
        fig.add_trace(go.Bar(x=df_is_np_eff['Net Point'], y=df_is_np_eff['AttackOutcome'], name='Net Point'), row=1, col=2)
        fig.add_trace(go.Bar(x=df_is_abt_eff['Attack Block Type'], y=df_is_abt_eff['AttackOutcome'], name='Attack Block Type'), row=2, col=1)
        fig.add_trace(go.Bar(x=df_is_settype_eff['Set Type'], y=df_is_settype_eff['AttackOutcome'], name='Set Type'), row=2, col=2)

        fig.update_yaxes(range=[.4, .8])
        fig.update_layout(title_text="In System Attack Efficiency")

        return fig
    elif radios == 1 and lowlevelbutton == "In System - Quick":
        fig = make_subplots(rows=2, cols=2,)
        
        fig.add_trace(go.Bar(x=df_quick_attackstyle_eff['Attack Style'], y=df_quick_attackstyle_eff['AttackOutcome'], name='Attack Style'), row=1, col=1)
        fig.add_trace(go.Bar(x=df_quick_np_eff['Net Point'], y=df_quick_np_eff['AttackOutcome'], name='Net Point'), row=1, col=2)
        fig.add_trace(go.Bar(x=df_quick_abt_eff['Attack Block Type'], y=df_quick_abt_eff['AttackOutcome'], name='Attack Block Type'), row=2, col=1)
        fig.add_trace(go.Bar(x=df_quick_settype_eff['Set Type'], y=df_quick_settype_eff['AttackOutcome'], name='Set Type'), row=2, col=2)

        fig.update_yaxes(range=[.4, .8])
        fig.update_layout(title_text="In System - Quick Attack Efficiency")
        return fig
    elif radios == 1 and lowlevelbutton == "Out of System":
        fig = make_subplots(rows=2, cols=2,)
        
        fig.add_trace(go.Bar(x=df_os_attackstyle_eff['Attack Style'], y=df_os_attackstyle_eff['AttackOutcome'], name='Attack Style'), row=1, col=1)
        fig.add_trace(go.Bar(x=df_os_np_eff['Net Point'], y=df_os_np_eff['AttackOutcome'], name='Net Point'), row=1, col=2)
        fig.add_trace(go.Bar(x=df_os_abt_eff['Attack Block Type'], y=df_os_abt_eff['AttackOutcome'], name='Attack Block Type'), row=2, col=1)
        fig.add_trace(go.Bar(x=df_os_settype_eff['Set Type'], y=df_os_settype_eff['AttackOutcome'], name='Set Type'), row=2, col=2)

        fig.update_yaxes(range=[.4, .8])
        fig.update_layout(title_text="Out of System Attack Efficiency")
        return fig
    elif radios == 1 and lowlevelbutton == "Two Ball":
        fig = make_subplots(rows=2, cols=2,)
        
        fig.add_trace(go.Bar(x=df_two_attackstyle_eff['Attack Style'], y=df_two_attackstyle_eff['AttackOutcome'], name='Attack Style'), row=1, col=1)
        fig.add_trace(go.Bar(x=df_two_np_eff['Net Point'], y=df_two_np_eff['AttackOutcome'], name='Net Point'), row=1, col=2)
        fig.add_trace(go.Bar(x=df_two_abt_eff['Attack Block Type'], y=df_two_abt_eff['AttackOutcome'], name='Attack Block Type'), row=2, col=1)
        fig.add_trace(go.Bar(x=df_two_settype_eff['Set Type'], y=df_two_settype_eff['AttackOutcome'], name='Set Type'), row=2, col=2)

        fig.update_yaxes(range=[.3, .8])
        fig.update_layout(title_text="Two Ball Attack Efficiency")
        return fig
    elif radios == 1 and lowlevelbutton == "Disruptive Serving":
        fig = make_subplots(rows=2, cols=2,)
        
        fig.add_trace(go.Bar(x=df_jf_eff['Receive Side'], y=df_jf_eff['Attack Outcome'], name='Jump Float'), row=1, col=1)
        fig.add_trace(go.Bar(x=df_js_eff['Receive Side'], y=df_js_eff['Attack Outcome'], name='Jump Spin'), row=1, col=2)
        fig.add_trace(go.Bar(x=df_fn_eff['Receive Side'], y=df_fn_eff['Attack Outcome'], name='Float Near'), row=2, col=1)
        fig.add_trace(go.Bar(x=df_ff_eff['Receive Side'], y=df_ff_eff['Attack Outcome'], name='Float Far'), row=2, col=2)

        fig.update_yaxes(range=[.3, .8])
        fig.update_layout(title_text="Attack Efficiency based on Serve Type & Location")
        return fig
    elif radios == 2 and lowlevelbutton == "Passing" and lowlowlevelbutton == 'Player 2':
        fig = px.scatter(df_sp_receive_rand, x="From_Zone_X", y="From_Zone_Y", color='Receive')
        fig.update_layout(title_text='Player 2 Receive Chart')
        
        return fig
    elif radios == 1 and lowlevelbutton == "High Percentage Shots":
        fig = px.scatter(df_hps, x="To_Zone_X", y="To_Zone_Y", color='AttackOutcome')
        fig.update_layout(title_text='High Percentage Shots', autosize=False, width=1000, height=700,)

        return fig
    elif radios == 1 and lowlevelbutton == "Defensive Positioning" and lowlowlevelbutton == "Right":
        fig = px.scatter(df_np_right, x="To_Zone_X", y="To_Zone_Y", color='Zone_Y')
        fig.update_layout(title_text= 'Attacks from the Right Side', autosize=False, width=1000, height=700)
        
        return fig
    elif radios == 1 and lowlevelbutton == "Defensive Positioning" and lowlowlevelbutton == "Left":
        fig = px.scatter(df_np_left, x="To_Zone_X", y="To_Zone_Y", color='Zone_Y')
        fig.update_layout(title_text= 'Attacks from the Left Side', autosize=False, width=1000, height=700)
        
        return fig
    elif radios == 1 and lowlevelbutton == "Defensive Positioning" and lowlowlevelbutton == "Middle":
        fig = px.scatter(df_np_mid, x="To_Zone_X", y="To_Zone_Y", color='Zone_Y')
        fig.update_layout(title_text= 'Attacks from the Middle', autosize=False, width=1000, height=700)
        
        return fig
    elif radios == 2 and lowlevelbutton == "EO vs AO Per Round" and lowlowlevelbutton == "Player 1":
        fig = make_subplots(rows=2, cols=2,)
        
        fig.add_trace(go.Bar(x=df_mhp_maindraw['Metric'], y=df_mhp_maindraw['Attack Outcome'], name='Main Draw'), row=1, col=1)
        fig.add_trace(go.Bar(x=df_mhp_quarters['Metric'], y=df_mhp_quarters['Attack Outcome'], name='Quarters'), row=1, col=2)
        fig.add_trace(go.Bar(x=df_mhp_finals['Metric'], y=df_mhp_finals['Attack Outcome'], name='Finals'), row=2, col=1)

        fig.update_yaxes(range=[.3, .8])
        fig.update_layout(title_text="Player 1 Efficiency Per Round")
        return fig
    elif radios == 3 and lowlevelbutton == "Feature Selection" and lowlowlevelbutton == "Random Forest":
        table_trace = go.Table(header=dict(values=['features', 'importance']), cells=dict(values=[feat_imp_rfr.feature, feat_imp_rfr.importance]))
        bar_trace = go.Bar(x=feat_imp_rfr.feature, y=feat_imp_rfr.importance)
        fig = make_subplots(rows=2, cols=1, specs=[[{"type": "table"}], [{"type": "bar"}]])
        fig.add_trace(table_trace, row=1, col=1)
        fig.add_trace(bar_trace, row=2, col=1)

        fig.update_layout(title_text='Random Forest Feature Selection')

        return fig
    elif radios == 3 and lowlevelbutton == "Feature Selection" and lowlowlevelbutton == "Extra Trees":
        table_trace = go.Table(header=dict(values=['features', 'importance']), cells=dict(values=[feat_imp_etr.feature, feat_imp_etr.importance]))
        bar_trace = go.Bar(x=feat_imp_etr.feature, y=feat_imp_etr.importance)
        fig = make_subplots(rows=2, cols=1, specs=[[{"type": "table"}], [{"type": "bar"}]])
        fig.add_trace(table_trace, row=1, col=1)
        fig.add_trace(bar_trace, row=2, col=1)

        fig.update_layout(title_text='Extra Tree Feature Selection')

        return fig
    elif radios == 3 and lowlevelbutton == "Feature Selection" and lowlowlevelbutton == "Gradient Boost":
        table_trace = go.Table(header=dict(values=['features', 'importance']), cells=dict(values=[feat_imp_gbr.feature, feat_imp_gbr.importance]))
        bar_trace = go.Bar(x=feat_imp_gbr.feature, y=feat_imp_gbr.importance)
        fig = make_subplots(rows=2, cols=1, specs=[[{"type": "table"}], [{"type": "bar"}]])
        fig.add_trace(table_trace, row=1, col=1)
        fig.add_trace(bar_trace, row=2, col=1)

        fig.update_layout(title_text='Gradient Boost Feature Selection')

        return fig
    elif radios == 3 and lowlevelbutton == "Feature Selection" and lowlowlevelbutton == "Average":
        table_trace = go.Table(header=dict(values=['features', 'importance']), cells=dict(values=[feat_imp_avg.feature, feat_imp_avg.avg_importance]))
        bar_trace = go.Bar(x=feat_imp_avg.feature, y=feat_imp_avg.avg_importance)
        fig = make_subplots(rows=2, cols=1, specs=[[{"type": "table"}], [{"type": "bar"}]])
        fig.add_trace(table_trace, row=1, col=1)
        fig.add_trace(bar_trace, row=2, col=1)

        fig.update_layout(title_text='Average Feature Selection')

        return fig
    else:
        return {}

       '''

# Run Server

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)