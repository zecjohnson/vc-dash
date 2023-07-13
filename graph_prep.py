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

'''
# Remove New_Player from df

df_ohe = df_ohe.drop('New_Player', axis=1)

# Remove Attack Location from df

df_ohe = df_ohe.drop('To_Zone_X', axis=1)
df_ohe = df_ohe.drop('To_Zone_Y', axis=1)

# Seperate df into mens and womens

men_attacks = df_ohe[df_ohe['Gender'] == 'Men']
women_attacks = df_ohe[df_ohe['Gender'] == 'Women']
men_attacks = men_attacks.drop('Gender', axis=1)
women_attacks = women_attacks.drop('Gender', axis=1)

# remove all missing values from womens attacks

women_attacks = women_attacks.dropna()

# create instance of Random Forest Regressor 

model = RandomForestRegressor()

# split data into training and testing for mens

X = men_attacks.drop('AttackOutcome', axis=1)
y = men_attacks['AttackOutcome']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# save mens model to a joblib file

model.fit(X_train, y_train)
joblib.dump(model, 'mens_chal_e16.joblib')



# split data into training and testing for womens

X = women_attacks.drop('AttackOutcome', axis=1)
y = women_attacks['AttackOutcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# save womens model to a joblib file



model.fit(X_train, y_train)
joblib.dump(model, 'womens_chal_e16.joblib')



# loading mens and womens model

mens_model = joblib.load('mens_chal_e16.joblib') 
womens_model = joblib.load('womens_chal_e16.joblib') '''

"""

param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(5, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20)
}

rfr = RandomForestRegressor()

cv = KFold(n_splits=5, shuffle=True, random_state=42)
rand_search = RandomizedSearchCV(
    estimator=rfr,
    param_distributions=param_dist,
    cv=cv,
    scoring='neg_mean_absolute_error',
    n_iter=20,
    random_state=42
)

rand_search.fit(Xc_train, yc_train)
rf_best_params = rand_search.best_params_
rfr_best = RandomForestRegressor(**rf_best_params)
rfr_best.fit(Xc_train, yc_train)
test_score = rfr_best.score(Xc_test, yc_test)


rf_best_params = {'max_depth': 15,
 'min_samples_leaf': 9,
 'min_samples_split': 16,
 'n_estimators': 64} 

rfr_best_params = RandomForestRegressor(**rf_best_params)
rfr_best_params.fit(Xc_train, yc_train) 
best_param_train_score = rfr_best_params.score(Xc_train, yc_train)
baseline_train_score = model.score(Xc_train, yc_train)
best_param_test_score = rfr_best_params.score(Xc_test, yc_test)
baseline_test_score = model.score(Xc_test, yc_test)

# testing performance of baseline against hypterparameter tuned model

Xc = df_ohe.drop('AttackOutcome', axis=1)
yc = df_ohe['AttackOutcome']

df_tuned_results = pd.DataFrame(columns=['Model', 'MAE_train', 'MAE_test', 'R2_train', 'R2_test'])

models_tuned = {}

models_tuned['RF_Baseline'] = RandomForestRegressor()
models_tuned['RF_Tuned'] = RandomForestRegressor(**rf_best_params)

for model_name in models_tuned:
    model = models_tuned[model_name]
    results = cross_validate(model, Xc, yc, cv=5, scoring=['neg_mean_absolute_error', 'r2'], return_train_score=True, error_score='raise')

    model_results = {
        'Model': model_name,
        'MAE_train': -results['train_neg_mean_absolute_error'].mean(),
        'MAE_test': -results['test_neg_mean_absolute_error'].mean(),
        'R2_train': results['train_r2'].mean(),
        'R2_test': results['test_r2'].mean()
    }

    df_tuned_results = df_tuned_results.append(model_results, ignore_index=True)

# hyperparameter testing for reduced feature sets

df_fs_avg = df_pp.copy()
df_fs_avg = df_fs_avg[['ScorePlusMinus', 'ScoreDifference', 'OpponentSetWon', 'ReceiveSide', 'AttackBlockType', 'ServeType', 'TournamentPhase', 'Set', 'SetType', 'TeamSetWon', 'SetStatus', 'ContactNumber', 'Wind', 'Light', 'Zone_Y', 'AttackLocation', 'AttackOutcome']]
df_fs_rfr = df_pp.copy()
df_fs_rfr = df_fs_rfr[['OpponentSetWon', 'Light', 'ContactNumber', 'Set', 'AttackLocation', 'Weather', 'From_Zone_X', 'OpponentScore', 'RedZone', 'Phase', 'TeamScore', 'Wind', 'GameState', 'From_Zone_Y', 'AttackOutcome']]

# creating dummy variables for all categorical variables except AttackOutcome

cols_to_encode = [col for col in df_fs_avg.columns if col != "AttackOutcome"]
df_fs_avg_encoded = pd.get_dummies(df_fs_avg[cols_to_encode])
df_fs_avg = pd.concat([df_fs_avg_encoded, df_fs_avg["AttackOutcome"]], axis=1)

cols_to_encode = [col for col in df_fs_rfr.columns if col != "AttackOutcome"]
df_fs_rfr_encoded = pd.get_dummies(df_fs_rfr[cols_to_encode])
df_fs_rfr = pd.concat([df_fs_rfr_encoded, df_fs_rfr["AttackOutcome"]], axis=1)


# split data into training and testing set at 80/20 ratio

Xc_avg = df_fs_avg.drop('AttackOutcome', axis=1)
yc_avg = df_fs_avg['AttackOutcome']

Xc_avg_train, Xc_avg_test, yc_avg_train, yc_avg_test = train_test_split(Xc_avg, yc_avg, test_size=0.2, random_state=42)

Xc_rfr = df_fs_rfr.drop('AttackOutcome', axis=1)
yc_rfr = df_fs_rfr['AttackOutcome']

Xc_rfr_train, Xc_rfr_test, yc_rfr_train, yc_rfr_test = train_test_split(Xc_rfr, yc_rfr, test_size=0.2, random_state=42)

df_fs_avg_results = pd.DataFrame(columns=['Model', 'MAE_train', 'MAE_test', 'R2_train', 'R2_test'])
df_fs_rfr_results = pd.DataFrame(columns=['Model', 'MAE_train', 'MAE_test', 'R2_train', 'R2_test'])

for model_name in models_tuned:
    model = models_tuned[model_name]
    results = cross_validate(model, Xc_avg, yc_avg, cv=5, scoring=['neg_mean_absolute_error', 'r2'], return_train_score=True, error_score='raise')

    model_results = {
        'Model': model_name,
        'MAE_train': -results['train_neg_mean_absolute_error'].mean(),
        'MAE_test': -results['test_neg_mean_absolute_error'].mean(),
        'R2_train': results['train_r2'].mean(),
        'R2_test': results['test_r2'].mean()
    }

    df_fs_avg_results = df_fs_avg_results.append(model_results, ignore_index=True)

for model_name in models_tuned:
    model = models_tuned[model_name]
    results = cross_validate(model, Xc_rfr, yc_rfr, cv=5, scoring=['neg_mean_absolute_error', 'r2'], return_train_score=True, error_score='raise')

    model_results = {
        'Model': model_name,
        'MAE_train': -results['train_neg_mean_absolute_error'].mean(),
        'MAE_test': -results['test_neg_mean_absolute_error'].mean(),
        'R2_train': results['train_r2'].mean(),
        'R2_test': results['test_r2'].mean()
    }

    df_fs_rfr_results = df_fs_rfr_results.append(model_results, ignore_index=True)

# establishing instance of RF Baseline Model

model = joblib.load('RF.joblib')

# feature selection

df_fs = df_pp.copy()

cols_to_encode = [col for col in df_fs.columns if col != "AttackOutcome"]
df_fs_encoded = pd.get_dummies(df_fs[cols_to_encode])
df_fs = pd.concat([df_fs_encoded, df_fs["AttackOutcome"]], axis=1)
encoded = list(df_fs.columns)

Xc = df_fs_encoded
yc = df_fs['AttackOutcome']

Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=42)

etr = ExtraTreesRegressor()
rfr = RandomForestRegressor()
gbr = GradientBoostingRegressor()
etr.fit(Xc_train, yc_train)
rfr.fit(Xc_train, yc_train)
gbr.fit(Xc_train, yc_train)

feat_imp_etr = pd.DataFrame(zip(cols_to_encode, etr.feature_importances_ * 100), columns=['feature', 'importance'])
feat_imp_etr = feat_imp_etr.sort_values(by='importance', ascending=False)
feat_imp_rfr = pd.DataFrame(zip(cols_to_encode, rfr.feature_importances_ * 100), columns=['feature', 'importance'])
feat_imp_rfr = feat_imp_rfr.sort_values(by='importance', ascending=False)
feat_imp_gbr = pd.DataFrame(zip(cols_to_encode, gbr.feature_importances_ * 100), columns=['feature', 'importance'])
feat_imp_gbr = feat_imp_gbr.sort_values(by='importance', ascending=False)

df_merged = feat_imp_etr.merge(feat_imp_rfr, on='feature').merge(feat_imp_gbr, on='feature')
df_merged['avg_importance'] = (df_merged['importance_x'] + df_merged['importance_y'] + df_merged['importance']) / 3
feat_imp_avg = df_merged[['feature', 'avg_importance']]
feat_imp_avg = feat_imp_avg.sort_values(by='avg_importance', ascending=False)
feat_imp_avg """

# prep for EO for MHP

''' df_mhp = df_attack.copy()

df_mhp = df_mhp[['Weather','TournamentPhase', 'SetType', 'SetStatus', 'Set', 'ServeType', 
               'ScorePlusMinus', 'ScoreDifference',
       'ReceiveSide', 'AttackBlockType', 'OpponentSetWon', 'TeamSetWon', 'Zone_X', 'AttackType', 'Wind', 
               'GameState', 'AttackLocation', 'From_Zone_X', 'Phase',
       'TeamScore', 'Zone_Y',
       'OpponentScore', 'Light', 'AttackStyle',
       'From_Zone_Y', 'ContactNumber', "Receive", "RedZone", "New_Player", 'AttackOutcome']]

df_mhp = df_mhp[df_mhp['New_Player'] == 'Melissa Humana-Paredes']
df_mhp = df_mhp.drop('New_Player', axis=1)

new_row = pd.DataFrame({
    
    'Weather': ['Clear'],
    'TournamentPhase': ['Final 3th Place'],
    'SetType': ['Hand'],
    'SetStatus': ['Tied 1-1'],
    'Set': [3],
    'ServeType': ['Float Far'],
    'ScorePlusMinus': [2],
    'ScoreDifference': [-1],
    'ReceiveSide': ['Middle'],
    'AttackBlockType': ['Peel'],
    'OpponentSetWon': [1],
    'TeamSetWon': [1],
    'Zone_X': [7],
    'AttackType': ['Quick'],
    'Wind': ['Strong Wind'],
    'GameState': ['Normal'],
    'AttackLocation': ['Net Point 2'],
    'From_Zone_X': [3],
    'Phase': ['Transition'],
    'TeamScore': [18],
    'Zone_Y': [1],
    'OpponentScore': [21],
    'Light': ['Artificial'],
    'AttackStyle': ['Pokey'],
    'From_Zone_Y': [4],
    'ContactNumber': [8],
    'Receive': ['Poor'],
    'RedZone': [0],
    'AttackOutcome': ['0.25']})

new_row2 = pd.DataFrame({
    
    'Weather': ['Clear'],
    'TournamentPhase': ['Qualification'],
    'SetType': ['Hand'],
    'SetStatus': ['Tied 1-1'],
    'Set': [3],
    'ServeType': ['Float Far'],
    'ScorePlusMinus': [2],
    'ScoreDifference': [-1],
    'ReceiveSide': ['Middle'],
    'AttackBlockType': ['Peel'],
    'OpponentSetWon': [1],
    'TeamSetWon': [1],
    'Zone_X': [7],
    'AttackType': ['Quick'],
    'Wind': ['Strong Wind'],
    'GameState': ['Normal'],
    'AttackLocation': ['Net Point 2'],
    'From_Zone_X': [3],
    'Phase': ['Transition'],
    'TeamScore': [18],
    'Zone_Y': [1],
    'OpponentScore': [21],
    'Light': ['Artificial'],
    'AttackStyle': ['Pokey'],
    'From_Zone_Y': [4],
    'ContactNumber': [8],
    'Receive': ['Poor'],
    'RedZone': [0],
    'AttackOutcome': ['0.25']})

df_mhp = df_mhp.append(new_row, ignore_index=True)
df_mhp = df_mhp.append(new_row2, ignore_index=True) 

cols_to_encode = [col for col in df_mhp.columns if col != "AttackOutcome"]
df_mhp_encoded = pd.get_dummies(df_mhp[cols_to_encode])
df_mhp = pd.concat([df_mhp_encoded, df_mhp["AttackOutcome"]], axis=1)
encoded = list(df_mhp.columns)

# Efficiency by Tournament Phase Graph Prep

df_quali = df_attack[df_attack['TournamentPhase'] == 'Qualification']
df_maindraw = df_attack[df_attack['TournamentPhase'] == 'Main Draw']
df_quarters = df_attack[df_attack['TournamentPhase'] == 'Quarterfinal']
df_finals = df_attack[df_attack['TournamentPhase'] == 'Final 1st Place']
df_bronze = df_attack[df_attack['TournamentPhase'] == 'Final 3th Place']
quali_eff = df_quali['AttackOutcome'].mean()
maindraw_eff = df_maindraw['AttackOutcome'].mean()
quarters_eff = df_quarters['AttackOutcome'].mean()
finals_eff = df_finals['AttackOutcome'].mean()
bronze_eff = df_bronze['AttackOutcome'].mean()

df_quali_hard = df_quali[df_quali['AttackStyle'] == 'Hard']
df_quali_shot = df_quali[df_quali['AttackStyle'] == 'Shot']
df_quali_pokey = df_quali[df_quali['AttackStyle'] == 'Pokey']
quali_eff_hard = df_quali_hard['AttackOutcome'].mean()
quali_eff_shot = df_quali_shot['AttackOutcome'].mean()
quali_eff_pokey = df_quali_pokey['AttackOutcome'].mean()

df_maindraw_hard = df_maindraw[df_maindraw['AttackStyle'] == 'Hard']
df_maindraw_shot = df_maindraw[df_maindraw['AttackStyle'] == 'Shot']
df_maindraw_pokey = df_maindraw[df_maindraw['AttackStyle'] == 'Pokey']
maindraw_eff_hard = df_maindraw_hard['AttackOutcome'].mean()
maindraw_eff_shot = df_maindraw_shot['AttackOutcome'].mean()
maindraw_eff_pokey = df_maindraw_pokey['AttackOutcome'].mean()

df_quarters_hard = df_quarters[df_quarters['AttackStyle'] == 'Hard']
df_quarters_shot = df_quarters[df_quarters['AttackStyle'] == 'Shot']
df_quarters_pokey = df_quarters[df_quarters['AttackStyle'] == 'Pokey']
quarters_eff_hard = df_quarters_hard['AttackOutcome'].mean()
quarters_eff_shot = df_quarters_shot['AttackOutcome'].mean()
quarters_eff_pokey = df_quarters_pokey['AttackOutcome'].mean()

df_bronze_hard = df_bronze[df_bronze['AttackStyle'] == 'Hard']
df_bronze_shot = df_bronze[df_bronze['AttackStyle'] == 'Shot']
df_bronze_pokey = df_bronze[df_bronze['AttackStyle'] == 'Pokey']
bronze_eff_hard = df_bronze_hard['AttackOutcome'].mean()
bronze_eff_shot = df_bronze_shot['AttackOutcome'].mean()
bronze_eff_pokey = df_bronze_pokey['AttackOutcome'].mean()

df_finals_hard = df_finals[df_finals['AttackStyle'] == 'Hard']
df_finals_shot = df_finals[df_finals['AttackStyle'] == 'Shot']
df_finals_pokey = df_finals[df_finals['AttackStyle'] == 'Pokey']
finals_eff_hard = df_finals_hard['AttackOutcome'].mean()
finals_eff_shot = df_finals_shot['AttackOutcome'].mean()
finals_eff_pokey = df_finals_pokey['AttackOutcome'].mean()

df_quali_eff = pd.DataFrame({
    'Attack Style': ['All', 'Hard', 'Shot', 'Pokey'],
    'Attack Outcome': [quali_eff, quali_eff_hard, quali_eff_shot, quali_eff_pokey]
})

df_maindraw_eff = pd.DataFrame({
    'Attack Style': ['All', 'Hard', 'Shot', 'Pokey'],
    'Attack Outcome': [maindraw_eff, maindraw_eff_hard, maindraw_eff_shot, maindraw_eff_pokey]
})

df_quarters_eff = pd.DataFrame({
    'Attack Style': ['All', 'Hard', 'Shot', 'Pokey'],
    'Attack Outcome': [quarters_eff, quarters_eff_hard, quarters_eff_shot, quarters_eff_pokey]
})

df_bronze_eff = pd.DataFrame({
    'Attack Style': ['All', 'Hard', 'Shot', 'Pokey'],
    'Attack Outcome': [bronze_eff, bronze_eff_hard, bronze_eff_shot, bronze_eff_pokey]
})

df_finals_eff = pd.DataFrame({
    'Attack Style': ['All', 'Hard', 'Shot', 'Pokey'],
    'Attack Outcome': [finals_eff, finals_eff_hard, finals_eff_shot, finals_eff_pokey]
})

# AttackStyle Graph Prep

df_hard = df_attack[df_attack['AttackStyle'] == 'Hard']
df_shot = df_attack[df_attack['AttackStyle'] == 'Shot']
df_pokey = df_attack[df_attack['AttackStyle'] == 'Pokey']
hard_eff = df_hard['AttackOutcome'].mean()
shot_eff = df_shot['AttackOutcome'].mean()
pokey_eff = df_pokey['AttackOutcome'].mean()
df_attack_eff = pd.DataFrame({
    'Attack Style': ['Hard', 'Shot', 'Pokey'],
    'AttackOutcome': [hard_eff, shot_eff, pokey_eff]
})

df_hard_angle = df_hard[df_hard['AttackBlockType'] == 'Angle']
hard_angle_eff = df_hard_angle['AttackOutcome'].mean()
df_hard_line = df_hard[df_hard['AttackBlockType'] == 'Line']
hard_line_eff = df_hard_line['AttackOutcome'].mean()
df_hard_peel = df_hard[df_hard['AttackBlockType'] == 'Peel']
hard_peel_eff = df_hard_peel['AttackOutcome'].mean()
df_hard_eff = pd.DataFrame({
    'Attack Block Type': ['Angle', 'Line', 'Peel'],
    'Hard Efficiency': [hard_angle_eff, hard_line_eff, hard_peel_eff]
})

df_pokey_angle = df_pokey[df_pokey['AttackBlockType'] == 'Angle']
pokey_angle_eff = df_pokey_angle['AttackOutcome'].mean()
df_pokey_line = df_pokey[df_pokey['AttackBlockType'] == 'Line']
pokey_line_eff = df_pokey_line['AttackOutcome'].mean()
df_pokey_peel = df_pokey[df_pokey['AttackBlockType'] == 'Peel']
pokey_peel_eff = df_pokey_peel['AttackOutcome'].mean()
df_pokey_eff = pd.DataFrame({
    'Attack Block Type': ['Angle', 'Line', 'Peel'],
    'Pokey Efficiency': [pokey_angle_eff, pokey_line_eff, pokey_peel_eff]
})

df_shot_angle = df_shot[df_shot['AttackBlockType'] == 'Angle']
shot_angle_eff = df_shot_angle['AttackOutcome'].mean()
df_shot_line = df_shot[df_shot['AttackBlockType'] == 'Line']
shot_line_eff = df_shot_line['AttackOutcome'].mean()
df_shot_peel = df_shot[df_shot['AttackBlockType'] == 'Peel']
shot_peel_eff = df_shot_peel['AttackOutcome'].mean()
df_shot_eff = pd.DataFrame({
    'Attack Block Type': ['Angle', 'Line', 'Peel'],
    'Shot Efficiency': [shot_angle_eff, shot_line_eff, shot_peel_eff]
})

# In System Regular Tempo & In System Quick Graph Prep

df_is = df_attack[df_attack['AttackType'] == 'In System']
df_is_hard = df_is[df_is['AttackStyle'] == 'Hard']
df_is_shot = df_is[df_is['AttackStyle'] == 'Shot']
df_is_pokey = df_is[df_is['AttackStyle'] == 'Pokey']
df_quick = df_attack[df_attack['AttackType'] == 'Quick']
df_quick_hard = df_quick[df_quick['AttackStyle'] == 'Hard']
df_quick_shot = df_quick[df_quick['AttackStyle'] == 'Shot']
df_quick_pokey = df_quick[df_quick['AttackStyle'] == 'Pokey']
is_hard_eff = df_is_hard['AttackOutcome'].mean()
is_shot_eff = df_is_shot['AttackOutcome'].mean()
is_pokey_eff = df_is_pokey['AttackOutcome'].mean()
quick_hard_eff = df_quick_hard['AttackOutcome'].mean()
quick_shot_eff = df_quick_shot['AttackOutcome'].mean()
quick_pokey_eff = df_quick_pokey['AttackOutcome'].mean()
df_is_attackstyle_eff = pd.DataFrame({
    'Attack Style': ['Hard', 'Shot', 'Pokey'],
    'AttackOutcome': [is_hard_eff, is_shot_eff, is_pokey_eff]
})
df_quick_attackstyle_eff = pd.DataFrame({
    'Attack Style': ['Hard', 'Shot', 'Pokey'],
    'AttackOutcome': [quick_hard_eff, quick_shot_eff, quick_pokey_eff]
})

df_is_np1 = df_is[df_is['AttackLocation'] == 'Net Point 1']
df_is_np2 = df_is[df_is['AttackLocation'] == 'Net Point 2']
df_is_np3 = df_is[df_is['AttackLocation'] == 'Net Point 3']
df_is_np4 = df_is[df_is['AttackLocation'] == 'Net Point 4']
df_is_np5 = df_is[df_is['AttackLocation'] == 'Net Point 5']
df_quick_np1 = df_quick[df_quick['AttackLocation'] == 'Net Point 1']
df_quick_np2 = df_quick[df_quick['AttackLocation'] == 'Net Point 2']
df_quick_np3 = df_quick[df_quick['AttackLocation'] == 'Net Point 3']
df_quick_np4 = df_quick[df_quick['AttackLocation'] == 'Net Point 4']
df_quick_np5 = df_quick[df_quick['AttackLocation'] == 'Net Point 5']
is_np1_eff = df_is_np1['AttackOutcome'].mean()
is_np2_eff = df_is_np2['AttackOutcome'].mean()
is_np3_eff = df_is_np3['AttackOutcome'].mean()
is_np4_eff = df_is_np4['AttackOutcome'].mean()
is_np5_eff = df_is_np5['AttackOutcome'].mean()
quick_np1_eff = df_quick_np1['AttackOutcome'].mean()
quick_np2_eff = df_quick_np2['AttackOutcome'].mean()
quick_np3_eff = df_quick_np3['AttackOutcome'].mean()
quick_np4_eff = df_quick_np4['AttackOutcome'].mean()
quick_np5_eff = df_quick_np5['AttackOutcome'].mean()
df_is_np_eff = pd.DataFrame({
    'Net Point': ['1', '2', '3', '4', '5'],
    'AttackOutcome': [is_np1_eff, is_np2_eff, is_np3_eff, is_np4_eff, is_np5_eff,]
})
df_quick_np_eff = pd.DataFrame({
    'Net Point': ['1', '2', '3', '4', '5'],
    'AttackOutcome': [quick_np1_eff, quick_np2_eff, quick_np3_eff, quick_np4_eff, quick_np5_eff,]
})

df_is_angle = df_is[df_is['AttackBlockType'] == 'Angle']
df_is_line = df_is[df_is['AttackBlockType'] == 'Line']
df_is_peel = df_is[df_is['AttackBlockType'] == 'Peel']
df_quick_angle = df_quick[df_quick['AttackBlockType'] == 'Angle']
df_quick_line = df_quick[df_quick['AttackBlockType'] == 'Line']
df_quick_peel = df_quick[df_quick['AttackBlockType'] == 'Peel']
is_angle_eff = df_is_angle['AttackOutcome'].mean()
is_line_eff = df_is_line['AttackOutcome'].mean()
is_peel_eff = df_is_peel['AttackOutcome'].mean()
quick_angle_eff = df_quick_angle['AttackOutcome'].mean()
quick_line_eff = df_quick_line['AttackOutcome'].mean()
quick_peel_eff = df_quick_peel['AttackOutcome'].mean()
df_is_abt_eff = pd.DataFrame({
    'Attack Block Type': ['Angle', 'Line', 'Peel',],
    'AttackOutcome': [is_angle_eff, is_line_eff, is_peel_eff]
})
df_quick_abt_eff = pd.DataFrame({
    'Attack Block Type': ['Angle', 'Line', 'Peel',],
    'AttackOutcome': [quick_angle_eff, quick_line_eff, quick_peel_eff]
})

df_is_hand = df_is[df_is['SetType'] == 'Hand']
df_is_bump = df_is[df_is['SetType'] == 'Bump']
df_quick_hand = df_quick[df_quick['SetType'] == 'Hand']
df_quick_bump = df_quick[df_quick['SetType'] == 'Bump']
is_hand_eff = df_is_hand['AttackOutcome'].mean()
is_bump_eff = df_is_bump['AttackOutcome'].mean()
quick_hand_eff = df_quick_hand['AttackOutcome'].mean()
quick_bump_eff = df_quick_bump['AttackOutcome'].mean()
df_is_settype_eff = pd.DataFrame({
    'Set Type': ['Hand', 'Bump'],
    'AttackOutcome': [is_hand_eff, is_bump_eff]
})
df_quick_settype_eff = pd.DataFrame({
    'Set Type': ['Hand', 'Bump'],
    'AttackOutcome': [quick_hand_eff, quick_bump_eff]
})

# Out of System & Two Ball Graph Prep

df_os = df_attack[df_attack['AttackType'] == 'Out of System']
df_os_hard = df_os[df_os['AttackStyle'] == 'Hard']
df_os_shot = df_os[df_os['AttackStyle'] == 'Shot']
df_os_pokey = df_os[df_os['AttackStyle'] == 'Pokey']
df_two = df_attack[df_attack['AttackType'] == 'Second Touch']
df_two_hard = df_two[df_two['AttackStyle'] == 'Hard']
df_two_shot = df_two[df_two['AttackStyle'] == 'Shot']
df_two_pokey = df_two[df_two['AttackStyle'] == 'Pokey']
os_hard_eff = df_os_hard['AttackOutcome'].mean()
os_shot_eff = df_os_shot['AttackOutcome'].mean()
os_pokey_eff = df_os_pokey['AttackOutcome'].mean()
two_hard_eff = df_two_hard['AttackOutcome'].mean()
two_shot_eff = df_two_shot['AttackOutcome'].mean()
two_pokey_eff = df_two_pokey['AttackOutcome'].mean()
df_os_attackstyle_eff = pd.DataFrame({
    'Attack Style': ['Hard', 'Shot', 'Pokey'],
    'AttackOutcome': [os_hard_eff, os_shot_eff, os_pokey_eff]
})
df_two_attackstyle_eff = pd.DataFrame({
    'Attack Style': ['Hard', 'Shot', 'Pokey'],
    'AttackOutcome': [two_hard_eff, two_shot_eff, two_pokey_eff]
})

df_os_np1 = df_os[df_os['AttackLocation'] == 'Net Point 1']
df_os_np2 = df_os[df_os['AttackLocation'] == 'Net Point 2']
df_os_np3 = df_os[df_os['AttackLocation'] == 'Net Point 3']
df_os_np4 = df_os[df_os['AttackLocation'] == 'Net Point 4']
df_os_np5 = df_os[df_os['AttackLocation'] == 'Net Point 5']
df_two_np1 = df_two[df_two['AttackLocation'] == 'Net Point 1']
df_two_np2 = df_two[df_two['AttackLocation'] == 'Net Point 2']
df_two_np3 = df_two[df_two['AttackLocation'] == 'Net Point 3']
df_two_np4 = df_two[df_two['AttackLocation'] == 'Net Point 4']
df_two_np5 = df_two[df_two['AttackLocation'] == 'Net Point 5']
os_np1_eff = df_os_np1['AttackOutcome'].mean()
os_np2_eff = df_os_np2['AttackOutcome'].mean()
os_np3_eff = df_os_np3['AttackOutcome'].mean()
os_np4_eff = df_os_np4['AttackOutcome'].mean()
os_np5_eff = df_os_np5['AttackOutcome'].mean()
two_np1_eff = df_two_np1['AttackOutcome'].mean()
two_np2_eff = df_two_np2['AttackOutcome'].mean()
two_np3_eff = df_two_np3['AttackOutcome'].mean()
two_np4_eff = df_two_np4['AttackOutcome'].mean()
two_np5_eff = df_two_np5['AttackOutcome'].mean()
df_os_np_eff = pd.DataFrame({
    'Net Point': ['1', '2', '3', '4', '5'],
    'AttackOutcome': [os_np1_eff, os_np2_eff, os_np3_eff, os_np4_eff, os_np5_eff,]
})
df_two_np_eff = pd.DataFrame({
    'Net Point': ['1', '2', '3', '4', '5'],
    'AttackOutcome': [two_np1_eff, two_np2_eff, two_np3_eff, two_np4_eff, two_np5_eff,]
})

df_os_angle = df_os[df_os['AttackBlockType'] == 'Angle']
df_os_line = df_os[df_os['AttackBlockType'] == 'Line']
df_os_peel = df_os[df_os['AttackBlockType'] == 'Peel']
df_two_angle = df_two[df_two['AttackBlockType'] == 'Angle']
df_two_line = df_two[df_two['AttackBlockType'] == 'Line']
df_two_peel = df_two[df_two['AttackBlockType'] == 'Peel']
os_angle_eff = df_os_angle['AttackOutcome'].mean()
os_line_eff = df_os_line['AttackOutcome'].mean()
os_peel_eff = df_os_peel['AttackOutcome'].mean()
two_angle_eff = df_two_angle['AttackOutcome'].mean()
two_line_eff = df_two_line['AttackOutcome'].mean()
two_peel_eff = df_two_peel['AttackOutcome'].mean()
df_os_abt_eff = pd.DataFrame({
    'Attack Block Type': ['Angle', 'Line', 'Peel',],
    'AttackOutcome': [os_angle_eff, os_line_eff, os_peel_eff]
})
df_two_abt_eff = pd.DataFrame({
    'Attack Block Type': ['Angle', 'Line', 'Peel',],
    'AttackOutcome': [two_angle_eff, two_line_eff, two_peel_eff]
})

df_os_hand = df_os[df_os['SetType'] == 'Hand']
df_os_bump = df_os[df_os['SetType'] == 'Bump']
df_two_hand = df_two[df_two['SetType'] == 'Hand']
df_two_bump = df_two[df_two['SetType'] == 'Bump']
os_hand_eff = df_os_hand['AttackOutcome'].mean()
os_bump_eff = df_os_bump['AttackOutcome'].mean()
two_hand_eff = df_two_hand['AttackOutcome'].mean()
two_bump_eff = df_two_bump['AttackOutcome'].mean()
df_os_settype_eff = pd.DataFrame({
    'Set Type': ['Hand', 'Bump'],
    'AttackOutcome': [os_hand_eff, os_bump_eff]
})
df_two_settype_eff = pd.DataFrame({
    'Set Type': ['Hand', 'Bump'],
    'AttackOutcome': [two_hand_eff, two_bump_eff]
})

# MHP Passing Chart Prep

df_mhp_receive_rand = df_attack[df_attack['New_Player'] == 'Melissa Humana-Paredes']
df_mhp_receive_rand = pd.DataFrame(df_mhp_receive_rand[['From_Zone_X', 'From_Zone_Y', 'Receive']])
df_mhp_receive_rand[['From_Zone_X', 'From_Zone_Y']] = df_mhp_receive_rand[['From_Zone_X', 'From_Zone_Y']].apply(lambda x: x + np.random.uniform(-0.5, 0.5, len(x)))

df_sp_receive_rand = df_attack[df_attack['New_Player'] == 'Sarah Pavan']
df_sp_receive_rand = pd.DataFrame(df_sp_receive_rand[['From_Zone_X', 'From_Zone_Y', 'Receive']])
df_sp_receive_rand[['From_Zone_X', 'From_Zone_Y']] = df_sp_receive_rand[['From_Zone_X', 'From_Zone_Y']].apply(lambda x: x + np.random.uniform(-0.5, 0.5, len(x)))

# High Percentage Shots Chart Prep

df_hps = pd.DataFrame(df_attack[['To_Zone_X', 'To_Zone_Y', 'AttackOutcome']])
df_hps[['To_Zone_X', 'To_Zone_Y']] = df_hps[['To_Zone_X', 'To_Zone_Y']].apply(lambda x: x + np.random.uniform(-0.5, 0.5, len(x)))

# Disruptive Serving Chart Prep

df_jf = df_attack[df_attack['ServeType'] == 'Jump Float']
df_js = df_attack[df_attack['ServeType'] == 'Jump Spin']
df_fn = df_attack[df_attack['ServeType'] == 'Float Near']
df_ff = df_attack[df_attack['ServeType'] == 'Float Far']
df_jf_right = df_jf[df_jf['ReceiveSide'] == 'Right']
df_jf_left = df_jf[df_jf['ReceiveSide'] == 'Left']
df_jf_mid = df_jf[df_jf['ReceiveSide'] == 'Middle']
df_jf_low = df_jf[df_jf['ReceiveSide'] == 'Low']
df_js_right = df_js[df_js['ReceiveSide'] == 'Right']
df_js_left = df_js[df_js['ReceiveSide'] == 'Left']
df_js_mid = df_js[df_js['ReceiveSide'] == 'Middle']
df_js_low = df_js[df_js['ReceiveSide'] == 'Low']
df_fn_right = df_fn[df_fn['ReceiveSide'] == 'Right']
df_fn_left = df_fn[df_fn['ReceiveSide'] == 'Left']
df_fn_mid = df_fn[df_fn['ReceiveSide'] == 'Middle']
df_fn_low = df_fn[df_fn['ReceiveSide'] == 'Low']
df_ff_right = df_ff[df_ff['ReceiveSide'] == 'Right']
df_ff_left = df_ff[df_ff['ReceiveSide'] == 'Left']
df_ff_mid = df_ff[df_ff['ReceiveSide'] == 'Middle']
df_ff_low = df_ff[df_ff['ReceiveSide'] == 'Low']
jf_right_eff = df_jf_right['AttackOutcome'].mean()
jf_left_eff = df_jf_left['AttackOutcome'].mean()
jf_mid_eff = df_jf_mid['AttackOutcome'].mean()
jf_low_eff = df_jf_low['AttackOutcome'].mean()
js_right_eff = df_js_right['AttackOutcome'].mean()
js_left_eff = df_js_left['AttackOutcome'].mean()
js_mid_eff = df_js_mid['AttackOutcome'].mean()
js_low_eff = df_js_low['AttackOutcome'].mean()
fn_right_eff = df_fn_right['AttackOutcome'].mean()
fn_left_eff = df_fn_left['AttackOutcome'].mean()
fn_mid_eff = df_fn_mid['AttackOutcome'].mean()
fn_low_eff = df_fn_low['AttackOutcome'].mean()
ff_right_eff = df_ff_right['AttackOutcome'].mean()
ff_left_eff = df_ff_left['AttackOutcome'].mean()
ff_mid_eff = df_ff_mid['AttackOutcome'].mean()
ff_low_eff = df_ff_low['AttackOutcome'].mean()

df_jf_eff = pd.DataFrame({
    'Receive Side': ['Left', 'Low', 'Middle', 'Right'],
    'Attack Outcome': [jf_left_eff, jf_low_eff, jf_mid_eff, jf_right_eff]
})
df_js_eff = pd.DataFrame({
    'Receive Side': ['Left', 'Low', 'Middle', 'Right'],
    'Attack Outcome': [js_left_eff, js_low_eff, js_mid_eff, js_right_eff]
})
df_ff_eff = pd.DataFrame({
    'Receive Side': ['Left', 'Low', 'Middle', 'Right'],
    'Attack Outcome': [ff_left_eff, ff_low_eff, ff_mid_eff, ff_right_eff]
})
df_fn_eff = pd.DataFrame({
    'Receive Side': ['Left', 'Low', 'Middle', 'Right'],
    'Attack Outcome': [fn_left_eff, fn_low_eff, fn_mid_eff, fn_right_eff]
})

# defensive position graph prep

df_def_pos = pd.DataFrame(df_attack[['To_Zone_X', 'To_Zone_Y', 'Zone_Y', 'AttackLocation']])
df_def_pos[['To_Zone_X', 'To_Zone_Y']] = df_def_pos[['To_Zone_X', 'To_Zone_Y']].apply(lambda x: x + np.random.uniform(-0.4, 0.4, len(x)))

df_np_right = df_def_pos[(df_def_pos['AttackLocation'] == 'Net Point 4') | (df_def_pos['AttackLocation'] == 'Net Point 5')]
df_np_left = df_def_pos[(df_def_pos['AttackLocation'] == 'Net Point 1') | (df_def_pos['AttackLocation'] == 'Net Point 2')]
df_np_mid = df_def_pos[df_def_pos['AttackLocation'] == 'Net Point 3']

# EO vs AO per round graph prep

df_mhp_predict = df_mhp.drop(columns=["AttackOutcome"])
df_ohe_predict = df_ohe.drop(columns=["AttackOutcome"])

df_mhp_predict_maindraw = df_mhp_predict[df_mhp_predict['TournamentPhase_Main Draw'] == 1]
df_mhp_predict_quarters = df_mhp_predict[df_mhp_predict['TournamentPhase_Quarterfinal'] == 1]
df_mhp_predict_finals = df_mhp_predict[df_mhp_predict['TournamentPhase_Final 1st Place'] == 1]

pred_mhp_ao = model.predict(df_mhp_predict)
exp_mhp_ao = sum(pred_mhp_ao) / len(pred_mhp_ao)
pred_all_ao = model.predict(df_ohe_predict)
exp_all_ao = sum(pred_all_ao) / len(pred_all_ao)

pred_mhp_ao_quarters = model.predict(df_mhp_predict_quarters)
exp_mhp_ao_quarters = sum(pred_mhp_ao_quarters) / len(pred_mhp_ao_quarters)
pred_mhp_ao_maindraw = model.predict(df_mhp_predict_maindraw)
exp_mhp_ao_maindraw = sum(pred_mhp_ao_maindraw) / len(pred_mhp_ao_maindraw)
pred_mhp_ao_finals = model.predict(df_mhp_predict_finals)
exp_mhp_ao_finals = sum(pred_mhp_ao_finals) / len(pred_mhp_ao_finals)

Tournament_MAO = df_ohe['AttackOutcome'].mean()

df_mhp_attack = df_attack[df_attack['New_Player'] == 'Melissa Humana-Paredes']
df_mhp_attack_quarters = df_mhp_attack[df_mhp_attack['TournamentPhase'] == 'Quarterfinal']
df_mhp_attack_maindraw = df_mhp_attack[df_mhp_attack['TournamentPhase'] == 'Main Draw']
df_mhp_attack_finals = df_mhp_attack[df_mhp_attack['TournamentPhase'] == 'Final 1st Place']

mhp_all_eff = df_mhp_attack['AttackOutcome'].mean()
mhp_quarters_eff = df_mhp_attack_quarters['AttackOutcome'].mean()
mhp_maindraw_eff = df_mhp_attack_maindraw['AttackOutcome'].mean()
mhp_finals_eff = df_mhp_attack_finals['AttackOutcome'].mean()

df_mhp_quarters = pd.DataFrame({
    'Metric': ['Expected', 'Actual'],
    'Attack Outcome': [exp_mhp_ao_quarters, mhp_quarters_eff]
})

df_mhp_maindraw = pd.DataFrame({
    'Metric': ['Expected', 'Actual'],
    'Attack Outcome': [exp_mhp_ao_maindraw, mhp_maindraw_eff]
})

df_mhp_finals = pd.DataFrame({
    'Metric': ['Expected', 'Actual'],
    'Attack Outcome': [exp_mhp_ao_finals, mhp_finals_eff]
})

'''