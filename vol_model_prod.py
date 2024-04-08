import requests
import pandas as pd
import numpy as np
import sqlalchemy
import mysql.connector
import matplotlib.pyplot as plt
import pytz
import sys

from datetime import timedelta, datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from email_setup import generate_message
from config import API_KEY

def Binary(number):
    if number <= 0:
        return 0
    elif number > 0:
        return 1
def return_probability(prediction_dataset):
    probabilities = []
    for row in prediction_dataset.index:
        prediction_data = prediction_dataset[prediction_dataset.index == row]
        prediction = prediction_data["prediction"].iloc[0]
        if prediction == 0:
            probabilities.append(prediction_data["probability_0"].iloc[0])
        elif prediction == 1:
            probabilities.append(prediction_data["probability_1"].iloc[0])
    return probabilities   

weekends = ["Saturday", "Sunday"]

if datetime.now(tz=pytz.timezone('America/New_York')).date().strftime("%A") in weekends:
    send_message(message = "Off Trading Hours", subject = f"System Status Offline")
    sys.exit()

polygon_api_key = API_KEY

engine = sqlalchemy.create_engine(API_KEY)

vol_dataset = pd.read_sql(sql = "asset_vol_dataset", con = engine).set_index("date")
features = ["year", "month", "day", "pre_volume", "pre_vol"]
target = "volatility_change"

underlying_symbol = "SPY"
date = datetime.now(tz=pytz.timezone("America/New_York")).strftime("%Y-%m-%d")

training_dataset = vol_dataset[vol_dataset.index < date].copy()

#

Y_Classification = training_dataset[target].apply(Binary).values
Y_Regression = training_dataset[target].values
X_Classification = training_dataset[features].values
X_Regression = training_dataset[features].values

#

RandomForest_Classification_Model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=21, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None).fit(X_Classification, Y_Classification)
RandomForest_Regression_Model = RandomForestRegressor(n_estimators=100, criterion='squared_error', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=1.0, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=21, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None).fit(X_Regression, Y_Regression)

#

underlying = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{underlying_symbol}/range/1/minute/{date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
underlying.index = pd.to_datetime(underlying.index, unit = "ms", utc = True).tz_convert("America/New_York")
underlying = underlying[(underlying.index.time >= pd.Timestamp("09:30").time())].add_prefix("asset_")
underlying["day"] = underlying.index.day
underlying["month"] = underlying.index.month
underlying["year"] = underlying.index.year

pre_session = underlying[underlying.index.hour < 14].copy()
pre_session["returns"] = abs(pre_session["asset_c"].pct_change().cumsum())


production_data = pd.DataFrame([{"year": pre_session["year"].iloc[-1], "month": pre_session["month"].iloc[-1],
                                  "day": pre_session["day"].iloc[-1],
                                  "pre_olume": round(pre_session["asset_v"].sum()),
                                "pre_vol": round(pre_session["returns"].iloc[-1]*100, 2)}])

session_vol = production_data["pre_vol"].iloc[0]

X_prod = production_data[features].values

# classification

random_forest_classification_prediction = RandomForest_Classification_Model.predict(X_prod)
random_forest_classification_prediction_probability = RandomForest_Classification_Model.predict_proba(X_prod)

random_forest_prediction_dataframe = pd.DataFrame({"prediction": random_forest_classification_prediction})
random_forest_prediction_dataframe["probability_0"] = random_forest_classification_prediction_probability[:,0]
random_forest_prediction_dataframe["probability_1"] = random_forest_classification_prediction_probability[:,1]
random_forest_prediction_dataframe["probability"] = return_probability(random_forest_prediction_dataframe)

probability = random_forest_prediction_dataframe["probability"].iloc[0]
prediction = random_forest_prediction_dataframe["prediction"].iloc[0]

# regression

random_forest_regression_prediction = RandomForest_Regression_Model.predict(X_prod)[0]
regression_expected_vol = session_vol + random_forest_regression_prediction

if prediction == 1:
    prediction_string = f"Higher vol expected: {round(probability*100,2)}# confidence. Predicted to move: {round(regression_expected_vol,2)}%"
    send_message(subject = "0-DTE Straddle Prediction")
elif prediction == 0:
    prediction_string = "Lower vol expected: {round(probability*100,2)% confidence."