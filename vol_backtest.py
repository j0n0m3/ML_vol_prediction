import requests
import pandas as pd
import numpy as np
import sqlalchemy
import mysql.connector
import matplotlib.pyplot as plt

from datetime import timedelta, datetime
from pandas_market_calendars import get_calendar
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from config import API_KEY, DB_LOGIN

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

polygon_api_key = API_KEY
engine = engine = sqlalchemy.create_engine(DB_LOGIN)
calendar = get_calendar("NYSE")

underlying_symbol = "SPY"

vol_dataset = pd.read_sql(sql = "SELECT * FROM asset_vol_dataset", con = engine).set_index("date")
features = ["year", "month", "day", "pre_14_volume", "pre_14_vol"]
target = "volatility_change"

#

start_date = "2022-01-01"
end_date = (datetime.today() - timedelta(days = 1)).strftime("%Y-%m-%d")

trading_dates = pd.DataFrame({"trading_dates": calendar.schedule(start_date = start_date, end_date = end_date).index.strftime("%Y-%m-%d")})

times = []
trades = []

profit_threshold = .20

trade_start = pd.Timestamp("11:00").time()
trade_end = pd.Timestamp("16:00").time()

for date in trading_dates["trading_dates"]:

    try:    
        
        start_time = datetime.now()
        underlying = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{underlying_symbol}/range/1/minute/{date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
        print(underlying)
        underlying.index = pd.to_datetime(underlying.index, unit = "ms", utc = True).tz_convert("America/New_York")
        underlying = underlying[(underlying.index.time >= pd.Timestamp("09:30").time()) & (underlying.index.time < trade_end)].add_prefix("asset_")
        underlying["year"] = underlying.index.year
        underlying["month"] = underlying.index.month
        underlying["day"] = underlying.index.day
        
        if len(underlying) < 350:
            continue
        
        underlying["atm_strike"] = round(underlying["asset_c"])
        
        pre_session = underlying[underlying.index.hour < 14].copy()
        pre_session["returns"] = abs(pre_session["asset_c"].pct_change().cumsum())
        
        post_session = underlying[underlying.index.hour >= 14].copy()
        post_session["returns"] = abs(post_session["asset_c"].pct_change().cumsum())
        
        training_data = vol_dataset[vol_dataset.index < date].copy()
        X = training_data[features].values
        Y = training_data[target].apply(Binary).values

        RandomForest_Model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2,
                                                    min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt',
                                                    max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False,
                                                    n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None,
                                                    ccp_alpha=0.0, max_samples=None).fit(X, Y)
        
        production_data = pd.DataFrame([{"year": pre_session["year"].iloc[-1], "month": pre_session["month"].iloc[-1],
                                          "day": pre_session["day"].iloc[-1],
                                          "pre_volume": round(pre_session["asset_v"].sum()),
                                        "pre_vol": round(pre_session["returns"].iloc[-1]*100, 2)}])
        
         

        X_prod = production_data[features].values
        
        random_forest_prediction = RandomForest_Model.predict(X_prod)
        random_forest_prediction_probability = RandomForest_Model.predict_proba(X_prod)
        
        random_forest_prediction_dataframe = pd.DataFrame({"prediction": random_forest_prediction})
        random_forest_prediction_dataframe["probability_0"] = random_forest_prediction_probability[:,0]
        random_forest_prediction_dataframe["probability_1"] = random_forest_prediction_probability[:,1]
        random_forest_prediction_dataframe["probability"] = return_probability(random_forest_prediction_dataframe)
        probability = random_forest_prediction_dataframe["probability"].iloc[0]
        prediction = random_forest_prediction_dataframe["prediction"].iloc[0]
        
        if prediction == 0:
            continue
        
        trades = post_14_session.head(1).copy()
        
        price = trades["asset_c"].iloc[0]
        returns = pre_session["returns"].iloc[-1]
        
        long_put_strike = trades["atm_strike"].iloc[0]
        long_call_strike  = trades["atm_strike"].iloc[0]
        
        # Options
        
        Put_Contracts = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={underlying_symbol}&contract_type=put&expiration_date={date}&as_of={date}&expired=false&limit=1000&apiKey={polygon_api_key}").json()["results"])
        Call_Contracts = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={underlying_tymbol&contract_type=call&expiration_date={date}&as_of={date}&expired=false&limit=1000&apiKey={polygon_api_key}").json()["results"])
        
        Long_Put_Symbol = Put_Contracts[Put_Contracts["strike_price"] == long_put_strike]["ticker"].iloc[0]
        Long_Call_Symbol = Call_Contracts[Call_Contracts["strike_price"] == long_call_strike]["ticker"].iloc[0]
        long_put_ohlcv = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{Long_Put_Symbol}/range/1/second/{date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
        long_put_ohlcv.index = pd.to_datetime(long_put_ohlcv.index, unit = "ms", utc = True).tz_convert("America/New_York")
        long_call_ohlcv = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{Long_Call_Symbol}/range/1/second/{date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
        long_call_ohlcv.index = pd.to_datetime(long_call_ohlcv.index, unit = "ms", utc = True).tz_convert("America/New_York")
        
        straddle = pd.concat([long_put_ohlcv.add_prefix("put_"), long_call_ohlcv.add_prefix("call_")], axis  = 1).dropna()
        straddle = straddle[straddle.index.time >= trade_start]
        straddle["straddle_value"] = straddle["put_c"] + straddle["call_c"]
        original_cost = straddle["straddle_value"].iloc[0]
        straddle["straddle_pnl"] = (straddle["straddle_value"] - original_cost) / original_cost
        
        closing_straddle = []
        
        for minute in straddle.index:
            minute_data = straddle[straddle.index == minute].copy()
            current_pnl = minute_data["straddle_pnl"].iloc[0]
            
            if current_pnl >= profit_threshold:
                closing_straddle = minute_data.copy()
                break
            # if no trigger event happens by 15:00
            elif (len(closing_straddle) < 1) and minute.time() >= pd.Timestamp("15:00").time():
                closing_straddle = minute_data.copy()
                break
            
        open_price = original_cost
        closing_price = closing_straddle["straddle_value"].iloc[0]
        
        gross_pnl = closing_price - open_price
        actual = Binarizer(post_14_session["returns"].iloc[-1] - pre_14_session["returns"].iloc[-1])
        print(gross_pnl)
        trade_dataframe = pd.DataFrame([{"date": pd.to_datetime(date), "prediction": prediction,
                                          "probability": probability, "open_price": open_price,
                                          "closing_price": closing_price, "gross_pnl": gross_pnl,
                                          "actual": actual, "closing_time": closing_straddle.index[0],
                                          "pnl_percent": gross_pnl / original_cost,
                                          "pre_14_vol": pre_14_session["returns"].iloc[-1]*100,
                                          "post_14_vol": post_14_session["returns"].iloc[-1]*100}])
        
        trades.append(trade_dataframe)
        print("trades: ", trades)        

        end_time = datetime.now()
        sec_to_complete = (end_time - start_time).total_seconds()
        times.append(sec_to_complete)
        iteration = round((np.where(trading_dates["trading_dates"]==date)[0][0]/len(trading_dates.index))*100,2)
        iterations_remaining = len(trading_dates["trading_dates"]) - np.where(trading_dates["trading_dates"]==date)[0][0]
        avg_time_to_complete = np.mean(times)
        est_completion_time = (datetime.now() + timedelta(seconds = int(average_time_to_complete*iterations_remaining)))
        time_remaining = estimated_completion_time - datetime.now()
        
        print(f"{iteration}% complete, {time_remaining} left, ETA: {estimated_completion_time}")
    except Exception as error_message:
        print(error_message)
        continue
    

trading_records = pd.concat(trades).set_index("date")
print("trading_records: ", trading_records)

trading_records = trading_records[trading_records["open_price"] <= .5]
trading_records = trading_records[trading_records["probability"] >= .65]

# We set the position size to 10 contracts
trading_records["gross_pnl"] = trading_records["gross_pnl"] * 10
trading_records["capital"] = 1000 + (trading_records["gross_pnl"].cumsum())*100

wins = trading_records[(trading_records["gross_pnl"] > 0)].copy()
losses = trading_records[(trading_records["gross_pnl"] < 0)].copy()

average_win = wins["gross_pnl"].mean()
average_loss = losses["gross_pnl"].mean()

monthly_sum = trading_records.resample('M').sum(numeric_only = True)

accuracy_rate = len(trading_records[trading_records["prediction"] == trading_records["actual"]]) / len(trading_records)
win_rate = len(trading_records[trading_records["gross_pnl"] > 0]) / len(trading_records)

expected_value = (win_rate * average_win) + ((1-win_rate) * average_loss)
print(f"\nAccuracy Rate: {round(accuracy_rate*100, 2)}%")
print(f"Win Rate: {round(win_rate*100, 2)}%")
print(f"Expected Value per Trade ${expected_value*100}")
print(f"Average Monthly Profit: ${monthly_sum['gross_pnl'].mean()*100}")

plt.figure(dpi = 200)
plt.xticks(rotation = 45)
plt.title("Buy Straddle When Prediction > 1")

plt.plot(trading_records["capital"])

plt.legend(["gross_pnl"])
plt.show()