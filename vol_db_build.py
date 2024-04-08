import requests
import pandas as pd
import numpy as np
import sqlalchemy
import mysql.connector
import pytz
import sys

from datetime import timedelta, datetime
from pandas_market_calendars import get_calendar
from email_setup import generate_message
from config import API_KEY, DB_LOGIN

cal = get_calendar("NYSE")
polygon_api_key=API_KEY

initial_time = datetime.now()

weekends = ["Saturday", "Sunday"]
today = datetime.now(tz=pytz.timezone("America/New_York")).date()

if datetime.now(tz=pytz.timezone("America/New_York")) in weekends:
    sys.exit()

start_date = "2015-01-01"
end_date = (today - timedelta(days = 1)).strftime("%Y-%m-%d")

trade_dates = pd.DataFrame({"trade_dates": cal.schedule(start_date = start_date, end_date = end_date).index.strftime("%Y-%m-%d")})

underlying_symbol = "SPY"

volatility_list = []
times = []

for date in trade_dates["trade_dates"]:
    
    try:

        start_time = datetime.now()

        underlying = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{underlying_symbol}/range/1/minute/{date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
        underlying.index = pd.to_datetime(underlying.index, unit = "ms", utc = True).tz_convert("America/New_York")
        underlying = underlying[(underlying.index.time >= pd.Timestamp("09:30").time()) & (underlying.index.time < pd.Timestamp("16:01").time())]

        if len(underlying) < 350:
            continue

        pre_session = underlying[underlying.index.hour < 14].copy()
        pre_session["returns"] = abs(pre_session["c"].pct_change().cumsum())

        post_session = underlying[underlying.index.hour >= 14].copy()
        post_session["returns"] = abs(post_session["c"].pct_change().cumsum())

        vol_dataframe = pd.DataFrame([{"date": pd.to_datetime(date),
                                                "pre_vol": round(pre_session["returns"].iloc[-1]*100, 2),
                                                "pre_volume": round(pre_session["v"].sum()),
                                                "pre_price": pre_session["c"].iloc[-1],
                                                "post_vol": round(post_session["returns"].iloc[-1]*100, 2),
                                                "post_volume": round(post_session["v"].sum())}])

        vol_list.append(vol_dataframe)
        end_time = datetime.now()
        seconds_to_complete = (end_time - start_time).total_seconds()
        times.append(seconds_to_complete)
        iteration = round((np.where(trade_dates["trade_dates"]==date)[0][0]/len(trade_dates.index))*100, 2)
        iterations_remaining = len(trade_dates["trade_dates"]) - np.where(trade_dates["trade_dates"]==date)[0][0]
        average_time_to_complete = np.mean(times)
        estimated_completion_time = (datetime.now() + timedelta(seconds = int(average_time_to_complete*iterations_remaining)))
        time_remaining = estimated_completion_time - datetime.now()

        print(f"{iteration}% complete, {time_remaining} left, ETA: {estimated_completion_time}")

    except Exception as error:
        print(error)
        continue

vol_dataset = pd.concat(vol_list).set_index("date")
vol_dataset["volatility_change"] = vol_dataset["post_14_vol"] - vol_dataset["pre_14_vol"]
print(len(vol_dataset[vol_dataset["volatility_change"] > 0]) / len(vol_dataset))

training_dataset = vol_dataset.copy()
training_dataset["year"] = training_dataset.index.year
training_dataset["month"] = training_dataset.index.month
training_dataset["day"] = training_dataset.index.day

engine = engine = sqlalchemy.create_engine(DB_LOGIN)
training_dataset.to_sql("asset_vol_dataset", con = engine, if_exists = "replace")

final_time = datetime.now()

database_string = f"Vol DB built on: {end_time}, Build Duration: {final_time-initial_time}, Last Data Point: {training_dataset.index[-1].strftime('%Y-%m-%d')}"
print(database_string)
generate_message(message = database_string, subject = f"Vol DB Op on {today.strftime('%A')}, {end_time}")