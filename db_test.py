import pandas as pd
import sqlalchemy
import mysql.connector
import yfinance

from datetime import datetime, timedelta
from config import DB_LOGIN

initial_engine = sqlalchemy.create_engine(DB_LOGIN)

with initial_engine.connect() as conn:
    conn.execute("CREATE DATABASE quant_data")

engine = sqlalchemy.create_engine(DB_LOGIN)

historical_data = yfinance.download("SPY", start = (datetime.today() - timedelta(days = 30)).strftime("%Y-%m-%d"), end = datetime.today().strftime("%Y-%m-%d"))

historical_data.to_sql("asset_data", con = engine, if_exists = "replace")

stored_data = pd.read_sql("SELECT * FROM asset_data", con = engine)