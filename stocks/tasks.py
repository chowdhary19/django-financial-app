# tasks.py

from celery import shared_task
from .models import StockData
import requests
from datetime import datetime
from requests.exceptions import RequestException
from django.db import IntegrityError
import itertools
from django.core.cache import cache
import time
import logging
from decouple import config

# Setup logging for production
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load API keys from environment variable
api_keys_string = config('ALPHAVANTAGE_API_KEYS')
API_KEYS = api_keys_string.split(',')  # Split the comma-separated string into a list
api_key_cycle = itertools.cycle(API_KEYS)  # Create an infinite cycle of API keys for rate limit handling

@shared_task
def fetch_stock_data_task(symbol=None):
    """
    Celery task to fetch stock data for a given symbol or default symbols if none provided.
    The data is fetched using the Alpha Vantage API and saved into the database.
    """
    symbols = [symbol] if symbol else [
        'AAPL', 'IBM', 'GOOG', 'AMZN', 'MSFT', 
        'META', 'NFLX', 'NVDA', 'TSLA', 'ORCL'
    ]
    
    for symbol in symbols:
        # Ensure symbol is in uppercase
        symbol = symbol.upper()

        # Get the latest available date for this symbol in the database
        last_date_entry = StockData.objects.filter(symbol=symbol).order_by('-date').first()
        last_date = last_date_entry.date if last_date_entry else None

        # Get the next available API key from the cycle
        api_key = next(api_key_cycle)

        # Construct URL to fetch data
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}'

        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise exception for HTTP errors
        except RequestException as e:
            logger.error(f"Network error for {symbol}: {e}")
            continue

        data = response.json()

        # Check for rate limit or other API errors
        if 'Time Series (Daily)' not in data:
            if 'Note' in data or 'Information' in data:
                logger.warning(f"Rate limit reached or issue with API key for symbol {symbol}. Message: {data.get('Note') or data.get('Information')}")
                time.sleep(60)  # Wait for a minute to handle rate limit
                continue
            logger.error(f"Unexpected API response for {symbol}: {data}")
            continue

        # Parse time series data
        time_series = data['Time Series (Daily)']
        new_records = 0
        for date_str, daily_data in time_series.items():
            date = datetime.strptime(date_str, '%Y-%m-%d').date()

            # Skip already existing dates in the database
            if last_date and date <= last_date:
                continue

            try:
                # Save or update stock data record
                StockData.objects.update_or_create(
                    symbol=symbol,
                    date=date,
                    defaults={
                        'open_price': daily_data['1. open'],
                        'close_price': daily_data['4. close'],
                        'high_price': daily_data['2. high'],
                        'low_price': daily_data['3. low'],
                        'volume': daily_data['5. volume'],
                    }
                )
                new_records += 1
            except IntegrityError as e:
                logger.error(f"Error saving data for {symbol} on {date}: {e}")

        if new_records > 0:
            logger.info(f"Successfully fetched and stored {new_records} records for symbol: {symbol}")
        else:
            logger.info(f"No new data to update for symbol: {symbol}")

        time.sleep(12)  # Delay to prevent rate limiting

def backtest_strategy(initial_investment, df, symbol, buy_window, sell_window):
    """
    Backtests a trading strategy based on the provided data frame and parameters.
    The strategy is a moving average crossover strategy where buy and sell signals are generated based on two moving averages.
    
    Args:
    - initial_investment (float): The amount of money to start with.
    - df (DataFrame): A DataFrame containing stock data.
    - symbol (str): The stock symbol to backtest.
    - buy_window (int): The window size for the buy moving average.
    - sell_window (int): The window size for the sell moving average.

    Returns:
    - dict: A dictionary containing the backtest results.
    """
    # Validate input parameters
    if initial_investment <= 0:
        return {"error": "Initial investment must be greater than zero", "status": 400}
    if buy_window >= sell_window:
        return {"error": "Buy window must be smaller than the Sell window", "status": 400}
    if df.empty or 'close_price' not in df.columns:
        return {"error": "Insufficient data for backtesting. Please check the symbol or data.", "status": 400}

    # Adding moving averages to the data
    df['buy_MA'] = df['close_price'].rolling(window=buy_window, min_periods=1).mean()
    df['sell_MA'] = df['close_price'].rolling(window=sell_window, min_periods=1).mean()

    # Ensure there is enough data to perform backtesting
    if df.shape[0] < sell_window:
        return {"error": f"Not enough data for the given moving average windows (buy: {buy_window}, sell: {sell_window})", "status": 400}

    # Initialize backtesting parameters
    balance = initial_investment
    position = 0  # 1 if holding stock, 0 otherwise
    shares = 0
    max_balance = initial_investment
    max_drawdown = 0
    trades = 0

    # Iterate over each row in the DataFrame
    for current_date, row in df.iterrows():
        # Buy Signal - Buy when price drops below the buy moving average
        if position == 0 and row['close_price'] < row['buy_MA']:
            shares = balance / row['close_price']
            balance = 0
            position = 1
            trades += 1

        # Sell Signal - Sell when price rises above the sell moving average
        elif position == 1 and row['close_price'] > row['sell_MA']:
            balance = shares * row['close_price']
            shares = 0
            position = 0
            trades += 1

        # Calculate current balance for drawdown calculations
        current_balance = balance + (shares * row['close_price']) if position == 1 else balance
        max_balance = max(max_balance, current_balance)
        if max_balance > 0:
            drawdown = (max_balance - current_balance) / max_balance
            max_drawdown = max(max_drawdown, drawdown)

    # Finalize balance if still holding a position
    if position == 1:
        balance = shares * df['close_price'].iloc[-1]

    # Calculate profit and return metrics
    profit = balance - initial_investment
    total_return = (profit / initial_investment) * 100  # Return as a percentage

    # Store results in cache for further use (e.g., generating reports)
    result = {
        "initial_investment": initial_investment,
        "final_balance": balance,
        "profit": profit,
        "total_ROI_percentage": total_return,
        "max_drawdown_percentage": max_drawdown * 100,
        "number_of_trades": trades,
        "status": 200
    }
    cache.set(f'backtest_result_{symbol}_{buy_window}_{sell_window}', result, timeout=300)

    return result
