# stocks/views.py

from django.shortcuts import render
from .models import StockData
from django.http import JsonResponse
import pandas as pd
import csv
from django.http import HttpResponse
import pickle
import numpy as np
from django.http import HttpResponse
import io
from django.core.cache import cache
from .tasks import fetch_stock_data_task
from .tasks import backtest_strategy
from datetime import timedelta
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from django.http import HttpResponse
from django.core.cache import cache
from .models import StockData
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import base64
from decouple import config
from django.core.cache import caches
import redis
from django_redis import get_redis_connection


# View to trigger fetching stock data
def fetch_stock_data_view(request):
    """
    Handles user requests to fetch stock data for a specific symbol or for default symbols.
    Uses Celery task to perform the data fetching in the background.
    """
    symbol = request.GET.get('symbol', '').upper()
    if symbol:
        # Fetch data for a specific symbol
        fetch_stock_data_task.delay(symbol)
        return JsonResponse({"message": f"Data fetching for symbol {symbol} initiated, please check back later."})
    else:
        # Fetch data for default symbols
        fetch_stock_data_task.delay()
        return JsonResponse({"message": "Data fetching for default symbols initiated, please check back later."})


# View to backtest a trading strategy
def backtest_strategy_view(request):
    """
    Runs a backtest on the requested symbol based on user-supplied buy and sell window parameters.
    If no data is available, triggers data fetching first.
    """
    initial_investment = float(request.GET.get("initial_investment", 10000))
    symbol = request.GET.get("symbol", "").upper()
    buy_window = int(request.GET.get("buy_window", 50))
    sell_window = int(request.GET.get("sell_window", 200))

    # Retrieve stock data for the given symbol
    stock_data = StockData.objects.filter(
        symbol=symbol,
        prediction__isnull=True,  # Only include rows that are not predictions
        close_price__gt=0  # Only include rows with a valid close price (greater than zero)
    ).order_by('date')

    if not stock_data.exists():
        # If no data exists, trigger fetching first
        fetch_stock_data_task.delay(symbol)
        return JsonResponse({
            "error": f"No data available for the symbol {symbol}. Data fetch initiated, please try again in a few moments.",
            "status": 202  # 202 Accepted - data processing has been started
        }, status=202)

    # Prepare the data for backtesting
    df = pd.DataFrame(list(stock_data.values('date', 'close_price')))
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Validate that there is enough data for backtesting
    if df.shape[0] < max(buy_window, sell_window):
        return JsonResponse({
            "error": f"Not enough data for the given moving average windows (buy: {buy_window}, sell: {sell_window})",
            "status": 400
        }, status=400)

    # Perform backtesting and return the result
    result = backtest_strategy(initial_investment=initial_investment, df=df, symbol=symbol, buy_window=buy_window, sell_window=sell_window)
    return JsonResponse(result)

# View to export stock data to CSV
def export_stock_data(request, symbol=None):
    """
    Exports stock data for a given symbol or all symbols as a CSV file.
    If no symbol is provided, exports data for all available symbols.
    """
    symbol = request.GET.get('symbol', symbol)

    if symbol:
        # Query data for the specific symbol
        stock_data = StockData.objects.filter(symbol=symbol.upper()).order_by('date')
        file_name = f"{symbol.upper()}_stock_data.csv"
    else:
        # Query data for all symbols
        stock_data = StockData.objects.all().order_by('symbol', 'date')
        file_name = "all_symbols_stock_data.csv"

    # Prepare CSV response
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{file_name}"'

    # Write CSV header and data rows
    writer = csv.writer(response)
    writer.writerow(['id', 'symbol', 'date', 'open_price', 'close_price', 'high_price', 'low_price', 'volume', 'prediction'])
    for record in stock_data:
        writer.writerow([
            record.id,
            record.symbol,
            record.date,
            record.open_price,
            record.close_price,
            record.high_price,
            record.low_price,
            record.volume,
            record.prediction if record.prediction is not None else ''  # Leave empty if prediction is not available
        ])

    return response

# Utility function to calculate RSI
def calculate_rsi(series, period=14):
    """
    Calculates the Relative Strength Index (RSI) of a given price series.
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# View to predict future stock prices using a pre-trained model
def predict_stock_prices(request):
    """
    Predicts future stock prices for the next 30 days based on historical data and a pre-trained model i.e Random Forest.
    Stores the predicted data in the database.
    """
    symbol = request.GET.get('symbol', 'AAPL').upper()
    try:
        # Load the pre-trained model
        model_path = config('MODEL_FILE_PATH')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Fetch the latest 50 rows of data for the requested symbol
        stock_data = StockData.objects.filter(symbol=symbol).order_by('-date')[:50]
        if not stock_data.exists():
            return JsonResponse({"error": f"No data available for symbol {symbol}. Please fetch the data first."}, status=400)

        # Convert the data to a DataFrame
        stock_data = sorted(stock_data, key=lambda x: x.date)
        data_dicts = [record.__dict__ for record in stock_data]
        df = pd.DataFrame(data_dicts)

        # Convert date to numeric feature and calculate other features
        df['date'] = pd.to_datetime(df['date'])
        df['date_ordinal'] = df['date'].map(lambda d: d.toordinal())
        df['7_day_avg'] = df['close_price'].rolling(window=7).mean()
        df['14_day_avg'] = df['close_price'].rolling(window=14).mean()
        df['30_day_avg'] = df['close_price'].rolling(window=30).mean()
        df['50_day_avg'] = df['close_price'].rolling(window=50).mean()
        df['ema_14'] = df['close_price'].ewm(span=14, adjust=False).mean()
        df['rolling_std_20'] = df['close_price'].rolling(window=20).std()
        df['bollinger_upper'] = df['30_day_avg'] + (df['rolling_std_20'] * 2)
        df['bollinger_lower'] = df['30_day_avg'] - (df['rolling_std_20'] * 2)
        df['rsi'] = calculate_rsi(df['close_price'])
        df['momentum'] = df['close_price'].diff(4)

        # Fill NaN values due to rolling calculations
        df.fillna(0, inplace=True)

        # Define the features for the model
        features = [
            'date_ordinal', '7_day_avg', '14_day_avg', '30_day_avg', '50_day_avg', 'volume',
            'ema_14', 'bollinger_upper', 'bollinger_lower', 'rsi', 'momentum'
        ]

        # Use the most recent data row for prediction
        recent_features = df[features].values[-1]

        # Predict stock prices for the next 30 days
        predictions = []
        last_entry = stock_data[-1]
        for day in range(1, 31):
            next_features = np.array(recent_features).reshape(1, -1)
            predicted_price = model.predict(next_features)[0]
            predicted_price = float(predicted_price)

            # Append the prediction to the results
            prediction_date = last_entry.date + timedelta(days=day)
            predictions.append({'date': str(prediction_date), 'predicted_price': predicted_price})

            # Update features for subsequent predictions
            recent_features[0] = prediction_date.toordinal()  # Update date
            recent_features[1] = (recent_features[1] * 6 + predicted_price) / 7  # Update 7-day average
            recent_features[2] = (recent_features[2] * 13 + predicted_price) / 14  # Update 14-day average
            recent_features[3] = (recent_features[3] * 29 + predicted_price) / 30  # Update 30-day average
            recent_features[4] = (recent_features[4] * 49 + predicted_price) / 50  # Update 50-day average

            # Volume update assumption: rolling average of recent volumes
            previous_volumes = df['volume'].tail(7).tolist()
            avg_volume = np.mean(previous_volumes) if previous_volumes else last_entry.volume
            recent_features[5] = avg_volume

            # Update EMA using the smoothing formula
            alpha = 2 / (14 + 1)
            recent_features[6] = (predicted_price * alpha) + (recent_features[6] * (1 - alpha))

            # Update Bollinger Bands with updated standard deviation
            historical_prices = df['close_price'].tolist() + [predicted_price]
            rolling_std = pd.Series(historical_prices).rolling(window=20).std().iloc[-1]
            recent_features[7] = recent_features[3] + (rolling_std * 2)  # Update upper band
            recent_features[8] = recent_features[3] - (rolling_std * 2)  # Update lower band

            # Update RSI with new data
            historical_prices_series = pd.Series(historical_prices)
            recent_features[9] = calculate_rsi(historical_prices_series, period=14).iloc[-1]

            # Update momentum (last 4-day difference)
            if len(historical_prices) >= 5:
                recent_features[10] = historical_prices[-1] - historical_prices[-5]
            else:
                recent_features[10] = 0

            # Store prediction in the database
            StockData.objects.update_or_create(
                symbol=symbol,
                date=prediction_date,
                defaults={
                    'open_price': 0,
                    'close_price': 0,
                    'high_price': 0,
                    'low_price': 0,
                    'volume': 0,
                    'prediction': predicted_price
                }
            )

        return JsonResponse({"symbol": symbol, "predictions": predictions})

    except FileNotFoundError:
        return JsonResponse({"error": "Model file not found. Please ensure the model is available."}, status=500)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)



# View to generate a stock report as a PDF or JSON response
def generate_report(request):
    """
    Generates a backtesting and prediction report for the given stock symbol.
    Supports both JSON and PDF formats based on the 'format' query parameter.
    """
    symbol = request.GET.get('symbol', 'AAPL').upper()
    format_type = request.GET.get('format', 'pdf').lower()

    # Get Redis connection and check the key with the version handled
    cache_instance = caches['default']  # Use Django cache instance
    redis_conn = get_redis_connection('default')

    # Dynamically search for the correct key based on symbol
    keys = redis_conn.keys(f"*backtest_result_{symbol}_*")
    if not keys:
        return JsonResponse({"error": f"No backtesting result found for symbol {symbol}. Please run the backtest first."}, status=400)

    # Assuming we use the first matching key found
    backtest_key = keys[0].decode('utf-8')

    # Extract the base key without prefix or version
    if ":" in backtest_key:
        base_key = backtest_key.split(":", 2)[-1]  # Only take the part after prefixes

    # Fetch the backtest result from cache, letting the cache system handle version and prefix
    backtest_result = cache_instance.get(base_key)


    if not backtest_result:
        return JsonResponse({"error": "No backtesting result found. Please run the backtest first."}, status=400)

    # Extract backtest metrics
    initial_investment = backtest_result.get('initial_investment', 10000)
    final_balance = backtest_result.get('final_balance', 10000)
    profit = backtest_result.get('profit', 0)
    total_roi = backtest_result.get('total_ROI_percentage', 0)
    max_drawdown = backtest_result.get('max_drawdown_percentage', 0)
    number_of_trades = backtest_result.get('number_of_trades', 0)

    # Fetch historical stock data
    stock_data = StockData.objects.filter(symbol=symbol).order_by('date')
    if not stock_data.exists():
        return JsonResponse({"error": f"No data available for symbol {symbol}."}, status=400)

    # Convert stock data to DataFrame
    data_dicts = [record.__dict__ for record in stock_data]
    df = pd.DataFrame(data_dicts)

    # Check if predictions are available
    predictions_available = 'prediction' in df.columns and not df['prediction'].isna().all()

    # Separate historical and predicted data
    historical_data = df[df['prediction'].isna()] if predictions_available else df
    predicted_data = df[~df['prediction'].isna()] if predictions_available else pd.DataFrame()

    # Create a plot if predictions are available
    encoded_chart_image = None
    if predictions_available and not predicted_data.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(historical_data['date'], historical_data['close_price'], label='Actual Price', color='blue', linewidth=2)
        plt.plot(predicted_data['date'], predicted_data['prediction'], label='Predicted Price', color='red', linestyle='--', linewidth=2)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Stock Price ($)', fontsize=12)
        plt.title(f'Actual vs Predicted Stock Prices for {symbol}', fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot to a BytesIO object and encode it in base64
        plot_buffer = io.BytesIO()
        plt.savefig(plot_buffer, format='png', dpi=300)
        plt.close()
        plot_buffer.seek(0)
        encoded_chart_image = base64.b64encode(plot_buffer.read()).decode('utf-8')

    # Return JSON report if requested
    if format_type == 'json':
        json_response = {
            "symbol": symbol,
            "initial_investment": initial_investment,
            "final_balance": final_balance,
            "profit": profit,
            "total_roi_percentage": total_roi,
            "max_drawdown_percentage": max_drawdown,
            "number_of_trades": number_of_trades,
        }

        if predictions_available and not predicted_data.empty:
            json_response["predictions"] = predicted_data[['date', 'prediction']].to_dict(orient='records')
        if encoded_chart_image:
            json_response["chart_image_base64"] = encoded_chart_image

        return JsonResponse(json_response)

    # Generate a PDF report if requested
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    styles = getSampleStyleSheet()

    # Add custom styles
    title_style = ParagraphStyle(
        name='TitleStyle',
        fontSize=24,
        leading=28,
        alignment=TA_CENTER,
        spaceAfter=24,
        textColor=colors.HexColor('#4B89DC'),
        fontName='Helvetica-Bold'
    )
    content_style = ParagraphStyle(
        name='ContentStyle',
        fontSize=12,
        leading=15,
        alignment=TA_LEFT,
        spaceAfter=12,
        fontName='Helvetica'
    )

    elements = []

    # Title Page
    elements.append(Paragraph(f"Stock Backtesting and Prediction Report", title_style))
    elements.append(Spacer(1, 24))
    elements.append(Paragraph(f"Symbol: {symbol}", content_style))
    elements.append(Spacer(1, 48))
    elements.append(Paragraph("Prepared by: Yuvraj Singh Chowdhary", content_style))
    elements.append(Paragraph("Prepared for: Blockhouse Work Trial Task", content_style))
    elements.append(Paragraph(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}", content_style))
    elements.append(PageBreak())

    # Backtesting Summary
    elements.append(Paragraph("Backtesting Summary", styles['Heading2']))
    metrics_data = [
        ['Metric', 'Value'],
        ['Initial Investment', f"${initial_investment:,.2f}"],
        ['Final Balance', f"${final_balance:,.2f}"],
        ['Profit', f"${profit:,.2f}"],
        ['Total ROI (%)', f"{total_roi:.2f}%"],
        ['Max Drawdown (%)', f"{max_drawdown:.2f}%"],
        ['Number of Trades', number_of_trades]
    ]
    metrics_table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4B89DC')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F5F5F5')),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ])
    metrics_table = Table(metrics_data, style=metrics_table_style, colWidths=[200, 200])
    elements.append(metrics_table)
    elements.append(Spacer(1, 24))

    # Add Line Chart Image to Report if predictions are available
    if predictions_available and encoded_chart_image:
        elements.append(Paragraph("Actual vs Predicted Stock Prices", styles['Heading2']))
        img = Image(io.BytesIO(base64.b64decode(encoded_chart_image)))
        img.drawHeight = 4 * 72  # 4 inches
        img.drawWidth = 6.5 * 72  # 6.5 inches
        elements.append(img)
        elements.append(Spacer(1, 24))

    # Summary of Predictions if available
    if predictions_available and not predicted_data.empty:
        elements.append(Paragraph(f"Predicted Stock Prices for the Next {predicted_data.shape[0]} Days", styles['Heading2']))
        predicted_summary = predicted_data[['date', 'prediction']].copy()
        predicted_summary['date'] = predicted_summary['date'].astype(str)
        predictions_table_data = [['Date', 'Predicted Price ($)']] + list(predicted_summary.to_records(index=False))
        predictions_table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4B89DC')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F5F5F5')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ])
        predictions_table = Table(predictions_table_data, style=predictions_table_style, colWidths=[200, 200])
        elements.append(predictions_table)

    # Build and return the PDF response
    doc.build(elements)
    buffer.seek(0)
    return HttpResponse(buffer, content_type='application/pdf')

