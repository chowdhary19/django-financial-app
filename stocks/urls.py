# stocks/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('fetch/', views.fetch_stock_data_view, name='fetch_stock_data'),  # Endpoint to fetch and store stock data from external sources
    path('backtest/', views.backtest_strategy_view, name='backtest_strategy'),  # Endpoint to run backtesting based on stored data
    path('export/', views.export_stock_data, name='export_all_stock_data'),  # Export all stored stock data as a downloadable file
    path('export/<str:symbol>/', views.export_stock_data, name='export_stock_data'),  # Export specific stock data for a given symbol
    path('predict/', views.predict_stock_prices, name='predict_stock_prices'),  # Endpoint for predicting stock prices based on historical data
    path('report/', views.generate_report, name='generate_report'),  # Endpoint to generate analysis report on the stock data
]



