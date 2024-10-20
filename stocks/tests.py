from django.test import TestCase
from .tasks import backtest_strategy
from .models import StockData
from datetime import datetime
import pandas as pd


class BacktestStrategyTest(TestCase):
    
    # Constants for repeated values
    SYMBOL = "AAPL"
    INITIAL_INVESTMENT = 10000

    def setUp(self):
        """
        Set up the database with sample data for the "AAPL" stock.
        This data will be used across multiple test cases.
        """
        self.create_stock_data(self.SYMBOL, [
            (datetime(2022, 1, 1), 100, 105, 110, 95, 1000),
            (datetime(2022, 1, 2), 105, 103, 108, 102, 1200),
            (datetime(2022, 1, 3), 103, 101, 106, 100, 1100),
            (datetime(2022, 1, 4), 101, 107, 110, 99, 1150),
            (datetime(2022, 1, 5), 107, 109, 115, 105, 1250)
        ])

    def tearDown(self):
        """
        Clean up any created data after each test to ensure test isolation.
        """
        StockData.objects.all().delete()

    def create_stock_data(self, symbol, data):
        """
        Helper function to create stock data.
        :param symbol: Stock symbol (e.g., 'AAPL')
        :param data: List of tuples containing (date, open_price, close_price, high_price, low_price, volume)
        """
        for date, open_price, close_price, high_price, low_price, volume in data:
            StockData.objects.create(
                symbol=symbol,
                date=date,
                open_price=open_price,
                close_price=close_price,
                high_price=high_price,
                low_price=low_price,
                volume=volume
            )

    def get_stock_data_df(self, symbol):
        """ 
        Utility function to retrieve stock data for a given symbol and convert it to a pandas DataFrame.
        This function facilitates the use of stock data within test cases.
        """
        stock_data = StockData.objects.filter(symbol=symbol)
        if not stock_data.exists():
            return pd.DataFrame()  # Return empty DataFrame if no data is found

        data_dicts = list(stock_data.values('date', 'close_price'))
        df = pd.DataFrame(data_dicts)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df

    def test_backtest_strategy_success(self):
        """ 
        Test a successful scenario for backtesting, ensuring that valid input parameters produce expected results.
        """
        df = self.get_stock_data_df(self.SYMBOL)
        result = backtest_strategy(initial_investment=self.INITIAL_INVESTMENT, symbol=self.SYMBOL, buy_window=2, sell_window=3, df=df)
        self.assertIn('final_balance', result)
        self.assertGreaterEqual(result['final_balance'], 0, "Final balance should be non-negative")
        self.assertGreaterEqual(result['number_of_trades'], 0, "Number of trades should be non-negative")
        self.assertGreaterEqual(result['total_ROI_percentage'], -100, "Total return should not be unrealistic")

    def test_backtest_strategy_invalid_investment(self):
        """ 
        Test the behavior when an invalid initial investment (negative value) is provided.
        This should result in an appropriate error response.
        """
        df = self.get_stock_data_df(self.SYMBOL)
        result = backtest_strategy(initial_investment=-1000, symbol=self.SYMBOL, buy_window=2, sell_window=3, df=df)
        self.assertEqual(result.get('status'), 400, "Invalid investment should return status 400")
        self.assertIn('error', result, "Response should contain an error message for negative investment")

    def test_backtest_strategy_invalid_window(self):
        """ 
        Test when the buy window is greater than or equal to the sell window.
        This should result in an appropriate error response.
        """
        df = self.get_stock_data_df(self.SYMBOL)
        result = backtest_strategy(initial_investment=self.INITIAL_INVESTMENT, symbol=self.SYMBOL, buy_window=3, sell_window=2, df=df)
        self.assertEqual(result.get('status'), 400, "Buy window greater than or equal to sell window should return status 400")
        self.assertIn('error', result, "Response should contain an error message for invalid window sizes")

    def test_backtest_strategy_no_data(self):
        """ 
        Test the behavior when attempting to backtest on a symbol that doesn't exist in the database.
        The response should indicate a data availability issue.
        """
        df = self.get_stock_data_df('NON_EXISTENT')
        result = backtest_strategy(initial_investment=self.INITIAL_INVESTMENT, symbol='NON_EXISTENT', buy_window=2, sell_window=3, df=df)
        self.assertEqual(result.get('status'), 400, "Backtest on non-existent symbol should return status 400")
        self.assertIn('error', result, "Response should contain an error message for missing data")

    def test_backtest_strategy_edge_case_single_data_point(self):
        """ 
        Test the backtest logic with only a single data point available.
        This should result in an error due to insufficient data.
        """
        StockData.objects.filter(symbol=self.SYMBOL).delete()
        self.create_stock_data(self.SYMBOL, [(datetime(2022, 1, 1), 100, 105, 110, 95, 1000)])
        
        df = self.get_stock_data_df(self.SYMBOL)
        result = backtest_strategy(initial_investment=self.INITIAL_INVESTMENT, symbol=self.SYMBOL, buy_window=2, sell_window=3, df=df)
        self.assertEqual(result.get('status'), 400, "Not enough data should return status 400")
        self.assertIn('error', result, "Response should contain an error message for insufficient data")

    def test_backtest_strategy_insufficient_data_for_ma(self):
        """ 
        Test the scenario where there isn't enough data to calculate the moving averages required by the strategy.
        """
        StockData.objects.filter(symbol=self.SYMBOL).delete()
        self.create_stock_data(self.SYMBOL, [
            (datetime(2022, 1, 1), 100, 105, 110, 95, 1000),
            (datetime(2022, 1, 2), 105, 103, 108, 102, 1200)
        ])
        
        df = self.get_stock_data_df(self.SYMBOL)
        result = backtest_strategy(initial_investment=self.INITIAL_INVESTMENT, symbol=self.SYMBOL, buy_window=5, sell_window=10, df=df)
        self.assertEqual(result.get('status'), 400, "Not enough data to calculate moving averages should return status 400")
        self.assertIn('error', result, "Response should contain an error message for insufficient data for moving averages")

    def test_backtest_strategy_large_investment(self):
        """ 
        Test the backtest with a very large initial investment to ensure no overflow issues or unexpected failures.
        """
        df = self.get_stock_data_df(self.SYMBOL)
        result = backtest_strategy(initial_investment=1000000000, symbol=self.SYMBOL, buy_window=2, sell_window=3, df=df)
        self.assertIn('final_balance', result, "Large investment should still provide a final balance")
        self.assertGreaterEqual(result['final_balance'], 0, "Final balance should be non-negative even for large investments")

    def test_backtest_strategy_moving_average_edge_case(self):
        """ 
        Test the scenario where the buy and sell windows are set very high to assess the strategy's handling of such parameters.
        """
        df = self.get_stock_data_df(self.SYMBOL)
        result = backtest_strategy(initial_investment=self.INITIAL_INVESTMENT, symbol=self.SYMBOL, buy_window=100, sell_window=200, df=df)
        self.assertIn('status', result)
        if result['status'] == 400:
            self.assertIn('error', result, "Should return an error if not enough data is available for the given window sizes")
        else:
            self.assertIn('final_balance', result, "Should calculate final balance if enough data exists")

    def test_backtest_strategy_zero_investment(self):
        """ 
        Test backtest with zero initial investment to verify correct error handling.
        """
        df = self.get_stock_data_df(self.SYMBOL)
        result = backtest_strategy(initial_investment=0, symbol=self.SYMBOL, buy_window=2, sell_window=3, df=df)
        self.assertEqual(result.get('status'), 400, "Zero initial investment should return status 400")
        self.assertIn('error', result, "Response should contain an error message for zero investment")
