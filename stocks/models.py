# stocks/models.py

from django.db import models

class StockData(models.Model):
    # Defines the stock symbol (e.g., AAPL) with a maximum length of 10 characters and a database index for fast lookup.
    symbol = models.CharField(max_length=10, db_index=True)  
    
    # The date when the stock data was recorded; also indexed to speed up queries involving date ranges.
    date = models.DateField(db_index=True)  
    
    # Price fields representing daily open, close, high, and low values of the stock.
    open_price = models.FloatField()
    close_price = models.FloatField()
    high_price = models.FloatField()
    low_price = models.FloatField()
    
    # Total volume of stocks traded on the given date.
    volume = models.BigIntegerField()
    
    # Optional field to store predicted closing price, which may or may not be available.
    prediction = models.FloatField(null=True, blank=True)   

    # String representation for debugging and admin purposes.
    def __str__(self):
        return f"{self.symbol} - {self.date}"

    class Meta:
        # Ensures the combination of symbol and date is unique in the database, preventing duplicate entries.
        unique_together = ('symbol', 'date')





