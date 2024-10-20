# financial_project/celery.py

from __future__ import absolute_import, unicode_literals
import os
from celery import Celery
from django.conf import settings
from celery.schedules import crontab
from decouple import config

# Set the default settings module for Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'financial_project.settings')

# Create the Celery application instance
app = Celery('financial_project')

# Load Celery configuration from Django's settings, using the 'CELERY_' namespace for all related settings
app.config_from_object('django.conf:settings', namespace='CELERY')

# Automatically discover tasks defined in all installed Django apps
app.autodiscover_tasks(lambda: settings.INSTALLED_APPS)

@app.task(bind=True)
def debug_task(self):
    """
    A debug task that prints the current request, useful for testing and debugging.
    """
    print(f'Request: {self.request!r}')

# Configure Celery Beat to run periodic tasks
app.conf.beat_schedule = {
    # Task to fetch stock data every 6 hours
    'fetch-stock-data-every-six-hours': {
        'task': 'stocks.tasks.fetch_stock_data_task',
        'schedule': crontab(minute=0, hour='*/6'),  # Scheduled to run every 6 hours at minute 0
    },
}



