# Generated by Django 5.1.2 on 2024-10-18 15:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('stocks', '0002_alter_stockdata_date_alter_stockdata_symbol_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='stockdata',
            name='prediction',
            field=models.FloatField(blank=True, null=True),
        ),
    ]
