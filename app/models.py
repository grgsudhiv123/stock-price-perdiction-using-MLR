from django.db import models

class StockLog(models.Model):
    open_price = models.FloatField()
    high_price = models.FloatField()
    low_price = models.FloatField()
    volume = models.FloatField()
    result = models.DecimalField(max_digits=50, decimal_places=2)

