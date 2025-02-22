from django.db import models
from django.utils.timezone import now  # Import now()


class StockLog(models.Model):
    company = models.CharField(max_length=50, default="DefaultCompany")  # New field for company name
    open_price = models.DecimalField(max_digits=15, decimal_places=2)
    high_price = models.DecimalField(max_digits=15, decimal_places=2)
    low_price = models.DecimalField(max_digits=15, decimal_places=2)
    volume = models.DecimalField(max_digits=20, decimal_places=0)  # Integer-like for volume
    result = models.DecimalField(max_digits=15, decimal_places=2)
    
    # Fields for percentage change and difference
    percentage_change = models.DecimalField(max_digits=7, decimal_places=2, null=True, blank=True)
    difference = models.DecimalField(max_digits=15, decimal_places=2, null=True, blank=True)
    timestamp = models.DateTimeField(default=now)

