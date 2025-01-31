from django.contrib import admin
from .models import StockLog

# Register the Products model with the admin site

class StockLogAdmin(admin.ModelAdmin):
    list_display = ('id', 'open_price', 'high_price', 'low_price', 'volume', 'result', 'percentage_change', 'difference')


admin.site.register(StockLog, StockLogAdmin)