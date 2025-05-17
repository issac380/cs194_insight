import numpy as np
import pandas as pd

# --- 1. Simulate retail data with most-stolen goods and theft events ---
def make_sim_data():
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2024-03-31', freq='D')
    stores = ['Store A', 'Store B', 'Store C']

    categories = [
        'Electronics', 'Health & Beauty', 'Alcoholic Beverages', 
        'Food', 'Household', 'Clothing & Accessories', 'Miscellaneous'
    ]

    items = {
        'Electronics': [
            'Smartphone', 'Tablet', 'Laptop', 'Gaming Console', 'Headphones', 
            'Smartwatch', 'USB Flash Drive', 'Phone Accessory'
        ],
        'Health & Beauty': [
            'Cosmetics', 'Skincare', 'Hair Care', 'Deodorant', 
            'Toothpaste', 'Vitamins', 'OTC Medication'
        ],
        'Alcoholic Beverages': [
            'Whisky', 'Champagne', 'Gin', 'Prosecco'
        ],
        'Food': [
            'Steak', 'Cheese', 'Frozen Seafood', 'Chocolate', 'Bacon'
        ],
        'Household': [
            'Razor Blades', 'Batteries', 'Paper Towels', 'Diapers', 'Detergent'
        ],
        'Clothing & Accessories': [
            'Designer Bag', 'Watch', 'Sunglasses', 'Scarf', 'Gloves'
        ]
    }

    # Assign higher theft probability to items known to be commonly stolen
    high_theft_categories = ['Electronics', 'Health & Beauty', 'Alcoholic Beverages', 'Clothing & Accessories']
    records = []
    for store in stores:
        for category in categories:
            for item in items[category]:
                stock = 500 if category in high_theft_categories else 800
                for date in dates:
                    sales = np.random.poisson(3 if category in high_theft_categories else 5)
                    theft_prob = 0.07 if category in high_theft_categories else 0.02
                    theft = np.random.binomial(1, theft_prob)
                    lost_qty = np.random.randint(1, 3) if theft else 0
                    stock = max(0, stock - sales - lost_qty)
                    records.append({
                        'date': date,
                        'store': store,
                        'category': category,
                        'item': item,
                        'sales': sales,
                        'theft_event': bool(theft),
                        'theft_qty': lost_qty,
                        'stock': stock
                    })
    return pd.DataFrame(records)