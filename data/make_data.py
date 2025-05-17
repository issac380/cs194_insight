import numpy as np
import pandas as pd

def make_sim_data():
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2024-03-31', freq='D')
    stores = ['Store A', 'Store B', 'Store C']

    # Electronics models (brand, model, color, specs)
    electronics = [
        {'item': 'Apple iPhone 15', 'color': 'Blue', 'spec': '128GB'},
        {'item': 'Apple iPhone 15', 'color': 'Blue', 'spec': '256GB'},
        {'item': 'Apple iPhone 15', 'color': 'Blue', 'spec': '512GB'},
        {'item': 'Apple iPhone 15', 'color': 'Yellow', 'spec': '128GB'},
        {'item': 'Apple iPhone 15', 'color': 'Yellow', 'spec': '256GB'},
        {'item': 'Apple iPhone 15', 'color': 'Yellow', 'spec': '512GB'},
        {'item': 'Apple iPhone 15', 'color': 'Black', 'spec': '128GB'},
        {'item': 'Apple iPhone 15', 'color': 'Black', 'spec': '256GB'},
        {'item': 'Apple iPhone 15', 'color': 'Black', 'spec': '512GB'},
        {'item': 'Apple iPhone 15 Pro Max', 'color': 'Black', 'spec': '128GB'},
        {'item': 'Apple iPhone 15 Pro Max', 'color': 'Black', 'spec': '256GB'},
        {'item': 'Apple iPhone 15 Pro Max', 'color': 'Black', 'spec': '512GB'},
        {'item': 'Apple iPhone 15 Pro Max', 'color': 'Silver', 'spec': '128GB'},
        {'item': 'Apple iPhone 15 Pro Max', 'color': 'Silver', 'spec': '256GB'},
        {'item': 'Apple iPhone 15 Pro Max', 'color': 'Silver', 'spec': '512GB'},
        {'item': 'Samsung Galaxy S24 Ultra', 'color': 'Phantom Black', 'spec': '256GB'},
        {'item': 'Samsung Galaxy S24 Ultra', 'color': 'Phantom Black', 'spec': '512GB'},
        {'item': 'Samsung Galaxy S24 Ultra', 'color': 'Phantom Black', 'spec': '1TB'},
        {'item': 'Samsung Galaxy S24 Ultra', 'color': 'Titanium Gray', 'spec': '256GB'},
        {'item': 'Samsung Galaxy S24 Ultra', 'color': 'Titanium Gray', 'spec': '512GB'},
        {'item': 'Samsung Galaxy S24 Ultra', 'color': 'Titanium Gray', 'spec': '1TB'},
        {'item': 'Samsung Galaxy S24 Ultra', 'color': 'Orange', 'spec': '256GB'},
        {'item': 'Samsung Galaxy S24 Ultra', 'color': 'Orange', 'spec': '512GB'},
        {'item': 'Samsung Galaxy S24 Ultra', 'color': 'Orange', 'spec': '1TB'},
        {'item': 'Sony WH-1000XM5 Headphones', 'color': 'Silver', 'spec': 'Noise Cancelling'},
        {'item': 'Sony WH-1000XM5 Headphones', 'color': 'Black', 'spec': 'Noise Cancelling'},
        {'item': 'Sony WH-1000XM5 Headphones', 'color': 'Blue', 'spec': 'Noise Cancelling'},
        {'item': 'Apple MacBook Pro 16"', 'color': 'Space Gray', 'spec': 'M3 Pro, 16GB RAM, 1TB SSD'},
        {'item': 'Apple MacBook Air 13"', 'color': 'Midnight', 'spec': 'M2, 8GB RAM, 256GB SSD'},
        {'item': 'Nintendo Switch OLED', 'color': 'White', 'spec': '64GB'},
        {'item': 'Bose QuietComfort Ultra', 'color': 'White Smoke', 'spec': 'Wireless'},
        {'item': 'Microsoft Surface Pro 9', 'color': 'Platinum', 'spec': '16GB RAM, 512GB SSD'},
        {'item': 'Google Pixel 8 Pro', 'color': 'Bay', 'spec': '128GB'},
        {'item': 'JBL Charge 5 Speaker', 'color': 'Blue', 'spec': 'Waterproof'},
    ]

    records = []
    for store in stores:
        for e in electronics:
            stock = 100  # Lower for high value/demand items
            for date in dates:
                sales = np.random.poisson(1.8)
                theft_prob = 0.08 if "iPhone" in e['item'] or "MacBook" in e['item'] else 0.04
                theft = np.random.binomial(1, theft_prob)
                lost_qty = np.random.randint(1, 2) if theft else 0
                stock = max(0, stock - sales - lost_qty)
                records.append({
                    'date': date,
                    'store': store,
                    'category': 'Electronics',
                    'item': e['item'],
                    'color': e['color'],
                    'spec': e['spec'],
                    'sales': sales,
                    'theft_event': bool(theft),
                    'theft_qty': lost_qty,
                    'stock': stock
                })
    return pd.DataFrame(records)