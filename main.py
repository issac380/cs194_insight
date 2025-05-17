import os
import pandas as pd
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv

# --- Setup OpenAI and env ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- 1. Simulate detailed electronics data with theft events ---
def make_electronics_data():
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2024-03-31', freq='D')
    stores = ['Store A', 'Store B', 'Store C']
    electronics = [
        {'item': 'Apple iPhone 15 Pro Max', 'color': 'Black', 'spec': '256GB'},
        {'item': 'Apple iPhone 15 Pro Max', 'color': 'Silver', 'spec': '512GB'},
        {'item': 'Samsung Galaxy S24 Ultra', 'color': 'Phantom Black', 'spec': '256GB'},
        {'item': 'Samsung Galaxy S24 Ultra', 'color': 'Titanium Gray', 'spec': '512GB'},
        {'item': 'Sony WH-1000XM5 Headphones', 'color': 'Silver', 'spec': 'Noise Cancelling'},
        {'item': 'Sony WH-1000XM5 Headphones', 'color': 'Black', 'spec': 'Noise Cancelling'},
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
            stock = 100
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

# Only generate data if not already done
DATA_CSV = "electronics_detailed_data.csv"
if not os.path.exists(DATA_CSV):
    df = make_electronics_data()
    df.to_csv(DATA_CSV, index=False)
else:
    df = pd.read_csv(DATA_CSV, parse_dates=['date'])

# --- 2. LLM agent function ---
def generate_insight(store, item, color, spec, theft_focus):
    sub = df[(df['store'] == store)]
    if item != "All":
        sub = sub[sub['item'] == item]
    if color != "All":
        sub = sub[sub['color'] == color]
    if spec != "All":
        sub = sub[sub['spec'] == spec]
    thefts = sub['theft_qty'].sum()
    total_sales = sub['sales'].sum()
    selection_desc = f"Store: {store}, Item: {item}, Color: {color}, Spec: {spec}"
    prompt = (
        f"Retail data for electronics. {selection_desc}.\n"
        f"Total sales: {total_sales}\n"
        f"Total theft quantity: {thefts}\n"
        f"{'Focus especially on theft and loss patterns.' if theft_focus else ''}\n"
        f"Give 2-3 actionable insights for a store manager to improve performance and reduce theft. Format the response using Markdown."
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a retail analytics assistant who specializes in smart, actionable insights. Format your response using Markdown."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.6,
    )
    insight = response.choices[0].message.content.strip()
    return insight

# --- 3. Dashboard plot function ---
def plot_trends(store, item, color, spec):
    sub = df[(df['store'] == store)]
    if item != "All":
        sub = sub[sub['item'] == item]
    if color != "All":
        sub = sub[sub['color'] == color]
    if spec != "All":
        sub = sub[sub['spec'] == spec]
    daily = sub.groupby('date').agg({'sales': 'sum', 'theft_qty': 'sum', 'stock': 'sum'}).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(daily['date'], daily['sales'], label="Sales", marker="o", linewidth=1)
    ax.plot(daily['date'], daily['theft_qty'], label="Theft", marker="x", linestyle="--")
    ax.set_ylabel("Count")
    ax.set_xlabel("Date")
    ax.set_title(f"Sales & Theft Trends - {store}" + (f" / {item}" if item != "All" else ""))
    ax.legend()
    fig.tight_layout()
    return fig

# --- 4. Gradio front end ---
def dashboard(store, item, color, spec, theft_focus):
    insights = generate_insight(store, item, color, spec, theft_focus)
    fig = plot_trends(store, item, color, spec)
    return insights, fig

with gr.Blocks(title="Electronics Retail LLM Dashboard") as demo:
    gr.Markdown("# 📱 Detailed Electronics Retail Insights Dashboard")
    gr.Markdown("**Filter by store, model, color, and specs to see sales & theft trends and LLM insights.**")
    with gr.Row():
        store = gr.Dropdown(list(df['store'].unique()), label="Store", value="Store A")
        def item_choices(): return ["All"] + list(df['item'].unique())
        item = gr.Dropdown(choices=item_choices(), label="Model", value="All")
        def color_choices(): return ["All"] + list(df['color'].unique())
        color = gr.Dropdown(choices=color_choices(), label="Color", value="All")
        def spec_choices(): return ["All"] + list(df['spec'].unique())
        spec = gr.Dropdown(choices=spec_choices(), label="Spec", value="All")
        theft_focus = gr.Checkbox(label="Focus insights on theft/loss?", value=True)
    with gr.Row():
        insights = gr.Markdown(label="LLM Smart Insights")
    chart = gr.Plot(label="Trends Dashboard")
    btn = gr.Button("Generate Insights & Dashboard")
    btn.click(
        dashboard, 
        inputs=[store, item, color, spec, theft_focus], 
        outputs=[insights, chart]
    )

    # Initial auto-run
    demo.load(
        dashboard,
        inputs=[store, item, color, spec, theft_focus],
        outputs=[insights, chart]
    )

if __name__ == '__main__':
    demo.launch()