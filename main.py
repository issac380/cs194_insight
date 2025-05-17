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

# Only generate data if not already done
DATA_CSV = "retail_moststolen_data.csv"
if not os.path.exists(DATA_CSV):
    df = make_sim_data()
    df.to_csv(DATA_CSV, index=False)
else:
    df = pd.read_csv(DATA_CSV, parse_dates=['date'])

# --- 2. LLM agent function ---
def generate_insight(store, category, item, theft_focus):
    sub = df[(df['store'] == store) & (df['category'] == category)]
    if item != "All":
        sub = sub[sub['item'] == item]
    thefts = sub['theft_qty'].sum()
    total_sales = sub['sales'].sum()
    prompt = (
        f"Retail data for {store}, category {category}"
        + (f", item: {item}" if item != "All" else "")
        + f".\nTotal sales: {total_sales}\nTotal theft quantity: {thefts}\n"
        + ("Focus especially on theft and loss patterns." if theft_focus else "")
        + "\nGive 2-3 actionable insights for a store manager to improve performance and reduce theft, knowing this is a commonly lost or stolen item."
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a retail analytics assistant who specializes in smart, actionable insights."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.6,
    )
    insight = response.choices[0].message.content.strip()
    return insight

# --- 3. Dashboard plot function ---
def plot_trends(store, category, item):
    sub = df[(df['store'] == store) & (df['category'] == category)]
    if item != "All":
        sub = sub[sub['item'] == item]
    daily = sub.groupby('date').agg({'sales': 'sum', 'theft_qty': 'sum', 'stock': 'sum'}).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(daily['date'], daily['sales'], label="Sales", marker="o", linewidth=1)
    ax.plot(daily['date'], daily['theft_qty'], label="Theft", marker="x", linestyle="--")
    ax.set_ylabel("Count")
    ax.set_xlabel("Date")
    ax.set_title(f"Daily Sales & Theft - {store} / {category}" + (f" / {item}" if item != "All" else ""))
    ax.legend()
    fig.tight_layout()
    return fig

# --- 4. Gradio front end ---
def dashboard(store, category, item, theft_focus):
    insights = generate_insight(store, category, item, theft_focus)
    fig = plot_trends(store, category, item)
    return insights, fig

with gr.Blocks(title="Retail LLM Agentic Dashboard") as demo:
    gr.Markdown("# 🛒 Most Commonly Stolen Retail Items — Agentic Insights Dashboard")
    gr.Markdown("**Select store, category, and item to see trends and actionable LLM-powered insights focused on loss prevention.**")
    with gr.Row():
        store = gr.Dropdown(list(df['store'].unique()), label="Store", value="Store A")
        category = gr.Dropdown(list(df['category'].unique()), label="Category", value="Electronics")
        def item_choices(category):
            return ["All"] + list(df[df['category']==category]['item'].unique())
        item = gr.Dropdown(choices=item_choices("Electronics"), label="Item", value="All")
        def update_items(c): return gr.Dropdown.update(choices=item_choices(c), value="All")
        category.change(fn=update_items, inputs=category, outputs=item)
        theft_focus = gr.Checkbox(label="Focus insights on theft/loss?", value=True)
    with gr.Row():
        insights = gr.Textbox(label="LLM Smart Insights", lines=5)
    chart = gr.Plot(label="Trends Dashboard")
    btn = gr.Button("Generate Insights & Dashboard")
    btn.click(
        dashboard, 
        inputs=[store, category, item, theft_focus], 
        outputs=[insights, chart]
    )

    # Initial auto-run
    demo.load(
        dashboard,
        inputs=[store, category, item, theft_focus],
        outputs=[insights, chart]
    )

if __name__ == '__main__':
    demo.launch()