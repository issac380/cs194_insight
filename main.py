import os
import pandas as pd
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
import io
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image

# ---------- Setup OpenAI ----------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- 1. Simulate RFID Retail Data ----------
def make_rfid_shelf_data():
    np.random.seed(102)
    dates = pd.date_range('2024-01-01', '2024-01-15', freq='D')
    stores = ['Store A', 'Store B', 'Store C']
    items = [
        {'item': 'Apple iPhone 15', 'colors': ['Black', 'Silver'], 'specs': ['128GB', '256GB']},
        {'item': 'Samsung S24 Ultra', 'colors': ['Black', 'Gray'], 'specs': ['256GB', '512GB']},
        {'item': 'Sony Headphones', 'colors': ['Black'], 'specs': ['Noise Cancelling']},
        {'item': 'Bose Speaker', 'colors': ['White', 'Blue'], 'specs': ['Wireless']},
    ]
    n_customers = 60
    customer_ids = [f"C{str(i).zfill(3)}" for i in range(n_customers)]
    records = []
    session_id = 0
    for date in dates:
        for store in stores:
            for _ in range(np.random.randint(10, 19)):
                customer = np.random.choice(customer_ids)
                session_id += 1
                n_grabbed = np.random.choice([1, 2, 3], p=[0.65, 0.25, 0.1])
                chosen_items = np.random.choice(len(items), size=n_grabbed, replace=False)
                purchased_idx = np.random.choice(chosen_items) if np.random.rand() < 0.75 else None
                for idx in chosen_items:
                    prod = items[idx]
                    color = np.random.choice(prod['colors'])
                    spec = np.random.choice(prod['specs'])
                    records.append({
                        'session_id': session_id,
                        'date': date,
                        'store': store,
                        'customer_id': customer,
                        'item': prod['item'],
                        'color': color,
                        'spec': spec,
                        'event': 'grab'
                    })
                    if purchased_idx != idx:
                        if np.random.rand() < 0.92:
                            records.append({
                                'session_id': session_id,
                                'date': date,
                                'store': store,
                                'customer_id': customer,
                                'item': prod['item'],
                                'color': color,
                                'spec': spec,
                                'event': 'putback'
                            })
                    if purchased_idx == idx:
                        records.append({
                            'session_id': session_id,
                            'date': date,
                            'store': store,
                            'customer_id': customer,
                            'item': prod['item'],
                            'color': color,
                            'spec': spec,
                            'event': 'purchase'
                        })
    return pd.DataFrame(records)

# ---------- 2. Load or Generate Data ----------
DATA_CSV = "rfid_shelf_data.csv"
if not os.path.exists(DATA_CSV):
    df = make_rfid_shelf_data()
    df.to_csv(DATA_CSV, index=False)
else:
    df = pd.read_csv(DATA_CSV, parse_dates=['date'])

# ---------- 3. Backend Analysis Functions ----------
def plot_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return Image.open(buf)

def get_product_list():
    return sorted(df['item'].unique())

def get_store_list():
    return sorted(df['store'].unique())

def analyze_grab_putback(item=None, store=None):
    sub = df.copy()
    if item and item != "All":
        sub = sub[sub['item'] == item]
    if store and store != "All":
        sub = sub[sub['store'] == store]
    if len(sub) == 0:
        return "No product data for your selection.", None
    grabs = sub[sub['event'] == 'grab']
    putbacks = sub[sub['event'] == 'putback']
    purchases = sub[sub['event'] == 'purchase']
    sessions_with_grab = grabs['session_id'].nunique()
    sessions_with_purchase = purchases['session_id'].nunique()
    conversion_rate = sessions_with_purchase / sessions_with_grab if sessions_with_grab else 0
    insight = (
        f"**Product:** {item or 'All'}  \n"
        f"**Store:** {store or 'All'}  \n"
        f"Total Grabs: {len(grabs)}  \n"
        f"Total Putbacks: {len(putbacks)}  \n"
        f"Total Purchases: {len(purchases)}  \n"
        f"Session Conversion Rate: {conversion_rate:.0%}  \n"
    )
    fig, ax = plt.subplots(figsize=(5,3))
    ax.bar(['Grabs', 'Putbacks', 'Purchases'], [len(grabs), len(putbacks), len(purchases)], color=["#9cf","#fa8","#7a8"])
    ax.set_ylabel("Count")
    ax.set_title("Product Interactions")
    fig.tight_layout()
    return insight, plot_to_pil(fig)

# --- Chat LLM/Agent ---
def run_agentic_analysis(user_query):
    q = user_query.lower()
    item_match = None
    store_match = None
    for prod in get_product_list():
        if prod.lower() in q:
            item_match = prod
            break
    for st in get_store_list():
        if st.lower() in q:
            store_match = st
            break
    # Keywords trigger dashboard analysis with plot
    if ('grab' in q or 'putback' in q or 'purchase' in q or 'conversion' in q or 'frequency' in q or 'rate' in q or 'interact' in q):
        return analyze_grab_putback(item=item_match, store=store_match)
    # Fallback: LLM text only
    llm_prompt = (
        "You are an expert retail analytics assistant. "
        "The user has queried: " + user_query +
        "\nIf you can answer with the available data, please do, otherwise explain what data would be needed."
    )
    llm_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"system", "content":"You are a retail analytics expert."},
                  {"role":"user", "content": llm_prompt}],
        temperature=0.4,
    ).choices[0].message.content.strip()
    return llm_response, None

# ---------- 4. Gradio UI (Chat + Dashboard) ----------
with gr.Blocks(title="RFID Shelf Dashboard + Chat") as demo:
    gr.Markdown("# 🛒 RFID Shelf Dashboard + Agentic Chat")
    gr.Markdown("Interact by **dropdown** or **chat**. See product grabs/putbacks/purchases and conversion rates for any product and store.")
    
    with gr.Tab("Dashboard"):
        with gr.Row():
            store_dd = gr.Dropdown(["All"] + get_store_list(), value="All", label="Store")
            item_dd = gr.Dropdown(["All"] + get_product_list(), value="All", label="Product")
        with gr.Row():
            btn = gr.Button("Generate Analysis")
        insight = gr.Markdown(label="Insight")
        plot = gr.Image(type="pil", label="Plot")
        btn.click(analyze_grab_putback, inputs=[item_dd, store_dd], outputs=[insight, plot])
        # Optional: auto-run on page load (uncomment if desired)
        # demo.load(analyze_grab_putback, inputs=[item_dd, store_dd], outputs=[insight, plot])

    with gr.Tab("Chat"):
        chat = gr.ChatInterface(
            fn=lambda msg, history: run_agentic_analysis(msg),
            additional_outputs=[gr.Image(type="pil", label="Chart/Plot (if relevant)")],
            examples=[
                "Show interaction rates for Apple iPhone 15 at Store B.",
                "Which products are most frequently grabbed but not purchased?",
                "Show conversion rates for Sony Headphones.",
                "What is the putback rate for Bose Speaker at Store C?",
            ]
        )

if __name__ == "__main__":
    demo.launch()