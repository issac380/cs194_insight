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
        {'item': 'Sony WH-1000XM5', 'colors': ['Black', 'Cream', 'Blue'], 'specs': ['Noise Cancelling']},
        {'item': 'Bose QC45', 'colors': ['Black', 'Silver', 'White'], 'specs': ['Noise Cancelling']},
        {'item': 'Beats Solo 4', 'colors': ['Black', 'Red'], 'specs': ['Standard']},
    ]
    n_customers = 50
    customer_ids = [f"C{str(i).zfill(3)}" for i in range(n_customers)]
    records = []
    session_id = 0

    for date in dates:
        for store in stores:
            for _ in range(np.random.randint(12, 20)):
                customer = np.random.choice(customer_ids)
                session_id += 1
                prob = [0.25, 0.35, 0.4]  # Sony, Bose, Beats
                item_idx = np.random.choice(len(items), p=prob)
                prod = items[item_idx]
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
                if prod['item'] == 'Sony WH-1000XM5':
                    if np.random.rand() < 0.85:
                        records.append({**records[-1], 'event': 'purchase'})
                    else:
                        records.append({**records[-1], 'event': 'putback'})
                elif prod['item'] == 'Beats Solo 4':
                    red_penalty = 0.05 if color == 'Red' else 0.15
                    if np.random.rand() > red_penalty:
                        records.append({**records[-1], 'event': 'putback'})
                    else:
                        records.append({**records[-1], 'event': 'purchase'})
                else:
                    if np.random.rand() < 0.5:
                        records.append({**records[-1], 'event': 'purchase'})
                    else:
                        records.append({**records[-1], 'event': 'putback'})
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
    if ('grab' in q or 'putback' in q or 'purchase' in q or 'conversion' in q or 'frequency' in q or 'rate' in q or 'interact' in q):
        return analyze_grab_putback(item=item_match, store=store_match)
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

# ---------- 4. Chain-of-Verification Full Pipeline ----------
def generate_cov_chain():
    # Step 1: Draft
    draft = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a retail assistant generating business insights from RFID data."},
            {"role": "user", "content": "Draft an initial business insight based on headphone grab/putback/purchase data."}
        ]
    ).choices[0].message.content.strip()

    # Step 2: Plan Verification Questions
    questions = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI analyst that generates verification questions."},
            {"role": "user", "content": f"Given this draft insight: \"{draft}\"\n\nWrite 2-3 verification questions to fact-check the claim."}
        ]
    ).choices[0].message.content.strip()

    # Step 3: Compute answers from the data (hardcoded logic)
    agg = df.groupby(['item', 'color'])['event'].value_counts().unstack(fill_value=0).reset_index()
    agg['putback_rate'] = agg['putback'] / agg['grab'].replace(0, np.nan)
    agg['conversion_rate'] = agg['purchase'] / agg['grab'].replace(0, np.nan)
    agg = agg.fillna(0).sort_values('putback_rate', ascending=False).head(10)
    answer_summary = agg.to_csv(index=False)

    # Step 4: Final verified answer
    final = client.chat.completions.create(
    model="gpt-4o",
    messages=[
            {"role": "system", "content": "You are a strategic retail advisor."},
            {"role": "user", "content": (
                f"Here was your draft:\n{draft}\n\n"
                f"Here are your verification questions:\n{questions}\n\n"
                f"Here is a verified summary of grab/putback data:\n{answer_summary}\n\n"
                "Now do the following:\n"
                "1. List the top 3 items with the highest putback rate.\n"
                "2. Recommend 1–2 items that should be discounted based on this evidence.\n"
                "3. Justify the recommendation in plain English for a business operator."
            )}
        ],
        temperature=0.4,
    ).choices[0].message.content.strip()


    # Return both outputs
    full_text = f"### Draft Insight:\n{draft}\n\n### Verification Questions:\n{questions}\n\n### Verified Data Summary:\n```csv\n{answer_summary}\n```\n\n### Final Recommendation:\n{final}"
    return final, full_text

# ---------- 5. Gradio UI ----------
with gr.Blocks(title="RFID Shelf Dashboard + Chat") as demo:
    gr.Markdown("# \U0001F6D2 RFID Shelf Dashboard + Agentic Chat")
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

    with gr.Tab("Chain-of-Verification"):
        gr.Markdown("## \U0001F9EA Chain-of-Verification Report")
        cov_btn = gr.Button("Generate Full CoVe Report")
        short_output = gr.Markdown()
        full_output = gr.Markdown(visible=False)
        toggle = gr.Checkbox(label="See full Chain of Verification output")

        cov_btn.click(generate_cov_chain, outputs=[short_output, full_output])
        toggle.change(lambda visible: gr.update(visible=visible), inputs=toggle, outputs=[full_output])

    with gr.Tab("Chat"):
        chat = gr.ChatInterface(
            fn=lambda msg, history: run_agentic_analysis(msg),
            additional_outputs=[gr.Image(type="pil", label="Chart/Plot (if relevant)")],
            examples=[
                "Show interaction rates for Sony WH-1000XM5 at Store A.",
                "What is the putback rate for Beats Solo 4?",
                "How often is the red Beats Solo 4 put back without purchase?",
                "Which color of Sony WH-1000XM5 is most purchased?",
                "Compare purchases vs putbacks for Bose QC45.",
                "What’s the conversion rate for noise cancelling headphones at Store B?",
                "Do customers prefer Sony or Beats?",
                "Which headphone model has the highest session conversion rate?",
                "Show product interaction stats for all headphones at Store C."
            ]
        )

if __name__ == "__main__":
    demo.launch()
