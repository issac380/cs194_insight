# 🎧 CS194 Spring 2025 Project – RFID Shelf Retail Analytics

A fully interactive dashboard and conversational agent for analyzing simulated retail RFID shelf data—including customer “grab”, “putback”, and “purchase” behaviors—across headphones in multiple stores.

---

## 🚀 Features

- **Dashboard Tab:**  
  Select any store and product to view grabs, putbacks, purchases, and conversion rates, with instant visual charts.
- **Sales Report Tab:**  
  One-click sales summary and smart business recommendations using agnetic AI pipeline, including which products to discount or promote.
- **Chat Tab:**  
  Ask natural-language questions about product behavior, conversion, color preferences, or sales trends. The AI agent generates real insights and plots as appropriate.

---

## 🏷️ Example Use Cases

- Discover which product-color combos are most/least likely to convert.
- Find which items are most often put back (potential targets for discounts).
- See sales patterns over time or by location.
- Get actionable AI recommendations to optimize inventory or promotions.

---

## 🛠️ Installation

```bash
# 1. Clone this repo (or download all files)
git clone <repo-url>

# 2. Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

---
## 🖥️ App Overview
- Dashboard Tab:
Analyze a single product/store. Click Generate Analysis for detailed interaction and conversion stats, plus a chart.
- Sales Report Tab:
Click Generate Sales Report to see the overall dataset analyzed by GPT-4o, with markdown recommendations for which products to discount.
- Chat Tab:
Ask open-ended or specific questions about sales trends, conversion rates, putbacks, product comparisons, and more.

---

## 📈 Customization
- Product/Store List:
Modify the items or stores list in make_rfid_shelf_data() to simulate other products or categories.
- Agent/LLM Logic:
Adjust the prompts or output formatting to tune the business advice or AI explanations.
