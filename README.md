📊 Marketing Intelligence Dashboard
📌 Overview

This project is an interactive BI dashboard built with Streamlit to analyze how marketing activities across different channels (Facebook, Google, TikTok) influence business outcomes (orders, revenue, profit, etc.).

The dashboard connects campaign-level marketing data with daily business performance data, helping decision-makers understand ROI, efficiency, and performance trends in one place.

📂 Datasets

You are provided with 120 days of daily activity:

Facebook.csv, Google.csv, TikTok.csv → Campaign-level marketing data

Columns: date, tactic, state, campaign, impression, clicks, spend, attributed_revenue

Business.csv → Daily business performance data

Columns: date, orders, new_orders, new_customers, total_revenue, gross_profit, COGS

⚙️ Features

✔️ Marketing KPIs (Spend, Attributed Revenue, Impressions, Clicks, CTR, CPC, ROAS)
✔️ Business KPIs (Orders, Revenue, Gross Profit, COGS)
✔️ Channel-level performance comparison (Facebook vs Google vs TikTok)
✔️ Correlation between Marketing Spend & Business Outcomes
✔️ Trend charts for spend, clicks, and attributed revenue
✔️ Download prepared datasets (Excel export)
✔️ Interactive date filter

🚀 Tech Stack

Python 3.13

Streamlit → App framework

Pandas / NumPy → Data preparation

Plotly Express / Matplotlib → Visualizations

Statsmodels → Trendline fitting

XlsxWriter → Excel export

📥 Installation

Clone the repo:

git clone https://github.com/mehga09/marketing_dashboard.git
cd marketing_dashboard


Create a virtual environment (recommended):

python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Mac/Linux


Install dependencies:

pip install -r requirements.txt

▶️ Running the App

Inside your project folder, run:

streamlit run marketing_dashboard/streamlit_app.py


Then open the link (usually http://localhost:8501/) in your browser.

🌍 Hosting (Optional)

You can host this dashboard on Streamlit Cloud:

Push this repo to GitHub.

Go to share.streamlit.io
.

Connect your GitHub repo.

Select marketing_dashboard/streamlit_app.py as the entry point.

📊 Example Insights

Which channel drives the highest ROAS (Return on Ad Spend)?

How does daily marketing spend correlate with orders?

Are certain tactics/states underperforming?

Which channel contributes most to new customer acquisition?

👩‍💻 Author

Mehga Rani
📌 Built as part of Marketing Intelligence Assessment 1.

