# streamlit_app.py — Final updated & production-ready
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import io

st.set_page_config(page_title="Marketing Intelligence Dashboard", layout="wide", page_icon="📊")

# ---------- Utilities ----------
def find_col(df, candidates):
    """Return first matching column name from list of candidates (case-insensitive) or None."""
    low = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    for col in df.columns:
        cl = col.lower()
        for cand in candidates:
            if cand.lower() in cl or cl in cand.lower():
                return col
    return None

def standardize_marketing_df(df, source_name):
    df = df.copy()
    df['source'] = source_name
    mapping = {
        'date': find_col(df, ['date','day']),
        'campaign': find_col(df, ['campaign','campaign_name','ad_campaign']),
        'tactic': find_col(df, ['tactic','adset','ad_group']),
        'state': find_col(df, ['state','region','location']),
        'impressions': find_col(df, ['impressions','impression','views','reach']),
        'clicks': find_col(df, ['clicks','click']),
        'spend': find_col(df, ['spend','cost','amount_spent']),
        'attributed_revenue': find_col(df, ['attributed_revenue','attributed revenue','revenue'])
    }
    rename_map = {v:k for k,v in mapping.items() if v is not None}
    if rename_map:
        df = df.rename(columns=rename_map)
    for want in ['date','campaign','tactic','state','impressions','clicks','spend','attributed_revenue','source']:
        if want not in df.columns:
            df[want] = pd.NA
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    for n in ['impressions','clicks','spend','attributed_revenue']:
        df[n] = pd.to_numeric(df[n], errors='coerce')
    return df[['date','source','tactic','state','campaign','impressions','clicks','spend','attributed_revenue']]

def standardize_business_df(df):
    df = df.copy()
    mapping = {
        'date': find_col(df, ['date','day']),
        'orders': find_col(df, ['orders','order_count']),
        'new_orders': find_col(df, ['new_orders','new orders']),
        'new_customers': find_col(df, ['new_customers','new customers']),
        'total_revenue': find_col(df, ['total_revenue','total revenue','revenue']),
        'gross_profit': find_col(df, ['gross_profit','gross profit']),
        'cogs': find_col(df, ['cogs','cost_of_goods_sold','cost'])
    }
    rename_map = {v:k for k,v in mapping.items() if v is not None}
    if rename_map:
        df = df.rename(columns=rename_map)
    for want in ['date','orders','new_orders','new_customers','total_revenue','gross_profit','cogs']:
        if want not in df.columns:
            df[want] = pd.NA
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    for n in ['orders','new_orders','new_customers','total_revenue','gross_profit','cogs']:
        df[n] = pd.to_numeric(df[n], errors='coerce')
    return df[['date','orders','new_orders','new_customers','total_revenue','gross_profit','cogs']]

# ---------- Load & prepare ----------
@st.cache_data
def load_prepare():
    base = Path('.')
    files = {}
    for fname in ['Facebook.csv','Google.csv','TikTok.csv','Business.csv']:
        p = base / fname
        if p.exists():
            try:
                files[fname] = pd.read_csv(p)
            except Exception:
                files[fname] = pd.read_csv(p, encoding='latin1')
        else:
            files[fname] = None

    marketing_list = []
    for name in ['Facebook.csv','Google.csv','TikTok.csv']:
        df = files.get(name)
        if df is not None:
            src = Path(name).stem
            marketing_list.append(standardize_marketing_df(df, src))
    if marketing_list:
        marketing = pd.concat(marketing_list, ignore_index=True)
    else:
        marketing = pd.DataFrame(columns=['date','source','tactic','state','campaign','impressions','clicks','spend','attributed_revenue'])

    biz_df = files.get('Business.csv')
    if biz_df is not None:
        business = standardize_business_df(biz_df)
    else:
        business = pd.DataFrame(columns=['date','orders','new_orders','new_customers','total_revenue','gross_profit','cogs'])

    agg_daily = marketing.groupby(['date','source']).agg({
        'impressions':'sum','clicks':'sum','spend':'sum','attributed_revenue':'sum'
    }).reset_index()

    for df in [marketing, agg_daily]:
        df['ctr'] = np.where(df['impressions']>0, df['clicks'] / df['impressions'], np.nan)
        df['cpc'] = np.where(df['clicks']>0, df['spend'] / df['clicks'], np.nan)
        df['roas'] = np.where(df['spend']>0, df['attributed_revenue'] / df['spend'], np.nan)

    joined = pd.merge(business, agg_daily, on='date', how='left')

    campaign_summary = marketing.groupby(['source','campaign']).agg({
        'impressions':'sum','clicks':'sum','spend':'sum','attributed_revenue':'sum'
    }).reset_index()
    campaign_summary['ctr'] = np.where(campaign_summary['impressions']>0, campaign_summary['clicks']/campaign_summary['impressions'], np.nan)
    campaign_summary['roas'] = np.where(campaign_summary['spend']>0, campaign_summary['attributed_revenue']/campaign_summary['spend'], np.nan)

    return marketing, agg_daily, business, joined, campaign_summary

marketing, agg_daily, business, joined, campaign_summary = load_prepare()

# ---------- Styling ----------
st.markdown("""
    <style>
    .stApp { background: #ff7f24; }
    .kpi { background: #ffebcd; border-radius:10px; padding:12px; 
           box-shadow: 0 2px 6px rgba(0,0,0,0.1); text-align:center; }
    .big-cta { font-size:16px; font-weight:600; color:#444; }
    .kpi-value { font-size:20px; font-weight:700; color:#111; }
    </style>
""", unsafe_allow_html=True)

# ---------- Controls ----------
st.sidebar.image("https://streamlit.io/images/brand/streamlit-mark.png", width=80)
st.sidebar.header("Filters & Controls")

min_date = pd.to_datetime(marketing['date'].min())
max_date = pd.to_datetime(marketing['date'].max())
if pd.isna(min_date): min_date = pd.to_datetime("today") - pd.Timedelta(days=120)
if pd.isna(max_date): max_date = pd.to_datetime("today")

date_range = st.sidebar.date_input(
    "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
)

# Ensure tuple always has 2 values
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

sources_available = marketing['source'].dropna().unique().tolist()
if not sources_available:
    sources_available = ['Facebook','Google','TikTok']
channels = st.sidebar.multiselect("Channels", options=sources_available, default=sources_available)

unique_campaigns = marketing['campaign'].dropna().unique().tolist()
campaign_sel = st.sidebar.selectbox("Campaign (optional)", options=["(All)"] + unique_campaigns, index=0)
unique_states = marketing['state'].dropna().unique().tolist()
state_sel = st.sidebar.selectbox("State (optional)", options=["(All)"] + unique_states, index=0)

mask = (marketing['date'] >= pd.to_datetime(start_date)) & (marketing['date'] <= pd.to_datetime(end_date))
mask = mask & marketing['source'].isin(channels)
if campaign_sel != "(All)": mask = mask & (marketing['campaign']==campaign_sel)
if state_sel != "(All)": mask = mask & (marketing['state']==state_sel)
marketing_f = marketing[mask].copy()

mask_agg = (agg_daily['date'] >= pd.to_datetime(start_date)) & (agg_daily['date'] <= pd.to_datetime(end_date))
mask_agg = mask_agg & agg_daily['source'].isin(channels)
agg_f = agg_daily[mask_agg].copy()

mask_joined = (joined['date'] >= pd.to_datetime(start_date)) & (joined['date'] <= pd.to_datetime(end_date))
joined_f = joined[mask_joined].copy()

# ---------- KPIs ----------
st.title("📊 Marketing Intelligence Dashboard")
st.write("Interactive dashboard connecting marketing activity to business outcomes.")

kpi1 = marketing_f['spend'].sum(skipna=True)
kpi2 = marketing_f['attributed_revenue'].sum(skipna=True)
kpi3 = marketing_f['impressions'].sum(skipna=True)
kpi4 = marketing_f['clicks'].sum(skipna=True)

c1,c2,c3,c4 = st.columns(4)
c1.markdown(f"<div class='kpi'><div class='big-cta'>Total Spend</div><div class='kpi-value'>₹{kpi1:,.2f}</div></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='kpi'><div class='big-cta'>Attributed Revenue</div><div class='kpi-value'>₹{kpi2:,.2f}</div></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='kpi'><div class='big-cta'>Impressions</div><div class='kpi-value'>{int(kpi3) if not np.isnan(kpi3) else '—'}</div></div>", unsafe_allow_html=True)
c4.markdown(f"<div class='kpi'><div class='big-cta'>Clicks</div><div class='kpi-value'>{int(kpi4) if not np.isnan(kpi4) else '—'}</div></div>", unsafe_allow_html=True)

# ---------- Trends ----------
st.markdown("### Spend & Attributed Revenue — 7-day rolling")
if not agg_f.empty:
    ts = agg_f.groupby('date').agg({'spend':'sum','attributed_revenue':'sum'}).reset_index().sort_values('date')
    ts['spend_7d'] = ts['spend'].rolling(7, min_periods=1).mean()
    ts['rev_7d'] = ts['attributed_revenue'].rolling(7, min_periods=1).mean()
    fig = px.line(ts, x='date', y=['spend_7d','rev_7d'], labels={'value':'Amount','variable':'Metric'}, title="7-day Rolling Spend vs Revenue")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Not enough spend/revenue data to plot trend.")

# ---------- Performance ----------
st.markdown("### Performance — CTR & CPC (7-day rolling)")
if not agg_f.empty:
    perf = agg_f.groupby('date').agg({'ctr':'mean','cpc':'mean'}).reset_index().sort_values('date')
    perf['ctr_7d'] = perf['ctr'].rolling(7, min_periods=1).mean()
    perf['cpc_7d'] = perf['cpc'].rolling(7, min_periods=1).mean()
    st.plotly_chart(px.line(perf, x='date', y=['ctr_7d'], title='CTR (7-day rolling)'), use_container_width=True)
    st.plotly_chart(px.line(perf, x='date', y=['cpc_7d'], title='CPC (7-day rolling)'), use_container_width=True)
else:
    st.info("CTR/CPC not available in your dataset.")

# ---------- Channel breakdown ----------
st.markdown("### Channel breakdown")
if not agg_f.empty:
    channel_totals = agg_f.groupby('source').agg({'spend':'sum','attributed_revenue':'sum','impressions':'sum','clicks':'sum'}).reset_index()
    channel_totals['roas'] = np.where(channel_totals['spend']>0, channel_totals['attributed_revenue']/channel_totals['spend'], np.nan)
    st.plotly_chart(px.bar(channel_totals, x='source', y='spend', text='spend', title='Total Spend by Channel'), use_container_width=True)
    st.plotly_chart(px.bar(channel_totals, x='source', y='roas', text='roas', title='ROAS by Channel'), use_container_width=True)
else:
    st.info("No channel data available for breakdown.")

# ---------- Campaigns ----------
st.markdown("### Campaign performance (top sample)")
if not campaign_summary.empty:
    cs = campaign_summary.copy()
    if campaign_sel != "(All)": cs = cs[cs['campaign']==campaign_sel]
    cs = cs.sort_values('roas', ascending=False).head(20)
    st.dataframe(cs[['source','campaign','spend','attributed_revenue','roas','ctr']].round(2))
    st.plotly_chart(px.bar(cs, x='campaign', y='roas', color='source', title='Top campaigns by ROAS (top 20)'), use_container_width=True)
else:
    st.info("No campaign-level data found.")

# ---------- Business outcomes ----------
st.markdown("### Business outcomes vs marketing")
if not joined_f.empty and 'orders' in joined_f.columns:
    jagg = joined_f.groupby('date').agg({'spend':'sum','orders':'sum','total_revenue':'sum'}).reset_index()
    fig_scatter = px.scatter(jagg, x='spend', y='orders', title='Daily Spend vs Orders (aggregated)')
    st.plotly_chart(fig_scatter, use_container_width=True)
    corr = jagg['spend'].corr(jagg['orders'])
    st.write(f"Correlation (spend vs orders): **{corr:.2f}**")
else:
    st.info("Orders or spend data are missing — cannot show business impact scatter.")

# ---------- Funnel ----------
st.markdown("### Conversion funnel (if available)")
has_clicks = marketing_f['clicks'].sum(skipna=True) > 0
has_orders = business['orders'].sum(skipna=True) > 0
if has_clicks and has_orders:
    total_impr = marketing_f['impressions'].sum(skipna=True)
    total_clicks = marketing_f['clicks'].sum(skipna=True)
    total_orders = business[(business['date']>=pd.to_datetime(start_date)) & (business['date']<=pd.to_datetime(end_date))]['orders'].sum(skipna=True)
    funnel = pd.DataFrame({'stage':['Impressions','Clicks','Orders'],'value':[int(total_impr),int(total_clicks),int(total_orders)]})
    st.plotly_chart(px.funnel(funnel, x='value', y='stage', title='Impression → Click → Order Funnel'), use_container_width=True)
else:
    st.info("Not enough data to build funnel (need clicks and orders).")

# ---------- Download prepared data ----------
st.markdown("### Download prepared datasets")
buf = io.BytesIO()
dfs_to_export = {
    "marketing_filtered": marketing_f,
    "agg_daily": agg_f,
    "campaign_summary": campaign_summary,
    "business": business,
    "joined": joined_f
}
with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
    for sheet_name, df in dfs_to_export.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)
buf.seek(0)

st.download_button(
    label="⬇️ Download prepared Excel",
    data=buf,
    file_name="prepared_marketing_data.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
