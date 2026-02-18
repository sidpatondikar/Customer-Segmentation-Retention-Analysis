import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global */
    .stApp { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3, h4 { font-family: 'DM Sans', sans-serif; font-weight: 700; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 12px;
        padding: 20px 24px;
        color: white;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .metric-card .label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .metric-card .value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 28px;
        font-weight: 700;
        color: #f1f5f9;
    }
    .metric-card .sub {
        font-size: 13px;
        color: #64748b;
        margin-top: 2px;
    }

    /* Segment badge */
    .segment-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
        letter-spacing: 0.5px;
    }

    /* Action badges */
    .action-priority { background: #fecaca; color: #991b1b; }
    .action-protect { background: #bbf7d0; color: #166534; }
    .action-nurture { background: #bfdbfe; color: #1e40af; }
    .action-reengage { background: #fde68a; color: #92400e; }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        font-size: 15px;
        padding: 10px 24px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    clean = pd.read_csv('data/processed/clean_data.csv', parse_dates=['InvoiceDate'])
    rfm = pd.read_csv('data/processed/rfm_segmented.csv')
    churn = pd.read_csv('data/processed/churn_clv.csv')
    return clean, rfm, churn

try:
    df_clean, rfm, churn = load_data()
except FileNotFoundError:
    st.error("⚠️ Data files not found. Make sure these exist:\n"
             "- `data/processed/clean_data.csv`\n"
             "- `data/processed/rfm_segmented.csv`\n"
             "- `data/processed/churn_clv.csv`")
    st.stop()


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────
def metric_card(label, value, sub=""):
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {sub_html}
    </div>
    """

def action_badge(action):
    css_map = {
        'Priority Retention': 'action-priority',
        'Protect & Reward': 'action-protect',
        'Nurture & Grow': 'action-nurture',
        'Low-Cost Re-engage': 'action-reengage'
    }
    css = css_map.get(action, '')
    return f'<span class="segment-badge {css}">{action}</span>'

SEGMENT_COLORS = {
    'Champions': '#22c55e', 'Loyal': '#3b82f6', 'New Customers': '#a855f7',
    'At Risk': '#f97316', 'Need Attention': '#eab308', 'Cant Lose Them': '#ef4444',
    'Lost': '#6b7280'
}

ACTION_COLORS = {
    'Priority Retention': '#ef4444', 'Protect & Reward': '#22c55e',
    'Nurture & Grow': '#3b82f6', 'Low-Cost Re-engage': '#eab308'
}


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("# 📊 Customer Intelligence Dashboard")
st.markdown("*Segmentation · Churn Prediction · Lifetime Value*")
st.markdown("---")


# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Customer Lookup", "👥 Segment Explorer",
    "⚠️ Churn Risk Dashboard", "📈 Business Overview"
])


# ═════════════════════════════════════════════
# TAB 1: Customer Lookup
# ═════════════════════════════════════════════
with tab1:
    st.markdown("### Customer Lookup")
    st.markdown("Search for any customer to see their complete profile: segment, churn risk, predicted lifetime value, and recommended action.")

    # Customer selector
    all_customers = sorted(rfm['Customer ID'].unique())
    selected_id = st.selectbox("Select Customer ID", all_customers,
                                index=0, key="customer_lookup")

    # Get customer data from both tables
    rfm_row = rfm[rfm['Customer ID'] == selected_id]
    churn_row = churn[churn['Customer ID'] == selected_id]

    if rfm_row.empty:
        st.warning(f"Customer {selected_id} not found in segmentation data.")
    else:
        r = rfm_row.iloc[0]

        # Row 1: Segment + Cluster
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(metric_card("RFM Segment", r['Segment'],
                                     f"Cluster: {r['Cluster_Name']}"),
                        unsafe_allow_html=True)
        with col2:
            st.markdown(metric_card("RFM Score", f"{int(r['RFM_Score'])} / 12",
                                     f"R={int(r['R_Score'])}  F={int(r['F_Score'])}  M={int(r['M_Score'])}"),
                        unsafe_allow_html=True)
        with col3:
            st.markdown(metric_card("Recency", f"{int(r['Recency'])} days",
                                     "Days since last purchase"),
                        unsafe_allow_html=True)

        st.markdown("")

        # Row 2: RFM values
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(metric_card("Frequency", f"{int(r['Frequency'])} orders"),
                        unsafe_allow_html=True)
        with col2:
            st.markdown(metric_card("Monetary", f"£{r['Monetary']:,.2f}",
                                     "Total lifetime spend"),
                        unsafe_allow_html=True)
        with col3:
            if not churn_row.empty:
                c = churn_row.iloc[0]
                st.markdown(metric_card("Cancel Rate",
                                         f"{c['Cancel_Rate']*100:.1f}%",
                                         f"{int(c['Cancellation_Count'])} cancellations"),
                            unsafe_allow_html=True)
            else:
                st.markdown(metric_card("Cancel Rate", "N/A"), unsafe_allow_html=True)

        st.markdown("")

        # Row 3: Churn + CLV (if available)
        if not churn_row.empty:
            c = churn_row.iloc[0]
            col1, col2, col3 = st.columns(3)
            with col1:
                churn_pct = c['Churn_Probability'] * 100
                risk = "🔴 High" if churn_pct > 50 else "🟡 Medium" if churn_pct > 30 else "🟢 Low"
                st.markdown(metric_card("Churn Probability",
                                         f"{churn_pct:.1f}%", risk),
                            unsafe_allow_html=True)
            with col2:
                st.markdown(metric_card("Predicted CLV",
                                         f"£{c['Predicted_CLV']:,.2f}",
                                         "Estimated 6-month future spend"),
                            unsafe_allow_html=True)
            with col3:
                st.markdown(metric_card("Recommended Action",
                                         c['Action']),
                            unsafe_allow_html=True)

            st.markdown("")

            # Action explanation
            action_descriptions = {
                'Priority Retention': '🚨 **High value customer at risk of leaving.** Immediate personal outreach recommended — exclusive offers, loyalty rewards, dedicated account manager.',
                'Protect & Reward': '🛡️ **High value customer, likely to stay.** Maintain VIP treatment — early access to new products, appreciation gestures, premium support.',
                'Nurture & Grow': '🌱 **Active but low spend.** Upsell and cross-sell opportunity — product recommendations, bundle deals, volume discounts.',
                'Low-Cost Re-engage': '📧 **Low value and likely leaving.** Automated win-back emails only — don\'t overspend on retention.'
            }
            st.info(action_descriptions.get(c['Action'], ''))
        else:
            st.caption("⚠️ Churn/CLV data not available for this customer (may not be in the training period).")


# ═════════════════════════════════════════════
# TAB 2: Segment Explorer
# ═════════════════════════════════════════════
with tab2:
    st.markdown("### Segment Explorer")
    st.markdown("Compare customer segments and drill down into each group.")

    # Segment selector
    segments = sorted(rfm['Segment'].unique())
    selected_segment = st.selectbox("Select Segment", ['All Segments'] + segments)

    # Segment summary
    seg_summary = rfm.groupby('Segment').agg(
        Customers=('Customer ID', 'count'),
        Avg_Recency=('Recency', 'mean'),
        Avg_Frequency=('Frequency', 'mean'),
        Avg_Monetary=('Monetary', 'mean'),
        Total_Revenue=('Monetary', 'sum')
    ).reset_index()
    seg_summary['Revenue_Share'] = (seg_summary['Total_Revenue'] /
                                     seg_summary['Total_Revenue'].sum() * 100)

    if selected_segment == 'All Segments':
        # Overview charts
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(seg_summary, values='Customers', names='Segment',
                         title='Customer Distribution by Segment',
                         color='Segment', color_discrete_map=SEGMENT_COLORS)
            fig.update_layout(template='plotly_white', height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.pie(seg_summary, values='Total_Revenue', names='Segment',
                         title='Revenue Distribution by Segment',
                         color='Segment', color_discrete_map=SEGMENT_COLORS)
            fig.update_layout(template='plotly_white', height=400)
            st.plotly_chart(fig, use_container_width=True)

        # RFM comparison
        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=['Avg Recency (days)', 'Avg Frequency (orders)', 'Avg Monetary (£)'])
        seg_order = seg_summary.sort_values('Avg_Monetary', ascending=False)['Segment'].tolist()
        colors = [SEGMENT_COLORS.get(s, '#6b7280') for s in seg_order]

        fig.add_trace(go.Bar(x=seg_order,
                             y=seg_summary.set_index('Segment').loc[seg_order, 'Avg_Recency'],
                             marker_color=colors), row=1, col=1)
        fig.add_trace(go.Bar(x=seg_order,
                             y=seg_summary.set_index('Segment').loc[seg_order, 'Avg_Frequency'],
                             marker_color=colors), row=1, col=2)
        fig.add_trace(go.Bar(x=seg_order,
                             y=seg_summary.set_index('Segment').loc[seg_order, 'Avg_Monetary'],
                             marker_color=colors), row=1, col=3)

        fig.update_layout(title='Segment Profiles — RFM Comparison',
                          template='plotly_white', height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Summary table
        st.markdown("#### Segment Summary Table")
        display_summary = seg_summary.copy()
        display_summary['Avg_Recency'] = display_summary['Avg_Recency'].round(0).astype(int)
        display_summary['Avg_Frequency'] = display_summary['Avg_Frequency'].round(1)
        display_summary['Avg_Monetary'] = display_summary['Avg_Monetary'].apply(lambda x: f"£{x:,.0f}")
        display_summary['Total_Revenue'] = display_summary['Total_Revenue'].apply(lambda x: f"£{x:,.0f}")
        display_summary['Revenue_Share'] = display_summary['Revenue_Share'].apply(lambda x: f"{x:.1f}%")
        st.dataframe(display_summary.sort_values('Customers', ascending=False),
                     use_container_width=True, hide_index=True)

    else:
        # Specific segment detail
        seg_data = rfm[rfm['Segment'] == selected_segment]
        seg_stats = seg_summary[seg_summary['Segment'] == selected_segment].iloc[0]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(metric_card("Customers", f"{int(seg_stats['Customers']):,}",
                                     f"{seg_stats['Customers']/len(rfm)*100:.1f}% of total"),
                        unsafe_allow_html=True)
        with col2:
            st.markdown(metric_card("Avg Recency", f"{seg_stats['Avg_Recency']:.0f} days"),
                        unsafe_allow_html=True)
        with col3:
            st.markdown(metric_card("Avg Frequency", f"{seg_stats['Avg_Frequency']:.1f} orders"),
                        unsafe_allow_html=True)
        with col4:
            st.markdown(metric_card("Avg Monetary", f"£{seg_stats['Avg_Monetary']:,.0f}",
                                     f"Revenue share: {seg_stats['Revenue_Share']:.1f}%"),
                        unsafe_allow_html=True)

        st.markdown("")

        # Distribution charts for this segment
        col1, col2, col3 = st.columns(3)
        with col1:
            fig = px.histogram(seg_data, x='Recency', nbins=30,
                               title='Recency Distribution',
                               color_discrete_sequence=[SEGMENT_COLORS.get(selected_segment, '#636EFA')])
            fig.update_layout(template='plotly_white', height=300)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.histogram(seg_data, x='Frequency', nbins=30,
                               title='Frequency Distribution',
                               color_discrete_sequence=[SEGMENT_COLORS.get(selected_segment, '#636EFA')])
            fig.update_layout(template='plotly_white', height=300)
            st.plotly_chart(fig, use_container_width=True)
        with col3:
            fig = px.histogram(seg_data[seg_data['Monetary'] < seg_data['Monetary'].quantile(0.99)],
                               x='Monetary', nbins=30,
                               title='Monetary Distribution (< 99th pctl)',
                               color_discrete_sequence=[SEGMENT_COLORS.get(selected_segment, '#636EFA')])
            fig.update_layout(template='plotly_white', height=300)
            st.plotly_chart(fig, use_container_width=True)

        # Customer list with churn data if available
        st.markdown(f"#### Customers in {selected_segment}")
        merged = seg_data.merge(
            churn[['Customer ID', 'Churn_Probability', 'Predicted_CLV', 'Action']],
            on='Customer ID', how='left'
        )
        display_cols = ['Customer ID', 'Recency', 'Frequency', 'Monetary',
                        'RFM_Score', 'Cluster_Name']
        if 'Churn_Probability' in merged.columns:
            display_cols += ['Churn_Probability', 'Predicted_CLV', 'Action']

        st.dataframe(
            merged[display_cols].sort_values('Monetary', ascending=False).head(50),
            use_container_width=True, hide_index=True
        )
        st.caption(f"Showing top 50 by spend out of {len(seg_data):,} customers")


# ═════════════════════════════════════════════
# TAB 3: Churn Risk Dashboard
# ═════════════════════════════════════════════
with tab3:
    st.markdown("### Churn Risk Dashboard")
    st.markdown("Adjust the churn threshold to see how it changes business recommendations and at-risk revenue.")

    # Interactive threshold slider
    col1, col2 = st.columns([1, 1])
    with col1:
        churn_thresh = st.slider("Churn Probability Threshold",
                                  min_value=0.1, max_value=0.9, value=0.5, step=0.05,
                                  help="Customers above this threshold are flagged as high churn risk")
    with col2:
        clv_percentile = st.slider("CLV Threshold (percentile)",
                                    min_value=50, max_value=95, value=75, step=5,
                                    help="Customers above this CLV percentile are considered high value")

    clv_thresh = churn['Predicted_CLV'].quantile(clv_percentile / 100)

    # Recalculate actions with new thresholds
    def assign_action(row):
        high_clv = row['Predicted_CLV'] > clv_thresh
        high_churn = row['Churn_Probability'] > churn_thresh
        if high_clv and high_churn:
            return 'Priority Retention'
        elif high_clv and not high_churn:
            return 'Protect & Reward'
        elif not high_clv and high_churn:
            return 'Low-Cost Re-engage'
        else:
            return 'Nurture & Grow'

    churn_dynamic = churn.copy()
    churn_dynamic['Action_Dynamic'] = churn_dynamic.apply(assign_action, axis=1)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    for col, action, emoji in zip(
        [col1, col2, col3, col4],
        ['Priority Retention', 'Protect & Reward', 'Nurture & Grow', 'Low-Cost Re-engage'],
        ['🚨', '🛡️', '🌱', '📧']
    ):
        subset = churn_dynamic[churn_dynamic['Action_Dynamic'] == action]
        with col:
            st.markdown(metric_card(
                f"{emoji} {action}",
                f"{len(subset):,}",
                f"£{subset['Predicted_CLV'].sum():,.0f} total value"
            ), unsafe_allow_html=True)

    st.markdown("")

    # Action matrix scatter plot
    plot_df = churn_dynamic[churn_dynamic['Predicted_CLV'] > 1].copy()
    fig = px.scatter(plot_df, x='Churn_Probability', y='Predicted_CLV',
                     color='Action_Dynamic', opacity=0.5,
                     color_discrete_map=ACTION_COLORS,
                     log_y=True,
                     hover_data=['Customer ID', 'Recency', 'Frequency'])
    fig.add_hline(y=clv_thresh, line_dash='dash', line_color='gray',
                  annotation_text=f'CLV Threshold: £{clv_thresh:,.0f}')
    fig.add_vline(x=churn_thresh, line_dash='dash', line_color='gray',
                  annotation_text=f'Churn Threshold: {churn_thresh}')
    fig.update_layout(
        title='Customer Action Matrix — Churn Risk vs Lifetime Value',
        template='plotly_white', height=550,
        xaxis_title='Churn Probability',
        yaxis_title='Predicted CLV (£, log scale)',
        yaxis_range=[0, np.log10(plot_df['Predicted_CLV'].quantile(0.999)) + 0.5],
        legend_title='Action'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Priority Retention list
    priority = churn_dynamic[churn_dynamic['Action_Dynamic'] == 'Priority Retention'].sort_values(
        'Predicted_CLV', ascending=False)

    if len(priority) > 0:
        st.markdown(f"#### 🚨 Priority Retention — {len(priority)} Customers (£{priority['Predicted_CLV'].sum():,.0f} at risk)")
        st.dataframe(
            priority[['Customer ID', 'Predicted_CLV', 'Churn_Probability',
                       'Recency', 'Frequency', 'Monetary', 'Cancel_Rate']].head(20),
            use_container_width=True, hide_index=True
        )
    else:
        st.info("No customers in Priority Retention with current thresholds.")

    # Churn rate by recency bucket
    st.markdown("#### Churn Rate by Recency")
    churn_dynamic['Recency_Bucket'] = pd.cut(churn_dynamic['Recency'],
                                              bins=[0, 30, 90, 180, 365, 600],
                                              labels=['0-30', '31-90', '91-180', '181-365', '365+'])
    churn_by_recency = churn_dynamic.groupby('Recency_Bucket', observed=False)['Churned'].mean() * 100

    fig = go.Figure(go.Bar(
        x=churn_by_recency.index.astype(str),
        y=churn_by_recency.values,
        marker_color=['#22c55e', '#84cc16', '#eab308', '#f97316', '#ef4444'],
        text=[f'{v:.1f}%' for v in churn_by_recency.values],
        textposition='outside'
    ))
    fig.update_layout(title='Churn Rate by Days Since Last Purchase',
                      xaxis_title='Recency Bucket (days)',
                      yaxis_title='Churn Rate (%)',
                      template='plotly_white', height=400)
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════
# TAB 4: Business Overview
# ═════════════════════════════════════════════
with tab4:
    st.markdown("### Business Overview")
    st.markdown("Key metrics and trends from the underlying transaction data.")

    purchases = df_clean[~df_clean['IsCancelled']].copy()

    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(metric_card("Total Revenue",
                                 f"£{purchases['TotalAmount'].sum():,.0f}",
                                 f"{purchases['Invoice'].nunique():,} orders"),
                    unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card("Unique Customers",
                                 f"{purchases['Customer ID'].nunique():,}",
                                 f"{purchases['Country'].nunique()} countries"),
                    unsafe_allow_html=True)
    with col3:
        aov = purchases.groupby('Invoice')['TotalAmount'].sum().mean()
        st.markdown(metric_card("Avg Order Value", f"£{aov:,.2f}"),
                    unsafe_allow_html=True)
    with col4:
        avg_orders = purchases.groupby('Customer ID')['Invoice'].nunique().mean()
        st.markdown(metric_card("Avg Orders/Customer", f"{avg_orders:.1f}"),
                    unsafe_allow_html=True)

    st.markdown("")

    # Monthly revenue trend
    purchases['YearMonth'] = purchases['InvoiceDate'].dt.to_period('M').astype(str)
    monthly = purchases.groupby('YearMonth').agg(
        Revenue=('TotalAmount', 'sum'),
        Orders=('Invoice', 'nunique'),
        Customers=('Customer ID', 'nunique')
    ).reset_index()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=['Monthly Revenue (£)', 'Monthly Active Customers'],
                        vertical_spacing=0.12)
    fig.add_trace(go.Bar(x=monthly['YearMonth'], y=monthly['Revenue'],
                         marker_color='#3b82f6'), row=1, col=1)
    fig.add_trace(go.Scatter(x=monthly['YearMonth'], y=monthly['Customers'],
                             mode='lines+markers', marker_color='#22c55e'), row=2, col=1)
    fig.update_layout(template='plotly_white', height=500, showlegend=False,
                      title='Monthly Business Trends')
    st.plotly_chart(fig, use_container_width=True)

    # Two columns: Geography + Pareto
    col1, col2 = st.columns(2)

    with col1:
        # Top countries (excluding UK)
        country_rev = purchases.groupby('Country')['TotalAmount'].sum().sort_values(ascending=False)
        uk_share = country_rev.iloc[0] / country_rev.sum() * 100
        country_no_uk = country_rev.iloc[1:11].reset_index()
        country_no_uk.columns = ['Country', 'Revenue']

        fig = px.bar(country_no_uk, x='Revenue', y='Country', orientation='h',
                     title=f'Top 10 Countries by Revenue (excl. UK — {uk_share:.0f}%)',
                     color='Revenue', color_continuous_scale='Blues')
        fig.update_layout(template='plotly_white', height=400,
                          yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Pareto curve
        customer_spend = purchases.groupby('Customer ID')['TotalAmount'].sum().sort_values(ascending=False)
        cumulative_pct = customer_spend.cumsum() / customer_spend.sum() * 100
        customer_pct = np.arange(1, len(customer_spend) + 1) / len(customer_spend) * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=customer_pct, y=cumulative_pct,
                                 mode='lines', line=dict(color='#3b82f6', width=2)))
        fig.add_hline(y=80, line_dash='dash', line_color='red',
                      annotation_text='80% Revenue')
        pct_at_80 = customer_pct[np.searchsorted(cumulative_pct.values, 80)]
        fig.update_layout(title=f'Revenue Concentration (Top {pct_at_80:.0f}% → 80% Revenue)',
                          xaxis_title='% of Customers', yaxis_title='% Cumulative Revenue',
                          template='plotly_white', height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Purchase timing
    col1, col2 = st.columns(2)
    with col1:
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        purchases['DayOfWeek'] = purchases['InvoiceDate'].dt.day_name()
        day_counts = purchases.groupby('DayOfWeek')['Invoice'].nunique().reindex(day_order)
        fig = go.Figure(go.Bar(x=day_counts.index, y=day_counts.values,
                               marker_color='#8b5cf6'))
        fig.update_layout(title='Orders by Day of Week',
                          template='plotly_white', height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        purchases['Hour'] = purchases['InvoiceDate'].dt.hour
        hour_counts = purchases.groupby('Hour')['Invoice'].nunique().sort_index()
        fig = go.Figure(go.Bar(x=hour_counts.index, y=hour_counts.values,
                               marker_color='#ec4899'))
        fig.update_layout(title='Orders by Hour of Day',
                          xaxis_title='Hour', template='plotly_white', height=350)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#64748b; font-size:13px;'>"
    "Customer Intelligence Dashboard · Built with Streamlit · Data: Online Retail II (UCI)"
    "</div>",
    unsafe_allow_html=True
)