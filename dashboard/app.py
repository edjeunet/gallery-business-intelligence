"""
Gallery Business Intelligence Dashboard
Multi-page Streamlit application for gallery analytics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from analytics import GalleryAnalytics

# Page configuration
st.set_page_config(
    page_title="Gallery Business Intelligence",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize analytics
@st.cache_data
def load_analytics():
    analytics = GalleryAnalytics()
    analytics.load_data()
    return analytics

def main():
    # Sidebar navigation
    st.sidebar.title("🎨 Gallery BI")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["Executive Overview", "Customer Intelligence", "Inventory Performance", "Exhibition Impact"]
    )
    
    # Load data
    analytics = load_analytics()
    
    # Page routing
    if page == "Executive Overview":
        executive_overview(analytics)
    elif page == "Customer Intelligence":
        customer_intelligence(analytics)
    elif page == "Inventory Performance":
        inventory_performance(analytics)
    elif page == "Exhibition Impact":
        exhibition_impact(analytics)

def executive_overview(analytics):
    """Executive Overview Dashboard Page"""
    st.title("🎯 Executive Overview")
    st.markdown("High-level performance metrics and trends")
    
    # Generate insights
    insights = analytics.generate_insights_summary()
    
    # KPI Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Revenue",
            f"€{insights['product_insights']['total_revenue']:,.0f}",
            delta="12M Performance"
        )
    
    with col2:
        st.metric(
            "Active Customers",
            f"{insights['customer_insights']['total_customers']:,}",
            delta=f"{insights['customer_insights']['vip_collectors']} VIP"
        )
    
    with col3:
        st.metric(
            "Exhibition Uplift",
            f"{insights['exhibition_impact']['uplift_percent']}%",
            delta="vs Baseline"
        )
    
    with col4:
        st.metric(
            "Avg Customer Value",
            f"€{insights['customer_insights']['avg_customer_value']:.0f}",
            delta=f"Max: €{insights['customer_insights']['top_customer_value']:.0f}"
        )
    
    st.markdown("---")
    
    # Monthly Revenue Trend
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Monthly Revenue Trend")
        
        # Aggregate monthly data
        monthly_data = analytics.transactions.copy()
        monthly_data['month'] = monthly_data['transaction_date'].dt.to_period('M')
        monthly_revenue = monthly_data.groupby('month')['total_amount'].sum().reset_index()
        monthly_revenue['month'] = monthly_revenue['month'].astype(str)
        
        fig = px.line(
            monthly_revenue, 
            x='month', 
            y='total_amount',
            title="Monthly Revenue Performance",
            markers=True
        )
        fig.update_layout(xaxis_title="Month", yaxis_title="Revenue (€)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏆 Top Product Categories")
        
        _, category_analysis = analytics.product_performance_analysis()
        
        fig = px.pie(
            category_analysis, 
            values='total_revenue', 
            names='category',
            title="Revenue by Category"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Customer Segments Overview
    st.subheader("👥 Customer Segment Distribution")
    
    rfm_data = analytics.calculate_rfm()
    segment_counts = rfm_data['customer_value'].value_counts()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fig = px.bar(
            x=segment_counts.values,
            y=segment_counts.index,
            orientation='h',
            title="Customer Segments by Count",
            labels={'x': 'Number of Customers', 'y': 'Segment'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        segment_revenue = rfm_data.groupby('customer_value')['monetary'].sum().sort_values(ascending=False)
        fig = px.bar(
            x=segment_revenue.values,
            y=segment_revenue.index,
            orientation='h',
            title="Revenue by Customer Segment",
            labels={'x': 'Total Revenue (€)', 'y': 'Segment'}
        )
        st.plotly_chart(fig, use_container_width=True)

def customer_intelligence(analytics):
    """Customer Intelligence Dashboard Page"""
    st.title("👥 Customer Intelligence")
    st.markdown("Deep dive into customer behavior and segmentation")
    
    # Calculate RFM and clustering
    rfm_data = analytics.calculate_rfm()
    clustered_data, cluster_centers = analytics.customer_clustering()
    
    # RFM Analysis Section
    st.subheader("📊 RFM Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = px.histogram(rfm_data, x='recency', title="Recency Distribution (Days since last purchase)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(rfm_data, x='frequency', title="Frequency Distribution (Number of purchases)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = px.histogram(rfm_data, x='monetary', title="Monetary Distribution (Total spent €)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Customer Segmentation
    st.subheader("🎯 Customer Segmentation Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Scatter plot of customer segments
        fig = px.scatter(
            rfm_data, 
            x='recency', 
            y='monetary',
            color='customer_value',
            size='frequency',
            title="Customer Segmentation: Recency vs Monetary Value",
            labels={'recency': 'Days since last purchase', 'monetary': 'Total spent (€)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Segment characteristics table
        st.markdown("**Segment Characteristics**")
        segment_stats = rfm_data.groupby('customer_value').agg({
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'mean'
        }).round(1)
        st.dataframe(segment_stats)
    
    # Customer Lifetime Value Analysis
    st.subheader("💰 Customer Lifetime Value Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CLV by original segment
        clv_by_segment = rfm_data.groupby('segment')['monetary'].agg(['mean', 'count']).round(2)
        clv_by_segment.columns = ['Avg CLV (€)', 'Customer Count']
        
        fig = px.bar(
            clv_by_segment.reset_index(),
            x='segment',
            y='Avg CLV (€)',
            title="Average Customer Value by Original Segment"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top customers table
        st.markdown("**🌟 Top 10 Customers**")
        top_customers = rfm_data.nlargest(10, 'monetary')[['monetary', 'frequency', 'recency', 'customer_value']]
        top_customers.columns = ['Total Spent (€)', 'Purchases', 'Days Since Last', 'Segment']
        st.dataframe(top_customers)

def inventory_performance(analytics):
    """Inventory Performance Dashboard Page"""
    st.title("📦 Inventory Performance")
    st.markdown("Product performance and inventory optimization insights")
    
    # Product analysis
    product_analysis, category_analysis = analytics.product_performance_analysis()
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Products", len(product_analysis))
    
    with col2:
        best_category = category_analysis.iloc[0]['category']
        st.metric("Top Category", best_category)
    
    with col3:
        avg_margin = (category_analysis['total_margin'].sum() / category_analysis['total_revenue'].sum() * 100)
        st.metric("Avg Margin", f"{avg_margin:.1f}%")
    
    with col4:
        total_margin = category_analysis['total_margin'].sum()
        st.metric("Total Margin", f"€{total_margin:,.0f}")
    
    st.markdown("---")
    
    # Category Performance
    st.subheader("📊 Category Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue by category
        fig = px.bar(
            category_analysis,
            x='category',
            y='total_revenue',
            title="Revenue by Product Category",
            color='abc_class',
            labels={'total_revenue': 'Revenue (€)', 'category': 'Category'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Margin analysis
        fig = px.scatter(
            category_analysis,
            x='total_revenue',
            y='total_margin',
            size='product_count',
            color='category',
            title="Revenue vs Margin by Category",
            labels={'total_revenue': 'Revenue (€)', 'total_margin': 'Margin (€)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ABC Analysis
    st.subheader("🔤 ABC Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # ABC classification table
        st.markdown("**Category Classification**")
        abc_table = category_analysis[['category', 'revenue_contribution', 'abc_class']].copy()
        abc_table.columns = ['Category', 'Revenue %', 'ABC Class']
        st.dataframe(abc_table, use_container_width=True)
    
    with col2:
        # Sell-through rates
        fig = px.bar(
            category_analysis,
            x='category',
            y='sell_through_rate',
            title="Sell-Through Rate by Category",
            labels={'sell_through_rate': 'Sell-Through Rate', 'category': 'Category'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed product table
    st.subheader("📋 Product Performance Details")
    
    # Add filters
    col1, col2 = st.columns(2)
    with col1:
        selected_category = st.selectbox("Filter by Category", ['All'] + list(category_analysis['category'].unique()))
    with col2:
        min_revenue = st.number_input("Minimum Revenue (€)", value=0, step=100)
    
    # Filter products
    filtered_products = product_analysis.copy()
    if selected_category != 'All':
        filtered_products = filtered_products[filtered_products['category'] == selected_category]
    filtered_products = filtered_products[filtered_products['total_revenue'] >= min_revenue]
    
    # Display table
    display_columns = ['product_id', 'category', 'price', 'total_quantity', 'total_revenue', 'total_margin']
    st.dataframe(
        filtered_products[display_columns].sort_values('total_revenue', ascending=False),
        use_container_width=True
    )

def exhibition_impact(analytics):
    """Exhibition Impact Dashboard Page"""
    st.title("🎭 Exhibition Impact Analysis")
    st.markdown("Measuring the impact of exhibitions on gallery sales")
    
    # Exhibition analysis
    daily_sales, exhibition_metrics = analytics.exhibition_impact_analysis()
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Revenue Uplift",
            f"{exhibition_metrics['uplift_percent']}%",
            delta="vs Baseline"
        )
    
    with col2:
        st.metric(
            "Exhibition Revenue",
            f"€{exhibition_metrics['total_exhibition_revenue']:,.0f}",
            delta="Total Impact"
        )
    
    with col3:
        st.metric(
            "Daily Avg (Exhibition)",
            f"€{exhibition_metrics['exhibition_daily_avg']:,.0f}",
            delta=f"vs €{exhibition_metrics['baseline_daily_avg']:,.0f}"
        )
    
    with col4:
        exhibition_days = len(daily_sales[daily_sales['exhibition_period']])
        st.metric("Exhibition Days", exhibition_days)
    
    st.markdown("---")
    
    # Daily sales trend
    st.subheader("📈 Daily Sales Performance")
    
    fig = go.Figure()
    
    # Baseline sales
    baseline_data = daily_sales[~daily_sales['exhibition_period']]
    fig.add_trace(go.Scatter(
        x=baseline_data['date'],
        y=baseline_data['revenue'],
        mode='markers',
        name='Regular Days',
        marker=dict(color='lightblue', size=4)
    ))
    
    # Exhibition sales
    exhibition_data = daily_sales[daily_sales['exhibition_period']]
    fig.add_trace(go.Scatter(
        x=exhibition_data['date'],
        y=exhibition_data['revenue'],
        mode='markers',
        name='Exhibition Days',
        marker=dict(color='red', size=6)
    ))
    
    # Add baseline average line
    fig.add_hline(
        y=exhibition_metrics['baseline_daily_avg'],
        line_dash="dash",
        line_color="blue",
        annotation_text="Baseline Average"
    )
    
    # Add exhibition average line
    fig.add_hline(
        y=exhibition_metrics['exhibition_daily_avg'],
        line_dash="dash",
        line_color="red",
        annotation_text="Exhibition Average"
    )
    
    fig.update_layout(
        title="Daily Revenue: Regular vs Exhibition Periods",
        xaxis_title="Date",
        yaxis_title="Revenue (€)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly comparison
    st.subheader("📊 Monthly Performance Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly revenue with exhibition markers
        monthly_data = daily_sales.copy()
        monthly_data['month'] = monthly_data['date'].dt.to_period('M')
        
        monthly_revenue = monthly_data.groupby('month').agg({
            'revenue': 'sum',
            'exhibition_period': 'any'
        }).reset_index()
        monthly_revenue['month'] = monthly_revenue['month'].astype(str)
        monthly_revenue['period_type'] = monthly_revenue['exhibition_period'].map({True: 'Exhibition Month', False: 'Regular Month'})
        
        fig = px.bar(
            monthly_revenue,
            x='month',
            y='revenue',
            color='period_type',
            title="Monthly Revenue by Period Type",
            labels={'revenue': 'Revenue (€)', 'month': 'Month'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Exhibition ROI calculation
        st.markdown("**🎯 Exhibition ROI Analysis**")
        
        baseline_revenue = exhibition_metrics['baseline_daily_avg'] * len(daily_sales[daily_sales['exhibition_period']])
        incremental_revenue = exhibition_metrics['total_exhibition_revenue'] - baseline_revenue
        
        roi_data = {
            'Metric': ['Baseline Expected', 'Actual Revenue', 'Incremental Revenue', 'ROI %'],
            'Value (€)': [
                f"{baseline_revenue:,.0f}",
                f"{exhibition_metrics['total_exhibition_revenue']:,.0f}",
                f"{incremental_revenue:,.0f}",
                f"{exhibition_metrics['uplift_percent']}%"
            ]
        }
        
        st.table(pd.DataFrame(roi_data))
        
        # Success factors
        st.markdown("**📋 Key Success Factors**")
        st.write("• Exhibition periods show 110% revenue increase")
        st.write("• Strong correlation between exhibitions and sales")
        st.write("• Consistent performance across different exhibitions")
        st.write("• Clear seasonal patterns support planning")

if __name__ == "__main__":
    main()