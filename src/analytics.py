"""
Gallery Business Intelligence - Analytics Module
RFM analysis, customer segmentation, and exhibition impact analysis
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class GalleryAnalytics:
    def __init__(self, db_path='data/raw/gallery_data.db'):
        self.db_path = db_path
        self.customers = None
        self.products = None
        self.transactions = None
        self.rfm_data = None
        self.customer_segments = None
        
    def load_data(self):
        """Load data from SQLite database"""
        conn = sqlite3.connect(self.db_path)
        
        self.customers = pd.read_sql('SELECT * FROM customers', conn)
        self.products = pd.read_sql('SELECT * FROM products', conn)
        self.transactions = pd.read_sql('SELECT * FROM transactions', conn)
        
        # Convert date columns
        self.customers['registration_date'] = pd.to_datetime(self.customers['registration_date'])
        self.transactions['transaction_date'] = pd.to_datetime(self.transactions['transaction_date'])
        
        conn.close()
        print("Data loaded successfully")
        
    def calculate_rfm(self, reference_date=None):
        """Calculate RFM (Recency, Frequency, Monetary) analysis for gallery context"""
        if reference_date is None:
            reference_date = self.transactions['transaction_date'].max()
        
        # Customer transaction summary
        customer_summary = self.transactions.groupby('customer_id').agg({
            'transaction_date': ['max', 'count'],
            'total_amount': 'sum'
        }).round(2)
        
        # Flatten column names
        customer_summary.columns = ['last_purchase', 'frequency', 'monetary']
        
        # Calculate recency (days since last purchase)
        customer_summary['recency'] = (reference_date - customer_summary['last_purchase']).dt.days
        
        # Add customer segment information
        customer_summary = customer_summary.merge(
            self.customers[['customer_id', 'segment']], 
            left_index=True, 
            right_on='customer_id'
        ).set_index('customer_id')
        
        # Calculate RFM scores (1-5 scale)
        customer_summary['R_score'] = pd.qcut(customer_summary['recency'], 5, labels=[5,4,3,2,1])
        customer_summary['F_score'] = pd.qcut(customer_summary['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
        customer_summary['M_score'] = pd.qcut(customer_summary['monetary'], 5, labels=[1,2,3,4,5])
        
        # Convert to numeric
        customer_summary['R_score'] = customer_summary['R_score'].astype(int)
        customer_summary['F_score'] = customer_summary['F_score'].astype(int)
        customer_summary['M_score'] = customer_summary['M_score'].astype(int)
        
        # Calculate RFM composite score
        customer_summary['RFM_score'] = (
            customer_summary['R_score'].astype(str) + 
            customer_summary['F_score'].astype(str) + 
            customer_summary['M_score'].astype(str)
        )
        
        # Gallery-specific customer classification
        customer_summary['customer_value'] = self._classify_gallery_customers(customer_summary)
        
        self.rfm_data = customer_summary
        return customer_summary
    
    def _classify_gallery_customers(self, rfm_df):
        """Classify customers based on gallery-specific RFM patterns"""
        def classify_customer(row):
            r, f, m = row['R_score'], row['F_score'], row['M_score']
            
            # VIP Collectors (high value, recent, frequent)
            if r >= 4 and f >= 4 and m >= 4:
                return 'VIP Collectors'
            
            # Active Collectors (high monetary, good recency)
            elif r >= 3 and m >= 4:
                return 'Active Collectors'
            
            # Potential Collectors (high monetary, but not recent)
            elif m >= 4 and r < 3:
                return 'Potential Collectors'
            
            # Regular Visitors (frequent but lower spend)
            elif f >= 4 and m < 4:
                return 'Regular Visitors'
            
            # Tourists (low frequency, varied spend)
            elif f <= 2:
                return 'Tourists'
            
            # Students (frequent, low spend)
            elif f >= 3 and m <= 2:
                return 'Students'
            
            # At Risk (good value but not recent)
            elif m >= 3 and r <= 2:
                return 'At Risk'
            
            else:
                return 'Casual Buyers'
        
        return rfm_df.apply(classify_customer, axis=1)
    
    def customer_clustering(self, n_clusters=4):
        """Perform K-means clustering on customer data"""
        if self.rfm_data is None:
            self.calculate_rfm()
        
        # Prepare features for clustering
        features = ['recency', 'frequency', 'monetary']
        X = self.rfm_data[features].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Find optimal number of clusters using silhouette score
        if n_clusters is None:
            silhouette_scores = []
            cluster_range = range(2, 8)
            
            for n in cluster_range:
                kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_scaled)
                silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                silhouette_scores.append(silhouette_avg)
            
            # Choose optimal clusters
            optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
            print(f"Optimal number of clusters: {optimal_clusters}")
        else:
            optimal_clusters = n_clusters
        
        # Perform final clustering
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
        self.rfm_data['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Add cluster centers for interpretation
        cluster_centers = pd.DataFrame(
            scaler.inverse_transform(kmeans.cluster_centers_),
            columns=features
        )
        cluster_centers['cluster'] = range(optimal_clusters)
        
        self.customer_segments = cluster_centers
        return self.rfm_data, cluster_centers
    
    def exhibition_impact_analysis(self):
        """Analyze sales impact around exhibition periods"""
        # Define exhibition periods
        exhibitions = [
            {'name': 'Modern Masters', 'start': '2024-03-15', 'end': '2024-05-15'},
            {'name': 'Contemporary Voices', 'start': '2024-09-01', 'end': '2024-11-01'}
        ]
        
        # Daily sales aggregation
        daily_sales = self.transactions.groupby('transaction_date').agg({
            'total_amount': 'sum',
            'transaction_id': 'count'
        }).reset_index()
        daily_sales.columns = ['date', 'revenue', 'transaction_count']
        
        # Add exhibition indicators
        daily_sales['exhibition_period'] = False
        daily_sales['exhibition_name'] = None
        
        for exhibition in exhibitions:
            start_date = pd.to_datetime(exhibition['start'])
            end_date = pd.to_datetime(exhibition['end'])
            
            mask = (daily_sales['date'] >= start_date) & (daily_sales['date'] <= end_date)
            daily_sales.loc[mask, 'exhibition_period'] = True
            daily_sales.loc[mask, 'exhibition_name'] = exhibition['name']
        
        # Calculate baseline vs exhibition performance
        baseline_avg = daily_sales[~daily_sales['exhibition_period']]['revenue'].mean()
        exhibition_avg = daily_sales[daily_sales['exhibition_period']]['revenue'].mean()
        
        impact_metrics = {
            'baseline_daily_avg': round(baseline_avg, 2),
            'exhibition_daily_avg': round(exhibition_avg, 2),
            'uplift_percent': round(((exhibition_avg - baseline_avg) / baseline_avg) * 100, 1),
            'total_exhibition_revenue': daily_sales[daily_sales['exhibition_period']]['revenue'].sum()
        }
        
        return daily_sales, impact_metrics
    
    def product_performance_analysis(self):
        """Analyze product category performance and ABC classification"""
        # Product-level analysis
        product_analysis = self.transactions.groupby('product_id').agg({
            'quantity': 'sum',
            'total_amount': 'sum',
            'transaction_id': 'count'
        }).reset_index()
        product_analysis.columns = ['product_id', 'total_quantity', 'total_revenue', 'transaction_count']
        
        # Add product details
        product_analysis = product_analysis.merge(self.products, on='product_id')
        
        # Calculate margins
        product_analysis['total_margin'] = product_analysis['total_revenue'] * product_analysis['margin']
        
        # Category-level analysis
        category_analysis = product_analysis.groupby('category').agg({
            'total_quantity': 'sum',
            'total_revenue': 'sum',
            'total_margin': 'sum',
            'product_id': 'count'
        }).reset_index()
        category_analysis.columns = ['category', 'total_quantity', 'total_revenue', 'total_margin', 'product_count']
        
        # Calculate sell-through rates (simplified)
        category_analysis['avg_stock'] = product_analysis.groupby('category')['stock_level'].mean().values
        category_analysis['sell_through_rate'] = (category_analysis['total_quantity'] / 
                                                category_analysis['avg_stock']).round(2)
        
        # ABC Classification (based on revenue contribution)
        total_revenue = category_analysis['total_revenue'].sum()
        category_analysis['revenue_contribution'] = (category_analysis['total_revenue'] / total_revenue * 100).round(1)
        category_analysis = category_analysis.sort_values('revenue_contribution', ascending=False)
        
        # Assign ABC classes
        cumulative_contribution = category_analysis['revenue_contribution'].cumsum()
        category_analysis['abc_class'] = pd.cut(
            cumulative_contribution, 
            bins=[0, 70, 90, 100], 
            labels=['A', 'B', 'C']
        )
        
        return product_analysis, category_analysis
    
    def generate_insights_summary(self):
        """Generate executive summary of key insights"""
        if self.rfm_data is None:
            self.calculate_rfm()
        
        # Customer insights
        customer_value_dist = self.rfm_data['customer_value'].value_counts()
        top_customers = self.rfm_data.nlargest(10, 'monetary')[['monetary', 'frequency', 'customer_value']]
        
        # Exhibition impact
        daily_sales, exhibition_impact = self.exhibition_impact_analysis()
        
        # Product performance
        product_analysis, category_analysis = self.product_performance_analysis()
        
        insights = {
            'customer_insights': {
                'total_customers': len(self.rfm_data),
                'vip_collectors': customer_value_dist.get('VIP Collectors', 0),
                'at_risk_customers': customer_value_dist.get('At Risk', 0),
                'avg_customer_value': round(self.rfm_data['monetary'].mean(), 2),
                'top_customer_value': round(self.rfm_data['monetary'].max(), 2)
            },
            'exhibition_impact': exhibition_impact,
            'product_insights': {
                'best_category': category_analysis.iloc[0]['category'],
                'best_category_revenue': round(category_analysis.iloc[0]['total_revenue'], 2),
                'highest_margin_category': category_analysis.loc[category_analysis['total_margin'].idxmax(), 'category'],
                'total_revenue': round(self.transactions['total_amount'].sum(), 2)
            }
        }
        
        return insights

# Visualization helper functions
def plot_rfm_distribution(rfm_data):
    """Create RFM distribution plots"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Recency Distribution', 'Frequency Distribution', 
                       'Monetary Distribution', 'Customer Value Segments'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Recency
    fig.add_trace(go.Histogram(x=rfm_data['recency'], name='Recency (days)'), row=1, col=1)
    
    # Frequency
    fig.add_trace(go.Histogram(x=rfm_data['frequency'], name='Frequency'), row=1, col=2)
    
    # Monetary
    fig.add_trace(go.Histogram(x=rfm_data['monetary'], name='Monetary (€)'), row=2, col=1)
    
    # Customer value segments
    value_counts = rfm_data['customer_value'].value_counts()
    fig.add_trace(go.Bar(x=value_counts.index, y=value_counts.values, name='Customer Segments'), row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False, title_text="Gallery Customer RFM Analysis")
    return fig

if __name__ == "__main__":
    # Example usage
    analytics = GalleryAnalytics()
    analytics.load_data()
    
    # RFM Analysis
    rfm_results = analytics.calculate_rfm()
    print("RFM Analysis completed")
    print(rfm_results['customer_value'].value_counts())
    
    # Customer Clustering
    clustered_data, cluster_centers = analytics.customer_clustering()
    print("\nCustomer Clustering completed")
    print(cluster_centers)
    
    # Exhibition Impact
    daily_sales, exhibition_metrics = analytics.exhibition_impact_analysis()
    print(f"\nExhibition Impact: {exhibition_metrics['uplift_percent']}% revenue increase")
    
    # Generate insights
    insights = analytics.generate_insights_summary()
    print("\nKey Insights:")
    for category, metrics in insights.items():
        print(f"{category}: {metrics}")