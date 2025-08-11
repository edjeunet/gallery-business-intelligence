"""
Gallery Business Intelligence - Data Generation
Generates realistic gallery transaction data for BI analysis
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class GalleryDataGenerator:
    def __init__(self):
        self.start_date = datetime(2024, 1, 1)
        self.end_date = datetime(2024, 12, 31)
        
        # Product categories with realistic pricing
        self.products = {
            'Artist Monographs': {'min_price': 25, 'max_price': 80, 'margin': 0.45},
            'Exhibition Catalogues': {'min_price': 15, 'max_price': 45, 'margin': 0.40},
            'Limited Edition Prints': {'min_price': 150, 'max_price': 800, 'margin': 0.65},
            'Merchandise': {'min_price': 10, 'max_price': 35, 'margin': 0.55}
        }
        
        # Customer segments with behavior patterns
        self.customer_segments = {
            'Collectors': {'frequency': 0.15, 'avg_spend': 400, 'repeat_rate': 0.8},
            'Students': {'frequency': 0.35, 'avg_spend': 45, 'repeat_rate': 0.3},
            'Tourists': {'frequency': 0.35, 'avg_spend': 85, 'repeat_rate': 0.1},
            'Institutional': {'frequency': 0.15, 'avg_spend': 250, 'repeat_rate': 0.6}
        }
        
        # Exhibition schedule (2 peaks: spring/fall)
        self.exhibitions = [
            {'name': 'Modern Masters', 'start': '2024-03-15', 'end': '2024-05-15'},
            {'name': 'Contemporary Voices', 'start': '2024-09-01', 'end': '2024-11-01'}
        ]
    
    def generate_customers(self, n_customers=800):
        """Generate customer database"""
        customers = []
        
        for i in range(n_customers):
            # Assign segment based on frequency
            segment = np.random.choice(
                list(self.customer_segments.keys()),
                p=[0.15, 0.35, 0.35, 0.15]
            )
            
            customer = {
                'customer_id': f'CUST_{i+1:04d}',
                'segment': segment,
                'registration_date': self.start_date + timedelta(
                    days=np.random.randint(0, 365)
                ),
                'country': np.random.choice(['Switzerland', 'Germany', 'France', 'Italy', 'USA'], 
                                          p=[0.4, 0.2, 0.15, 0.15, 0.1])
            }
            customers.append(customer)
        
        return pd.DataFrame(customers)
    
    def generate_products(self, n_products=200):
        """Generate product catalog"""
        products = []
        
        for i, (category, details) in enumerate(self.products.items()):
            # Number of products per category
            n_cat = n_products // 4
            
            for j in range(n_cat):
                product = {
                    'product_id': f'PROD_{category[:3].upper()}_{j+1:03d}',
                    'category': category,
                    'price': round(np.random.uniform(
                        details['min_price'], details['max_price']
                    ), 2),
                    'margin': details['margin'],
                    'stock_level': np.random.randint(5, 100)
                }
                products.append(product)
        
        return pd.DataFrame(products)
    
    def generate_transactions(self, customers_df, products_df, n_transactions=3000):
        """Generate transaction history with seasonal patterns"""
        transactions = []
        
        for i in range(n_transactions):
            # Select customer
            customer = customers_df.sample(1).iloc[0]
            
            # Generate transaction date with seasonal bias
            if np.random.random() < 0.3:  # 30% during exhibitions
                # Exhibition periods
                if np.random.random() < 0.5:
                    # Spring exhibition
                    trans_date = datetime(2024, 3, 15) + timedelta(
                        days=np.random.randint(0, 61)
                    )
                else:
                    # Fall exhibition
                    trans_date = datetime(2024, 9, 1) + timedelta(
                        days=np.random.randint(0, 61)
                    )
            else:
                # Regular periods
                trans_date = self.start_date + timedelta(
                    days=np.random.randint(0, 365)
                )
            
            # Select products based on customer segment
            segment_prefs = self._get_segment_preferences(customer['segment'])
            product = products_df[
                products_df['category'].isin(segment_prefs)
            ].sample(1).iloc[0]
            
            # Quantity based on segment
            if customer['segment'] == 'Institutional':
                quantity = np.random.randint(1, 5)
            else:
                quantity = 1 if np.random.random() < 0.8 else 2
            
            transaction = {
                'transaction_id': f'TXN_{i+1:06d}',
                'customer_id': customer['customer_id'],
                'product_id': product['product_id'],
                'transaction_date': trans_date,
                'quantity': quantity,
                'unit_price': product['price'],
                'total_amount': round(product['price'] * quantity, 2),
                'channel': np.random.choice(['In-store', 'Online'], p=[0.7, 0.3])
            }
            transactions.append(transaction)
        
        return pd.DataFrame(transactions)
    
    def _get_segment_preferences(self, segment):
        """Define product preferences by customer segment"""
        preferences = {
            'Collectors': ['Limited Edition Prints', 'Artist Monographs'],
            'Students': ['Exhibition Catalogues', 'Merchandise'],
            'Tourists': ['Merchandise', 'Exhibition Catalogues', 'Artist Monographs'],
            'Institutional': ['Artist Monographs', 'Exhibition Catalogues']
        }
        return preferences.get(segment, list(self.products.keys()))
    
    def save_to_sqlite(self, customers_df, products_df, transactions_df, db_path='data/raw/gallery_data.db'):
        """Save all data to SQLite database"""
        conn = sqlite3.connect(db_path)
        
        customers_df.to_sql('customers', conn, if_exists='replace', index=False)
        products_df.to_sql('products', conn, if_exists='replace', index=False)
        transactions_df.to_sql('transactions', conn, if_exists='replace', index=False)
        
        conn.close()
        print(f"Data saved to {db_path}")
    
    def generate_all_data(self):
        """Generate complete dataset"""
        print("Generating gallery business intelligence dataset...")
        
        # Generate data
        customers = self.generate_customers()
        products = self.generate_products()
        transactions = self.generate_transactions(customers, products)
        
        # Save to database
        self.save_to_sqlite(customers, products, transactions)
        
        # Save to CSV for easy access
        customers.to_csv('data/raw/customers.csv', index=False)
        products.to_csv('data/raw/products.csv', index=False)
        transactions.to_csv('data/raw/transactions.csv', index=False)
        
        print(f"Generated:")
        print(f"- {len(customers)} customers")
        print(f"- {len(products)} products")
        print(f"- {len(transactions)} transactions")
        
        return customers, products, transactions

if __name__ == "__main__":
    generator = GalleryDataGenerator()
    customers, products, transactions = generator.generate_all_data()