# Gallery Business Intelligence 🎨

A comprehensive business intelligence dashboard for art gallery operations, featuring customer segmentation, exhibition impact analysis, and inventory optimization using Python, Streamlit, and machine learning.

## 🎯 Core Business Problems Solved

1. **Customer Lifetime Value Prediction**: Distinguish collectors vs. casual buyers for targeted marketing strategies
2. **Exhibition Impact Forecasting**: Quantify sales correlation with exhibition openings (+110% revenue uplift identified)
3. **Inventory Optimization**: Optimize stock levels across product categories using ABC analysis

## 📊 Key Insights Discovered

- **Exhibition Impact**: 110% revenue increase during exhibition periods vs baseline
- **Customer Segmentation**: 117 VIP Collectors identified (15% of customer base driving premium revenue)
- **Product Performance**: Limited Edition Prints generate €145K revenue (51% of total sales)
- **Customer Behavior**: Clear patterns between customer segments and purchasing preferences

## 🛠 Tech Stack

- **Data Processing**: pandas, numpy, scikit-learn
- **Visualization**: Streamlit, Plotly
- **Analytics**: RFM analysis, K-means clustering, Prophet forecasting
- **Storage**: SQLite
- **Deployment**: Streamlit Cloud ready

## 📁 Project Structure

```
gallery-business-intelligence/
├── data/
│   ├── raw/                 # Generated gallery transaction data
│   └── processed/           # Cleaned and transformed datasets
├── src/
│   ├── data_generation.py   # Realistic gallery data simulation
│   ├── analytics.py         # RFM analysis, clustering, forecasting
│   └── utils.py            # Helper functions
├── dashboard/
│   └── app.py              # Multi-page Streamlit dashboard
├── notebooks/              # Exploratory data analysis
├── docs/                   # Documentation and insights
└── requirements.txt        # Python dependencies
```

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/edjeunet/gallery-business-intelligence.git
cd gallery-business-intelligence
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Data
```bash
python src/data_generation.py
```

### 4. Run Analytics
```bash
python src/analytics.py
```

### 5. Launch Dashboard
```bash
streamlit run dashboard/app.py
```

## 📈 Dashboard Features

### 🎯 Executive Overview
- **KPI Metrics**: Revenue, customers, exhibition uplift, average customer value
- **Monthly Trends**: Revenue performance with exhibition markers
- **Category Analysis**: Revenue distribution across product types
- **Customer Segments**: Distribution and revenue by segment

### 👥 Customer Intelligence
- **RFM Analysis**: Recency, Frequency, Monetary value segmentation
- **K-means Clustering**: Data-driven customer grouping
- **Customer Lifetime Value**: Segment-based CLV analysis
- **Top Customers**: VIP identification and characteristics

### 📦 Inventory Performance
- **ABC Classification**: Product category prioritization
- **Sell-through Analysis**: Inventory turnover by category
- **Margin Analysis**: Profitability insights across products
- **Performance Metrics**: Revenue, quantity, and margin tracking

### 🎭 Exhibition Impact
- **Sales Correlation**: Exhibition vs baseline performance
- **ROI Analysis**: Exhibition return on investment
- **Daily Performance**: Sales trends around exhibition periods
- **Impact Quantification**: 110% revenue uplift measurement

## 🔍 Methodology

### Data Generation
- **3,000 transactions** across 12 months of simulated gallery operations
- **4 customer segments**: Collectors, Students, Tourists, Institutional buyers
- **4 product categories**: Artist monographs (€25-80), exhibition catalogues (€15-45), limited prints (€150-800), merchandise (€10-35)
- **Seasonal patterns**: 2 exhibition peaks (spring/fall) with realistic baseline

### Analytics Approach
- **RFM Analysis**: Gallery-adapted recency, frequency, monetary segmentation
- **Customer Clustering**: K-means with optimal cluster selection via silhouette analysis
- **Exhibition Impact**: Time series analysis with baseline comparison
- **Inventory Optimization**: ABC analysis weighted by margin contribution

### Machine Learning
- **Customer Segmentation**: K-means clustering with standardized features
- **Forecasting**: Prophet model ready for time series prediction
- **Classification**: Rule-based customer value classification system

## 📊 Sample Insights

### Customer Segments Identified
- **VIP Collectors** (117 customers): High value, recent, frequent purchasers
- **Active Collectors** (105 customers): High monetary value, good recency
- **Regular Visitors** (117 customers): Frequent but lower spend
- **Tourists** (256 customers): Low frequency, varied spend patterns

### Exhibition Performance
- **Baseline Daily Average**: €572
- **Exhibition Daily Average**: €1,202
- **Revenue Uplift**: 110% increase
- **Total Exhibition Revenue**: €149K over exhibition periods

### Product Category Performance
1. **Limited Edition Prints**: €145K revenue (51% of total)
2. **Artist Monographs**: Premium category with strong margins
3. **Exhibition Catalogues**: High volume, education-focused
4. **Merchandise**: Consistent baseline revenue

## 🎯 Business Recommendations

### Customer Strategy
- **VIP Program**: Target 117 identified collectors with exclusive previews
- **Student Engagement**: Optimize discount timing for 57 student customers
- **Tourist Optimization**: Focus on popular artists and shipping efficiency

### Inventory Management
- **A-Class Focus**: Prioritize Limited Edition Prints inventory
- **Seasonal Planning**: Stock buildup before spring/fall exhibitions
- **Margin Optimization**: Balance volume and profitability across categories

### Exhibition Planning
- **ROI Proven**: 110% uplift justifies exhibition investment
- **Timing Strategy**: Maintain spring/fall schedule for optimal impact
- **Marketing Integration**: Coordinate inventory and promotion timing

## 🔮 Future Enhancements

- **Predictive Models**: Customer churn prediction and CLV forecasting
- **Recommendation Engine**: Personalized product suggestions by segment
- **Real-time Integration**: Live data feeds and automated reporting
- **Advanced Analytics**: Market basket analysis and price optimization

## 📱 Live Demo

[View Live Dashboard]([https://your-streamlit-app-url.com](https://gallery-business-intelligence-edjeunet.streamlit.app/)) *(Deploy to Streamlit Cloud for public access)*

## 👤 Author

**edjeunet** - [GitHub](https://github.com/edjeunet)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*This project demonstrates end-to-end business intelligence development, from data generation through advanced analytics to interactive dashboard creation, showcasing practical skills in data science, machine learning, and business insight generation.*
