"""
Comprehensive E-commerce Analytics System - Custom Dataset Version v4
Interactive Streamlit Dashboard with CLV, RFM, Churn Prediction, Market Basket Analysis,
Customer Segmentation, Sales Forecasting, User Behavior Analytics, and Behavioral Insights

Requirements:
pip install streamlit pandas numpy matplotlib seaborn scikit-learn mlxtend plotly

Usage:
streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="E-commerce Analytics Dashboard v4",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .header-info {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        font-family: monospace;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# EXPECTED HEADERS CONFIGURATION
# =============================================================================

EXPECTED_HEADERS = {
    'customers': {
        'description': 'Customer Information',
        'required': ['customer_id'],
        'optional': ['customer_unique_id', 'customer_zip_code_prefix', 'customer_city', 'customer_state'],
        'example': 'customer_id, customer_unique_id, customer_zip_code_prefix, customer_city, customer_state'
    },
    'orders': {
        'description': 'Order Details',
        'required': ['order_id', 'customer_id', 'order_status', 'order_purchase_timestamp'],
        'optional': ['order_approved_at', 'order_delivered_customer_date', 'order_estimated_delivery_date'],
        'example': 'order_id, customer_id, order_status, order_purchase_timestamp, order_approved_at, order_delivered_customer_date'
    },
    'order_items': {
        'description': 'Order Items/Products',
        'required': ['order_id', 'product_id', 'price'],
        'optional': ['order_item_id', 'seller_id', 'shipping_limit_date', 'freight_value'],
        'example': 'order_id, order_item_id, product_id, seller_id, price, freight_value'
    },
    'order_payments': {
        'description': 'Payment Information',
        'required': ['order_id', 'payment_value'],
        'optional': ['payment_sequential', 'payment_type', 'payment_installments'],
        'example': 'order_id, payment_sequential, payment_type, payment_installments, payment_value'
    },
    'products': {
        'description': 'Product Catalog',
        'required': ['product_id'],
        'optional': ['product_category_name', 'product_name_length', 'product_description_length', 
                    'product_photos_qty', 'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm'],
        'example': 'product_id, product_category_name, product_name_length, product_weight_g'
    },
    'product_translation': {
        'description': 'Product Category Translation (Optional)',
        'required': ['product_category_name'],
        'optional': ['product_category_name_english'],
        'example': 'product_category_name, product_category_name_english'
    },
    'reviews': {
        'description': 'Customer Reviews (Optional)',
        'required': ['order_id'],
        'optional': ['review_id', 'review_score', 'review_comment_title', 'review_comment_message', 
                    'review_creation_date', 'review_answer_timestamp'],
        'example': 'review_id, order_id, review_score, review_comment_title, review_creation_date'
    }
}

# =============================================================================
# DATA LOADING WITH CUSTOM UPLOAD
# =============================================================================

def show_expected_headers(dataset_name):
    """Display expected headers for each dataset"""
    headers = EXPECTED_HEADERS.get(dataset_name, {})
    
    st.markdown(f"**{headers.get('description', dataset_name)}**")
    
    required = headers.get('required', [])
    optional = headers.get('optional', [])
    
    st.markdown(f"🔴 **Required columns:** `{', '.join(required)}`")
    if optional:
        st.markdown(f"🟡 **Optional columns:** `{', '.join(optional[:5])}{'...' if len(optional) > 5 else ''}`")
    
    with st.expander("📋 See example format"):
        st.code(headers.get('example', ''), language='text')

def validate_dataset(df, dataset_name):
    """Validate that required columns are present"""
    headers = EXPECTED_HEADERS.get(dataset_name, {})
    required = headers.get('required', [])
    
    missing_cols = [col for col in required if col not in df.columns]
    
    if missing_cols:
        st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
        return False
    
    st.success(f"✅ All required columns found!")
    return True

def load_custom_datasets():
    """Load custom datasets via file upload"""
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.subheader("📤 Upload Your Datasets")
    st.markdown("Upload CSV files with the expected headers. **Minimum required: Customers, Orders, and Order Items**")
    
    datasets = {}
    
    # Create tabs for each dataset
    tabs = st.tabs(["👥 Customers", "📦 Orders", "🛍️ Order Items", "💳 Payments", 
                    "📦 Products", "🌐 Translations", "⭐ Reviews"])
    
    dataset_keys = ['customers', 'orders', 'order_items', 'order_payments', 
                    'products', 'product_translation', 'reviews']
    
    for tab, key in zip(tabs, dataset_keys):
        with tab:
            show_expected_headers(key)
            
            uploaded_file = st.file_uploader(
                f"Upload {key.replace('_', ' ').title()} CSV",
                type=['csv'],
                key=f"upload_{key}"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write(f"**Uploaded columns:** {', '.join(df.columns.tolist())}")
                    
                    if validate_dataset(df, key):
                        datasets[key] = df
                        st.info(f"📊 Loaded {len(df):,} rows")
                        
                        with st.expander("Preview data"):
                            st.dataframe(df.head(10))
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Check if minimum required datasets are uploaded
    required_datasets = ['customers', 'orders', 'order_items']
    has_minimum = all(key in datasets for key in required_datasets)
    
    if has_minimum:
        st.success(f"✅ Minimum required datasets uploaded! You can proceed with analysis.")
    else:
        missing = [key for key in required_datasets if key not in datasets]
        st.warning(f"⚠️ Please upload these required datasets: {', '.join(missing)}")
    
    return datasets if has_minimum else None

@st.cache_data
def preprocess_data(datasets):
    """Clean and merge datasets for analysis"""
    
    customers = datasets['customers']
    orders = datasets['orders']
    order_items = datasets['order_items']
    
    # Filter only delivered/shipped orders
    if 'order_status' in orders.columns:
        orders_clean = orders[orders['order_status'].isin(['delivered', 'shipped'])]
    else:
        orders_clean = orders
    
    # Start with order_items and orders
    transactions = order_items.merge(orders_clean, on='order_id', how='inner')
    
    # Merge customers
    transactions = transactions.merge(customers, on='customer_id', how='left')
    
    # Merge payments if available
    if 'order_payments' in datasets:
        payments = datasets['order_payments']
        payment_agg = payments.groupby('order_id').agg({
            'payment_value': 'sum',
            'payment_type': 'first'
        }).reset_index()
        transactions = transactions.merge(payment_agg, on='order_id', how='left')
    
    # Merge products if available
    if 'products' in datasets:
        products = datasets['products']
        transactions = transactions.merge(products, on='product_id', how='left')
    
    # Merge product translation if available
    if 'product_translation' in datasets and 'product_category_name' in transactions.columns:
        translation = datasets['product_translation']
        transactions = transactions.merge(translation, on='product_category_name', how='left')
    
    # Merge reviews if available
    if 'reviews' in datasets:
        reviews = datasets['reviews']
        review_agg = reviews.groupby('order_id').agg({
            'review_score': 'mean'
        }).reset_index()
        transactions = transactions.merge(review_agg, on='order_id', how='left')
    
    # Convert date columns
    date_columns = ['order_purchase_timestamp', 'order_approved_at', 
                   'order_delivered_customer_date', 'order_estimated_delivery_date']
    for col in date_columns:
        if col in transactions.columns:
            transactions[col] = pd.to_datetime(transactions[col], errors='coerce')
    
    # Calculate additional features
    if 'freight_value' not in transactions.columns:
        transactions['freight_value'] = 0
    
    if 'payment_value' in transactions.columns:
        transactions['order_total'] = transactions['payment_value']
    else:
        transactions['order_total'] = transactions['price'] + transactions['freight_value']
    
    # Time-based features
    if 'order_purchase_timestamp' in transactions.columns:
        transactions['order_month'] = transactions['order_purchase_timestamp'].dt.to_period('M')
        transactions['order_year_month'] = transactions['order_purchase_timestamp'].dt.strftime('%Y-%m')
        transactions['purchase_hour'] = transactions['order_purchase_timestamp'].dt.hour
        transactions['purchase_day'] = transactions['order_purchase_timestamp'].dt.day_name()
        transactions['purchase_month'] = transactions['order_purchase_timestamp'].dt.month_name()
        transactions['purchase_day_of_week'] = transactions['order_purchase_timestamp'].dt.dayofweek
        transactions['is_weekend'] = transactions['purchase_day_of_week'] >= 5
    
    # Behavioral features
    transactions['has_free_shipping'] = transactions['freight_value'] == 0
    transactions['freight_ratio'] = transactions['freight_value'] / (transactions['price'] + 1)
    
    return transactions

# =============================================================================
# BEHAVIORAL PSYCHOLOGY INSIGHTS
# =============================================================================

def behavioral_psychology_insights(transactions):
    """Analyze consumer behavior using psychological principles"""
    
    insights = {}
    
    # Prospect Theory Analysis
    free_shipping_effect = transactions.groupby('has_free_shipping').agg({
        'order_total': 'mean',
        'order_id': 'nunique'
    })
    
    insights['prospect_theory'] = {
        'free_shipping_orders': free_shipping_effect.loc[True, 'order_id'] if True in free_shipping_effect.index else 0,
        'paid_shipping_orders': free_shipping_effect.loc[False, 'order_id'] if False in free_shipping_effect.index else 0,
        'avg_order_free': free_shipping_effect.loc[True, 'order_total'] if True in free_shipping_effect.index else 0,
        'avg_order_paid': free_shipping_effect.loc[False, 'order_total'] if False in free_shipping_effect.index else 0
    }
    
    # Social Proof Analysis
    if 'review_score' in transactions.columns:
        high_rated = transactions[transactions['review_score'] >= 4]
        low_rated = transactions[transactions['review_score'] < 3]
        
        insights['social_proof'] = {
            'high_rated_orders': len(high_rated),
            'low_rated_orders': len(low_rated),
            'avg_high_rated_price': high_rated['price'].mean() if len(high_rated) > 0 else 0,
            'avg_low_rated_price': low_rated['price'].mean() if len(low_rated) > 0 else 0
        }
    
    # Choice Architecture
    if 'product_category_name_english' in transactions.columns:
        category_col = 'product_category_name_english'
    elif 'product_category_name' in transactions.columns:
        category_col = 'product_category_name'
    else:
        category_col = None
    
    if category_col:
        customer_choice = transactions.groupby('customer_id')[category_col].nunique()
        insights['choice_architecture'] = {
            'avg_categories': customer_choice.mean(),
            'max_categories': customer_choice.max()
        }
    
    # Temporal Patterns
    if 'purchase_hour' in transactions.columns:
        hour_patterns = transactions.groupby('purchase_hour')['order_id'].nunique()
        insights['temporal'] = {
            'peak_hours': hour_patterns.nlargest(3).index.tolist(),
            'peak_hour_orders': hour_patterns.max()
        }
    
    # Status Quo Bias
    customer_orders = transactions.groupby('customer_id')['order_id'].nunique()
    repeat_customers = customer_orders[customer_orders > 1]
    insights['cognitive_biases'] = {
        'status_quo_bias': len(repeat_customers) / len(customer_orders) if len(customer_orders) > 0 else 0,
        'total_customers': len(customer_orders),
        'repeat_customers': len(repeat_customers)
    }
    
    return insights

# =============================================================================
# CLV CALCULATION
# =============================================================================

def calculate_clv(transactions):
    """Calculate Customer Lifetime Value"""
    
    clv_data = transactions.groupby('customer_id').agg({
        'order_id': 'nunique',
        'order_total': 'sum',
        'order_purchase_timestamp': ['min', 'max']
    }).reset_index()
    
    clv_data.columns = ['customer_id', 'frequency', 'monetary', 'first_purchase', 'last_purchase']
    
    clv_data['lifespan_days'] = (clv_data['last_purchase'] - clv_data['first_purchase']).dt.days + 1
    clv_data['lifespan_days'] = clv_data['lifespan_days'].clip(lower=1)
    clv_data['avg_order_value'] = clv_data['monetary'] / clv_data['frequency']
    clv_data['purchase_frequency'] = clv_data['frequency'] / clv_data['lifespan_days']
    clv_data['clv'] = clv_data['avg_order_value'] * clv_data['purchase_frequency'] * 365
    
    return clv_data

# =============================================================================
# RFM ANALYSIS
# =============================================================================

def calculate_rfm(transactions):
    """Calculate RFM scores"""
    
    analysis_date = transactions['order_purchase_timestamp'].max() + timedelta(days=1)
    
    rfm = transactions.groupby('customer_id').agg({
        'order_purchase_timestamp': lambda x: (analysis_date - x.max()).days,
        'order_id': 'nunique',
        'order_total': 'sum'
    }).reset_index()
    
    rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
    
    # Create RFM scores
    try:
        rfm['R_score'] = pd.qcut(rfm['recency'], q=4, labels=[4,3,2,1], duplicates='drop').astype(int)
    except (ValueError, TypeError):
        rfm['R_score'] = pd.cut(rfm['recency'], bins=4, labels=[4,3,2,1]).astype(int)
    
    try:
        rfm['F_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=4, labels=[1,2,3,4], duplicates='drop').astype(int)
    except (ValueError, TypeError):
        rfm['F_score'] = pd.cut(rfm['frequency'], bins=4, labels=[1,2,3,4]).astype(int)
    
    try:
        rfm['M_score'] = pd.qcut(rfm['monetary'].rank(method='first'), q=4, labels=[1,2,3,4], duplicates='drop').astype(int)
    except (ValueError, TypeError):
        rfm['M_score'] = pd.cut(rfm['monetary'], bins=4, labels=[1,2,3,4]).astype(int)
    
    rfm['RFM_score'] = rfm['R_score'].astype(str) + rfm['F_score'].astype(str) + rfm['M_score'].astype(str)
    
    # Segment customers
    def segment_customers(row):
        try:
            r = int(row['R_score'])
            f = int(row['F_score'])
            m = int(row['M_score'])
            
            if r >= 3 and f >= 3 and m >= 3:
                return 'Champions'
            elif r >= 3 and f >= 2:
                return 'Loyal Customers'
            elif r >= 3:
                return 'Potential Loyalists'
            elif f >= 3 and m >= 3:
                return 'At Risk'
            elif r <= 2:
                return 'Hibernating'
            else:
                return 'Need Attention'
        except:
            return 'Need Attention'
    
    rfm['segment'] = rfm.apply(segment_customers, axis=1)
    
    return rfm

# =============================================================================
# CUSTOMER SEGMENTATION (K-MEANS)
# =============================================================================

def perform_customer_segmentation(transactions):
    """Perform K-Means clustering for customer segmentation"""
    
    # Prepare features
    customer_features = transactions.groupby('customer_id').agg({
        'order_id': 'nunique',
        'order_total': ['sum', 'mean'],
        'price': 'mean',
        'freight_value': 'mean'
    }).reset_index()
    
    customer_features.columns = ['customer_id', 'order_count', 'total_spent', 'avg_order_value', 'avg_price', 'avg_shipping']
    
    # Add recency
    analysis_date = transactions['order_purchase_timestamp'].max()
    recency = transactions.groupby('customer_id')['order_purchase_timestamp'].max().reset_index()
    recency['recency_days'] = (analysis_date - recency['order_purchase_timestamp']).dt.days
    customer_features = customer_features.merge(recency[['customer_id', 'recency_days']], on='customer_id')
    
    # Scale features
    features_to_scale = ['order_count', 'total_spent', 'avg_order_value', 'recency_days']
    scaler = StandardScaler()
    customer_features_scaled = customer_features.copy()
    customer_features_scaled[features_to_scale] = scaler.fit_transform(customer_features[features_to_scale])
    
    # Determine optimal clusters using elbow method
    inertias = []
    K_range = range(2, min(11, len(customer_features)//10 + 2))
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(customer_features_scaled[features_to_scale])
        inertias.append(kmeans.inertia_)
    
    # Use 4 clusters as default
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    customer_features['cluster'] = kmeans.fit_predict(customer_features_scaled[features_to_scale])
    
    # Label clusters
    cluster_summary = customer_features.groupby('cluster').agg({
        'total_spent': 'mean',
        'order_count': 'mean',
        'recency_days': 'mean'
    }).reset_index()
    
    def label_cluster(cluster_id):
        stats = cluster_summary[cluster_summary['cluster'] == cluster_id].iloc[0]
        if stats['total_spent'] > cluster_summary['total_spent'].median() and stats['order_count'] > cluster_summary['order_count'].median():
            return 'High Value'
        elif stats['recency_days'] > cluster_summary['recency_days'].median():
            return 'Dormant'
        elif stats['order_count'] <= cluster_summary['order_count'].median():
            return 'New/Low Frequency'
        else:
            return 'Medium Value'
    
    customer_features['segment_label'] = customer_features['cluster'].apply(label_cluster)
    
    return customer_features, inertias, K_range

# =============================================================================
# CHURN PREDICTION
# =============================================================================

def predict_churn(transactions, churn_days=90):
    """Predict customer churn using Random Forest"""
    
    analysis_date = transactions['order_purchase_timestamp'].max()
    
    # Create customer features
    customer_data = transactions.groupby('customer_id').agg({
        'order_id': 'nunique',
        'order_total': ['sum', 'mean', 'std'],
        'order_purchase_timestamp': ['min', 'max'],
        'freight_value': 'mean',
        'price': 'mean'
    }).reset_index()
    
    customer_data.columns = ['customer_id', 'order_count', 'total_spent', 'avg_order_value', 
                             'std_order_value', 'first_purchase', 'last_purchase', 'avg_shipping', 'avg_price']
    
    # Fill NaN in std_order_value
    customer_data['std_order_value'] = customer_data['std_order_value'].fillna(0)
    
    # Calculate features
    customer_data['recency_days'] = (analysis_date - customer_data['last_purchase']).dt.days
    customer_data['lifespan_days'] = (customer_data['last_purchase'] - customer_data['first_purchase']).dt.days + 1
    customer_data['purchase_frequency'] = customer_data['order_count'] / customer_data['lifespan_days']
    
    # Define churn (no purchase in last X days)
    customer_data['is_churned'] = (customer_data['recency_days'] > churn_days).astype(int)
    
    # Features for model
    feature_cols = ['order_count', 'total_spent', 'avg_order_value', 'std_order_value', 
                    'recency_days', 'lifespan_days', 'purchase_frequency', 'avg_shipping', 'avg_price']
    
    X = customer_data[feature_cols].fillna(0)
    y = customer_data['is_churned']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Train model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Predict churn probability for all customers
    customer_data['churn_probability'] = rf_model.predict_proba(X)[:, 1]
    customer_data['churn_risk'] = pd.cut(customer_data['churn_probability'], 
                                         bins=[0, 0.3, 0.7, 1.0],
                                         labels=['Low Risk', 'Medium Risk', 'High Risk'])
    
    return customer_data, feature_importance, rf_model, (X_test, y_test, y_pred, y_pred_proba)

# =============================================================================
# MARKET BASKET ANALYSIS
# =============================================================================

def market_basket_analysis(transactions):
    """Perform market basket analysis using Apriori algorithm"""
    
    # Get category column
    if 'product_category_name_english' in transactions.columns:
        category_col = 'product_category_name_english'
    elif 'product_category_name' in transactions.columns:
        category_col = 'product_category_name'
    else:
        return None, None
    
    # Create basket
    basket = transactions.groupby(['order_id', category_col])['order_id'].count().unstack(fill_value=0)
    basket = (basket > 0).astype(int)
    
    # Apply Apriori
    try:
        frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True, max_len=3)
        
        if len(frequent_itemsets) > 0:
            # Generate rules
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
            rules = rules.sort_values('lift', ascending=False)
            
            # Format antecedents and consequents
            rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
            
            return frequent_itemsets, rules
        else:
            return None, None
    except:
        return None, None

# =============================================================================
# SALES FORECASTING
# =============================================================================

def sales_forecasting(transactions):
    """Forecast future sales using time series analysis"""
    
    # Aggregate daily sales
    daily_sales = transactions.groupby(transactions['order_purchase_timestamp'].dt.date).agg({
        'order_total': 'sum',
        'order_id': 'nunique'
    }).reset_index()
    
    daily_sales.columns = ['date', 'revenue', 'orders']
    daily_sales['date'] = pd.to_datetime(daily_sales['date'])
    daily_sales = daily_sales.sort_values('date')
    
    # Create features
    daily_sales['day_of_week'] = daily_sales['date'].dt.dayofweek
    daily_sales['day_of_month'] = daily_sales['date'].dt.day
    daily_sales['month'] = daily_sales['date'].dt.month
    daily_sales['days_from_start'] = (daily_sales['date'] - daily_sales['date'].min()).dt.days
    
    # Rolling averages
    daily_sales['revenue_ma7'] = daily_sales['revenue'].rolling(window=7, min_periods=1).mean()
    daily_sales['revenue_ma30'] = daily_sales['revenue'].rolling(window=30, min_periods=1).mean()
    
    # Prepare training data
    feature_cols = ['day_of_week', 'day_of_month', 'month', 'days_from_start', 'revenue_ma7', 'revenue_ma30']
    X = daily_sales[feature_cols].fillna(method='bfill').fillna(method='ffill')
    y = daily_sales['revenue']
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train model
    gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    gbm.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = gbm.predict(X_train)
    y_pred_test = gbm.predict(X_test)
    
    # Metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    # Future predictions (next 30 days)
    last_date = daily_sales['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30)
    
    future_df = pd.DataFrame({
        'date': future_dates,
        'day_of_week': future_dates.dayofweek,
        'day_of_month': future_dates.day,
        'month': future_dates.month,
        'days_from_start': [(d - daily_sales['date'].min()).days for d in future_dates]
    })
    
    # Use last known moving averages
    future_df['revenue_ma7'] = daily_sales['revenue_ma7'].iloc[-1]
    future_df['revenue_ma30'] = daily_sales['revenue_ma30'].iloc[-1]
    
    future_predictions = gbm.predict(future_df[feature_cols])
    future_df['predicted_revenue'] = future_predictions
    
    return daily_sales, future_df, (train_r2, test_r2, test_rmse), (X_test, y_test, y_pred_test)

# =============================================================================
# MAIN STREAMLIT APP
# =============================================================================

def main():
    st.markdown('<h1 class="main-header">📊 E-commerce Analytics Dashboard v4</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔍 Navigation")
    st.sidebar.markdown("---")
    
    # Check if data is already loaded in session state
    if 'transactions' not in st.session_state:
        st.info("👋 Welcome! Please upload your datasets to begin analysis.")
        
        datasets = load_custom_datasets()
        
        if datasets:
            if st.button("🚀 Process Data and Start Analysis", type="primary"):
                with st.spinner("Processing data..."):
                    transactions = preprocess_data(datasets)
                    st.session_state.transactions = transactions
                    st.session_state.datasets = datasets
                    st.rerun()
        else:
            st.stop()
    else:
        transactions = st.session_state.transactions
        
        st.sidebar.success(f"✅ Loaded {len(transactions):,} transactions")
        
        if st.sidebar.button("🔄 Upload New Data"):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Navigation menu
    analysis_type = st.sidebar.selectbox(
        "Select Analysis",
        ["📈 Overview Dashboard", 
         "🧠 Behavioral Insights", 
         "💰 Customer Lifetime Value",
         "🎯 RFM Analysis",
         "👥 Customer Segmentation",
         "⚠️ Churn Prediction",
         "🛒 Market Basket Analysis",
         "📈 Sales Forecasting",
         "📊 Sales Analytics",
         "⭐ Product Analytics"]
    )
    
    # =============================================================================
    # OVERVIEW DASHBOARD
    # =============================================================================
    
    if analysis_type == "📈 Overview Dashboard":
        st.header("📈 Business Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_revenue = transactions['order_total'].sum()
            st.metric("Total Revenue", f"${total_revenue:,.2f}")
        
        with col2:
            total_orders = transactions['order_id'].nunique()
            st.metric("Total Orders", f"{total_orders:,}")
        
        with col3:
            total_customers = transactions['customer_id'].nunique()
            st.metric("Total Customers", f"{total_customers:,}")
        
        with col4:
            avg_order_value = transactions['order_total'].mean()
            st.metric("Avg Order Value", f"${avg_order_value:.2f}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'order_year_month' in transactions.columns:
                st.subheader("📅 Sales Trend Over Time")
                sales_trend = transactions.groupby('order_year_month')['order_total'].sum().reset_index()
                fig = px.line(sales_trend, x='order_year_month', y='order_total',
                             labels={'order_year_month': 'Month', 'order_total': 'Revenue ($)'})
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            category_col = None
            if 'product_category_name_english' in transactions.columns:
                category_col = 'product_category_name_english'
            elif 'product_category_name' in transactions.columns:
                category_col = 'product_category_name'
            
            if category_col:
                st.subheader("🏆 Top Product Categories")
                top_categories = transactions[category_col].value_counts().head(10)
                fig = px.bar(x=top_categories.values, y=top_categories.index, orientation='h',
                            labels={'x': 'Number of Orders', 'y': 'Category'})
                st.plotly_chart(fig, use_container_width=True)
        
        # Additional metrics
        st.markdown("---")
        st.subheader("📊 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            repeat_rate = (transactions.groupby('customer_id')['order_id'].nunique() > 1).mean()
            st.metric("Repeat Customer Rate", f"{repeat_rate:.1%}")
        
        with col2:
            avg_items = transactions.groupby('order_id')['product_id'].count().mean()
            st.metric("Avg Items per Order", f"{avg_items:.1f}")
        
        with col3:
            if 'review_score' in transactions.columns:
                avg_rating = transactions['review_score'].mean()
                st.metric("Avg Review Score", f"{avg_rating:.2f}/5")
        
        with col4:
            free_shipping_rate = transactions['has_free_shipping'].mean()
            st.metric("Free Shipping Rate", f"{free_shipping_rate:.1%}")
    
    # =============================================================================
    # BEHAVIORAL INSIGHTS
    # =============================================================================
    
    elif analysis_type == "🧠 Behavioral Insights":
        st.header("🧠 Consumer Behavioral Psychology Insights")
        
        with st.spinner("Analyzing behavioral patterns..."):
            insights = behavioral_psychology_insights(transactions)
        
        # Prospect Theory
        st.subheader("📊 1. Prospect Theory (Loss Aversion)")
        col1, col2 = st.columns(2)
        
        with col1:
            pt = insights['prospect_theory']
            st.metric("Free Shipping Orders", f"{pt['free_shipping_orders']:,}")
            st.metric("Avg Order (Free Shipping)", f"${pt['avg_order_free']:.2f}")
        
        with col2:
            st.metric("Paid Shipping Orders", f"{pt['paid_shipping_orders']:,}")
            st.metric("Avg Order (Paid Shipping)", f"${pt['avg_order_paid']:.2f}")
        
        # Social Proof
        if 'social_proof' in insights:
            st.subheader("⭐ 2. Social Proof & Review Impact")
            col1, col2 = st.columns(2)
            
            with col1:
                if 'review_score' in transactions.columns:
                    review_dist = transactions['review_score'].value_counts().sort_index()
                    fig = px.bar(x=review_dist.index, y=review_dist.values,
                               labels={'x': 'Review Score', 'y': 'Number of Orders'})
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                sp = insights['social_proof']
                st.metric("High-Rated Orders (≥4)", f"{sp['high_rated_orders']:,}")
                st.metric("Low-Rated Orders (<3)", f"{sp['low_rated_orders']:,}")
        
        # Temporal Patterns
        if 'purchase_hour' in transactions.columns:
            st.subheader("⏰ 3. Temporal Purchasing Patterns")
            hour_patterns = transactions.groupby('purchase_hour')['order_id'].nunique()
            fig = px.line(x=hour_patterns.index, y=hour_patterns.values,
                         labels={'x': 'Hour of Day', 'y': 'Number of Orders'},
                         markers=True)
            st.plotly_chart(fig, use_container_width=True)
        
        # Weekend vs Weekday
        if 'is_weekend' in transactions.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                weekend_data = transactions.groupby('is_weekend')['order_total'].mean()
                fig = px.bar(x=['Weekday', 'Weekend'], y=weekend_data.values,
                            labels={'x': '', 'y': 'Avg Order Value ($)'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'purchase_day' in transactions.columns:
                    day_dist = transactions['purchase_day'].value_counts()
                    fig = px.pie(values=day_dist.values, names=day_dist.index,
                                title='Orders by Day of Week')
                    st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.subheader("🎯 Actionable Recommendations")
        
        recommendations = [
            "✅ **Loss Aversion**: Highlight free shipping thresholds prominently",
            "🔥 **Scarcity Principle**: Use limited-time offers for slow-moving inventory",
            "⭐ **Social Proof**: Display review scores and testimonials prominently",
            "🎯 **Choice Architecture**: Curate 'Recommended for You' sections"
        ]
        
        if 'temporal' in insights:
            recommendations.append(f"⏰ **Temporal Patterns**: Schedule campaigns during peak hours {insights['temporal']['peak_hours']}")
        
        if 'cognitive_biases' in insights:
            recommendations.append(f"🔄 **Status Quo Bias**: {insights['cognitive_biases']['status_quo_bias']:.1%} repeat rate - implement loyalty programs")
        
        for rec in recommendations:
            st.markdown(rec)
    
    # =============================================================================
    # CLV ANALYSIS
    # =============================================================================
    
    elif analysis_type == "💰 Customer Lifetime Value":
        st.header("💰 Customer Lifetime Value Analysis")
        
        with st.spinner("Calculating CLV..."):
            clv_data = calculate_clv(transactions)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average CLV", f"${clv_data['clv'].mean():.2f}")
        
        with col2:
            st.metric("Median CLV", f"${clv_data['clv'].median():.2f}")
        
        with col3:
            st.metric("Top Customer CLV", f"${clv_data['clv'].max():.2f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("CLV Distribution")
            fig = px.histogram(clv_data, x='clv', nbins=50,
                             labels={'clv': 'Customer Lifetime Value ($)'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Top 20 Customers by CLV")
            top_clv = clv_data.nlargest(20, 'clv')
            fig = px.bar(top_clv, x=top_clv.index, y='clv',
                        labels={'x': 'Customer Rank', 'clv': 'CLV ($)'})
            st.plotly_chart(fig, use_container_width=True)
        
        # CLV Segments
        st.subheader("CLV Segmentation")
        clv_data['clv_segment'] = pd.qcut(clv_data['clv'], q=4, 
                                          labels=['Low Value', 'Medium Value', 'High Value', 'VIP'],
                                          duplicates='drop')
        
        segment_stats = clv_data.groupby('clv_segment').agg({
            'customer_id': 'count',
            'clv': 'mean',
            'monetary': 'sum'
        }).reset_index()
        segment_stats.columns = ['Segment', 'Customer Count', 'Avg CLV', 'Total Revenue']
        
        st.dataframe(segment_stats, use_container_width=True)
        
        # Download
        csv = clv_data.to_csv(index=False)
        st.download_button("📥 Download CLV Data", csv, "clv_analysis.csv", "text/csv")
    
    # =============================================================================
    # RFM ANALYSIS
    # =============================================================================
    
    elif analysis_type == "🎯 RFM Analysis":
        st.header("🎯 RFM (Recency, Frequency, Monetary) Analysis")
        
        with st.spinner("Calculating RFM scores..."):
            rfm = calculate_rfm(transactions)
        
        # Segment distribution
        st.subheader("Customer Segment Distribution")
        segment_counts = rfm['segment'].value_counts()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.pie(values=segment_counts.values, names=segment_counts.index,
                        title='Customer Segments')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(segment_counts.reset_index(), use_container_width=True)
        
        # Segment analysis
        st.subheader("Segment Performance Metrics")
        segment_stats = rfm.groupby('segment').agg({
            'customer_id': 'count',
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'mean'
        }).reset_index()
        segment_stats.columns = ['Segment', 'Customer Count', 'Avg Recency (days)', 'Avg Frequency', 'Avg Monetary ($)']
        
        st.dataframe(segment_stats, use_container_width=True)
        
        # RFM scatter
        st.subheader("RFM Relationship Analysis")
        fig = px.scatter(rfm, x='frequency', y='monetary', color='segment',
                        size='recency', hover_data=['customer_id'],
                        labels={'frequency': 'Frequency', 'monetary': 'Monetary Value ($)'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Download
        csv = rfm.to_csv(index=False)
        st.download_button("📥 Download RFM Analysis", csv, "rfm_analysis.csv", "text/csv")
    
    # =============================================================================
    # CUSTOMER SEGMENTATION
    # =============================================================================
    
    elif analysis_type == "👥 Customer Segmentation":
        st.header("👥 Customer Segmentation (K-Means Clustering)")
        
        with st.spinner("Performing customer segmentation..."):
            customer_features, inertias, K_range = perform_customer_segmentation(transactions)
        
        # Elbow plot
        st.subheader("📊 Optimal Number of Clusters (Elbow Method)")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.line(x=list(K_range), y=inertias, markers=True,
                         labels={'x': 'Number of Clusters (K)', 'y': 'Inertia'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.info("The 'elbow' point suggests the optimal number of clusters. We're using K=4 for this analysis.")
        
        # Segment distribution
        st.subheader("Customer Segment Distribution")
        segment_dist = customer_features['segment_label'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(values=segment_dist.values, names=segment_dist.index,
                        title='Segment Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Segment characteristics
            segment_chars = customer_features.groupby('segment_label').agg({
                'total_spent': 'mean',
                'order_count': 'mean',
                'recency_days': 'mean',
                'customer_id': 'count'
            }).reset_index()
            segment_chars.columns = ['Segment', 'Avg Spent ($)', 'Avg Orders', 'Avg Recency (days)', 'Count']
            st.dataframe(segment_chars, use_container_width=True)
        
        # 3D Scatter plot
        st.subheader("3D Visualization of Customer Segments")
        fig = px.scatter_3d(customer_features, x='order_count', y='total_spent', z='recency_days',
                           color='segment_label', hover_data=['customer_id'],
                           labels={'order_count': 'Order Count', 'total_spent': 'Total Spent ($)', 
                                  'recency_days': 'Recency (days)'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Download
        csv = customer_features.to_csv(index=False)
        st.download_button("📥 Download Segmentation Data", csv, "customer_segments.csv", "text/csv")
    
    # =============================================================================
    # CHURN PREDICTION
    # =============================================================================
    
    elif analysis_type == "⚠️ Churn Prediction":
        st.header("⚠️ Customer Churn Prediction")
        
        churn_threshold = st.slider("Define Churn Threshold (days of inactivity)", 30, 180, 90)
        
        with st.spinner("Training churn prediction model..."):
            customer_data, feature_importance, model, test_data = predict_churn(transactions, churn_threshold)
        
        X_test, y_test, y_pred, y_pred_proba = test_data
        
        # Model performance
        st.subheader("📊 Model Performance")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            accuracy = (y_pred == y_test).mean()
            st.metric("Accuracy", f"{accuracy:.2%}")
        
        with col2:
            churn_rate = customer_data['is_churned'].mean()
            st.metric("Overall Churn Rate", f"{churn_rate:.2%}")
        
        with col3:
            high_risk = (customer_data['churn_risk'] == 'High Risk').sum()
            st.metric("High Risk Customers", f"{high_risk:,}")
        
        # Feature importance
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Feature Importance")
            fig = px.bar(feature_importance.head(10), x='importance', y='feature', orientation='h',
                        labels={'importance': 'Importance', 'feature': 'Feature'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Churn Risk Distribution")
            risk_dist = customer_data['churn_risk'].value_counts()
            fig = px.pie(values=risk_dist.values, names=risk_dist.index)
            st.plotly_chart(fig, use_container_width=True)
        
        # High risk customers
        st.subheader("🚨 High Risk Customers (Top 20)")
        high_risk_customers = customer_data[customer_data['churn_risk'] == 'High Risk'].nlargest(20, 'churn_probability')
        high_risk_display = high_risk_customers[['customer_id', 'churn_probability', 'recency_days', 
                                                  'order_count', 'total_spent']].copy()
        high_risk_display.columns = ['Customer ID', 'Churn Probability', 'Days Since Last Order', 
                                     'Total Orders', 'Total Spent ($)']
        st.dataframe(high_risk_display, use_container_width=True)
        
        # Recommendations
        st.subheader("💡 Churn Prevention Strategies")
        st.markdown("""
        - 🎁 **Win-back campaigns**: Target high-risk customers with special offers
        - 📧 **Re-engagement emails**: Send personalized recommendations based on past purchases
        - 💰 **Loyalty rewards**: Offer points or discounts to encourage repeat purchases
        - 📞 **Proactive outreach**: Contact high-value churning customers directly
        - 🔔 **Push notifications**: Remind customers of abandoned carts or wishlist items
        """)
        
        # Download
        csv = customer_data.to_csv(index=False)
        st.download_button("📥 Download Churn Predictions", csv, "churn_predictions.csv", "text/csv")
    
    # =============================================================================
    # MARKET BASKET ANALYSIS
    # =============================================================================
    
    elif analysis_type == "🛒 Market Basket Analysis":
        st.header("🛒 Market Basket Analysis")
        
        with st.spinner("Analyzing product associations..."):
            frequent_itemsets, rules = market_basket_analysis(transactions)
        
        if rules is not None and len(rules) > 0:
            st.subheader("🔍 Association Rules")
            st.info(f"Found {len(rules)} association rules")
            
            # Filter controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.3, 0.05)
            
            with col2:
                min_lift = st.slider("Minimum Lift", 1.0, 10.0, 2.0, 0.5)
            
            with col3:
                top_n = st.slider("Show Top N Rules", 10, 100, 20, 10)
            
            # Filter rules
            filtered_rules = rules[
                (rules['confidence'] >= min_confidence) & 
                (rules['lift'] >= min_lift)
            ].head(top_n)
            
            # Display rules
            display_rules = filtered_rules[['antecedents', 'consequents', 'support', 
                                           'confidence', 'lift']].copy()
            display_rules.columns = ['If Customer Buys', 'They Also Buy', 'Support', 'Confidence', 'Lift']
            
            st.dataframe(display_rules, use_container_width=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Support vs Confidence")
                fig = px.scatter(filtered_rules, x='support', y='confidence', size='lift',
                               hover_data=['antecedents', 'consequents'],
                               labels={'support': 'Support', 'confidence': 'Confidence'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Top Rules by Lift")
                top_lift = filtered_rules.nlargest(10, 'lift')
                fig = px.bar(top_lift, x='lift', y=top_lift.index, orientation='h',
                           labels={'lift': 'Lift', 'index': 'Rule'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Business recommendations
            st.subheader("💡 Actionable Insights")
            st.markdown("""
            - 🎁 **Product Bundling**: Create bundles with frequently bought together items
            - 📍 **Store Layout**: Place associated products near each other
            - 🎯 **Cross-selling**: Recommend complementary products at checkout
            - 📧 **Email Marketing**: Send targeted promotions based on purchase history
            - 💼 **Inventory Management**: Stock associated products together
            """)
            
            # Download
            csv = filtered_rules.to_csv(index=False)
            st.download_button("📥 Download Association Rules", csv, "basket_analysis.csv", "text/csv")
        
        else:
            st.warning("⚠️ Not enough data to generate meaningful association rules. Try lowering the minimum support threshold or ensure you have product category data.")
    
    # =============================================================================
    # SALES FORECASTING
    # =============================================================================
    
    elif analysis_type == "📈 Sales Forecasting":
        st.header("📈 Sales Forecasting")
        
        with st.spinner("Building forecasting model..."):
            daily_sales, future_predictions, metrics, test_data = sales_forecasting(transactions)
        
        train_r2, test_r2, test_rmse = metrics
        X_test, y_test, y_pred_test = test_data
        
        # Model performance
        st.subheader("📊 Model Performance")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Train R² Score", f"{train_r2:.3f}")
        
        with col2:
            st.metric("Test R² Score", f"{test_r2:.3f}")
        
        with col3:
            st.metric("Test RMSE", f"${test_rmse:,.2f}")
        
        # Historical vs Predicted
        st.subheader("📅 Historical Sales and Forecast")
        
        # Combine historical and future
        historical_chart = daily_sales[['date', 'revenue']].copy()
        historical_chart['type'] = 'Historical'
        
        future_chart = future_predictions[['date', 'predicted_revenue']].copy()
        future_chart.columns = ['date', 'revenue']
        future_chart['type'] = 'Forecast'
        
        combined_chart = pd.concat([historical_chart, future_chart], ignore_index=True)
        
        fig = px.line(combined_chart, x='date', y='revenue', color='type',
                     labels={'date': 'Date', 'revenue': 'Revenue ($)', 'type': 'Type'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Future predictions table
        st.subheader("🔮 30-Day Revenue Forecast")
        forecast_display = future_predictions[['date', 'predicted_revenue']].copy()
        forecast_display['date'] = forecast_display['date'].dt.strftime('%Y-%m-%d')
        forecast_display.columns = ['Date', 'Predicted Revenue ($)']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(forecast_display, use_container_width=True)
        
        with col2:
            total_forecast = future_predictions['predicted_revenue'].sum()
            avg_daily = future_predictions['predicted_revenue'].mean()
            
            st.metric("Total Forecast (30 days)", f"${total_forecast:,.2f}")
            st.metric("Avg Daily Revenue", f"${avg_daily:,.2f}")
        
        # Download
        csv = forecast_display.to_csv(index=False)
        st.download_button("📥 Download Forecast", csv, "sales_forecast.csv", "text/csv")
    
    # =============================================================================
    # SALES ANALYTICS
    # =============================================================================
    
    elif analysis_type == "📊 Sales Analytics":
        st.header("📊 Sales Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'order_year_month' in transactions.columns:
                st.subheader("Monthly Sales Trend")
                monthly_sales = transactions.groupby('order_year_month').agg({
                    'order_total': 'sum',
                    'order_id': 'nunique'
                }).reset_index()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=monthly_sales['order_year_month'], 
                                        y=monthly_sales['order_total'],
                                        mode='lines+markers', name='Revenue'))
                fig.update_layout(title='Top 10 States by Orders')
                st.plotly_chart(fig, use_container_width=True)
    
    # =============================================================================
    # PRODUCT ANALYTICS
    # =============================================================================
    
    elif analysis_type == "⭐ Product Analytics":
        st.header("⭐ Product Analytics")
        
        col1, col2 = st.columns(2)
        
        category_col = None
        if 'product_category_name_english' in transactions.columns:
            category_col = 'product_category_name_english'
        elif 'product_category_name' in transactions.columns:
            category_col = 'product_category_name'
        
        with col1:
            if category_col and 'review_score' in transactions.columns:
                st.subheader("Average Review Score by Category")
                category_reviews = transactions.groupby(category_col)['review_score'].mean().sort_values(ascending=False).head(15)
                fig = px.bar(x=category_reviews.values, y=category_reviews.index, orientation='h',
                           labels={'x': 'Avg Review Score', 'y': 'Category'})
                st.plotly_chart(fig, use_container_width=True)
            elif 'review_score' in transactions.columns:
                st.subheader("Review Score Distribution")
                review_dist = transactions['review_score'].value_counts().sort_index()
                fig = px.bar(x=review_dist.index, y=review_dist.values,
                           labels={'x': 'Review Score', 'y': 'Count'})
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Price Distribution")
            fig = px.histogram(transactions, x='price', nbins=50,
                             labels={'price': 'Price ($)'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Product popularity
        st.subheader("Most Popular Products")
        product_stats = transactions.groupby('product_id').agg({
            'order_id': 'nunique',
            'price': 'mean'
        }).reset_index()
        
        if 'review_score' in transactions.columns:
            review_by_product = transactions.groupby('product_id')['review_score'].mean().reset_index()
            product_stats = product_stats.merge(review_by_product, on='product_id', how='left')
        
        product_stats.columns = ['product_id', 'orders', 'avg_price'] + (['avg_review'] if 'review_score' in transactions.columns else [])
        product_stats = product_stats.sort_values('orders', ascending=False).head(20)
        
        st.dataframe(product_stats, use_container_width=True)
        
        # Category insights
        if category_col:
            st.subheader("Category Insights")
            
            category_insights = transactions.groupby(category_col).agg({
                'order_id': 'nunique',
                'price': ['mean', 'sum'],
                'freight_value': 'mean'
            }).reset_index()
            
            category_insights.columns = ['category', 'total_orders', 'avg_price', 'total_revenue', 'avg_shipping']
            category_insights = category_insights.sort_values('total_revenue', ascending=False).head(20)
            
            st.dataframe(category_insights, use_container_width=True)
            
            # Download category insights
            csv = category_insights.to_csv(index=False)
            st.download_button(
                label="📥 Download Category Insights",
                data=csv,
                file_name="category_insights.csv",
                mime="text/csv"
            )
        
        # Product performance metrics
        st.subheader("Product Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_product_price = transactions['price'].mean()
            st.metric("Average Product Price", f"${avg_product_price:.2f}")
        
        with col2:
            total_products = transactions['product_id'].nunique()
            st.metric("Total Unique Products", f"{total_products:,}")
        
        with col3:
            avg_items_per_order = transactions.groupby('order_id')['product_id'].count().mean()
            st.metric("Avg Items per Order", f"{avg_items_per_order:.2f}")
        
        # Price vs Review correlation
        if 'review_score' in transactions.columns:
            st.subheader("Price vs Review Score Analysis")
            
            price_review = transactions.groupby('product_id').agg({
                'price': 'mean',
                'review_score': 'mean'
            }).reset_index()
            
            fig = px.scatter(price_review, x='price', y='review_score',
                           labels={'price': 'Average Price ($)', 'review_score': 'Average Review Score'},
                           trendline="ols")
            st.plotly_chart(fig, use_container_width=True)
            
            correlation = price_review['price'].corr(price_review['review_score'])
            st.info(f"💡 **Correlation between Price and Review Score:** {correlation:.3f}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <p><strong>📊 E-commerce Analytics Dashboard v4</strong> | Built with Streamlit</p>
            <p style='font-size: 0.9rem;'>Complete analytics suite with CLV, RFM, Segmentation, Churn Prediction, Market Basket Analysis & Sales Forecasting</p>
            <p style='font-size: 0.8rem; color: #999;'>Upload your custom datasets to unlock powerful insights</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()xaxis_title='Month', yaxis_title='Revenue ($)')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'payment_type' in transactions.columns:
                st.subheader("Payment Method Distribution")
                payment_dist = transactions['payment_type'].value_counts()
                fig = px.pie(values=payment_dist.values, names=payment_dist.index)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Payment type data not available")
        
        # Category performance
        category_col = None
        if 'product_category_name_english' in transactions.columns:
            category_col = 'product_category_name_english'
        elif 'product_category_name' in transactions.columns:
            category_col = 'product_category_name'
        
        if category_col:
            st.subheader("Category Performance")
            category_perf = transactions.groupby(category_col).agg({
                'order_total': 'sum',
                'order_id': 'nunique',
                'price': 'mean'
            }).reset_index().sort_values('order_total', ascending=False).head(15)
            
            fig = px.bar(category_perf, x=category_col, y='order_total',
                        labels={category_col: 'Category', 'order_total': 'Revenue ($)'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(category_perf, use_container_width=True)
        
        # Geographic analysis
        if 'customer_state' in transactions.columns:
            st.subheader("Revenue by Geographic Location")
            col1, col2 = st.columns(2)
            
            with col1:
                state_revenue = transactions.groupby('customer_state')['order_total'].sum().sort_values(ascending=False).head(10)
                fig = px.bar(x=state_revenue.values, y=state_revenue.index, orientation='h',
                           labels={'x': 'Revenue ($)', 'y': 'State'})
                fig.update_layout(title='Top 10 States by Revenue')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                state_orders = transactions.groupby('customer_state')['order_id'].nunique().sort_values(ascending=False).head(10)
                fig = px.bar(x=state_orders.values, y=state_orders.index, orientation='h',
                           labels={'x': 'Number of Orders', 'y': 'State'})
                fig.update_layout(
