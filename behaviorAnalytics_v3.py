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
            return 'Inactive'
        elif stats['order_count'] <= cluster_summary['order_count'].median():
            return 'New/Low Frequency'
        else:
            return 'Medium Value'
    
    customer_features['cluster_label'] = customer_features['cluster'].apply(label_cluster)
    
    return customer_features, inertias, list(K_range)

# =============================================================================
# CHURN PREDICTION
# =============================================================================

def predict_churn(transactions):
    """Predict customer churn using Random Forest"""
    
    analysis_date = transactions['order_purchase_timestamp'].max()
    
    # Prepare features
    customer_features = transactions.groupby('customer_id').agg({
        'order_id': 'nunique',
        'order_total': ['sum', 'mean', 'std'],
        'order_purchase_timestamp': ['min', 'max'],
        'freight_value': 'mean'
    }).reset_index()
    
    customer_features.columns = ['customer_id', 'order_count', 'total_spent', 'avg_order_value', 
                                  'std_order_value', 'first_purchase', 'last_purchase', 'avg_shipping']
    
    # Calculate features
    customer_features['recency_days'] = (analysis_date - customer_features['last_purchase']).dt.days
    customer_features['customer_age_days'] = (customer_features['last_purchase'] - customer_features['first_purchase']).dt.days + 1
    customer_features['avg_days_between_orders'] = customer_features['customer_age_days'] / customer_features['order_count']
    
    # Define churn (no purchase in last 90 days)
    customer_features['is_churned'] = (customer_features['recency_days'] > 90).astype(int)
    
    # Fill NaN values
    customer_features['std_order_value'] = customer_features['std_order_value'].fillna(0)
    
    # Features for model
    feature_cols = ['order_count', 'total_spent', 'avg_order_value', 'std_order_value', 
                    'recency_days', 'customer_age_days', 'avg_days_between_orders', 'avg_shipping']
    
    X = customer_features[feature_cols]
    y = customer_features['is_churned']
    
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
    customer_features['churn_probability'] = rf_model.predict_proba(X)[:, 1]
    customer_features['churn_risk'] = pd.cut(customer_features['churn_probability'], 
                                              bins=[0, 0.3, 0.7, 1.0], 
                                              labels=['Low', 'Medium', 'High'])
    
    return customer_features, rf_model, feature_importance, y_test, y_pred

# =============================================================================
# MARKET BASKET ANALYSIS
# =============================================================================

def market_basket_analysis(transactions):
    """Perform market basket analysis using Apriori algorithm"""
    
    # Determine category column
    if 'product_category_name_english' in transactions.columns:
        category_col = 'product_category_name_english'
    elif 'product_category_name' in transactions.columns:
        category_col = 'product_category_name'
    else:
        return None, None
    
    # Create basket
    basket = transactions.groupby(['order_id', category_col])['product_id'].count().unstack().fillna(0)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    
    # Apply Apriori
    frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
    
    if len(frequent_itemsets) == 0:
        return None, None
    
    # Generate rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    rules = rules.sort_values('lift', ascending=False)
    
    return frequent_itemsets, rules

# =============================================================================
# SALES FORECASTING
# =============================================================================

def forecast_sales(transactions):
    """Forecast future sales using Gradient Boosting"""
    
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
    daily_sales['day_num'] = (daily_sales['date'] - daily_sales['date'].min()).dt.days
    
    # Lag features
    for lag in [1, 7, 14]:
        daily_sales[f'revenue_lag_{lag}'] = daily_sales['revenue'].shift(lag)
    
    daily_sales = daily_sales.dropna()
    
    # Features for model
    feature_cols = ['day_of_week', 'day_of_month', 'month', 'day_num', 
                    'revenue_lag_1', 'revenue_lag_7', 'revenue_lag_14']
    
    X = daily_sales[feature_cols]
    y = daily_sales['revenue']
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train model
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
    gb_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = gb_model.predict(X_test)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Forecast next 30 days
    last_date = daily_sales['date'].max()
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30)
    
    forecast_df = pd.DataFrame({'date': forecast_dates})
    forecast_df['day_of_week'] = forecast_df['date'].dt.dayofweek
    forecast_df['day_of_month'] = forecast_df['date'].dt.day
    forecast_df['month'] = forecast_df['date'].dt.month
    forecast_df['day_num'] = (forecast_df['date'] - daily_sales['date'].min()).dt.days
    
    # Use last known values for lag features
    last_revenue = daily_sales['revenue'].iloc[-14:].values
    
    forecast_revenues = []
    for i in range(30):
        if i == 0:
            lag_1 = last_revenue[-1]
            lag_7 = last_revenue[-7]
            lag_14 = last_revenue[-14]
        elif i < 7:
            lag_1 = forecast_revenues[-1] if i > 0 else last_revenue[-1]
            lag_7 = last_revenue[-(7-i)]
            lag_14 = last_revenue[-(14-i)]
        elif i < 14:
            lag_1 = forecast_revenues[-1]
            lag_7 = forecast_revenues[-(7-1)] if i >= 7 else last_revenue[-(7-i)]
            lag_14 = last_revenue[-(14-i)]
        else:
            lag_1 = forecast_revenues[-1]
            lag_7 = forecast_revenues[-7]
            lag_14 = forecast_revenues[-14]
        
        forecast_df.loc[i, 'revenue_lag_1'] = lag_1
        forecast_df.loc[i, 'revenue_lag_7'] = lag_7
        forecast_df.loc[i, 'revenue_lag_14'] = lag_14
        
        pred = gb_model.predict(forecast_df.loc[[i], feature_cols])[0]
        forecast_revenues.append(pred)
    
    forecast_df['predicted_revenue'] = forecast_revenues
    
    return daily_sales, forecast_df, gb_model, rmse, r2

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_clv_distribution(clv_data):
    """Plot CLV distribution"""
    fig = px.histogram(clv_data, x='clv', nbins=50, 
                       title='Customer Lifetime Value Distribution',
                       labels={'clv': 'CLV ($)', 'count': 'Number of Customers'})
    fig.update_layout(showlegend=False)
    return fig

def plot_rfm_segments(rfm):
    """Plot RFM segment distribution"""
    segment_counts = rfm['segment'].value_counts().reset_index()
    segment_counts.columns = ['segment', 'count']
    
    fig = px.bar(segment_counts, x='segment', y='count',
                 title='Customer Segmentation by RFM',
                 labels={'segment': 'Customer Segment', 'count': 'Number of Customers'},
                 color='count', color_continuous_scale='Blues')
    return fig

def plot_cluster_3d(customer_features):
    """Plot 3D cluster visualization"""
    fig = px.scatter_3d(customer_features, 
                        x='total_spent', 
                        y='order_count', 
                        z='recency_days',
                        color='cluster_label',
                        title='Customer Segmentation - 3D View',
                        labels={'total_spent': 'Total Spent ($)', 
                               'order_count': 'Order Count',
                               'recency_days': 'Recency (Days)'})
    return fig

def plot_churn_risk(churn_data):
    """Plot churn risk distribution"""
    risk_counts = churn_data['churn_risk'].value_counts().reset_index()
    risk_counts.columns = ['risk', 'count']
    
    fig = px.pie(risk_counts, values='count', names='risk',
                 title='Customer Churn Risk Distribution',
                 color='risk',
                 color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'})
    return fig

def plot_sales_forecast(daily_sales, forecast_df):
    """Plot sales forecast"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=daily_sales['date'], y=daily_sales['revenue'],
                            mode='lines', name='Historical',
                            line=dict(color='blue')))
    
    fig.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['predicted_revenue'],
                            mode='lines', name='Forecast',
                            line=dict(color='red', dash='dash')))
    
    fig.update_layout(title='Sales Forecast - Next 30 Days',
                     xaxis_title='Date',
                     yaxis_title='Revenue ($)')
    return fig

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    st.markdown('<h1 class="main-header">🚀 E-commerce Analytics Dashboard v4</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'datasets' not in st.session_state:
        st.session_state.datasets = None
    if 'transactions' not in st.session_state:
        st.session_state.transactions = None
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    
    if st.session_state.datasets is None:
        page = "Upload Data"
    else:
        page = st.sidebar.radio(
            "Select Analysis",
            ["📈 Overview Dashboard", "🧠 Behavioral Insights", "💰 Customer Lifetime Value",
             "🎯 RFM Analysis", "👥 Customer Segmentation", "⚠️ Churn Prediction",
             "🛒 Market Basket Analysis", "📈 Sales Forecasting", "📊 Sales Analytics",
             "⭐ Product Analytics", "🔄 Upload New Data"]
        )
    
    # Upload Data Page
    if page == "Upload Data" or page == "🔄 Upload New Data" or st.session_state.datasets is None:
        st.header("📤 Data Upload")
        datasets = load_custom_datasets()
        
        if datasets is not None:
            if st.button("🚀 Process Data & Start Analysis", type="primary"):
                with st.spinner("Processing data..."):
                    st.session_state.datasets = datasets
                    st.session_state.transactions = preprocess_data(datasets)
                    st.success("✅ Data processed successfully!")
                    st.rerun()
    
    elif st.session_state.transactions is not None:
        transactions = st.session_state.transactions
        
        # Overview Dashboard
        if page == "📈 Overview Dashboard":
            st.header("📈 Overview Dashboard")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_revenue = transactions['order_total'].sum()
                st.metric("💰 Total Revenue", f"${total_revenue:,.2f}")
            
            with col2:
                total_orders = transactions['order_id'].nunique()
                st.metric("📦 Total Orders", f"{total_orders:,}")
            
            with col3:
                total_customers = transactions['customer_id'].nunique()
                st.metric("👥 Total Customers", f"{total_customers:,}")
            
            with col4:
                avg_order_value = transactions.groupby('order_id')['order_total'].sum().mean()
                st.metric("💵 Avg Order Value", f"${avg_order_value:,.2f}")
            
            st.markdown("---")
            
            # Monthly trends
            col1, col2 = st.columns(2)
            
            with col1:
                monthly_revenue = transactions.groupby('order_year_month')['order_total'].sum().reset_index()
                fig = px.line(monthly_revenue, x='order_year_month', y='order_total',
                             title='Monthly Revenue Trend',
                             labels={'order_year_month': 'Month', 'order_total': 'Revenue ($)'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                monthly_orders = transactions.groupby('order_year_month')['order_id'].nunique().reset_index()
                fig = px.bar(monthly_orders, x='order_year_month', y='order_id',
                            title='Monthly Orders',
                            labels={'order_year_month': 'Month', 'order_id': 'Number of Orders'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Top categories
            if 'product_category_name_english' in transactions.columns:
                category_col = 'product_category_name_english'
            elif 'product_category_name' in transactions.columns:
                category_col = 'product_category_name'
            else:
                category_col = None
            
            if category_col:
                st.subheader("📊 Top 10 Product Categories")
                top_categories = transactions.groupby(category_col)['order_total'].sum().nlargest(10).reset_index()
                fig = px.bar(top_categories, x='order_total', y=category_col, orientation='h',
                            title='Top 10 Categories by Revenue',
                            labels={'order_total': 'Revenue ($)', category_col: 'Category'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Additional metrics
            st.subheader("📊 Additional Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                repeat_rate = (transactions.groupby('customer_id')['order_id'].nunique() > 1).mean() * 100
                st.metric("🔄 Repeat Customer Rate", f"{repeat_rate:.1f}%")
            
            with col2:
                avg_items = transactions.groupby('order_id')['product_id'].count().mean()
                st.metric("🛍️ Avg Items per Order", f"{avg_items:.1f}")
            
            with col3:
                if 'review_score' in transactions.columns:
                    avg_rating = transactions['review_score'].mean()
                    st.metric("⭐ Avg Review Score", f"{avg_rating:.2f}/5")
        
        # Behavioral Insights
        elif page == "🧠 Behavioral Insights":
            st.header("🧠 Behavioral Psychology Insights")
            
            insights = behavioral_psychology_insights(transactions)
            
            # Prospect Theory
            st.subheader("🎯 Prospect Theory: Loss Aversion (Free Shipping Effect)")
            pt = insights['prospect_theory']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Orders with Free Shipping", f"{pt['free_shipping_orders']:,}")
                st.metric("Avg Order Value (Free)", f"${pt['avg_order_free']:,.2f}")
            with col2:
                st.metric("Orders with Paid Shipping", f"{pt['paid_shipping_orders']:,}")
                st.metric("Avg Order Value (Paid)", f"${pt['avg_order_paid']:,.2f}")
            
            st.info("💡 **Insight**: Free shipping leverages loss aversion - customers perceive shipping costs as a loss.")
            
            # Social Proof
            if 'social_proof' in insights:
                st.subheader("⭐ Social Proof: Review Impact")
                sp = insights['social_proof']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("High-Rated Orders (≥4)", f"{sp['high_rated_orders']:,}")
                    st.metric("Avg Price (High-Rated)", f"${sp['avg_high_rated_price']:,.2f}")
                with col2:
                    st.metric("Low-Rated Orders (<3)", f"{sp['low_rated_orders']:,}")
                    st.metric("Avg Price (Low-Rated)", f"${sp['avg_low_rated_price']:,.2f}")
                
                st.info("💡 **Insight**: Positive reviews build trust and influence purchasing decisions.")
            
            # Temporal Patterns
            if 'temporal' in insights:
                st.subheader("⏰ Temporal Decision-Making")
                temp = insights['temporal']
                
                st.write(f"**Peak Shopping Hours:** {', '.join(map(str, temp['peak_hours']))}")
                st.write(f"**Peak Hour Orders:** {temp['peak_hour_orders']:,}")
                
                if 'purchase_hour' in transactions.columns:
                    hourly = transactions.groupby('purchase_hour')['order_id'].nunique().reset_index()
                    fig = px.bar(hourly, x='purchase_hour', y='order_id',
                                title='Orders by Hour of Day',
                                labels={'purchase_hour': 'Hour', 'order_id': 'Orders'})
                    st.plotly_chart(fig, use_container_width=True)
                
                st.info("💡 **Insight**: Shopping patterns reveal when customers are most engaged.")
            
            # Cognitive Biases
            st.subheader("🧩 Cognitive Biases: Status Quo & Habit Formation")
            cb = insights['cognitive_biases']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Customers", f"{cb['total_customers']:,}")
            with col2:
                st.metric("Repeat Customers", f"{cb['repeat_customers']:,}")
            with col3:
                st.metric("Repeat Rate", f"{cb['status_quo_bias']*100:.1f}%")
            
            st.info("💡 **Insight**: Status quo bias keeps customers returning to familiar brands.")
            
            # Recommendations
            st.subheader("🚀 Actionable Recommendations")
            st.markdown("""
            1. **Free Shipping Strategy**: Offer free shipping thresholds to increase average order value
            2. **Social Proof**: Prominently display reviews and ratings to build trust
            3. **Peak Time Marketing**: Schedule promotions during peak shopping hours
            4. **Loyalty Programs**: Encourage repeat purchases through rewards
            5. **Choice Architecture**: Simplify product categories to reduce decision fatigue
            """)
        
        # Customer Lifetime Value
        elif page == "💰 Customer Lifetime Value":
            st.header("💰 Customer Lifetime Value Analysis")
            
            clv_data = calculate_clv(transactions)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg CLV", f"${clv_data['clv'].mean():,.2f}")
            with col2:
                st.metric("Median CLV", f"${clv_data['clv'].median():,.2f}")
            with col3:
                st.metric("Total CLV", f"${clv_data['clv'].sum():,.2f}")
            
            # Distribution
            fig = plot_clv_distribution(clv_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Segments
            st.subheader("💎 CLV Segments")
            clv_data['clv_segment'] = pd.qcut(clv_data['clv'], q=4, labels=['Low', 'Medium', 'High', 'Very High'], duplicates='drop')
            segment_summary = clv_data.groupby('clv_segment').agg({
                'customer_id': 'count',
                'clv': 'mean',
                'monetary': 'mean'
            }).reset_index()
            segment_summary.columns = ['CLV Segment', 'Customer Count', 'Avg CLV', 'Total Spent']
            st.dataframe(segment_summary, use_container_width=True)
            
            # Top customers
            st.subheader("🏆 Top 20 Customers by CLV")
            top_clv = clv_data.nlargest(20, 'clv')[['customer_id', 'clv', 'frequency', 'monetary', 'avg_order_value']]
            st.dataframe(top_clv, use_container_width=True)
            
            # Download
            csv = clv_data.to_csv(index=False)
            st.download_button("📥 Download CLV Data", csv, "clv_analysis.csv", "text/csv")
        
        # RFM Analysis
        elif page == "🎯 RFM Analysis":
            st.header("🎯 RFM Analysis")
            
            rfm = calculate_rfm(transactions)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = plot_rfm_segments(rfm)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📊 Segment Summary")
                segment_counts = rfm['segment'].value_counts().reset_index()
                segment_counts.columns = ['Segment', 'Count']
                st.dataframe(segment_counts, use_container_width=True)
            
            # Segment details
            st.subheader("📈 Segment Characteristics")
            segment_details = rfm.groupby('segment').agg({
                'recency': 'mean',
                'frequency': 'mean',
                'monetary': 'mean',
                'customer_id': 'count'
            }).reset_index()
            segment_details.columns = ['Segment', 'Avg Recency (days)', 'Avg Frequency', 'Avg Monetary ($)', 'Customer Count']
            st.dataframe(segment_details, use_container_width=True)
            
            # RFM distribution
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig = px.histogram(rfm, x='recency', nbins=30, title='Recency Distribution')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.histogram(rfm, x='frequency', nbins=30, title='Frequency Distribution')
                st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                fig = px.histogram(rfm, x='monetary', nbins=30, title='Monetary Distribution')
                st.plotly_chart(fig, use_container_width=True)
            
            # Download
            csv = rfm.to_csv(index=False)
            st.download_button("📥 Download RFM Data", csv, "rfm_analysis.csv", "text/csv")
        
        # Customer Segmentation
        elif page == "👥 Customer Segmentation":
            st.header("👥 Customer Segmentation (K-Means)")
            
            customer_features, inertias, k_range = perform_customer_segmentation(transactions)
            
            # Elbow plot
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("📉 Elbow Method")
                fig = px.line(x=list(k_range), y=inertias, markers=True,
                             labels={'x': 'Number of Clusters', 'y': 'Inertia'},
                             title='Optimal Clusters')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🎯 Cluster Distribution")
                cluster_counts = customer_features['cluster_label'].value_counts().reset_index()
                cluster_counts.columns = ['Cluster', 'Count']
                fig = px.pie(cluster_counts, values='Count', names='Cluster',
                            title='Customer Distribution by Cluster')
                st.plotly_chart(fig, use_container_width=True)
            
            # 3D visualization
            st.subheader("🌐 3D Cluster Visualization")
            fig = plot_cluster_3d(customer_features)
            st.plotly_chart(fig, use_container_width=True)
            
            # Cluster characteristics
            st.subheader("📊 Cluster Characteristics")
            cluster_summary = customer_features.groupby('cluster_label').agg({
                'customer_id': 'count',
                'total_spent': 'mean',
                'order_count': 'mean',
                'avg_order_value': 'mean',
                'recency_days': 'mean'
            }).reset_index()
            cluster_summary.columns = ['Cluster', 'Customer Count', 'Avg Total Spent', 'Avg Orders', 'Avg Order Value', 'Avg Recency']
            st.dataframe(cluster_summary, use_container_width=True)
            
            # Download
            csv = customer_features.to_csv(index=False)
            st.download_button("📥 Download Segmentation Data", csv, "customer_segmentation.csv", "text/csv")
        
        # Churn Prediction
        elif page == "⚠️ Churn Prediction":
            st.header("⚠️ Churn Prediction")
            
            churn_data, model, feature_importance, y_test, y_pred = predict_churn(transactions)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                churn_rate = churn_data['is_churned'].mean() * 100
                st.metric("📉 Churn Rate", f"{churn_rate:.1f}%")
            
            with col2:
                high_risk = (churn_data['churn_risk'] == 'High').sum()
                st.metric("⚠️ High Risk Customers", f"{high_risk:,}")
            
            with col3:
                medium_risk = (churn_data['churn_risk'] == 'Medium').sum()
                st.metric("⚡ Medium Risk Customers", f"{medium_risk:,}")
            
            # Risk distribution
            col1, col2 = st.columns([1, 1])
            
            with col1:
                fig = plot_churn_risk(churn_data)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🎯 Feature Importance")
                fig = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                            title='Churn Prediction Feature Importance')
                st.plotly_chart(fig, use_container_width=True)
            
            # Model performance
            st.subheader("📊 Model Performance")
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                accuracy = accuracy_score(y_test, y_pred)
                st.metric("Accuracy", f"{accuracy:.2%}")
            with col2:
                precision = precision_score(y_test, y_pred, zero_division=0)
                st.metric("Precision", f"{precision:.2%}")
            with col3:
                recall = recall_score(y_test, y_pred, zero_division=0)
                st.metric("Recall", f"{recall:.2%}")
            with col4:
                f1 = f1_score(y_test, y_pred, zero_division=0)
                st.metric("F1 Score", f"{f1:.2%}")
            
            # High risk customers
            st.subheader("🚨 Top 20 High-Risk Customers")
            high_risk_customers = churn_data[churn_data['churn_risk'] == 'High'].nlargest(20, 'churn_probability')
            display_cols = ['customer_id', 'churn_probability', 'recency_days', 'order_count', 'total_spent']
            st.dataframe(high_risk_customers[display_cols], use_container_width=True)
            
            # Recommendations
            st.subheader("💡 Churn Prevention Strategies")
            st.markdown("""
            1. **Re-engagement Campaigns**: Target high-risk customers with personalized offers
            2. **Win-back Programs**: Special discounts for inactive customers
            3. **Loyalty Rewards**: Incentivize frequent purchases
            4. **Customer Feedback**: Understand reasons for inactivity
            5. **Proactive Support**: Reach out before customers churn
            """)
            
            # Download
            csv = churn_data.to_csv(index=False)
            st.download_button("📥 Download Churn Data", csv, "churn_prediction.csv", "text/csv")
        
        # Market Basket Analysis
        elif page == "🛒 Market Basket Analysis":
            st.header("🛒 Market Basket Analysis")
            
            frequent_itemsets, rules = market_basket_analysis(transactions)
            
            if rules is not None and len(rules) > 0:
                st.subheader("🔗 Association Rules")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    min_support = st.slider("Min Support", 0.0, 0.1, 0.01, 0.001)
                with col2:
                    min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.3, 0.05)
                with col3:
                    min_lift = st.slider("Min Lift", 1.0, 5.0, 1.5, 0.1)
                
                filtered_rules = rules[
                    (rules['support'] >= min_support) &
                    (rules['confidence'] >= min_confidence) &
                    (rules['lift'] >= min_lift)
                ]
                
                st.write(f"**Found {len(filtered_rules)} rules matching criteria**")
                
                if len(filtered_rules) > 0:
                    # Convert frozensets to strings for display
                    display_rules = filtered_rules.copy()
                    display_rules['antecedents'] = display_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                    display_rules['consequents'] = display_rules['consequents'].apply(lambda x: ', '.join(list(x)))
                    
                    display_cols = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
                    st.dataframe(display_rules[display_cols].head(20), use_container_width=True)
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.scatter(filtered_rules, x='support', y='confidence',
                                       size='lift', color='lift',
                                       title='Support vs Confidence (sized by Lift)',
                                       labels={'support': 'Support', 'confidence': 'Confidence'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        top_rules = filtered_rules.nlargest(10, 'lift')
                        fig = px.bar(top_rules, x='lift', y=top_rules.index,
                                    orientation='h', title='Top 10 Rules by Lift')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommendations
                    st.subheader("💡 Product Bundling Recommendations")
                    st.markdown("""
                    1. **Cross-selling**: Recommend frequently bought together items
                    2. **Product Bundles**: Create bundles based on high-lift associations
                    3. **Store Layout**: Place associated items near each other
                    4. **Personalization**: Use rules for personalized recommendations
                    """)
                    
                    # Download
                    csv = display_rules.to_csv(index=False)
                    st.download_button("📥 Download Association Rules", csv, "market_basket_rules.csv", "text/csv")
                else:
                    st.warning("No rules found with the selected criteria. Try lowering the thresholds.")
            else:
                st.warning("Not enough data for market basket analysis. Need more varied product purchases.")
        
        # Sales Forecasting
        elif page == "📈 Sales Forecasting":
            st.header("📈 Sales Forecasting")
            
            daily_sales, forecast_df, model, rmse, r2 = forecast_sales(transactions)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📊 Model R² Score", f"{r2:.3f}")
            with col2:
                st.metric("📉 RMSE", f"${rmse:,.2f}")
            
            # Forecast visualization
            st.subheader("📈 30-Day Revenue Forecast")
            fig = plot_sales_forecast(daily_sales, forecast_df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast summary
            st.subheader("📊 Forecast Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_forecast = forecast_df['predicted_revenue'].sum()
                st.metric("Total Forecast (30 days)", f"${total_forecast:,.2f}")
            
            with col2:
                avg_daily = forecast_df['predicted_revenue'].mean()
                st.metric("Avg Daily Revenue", f"${avg_daily:,.2f}")
            
            with col3:
                max_day = forecast_df.loc[forecast_df['predicted_revenue'].idxmax(), 'date']
                st.metric("Peak Revenue Day", max_day.strftime('%Y-%m-%d'))
            
            # Forecast table
            st.subheader("📅 Daily Forecast")
            forecast_display = forecast_df[['date', 'predicted_revenue']].copy()
            forecast_display['predicted_revenue'] = forecast_display['predicted_revenue'].apply(lambda x: f"${x:,.2f}")
            st.dataframe(forecast_display, use_container_width=True)
            
            # Download
            csv = forecast_df.to_csv(index=False)
            st.download_button("📥 Download Forecast", csv, "sales_forecast.csv", "text/csv")
        
        # Sales Analytics
        elif page == "📊 Sales Analytics":
            st.header("📊 Sales Analytics")
            
            # Payment methods
            if 'payment_type' in transactions.columns:
                st.subheader("💳 Payment Method Distribution")
                payment_dist = transactions.groupby('payment_type')['order_id'].nunique().reset_index()
                payment_dist.columns = ['Payment Method', 'Orders']
                fig = px.pie(payment_dist, values='Orders', names='Payment Method',
                            title='Orders by Payment Method')
                st.plotly_chart(fig, use_container_width=True)
            
            # Geographic analysis
            if 'customer_state' in transactions.columns:
                st.subheader("🗺️ Geographic Revenue Distribution")
                state_revenue = transactions.groupby('customer_state').agg({
                    'order_total': 'sum',
                    'order_id': 'nunique'
                }).reset_index()
                state_revenue.columns = ['State', 'Revenue', 'Orders']
                state_revenue = state_revenue.nlargest(15, 'Revenue')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(state_revenue, x='State', y='Revenue',
                                title='Top 15 States by Revenue',
                                labels={'Revenue': 'Revenue ($)'})
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(state_revenue, x='State', y='Orders',
                                title='Top 15 States by Orders',
                                labels={'Orders': 'Number of Orders'})
                    st.plotly_chart(fig, use_container_width=True)
            
            # Category performance
            if 'product_category_name_english' in transactions.columns:
                category_col = 'product_category_name_english'
            elif 'product_category_name' in transactions.columns:
                category_col = 'product_category_name'
            else:
                category_col = None
            
            if category_col:
                st.subheader("📦 Category Performance")
                category_performance = transactions.groupby(category_col).agg({
                    'order_total': 'sum',
                    'order_id': 'nunique',
                    'product_id': 'count'
                }).reset_index()
                category_performance.columns = ['Category', 'Revenue', 'Orders', 'Items Sold']
                category_performance = category_performance.nlargest(15, 'Revenue')
                
                fig = px.scatter(category_performance, x='Orders', y='Revenue',
                               size='Items Sold', hover_data=['Category'],
                               title='Category Performance: Orders vs Revenue',
                               labels={'Orders': 'Number of Orders', 'Revenue': 'Revenue ($)'})
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(category_performance, use_container_width=True)
        
        # Product Analytics
        elif page == "⭐ Product Analytics":
            st.header("⭐ Product Analytics")
            
            # Review scores by category
            if 'review_score' in transactions.columns:
                if 'product_category_name_english' in transactions.columns:
                    category_col = 'product_category_name_english'
                elif 'product_category_name' in transactions.columns:
                    category_col = 'product_category_name'
                else:
                    category_col = None
                
                if category_col:
                    st.subheader("⭐ Review Scores by Category")
                    category_reviews = transactions.groupby(category_col)['review_score'].agg(['mean', 'count']).reset_index()
                    category_reviews.columns = ['Category', 'Avg Rating', 'Review Count']
                    category_reviews = category_reviews[category_reviews['Review Count'] >= 10]
                    category_reviews = category_reviews.nlargest(15, 'Avg Rating')
                    
                    fig = px.bar(category_reviews, x='Avg Rating', y='Category', orientation='h',
                                title='Top 15 Categories by Average Rating',
                                labels={'Avg Rating': 'Average Rating', 'Category': 'Product Category'})
                    st.plotly_chart(fig, use_container_width=True)
            
            # Price distribution
            st.subheader("💰 Price Distribution")
            fig = px.histogram(transactions, x='price', nbins=50,
                             title='Product Price Distribution',
                             labels={'price': 'Price ($)', 'count': 'Number of Products'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Price statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Min Price", f"${transactions['price'].min():.2f}")
            with col2:
                st.metric("Max Price", f"${transactions['price'].max():.2f}")
            with col3:
                st.metric("Avg Price", f"${transactions['price'].mean():.2f}")
            with col4:
                st.metric("Median Price", f"${transactions['price'].median():.2f}")
            
            # Most popular products
            st.subheader("🏆 Most Popular Products")
            product_popularity = transactions.groupby('product_id').agg({
                'order_id': 'nunique',
                'price': 'mean'
            }).reset_index()
            product_popularity.columns = ['Product ID', 'Orders', 'Avg Price']
            product_popularity = product_popularity.nlargest(20, 'Orders')
            
            fig = px.scatter(product_popularity, x='Avg Price', y='Orders',
                           hover_data=['Product ID'],
                           title='Top 20 Products: Price vs Popularity',
                           labels={'Avg Price': 'Average Price ($)', 'Orders': 'Number of Orders'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(product_popularity, use_container_width=True)
            
            # Price vs Review correlation
            if 'review_score' in transactions.columns:
                st.subheader("💰⭐ Price vs Review Score Correlation")
                
                # Create price bins
                transactions_copy = transactions.copy()
                transactions_copy['price_bin'] = pd.cut(transactions_copy['price'], bins=10)
                price_review = transactions_copy.groupby('price_bin')['review_score'].mean().reset_index()
                price_review['price_bin'] = price_review['price_bin'].astype(str)
                
                fig = px.line(price_review, x='price_bin', y='review_score',
                             title='Average Review Score by Price Range',
                             labels={'price_bin': 'Price Range', 'review_score': 'Avg Review Score'})
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
                
                correlation = transactions[['price', 'review_score']].corr().iloc[0, 1]
                st.metric("Price-Review Correlation", f"{correlation:.3f}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>📊 E-commerce Analytics Dashboard v4 | Built with Streamlit & Python</p>
        <p>💡 Upload your data and explore comprehensive analytics including CLV, RFM, Churn Prediction, and more!</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
