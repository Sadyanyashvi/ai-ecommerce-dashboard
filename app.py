"""
AI-Powered E-Commerce Dashboard
A Streamlit app for real-time analytics with Gemini AI insights
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import google.generativeai as genai
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import logging

# ═══════════════════════════════════════════════════════════════════════════════
# SETUP & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Gemini API Configuration
try:
    gemini_api_key = os.getenv('GEMINI_API_KEY') or st.secrets.get('GEMINI_API_KEY')
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    genai.configure(api_key=gemini_api_key)
    GEMINI_MODEL = 'gemini-2.0-flash'  # Centralized model config
except Exception as e:
    logger.error(f"Gemini API configuration failed: {e}")
    st.error("❌ Gemini API configuration failed. Check your GEMINI_API_KEY.")

# Streamlit page configuration
st.set_page_config(
    page_title="AI E-Commerce Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown('''
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1rem;
        border-left: 4px solid #185FA5;
    }
    .insight-card {
        background: #EAF3DE;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #3B6D11;
    }
    .alert-card {
        background: #FAEEDA;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #854F0B;
    }
    .error-card {
        background: #FFE5E5;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #993C1D;
    }
</style>
''', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def import_data_generator():
    """Dynamically import data generator to avoid module not found errors."""
    try:
        from data_generator import generate_orders, add_live_orders
        return generate_orders, add_live_orders
    except ImportError:
        logger.warning("data_generator module not found. Using fallback data.")
        return None, None


@st.cache_data(ttl=30)
def load_ecommerce_data(time_key):
    """
    Load and prepare e-commerce data with live updates every 30 seconds.
    
    Args:
        time_key: Time-based cache invalidation key
        
    Returns:
        pd.DataFrame: Processed e-commerce data
    """
    generate_orders, add_live_orders = import_data_generator()
    
    if generate_orders is None:
        # Fallback: Create sample data structure
        st.warning("⚠️ Using sample data. Install data_generator for live data.")
        df = pd.DataFrame({
            'date': pd.date_range(start='2024-01-01', periods=100),
            'product': ['Product A', 'Product B', 'Product C'] * 33 + ['Product A'],
            'category': ['Electronics', 'Clothing', 'Home'] * 33 + ['Electronics'],
            'revenue': [100, 150, 200] * 33 + [100],
            'status': ['completed', 'pending', 'shipped', 'cancelled'] * 25,
        })
    else:
        df = generate_orders(n_days=90, orders_per_day=20)
        df = add_live_orders(df)
    
    df['date'] = pd.to_datetime(df['date'])
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# GEMINI AI FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def safe_gemini_call(prompt, max_retries=2):
    """
    Safely call Gemini API with error handling and retries.
    
    Args:
        prompt: The prompt to send to Gemini
        max_retries: Number of retries on failure
        
    Returns:
        str: Generated response or error message
    """
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return f"❌ AI analysis unavailable. Error: {str(e)[:100]}"
            continue


def generate_ai_insights(df, period_label):
    """
    Generate 3 business insights from e-commerce data using Gemini.
    
    Args:
        df: DataFrame with e-commerce data
        period_label: Human-readable period label (e.g., "Last 7 days")
        
    Returns:
        str: Formatted business insights
    """
    if df.empty:
        return "⚠️ No data available for the selected filters."
    
    try:
        # Calculate key metrics
        top_cat = df.groupby('category')['revenue'].sum().idxmax()
        worst_cat = df.groupby('category')['revenue'].sum().idxmin()
        top_product = df.groupby('product')['revenue'].sum().idxmax()
        cancel_rate = (df['status'] == 'cancelled').sum() / len(df) * 100
        
        # Weekend vs weekday analysis
        df_temp = df.copy()
        df_temp['dow'] = pd.to_datetime(df_temp['date']).dt.dayofweek
        weekend_rev = df_temp[df_temp['dow'] >= 5]['revenue'].mean()
        weekday_rev = df_temp[df_temp['dow'] < 5]['revenue'].mean()
        
        summary = f"""
        E-Commerce Performance Summary ({period_label})
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        • Total Revenue: ${df['revenue'].sum():,.0f}
        • Total Orders: {len(df):,}
        • Avg Order Value: ${df['revenue'].mean():.2f}
        • Cancellation Rate: {cancel_rate:.1f}%
        • Best Category: {top_cat}
        • Worst Category: {worst_cat}
        • Top Product: {top_product}
        • Weekend Avg Revenue: ${weekend_rev:.0f}
        • Weekday Avg Revenue: ${weekday_rev:.0f}
        """
        
        prompt = f"""You are a senior e-commerce business analyst.

Analyze this data and provide exactly 3 business insights.

Each insight MUST:
1. Identify a specific opportunity or issue
2. Explain why it matters (impact)
3. Suggest one concrete action

Format each as:
INSIGHT [N]: [Title]
Finding: [What you discovered]
Impact: [Why it matters]
ACTION: [Specific step to take]

Keep concise. No fluff.

{summary}"""
        
        return safe_gemini_call(prompt)
        
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        return f"❌ Failed to generate insights: {str(e)}"


def answer_business_question(df, user_question, selected_range, selected_cat):
    """
    Answer a user's business question using Gemini with current data context.
    
    Args:
        df: DataFrame with current filtered data
        user_question: User's business question
        selected_range: Time range being analyzed
        selected_cat: Product category filter
        
    Returns:
        str: AI-generated answer
    """
    if not user_question or user_question.strip() == "":
        return "Please enter a question."
    
    try:
        context = f"""
        Current E-Commerce Data Context
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        • Total Revenue: ${df['revenue'].sum():,.0f}
        • Total Orders: {len(df):,}
        • Average Order Value: ${df['revenue'].mean():.2f}
        • Top Category: {df.groupby('category')['revenue'].sum().idxmax() if not df.empty else 'N/A'}
        • Top Product: {df.groupby('product')['revenue'].sum().idxmax() if not df.empty else 'N/A'}
        • Time Range: {selected_range}
        • Category Filter: {selected_cat}
        """
        
        prompt = f"""{context}

User Question: {user_question}

Provide a concise, actionable answer (2-3 sentences max) with specific recommendations based on the data context."""
        
        return safe_gemini_call(prompt)
        
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        return f"❌ Error processing question: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS & KPI CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_kpis(df, df_prev):
    """
    Calculate key performance indicators with period-over-period comparison.
    
    Args:
        df: Current period data
        df_prev: Previous period data for comparison
        
    Returns:
        dict: Dictionary of KPIs
    """
    total_revenue = df['revenue'].sum()
    prev_revenue = df_prev['revenue'].sum() if not df_prev.empty else total_revenue
    revenue_delta = ((total_revenue - prev_revenue) / prev_revenue * 100) if prev_revenue > 0 else 0
    
    return {
        'total_revenue': total_revenue,
        'revenue_delta': revenue_delta,
        'total_orders': len(df),
        'avg_order_value': df['revenue'].mean() if len(df) > 0 else 0,
        'completion_rate': (df['status'] == 'completed').sum() / len(df) * 100 if len(df) > 0 else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_revenue_trend(df):
    """Create revenue trend line chart."""
    if df.empty:
        st.info("No data available for revenue trend.")
        return
    
    daily_rev = df.groupby('date')['revenue'].sum().reset_index()
    fig = px.line(
        daily_rev,
        x='date',
        y='revenue',
        title='Daily Revenue Trend',
        color_discrete_sequence=['#185FA5'],
        labels={'revenue': 'Revenue ($)', 'date': 'Date'}
    )
    fig.update_layout(
        showlegend=False,
        height=320,
        hovermode='x unified',
        yaxis_tickprefix='$'
    )
    st.plotly_chart(fig, width='stretch')


def plot_top_products(df):
    """Create top products bar chart."""
    if df.empty:
        st.info("No data available for product analysis.")
        return
    
    top_products = (df.groupby('product')['revenue']
                    .sum()
                    .sort_values(ascending=True)
                    .tail(6)
                    .reset_index())
    
    fig = px.bar(
        top_products,
        x='revenue',
        y='product',
        orientation='h',
        color='revenue',
        color_continuous_scale='Blues',
        title='Top 6 Products by Revenue',
        labels={'revenue': 'Revenue ($)', 'product': 'Product'}
    )
    fig.update_layout(
        height=320,
        showlegend=False,
        xaxis_tickprefix='$'
    )
    st.plotly_chart(fig, width='stretch')


def plot_sales_by_category(df):
    """Create category breakdown pie chart."""
    if df.empty:
        st.info("No data available for category analysis.")
        return
    
    cat_rev = df.groupby('category')['revenue'].sum().reset_index()
    fig = px.pie(
        cat_rev,
        values='revenue',
        names='category',
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set2,
        title='Sales Distribution by Category'
    )
    fig.update_layout(height=320)
    st.plotly_chart(fig, width='stretch')


def plot_order_status(df):
    """Create order status breakdown bar chart."""
    if df.empty:
        st.info("No data available for order status.")
        return
    
    status_counts = df['status'].value_counts().reset_index()
    status_counts.columns = ['status', 'count']
    
    fig = px.bar(
        status_counts,
        x='status',
        y='count',
        color='status',
        color_discrete_map={
            'completed': '#3B6D11',
            'pending': '#854F0B',
            'shipped': '#185FA5',
            'cancelled': '#993C1D'
        },
        title='Order Status Distribution',
        labels={'status': 'Status', 'count': 'Count'}
    )
    fig.update_layout(
        height=320,
        showlegend=False,
    )
    st.plotly_chart(fig, width='stretch')


# ═══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar_filters(df_all):
    """
    Render sidebar filter controls.
    
    Returns:
        tuple: (filtered_df, selected_range_label, selected_category)
    """
    with st.sidebar:
        st.title("🎛️ Dashboard Controls")
        st.markdown("---")
        
        # Date range filter
        st.subheader("📅 Date Range")
        date_options = {
            "Last 7 days": 7,
            "Last 30 days": 30,
            "Last 90 days": 90,
        }
        selected_range = st.selectbox("Select period", list(date_options.keys()))
        n_days = date_options[selected_range]
        cutoff = datetime.now() - timedelta(days=n_days)
        
        # Category filter
        st.subheader("🏷️ Product Category")
        categories = ['All'] + sorted(df_all['category'].unique().tolist())
        selected_cat = st.selectbox("Filter by category", categories)
        
        # Apply filters
        df = df_all[df_all['date'] >= cutoff]
        if selected_cat != 'All':
            df = df[df['category'] == selected_cat]
        
        # Info footer
        st.markdown("---")
        st.caption("ℹ️ Data refreshes every 30 seconds")
        st.caption(f"⏰ Last update: {datetime.now().strftime('%H:%M:%S')}")
        
        return df, selected_range, selected_cat


def render_kpi_cards(kpis):
    """Display KPI metric cards."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            '💰 Total Revenue',
            f'${kpis["total_revenue"]:,.0f}',
            f'{kpis["revenue_delta"]:+.1f}% vs prev period'
        )
    
    with col2:
        st.metric(
            '📦 Total Orders',
            f'{kpis["total_orders"]:,}'
        )
    
    with col3:
        st.metric(
            '🎯 Avg Order Value',
            f'${kpis["avg_order_value"]:.2f}'
        )
    
    with col4:
        st.metric(
            '✅ Completion Rate',
            f'{kpis["completion_rate"]:.1f}%'
        )


def render_insights_section(df, selected_range):
    """Render AI insights generation section."""
    st.subheader("🤖 AI Business Insights")
    
    if st.button("🔄 Generate Fresh Insights", type="primary", use_container_width=False):
        with st.spinner('🔍 Analyzing data with Gemini...'):
            insights = generate_ai_insights(df, selected_range)
            st.session_state["insights"] = insights
    
    if "insights" in st.session_state:
        insights_text = st.session_state["insights"]
        
        # Parse and format insights
        lines = insights_text.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if 'INSIGHT' in line.upper():
                st.markdown(f'<div class="insight-card"><strong>{line}</strong></div>',
                           unsafe_allow_html=True)
            elif 'ACTION:' in line.upper():
                st.markdown(f'<div class="alert-card">{line}</div>',
                           unsafe_allow_html=True)
            elif line:
                st.markdown(line)
    else:
        st.info("💡 Click the button above to generate AI insights")


def render_chat_section(df, selected_range, selected_cat):
    """Render ask-a-question chat interface."""
    st.subheader("💬 Ask a Business Question")
    
    user_question = st.text_input(
        "Ask anything about your data...",
        placeholder="e.g., Which product should I promote this weekend?",
        label_visibility="collapsed"
    )
    
    if user_question:
        with st.spinner('🤔 Thinking...'):
            answer = answer_business_question(df, user_question, selected_range, selected_cat)
            st.markdown(f'<div class="insight-card">{answer}</div>',
                       unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main app execution."""
    
    # Load data
    time_key = datetime.now().strftime('%Y-%m-%d %H:%M')[:-1]  # Round to 30-sec
    df_all = load_ecommerce_data(time_key)
    
    # Sidebar filters
    df, selected_range, selected_cat = render_sidebar_filters(df_all)
    
    # Header
    st.title("🛒 AI-Powered E-Commerce Dashboard")
    st.markdown(f"**Live data** · {selected_range} · {selected_cat}")
    st.markdown("---")
    
    # KPI Section
    st.subheader("📊 Key Performance Indicators")
    
    # Calculate comparison period
    n_days = {'Last 7 days': 7, 'Last 30 days': 30, 'Last 90 days': 90}[selected_range]
    cutoff = datetime.now() - timedelta(days=n_days)
    prev_cutoff = cutoff - timedelta(days=n_days)
    df_prev = df_all[(df_all['date'] >= prev_cutoff) & (df_all['date'] < cutoff)]
    
    kpis = calculate_kpis(df, df_prev)
    render_kpi_cards(kpis)
    
    st.markdown("---")
    
    # Charts Section
    st.subheader("📈 Analytics & Performance")
    
    col1, col2 = st.columns(2)
    with col1:
        plot_revenue_trend(df)
    with col2:
        plot_top_products(df)
    
    col3, col4 = st.columns(2)
    with col3:
        plot_sales_by_category(df)
    with col4:
        plot_order_status(df)
    
    st.markdown("---")
    
    # AI Section
    st.subheader("🤖 AI-Powered Analytics")
    col_insights, col_chat = st.columns(2)
    
    with col_insights:
        render_insights_section(df, selected_range)
    
    with col_chat:
        render_chat_section(df, selected_range, selected_cat)
    
    # Footer
    st.markdown("---")
    st.caption("🚀 Powered by Gemini AI | Data updates every 30 seconds")


if __name__ == "__main__":
    main()