import streamlit as st
import pl
import pandas as pd
import os
from pulp import LpStatus
import plotly.express as px
import plotly.graph_objects as go
import time 

# --- 1. CONFIG & SETUP ---
st.set_page_config(
    page_title="FPL AI Optimizer",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="⚽"
)

# --- 2. Caching & Data Loading ---
@st.cache_data(ttl=3600)
def load_data():
    """Loads FPL player and position data."""
    players_df, positions_df = pl.load_fpl_data()
    return players_df, positions_df

@st.cache_data(ttl=3600)
def fetch_and_forecast_data():
    """Fetches history and forecasts points for the next GW."""
    forecast_df = pl.fetch_and_forecast_players()
    return forecast_df

# --- 3. STARTUP LOADING SCREEN ---
if 'app_ready' not in st.session_state:
    with st.status("🚀 Initializing FPL Optimizer...", expanded=True) as status:
        st.write("📡 Connecting to Premier League API...")
        players_df, positions_df = load_data()
        time.sleep(0.3)
        
        st.write("📊 Processing player statistics...")
        time.sleep(0.3)
        
        st.session_state['app_ready'] = True
        status.update(label="✅ Ready to optimize!", state="complete", expanded=False)

# Modern, Clean CSS
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main {
        padding: 1rem 2rem;
        background: #fafbfc;
    }
    
    /* Header Styles */
    .hero-header {
        background: linear-gradient(135deg, #37003c 0%, #570051 50%, #37003c 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(55, 0, 60, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(4, 245, 255, 0.1) 0%, transparent 70%);
        border-radius: 50%;
    }
    
    .hero-title {
        color: white;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        color: rgba(255, 255, 255, 0.85);
        font-size: 1.1rem;
        margin-top: 0.8rem;
        font-weight: 500;
        position: relative;
        z-index: 1;
    }
    
    .hero-badge {
        display: inline-block;
        background: rgba(4, 245, 255, 0.2);
        color: #04f5ff;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 1rem;
        border: 1px solid rgba(4, 245, 255, 0.3);
    }
    
    /* Tab Styles - More Modern */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: white;
        padding: 0.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #e8eaed;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        background: transparent;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        color: #5f6368;
        transition: all 0.2s ease;
        font-size: 0.95rem;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #f8f9fa;
        color: #37003c;
    }
    
    .stTabs [aria-selected="true"] {
        background: #37003c !important;
        color: white !important;
        box-shadow: 0 2px 8px rgba(55, 0, 60, 0.25);
    }
    
    /* Card Styles - Cleaner */
    .modern-card {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border: 1px solid #e8eaed;
        margin: 1rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .modern-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        transform: translateY(-2px);
    }
    
    .success-banner {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 4px solid #28a745;
        padding: 1.25rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .warning-banner {
        background: linear-gradient(135deg, #fff3cd 0%, #ffe8a1 100%);
        border-left: 4px solid #ffc107;
        padding: 1.25rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .info-banner {
        background: linear-gradient(135deg, #cfe2ff 0%, #b6d4fe 100%);
        border-left: 4px solid #0d6efd;
        padding: 1.25rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    /* Metric Cards - More Visual */
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: all 0.3s ease;
        border: 2px solid transparent;
        position: relative;
        overflow: hidden;
    }
    
    .stat-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #37003c, #04f5ff);
    }
    
    .stat-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        border-color: #37003c;
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #37003c, #04f5ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        color: #5f6368;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Button Styles - More Premium */
    .stButton > button {
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #37003c, #570051);
    }
    
    /* Player Pills - Better Design */
    .player-pill {
        display: inline-flex;
        align-items: center;
        background: white;
        border: 2px solid #e8eaed;
        border-radius: 24px;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        font-weight: 600;
        color: #37003c;
        transition: all 0.2s ease;
        cursor: pointer;
    }
    
    .player-pill:hover {
        background: #37003c;
        color: white;
        border-color: #37003c;
        transform: scale(1.05);
    }
    
    /* Bench Card - Improved */
    .bench-player-card {
        background: white;
        border: 2px solid #e8eaed;
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .bench-player-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: #dee2e6;
        border-radius: 16px 16px 0 0;
    }
    
    .bench-player-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
        border-color: #37003c;
    }
    
    .bench-player-card:hover::before {
        background: linear-gradient(90deg, #37003c, #04f5ff);
    }
    
    /* Section Headers */
    h1, h2, h3 {
        color: #202124;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    h2 {
        font-size: 1.8rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        font-size: 1.4rem;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    
    /* Sidebar Enhancement */
    section[data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #fafbfc, #ffffff);
        border-right: 1px solid #e8eaed;
    }
    
    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }
    
    /* DataFrames */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #e8eaed;
    }
    
    /* Input Styles */
    .stTextInput input, .stSelectbox select, .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #e8eaed;
        transition: all 0.2s;
        font-size: 0.95rem;
    }
    
    .stTextInput input:focus, .stSelectbox select:focus, .stTextArea textarea:focus {
        border-color: #37003c;
        box-shadow: 0 0 0 3px rgba(55, 0, 60, 0.1);
    }
    
    /* Loading Spinner */
    .stSpinner > div {
        border-color: #37003c transparent transparent transparent;
    }
    
    /* Section Divider */
    hr {
        margin: 2.5rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, #e8eaed, transparent);
    }
    
    /* Captain Badge */
    .captain-badge {
        display: inline-block;
        background: linear-gradient(135deg, #ffd700, #ffed4e);
        color: #856404;
        padding: 0.3rem 0.8rem;
        border-radius: 16px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-left: 0.5rem;
        box-shadow: 0 2px 6px rgba(212, 175, 55, 0.3);
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2rem;
        }
        
        .stat-value {
            font-size: 1.8rem;
        }
    }
    
    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #37003c;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #570051;
    }
    </style>
""", unsafe_allow_html=True)

# Enhanced Header with badge
st.markdown("""
    <div class='hero-header'>
        <h1 class='hero-title'>⚽ FPL AI Optimizer</h1>
        <p class='hero-subtitle'>
            Build your dream Fantasy Premier League team with AI-powered insights and advanced analytics
        </p>
        <span class='hero-badge'>🤖 Powered by AI</span>
    </div>
""", unsafe_allow_html=True)

# Load data
players_df, positions_df = load_data()

def create_pitch_visualization(squad_df, starting_11_df=None):
    """Enhanced pitch visualization with better aesthetics"""
    import plotly.graph_objects as go
    import numpy as np
    
    if starting_11_df is not None and not starting_11_df.empty:
        pitch_players = squad_df[squad_df['name'].isin(starting_11_df['name'])].copy()
    else:
        pitch_players = squad_df.copy()
    
    POSITION_MAP = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    if 'position_name' not in pitch_players.columns:
        pitch_players['position_name'] = pitch_players['element_type'].map(POSITION_MAP)
    
    total_value = pitch_players['now_cost'].sum() if 'now_cost' in pitch_players.columns else 0
    total_projected = int(pitch_players['projected_points'].sum()) if 'projected_points' in pitch_players.columns else 0
    
    fig = go.Figure()
    
    # Enhanced pitch background
    fig.add_shape(
        type="rect", x0=0, y0=0, x1=100, y1=100,
        fillcolor="rgba(34, 139, 34, 0.15)",
        line=dict(color="rgba(255, 255, 255, 0.6)", width=2),
        layer='below'
    )
    
    # Grass stripes
    for i in range(10):
        if i % 2 == 0:
            fig.add_shape(
                type="rect", x0=0, y0=i*10, x1=100, y1=(i+1)*10,
                fillcolor="rgba(34, 139, 34, 0.08)",
                line=dict(width=0), layer='below'
            )
    
    # Center line
    fig.add_shape(
        type="line", x0=50, y0=0, x1=50, y1=100,
        line=dict(color="rgba(255, 255, 255, 0.4)", width=2), layer='below'
    )
    
    # Center circle
    fig.add_shape(
        type="circle", x0=40, y0=40, x1=60, y1=60,
        line=dict(color="rgba(255, 255, 255, 0.3)", width=2), 
        fillcolor="rgba(0,0,0,0)", layer='below'
    )
    
    # Center spot
    fig.add_shape(
        type="circle", x0=49, y0=49, x1=51, y1=51,
        fillcolor="rgba(255, 255, 255, 0.6)",
        line=dict(color="white", width=1), layer='below'
    )
    
    # Penalty boxes
    fig.add_shape(
        type="rect", x0=0, y0=30, x1=15, y1=70,
        line=dict(color="rgba(255, 255, 255, 0.3)", width=2),
        fillcolor="rgba(0,0,0,0)", layer='below'
    )
    fig.add_shape(
        type="rect", x0=85, y0=30, x1=100, y1=70,
        line=dict(color="rgba(255, 255, 255, 0.3)", width=2),
        fillcolor="rgba(0,0,0,0)", layer='below'
    )

    # Player positioning
    x_positions = {'GKP': 10, 'DEF': 30, 'MID': 58, 'FWD': 85}
    position_colors = {
        'GKP': '#FFD700',
        'DEF': '#00E5FF',
        'MID': '#FF1493',
        'FWD': '#FF4500'
    }
    
    for pos in ['GKP', 'DEF', 'MID', 'FWD']:
        players = pitch_players[pitch_players['position_name'] == pos]
        if players.empty:
            continue
        
        players = players.sort_values('projected_points', ascending=False)
        n = len(players)
        
        if n == 1: 
            y_vals = [50]
        elif n == 2: 
            y_vals = [37, 63]
        elif n == 3: 
            y_vals = [28, 50, 72]
        elif n == 4: 
            y_vals = [22, 42, 58, 78]
        elif n == 5:
            y_vals = [18, 34, 50, 66, 82]
        else: 
            y_vals = np.linspace(15, 85, n)
        
        for y, (_, p) in zip(y_vals, players.iterrows()):
            # Player marker
            fig.add_trace(go.Scatter(
                x=[x_positions[pos]], y=[y],
                mode='markers',
                marker=dict(
                    size=38, 
                    color=position_colors[pos],
                    line=dict(color='white', width=3),
                    opacity=0.95
                ),
                showlegend=False, hoverinfo='skip'
            ))
            
            # Player initials
            name_parts = p['name'].strip().split()
            if len(name_parts) >= 2:
                initials = f"{name_parts[0][0]}{name_parts[-1][0]}".upper()
            else:
                initials = name_parts[0][:2].upper()
            
            fig.add_trace(go.Scatter(
                x=[x_positions[pos]], y=[y],
                mode='text',
                text=initials,
                textfont=dict(color='white', size=12, family='Arial Black'),
                showlegend=False, hoverinfo='skip'
            ))
            
            # Player surname
            last_name = p['name'].split()[-1].upper()
            fig.add_trace(go.Scatter(
                x=[x_positions[pos]], y=[y - 4.5],
                mode='text',
                text=last_name,
                textfont=dict(color='white', size=9, family='Arial Black'),
                showlegend=False, hoverinfo='skip'
            ))
            
            # Hover area
            fig.add_trace(go.Scatter(
                x=[x_positions[pos]], y=[y],
                mode='markers',
                marker=dict(size=50, color='rgba(0,0,0,0)'),
                hovertemplate=(
                    f"<b>{p['name']}</b><br>"
                    f"<b>Position:</b> {pos}<br>"
                    f"<b>Team:</b> {p.get('team', 'N/A')}<br>"
                    f"<b>Cost:</b> £{p.get('now_cost', 0):.1f}m<br>"
                    f"<b>Projected:</b> {p.get('projected_points', 0):.1f} pts<br>"
                    f"<b>Form:</b> {p.get('form', 'N/A')}<br>"
                    "<extra></extra>"
                ),
                showlegend=False
            ))

    fig.update_layout(
        title=dict(
            text=(
                f"<b style='color:#00FF87;'>⚽ YOUR STARTING XI</b><br>"
                f"<span style='font-size:15px; color:#FFD700;'>Projected Points: {total_projected:.1f}</span> | "
                f"<span style='font-size:15px; color:#00E5FF;'>Team Value: £{total_value:.1f}m</span>"
            ),
            x=0.5, 
            xanchor='center',
            font=dict(size=24, color='white', family='Arial Black')
        ),
        xaxis=dict(range=[0, 100], visible=False, fixedrange=True),
        yaxis=dict(range=[0, 100], visible=False, scaleanchor="x", fixedrange=True),
        plot_bgcolor='#1a472a',
        paper_bgcolor='#37003C',
        height=1100,
        margin=dict(l=40, r=40, t=120, b=40),
        hoverlabel=dict(
            bgcolor="rgba(20, 20, 30, 0.95)", 
            font_size=13,
            font_family="Arial",
            bordercolor="#00FF87"
        )
    )
    
    return fig

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.caption("Customize your optimization parameters")
    
    budget = st.slider(
        "💰 Maximum Budget",
        min_value=90.0,
        max_value=100.0,
        value=100.0,
        step=0.1,
        help="Set your total squad budget constraint"
    )
    
    st.markdown("---")
    
    with st.expander("🔧 Advanced Settings", expanded=False):
        st.caption("""
        **AI Features Require API Keys:**
        - `GOOGLE_API_KEY`
        - `GOOGLE_CSE_ID`
        
        Configure in Streamlit Secrets or environment variables.
        """)
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    
    if 'selected_squad_df' in st.session_state:
        squad_df = st.session_state['selected_squad_df']
        st.markdown(f"""
            <div class='modern-card' style='padding: 1rem;'>
                <div style='text-align: center;'>
                    <div style='font-size: 2rem; font-weight: 800; color: #37003c;'>{len(squad_df)}</div>
                    <div style='color: #5f6368; font-size: 0.85rem; font-weight: 600;'>PLAYERS</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class='modern-card' style='padding: 1rem; margin-top: 0.5rem;'>
                <div style='text-align: center;'>
                    <div style='font-size: 2rem; font-weight: 800; color: #28a745;'>{squad_df['projected_points'].sum():.0f}</div>
                    <div style='color: #5f6368; font-size: 0.85rem; font-weight: 600;'>TOTAL POINTS</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("📊 Run optimization to see stats")

# --- MAIN CONTENT ---
tab1, tab_my_team, tab2, tab5, tab4, tab3 = st.tabs([
    "📊 Overview",
    "📋 My Team",
    "🎯 Build Squad",
    "📈 Analytics",
    "🤖 AI Advisor",
    "🚑 Injury News"
])

# --- NEW TAB: MY TEAM ---
# --- UI/UX IMPROVEMENT: FILTERS IN SIDEBAR OR EXPANDER ---
with st.sidebar:
    st.header("🎯 Squad Discovery")
    st.info("Filter these options to narrow down the dropdown list below. This won't hide players from the analysis graph.")
    
    # Global filters that only affect the DROPDOWN content
    teams = sorted(players_df['team'].unique().tolist())
    filter_teams = st.multiselect("Focus on Teams:", options=teams)
    
    positions = sorted(players_df['position_name'].unique().tolist())
    filter_positions = st.multiselect("Focus on Positions:", options=positions)

# --- MAIN TAB CONTENT ---
# --- TAB: MY TEAM (THE NEW TAB) ---
with tab_my_team:
    st.title("📋 My Current Squad Analysis")
    st.markdown("Select your current 15 players to see how they stack up against the rest of the league.")
    # --- SESSION STATE INITIALIZATION ---
# This ensures 'my_squad_names' exists the first time the app loads
    if 'my_squad_names' not in st.session_state:
        st.session_state.my_squad_names = []
    # UI/UX: Improved Filter Interface in an Expander
    with st.expander("🔍 Manage Squad & Search Filters", expanded=True):
        f_col1, f_col2, f_col3 = st.columns([1, 1, 2])
        
        with f_col1:
            teams = sorted(players_df['team'].unique().tolist())
            search_team = st.multiselect("Focus on Teams", options=teams)
        
        with f_col2:
            positions = ["GKP", "DEF", "MID", "FWD"]
            search_pos = st.multiselect("Focus on Positions", options=positions)
            
        # Apply filters ONLY to the dropdown list
        dropdown_df = players_df.copy()
        if search_team:
            dropdown_df = dropdown_df[dropdown_df['team'].isin(search_team)]
        if search_pos:
            dropdown_df = dropdown_df[dropdown_df['position_name'].isin(search_pos)]
        
        with f_col3:
            # We use a set to ensure we don't lose players previously selected but currently filtered out
            all_options = sorted(list(set(dropdown_df['name'].tolist() + st.session_state.my_squad_names)))
            selected = st.multiselect(
                "Select Players (Max 15)",
                options=all_options,
                default=st.session_state.my_squad_names,
                placeholder="Search for players..."
            )
            st.session_state.my_squad_names = selected

    if st.session_state.my_squad_names:
        # Preparation for Analytics
        plot_df = players_df.copy()
        plot_df['Status'] = 'League'
        plot_df.loc[plot_df['name'].isin(st.session_state.my_squad_names), 'Status'] = 'My Squad'
        
        # Metric Bar
        my_team_stats = plot_df[plot_df['Status'] == 'My Squad']
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Squad Size", f"{len(my_team_stats)}/15")
        m2.metric("Total Points", f"{my_team_stats['total_points'].sum()}")
        m3.metric("Avg Points/Player", f"{my_team_stats['total_points'].mean():.1f}")
        m4.metric("Total Cost", f"£{my_team_stats['now_cost'].sum()/10:.1f}m")

        st.divider()

        # GRAPH 1: Main Scatter Plot (High-Level Context)
        st.subheader("💰 Value Analysis: Total Points vs. Cost")
        st.caption("Faded dots represent the rest of the league. Neon green dots are your players.")
        
        fig_main = px.scatter(
            plot_df, x='now_cost', y='total_points',
            color='Status', hover_name='name',
            color_discrete_map={'My Squad': '#00FF87', 'League': '#37003c'},
            labels={'now_cost': 'Cost (£m)', 'total_points': 'Total Points'},
            template="plotly_white", height=500
        )
        
        # UX: Highlight Selected Trace
        fig_main.update_traces(marker=dict(size=15, line=dict(width=2, color='white')), selector=dict(name='My Squad'))
        fig_main.update_traces(marker=dict(size=7, opacity=0.2), selector=dict(name='League'))
        st.plotly_chart(fig_main, use_container_width=True)

        st.divider()

        # GRAPH 2: Position-Split Scatter Plot (Faceted)
        st.subheader("🛡️ Position Breakdown: Points vs. Cost")
        st.caption("Comparing your squad efficiency within specific roles (GK, DEF, MID, FWD)")

        fig_pos = px.scatter(
            plot_df, x='now_cost', y='total_points',
            color='Status', facet_col='position_name',
            facet_col_wrap=2, hover_name='name',
            color_discrete_map={'My Squad': '#00FF87', 'League': '#37003c'},
            labels={'now_cost': 'Cost', 'total_points': 'Points'},
            template="plotly_white", height=700
        )
        
        # UX: Highlight across all facets
        fig_pos.update_traces(marker=dict(size=14, line=dict(width=2, color='white')), selector=dict(name='My Squad'))
        fig_pos.update_traces(marker=dict(size=6, opacity=0.2), selector=dict(name='League'))
        
        # Polish facet titles
        fig_pos.for_each_annotation(lambda a: a.update(text=f"<b>{a.text.split('=')[-1]}</b>"))
        
        st.plotly_chart(fig_pos, use_container_width=True)

    else:
        st.info("💡 Start by adding players in the section above to visualize your team's performance.")

# TAB 1: Overview
with tab1:
    st.markdown("## 📊 Player Performance Analysis")
    st.caption("Compare players across time and analyze historical performance")
    
    if 'comparison_list' not in st.session_state:
        st.session_state.comparison_list = players_df.sort_values('total_points', ascending=False)['name'].head(2).tolist()

    # --- 1. Selection interface ---
    st.markdown("### 🔍 Add Players to Compare")
    col1, col2, col3, col4 = st.columns([2, 2, 3, 1])
    
    with col1:
        teams = sorted(players_df['team'].unique())
        sel_team = st.selectbox("Filter by Team", ["All Teams"] + teams, key="overview_team")
        team_mask = pd.Series([True] * len(players_df)) if sel_team == "All Teams" else players_df['team'] == sel_team
    
    with col2:
        positions = sorted(players_df[team_mask]['position_name'].unique())
        sel_pos = st.selectbox("Position", positions, key="overview_pos")
        pos_mask = players_df['position_name'] == sel_pos
        
    with col3:
        names = sorted(players_df[team_mask & pos_mask]['name'].unique())
        sel_player = st.selectbox("Select Player", names, key="overview_player")

    with col4:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("➕ Add", use_container_width=True, type="primary"):
            if sel_player not in st.session_state.comparison_list:
                st.session_state.comparison_list.append(sel_player)
                st.rerun()

    # --- 2. Display selected players (Pills) ---
    st.markdown("""
        <style>
        div[data-testid="column"] button {
            border-radius: 20px !important;
            padding: 4px 12px !important;
            border: 1px solid #e0e0e0 !important;
            background-color: #f8f9fa !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    if st.session_state.comparison_list:
        st.markdown("### 👥 Selected Players")
        with st.container():
            pills_row = st.columns([1] * len(st.session_state.comparison_list) + [0.5, 2])
            for i, p_name in enumerate(st.session_state.comparison_list):
                last_name = p_name.split()[-1]
                with pills_row[i]:
                    if st.button(f"{last_name}", key=f"pill_{p_name}", icon=":material/close:"):
                        st.session_state.comparison_list.remove(p_name)
                        st.rerun()
            
            with pills_row[len(st.session_state.comparison_list)]:
                if st.button("🗑️", type="tertiary", help="Clear all selections"):
                    st.session_state.comparison_list = []
                    st.rerun()

    st.divider()

    # --- 3. NEW: Historical Time-Series Analysis ---
    if st.session_state.comparison_list:
        st.markdown("### 📈 Historical Trends")
        st.caption("Tracking performance progression throughout the season")
        
        with st.spinner("Fetching live historical data..."):
            # Call the function you provided
            fig_cum, fig_weekly = pl.plot_player_performance_timeseries(
                st.session_state.comparison_list, 
                players_df
            )
            
            if fig_cum and fig_weekly:
                # Layout the two charts side-by-side or stacked
                # We'll use tabs to keep the UI clean
                chart_tab1, chart_tab2 = st.tabs(["Total Growth", "Weekly Performance"])
                
                with chart_tab1:
                    st.plotly_chart(fig_cum, use_container_width=True)
                with chart_tab2:
                    st.plotly_chart(fig_weekly, use_container_width=True)
            else:
                st.info("No historical data found for selected players.")
    
    st.divider()

    # --- 4. Value Analysis (Original) ---
    st.markdown("### 💰 Player Value Analysis")
    st.caption("Explore the relationship between player cost and performance")
    try:
        fig_vfm = pl.plot_points_vs_cost(players_df, positions_df)
        st.plotly_chart(fig_vfm, use_container_width=True)
    except Exception as e:
        st.error(f"❌ Error loading chart: {e}")

# TAB 2: Build Squad
with tab2:
    st.markdown("## 🎯 Build Your Optimal Squad")
    st.caption("Set your constraints and let AI find the best team for you")

    if 'players_df' in locals() or 'players_df' in globals():
        # Ensure required columns exist
        if 'name' not in players_df.columns:
            players_df['name'] = players_df['first_name'] + ' ' + players_df['second_name']
        if 'position_name' not in players_df.columns:
            pos_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
            players_df['position_name'] = players_df['element_type'].map(pos_map)

        # Initialize session state
        if 'players_to_keep' not in st.session_state:
            st.session_state.players_to_keep = []
        if 'players_to_exclude' not in st.session_state:
            st.session_state.players_to_exclude = []

        # Header with reset button
        col_header, col_reset = st.columns([3, 1])
        with col_header:
            st.markdown("### 👥 Player Constraints")
            st.caption("Force include or exclude specific players from your squad")
        with col_reset:
            if st.button("🗑️ Reset", use_container_width=True):
                st.session_state.players_to_keep = []
                st.session_state.players_to_exclude = []
                st.rerun()

        # Summary metrics
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""
                <div class='stat-card'>
                    <div class='stat-value'>{len(st.session_state.players_to_keep)}</div>
                    <div class='stat-label'>🔒 Locked In</div>
                </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
                <div class='stat-card'>
                    <div class='stat-value'>{len(st.session_state.players_to_exclude)}</div>
                    <div class='stat-label'>🚫 Excluded</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Team and Player Selection
        teams_list = sorted(players_df['team'].unique().tolist())
        col1, col2 = st.columns(2)

        # LEFT: Include Players
        with col1:
            st.markdown("""
                <div class='modern-card'>
                    <h4 style='margin-top: 0; color: #28a745;'>🔒 Force Include</h4>
                    <p style='color: #6c757d; font-size: 0.9rem;'>Players that must be in your squad</p>
                </div>
            """, unsafe_allow_html=True)
            
            sel_team_inc = st.selectbox("Select Team", [""] + teams_list, key="inc_team")
            
            if sel_team_inc:
                team_p = players_df[players_df['team'] == sel_team_inc].sort_values('web_name')
                p_options = [f"{row['web_name']} ({row['position_name']} - £{row['now_cost']:.1f}m)" 
                             for _, row in team_p.iterrows()]
                
                selected = st.multiselect("Choose Players", options=p_options, key="inc_p_multi")
                
                if st.button("➕ Add to Squad", key="btn_add_inc", use_container_width=True):
                    for item in selected:
                        p_web_name = item.split(" (")[0]
                        full_name = team_p[team_p['web_name'] == p_web_name]['name'].values[0]
                        if full_name not in st.session_state.players_to_keep:
                            st.session_state.players_to_keep.append(full_name)
                    st.rerun()

            # Display Keep List
            if st.session_state.players_to_keep:
                st.markdown("<br>", unsafe_allow_html=True)
                for p_name in st.session_state.players_to_keep:
                    p_data = players_df[players_df['name'] == p_name].iloc[0]
                    col_a, col_b = st.columns([5, 1])
                    with col_a:
                        st.markdown(f"""
                            <div class='success-banner' style='padding: 0.75rem; margin-bottom: 0.5rem;'>
                                <strong>{p_name}</strong> ({p_data['team']})
                            </div>
                        """, unsafe_allow_html=True)
                    with col_b:
                        if st.button("✕", key=f"del_inc_{p_name}"):
                            st.session_state.players_to_keep.remove(p_name)
                            st.rerun()

        # RIGHT: Exclude Players
        with col2:
            st.markdown("""
                <div class='modern-card'>
                    <h4 style='margin-top: 0; color: #dc3545;'>🚫 Force Exclude</h4>
                    <p style='color: #6c757d; font-size: 0.9rem;'>Players to avoid in optimization</p>
                </div>
            """, unsafe_allow_html=True)
            
            sel_team_exc = st.selectbox("Select Team", [""] + teams_list, key="exc_team")
            
            if sel_team_exc:
                team_p = players_df[players_df['team'] == sel_team_exc].sort_values('web_name')
                p_options = [f"{row['web_name']} ({row['position_name']} - £{row['now_cost']:.1f}m)" 
                             for _, row in team_p.iterrows()]
                
                selected = st.multiselect("Choose Players", options=p_options, key="exc_p_multi")
                
                if st.button("➕ Add to Exclusions", key="btn_add_exc", use_container_width=True):
                    for item in selected:
                        p_web_name = item.split(" (")[0]
                        full_name = team_p[team_p['web_name'] == p_web_name]['name'].values[0]
                        if full_name not in st.session_state.players_to_exclude:
                            st.session_state.players_to_exclude.append(full_name)
                    st.rerun()

            # Display Exclude List
            if st.session_state.players_to_exclude:
                st.markdown("<br>", unsafe_allow_html=True)
                for p_name in st.session_state.players_to_exclude:
                    p_data = players_df[players_df['name'] == p_name].iloc[0]
                    col_a, col_b = st.columns([5, 1])
                    with col_a:
                        st.markdown(f"""
                            <div class='warning-banner' style='padding: 0.75rem; margin-bottom: 0.5rem;'>
                                <strong>{p_name}</strong> ({p_data['team']})
                            </div>
                        """, unsafe_allow_html=True)
                    with col_b:
                        if st.button("✕", key=f"del_exc_{p_name}"):
                            st.session_state.players_to_exclude.remove(p_name)
                            st.rerun()

        players_to_keep = st.session_state.players_to_keep
        players_to_exclude = st.session_state.players_to_exclude

    else:
        st.error("Please load the FPL data first.")

    st.markdown("---")
    
    # Run Optimization
    st.markdown("### 🚀 Generate Optimal Squad")
    run_opt = st.button("⚡ Optimize My Team", type="primary", use_container_width=True)

    if run_opt:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.info("🔍 Step 1/3: Analyzing fixture difficulty and player form...")
        forecast_df = pl.fetch_and_forecast_players()
        progress_bar.progress(33)
        
        status_text.info("🧠 Step 2/3: Running optimization algorithm...")
        selected_names, model, fig_optimization = pl.optimize_fpl_team(
            forecast_df,
            st.session_state.players_to_keep,
            st.session_state.players_to_exclude
        )
        progress_bar.progress(66)
        
        status_text.info("📋 Step 3/3: Selecting optimal Starting XI...")
        selected_df = forecast_df[forecast_df["name"].isin(selected_names)].copy()
        POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
        selected_df["position_name"] = selected_df["element_type"].map(POSITION_MAP)
        
        starting_11 = pl.optimize_starting_11(selected_df)
        
        st.session_state["selected_squad_df"] = selected_df
        st.session_state["starting_11_df"] = starting_11
        st.session_state["fig_optimization"] = fig_optimization
        
        progress_bar.progress(100)
        status_text.success(f"✅ Optimization Complete: {LpStatus[model.status]}")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()

    # Display Results
    if "selected_squad_df" in st.session_state:
        selected_df = st.session_state["selected_squad_df"]
        starting_11 = st.session_state["starting_11_df"]

        st.markdown("---")
        
        # Formation calculation
        counts = starting_11['position_name'].value_counts()
        formation_str = f"{counts.get('DEF', 0)}-{counts.get('MID', 0)}-{counts.get('FWD', 0)}"

        # Squad Summary
        st.markdown("## 📊 Squad Summary")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.markdown(f"""
                <div class='stat-card'>
                    <div class='stat-value'>£{selected_df['now_cost'].sum():.1f}m</div>
                    <div class='stat-label'>Squad Cost</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class='stat-card'>
                    <div class='stat-value'>{selected_df['projected_points'].sum():.0f}</div>
                    <div class='stat-label'>Total Points</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class='stat-card'>
                    <div class='stat-value'>{starting_11['projected_points'].sum():.0f}</div>
                    <div class='stat-label'>XI Points</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
                <div class='stat-card'>
                    <div class='stat-value'>{formation_str}</div>
                    <div class='stat-label'>Formation</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
                <div class='stat-card'>
                    <div class='stat-value'>{len(selected_df)}/15</div>
                    <div class='stat-label'>Squad Size</div>
                </div>
            """, unsafe_allow_html=True)

        # Pitch Visualization
        st.markdown("---")
        st.markdown("## ⚽ Starting XI Formation")
        fig_pitch = create_pitch_visualization(selected_df, starting_11)
        st.plotly_chart(fig_pitch, use_container_width=True)

        # Bench Section
        st.markdown("---")
        st.markdown("## 🪑 Bench Players")
        st.caption("Ordered by projected points (GK always last)")

        bench_df = selected_df[~selected_df["name"].isin(starting_11["name"])]
        outfield_bench = bench_df[bench_df['position_name'] != 'GKP'].sort_values("projected_points", ascending=False)
        gk_bench = bench_df[bench_df['position_name'] == 'GKP']
        final_bench = pd.concat([outfield_bench, gk_bench])

        cols = st.columns(4)
        for i, (_, p) in enumerate(final_bench.iterrows()):
            with cols[i % 4]:
                rank_label = f"#{i+1}" if p['position_name'] != 'GKP' else "GK"
                
                st.markdown(f"""
                    <div class='bench-player-card'>
                        <div style='position: absolute; top: 1rem; right: 1rem; font-weight: 700; color: #37003c;'>{rank_label}</div>
                        <div style='font-weight: 700; font-size: 1.2rem; color: #37003c; margin-bottom: 0.5rem;'>{p['name'].split()[-1]}</div>
                        <div style='font-size: 0.9rem; color: #6c757d;'>{p['position_name']} | {p['team']}</div>
                        <div style='margin-top: 1rem; padding-top: 1rem; border-top: 2px solid #e8eaed;'>
                            <span style='font-weight: 700; font-size: 1.1rem; color: #28a745;'>{p['projected_points']:.1f} pts</span>
                            <span style='color: #6c757d;'> | £{p['now_cost']:.1f}m</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

# TAB 3: Analytics
with tab3:
    st.markdown("## 📈 Advanced Analytics")
    st.caption("Deep dive into squad performance and optimization details")
    
    if 'selected_squad_df' in st.session_state:
        selected_df = st.session_state['selected_squad_df']
        starting_11 = st.session_state.get('starting_11_df', pd.DataFrame())
        
        # Optimization Results
        st.markdown("### 🎯 Optimization Visualization")
        try:
            if 'fig_optimization' in st.session_state:
                st.plotly_chart(st.session_state['fig_optimization'], use_container_width=True)
            else:
                st.info("📊 Optimization chart will appear after running the optimizer")
        except:
            st.info("📊 Run optimization to see visualization")
        
        st.markdown("---")
        
        # Squad Table
        st.markdown("### 📋 Complete Squad Breakdown")
        sorted_squad_df = selected_df.sort_values(
            by=['element_type', 'projected_points'], 
            ascending=[True, False]
        )
        
        display_df = sorted_squad_df[[
            'name', 'position_name', 'team', 'now_cost', 
            'projected_points', 'fixture_multiplier', 'form'
        ]].copy()
        
        display_df.columns = ['Player', 'Position', 'Team', 'Cost (£m)', 'Proj. Points', 'Fixture', 'Form']
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Cost (£m)": st.column_config.NumberColumn(format="£%.1f"),
                "Proj. Points": st.column_config.NumberColumn(format="%.1f"),
                "Fixture": st.column_config.NumberColumn(format="%.2f"),
                "Form": st.column_config.NumberColumn(format="%.1f")
            }
        )
        
        st.markdown("---")

        # Captain Selection
        st.markdown("### 🔴 Captain Recommendations")
        st.caption("Your captain scores double points - choose wisely!")
        
        if not starting_11.empty:
            captain_options = starting_11.sort_values('projected_points', ascending=False)
            
            col1, col2 = st.columns([2, 1])
            
            POSITION_MAP = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
            if 'element_type' not in captain_options.columns:
                captain_options = captain_options.merge(
                    selected_df[['name', 'element_type']], on='name', how='left'
                )
            if 'position_name' not in captain_options.columns:
                captain_options['position_name'] = captain_options['element_type'].map(POSITION_MAP)
            
            with col1:
                st.markdown("#### 🏆 Top 5 Captain Picks")
                top_5 = captain_options.head(5)[['name', 'position_name', 'team', 'projected_points', 'form']].copy()
                
                for idx, (_, row) in enumerate(top_5.iterrows()):
                    pts = float(row['projected_points'])
                    double_pts = pts * 2
                    
                    if idx == 0:
                        badge = "🥇"
                        bg_color = "linear-gradient(135deg, #ffd700 0%, #ffed4e 100%)"
                    elif idx == 1:
                        badge = "🥈"
                        bg_color = "linear-gradient(135deg, #e8e8e8 0%, #f5f5f5 100%)"
                    elif idx == 2:
                        badge = "🥉"
                        bg_color = "linear-gradient(135deg, #cd7f32 0%, #e8a87c 100%)"
                    else:
                        badge = "⭐"
                        bg_color = "white"
                    
                    st.markdown(f"""
                        <div class='modern-card' style='background: {bg_color}; margin-bottom: 0.8rem;'>
                            <div style='display: flex; justify-content: space-between; align-items: center;'>
                                <div>
                                    <span style='font-size: 1.5rem;'>{badge}</span>
                                    <strong style='font-size: 1.1rem; color: #37003c;'> {row['name']}</strong>
                                    <span style='color: #666; margin-left: 0.5rem;'>({row['position_name']} - {row['team']})</span>
                                </div>
                                <div style='text-align: right;'>
                                    <div style='font-size: 1.3rem; font-weight: 700; color: #28a745;'>{double_pts:.1f} pts</div>
                                    <div style='font-size: 0.85rem; color: #666;'>with captaincy</div>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### 💡 Captain Tips")
                st.markdown("""
                    <div class='info-banner'>
                        <strong>Key Factors:</strong>
                        <ul style='margin: 0.5rem 0; padding-left: 1.5rem;'>
                            <li>Fixture difficulty</li>
                            <li>Recent form</li>
                            <li>Home advantage</li>
                            <li>Historical returns</li>
                        </ul>
                        <strong>Pro Tip:</strong><br>
                        <small>Also select a vice-captain as backup!</small>
                    </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")

        # Squad Analysis
        st.markdown("### ⚽ Squad Distribution Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📊 By Position")
            pos_counts = selected_df['position_name'].value_counts()
            for pos, count in pos_counts.items():
                st.markdown(f"""
                    <div class='stat-card' style='margin-bottom: 0.8rem;'>
                        <div class='stat-value'>{count}</div>
                        <div class='stat-label'>{pos}</div>
                    </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 🏟️ By Team")
            team_counts = selected_df['team'].value_counts().head(5)
            for team, count in team_counts.items():
                st.markdown(f"""
                    <div class='modern-card' style='padding: 0.8rem; margin-bottom: 0.5rem;'>
                        <strong style='color: #37003c;'>{team}</strong>: {count} player(s)
                    </div>
                """, unsafe_allow_html=True)
            
            if len(selected_df['team'].unique()) > 5:
                others = len(selected_df['team'].unique()) - 5
                st.caption(f"+ {others} other team(s)")
        
        with col3:
            st.markdown("#### 💰 Cost Analysis")
            avg_cost = selected_df['now_cost'].mean()
            max_cost = selected_df['now_cost'].max()
            min_cost = selected_df['now_cost'].min()
            
            st.markdown(f"""
                <div class='stat-card' style='margin-bottom: 0.8rem;'>
                    <div class='stat-value'>£{avg_cost:.1f}m</div>
                    <div class='stat-label'>Average</div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class='modern-card' style='padding: 0.6rem; margin-bottom: 0.5rem;'>
                    <strong>Most Expensive:</strong> £{max_cost:.1f}m
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class='modern-card' style='padding: 0.6rem;'>
                    <strong>Cheapest:</strong> £{min_cost:.1f}m
                </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class='warning-banner'>
                <h3 style='margin:0; color:#856404;'>⚠️ No Squad Data</h3>
                <p style='margin:0.5rem 0 0 0;'>
                    Please run the Squad Optimizer first to view analytics.
                </p>
            </div>
        """, unsafe_allow_html=True)

# TAB 4: AI Advisor
with tab4:
    st.markdown("## 🤖 AI Squad Advisor")
    st.caption("Get intelligent insights and recommendations for your FPL team")
    
    if 'selected_squad_df' in st.session_state:
        st.markdown("""
            <div class='info-banner'>
                <strong>💬 Ask me anything about your squad!</strong><br>
                <p style='margin: 0.5rem 0 0 0;'>Examples: "Who should I captain?", "Any transfer suggestions?", "Is my defense strong enough?"</p>
            </div>
        """, unsafe_allow_html=True)
        
        advisor_question = st.text_input(
            "Your Question",
            placeholder="e.g., Who should I make captain this gameweek?",
            label_visibility="collapsed"
        )
        
        if advisor_question:
            with st.spinner("🤔 Analyzing your squad..."):
                response = pl.fpl_langchain_advisor(advisor_question, st.session_state['selected_squad_df'])
                
                st.markdown(f"""
                    <div class='success-banner'>
                        <h4 style='margin:0; color:#155724;'>🤖 AI Response:</h4>
                        <p style='margin:0.5rem 0 0 0; color:#155724; line-height:1.8;'>{response}</p>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class='warning-banner'>
                <h3 style='margin:0; color:#856404;'>⚠️ Optimization Required</h3>
                <p style='margin:0.5rem 0 0 0;'>
                    Please run Squad Optimization first to enable the AI Advisor.
                </p>
            </div>
        """, unsafe_allow_html=True)

# TAB 5: Injury News
with tab5:
    st.markdown("## 🚑 Latest Injury & Team News")
    st.caption("Stay updated with the latest injury reports and team news")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📰 Fetch Latest News", type="primary", use_container_width=True):
            with st.spinner("🔍 Gathering news from multiple sources..."):
                news = pl.get_fpl_injury_news()
                
                st.markdown(f"""
                    <div class='modern-card'>
                        <h4 style='margin:0; color:#37003c;'>🚑 Injury News Summary</h4>
                        <hr style='margin:1rem 0; border:none; height:1px; background:#e8eaed;'>
                        <div style='color:#202124; line-height:1.8;'>{news}</div>
                    </div>
                """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class='info-banner'>
            <h4 style='margin-top: 0;'>💡 Why Check Injury News?</h4>
            <ul style='margin: 0.5rem 0; padding-left: 1.5rem;'>
                <li>Avoid selecting injured or doubtful players</li>
                <li>Find differential picks when key players are out</li>
                <li>Plan transfers based on return dates</li>
                <li>Stay ahead of price changes</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align:center; padding:2rem; color:#5f6368;'>
        <p style='margin:0; font-size:0.95rem; font-weight: 500;'>
            💜 Built with <strong>Streamlit</strong> | 
            Data from <strong>Official FPL API</strong> | 
        </p>
        <p style='margin:0.5rem 0 0 0; font-size:0.85rem; opacity:0.8;'>
            © 2025 FPL AI Optimizer | Made with ⚽ for FPL managers
        </p>
    </div>
""", unsafe_allow_html=True)