import streamlit as st
import pl
import pandas as pd
import numpy as np
import os
from pulp import LpStatus
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

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
    players_df, injuries_df, positions_df = pl.load_fpl_data()
    return players_df, injuries_df, positions_df

@st.cache_data(ttl=3600)
def get_next_gw():
    return pl.get_next_gw()

@st.cache_data(ttl=3600)
def get_fixture_info() -> "pd.DataFrame":
    return pl.get_next_fixture_info()


# ── UI helpers ─────────────────────────────────────────────────────────────────

_POS_BAR_COLOR  = {"GKP": "#FFD700", "DEF": "#00B4D8", "MID": "#7B2FBE", "FWD": "#E63946"}
_POS_BADGE_CLS  = {"GKP": "pos-gkp", "DEF": "pos-def", "MID": "pos-mid", "FWD": "pos-fwd"}

def _bench_card_html(p: "pd.Series", rank_label: str) -> str:
    """Return HTML for a single bench player card (bench-v2 style)."""
    pos       = str(p.get("position_name", "MID"))
    bar_color = _POS_BAR_COLOR.get(pos, "#888")
    badge_cls = _POS_BADGE_CLS.get(pos, "pos-mid")
    name      = p.get("name", "")
    team      = p.get("team", "")
    pts       = float(p.get("projected_points", 0))
    cost      = float(p.get("now_cost", 0))
    return f"""
    <div class='bench-v2'>
        <div class='bench-v2-bar' style='background:{bar_color};'></div>
        <div class='bench-v2-body'>
            <div class='bench-rank'>{rank_label}</div>
            <div class='bench-player-name'>{name}</div>
            <div class='bench-meta'>
                <span class='pos-badge {badge_cls}'>{pos}</span>&nbsp;{team}
            </div>
            <div class='bench-footer'>
                <span class='bench-pts-val'>{pts:.1f} pts</span>
                <span class='bench-cost-val'>£{cost:.1f}m</span>
            </div>
        </div>
    </div>"""


# --- 3. STARTUP LOADING SCREEN ---
if 'app_ready' not in st.session_state:
    with st.status("🚀 Initializing FPL Optimizer...", expanded=True) as status:
        st.write("📡 Connecting to FPL API and loading player data...")
        players_df, injuries_df, positions_df = load_data()
        st.write("✅ Data loaded successfully.")
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
        padding: 1.25rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.25rem;
        box-shadow: 0 6px 24px rgba(55, 0, 60, 0.25);
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
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        position: relative;
        z-index: 1;
    }

    .hero-subtitle {
        color: rgba(255, 255, 255, 0.85);
        font-size: 0.95rem;
        margin-top: 0.4rem;
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

    /* ── Position badges ─────────────────────────────── */
    .pos-badge {
        display: inline-block;
        padding: 0.18rem 0.52rem;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.4px;
        text-transform: uppercase;
        line-height: 1.4;
    }
    .pos-gkp { background: #fff3cd; color: #856404; }
    .pos-def { background: #cfe2ff; color: #084298; }
    .pos-mid { background: #ead6fd; color: #5a0098; }
    .pos-fwd { background: #ffd6d6; color: #842029; }

    /* ── Bench card v2 ───────────────────────────────── */
    .bench-v2 {
        background: white;
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid #e8eaed;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        transition: all 0.25s ease;
        margin-bottom: 0.75rem;
    }
    .bench-v2:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
        border-color: rgba(55,0,60,0.25);
    }
    .bench-v2-bar   { height: 4px; }
    .bench-v2-body  { padding: 0.9rem 1.1rem 1rem; }
    .bench-rank     {
        font-size: 0.68rem; font-weight: 700; color: #9e9e9e;
        text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.3rem;
    }
    .bench-player-name {
        font-weight: 700; font-size: 0.95rem; color: #202124;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .bench-meta     { font-size: 0.8rem; color: #6c757d; margin-top: 0.25rem; }
    .bench-footer   {
        margin-top: 0.6rem; padding-top: 0.6rem;
        border-top: 1px solid #f1f3f4;
        display: flex; justify-content: space-between; align-items: center;
    }
    .bench-pts-val  { font-weight: 700; font-size: 0.95rem; color: #1e8a4b; }
    .bench-cost-val { font-size: 0.82rem; color: #6c757d; font-weight: 500; }

    /* ── Transfer cards ──────────────────────────────── */
    .transfer-in {
        background: linear-gradient(135deg, #f0fff4, #e6ffed);
        border-left: 4px solid #28a745;
        border-radius: 10px; padding: 0.75rem 1rem; margin-bottom: 0.5rem;
    }
    .transfer-out {
        background: linear-gradient(135deg, #fff5f0, #ffe8dc);
        border-left: 4px solid #fd7e14;
        border-radius: 10px; padding: 0.75rem 1rem; margin-bottom: 0.5rem;
    }
    .transfer-player { font-weight: 700; font-size: 0.95rem; color: #202124; }
    .transfer-meta   { font-size: 0.8rem; color: #555; margin-top: 0.2rem; }

    /* ── GW live pill (hero) ─────────────────────────── */
    .gw-live {
        display: inline-flex; align-items: center; gap: 0.4rem;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.28);
        color: white; padding: 0.32rem 0.85rem;
        border-radius: 20px; font-size: 0.82rem; font-weight: 600;
        backdrop-filter: blur(4px);
    }

    /* ── Empty state ─────────────────────────────────── */
    .empty-state {
        text-align: center; padding: 2.5rem 1.5rem;
        background: #fafbfc; border-radius: 16px;
        border: 2px dashed #dee2e6;
    }
    .empty-state-icon  { font-size: 2.5rem; margin-bottom: 0.75rem; }
    .empty-state-title { font-weight: 700; color: #495057; font-size: 1rem; }
    .empty-state-body  { color: #6c757d; font-size: 0.88rem; margin-top: 0.35rem; }

    /* ── Transfer arrow ──────────────────────────────── */
    .transfer-arrow-col {
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        gap: 0.5rem; padding-top: 2rem;
    }
    .transfer-arrow-icon { font-size: 2rem; opacity: 0.35; }
    </style>
""", unsafe_allow_html=True)

# Fetch early so the hero f-string can reference it (cached, no extra API call)
next_gw = get_next_gw()

# Hero header — includes live GW pill
st.markdown(f"""
    <div class='hero-header'>
        <div style='display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:0.75rem;'>
            <div>
                <h1 class='hero-title'>⚽ FPL AI Optimizer</h1>
                <p class='hero-subtitle'>
                    Build your dream team with AI-powered analytics and LP optimisation
                </p>
            </div>
            <div style='display:flex; flex-direction:column; align-items:flex-end; gap:0.5rem; padding-top:0.25rem;'>
                <span class='gw-live'>📅 Gameweek {next_gw}</span>
                <span class='hero-badge'>🤖 AI-Powered</span>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Load data
players_df, injuries_df, positions_df = load_data()
next_gw = get_next_gw()

# Algorithm defaults — overridden by sidebar sliders each run
xg_blend       = 0.35
baseline_blend = 0.25

def create_pitch_visualization(squad_df, starting_11_df=None):
    """Enhanced pitch visualization with better aesthetics"""

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
        height=720,
        margin=dict(l=20, r=20, t=90, b=20),
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

    gw_col, btn_col = st.columns([3, 2])
    with gw_col:
        st.metric("Next Gameweek", f"GW {next_gw}")
    with btn_col:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Refresh", use_container_width=True, key="sidebar_refresh"):
            st.cache_data.clear()
            st.rerun()

    st.markdown("---")
    st.markdown("#### 📊 My Squad")
    if "selected_squad_df" in st.session_state:
        _sq = st.session_state["selected_squad_df"]
        _budget_used = _sq["now_cost"].sum()
        _sb1, _sb2 = st.columns(2)
        _sb1.metric("Players",     f"{len(_sq)}/15")
        _sb2.metric("Cost",        f"£{_budget_used:.1f}m")
        _sb3, _sb4 = st.columns(2)
        _sb3.metric("Proj Pts",    f"{_sq['projected_points'].sum():.0f}")
        _sb4.metric("Budget Left", f"£{100.0 - _budget_used:.1f}m")
    else:
        st.caption("Run the optimizer to see squad stats.")

    st.markdown("---")

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
        st.markdown("**🧬 Algorithm Tuning**")
        st.caption(
            "The model blends 5-GW form (decay-weighted) with xG/xA signal "
            "and a season PPG baseline. Adjust the balance here."
        )
        xg_blend = st.slider(
            "xG Signal Weight",
            min_value=0.0, max_value=1.0, value=0.35, step=0.05,
            key="algo_xg_blend",
            help=(
                "0 = use raw GW points only · 1 = use xG/xA model only\n"
                "Higher values reduce luck noise at the cost of ignoring bonuses/cards."
            ),
        )
        baseline_blend = st.slider(
            "Season Baseline Weight",
            min_value=0.0, max_value=1.0, value=0.25, step=0.05,
            key="algo_baseline_blend",
            help=(
                "0 = pure recent form · 1 = pure season PPG average\n"
                "Higher values make the model more conservative."
            ),
        )
        st.caption(
            f"ℹ️ Final projection = "
            f"**{(1-baseline_blend)*100:.0f}%** 5-GW form × "
            f"(**{(1-xg_blend)*100:.0f}%** raw pts + **{xg_blend*100:.0f}%** xG model) "
            f"+ **{baseline_blend*100:.0f}%** season PPG"
        )

        st.markdown("---")
        st.caption("""
        **AI Features Require API Keys:**
        - `GOOGLE_API_KEY`
        - `GOOGLE_CSE_ID`

        Configure in Streamlit Secrets or environment variables.
        """)


# --- MAIN CONTENT ---
tab_sporting_dir, tab_analytics, tab_recruit_perf, tab5, tab4 = st.tabs([
    "🏆 Sporting Director",
    "📊 Analytics",
    "🎯 Squad Builder",
    "🏥 Injury News",
    "🤖 AI Advisor",
])

# ── TAB: SPORTING DIRECTOR ────────────────────────────────────────────────────
with tab_sporting_dir:
    st.markdown("## 🏆 Sporting Director")
    st.caption("Pre-computed optimal squad — unconstrained, pure AI selection")

    BASELINE_PATH = Path("./input/base_squad_result.parquet")

    # ── Load & validate ────────────────────────────────────────────────────
    if not BASELINE_PATH.exists():
        st.markdown("""
            <div class='warning-banner'>
                <h3 style='margin:0; color:#856404;'>⚠️ No Baseline Squad Found</h3>
                <p style='margin:0.5rem 0 0 0;'>
                    Click below to fetch live FPL data and generate an unconstrained baseline squad.
                    This takes ~2 minutes as it fetches history for every player.
                </p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Generate Baseline Squad", type="primary", use_container_width=True, key="gen_baseline_btn"):
            BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with st.status("⚙️ Generating baseline squad...", expanded=True) as _gen_status:
                st.write("📡 Step 1/3 — Fetching fixture difficulty and recent form...")
                pl.run_and_save_base_optimization(str(BASELINE_PATH))
                st.write("✅ Baseline squad saved!")
                _gen_status.update(label="✅ Ready!", state="complete", expanded=False)
            st.cache_data.clear()
            st.rerun()
    else:
        @st.cache_data(ttl=3600, show_spinner="Loading baseline squad...")
        def load_baseline(path: str) -> pd.DataFrame:
            df = pd.read_parquet(path)
            if "position_name" not in df.columns:
                df["position_name"] = df["element_type"].map({1:"GKP",2:"DEF",3:"MID",4:"FWD"})
            return df

        baseline_df    = load_baseline(str(BASELINE_PATH))
        baseline_xi    = baseline_df[baseline_df["in_starting_11"] == True].copy()
        baseline_bench = baseline_df[baseline_df["in_starting_11"] == False].copy()

        # ── Header row: file freshness + refresh ──────────────────────────────
        mtime = pd.Timestamp(BASELINE_PATH.stat().st_mtime, unit="s").strftime("%d %b %Y, %H:%M")
        hcol1, hcol2 = st.columns([4, 1])
        with hcol1:
            st.caption(f"📅 Last generated: **{mtime}**")
        with hcol2:
            if st.button("🔄 Refresh Data", use_container_width=True, key="sd_refresh"):
                st.cache_data.clear()
                st.rerun()

        # ── Squad summary metrics ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 📊 Squad Summary")

        counts        = baseline_xi["position_name"].value_counts()
        formation_str = f"{counts.get('DEF',0)}-{counts.get('MID',0)}-{counts.get('FWD',0)}"
        budget_remain = 100.0 - baseline_df["now_cost"].sum()

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Squad Cost",  f"£{baseline_df['now_cost'].sum():.1f}m")
        c2.metric("Remaining",   f"£{budget_remain:.1f}m")
        c3.metric("Total Pts",   f"{baseline_df['projected_points'].sum():.0f}")
        c4.metric("XI Pts",      f"{baseline_xi['projected_points'].sum():.0f}")
        c5.metric("Formation",   formation_str)
        c6.metric("Squad Size",  f"{len(baseline_df)}/15")

        # ── Pitch ──────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## ⚽ Starting XI")
        st.plotly_chart(
            create_pitch_visualization(baseline_df, baseline_xi),
            use_container_width=True, key="sd_pitch_1"
        )

        # ── Captain picks ──────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 🔴 Captain Recommendations")
        st.caption("Top performers from the unconstrained optimal XI")

        top_5 = baseline_xi.sort_values("projected_points", ascending=False).head(5)
        cap_col1, cap_col2 = st.columns([2, 1])

        with cap_col1:
            st.markdown("##### 🏆 Top 5 Picks")
            BADGES = [
                ("🥇", "linear-gradient(135deg, #ffd700 0%, #ffed4e 100%)"),
                ("🥈", "linear-gradient(135deg, #e8e8e8 0%, #f5f5f5 100%)"),
                ("🥉", "linear-gradient(135deg, #cd7f32 0%, #e8a87c 100%)"),
                ("⭐", "white"),
                ("⭐", "white"),
            ]
            for idx, (_, row) in enumerate(top_5.iterrows()):
                badge, bg = BADGES[idx]
                base_pts   = float(row["projected_points"])
                double_pts = base_pts * 2
                pos        = str(row.get("position_name", ""))
                pos_badge  = f"<span class='pos-badge {_POS_BADGE_CLS.get(pos, '')}' style='vertical-align:middle;'>{pos}</span>"
                st.markdown(f"""
                    <div class='modern-card' style='background:{bg}; margin-bottom:0.75rem;'>
                        <div style='display:flex; justify-content:space-between; align-items:center; gap:0.5rem;'>
                            <div style='display:flex; align-items:center; gap:0.6rem; flex:1; min-width:0;'>
                                <span style='font-size:1.4rem; flex-shrink:0;'>{badge}</span>
                                <div style='min-width:0;'>
                                    <div style='font-weight:700; font-size:1rem; color:#37003c;
                                                white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>
                                        {row['name']}
                                    </div>
                                    <div style='margin-top:0.2rem;'>
                                        {pos_badge}
                                        <span style='color:#6c757d; font-size:0.82rem; margin-left:0.35rem;'>
                                            {row['team']}
                                        </span>
                                    </div>
                                </div>
                            </div>
                            <div style='text-align:right; flex-shrink:0;'>
                                <div style='font-size:1.2rem; font-weight:800; color:#1e8a4b;'>{double_pts:.1f}</div>
                                <div style='font-size:0.75rem; color:#6c757d; white-space:nowrap;'>
                                    {base_pts:.1f} × 2 (C)
                                </div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        with cap_col2:
            st.markdown("##### 💡 About This Squad")
            st.markdown("""
                <div class='info-banner'>
                    <strong>Sporting Director mode:</strong>
                    <ul style='margin:0.5rem 0; padding-left:1.5rem;'>
                        <li>No forced inclusions</li>
                        <li>No exclusions applied</li>
                        <li>Pure LP optimisation</li>
                        <li>Max value within £100m</li>
                    </ul>
                    <strong>Use as your benchmark</strong> — compare against your
                    constrained squad in the Recruitment Hub to see the true cost
                    of your preferences.
                </div>
            """, unsafe_allow_html=True)

        # ── Bench ──────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 🪑 Bench")
        st.caption("Ordered by projected points — GK always last")

        outfield_bench = (
            baseline_bench[baseline_bench["position_name"] != "GKP"]
            .sort_values("projected_points", ascending=False)
        )
        gk_bench    = baseline_bench[baseline_bench["position_name"] == "GKP"]
        final_bench = pd.concat([outfield_bench, gk_bench]).reset_index(drop=True)

        bench_cols = st.columns(4)
        for i, (_, p) in enumerate(final_bench.iterrows()):
            with bench_cols[i % 4]:
                rank_label = f"#{i+1}" if p["position_name"] != "GKP" else "GK"
                st.markdown(_bench_card_html(p, rank_label), unsafe_allow_html=True)

        # ── Full squad table ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 📋 Full Squad Breakdown")
        st.caption("Difficulty: 1 = easiest · 5 = hardest  |  DGW teams show both fixtures")

        _fix = get_fixture_info()
        _enrich = baseline_df.copy()
        # convert string columns the FPL API returns as strings
        for _col in ["form", "points_per_game"]:
            if _col in _enrich.columns:
                _enrich[_col] = pd.to_numeric(_enrich[_col], errors="coerce")

        if not _fix.empty and "team_id" in _enrich.columns:
            _enrich = _enrich.merge(_fix[["team_id", "next_match", "difficulty"]], on="team_id", how="left")
        else:
            _enrich["next_match"] = "–"
            _enrich["difficulty"] = None

        _enrich["ppm"] = (_enrich["total_points"] / _enrich["now_cost"]).round(1)

        _COLS = {
            "name":              "Player",
            "position_name":     "Pos",
            "team":              "Team",
            "next_match":        "Next Match",
            "difficulty":        "Diff",
            "now_cost":          "Cost",
            "projected_points":  "Proj Pts",
            "total_points":      "Season Pts",
            "ppm":               "Pts/£m",
            "points_per_game":   "PPG",
            "form":              "Form",
            "selected_by_percent": "Ownership",
            "gw_1_points":       "GW-1",
            "gw_2_points":       "GW-2",
            "gw_3_points":       "GW-3",
            "in_starting_11":    "XI",
        }
        _avail = [c for c in _COLS if c in _enrich.columns]
        _display = (
            _enrich[_avail]
            .sort_values("projected_points", ascending=False)
            .rename(columns=_COLS)
        )
        st.dataframe(
            _display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Cost":       st.column_config.NumberColumn("Cost (£m)", format="£%.1f"),
                "Proj Pts":   st.column_config.NumberColumn(format="%.1f"),
                "Season Pts": st.column_config.NumberColumn(format="%d"),
                "Pts/£m":     st.column_config.NumberColumn(format="%.1f"),
                "PPG":        st.column_config.NumberColumn(format="%.1f"),
                "Form":       st.column_config.NumberColumn(format="%.1f"),
                "Ownership":  st.column_config.NumberColumn("Ownership (%)", format="%.1f%%"),
                "Diff":       st.column_config.ProgressColumn(
                                  "Difficulty", min_value=0, max_value=5, format="%d /5"
                              ),
                "GW-1":       st.column_config.NumberColumn(format="%d"),
                "GW-2":       st.column_config.NumberColumn(format="%d"),
                "GW-3":       st.column_config.NumberColumn(format="%d"),
                "XI":         st.column_config.CheckboxColumn("XI"),
            },
        )

        # ── Distribution analysis ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## ⚽ Squad Distribution")

        d1, d2, d3 = st.columns(3)

        with d1:
            st.markdown("**📊 By Position**")
            for pos in ["GKP", "DEF", "MID", "FWD"]:
                count = (baseline_df["position_name"] == pos).sum()
                st.markdown(f"""
                    <div class='stat-card' style='margin-bottom:0.8rem;'>
                        <div class='stat-value'>{count}</div>
                        <div class='stat-label'>{pos}</div>
                    </div>
                """, unsafe_allow_html=True)

        with d2:
            st.markdown("**🏟️ By Team**")
            for team, count in baseline_df["team"].value_counts().items():
                st.markdown(f"""
                    <div class='modern-card' style='padding:0.8rem; margin-bottom:0.5rem;'>
                        <strong style='color:#37003c;'>{team}</strong>: {count} player(s)
                    </div>
                """, unsafe_allow_html=True)

        with d3:
            st.markdown("**💰 Cost Breakdown**")
            for pos in ["GKP", "DEF", "MID", "FWD"]:
                pos_cost = baseline_df[baseline_df["position_name"] == pos]["now_cost"].sum()
                st.markdown(f"""
                    <div class='modern-card' style='padding:0.8rem; margin-bottom:0.5rem;'>
                        <strong style='color:#37003c;'>{pos}</strong>: £{pos_cost:.1f}m
                    </div>
                """, unsafe_allow_html=True)
            st.markdown(f"""
                <div class='stat-card' style='margin-top:0.5rem;'>
                    <div class='stat-value'>£{budget_remain:.1f}m</div>
                    <div class='stat-label'>ITB Remaining</div>
                </div>
            """, unsafe_allow_html=True)
 
# --- TAB: ANALYTICS & SQUAD ---
with tab_analytics:
    st.markdown("## 📊 Analytics Hub")
    st.caption("Analyse player performance and manage your squad in one place")

    # ── SESSION STATE ──────────────────────────────────────────────────────────
    if 'my_squad_names' not in st.session_state:
        st.session_state.my_squad_names = []
    if 'comparison_list' not in st.session_state:
        st.session_state.comparison_list = []

    # ── FILTER BAR ─────────────────────────────────────────────────────────────
    st.markdown("#### 🔍 Compare Players")

    fc1, fc2, fc3 = st.columns([1, 1, 3])

    with fc1:
        search_team = st.multiselect(
            "🏟️ Team",
            options=sorted(players_df['team'].unique().tolist()),
            placeholder="All teams",
            key="squad_search_team",
        )

    with fc2:
        search_pos = st.multiselect(
            "📌 Position",
            options=["GKP", "DEF", "MID", "FWD"],
            placeholder="All positions",
            key="squad_search_pos",
        )

    # Apply filters and exclude already-selected players from dropdown
    dropdown_df = players_df.copy()
    if search_team:
        dropdown_df = dropdown_df[dropdown_df['team'].isin(search_team)]
    if search_pos:
        dropdown_df = dropdown_df[dropdown_df['position_name'].isin(search_pos)]

    available_options = sorted([
        n for n in dropdown_df['name'].tolist()
        if n not in st.session_state.my_squad_names
    ])
    result_label = f"{len(available_options)} players shown" if (search_team or search_pos) else f"{len(available_options)} players available"

    with fc3:
        sel_player = st.selectbox(
            "🔎 Add Player",
            options=[""] + available_options,
            index=0,
            key="squad_player_select",
            format_func=lambda x: "Type to search players..." if x == "" else x,
        )
        st.caption(result_label)

    # Auto-add as soon as a name is selected
    if sel_player and sel_player not in st.session_state.my_squad_names:
        if len(st.session_state.my_squad_names) < 15:
            st.session_state.my_squad_names.append(sel_player)
            st.session_state.comparison_list = st.session_state.my_squad_names.copy()
            st.rerun()

    # ── SELECTED PLAYERS TABLE ─────────────────────────────────────────────────
    if st.session_state.my_squad_names:
        st.markdown("#### 👥 Selected Players")

        squad_data = players_df[players_df['name'].isin(st.session_state.my_squad_names)][
            ['name', 'team', 'position_name', 'total_points', 'now_cost']
        ].copy()
        squad_data['now_cost'] = squad_data['now_cost'] / 10
        squad_data['ppm'] = (squad_data['total_points'] / squad_data['now_cost']).round(1)
        # Preserve insertion order
        squad_data['_order'] = squad_data['name'].apply(
            lambda x: st.session_state.my_squad_names.index(x)
        )
        squad_data = squad_data.sort_values('_order').drop(columns='_order').reset_index(drop=True)

        # Table header
        h0, h1, h2, h3, h4, h5, h6 = st.columns([2.5, 1.6, 1.1, 0.8, 1.0, 0.8, 0.5])
        h0.markdown("**Player**")
        h1.markdown("**Team**")
        h2.markdown("**Pos**")
        h3.markdown("**Pts**")
        h4.markdown("**Cost**")
        h5.markdown("**Pts/£m**")
        h6.markdown("")

        remove_player = None
        for _, row in squad_data.iterrows():
            c0, c1, c2, c3, c4, c5, c6 = st.columns([2.5, 1.6, 1.1, 0.8, 1.0, 0.8, 0.5])
            c0.write(row['name'])
            c1.write(row['team'])
            c2.write(row['position_name'])
            c3.write(str(int(row['total_points'])))
            c4.write(f"£{row['now_cost']:.1f}m")
            c5.write(f"{row['ppm']:.1f}")
            with c6:
                if st.button("✕", key=f"remove_{row['name']}", help=f"Remove {row['name']}"):
                    remove_player = row['name']

        if remove_player:
            st.session_state.my_squad_names.remove(remove_player)
            st.session_state.comparison_list = st.session_state.my_squad_names.copy()
            st.rerun()

        # Totals + clear button on same row
        tot_cols = st.columns([2.5, 1.6, 1.1, 0.8, 1.0, 0.8, 2.0])
        tot_cols[0].markdown(f"**{len(squad_data)} player(s)**")
        tot_cols[3].markdown(f"**{int(squad_data['total_points'].sum())}**")
        tot_cols[4].markdown(f"**£{squad_data['now_cost'].sum():.1f}m**")
        tot_cols[5].markdown(f"**{(squad_data['total_points'].sum() / squad_data['now_cost'].sum()):.1f}**")
        with tot_cols[6]:
            if st.button("🗑️ Clear all", key="clear_all_squad_1", type="secondary"):
                st.session_state.my_squad_names = []
                st.session_state.comparison_list = []
                st.rerun()

    st.divider()

    # ── SQUAD METRICS ─────────────────────────────────────────────────────────
    if st.session_state.my_squad_names:
        my_team_stats = players_df[players_df['name'].isin(st.session_state.my_squad_names)]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Squad Size", f"{len(my_team_stats)}/15")
        m2.metric("Total Points", f"{my_team_stats['total_points'].sum()}")
        m3.metric("Avg Points/Player", f"{my_team_stats['total_points'].mean():.1f}")
        m4.metric("Total Cost", f"£{my_team_stats['now_cost'].sum()/10:.1f}m")
        st.divider()

    # ── BUILD PLOT DF ──────────────────────────────────────────────────────────
    plot_df = players_df.copy()
    plot_df['ppm'] = (plot_df['total_points'] / (plot_df['now_cost'] / 10)).round(1)
    plot_df['Status'] = 'League'
    if st.session_state.my_squad_names:
        plot_df.loc[plot_df['name'].isin(st.session_state.my_squad_names), 'Status'] = 'My Squad'

    has_squad = 'My Squad' in plot_df['Status'].values

    # ── 1. VALUE ANALYSIS ─────────────────────────────────────────────────────
    st.markdown("#### 💰 Points vs. Cost")
    st.caption("Your squad highlighted against the rest of the league." if has_squad else "All league players — add your squad above to highlight them.")

    fig_main = px.scatter(
        plot_df, x='now_cost', y='total_points',
        color='Status', hover_name='name',
        hover_data={'ppm': True, 'now_cost': ':.1f', 'total_points': True, 'Status': False},
        color_discrete_map={'My Squad': '#00FF87', 'League': '#37003c'},
        labels={'now_cost': 'Cost (£m)', 'total_points': 'Total Points', 'ppm': 'Pts/£m'},
        template="plotly_white", height=480
    )
    if has_squad:
        fig_main.update_traces(marker=dict(size=14, line=dict(width=2, color='white')), selector=dict(name='My Squad'))
    fig_main.update_traces(marker=dict(size=6, opacity=0.18 if has_squad else 0.5), selector=dict(name='League'))
    fig_main.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_main, use_container_width=True)

    # ── PPM leaderboard ───────────────────────────────────────────────────────
    with st.expander("📊 Best Value Players (Points per £m)", expanded=False):
        ppm_df = (
            plot_df[plot_df['total_points'] > 20]
            .sort_values('ppm', ascending=False)
            .head(20)[['name', 'position_name', 'team', 'total_points', 'now_cost', 'ppm']]
            .copy()
        )
        ppm_df.columns = ['Player', 'Pos', 'Team', 'Total Pts', 'Cost (raw)', 'Pts/£m']
        ppm_df['Cost (£m)'] = (ppm_df['Cost (raw)'] / 10).round(1)
        st.dataframe(
            ppm_df[['Player', 'Pos', 'Team', 'Total Pts', 'Cost (£m)', 'Pts/£m']],
            use_container_width=True, hide_index=True,
            column_config={
                'Total Pts': st.column_config.NumberColumn(format="%d"),
                'Cost (£m)': st.column_config.NumberColumn(format="£%.1f"),
                'Pts/£m':    st.column_config.NumberColumn(format="%.1f"),
            }
        )

    st.divider()

    # ── 2. POSITION BREAKDOWN ─────────────────────────────────────────────────
    st.markdown("#### 🛡️ Position Breakdown")
    st.caption("Points vs. cost by role — spot over- and under-performers in each position.")

    fig_pos = px.scatter(
        plot_df, x='now_cost', y='total_points',
        color='Status', facet_col='position_name',
        facet_col_wrap=2, hover_name='name',
        hover_data={'ppm': True, 'Status': False},
        color_discrete_map={'My Squad': '#00FF87', 'League': '#37003c'},
        labels={'now_cost': 'Cost', 'total_points': 'Points', 'ppm': 'Pts/£m'},
        template="plotly_white", height=620
    )
    if has_squad:
        fig_pos.update_traces(marker=dict(size=13, line=dict(width=2, color='white')), selector=dict(name='My Squad'))
    fig_pos.update_traces(marker=dict(size=5, opacity=0.18 if has_squad else 0.5), selector=dict(name='League'))
    fig_pos.for_each_annotation(lambda a: a.update(text=f"<b>{a.text.split('=')[-1]}</b>"))
    fig_pos.update_layout(showlegend=False)
    st.plotly_chart(fig_pos, use_container_width=True)

    st.divider()

    # ── 3. SEASON TRAJECTORY ──────────────────────────────────────────────────
    st.markdown("#### 📈 Season Trajectory")

    if st.session_state.comparison_list:
        chart_tab1, chart_tab2 = st.tabs(["📊 Weekly Performance", "📈 Cumulative Points"])

        with st.spinner("Loading trajectory data..."):
            fig_cum, fig_weekly = pl.plot_player_performance_timeseries(
                st.session_state.comparison_list,
                players_df
            )

        with chart_tab1:
            st.caption("Points scored each gameweek — includes negative returns from deductions or red cards.")
            if fig_weekly:
                fig_weekly.update_layout(
                    yaxis=dict(
                        zeroline=True,
                        zerolinecolor="#888888",
                        zerolinewidth=1.5,
                        rangemode="tozero" if False else "normal",
                    ),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(t=40, b=40)
                )
                for trace in fig_weekly.data:
                    if hasattr(trace, 'type') and getattr(trace, 'type', '') == 'bar':
                        trace.update(base=None)
                st.plotly_chart(fig_weekly, use_container_width=True)
            else:
                st.info("No weekly data found for the selected players.")

        with chart_tab2:
            st.caption("Total points accumulated over the season.")
            if fig_cum:
                fig_cum.update_layout(
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(t=40, b=40)
                )
                st.plotly_chart(fig_cum, use_container_width=True)
            else:
                st.info("No cumulative data found for the selected players.")
    else:
        st.markdown("""
            <div class='empty-state'>
                <div class='empty-state-icon'>📈</div>
                <div class='empty-state-title'>No players selected</div>
                <div class='empty-state-body'>Add players using the search above to compare their season trajectories</div>
            </div>
        """, unsafe_allow_html=True)

with tab_recruit_perf:
    st.markdown("## 🧠 Recruitment Hub")
    st.caption("Build your optimal squad and analyse team performance in one place")

    recruit_tab, perf_tab = st.tabs(["🎯 AI Scouts", "📈 Recruitment Analysts"])

    # ── SUB-TAB 1: RECRUITMENT ────────────────────────────────────────────────
    with recruit_tab:
        st.markdown("### 🎯 Build Your Optimal Squad")
        st.caption("Set your constraints and let AI find the best team for you")

        if 'players_df' not in locals() and 'players_df' not in globals():
            st.error("Please load the FPL data first.")
            st.stop()

        if 'name' not in players_df.columns:
            players_df['name'] = players_df['first_name'] + ' ' + players_df['second_name']
        if 'position_name' not in players_df.columns:
            pos_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
            players_df['position_name'] = players_df['element_type'].map(pos_map)

        if 'players_to_keep' not in st.session_state:
            st.session_state.players_to_keep = []
        if 'players_to_exclude' not in st.session_state:
            st.session_state.players_to_exclude = []

        # ── Constraint counters ────────────────────────────────────────────
        m1, m2, m3 = st.columns([2, 2, 1])
        m1.metric("🔒 Locked In", len(st.session_state.players_to_keep))
        m2.metric("🚫 Excluded",  len(st.session_state.players_to_exclude))
        with m3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️ Reset All", use_container_width=True, key="recruit_reset_1"):
                st.session_state.players_to_keep = []
                st.session_state.players_to_exclude = []
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        teams_list = sorted(players_df['team'].unique().tolist())
        col1, col2 = st.columns(2)

        # ── Force Include ──────────────────────────────────────────────────
        with col1:
            st.markdown("""
                <div class='modern-card'>
                    <h4 style='margin-top: 0; color: #28a745;'>🔒 Force Include</h4>
                    <p style='color: #6c757d; font-size: 0.9rem;'>Players that must be in your squad</p>
                </div>
            """, unsafe_allow_html=True)

            sel_team_inc = st.selectbox("Select Team", [""] + teams_list, key="inc_team_1")
            if sel_team_inc:
                team_p = players_df[players_df['team'] == sel_team_inc].sort_values('web_name')
                _inj_col = 'web_name' if 'web_name' in injuries_df.columns else ('name' if 'name' in injuries_df.columns else None)
                _injured = set(injuries_df[_inj_col].tolist()) if _inj_col else set()
                p_options = [
                    f"{'🚑 ' if row['web_name'] in _injured else ''}{row['web_name']} ({row['position_name']} - £{row['now_cost']/10:.1f}m - {int(row['total_points'])} pts)"
                    for _, row in team_p.iterrows()
                ]
                if _injured & set(team_p['web_name'].tolist()):
                    st.caption("🚑 = injury / doubt — check news before locking in")
                # Clear multiselect before instantiation if flagged by previous run
                if st.session_state.pop("_reset_inc_multi", False):
                    st.session_state.pop("inc_p_multi_1", None)
                selected = st.multiselect("Choose Players", options=p_options, key="inc_p_multi_1")
                if st.button("➕ Add to Squad", key="btn_add_inc_1", use_container_width=True):
                    for item in selected:
                        p_web_name = item.lstrip("🚑 ").split(" (")[0]
                        full_name = team_p[team_p['web_name'] == p_web_name]['name'].values[0]
                        if full_name not in st.session_state.players_to_keep:
                            st.session_state.players_to_keep.append(full_name)
                    st.session_state["_reset_inc_multi"] = True
                    st.rerun()

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
                        if st.button("✕", key=f"del_inc_{p_name}_1"):
                            st.session_state.players_to_keep.remove(p_name)
                            st.rerun()

        # ── Force Exclude ──────────────────────────────────────────────────
        with col2:
            st.markdown("""
                <div class='modern-card'>
                    <h4 style='margin-top: 0; color: #dc3545;'>🚫 Force Exclude</h4>
                    <p style='color: #6c757d; font-size: 0.9rem;'>Players to avoid in optimization</p>
                </div>
            """, unsafe_allow_html=True)

            sel_team_exc = st.selectbox("Select Team", [""] + teams_list, key="exc_team_1")
            if sel_team_exc:
                team_p = players_df[players_df['team'] == sel_team_exc].sort_values('web_name')
                _inj_col = 'web_name' if 'web_name' in injuries_df.columns else ('name' if 'name' in injuries_df.columns else None)
                _injured = set(injuries_df[_inj_col].tolist()) if _inj_col else set()
                p_options = [
                    f"{'🚑 ' if row['web_name'] in _injured else ''}{row['web_name']} ({row['position_name']} - £{row['now_cost']/10:.1f}m - {int(row['total_points'])} pts)"
                    for _, row in team_p.iterrows()
                ]
                # Clear multiselect before instantiation if flagged by previous run
                if st.session_state.pop("_reset_exc_multi", False):
                    st.session_state.pop("exc_p_multi_1", None)
                selected = st.multiselect("Choose Players", options=p_options, key="exc_p_multi_1")
                if st.button("➕ Add to Exclusions", key="btn_add_exc_1", use_container_width=True):
                    for item in selected:
                        p_web_name = item.lstrip("🚑 ").split(" (")[0]
                        full_name = team_p[team_p['web_name'] == p_web_name]['name'].values[0]
                        if full_name not in st.session_state.players_to_exclude:
                            st.session_state.players_to_exclude.append(full_name)
                    st.session_state["_reset_exc_multi"] = True
                    st.rerun()

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
                        if st.button("✕", key=f"del_exc_{p_name}_1"):
                            st.session_state.players_to_exclude.remove(p_name)
                            st.rerun()

        players_to_keep = st.session_state.players_to_keep
        players_to_exclude = st.session_state.players_to_exclude

        # ── Optimize CTA ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🚀 Generate Optimal Squad")
        run_opt = st.button("⚡ Optimize My Team", type="primary", use_container_width=True)

        if run_opt:
            # Step 1 occupies 0–70% of the bar (slowest: fetches 300+ player histories)
            # Step 2 occupies 70–90% (LP solver)
            # Step 3 occupies 90–100% (XI selection)
            bar = st.progress(0, text="🔍 Step 1/3 — Fetching fixture difficulty and recent form...")

            def on_player_fetched(pct, text):
                bar.progress(int(pct * 0.70), text=text)

            forecast_df = pl.fetch_and_forecast_players(
                progress_callback=on_player_fetched,
                xg_blend=xg_blend,
                baseline_blend=baseline_blend,
            )

            bar.progress(70, text="🧠 Step 2/3 — Running LP optimization algorithm...")
            selected_names, model, fig_optimization = pl.optimize_fpl_team(
                forecast_df,
                st.session_state.players_to_keep,
                st.session_state.players_to_exclude,
                budget=budget,
            )

            bar.progress(90, text="📋 Step 3/3 — Selecting optimal Starting XI...")
            selected_df = forecast_df[forecast_df["name"].isin(selected_names)].copy()
            POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
            selected_df["position_name"] = selected_df["element_type"].map(POSITION_MAP)
            starting_11 = pl.optimize_starting_11(selected_df)

            st.session_state["selected_squad_df"] = selected_df
            st.session_state["starting_11_df"] = starting_11
            st.session_state["fig_optimization"] = fig_optimization
            st.session_state["forecast_df"] = forecast_df

            bar.progress(100, text=f"✅ Optimization complete — {LpStatus[model.status]}")

        # ── Results (shown below CTA once optimized) ───────────────────────
        if "selected_squad_df" in st.session_state:
            selected_df = st.session_state["selected_squad_df"]
            starting_11 = st.session_state["starting_11_df"]

            counts = starting_11['position_name'].value_counts()
            formation_str = f"{counts.get('DEF',0)}-{counts.get('MID',0)}-{counts.get('FWD',0)}"

            st.markdown("---")
            st.markdown("## 📊 Squad Summary")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Squad Cost",   f"£{selected_df['now_cost'].sum():.1f}m")
            col2.metric("Total Points", f"{selected_df['projected_points'].sum():.0f}")
            col3.metric("XI Points",    f"{starting_11['projected_points'].sum():.0f}")
            col4.metric("Formation",    formation_str)
            col5.metric("Squad Size",   f"{len(selected_df)}/15")

            st.markdown("---")
            st.markdown("## ⚽ Starting XI Formation")
            fig_pitch = create_pitch_visualization(selected_df, starting_11)
            st.plotly_chart(fig_pitch, use_container_width=True)

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
                    st.markdown(_bench_card_html(p, rank_label), unsafe_allow_html=True)

            # ── BGW warning ────────────────────────────────────────────────
            bgw_players = selected_df[
                (selected_df["name"].isin(st.session_state.players_to_keep)) &
                (selected_df.get("fixture_multiplier", pd.Series(1, index=selected_df.index)) == 0)
            ]
            if not bgw_players.empty:
                names_str = ", ".join(bgw_players["name"].tolist())
                st.markdown(f"""
                    <div class='warning-banner'>
                        <strong>⚠️ Blank Gameweek Alert</strong><br>
                        The following force-included player(s) have no fixture in GW{next_gw}:
                        <strong>{names_str}</strong>. Consider removing them or using a wildcard.
                    </div>
                """, unsafe_allow_html=True)

            # ── CSV export ─────────────────────────────────────────────────
            st.markdown("---")
            _export = selected_df[['name', 'position_name', 'team', 'now_cost', 'projected_points']].copy()
            _export['in_starting_11'] = _export['name'].isin(starting_11['name'])
            st.download_button(
                "📥 Export Squad to CSV",
                data=_export.to_csv(index=False),
                file_name=f"fpl_squad_gw{next_gw}.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # ── Transfer Planner ───────────────────────────────────────────
            st.markdown("---")
            st.markdown("### 🔄 Transfer Planner")
            st.caption("Find the best transfer(s) to improve your current squad next gameweek")

            tp_col1, tp_col2 = st.columns([3, 1])
            with tp_col1:
                max_transfers = st.slider(
                    "Maximum free transfers", 1, 3, 1, key="tp_max_transfers",
                    help="How many transfers to allow (each beyond your free hit costs 4 pts)"
                )
            with tp_col2:
                st.markdown("<br>", unsafe_allow_html=True)
                run_transfers = st.button(
                    "🔍 Find Best Transfers", type="secondary",
                    use_container_width=True, key="tp_run_btn"
                )

            if run_transfers:
                if "forecast_df" not in st.session_state:
                    st.warning("Run the optimizer first so forecast data is available.")
                else:
                    with st.spinner("🔄 Calculating optimal transfers..."):
                        _, _t_in, _t_out = pl.optimize_transfers(
                            st.session_state["forecast_df"],
                            selected_df["name"].tolist(),
                            max_transfers=max_transfers,
                            budget=budget,
                        )
                    st.session_state["transfer_result"] = (_t_in, _t_out)

            if "transfer_result" in st.session_state:
                _t_in, _t_out = st.session_state["transfer_result"]
                if _t_in:
                    tc_out, tc_arrow, tc_in = st.columns([5, 1, 5])

                    with tc_out:
                        st.markdown(
                            "<p style='font-weight:700; color:#fd7e14; margin-bottom:0.6rem;'>"
                            "⬆️ Transfer OUT</p>",
                            unsafe_allow_html=True,
                        )
                        for name in _t_out:
                            row = selected_df[selected_df["name"] == name]
                            if not row.empty:
                                r = row.iloc[0]
                                pos = r.get("position_name", "")
                                badge = f"<span class='pos-badge {_POS_BADGE_CLS.get(pos, '')}' style='margin-right:0.4rem;'>{pos}</span>"
                                st.markdown(f"""
                                    <div class='transfer-out'>
                                        <div class='transfer-player'>{badge}{r['name']}</div>
                                        <div class='transfer-meta'>
                                            {r.get('team', '')} &nbsp;·&nbsp;
                                            £{r['now_cost']:.1f}m &nbsp;·&nbsp;
                                            {r['projected_points']:.1f} proj pts
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)

                    with tc_arrow:
                        for _ in _t_out:
                            st.markdown(
                                "<div class='transfer-arrow-col'>"
                                "<span class='transfer-arrow-icon'>→</span>"
                                "</div>",
                                unsafe_allow_html=True,
                            )

                    with tc_in:
                        st.markdown(
                            "<p style='font-weight:700; color:#28a745; margin-bottom:0.6rem;'>"
                            "⬇️ Transfer IN</p>",
                            unsafe_allow_html=True,
                        )
                        for name in _t_in:
                            row = st.session_state["forecast_df"][
                                st.session_state["forecast_df"]["name"] == name
                            ]
                            if not row.empty:
                                r = row.iloc[0]
                                pos = r.get("position_name", "")
                                badge = f"<span class='pos-badge {_POS_BADGE_CLS.get(pos, '')}' style='margin-right:0.4rem;'>{pos}</span>"
                                st.markdown(f"""
                                    <div class='transfer-in'>
                                        <div class='transfer-player'>{badge}{r['name']}</div>
                                        <div class='transfer-meta'>
                                            {r.get('team', '')} &nbsp;·&nbsp;
                                            £{r['now_cost']:.1f}m &nbsp;·&nbsp;
                                            {r['projected_points']:.1f} proj pts
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)
                else:
                    st.success("✅ Your current squad is already optimal — no transfers needed!")

    # ── SUB-TAB 2: RECRUITMENT ANALYSTS ───────────────────────────────────────
    with perf_tab:
        st.markdown("### 📈 Advanced Analytics")
        st.caption("Deep dive into squad performance and optimization details")

        if "selected_squad_df" not in st.session_state:
            st.markdown("""
                <div class='empty-state'>
                    <div class='empty-state-icon'>🔍</div>
                    <div class='empty-state-title'>No squad optimized yet</div>
                    <div class='empty-state-body'>
                        Head to the <strong>🎯 AI Scouts</strong> tab, run the Squad Optimizer,
                        then come back here for deep performance analytics.
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            selected_df = st.session_state['selected_squad_df']
            starting_11 = st.session_state.get('starting_11_df', pd.DataFrame())

            # ── 1. Captain picks (most actionable) ────────────────────────
            st.markdown("#### 🔴 Captain Recommendations")
            st.caption("Your captain scores double points - choose wisely!")

            if not starting_11.empty:
                captain_options = starting_11.sort_values('projected_points', ascending=False)
                POSITION_MAP = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
                if 'element_type' not in captain_options.columns:
                    captain_options = captain_options.merge(selected_df[['name', 'element_type']], on='name', how='left')
                if 'position_name' not in captain_options.columns:
                    captain_options['position_name'] = captain_options['element_type'].map(POSITION_MAP)

                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown("##### 🏆 Top 5 Captain Picks")
                    top_5 = captain_options.head(5)[['name', 'position_name', 'team', 'projected_points', 'form']].copy()
                    _CAP_BADGES = [
                        ("🥇", "linear-gradient(135deg, #ffd700 0%, #ffed4e 100%)"),
                        ("🥈", "linear-gradient(135deg, #e8e8e8 0%, #f5f5f5 100%)"),
                        ("🥉", "linear-gradient(135deg, #cd7f32 0%, #e8a87c 100%)"),
                        ("⭐", "white"), ("⭐", "white"),
                    ]
                    for idx, (_, row) in enumerate(top_5.iterrows()):
                        badge, bg_color = _CAP_BADGES[idx]
                        base_pts   = float(row['projected_points'])
                        double_pts = base_pts * 2
                        pos        = str(row.get('position_name', ''))
                        pos_badge  = f"<span class='pos-badge {_POS_BADGE_CLS.get(pos, '')}' style='vertical-align:middle;'>{pos}</span>"
                        st.markdown(f"""
                            <div class='modern-card' style='background:{bg_color}; margin-bottom:0.75rem;'>
                                <div style='display:flex; justify-content:space-between; align-items:center; gap:0.5rem;'>
                                    <div style='display:flex; align-items:center; gap:0.6rem; flex:1; min-width:0;'>
                                        <span style='font-size:1.4rem; flex-shrink:0;'>{badge}</span>
                                        <div style='min-width:0;'>
                                            <div style='font-weight:700; font-size:1rem; color:#37003c;
                                                        white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>
                                                {row['name']}
                                            </div>
                                            <div style='margin-top:0.2rem;'>
                                                {pos_badge}
                                                <span style='color:#6c757d; font-size:0.82rem; margin-left:0.35rem;'>
                                                    {row['team']}
                                                </span>
                                            </div>
                                        </div>
                                    </div>
                                    <div style='text-align:right; flex-shrink:0;'>
                                        <div style='font-size:1.2rem; font-weight:800; color:#1e8a4b;'>{double_pts:.1f}</div>
                                        <div style='font-size:0.75rem; color:#6c757d; white-space:nowrap;'>
                                            {base_pts:.1f} × 2 (C)
                                        </div>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                with col2:
                    st.markdown("##### 💡 Captain Tips")
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

            # ── 2. Full squad table ────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 📋 Complete Squad Breakdown")
            st.caption("Difficulty: 1 = easiest · 5 = hardest")

            _fix2 = get_fixture_info()
            _sq = selected_df.copy()
            for _col in ["form", "points_per_game"]:
                if _col in _sq.columns:
                    _sq[_col] = pd.to_numeric(_sq[_col], errors="coerce")
            if not _fix2.empty and "team_id" in _sq.columns:
                _sq = _sq.merge(_fix2[["team_id", "next_match", "difficulty"]], on="team_id", how="left")
            else:
                _sq["next_match"] = "–"
                _sq["difficulty"] = None
            _sq["ppm"] = (_sq["total_points"] / _sq["now_cost"]).round(1)

            _SQ_COLS = {
                "name":              "Player",
                "position_name":     "Pos",
                "team":              "Team",
                "next_match":        "Next Match",
                "difficulty":        "Diff",
                "now_cost":          "Cost",
                "projected_points":  "Proj Pts",
                "total_points":      "Season Pts",
                "ppm":               "Pts/£m",
                "points_per_game":   "PPG",
                "form":              "Form",
                "selected_by_percent": "Ownership",
                "gw_1_points":       "GW-1",
                "gw_2_points":       "GW-2",
                "gw_3_points":       "GW-3",
            }
            _sq_avail = [c for c in _SQ_COLS if c in _sq.columns]
            _sq_display = (
                _sq[_sq_avail]
                .sort_values(["element_type", "projected_points"], ascending=[True, False])
                .rename(columns=_SQ_COLS)
            )
            st.dataframe(
                _sq_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Cost":       st.column_config.NumberColumn("Cost (£m)", format="£%.1f"),
                    "Proj Pts":   st.column_config.NumberColumn(format="%.1f"),
                    "Season Pts": st.column_config.NumberColumn(format="%d"),
                    "Pts/£m":     st.column_config.NumberColumn(format="%.1f"),
                    "PPG":        st.column_config.NumberColumn(format="%.1f"),
                    "Form":       st.column_config.NumberColumn(format="%.1f"),
                    "Ownership":  st.column_config.NumberColumn("Ownership (%)", format="%.1f%%"),
                    "Diff":       st.column_config.ProgressColumn(
                                      "Difficulty", min_value=0, max_value=5, format="%d /5"
                                  ),
                    "GW-1":       st.column_config.NumberColumn(format="%d"),
                    "GW-2":       st.column_config.NumberColumn(format="%d"),
                    "GW-3":       st.column_config.NumberColumn(format="%d"),
                },
            )

            # ── 3. Distribution analysis ───────────────────────────────────
            st.markdown("---")
            st.markdown("#### ⚽ Squad Distribution Analysis")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**📊 By Position**")
                pos_counts = selected_df['position_name'].value_counts()
                for pos, count in pos_counts.items():
                    st.markdown(f"<div class='stat-card' style='margin-bottom: 0.8rem;'><div class='stat-value'>{count}</div><div class='stat-label'>{pos}</div></div>", unsafe_allow_html=True)

            with col2:
                st.markdown("**🏟️ By Team**")
                team_counts = selected_df['team'].value_counts().head(5)
                for team, count in team_counts.items():
                    st.markdown(f"<div class='modern-card' style='padding: 0.8rem; margin-bottom: 0.5rem;'><strong style='color: #37003c;'>{team}</strong>: {count} player(s)</div>", unsafe_allow_html=True)
                if len(selected_df['team'].unique()) > 5:
                    st.caption(f"+ {len(selected_df['team'].unique()) - 5} other team(s)")

            with col3:
                st.markdown("**💰 Cost Analysis**")
                avg_cost = selected_df['now_cost'].mean()
                max_cost = selected_df['now_cost'].max()
                min_cost = selected_df['now_cost'].min()
                st.markdown(f"<div class='stat-card' style='margin-bottom: 0.8rem;'><div class='stat-value'>£{avg_cost:.1f}m</div><div class='stat-label'>Average</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='modern-card' style='padding: 0.6rem; margin-bottom: 0.5rem;'><strong>Most Expensive:</strong> £{max_cost:.1f}m</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='modern-card' style='padding: 0.6rem;'><strong>Cheapest:</strong> £{min_cost:.1f}m</div>", unsafe_allow_html=True)

            # ── 4. Optimization chart (least actionable, last) ─────────────
            st.markdown("---")
            st.markdown("#### 🎯 Optimization Visualization")
            st.caption("Model output showing how players were scored and selected")
            if 'fig_optimization' in st.session_state:
                st.plotly_chart(st.session_state['fig_optimization'], use_container_width=True)
            else:
                st.info("📊 Optimization chart will appear after running the optimizer")
                
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

# TAB 5: Injury News
with tab5:
    st.markdown("## 🚑 Injury & team news")
    
    TEAM_COLORS = {
        # Existing teams
        "Arsenal": "#EF0107",        # Red
        "Aston Villa": "#670E36",    # Claret
        "Bournemouth": "#DA291C",    # Red
        "Brentford": "#E30613",      # Red & white stripes
        "Brighton": "#0057B8",       # Blue & white stripes
        "Chelsea": "#034694",        # Blue
        "Crystal Palace": "#1B458F", # Red & blue
        "Everton": "#003399",        # Blue
        "Fulham": "#FFFFFF",         # White
        "Leeds": "#FFFFFF",          # White
        "Liverpool": "#C8102E",      # Red
        "Man City": "#6CABDD",       # Sky blue
        "Man Utd": "#DA291C",        # Red
        "Newcastle": "#241F20",      # Black
        "Nott'm Forest": "#EF3340",     # Red
        "Sheffield Utd": "#EE2737",  # Red
        "Spurs": "#FFFFFF",      # White
        "West Ham": "#7A263A",       # Claret
        "Wolves": "#FDB913",         # Gold

        # 2025-26 promoted teams (replacing Leicester, Ipswich, Southampton)
        "Burnley": "#6C1D45",        # Claret
        "Sunderland": "#EB172B",     # Red
    }

    # ── Team summary ───────────────────────────────────────────────────────
    team_summary = (
        injuries_df
        .groupby("team").size()
        .reset_index(name="injured_players")
        .sort_values("team")
        .reset_index(drop=True)
    )

    # ── Bar chart via st.bar_chart alternative: Plotly with selectbox only ─
    bar_colors = [TEAM_COLORS.get(t, "#888888") for t in team_summary["team"]]
    border_colors = [
        "black" if TEAM_COLORS.get(t, "").upper() in ("#FFFFFF", "#CCCCCC", "WHITE")
        else "rgba(0,0,0,0)"
        for t in team_summary["team"]
    ]

    fig = go.Figure(go.Bar(
        x=team_summary["team"],
        y=team_summary["injured_players"],
        marker_color=bar_colors,
        marker_line_color=border_colors,
        marker_line_width=1.5,
        hovertemplate="<b>%{x}</b><br>%{y} injured<extra></extra>",
    ))

    fig.update_layout(
        title={
        'text': "🚑 Injuries per Team",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
        xaxis_title=None,
        yaxis_title="Injured players",
        showlegend=False,
        margin=dict(t=8, b=8, l=0, r=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickfont=dict(size=11)),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Top bar: dropdown + clear ──────────────────────────────────────────
    st.caption("Use the dropdown below to filter by team.")

    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        teams = ["All teams"] + sorted(injuries_df["team"].unique())
        selected_team = st.selectbox("Filter by team", teams, label_visibility="collapsed")
    with col_btn:
        if st.button("Clear", use_container_width=True):
            st.session_state["selected_team_chart"] = "All teams"
            st.rerun()

    # ── Filter logic ───────────────────────────────────────────────────────
    injuries_filtered = (
        injuries_df[injuries_df["team"] == selected_team]
        if selected_team != "All teams"
        else injuries_df.copy()
    )

    label = selected_team if selected_team != "All teams" else "All teams"
    st.markdown(f"**Showing:** {label} · {len(injuries_filtered)} player(s)")

    # ── Injury table ───────────────────────────────────────────────────────
    _STATUS_PILL = {
        "Injured":   ("background:#fde8ea; color:#c0392b; border:1px solid #f5c6cb;", "🔴"),
        "Doubtful":  ("background:#fff3cd; color:#856404; border:1px solid #ffeeba;", "🟡"),
        "Suspended": ("background:#ede7f6; color:#5e35b1; border:1px solid #d1c4e9;", "🟣"),
    }
    _POS_PILL_COLOR = {"GKP": "#FFD700", "DEF": "#00B4D8", "MID": "#7B2FBE", "FWD": "#E63946"}

    rows_html = ""
    for _, r in injuries_filtered.sort_values("team").iterrows():
        status     = str(r.get("status_readable", ""))
        pill_style, pill_icon = _STATUS_PILL.get(status, ("background:#eee; color:#333;", "⚪"))
        pos        = str(r.get("position_name", ""))
        pos_color  = _POS_PILL_COLOR.get(pos, "#888")
        cop        = r.get("chance_of_playing_this_round", None)
        cop_str    = f"{int(cop)}%" if cop is not None and not pd.isna(cop) else "—"
        news_txt   = str(r.get("news", "")) or "—"
        rows_html += f"""
        <tr>
            <td style='padding:10px 12px; font-weight:600;'>{r.get('name','')}</td>
            <td style='padding:10px 12px; color:#555;'>{r.get('team','')}</td>
            <td style='padding:10px 12px;'>
                <span style='display:inline-block; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:700;
                    background:{pos_color}22; color:{pos_color}; border:1px solid {pos_color}55;'>{pos}</span>
            </td>
            <td style='padding:10px 12px;'>
                <span style='display:inline-block; padding:3px 10px; border-radius:12px; font-size:0.78rem; font-weight:700; {pill_style}'>
                    {pill_icon} {status}
                </span>
            </td>
            <td style='padding:10px 12px; color:#666; font-size:0.85rem; font-weight:700;'>{cop_str}</td>
            <td style='padding:10px 12px; color:#555; font-size:0.83rem; max-width:320px;'>{news_txt}</td>
        </tr>"""

    st.markdown(f"""
        <table style='width:100%; border-collapse:collapse; font-size:0.9rem;'>
            <thead>
                <tr style='border-bottom:2px solid #e0e0e0; background:#f8f9fa;'>
                    <th style='padding:10px 12px; text-align:left; color:#333; font-weight:700;'>Player</th>
                    <th style='padding:10px 12px; text-align:left; color:#333; font-weight:700;'>Team</th>
                    <th style='padding:10px 12px; text-align:left; color:#333; font-weight:700;'>Pos</th>
                    <th style='padding:10px 12px; text-align:left; color:#333; font-weight:700;'>Status</th>
                    <th style='padding:10px 12px; text-align:left; color:#333; font-weight:700;'>Chance</th>
                    <th style='padding:10px 12px; text-align:left; color:#333; font-weight:700;'>News</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
    """, unsafe_allow_html=True)