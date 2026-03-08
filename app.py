"""
╔══════════════════════════════════════════════════════════════════════════╗
║  NexAiCare - AI-Powered Healthcare Platform                              ║
║  Developed by Jade Global                                                ║
║  Six Integrated AI Modules in One Unified Application                    ║
║  Version: 1.0.0                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════
import streamlit as st
import os
import sqlite3
import pandas as pd
import plotly.express as px
from pathlib import Path
from datetime import datetime
import warnings
from PIL import Image

from streamlit_extras.stylable_container import stylable_container
from streamlit_option_menu import option_menu

from openai import OpenAI

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
jadeimage = Image.open("logo/jadeglobalsmall.png")
st.set_page_config(
    page_title="NexAiCare - Healthcare AI Platform",
    page_icon=jadeimage,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# PATHS & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).parent
DB_PATH = str(BASE_DIR / "hc_data.db")
PDF_PATH = str(BASE_DIR / "medical_diagnosis_manual.pdf")
CSV_PATH = str(BASE_DIR / "finetuning_medical_testing.csv")
VECTOR_DB_DIR = str(BASE_DIR / "Healthcare_db")

MENU_ITEMS = {
    "Dashboard": "dashboard",
    "Medical Assistant AI": "module_1",
    "Claim Audit AI": "module_2",
    "Consultation Notes AI": "module_3",
    "PII/PHI Monitor": "module_4",
    "Medical Coding AI": "module_5",
    "Clinical Diagnostic AI": "module_6",
    "Settings": "settings",
}

STATUS_COLORS = {"Paid": "#175388", "Denied": "#ecb713", "Pending": "#2A7B9B"}

# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════
def inject_css():
    CUSTOM_CSS = """
    <style>
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 2rem !important;
        }
        .stAppHeader { visibility: hidden; }
        footer { visibility: hidden; }

        /* Professional Sidebar Logo Sizing */
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 55% !important; /* Reduced for professional look */
            margin-bottom: 5px;
        }

        .stButton > button {
            border-radius: 8px !important;
            border: 2px solid #144774 !important;
            background-color: #175388 !important;
            color: white !important;
            transition: all 0.2s ease-in-out !important;
            padding: 0.4rem 1rem !important;
        }
        .stButton > button:hover {
            background-color: #ecb713 !important;
            border-color: #c49e10 !important;
            color: white !important;
        }

        .insight-card {
            background: #fff; border-radius: 10px; padding: 18px 22px;
            margin: 8px 0; box-shadow: 0 1px 8px rgba(0,0,0,0.05);
            border-left: 4px solid #175388;
        }
        .coming-soon {
            text-align: center; padding: 60px 40px;
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            border-radius: 16px; margin: 30px 0;
        }
    </style>
    """
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# UTILITY HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def api_key_configured():
    if st.session_state.get("openai_api_key"):
        return True
    return False

def render_page_header(icon, title, description):
    full_title = f"{icon} {title}" if icon else title
    banner_html = f"""
    <div style="padding: 12px 25px; border-radius: 12px; width: 100%; 
                background: linear-gradient(135deg, #175388 0%, #2A7B9B 100%);
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15); margin-bottom: 1.2rem; text-align: center;">
        <div style="font-size: 26px; font-weight: bold; color: #FFFFFF; margin: 0;">{full_title}</div>
        <div style="font-size: 14px; color: #E0F7FA; margin-top: 4px;">{description}</div>
    </div>
    """
    st.markdown(banner_html, unsafe_allow_html=True)

def render_kpi_card(label, value):
    box_css = "{ background-color: #ffffff; padding: 15px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); border: 1px solid #e6e6e6; text-align: center; }"
    with stylable_container(key=label, css_styles=box_css):
        st.markdown(f"<p style='color: #175388; font-weight: bold; font-size: 13px; margin-bottom: 5px;'>{label}</p>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='color: #ecb713; font-size: 24px; margin-top: 0;'>{value}</h2>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        # Jade Logo (Width controlled via CSS at 55%)
        st.image('logo/jadeglobal.png')
        
        st.markdown("""
        <div style='text-align: center; color: #175388; margin-bottom: 15px;'>
            <h2 style='margin-bottom: 0; padding-bottom: 0; font-size: 18px;'>NexAiCare</h2>
            <p style='margin-top: 2px; padding-top: 0; font-size: 12px; opacity: 0.8;'>Healthcare AI Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        menu_options = list(MENU_ITEMS.keys())
        current_index = menu_options.index(st.session_state.selected_page)
        
        selected = option_menu(
            menu_title=None,
            options=menu_options,
            icons=['speedometer2', 'heart-pulse', 'search', 'journal-text', 'shield-lock', 'code-square', 'clipboard2-pulse', 'gear'],
            default_index=current_index,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "white", "font-size": "14px"},
                "nav-link": {
                    "font-size": "13px",
                    "text-align": "left",
                    "margin": "2px 0",
                    "padding": "8px 12px",
                    "color": "white",
                    "border-radius": "8px",
                    "background-color": "#175388",
                },
                "nav-link-selected": {"background-color": "#ecb713"},
            }
        )
        st.session_state.selected_page = selected
        
        st.divider()
        
        # Enhanced Status Visibility
        if api_key_configured():
            st.markdown("<p style='font-size: 14px; text-align: center; font-weight: 500;'>Status: <span style='color: #28a745;'>Connected 🟢</span></p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='font-size: 14px; text-align: center; font-weight: 500;'>Status: <span style='color: #dc3545;'>Setup Required 🔴</span></p>", unsafe_allow_html=True)

        # Copyright Section at the Bottom
        st.markdown(
            """<div style='font-size: 11px; color: rgba(255,255,255,0.6); text-align: center; margin-top: 30px; line-height: 1.5;'>
                © 2026 Jade Global. All rights reserved.<br>
                NexAiCare v1.0.0
            </div>""",
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════════════════════
#  DASHBOARD & MODULE RENDERING (Placeholders for Logic)
# ═══════════════════════════════════════════════════════════════════════════
def render_dashboard():
    render_page_header("📊", "Healthcare Operations Dashboard", "Real-time analytics across claims and clinical operations.")
    c1, c2, c3, c4 = st.columns(4)
    render_kpi_card("Total Claims", "100")
    render_kpi_card("Total Billed", "$409,010.39")
    render_kpi_card("Denial Rate", "24.0%")
    render_kpi_card("Avg Length of Stay", "4.0 days")

def render_settings():
    render_page_header("⚙️", "Settings", "Configure your AI credentials.")
    st.text_input("OpenAI API Key", type="password", value=st.session_state.get("openai_api_key", ""), placeholder="sk-...")
    if st.button("Save Configuration"):
        st.success("Configuration saved!")

def main():
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = "Dashboard"
    
    inject_css()
    render_sidebar()

    page = MENU_ITEMS[st.session_state.selected_page]
    if page == "dashboard": render_dashboard()
    elif page == "settings": render_settings()
    else: st.info(f"Module {page} is active. Content goes here.")

if __name__ == "__main__":
    main()
