import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import login

def check_auth():
    """
    Check if user is authenticated, redirect to login page if not.
    Use this at the beginning of each page to protect it.
    
    Returns:
        bool: True if user is authenticated, redirects to login if not
    """
    # Check if logged in
    if not login.check_login():
        st.warning("Please log in to access this page")
        login.show_login_page()
        st.stop()  # Stop execution of the current page
    
    return True

def init_sidebar():
    """
    Initialize the sidebar with logout button if user is logged in.
    """
    if login.check_login():
        with st.sidebar:
            login.show_logout_button()