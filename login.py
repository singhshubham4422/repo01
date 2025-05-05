import streamlit as st
import os
import sys
from typing import Dict, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import auth

def show_login_page():
    """Show the login page."""
    st.title("Login")
    
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        
    if "username" not in st.session_state:
        st.session_state.username = ""
    
    # Create tabs for login and registration
    login_tab, register_tab = st.tabs(["Login", "Register"])
    
    with login_tab:
        # Login form
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                success, message = auth.authenticate(username, password)
                
                if success:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
    
    with register_tab:
        # Registration form
        with st.form("register_form"):
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            register_button = st.form_submit_button("Register")
            
            if register_button:
                if not new_username or not new_password:
                    st.error("Username and password are required")
                elif new_password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    success, message = auth.register_user(new_username, new_password)
                    
                    if success:
                        st.success(message)
                        st.info("Please go to the Login tab to login")
                    else:
                        st.error(message)

def show_logout_button():
    """Show the logout button."""
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

def check_login() -> bool:
    """
    Check if the user is logged in.
    
    Returns:
        True if the user is logged in, False otherwise
    """
    return st.session_state.get("logged_in", False)