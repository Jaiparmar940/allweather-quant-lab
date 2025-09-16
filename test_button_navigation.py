#!/usr/bin/env python3
"""
Simple test to verify button navigation works
"""

import streamlit as st
import time

def test_navigation():
    """Test navigation functionality."""
    st.title("Navigation Test")
    
    # Initialize page in session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Page 1"
    
    # Display current page
    st.write(f"Current page: {st.session_state.current_page}")
    
    # Navigation buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Go to Page 1"):
            st.session_state.current_page = "Page 1"
            st.rerun()
    
    with col2:
        if st.button("Go to Page 2"):
            st.session_state.current_page = "Page 2"
            st.rerun()
    
    with col3:
        if st.button("Go to Page 3"):
            st.session_state.current_page = "Page 3"
            st.rerun()
    
    # Display content based on current page
    if st.session_state.current_page == "Page 1":
        st.write("This is Page 1 content")
    elif st.session_state.current_page == "Page 2":
        st.write("This is Page 2 content")
    elif st.session_state.current_page == "Page 3":
        st.write("This is Page 3 content")

if __name__ == "__main__":
    test_navigation()
