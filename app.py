import streamlit as st
import hw1
import hw2


# Title for the main page
st.title('HOMEWORK MANAGER')

# Sidebar selection
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Home", "Homework 1", "Homework 2"])

# Home Page
if selection == "Home":
    st.write("""
    ## Welcome to the Dhruv's Streamlit Homework App
    Use the sidebar to navigate to different Homework.
    """)

# HW 1 Page
elif selection == "Homework 1":
    hw1.run()

# HW 2 Page
elif selection == "Homework 2":
    hw2.run()
