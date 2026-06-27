import sys
import traceback
import streamlit as st

# Set page config at the very top level before any other streamlit rendering commands
st.set_page_config(
    page_title="Urban Heat Mitigation AI",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    import real_app
    if __name__ == "__main__":
        real_app.main()
except Exception as e:
    st.error("🚨 Critical Startup Error:")
    st.markdown("An unhandled exception occurred during app initialization. Details below:")
    st.code(traceback.format_exc())
