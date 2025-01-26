import streamlit as st

# Portfolio Title and Introduction
st.title("EcoSoil Insights: Capstone Project Portfolio")
st.markdown("""
Welcome to the portfolio for the **EcoSoil Insights** project. This project transforms static soil quality reports into an **interactive dashboard** 
to support Auckland's environmental sustainability goals. Explore the data cleaning process, dashboard features, and project outcomes below.
""")

# Navigation
st.sidebar.title("Navigation")
pages = st.sidebar.radio("Go to", ["Home", "Data Cleaning App", "Dashboard", "GitHub Repository"])

if pages == "Home":
    st.header("About the Project")
    st.markdown("""
    **Project Highlights:**
    - **Objective**: Create an interactive soil quality monitoring dashboard for Auckland.
    - **Tech Stack**: Python, Streamlit, Scikit-learn, Plotly.
    - **Outcome**: An interactive, scalable dashboard enabling informed decisions for soil management.
    """)

    st.subheader("Key Features")
    st.markdown("""
    - **Dynamic Dashboard**: Visualize trends, contamination levels, and soil health metrics.
    - **Data Cleaning App**: Automated preprocessing, missing value imputation, and contamination index calculation.
    - **Interactive Visualizations**: Enable stakeholders to filter and explore actionable insights.
    """)

elif pages == "Data Cleaning App":
    st.header("Data Cleaning App")
    st.markdown("Upload raw soil data, preprocess it, and download the cleaned dataset with contamination indexes.")
    st.code('streamlit run clean_with_encode_app.py', language='bash')
    st.markdown("""
    Run the above app locally or explore it [here](https://ecosoilinsightsdatacleaning.streamlit.app/).
    """)

elif pages == "Dashboard":
    st.header("Interactive Dashboard")
    st.markdown("Explore Auckland's soil quality trends and metrics with interactive filtering and visualizations.")
    st.code('streamlit run Soil_quality_Dashboard.py', language='bash')
    st.markdown("""
    Run the above dashboard locally or explore it [here](https://713-capstone-dashboard.streamlit.app/).
    """)

elif pages == "GitHub Repository":
    st.header("GitHub Repository")
    st.markdown("""
    View the complete project source code, including data cleaning and dashboard scripts:
    - [GitHub Repository](https://github.com/YiliaTao0122/713_Portfolio)
    """)

