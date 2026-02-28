import streamlit as st
import pandas as pd

st.title("📂 Načítanie a overenie dát")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview of dataset")
    st.dataframe(df.head())

    st.write("Number of rows:", df.shape[0])
    st.write("Number of columns:", df.shape[1])