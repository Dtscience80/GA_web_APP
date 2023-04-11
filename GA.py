import streamlit as st
import pandas as pd

st.title('Streamlit Genetic Algorithm')
#upload single file and display it as a dataframe
#file = st.file_uploader("Please select a file to upload")
file = st.file_uploader("Upload file CSV", type=["CSV", "csv"])
url = 'https://raw.githubusercontent.com/Dtscience80/Project_ISAST_2022/main/df_minmax.csv'
data = pd.read_csv(url)
if file is not None:
    #Can be used wherever a "file-like" object is accepted:
    #df= pd.read_csv(file)
    df = pd.read_excel(file)
    st.dataframe(df)

st.dataframe(data, width=1000)
st.dataframe(data.describe())

st.sidebar.text('Feature seleksi dengan ALgoritma Genetic ALgorithm')
header = data.columns.tolist()
st.subheader(" feature data anda : " + header )
