import streamlit as st
import pandas as pd

st.title('Streamlit Genetic Algorithm')
#upload single file and display it as a dataframe
file = st.file_uploader("Please select a file to upload")
#file = st.file_uploader("Upload file Excel", type=["xlsx", "xls"])
url = 'https://raw.githubusercontent.com/Dtscience80/Project_ISAST_2022/main/df_minmax.csv'
data = pd.read_csv(url)
st.text("Berikut tabel data anda :") 
if file is not None:
    #Can be used wherever a "file-like" object is accepted:
    #df= pd.read_csv(file)
    df = pd.read_excel(file)
    
    st.dataframe(data.head())

st.text("Berikut deskripsi data anda :") 
st.dataframe(data, width=1000)
st.dataframe(data.describe())
