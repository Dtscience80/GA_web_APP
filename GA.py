import streamlit as st
import pandas as pd

st.title('Streamlit Genetic Algorithm')
#upload single file and display it as a dataframe
file = st.file_uploader("Please select a file to upload")
#file = st.file_uploader("Upload file Excel", type=["xlsx", "xls"])
url = 'https://raw.githubusercontent.com/Dtscience80/Project_ISAST_2022/main/df_minmax.csv'
data = pd.read_csv(url)

if file is not None:
    #Can be used wherever a "file-like" object is accepted:
    #df= pd.read_csv(file)
    df = pd.read_excel(file) 
    st.dataframe(data.head())

st.text("Berikut tabel data anda :") 
st.dataframe(data, width=1000)
st.text("Berikut deskripsi data anda :") 
st.dataframe(data.describe())

st.sidebar.text('Feature seleksi ')
st.sidebar.text('Genetic ALgorithm')
header = data.columns.tolist()
st.text(" feature data anda : " )
st.text(header)

#Multi select
st.text(" feature data anda : " )
target = st.multiselect("Tentukan Target Feature seleksi", header)
st.write('Target selected:', target)
