import streamlit as st
import pandas as pd

st.title('Fitur Seleksi dengan Genetic Algorithm')
st.text("by @v1tr4") 
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
st.text(" Drop Fitur yang tidak dipakai  : " )
dropdata = st.multiselect("Tentukan Feature data yang perlu di hilangkan ", header)
st.write('Feature drop:', dropdata)

option = st.selectbox(
     'Tentukan target machine learning untuk fitur seleksi anda?',
     header)

st.write('Target Fitur Seleksi:', option)

aktif = st.button("Aktifkan")
st.text(aktif)
#if (aktif=='True')
    X = data.drop(columns=dropdata) 
    st.dataframe(X, width=1000)

