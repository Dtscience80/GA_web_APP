
import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from genetic_selection import GeneticSelectionCV
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from sklearn.metrics import *
from sklearn.model_selection import *
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import time

def forward_selection(data, target, significance_level=0.05):
    initial_features = data.columns.tolist()
    best_features = []
    while (len(initial_features)>0):
        remaining_features = list(set(initial_features)-set(best_features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = sm.OLS(target, sm.add_constant(data[best_features+[new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if(min_p_value<significance_level):
            best_features.append(new_pval.idxmin())
        else:
            break
    return best_features

html_temp = """
		<div style="background-color:#9900FF;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Feature Selection webb Application</h1>
		<h4 style="color:white;text-align:center;">by @v1tr4 </h4>
		</div>
		"""
stc.html(html_temp)

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

#st.sidebar.text('Feature seleksi ')
#st.sidebar.text('Genetic ALgorithm')
header = data.columns.tolist()
st.text(" feature data anda : " )
st.text(header)

target = st.selectbox('Tentukan target machine learning untuk fitur seleksi anda?', header)
st.write('Target Fitur Seleksi:', target)

#Multi select
#st.text(" Drop Input yang tidak dipakai  : " )
dropdata = st.multiselect("Tentukan Feature data yang perlu di hilangkan (termasuk target yang tidak dipakai) ", header)
st.write('Feature drop:', dropdata)

#st.text(dropdata)

st.text(" Data Anda setelah di filter, dengan target " + target )
X = data.drop(columns=dropdata) 
st.dataframe(X, width=1000)
Y = data[target].astype(float) 

st.header('Fitur Seleksi ')
st.subheader(" 1. Filter Methode (Pearson correlation coefficient) ")

t1=time.time()
st.write("Process Start", t1)
st.set_option('deprecation.showPyplotGlobalUse', False)
fig = plt.figure(figsize=(12,10))
#ax = sns.boxplot(x=data['CL'])
#fig = plt.show()

#plt.figure(figsize=(12,10))
cor = data.corr()
ax = sns.heatmap(abs(cor), cmap='PuBuGn' ,annot=True, fmt=".2f")
fig = plt.show()
st.pyplot(fig)

#Correlation with output variable ex. CL
cor_target = abs(cor[target])
#Selecting highly correlated features
relevant_features_CL = cor_target[cor_target>0.5]
st.write('Feature yang relevan untuk target ' + target + ' adalah : ', relevant_features_CL)

#Lama waktu Proses 
t2=time.time()
t_polyfit = float(t2-t1)
st.write("Time taken: {} seconds".format(t_polyfit))

st.subheader(" 2. Sequential Forward Selection (SFS) Algorithms ")
t1=time.time()
st.write("Process Start", t1)
st.write("Hasil SFS : ", forward_selection(X, Y))
t2=time.time()
t_polyfit = float(t2-t1)
st.write("Time taken: {} seconds".format(t_polyfit))

