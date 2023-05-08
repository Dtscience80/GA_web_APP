import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
from sklearn import linear_model
import sklearn.ensemble as ske
from sklearn.ensemble import RandomForestRegressor
#from sklearn.tree import DecisionTreeRegressor
#from genetic_selection import GeneticSelectionCV
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn import metrics
from sklearn.metrics import *
from sklearn.model_selection import *

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import time

#importing the necessary libraries
import joblib
import sys
sys.modules['sklearn.externals.joblib'] = joblib
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression

def embed(fung, x, y, model, target):
    import matplotlib
    reg = fung()
    reg.fit(x, y)
    st.write("Best alpha using " + model + ": %f" % reg.alpha_)
    st.write("Best score using " + model + ": %f" % reg.score(x,y))
    coef = pd.Series(reg.coef_, index = x.columns)
    st.write(model + " picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
    imp_coef = coef.sort_index()
    matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
    imp_coef.plot(kind = "barh")
    plt.title("Feature importance using " + model + " target " + target)
    fig = plt.show()
    st.pyplot(fig)
    emb = pd.DataFrame()
    emb["Scores"] = abs(imp_coef)
    #st.write("Score : ", emb.sort_values(by=['Score'], inplace=True))
    st.write("Score : ", emb)

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

def backward_elimination(data, target,significance_level = 0.1):
    features = data.columns.tolist()
    while(len(features)>0):
        features_with_constant = sm.add_constant(data[features])
        p_values = sm.OLS(target, features_with_constant).fit().pvalues[1:]
        max_p_value = p_values.max()
        if(max_p_value >= significance_level):
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
        else:
            break 
    return features

def read_file(filename):
    if not filename:
        raise ValueError("Nama file tidak boleh kosong")
    elif filename.endswith('.csv'):
        df = pd.read_csv(filename)
        return df
    elif filename.endswith('.xlsx'):
        df = pd.read_excel(filename)
        return df
    else:
        raise ValueError("File harus berformat CSV atau XLSX")

html_temp = """
    <div style="background-color:#9900FF;padding:10px;border-radius:10px">
    <h1 style="color:white;text-align:center;">Feature Selection webb Application</h1>
    <h4 style="color:white;text-align:center;">by @v1t </h4>
    </div>
    """
stc.html(html_temp)

#upload single file and display it as a dataframe
#file = st.file_uploader("Please select a file to upload")
#file = st.file_uploader("Upload file Excel", type=["xlsx", "xls"])
#url = 'https://raw.githubusercontent.com/Dtscience80/Project_ISAST_2022/main/df_minmax.csv'
#data = pd.read_csv(url)

#Multiple files
#adding a file uploader to accept multiple CSV file
delimiter = ','
data =[]

judul = st.text_input("Tuliskan Nama data anda")

st.markdown(' ##### Pilih Delimiter (separator) file yang dipakai ! ')
delimiter = st.selectbox('Tentukan Delimiter file tabel anda !', (':', ';', ',', '.', '/', '|', '+'))
st.write('Delimiter file tabel anda \'', delimiter + '\'')

uploaded_files = st.file_uploader("Please Upload data tabel dalam bentuk csv file", accept_multiple_files=True)
for file in uploaded_files:
    if uploaded_files is not None:
       #data = pd.read_csv(file, sep='[;:\s]+', engine='python')   
       data = pd.read_csv(file, sep=delimiter, engine='python')
       st.write("File uploaded:", file.name)
       st.dataframe(data.head())
       data = data.dropna()

       st.text("Berikut tabel data " + judul + " anda :") 
       st.dataframe(data, width=1000)
       st.text("Berikut deskripsi data " + judul + " anda :") 
       st.dataframe(data.describe())
       
       #st.sidebar.text('Feature seleksi ')
       #st.sidebar.text('Genetic ALgorithm')
       header = data.columns.tolist()
       
       st.text(" feature data anda : " )
       st.text(header)
       
       st.markdown(' ##### Filter data yang dipakai ! ')
       target = st.selectbox('Tentukan target machine learning untuk fitur seleksi anda !', header)
       st.write('Target Fitur Seleksi:', target)
       
       #Multi select
       st.text(" Hilangkan fitur / Input tidak dipakai, serta semua target !  " )
       dropdata = st.multiselect("Tentukan Feature data yang perlu di hilangkan (termasuk target) ", header)
       st.write('Feature drop:', dropdata)
       
       #st.text(dropdata)
       Xd = []
       st.text(" Data " + judul + " Anda setelah di filter, dengan target " + target )
       X = data.drop(columns=dropdata) 
       Y = data[target].astype(float) 
       Xd = pd.concat([X, Y], axis=1)
       st.dataframe(Xd)
       
       st.header('Fitur Seleksi ')
       st.subheader(" 1. Filter Methode (Pearson correlation coefficient) ")
       
       t1=time.time()
       #st.write("Process Start", t1)
       st.set_option('deprecation.showPyplotGlobalUse', False)
       fig = plt.figure(figsize=(12,10))
       #ax = sns.boxplot(x=data['CL'])
       #fig = plt.show()
       
       #plt.figure(figsize=(12,10))
       cor = Xd.corr()
       ax = sns.heatmap(abs(cor), cmap='PuBuGn' , annot=True, fmt=".2f")
       fig = plt.show()
       st.pyplot(fig)
       
       #Correlation with output variable ex. CL
       cor_target = abs(cor[target])
       #Selecting highly correlated features
       relevant_features_CL = cor_target[cor_target>0.5]
       st.write('Feature yang relevan untuk data ' + judul + ' dengan target ' + target + ' adalah : ', relevant_features_CL)
       
       #Lama waktu Proses 
       t2=time.time()
       t_polyfit = float(t2-t1)
       st.write("Time taken: {} seconds".format(t_polyfit))
       
       st.subheader(" 2. Sequential Forward Selection (SFS) Algorithms ")
       t1=time.time()
       #st.write("Process Start", t1)
       st.write("Hasil SFS data " + judul + " : ", forward_selection(X, Y))
       t2=time.time()
       t_polyfit = float(t2-t1)
       st.write("Time taken: {} seconds".format(t_polyfit))
       
       st.subheader(" 3. Sequential backward Selection (SBS) Algorithms ")
       t1=time.time()
       #st.write("Process Start", t1)
       st.write("Hasil SBS data " + judul + " : ", backward_elimination(X, Y))
       t2=time.time()
       t_polyfit = float(t2-t1)
       st.write("Time taken: {} seconds".format(t_polyfit))
       
       st.subheader(" 4. Sequential Floating Selection Algorithm")
       t1=time.time()
       #st.write("Process Start", t1)
       X1 = np.array(X)
       sbs = SFS(LinearRegression(),
                k_features=4,
                forward=False,
                floating=True,
                cv=0)
       sbs.fit(X1, Y)
       label = list(map(int, sbs.k_feature_names_))
       feature_name = X.columns.values
       labels = feature_name[label]
       sfls = pd.DataFrame()
       sfls["Feature Selected"] = labels
       st.write("Hasil Sequential Floating Selection data " + judul + " : ", sfls)
       t2=time.time()
       t_polyfit = float(t2-t1)
       st.write("Time taken: {} seconds".format(t_polyfit))
       
       st.subheader(" 5. Embedded Selection algorithm")
       t1=time.time()
       #st.write("Process Start", t1)
       st.write("Hasil untuk metode embedded feature selection untuk data " + judul + " adalah ")
       embed(LassoCV, X, Y, 'Lasso CV', target)
       embed(RidgeCV, X, Y, 'Ridge CV', target)
       
       t2=time.time()
       t_polyfit = float(t2-t1)
       st.write("Time taken: {} seconds".format(t_polyfit))
       
       st.subheader(" 6. Random Forest (RF) algorithm")
       t1=time.time()
       #st.write("Process Start", t1)
       
       reg = RandomForestRegressor()
       reg.fit(X, Y)
       fet_ind = np.argsort(reg.feature_importances_)[::-1]
       fet_imp = reg.feature_importances_[fet_ind]
       feature_name = X.columns.values
       labels = feature_name[fet_ind]
       
       rf = pd.DataFrame()
       rf["labels"] = labels
       rf["Score"] = fet_imp
       #display dataframe
       st.write("Feature selected untuk data " + judul + " adalah : ", rf)
       
       fig, ax = plt.subplots(1, 1, figsize=(8, 3))
       pd.Series(fet_imp, index=labels).plot(kind='bar', ax=ax)
       ax.set_title('Features importance ' + judul + ' data')
       fig = plt.show()
       st.pyplot(fig)
       
       t2=time.time()
       t_polyfit = float(t2-t1)
       st.write("Time taken: {} seconds".format(t_polyfit))
