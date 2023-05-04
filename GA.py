
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
import matplotlib.pyplot as plt
import seaborn as sns
import time



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

st.header('Fitur Seleksi ')

Y = data[target].astype(float) 

st.set_option('deprecation.showPyplotGlobalUse', False)
fig = plt.figure(figsize=(12,10))
#ax = sns.boxplot(x=data['CL'])
#fig = plt.show()

#plt.figure(figsize=(12,10))
cor = data.corr()
ax = sns.heatmap(abs(cor), cmap='PuBuGn' ,annot=True, fmt=".2f")
fig = plt.show()
st.pyplot(fig)

st.subheader(" 1. Filter Methode (Pearson correlation coefficient) ")

#estimators = linear_model.LinearRegression()
estimators = DecisionTreeRegressor()
#estimators = RandomForestRegressor()
#estimators = MLPRegressor()
t1=time.time()
st.write("Process Start", t1)

report = pd.DataFrame()
nofeats = [] 
chosen_feats = [] 
cvscore = [] 

st.write("Estimator dipakai : ", estimators )
st.write("Target : ", target )
for i in range(1,9):  
  selector = GeneticSelectionCV(estimators,
                                cv = 5,
                                verbose = 1,
                                scoring="neg_mean_squared_error", 
                                max_features = i,
                                n_population = 200,
                                crossover_proba = 0.5,
                                mutation_proba = 0.2,
                                n_generations = 50,
                                crossover_independent_proba=0.5,
                                mutation_independent_proba=0.1,
                                tournament_size = 3,
                                n_gen_no_change=10,
                                caching=True,
                                n_jobs=-1)
  selector = selector.fit(X, Y)
  genfeats = X.columns[selector.support_]
  genfeats = list(genfeats)
  #st.write("Chosen Feats: {} of {}, scores : {} " .format(genfeats, selector.n_features_, round(selector.generation_scores_[-1], 3)))

  cv_score = selector.generation_scores_[-1]
  nofeats.append(len(genfeats)) 
  chosen_feats.append(genfeats) 
  cvscore.append(cv_score)

report["No of Feats"] = nofeats
report["Chosen Feats"] = chosen_feats
report["Scores"] = cvscore
#Lama waktu Proses 
t2=time.time()
t_polyfit = float(t2-t1)
st.write("Time taken: {} seconds".format(t_polyfit))

#Print Reports 
report["Scores"] = np.round(report["Scores"], 3)
report.sort_values(by = "Scores", ascending = False, inplace = True)
#report.index
ga_feats = report.iloc[0]["Chosen Feats"]
DataTable(report)
st.write("Feature selection of '" + Y.name + "' recommend:", ga_feats)
st.write("Estimator : {}, reports : ". format(selector.estimator_))
report
