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

target = st.selectbox('Tentukan target machine learning untuk fitur seleksi anda?', header)

st.write('Target Fitur Seleksi:', target)
st.text(dropdata)

st.text(" Data Anda " )
X = data.drop(columns=dropdata) 
st.dataframe(X, width=1000)

st.header('Fitur Seleksi ')

Y = d.astype(target)

#estimators = linear_model.LinearRegression()
estimators = DecisionTreeRegressor()
#estimators = RandomForestRegressor()
#estimators = MLPRegressor()
t1=time.time()
st.text(t1)

report = pd.DataFrame()
nofeats = [] 
chosen_feats = [] 
cvscore = [] 

st.text("Estimator dipakai : ", estimators )
st.text("Target : ", Y )
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
  st.text("Chosen Feats: {} of {}, scores : {} " .format(genfeats, selector.n_features_, round(selector.generation_scores_[-1], 3)))

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
st.text("Time taken: {} seconds".format(t_polyfit))

