import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
import streamlit as st
from sklearn.metrics import classification_report
from pandas.core.algorithms import mode
 
st.header('SMOKING ANALYSIS') 
st.write('This dataset is a collection of basic health biological signal data. ')
st.write('The goal is to determine and predict the the smoking habit of people by their bio-signals.')
st.write('The Aim of this study is to find correlations among health signals to adopt a classification model to find an health algorithm for people')

smoking_dff = pd.read_csv('smoking.csv (1).zip')


if st.checkbox('DATA SET') :
  st.write('The raw data:')
  st.write(smoking_dff)

def get_downloadable_data (df):
  return df.to_csv().encode('utf-8')

st.download_button('DOWNLOAD DATA SET' , get_downloadable_data(smoking_dff), file_name='smoking.csv' )  
smoking_dff.describe()

smoking_df = smoking_dff.drop(columns= ["ID", "gender", "oral", "tartar" ]  )

numeric_columns = smoking_df.select_dtypes(['float64', 'float32', 'int32', 'int64']).columns
st.set_option('deprecation.showPyplotGlobalUse', False)

if st.checkbox('Describe') :
  st.write(smoking_df.describe())

if st.sidebar.checkbox('Histogram Plot') :
   st.sidebar.subheader("Histogram")
   select_box3 = st.sidebar.selectbox(label="Feature", options=numeric_columns)
   histogram_slider = st.sidebar.slider(label="Number of Bins",min_value=5, max_value=100, value=30)
   sns.distplot(smoking_df[select_box3], bins=histogram_slider)
   st.pyplot()    




smoking_df.tail(10)

smoking_df.head(10)

smoking_df.corr()

if st.checkbox('CORRELATIONS ') :
  st.write('THE CORRELATIONS:')
  H = plt.figure(figsize = (18,18))
  sns.heatmap(smoking_df.corr(), annot = True)
  st.write(H)

if st.checkbox('Distribution of LDL AND CHOLESTEROL Values Of CONDIDATES IN SCATTER ') :
  fig = px.scatter(smoking_df, x="LDL", y="Cholesterol", color="smoking", title=
        "LDL/Cholestrol")
  st.write(fig)

smoking_df = pd.read_csv('smoking.csv (1).zip')

smoking_df['BMI']= smoking_df ['weight(kg)'] / (smoking_df['height(cm)'] / 100 ) **2

if st.checkbox('Adding BMI Column'):
  st.write("Adding BMI Column :")
  st.write(smoking_df)

if st.checkbox('BMI INFORMATION'):
  #'BMI = weight(kg) / height(m) * height(m)'
  st.write(' BMI < 18.5     UNDERWEIGHT')
  st.write('18.5 <= BMI <= 24.5     HEALTYHY WEIGHT')
  st.write('24.5 < BMI <= 29.9    OVERWEIGHT') 
  st.write ('BMI >= 30    OBESITY')


smoking_df.nunique().sort_values()


st.header("Percentages of Smokers among Candidates")

col_1, col_2 = st.columns(2)

with col_1 :
  fig = plt.figure(figsize= (8,4))
  smoking_df['gender'].value_counts().plot.pie(explode=[0,0.3],autopct=lambda x: str(x)[:4] + '%', shadow =True)
  st.write(fig)
  st.caption('Gender Share among Candidates')

with col_2 :
  fig =plt.figure(figsize= (8,4))
  smoking_df['smoking'].value_counts().plot.pie(explode=[0,0.3],autopct=lambda x: str(x)[:4] + '%', shadow =True)
  st.write(fig)
  st.caption ('Smokers Proportion')


if st.checkbox('Age of condidates') :
   st.write("Age of oldest condidates  : ")
   st.write(smoking_df['age'].max())
   st.write("Mean of ages : ")
   st.write(smoking_df['age'].mean())
   st.write(" Age of Yongest condidates: ")
   st.write(smoking_df['age'].min())

if st.checkbox('Smoking condidates') :
  st.write("Smoking condidates : ")
  smoking_df.loc[smoking_df['smoking'] == 1]

smoking_df['age'].value_counts()

if st.checkbox('Number of Smoking condidates') :
   st.write("Number of Smoking condidates :")
   st.write(smoking_df['smoking'].sum())

if st.checkbox ('Number of Smokers in every Age group'):
  _x = smoking_df['age'].value_counts().index
  _y = smoking_df['age'].value_counts()
  fig = plt.figure(figsize= (10,6))
  plt.bar(_x,_y)
  plt.title('smoking_candidates')
  plt.xlabel('age')
  plt.ylabel(' number_of_smoking_condidates')
  st.write("AGE OF SMOKING :")
  st.write(fig)
  

st.header("SOME DISTRIBUTION OF SMOKERS ")
with st.expander('CHOOSE CRITERIAN'): 
 select_criterian = st.selectbox('select model :', ['CHOLESTEROL', 'weight(kg)' , 'hemoglobin' ])
 if select_criterian  == 'CHOLESTEROL' :
   A = smoking_df[smoking_df['smoking']==1]
   fig = plt.figure(figsize=(10,6) )
   sns.distplot(A['Cholesterol'],color = 'red' , bins= 90)
   plt.title("Cholesterol Distribution of Smokers")
   st.write(fig)
 elif select_criterian  == 'weight(kg)' :
     A = smoking_df[smoking_df['smoking']==1]
     fig = plt.figure(figsize=(10,6))
     sns.distplot(A['weight(kg)'],color = 'red' , bins= 60)
     plt.title("Weight Distribuuion of Smokers")
     st.write(fig)
 else :
     A = smoking_df[smoking_df['smoking']==1]
     fig = plt.figure(figsize=(10,6))
     sns.distplot(A['hemoglobin'],color = 'red' , bins= 90)
     plt.title("hemoglobin Distribution of Smokers")
     st.write(fig)
          
oldest_candidates = smoking_df['age'].sort_values(ascending = False)
if st.checkbox('oldest Candidates ID') :
  oldest_candidates[:10].index

youngest_condidates=smoking_df['age'].sort_values(ascending = True)
if st.checkbox(' 10 Youngest candidates ID') :
   youngest_condidates[:10].index

if st.checkbox('Mean of Smoking Age') :
  candidates_groupby_age_smoking = smoking_df.groupby('smoking')['age'].mean()
  candidates_groupby_age_smoking


if st.checkbox('Considering genders according to the mean of Age,LDL & Cholesterol'):
  candidates_groupby =smoking_df.groupby(['gender','smoking'])['age','LDL','Cholesterol'].mean()
  candidates_groupby
  
  

if st.checkbox("Consider tartar on candidates who are smoking ") :
  tartar_smoking_people =smoking_df.groupby('tartar')['smoking'].sum()
  fig = tartar_smoking_people
  st.write(fig)


if st.checkbox("Consider dental caries on candidates who are smoking") :
  dental_caries_smoking_people =smoking_df.groupby('dental caries')['smoking'].sum()
  dental_caries_smoking_people
  fig = plt.figure(figsize=(10,6))
  dental_caries_smoking_people.plot(kind='bar', title='comparing_dental_caries_smoking_people',figsize=(10, 6))
  st.write(fig)

# change datatype column gender 
le = LabelEncoder()
le.fit(smoking_df["gender"])
smoking_df["gender"]=le.transform(smoking_df["gender"])  

# change datatype column oral 
l = LabelEncoder()
l.fit(smoking_df["oral"])
smoking_df["oral"]=l.transform(smoking_df["oral"])

# change datatype column tartar 
a = LabelEncoder()
a.fit(smoking_df["tartar"])
smoking_df["tartar"]=a.transform(smoking_df["tartar"])

if st.checkbox('Pairplot') :
  fig = sns.pairplot(smoking_df, hue = 'smoking', vars = ['fasting blood sugar', 'hemoglobin', 'LDL','Cholesterol'])
  plt.subplots_adjust(top=0.9)
  st.pyplot(fig)


y_smoking_df = smoking_df['smoking']
x_smoking_df = smoking_df.drop( 'smoking' , axis = 1) 
x_train, x_test, y_train, y_test = train_test_split(x_smoking_df, y_smoking_df, test_size=0.2, random_state=42)
model = GaussianNB()
model.fit(x_train , y_train)
y_pred = model.predict(x_test)

if st.sidebar.checkbox(" GaussianNB Report") :
   #  GaussianNB Report
  st.header("GaussianNB") 
  st.write(classification_report(y_test, y_pred))


if st.sidebar.checkbox("CONFUSION MATRIX GaussianNB") :
  st.subheader("Confusion Matrix GaussianNB ")
  cf_matrix = confusion_matrix(y_test, y_pred)
  st.write(cf_matrix)

if st.sidebar.checkbox("CONFUSION MATRIX HEAT MAP GaussianNB") :
  st.subheader("Confusion Matrix Heat Map GaussianNB ")
  l= plt.figure(figsize = (10,6))
  sns.heatmap(cf_matrix,fmt='.0f' , annot = True)
  plt.title('confusion_matrix_heatmap_GaussianNB')
  plt.xlabel('Predicted Values')
  plt.ylabel('True Values')
  st.write(l)


#classification Random Forest
y_smoking_df = smoking_df['smoking']
x_smoking_df = smoking_df.drop( 'smoking' , axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x_smoking_df, y_smoking_df, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy_score(y_test, y_pred)

if st.sidebar.checkbox(" Feature Importance Random Forest ") :
  st.subheader("Feature Importance Plot Random Forest")
  i = plt.figure(figsize = (14,6))
  model = RandomForestClassifier()
  model.fit(x_train, y_train)
  sort = model.feature_importances_.argsort()
  plt.barh(smoking_df.columns[sort], model.feature_importances_[sort])
  plt.xlabel("Feature Importance")
  st.write(i) 


if st.sidebar.checkbox(" RandomForest Report") :
# RandomForest Report
  st.header("Random Forest")
  st.write(classification_report(y_test, y_pred))


if st.sidebar.checkbox("CONFUSION MATRIX RandomForest") :
  cf_matrix = confusion_matrix(y_test, y_pred)
  st.write(cf_matrix)

if st.sidebar.checkbox("CONFUSION MATRIX HEAT MAP RandomForest") :
  st.subheader("CONFUSION MATRIX HEAT MAP RandomForest")
  v = plt.figure(figsize = (10,6))
  sns.heatmap(cf_matrix,fmt='.0f' , annot = True)
  plt.title('confusion_matrix_heatmap_RandomForestClassifier')
  plt.xlabel('Predicted Values')
  plt.ylabel('True Values')
  st.write(v)


#clasification 3 Decision Tree
y_smoking_df = smoking_df['smoking']
x_smoking_df = smoking_df.drop( 'smoking' , axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x_smoking_df, y_smoking_df, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(x_train , y_train)
y_pred = model.predict(x_test)
accuracy_score(y_test, y_pred)

if st.sidebar.checkbox(" Feature Importance Decision Tree ") :
  st.subheader("Feature Importance Plot Decision Tree")
  o = plt.figure(figsize = (14,6))
  model = DecisionTreeClassifier()
  model.fit(x_train, y_train)
  sort = model.feature_importances_.argsort()
  plt.barh(smoking_df.columns[sort], model.feature_importances_[sort])
  plt.xlabel("Feature Importance")
  st.write(o)

if st.sidebar.checkbox(" Decision Tree Report") :
# Decision Tree Report
   st.header("DecisionTree")
   st.write(classification_report(y_test, y_pred))

if st.sidebar.checkbox("CONFUSION MATRIX Decisiontree") :
  st.subheader("Confusion Matrix DecisionTree")
  cf_matrix = confusion_matrix(y_test, y_pred)
  st.write(cf_matrix)


if st.sidebar.checkbox("CONFUSION MATRIX HEAT MAP Decisiontree") :
  st.subheader("Confusion Matrix Heat Map DecisionTree")
  p = plt.figure(figsize = (10,6))
  sns.heatmap(cf_matrix,fmt='.0f' , annot = True)
  plt.title('Confusion_matrix_heatmap_DecisionTreeClassifier')
  plt.xlabel('Predicted Values')
  plt.ylabel('True Values')
  st.write(p)



with st.expander('SHOW ACCURACY OF MODELS'): 
  select_model = st.selectbox('select model :', ['Randomforest' ,'GaussianNB', 'DecisionTreeClassifier'] )
  if select_model == 'GaussianNB' :
    y_smoking_df = smoking_df['smoking']
    x_smoking_df = smoking_df.drop( 'smoking' , axis = 1) 
    x_train, x_test, y_train, y_test = train_test_split(x_smoking_df, y_smoking_df, test_size=0.2, random_state=42)
    model = GaussianNB()
    model.fit(x_train , y_train)
    y_pred = model.predict(x_test)
    sum(y_pred == y_test) / len(y_pred)
    accuracy_score(y_test, y_pred)
  elif  select_model == 'Randomforest' :
       y_smoking_df = smoking_df['smoking']
       x_smoking_df = smoking_df.drop( 'smoking' , axis = 1) 
       x_train, x_test, y_train, y_test = train_test_split(x_smoking_df, y_smoking_df, test_size=0.2, random_state=42)
       model = RandomForestClassifier()
       model.fit(x_train, y_train)
       y_pred = model.predict(x_test)
       st.write(accuracy_score(y_test, y_pred))   
  else :
        model = DecisionTreeClassifier()
        model.fit(x_train , y_train)
        y_pred = model.predict(x_test)
        st.write(accuracy_score(y_test, y_pred))



with st.expander('SHOW KFOLD ACCURACIES '): 
    select_model = st.selectbox('select model :', ['RandomForestClassifier' ,'GaussiaNNB', 'Descisiontree'] )
    if select_model == 'GaussiaNNB' :
      model = GaussianNB()
      y_smoking_df = smoking_df['smoking']
      x_smoking_df = smoking_df.drop( 'smoking' , axis = 1) 
      x_train, x_test, y_train, y_test = train_test_split(x_smoking_df, y_smoking_df, test_size=0.2, random_state=42)
      kf = KFold(n_splits=10, shuffle=True, random_state=42)
      accuracies = []
      i = 0
      for train_index, test_index in kf.split(x_smoking_df,y_smoking_df):
         i += 1
         x_train, y_train = x_smoking_df.iloc[train_index], y_smoking_df.iloc[train_index]
         x_test, y_test =  x_smoking_df.iloc[test_index], y_smoking_df.iloc[test_index]
         model.fit(x_train, y_train)
         y_pred = model.predict(x_test)
         accuracy = accuracy_score(y_pred, y_test)
         accuracies.append(accuracy)
         st.write(i, ') Accuracy = ', accuracy)
  
  
      st.write('Mean accuracy: ', np.array(accuracies).mean())
    elif  select_model == 'RandomForestClassifier' :
          y_smoking_df = smoking_df['smoking']
          x_smoking_df = smoking_df.drop( 'smoking' , axis = 1) 
          x_train, x_test, y_train, y_test = train_test_split(x_smoking_df, y_smoking_df, test_size=0.2, random_state=42)
          model = RandomForestClassifier()  
          kf = KFold(n_splits=10, shuffle=True, random_state=42)
          accuracies = []
          i = 0
          for train_index, test_index in kf.split(x_smoking_df,y_smoking_df):
            i += 1
            model = RandomForestClassifier(random_state=42)
            x_train, y_train = x_smoking_df.iloc[train_index], y_smoking_df.iloc[train_index]
            x_test, y_test =  x_smoking_df.iloc[test_index], y_smoking_df.iloc[test_index]
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_pred, y_test)
            accuracies.append(accuracy)
            st.write(i, ') accuracy = ', accuracy)


          st.write('Mean accuracy: ', np.array(accuracies).mean())
    else :
            y_smoking_df = smoking_df['smoking']
            x_smoking_df = smoking_df.drop( 'smoking' , axis = 1) 
            x_train, x_test, y_train, y_test = train_test_split(x_smoking_df, y_smoking_df, test_size=0.2, random_state=42)
            model = DecisionTreeClassifier()
            kf = KFold(n_splits=10, shuffle=True, random_state=42)
            accuracies = []
            i = 0
            for train_index, test_index in kf.split(x_smoking_df,y_smoking_df):
              i += 1
              model = DecisionTreeClassifier(random_state=42)
              x_train, y_train = x_smoking_df.iloc[train_index], y_smoking_df.iloc[train_index]
              x_test, y_test =  x_smoking_df.iloc[test_index], y_smoking_df.iloc[test_index]
              model.fit(x_train, y_train)
              y_pred = model.predict(x_test)
              accuracy = accuracy_score(y_pred, y_test)
              accuracies.append(accuracy)
              st.write(i, ') accuracy = ', accuracy)

            st.write('Mean accuracy: ', np.array(accuracies).mean())




def train_model(x , y , model , random_state = 42 , test_size= 0.2):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print ('accuracy = ' , accuracy_score(y_pred, y_test ))
st.write('This Machine learning method was adopted as the RandomForest Classification had the highest accuracy among all, The five mentioned indexes had the highest influence on the dataframe according to the Feature Importance algorythm')
if st.checkbox("RandomForestClassifier on 'hemoglobin', 'Gtp' , 'triglyceride', 'height(cm)'") :
     x = smoking_df[['hemoglobin' , 'Gtp' , 'triglyceride', 'height(cm)']]
     y = smoking_df [ 'smoking']
     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
     model = RandomForestClassifier()
     model.fit(x_train, y_train)
     y_pred = model.predict(x_test)
     st.write(accuracy_score(y_test, y_pred))


x = smoking_df[['hemoglobin' , 'Gtp' , 'triglyceride', 'height(cm)']]
y = smoking_df [ 'smoking']
 
if st.checkbox("Random forest KFOLD (4 columns only)") :
  kf = KFold(n_splits=10, shuffle=True, random_state=42)
  accuracies = []
  i = 0
  for train_index, test_index in kf.split(x,y):
      i += 1
      model = RandomForestClassifier(random_state=42)
      x_train, y_train = x.iloc[train_index], y.iloc[train_index]
      x_test, y_test = x.iloc[test_index], y.iloc[test_index]
      model.fit(x_train, y_train)
      y_pred = model.predict(x_test)
      accuracy = accuracy_score(y_pred, y_test)
      accuracies.append(accuracy)
      st.write(i, ') Accuracy = ', accuracy)

      st.write('Mean accuracy: ', np.array(accuracies).mean())    
