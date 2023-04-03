import pickle
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

dftrain = pd.read_excel('train.xlsx')
dftest = pd.read_excel('test.xlsx')

dftrain['TotalCharges'] = pd.to_numeric(dftrain['TotalCharges'], errors='coerce')
dftest['TotalCharges'] = pd.to_numeric(dftest['TotalCharges'], errors='coerce')

dftrain.isna().sum()

dftrain['TotalCharges'].fillna(value=dftrain['TotalCharges'].mean(),inplace=True)

dftrain.isna().sum()

dftest.isna().sum()

dftrain.dtypes



CategoricalFeatures = [feature for feature in dftrain.columns if dftrain[feature].dtypes =='O']
NumericalFeatures = [feature for feature in dftrain.columns if feature not in CategoricalFeatures]

CategoricalFeatures.remove('customerID')

NumericalFeatures

NumericalFeatures.remove('SeniorCitizen')

CategoricalFeatures.remove('PhoneService')

CategoricalFeatures

len(CategoricalFeatures)

NullHypothesis=[]
AlternativeHypothesis=[]

for feature in CategoricalFeatures:
  ct_table_ind=pd.crosstab(dftrain[feature],dftrain["Churn"])
  #print('contingency_table :\n',ct_table_ind)
  stat, p, dof, expected = chi2_contingency(ct_table_ind)
  if p > 0.05:
    NullHypothesis.append(feature)
  else:
    AlternativeHypothesis.append(feature)

for feature in NumericalFeatures:
  ct_table_ind=pd.crosstab(dftrain[feature],dftrain["Churn"])
  #print('contingency_table :\n',ct_table_ind)
  stat, p, dof, expected = chi2_contingency(ct_table_ind)
  if p > 0.05:
    NullHypothesis.append


dftrain.nunique()

dftest.nunique()

dftrain['OnlineBackup'].value_counts()

dftrain['DeviceProtection'].value_counts()

dftest['OnlineBackup'].value_counts()

dftest['OnlineBackup'] = dftest['OnlineBackup'].replace('no','No')

dftest['OnlineBackup'].value_counts()

dftest['DeviceProtection'].value_counts()

dftest['DeviceProtection'] = dftest['DeviceProtection'].replace('yes','Yes')

dftest['DeviceProtection'].value_counts()

dftest.nunique()

dftrain[AlternativeHypothesis].nunique()

dftrain[NumericalFeatures].nunique()

dftrain.dtypes

df2=dftrain[AlternativeHypothesis]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for feature in AlternativeHypothesis:
  if df2[feature].nunique() == 2:
    df2[feature] = le.fit_transform(df2[feature])

df2['Churn']

for feature in AlternativeHypothesis:
  if feature !='TotalCharges' and feature !='customerID':
    if df2[feature].nunique() > 2 and df2[feature].nunique() <= 4:
      df_encoded = pd.get_dummies(df2[feature], prefix=feature)
      df2 = pd.concat([df2, df_encoded], axis=1)
      df2 = df2.drop(feature, axis=1)

df2

# Create a MinMaxScaler object
scaler = MinMaxScaler()
columns = df2.columns
# Apply the scaler to the data
scaled_data = scaler.fit_transform(df2)

# Convert the scaled data back to a DataFrame
df_scaled = pd.DataFrame(scaled_data, columns=columns)

# Print the first few rows of the scaled DataFra
df_scaled.head()

df_scaled

col = df_scaled.pop('Churn')
df_scaled = pd.concat([df_scaled, col], axis=1)

df_scaled.isna().sum()

df_scaled.drop_duplicates()

list2 = df_scaled.columns.values.tolist()

X=df_scaled[list2].iloc[:,:-1]
#X=X.drop(['3'],axis=1)
Y=df_scaled[df_scaled[list2].columns[-1]]

Y

AlternativeHypothesis.remove('Churn')

df3=dftest[AlternativeHypothesis]

df3

for feature in AlternativeHypothesis:
  #if feature!="Churn":
  if df3[feature].nunique() == 2:
      df3[feature] = le.fit_transform(df3[feature])

for feature in AlternativeHypothesis:
  if feature!="Churn":
    if df3[feature].nunique() > 2 and df3[feature].nunique() <= 4:
      df_encoded2 = pd.get_dummies(df3[feature], prefix=feature)
      df3 = pd.concat([df3, df_encoded2], axis=1)
      df3 = df3.drop(feature, axis=1)

testSize = len(df3)/len(df2)

X_train, X_test,Y_train,Y_test = train_test_split(X,Y ,test_size=testSize, random_state=123)

df3

Y_train

class_weights = class_weight.compute_class_weight('balanced', classes=[0.0, 1.0], y=Y_train)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

scaler = MinMaxScaler()
columns = df3.columns
# Apply the scaler to the data
scaled_data2 = scaler.fit_transform(df3)

# Convert the scaled data back to a DataFrame
df_scaled2 = pd.DataFrame(scaled_data2, columns=columns)

# Print the first few rows of the scaled DataFra
df_scaled2.head()

from sklearn.neighbors import KNeighborsClassifier
classifer = KNeighborsClassifier(n_neighbors=2)

# Train the model on the training data
classifer.fit(X_train, Y_train)
print("--------------------------->shape",df_scaled2.shape)
print("--------------------------->shape",df_scaled2.columns)
# Make predictions on the testing data
y_pred = classifer.predict(df_scaled2)

pickle.dump(classifer, open('model.pkl','wb'))

#load the model and test with a custom input
model = pickle.load( open('model.pkl','rb'))
y_pred2 = model.predict(df_scaled2)


# Calculate the accuracy score
accuracy = accuracy_score(Y_test, y_pred)
print('Accuracy:', accuracy)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)