import pandas as pd
import numpy as np
import pickle

df = pd.read_csv("//root//Desktop//Staff.csv", usecols = ['Age', 'Attrition' , 'JobSatisfaction','MonthlyIncome','YearsSinceLastPromotion'])

 
categoricals = []
for col, col_type in df.dtypes.iteritems():
     if col_type == 'O':
          categoricals.append(col)
     else:
          df[col].fillna(0, inplace= True)

df.info()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

 

df['Attrition'] = le.fit_transform(df.Attrition)
 

df.columns
x = df.iloc[ :, [0,2,3,4]].values
y = df.iloc[:, 1].values
 
from sklearn.model_selection import train_test_split
train_X,test_X,train_Y,test_Y = train_test_split(x,y, test_size = 0.25, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
train_X = sc_X.fit_transform(train_X)
test_X = sc_X.transform(test_X)

from sklearn.linear_model import LogisticRegression
lgr = LogisticRegression(random_state = 0)
lgr.fit(train_X, train_Y)


 
# Saving model to disk
pickle.dump(lgr, open('model.pkl','wb'))


# Load the model that was just saved
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[30, 1, 2000, 3]]))




""" Saving the data columns from training
model_columns = list(x)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")
"""
 
