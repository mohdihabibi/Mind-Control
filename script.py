from sklearn.externals import joblib
import numpy as np
import scipy as sp
import pandas as pd
import scipy.io
from sklearn.preprocessing import Imputer
from sklearn.cluster import KMeans
from numpy import genfromtxt
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

#read the csv into a dataframe
df = pd.read_csv('~/part1.csv')

imp = Imputer(missing_values='NaN', strategy='median', axis=1)
imputed_DF = pd.DataFrame(imp.fit_transform(df))
imputed_DF.columns = df.columns
imputed_DF.index = df.index

# creating a kmean object
kmean = KMeans(n_clusters=4) #n_clusters asks for only 4 groupings
kmean.fit(df.values)

row = 0;

#adding an extra column with the class
while(df.shape[0] > row ):
	df.loc[row, "Class"] = kmean.predict(df.loc[row][:25])
	row += 1


X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :25], df.Class, stratify= df.Class)

knnClassifier = KNeighborsClassifier(n_neighbors=2)
knnClassifier.fit(X_train, y_train)
train_accuracy = knnClassifier.score(X_train, y_train)
test_accuracy = knnClassifier.score(X_test, y_test))

scores = cross_val_score(knnClassifier, df.iloc[:, :25], df.Class, cv=5, scoring='f1_macro')
cross_accuracy = np.average(scores)

print("training accuracy is: \n", train_accuracy)
print("testing accuracy is: \n", test_accuracy)
print("cross fold accuracy is: \n", cross_accuracy)

forest = RandomForestClassifier(n_estimators=10, random_state=0) 
forest.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train))) 
print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))
random_score = cross_val_score(forest, df.iloc[:, :25], df.Class, cv=5, scoring='f1_macro')
print("random forest cross fold accuracy is: \n", np.average(random_score))
print("try a prediction using random forest:\n")
print(forest.predict(df.iloc[10, :25]))
print(df.iloc[:1,:]) 

filename = 'finalized_KNN_model.sav'
joblib.dump(knnClassifier, filename)

filename1 = 'finalized_random_forest_model.sav'
joblib.dump(forest, filename1)



