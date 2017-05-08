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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
import warnings
warnings.filterwarnings("ignore")

#read the csv into a dataframe
df = pd.read_csv('/Users/Mohdi/Downloads/CMPE188/test.csv')

#print the contents of the dataframe
# print(df)

#this will show the count of each column
# df.count()

# info about the dataframe
# df.describe()
# df.describe().transpose()

#create another dataframe with dropping 
# the rows contatinig empty cell
# df2 = df.dropna()

#another way to prepare your data is to impute it 
#can modify the strategy
imp = Imputer(missing_values='NaN', strategy='median', axis=1)
imputed_DF = pd.DataFrame(imp.fit_transform(df))
imputed_DF.columns = df.columns
imputed_DF.index = df.index

# creating a kmean object
kmean = KMeans(n_clusters=4) #n_clusters asks for only 4 groupings
kmean.fit(df.values)

#in order to print the cluster centers
print(kmean.cluster_centers_)

cluster_center1 = kmean.cluster_centers_[0][0]
cluster_center2 = kmean.cluster_centers_[1][0]
cluster_center3 = kmean.cluster_centers_[2][0]
cluster_center4 = kmean.cluster_centers_[3][0]

# giving the cluster a name
clusterDict = { cluster_center1: 1, #right hand
			    cluster_center2: 2, #left hand
			    cluster_center3: 3, #tongue
			    cluster_center4: 4} #legs

row = 0;

#adding an extra column with the class
while(df.shape[0] > row ):
	df.loc[row, "Class"] = kmean.predict(df.loc[row][:25])
	row += 1

#printing with the added column
# print(df)


# in order to visualize our clusters, we used feature engineering and dimensionality reduction 
# to bring our data into 3rd and show the clusters. Since our data is EEG sensor data, we decided 
# to use the first 3 sensor data and cluster it one more time to amek the new model

df2 = df.iloc[:, 1:4]
kmean3rd = KMeans(n_clusters=4)
kmean3rd.fit(df2.values)

row = 0;

#adding an extra column with the class
while(df2.shape[0] > row ):
	df2.loc[row, "Class"] = kmean3rd.predict(df2.loc[row][:3])
	row += 1


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

row = 0
while(df2.shape[0] > row):
	if df2.loc[row, "Class"] == 0:
		ax.scatter(df2.iloc[row, 0], df2.iloc[row, 1], df2.iloc[row , 2], c = 'r', marker = 'o')
	elif df2.loc[row, "Class"] == 1:
		ax.scatter(df2.iloc[row, 0], df2.iloc[row, 1], df2.iloc[row, 2], c = 'b', marker = '^')
	elif df2.loc[row, "Class"] == 2: 
		ax.scatter(df2.iloc[row, 0], df2.iloc[row, 1], df2.iloc[row, 2], c = 'g', marker = 's')
	else : 
		ax.scatter(df2.iloc[row, 0], df2.iloc[row, 1], df2.iloc[row, 2], c = 'black', marker = '*')
	row += 1

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()



# Now that we have the data ready we use KNN to make a ML model

# the first step we separate our training and testing data
# for having a better result we use 66 as split and random
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :25], df.Class, stratify= df.Class)

#Now we need to find the optimal N
training_accuracy = []
test_accuracy = []

cross_fold_accuracy = []

# try n_neighbors from 1 to 10.
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
    # build the model
    knnClassifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    knnClassifier.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(knnClassifier.score(X_train, y_train))
    # record generalization accuracy
    test_accuracy.append(knnClassifier.score(X_test, y_test))
    # using cross validation and random sampling to get the ideal k
    scores = cross_val_score(knnClassifier, df.iloc[:, :25], df.Class, cv=5, scoring='f1_macro')
    cross_fold_accuracy.append(np.average(scores))

for iterator in range(len(training_accuracy)):
	print("training accuracy is: \n", training_accuracy[iterator])
	print("testing accuracy is: \n", test_accuracy[iterator])
	print("cross fold accuracy is: \n", cross_fold_accuracy[iterator])

print(knnClassifier.predict(df.iloc[15, :25]))
print(df.iloc[15, 25])

for index in range(len(training_accuracy)):
	plt.plot(index, training_accuracy[index], c = 'r', marker = 'o')
	plt.plot(index, test_accuracy[index], c = 'g', marker = '^')
	plt.plot(index, cross_fold_accuracy[index], c = 'b', marker 3214 's')


plt.show()


# Now we try to use different classifier and compare it with KNN
# this time we use Random Forest

forest = RandomForestClassifier(n_estimators=100, random_state=0) 
forest.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train))) 
print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))
random_score = cross_val_score(forest, df.iloc[:, :25], df.Class, cv=5, scoring='f1_macro')
print(np.average(random_score))
print(forest.predict(df.iloc[10, :25]))
print(df.iloc[:1,:]) 

filename = 'finalized_model.sav'
joblib.dump(forest, filename)


