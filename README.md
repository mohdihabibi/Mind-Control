# Mind-Control
The data that is being used for this project was open  source. The goal of this project was to get the raw EEG data and make a ML model so that can predict the new reading.

Since the initial data did not have a class, using Kmean clustring to predict the actual class for each reading. This would consider
as part of your data preprocessing.

In order to use this with your own dataset, first get the data into pandas dataframe and change the location of the file. 
Using the matlab.py script, you can run your evaluation set to get the optimal K value for KNN. This will basically would be a 
portion of your dataset to find the optimal K value then using the actual dataset run the script.py file to make your models.
Using the script you are building two models, one KNN and Random Forest. Both of them will be saved on your directory and later
on you can just include them to any of your files and use that model to predict.

There are several examples on how to predict using the model in matlab.py file.
