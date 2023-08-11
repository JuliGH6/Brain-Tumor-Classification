# Brain-Tumor-Classification

In this program, I read in MRI pictures, stored on my laptop.
I use the kaggle dataset "Brain Tumor MRI dataset"


Each picture is getting transformed into grayscale as well as reshaped into a one-dimensional array.
Additionally, the grayscale values of the pixels is scaled to be between 0 and 1.
Now each array of pixel values (=each picture) is added to a list. Another list is created where the labels are stored.
The indeices of one picture and its label are equal.
This process is done with training data. 

I use a SVM - SVC (Support Vector Classifier) as my estimator. 
Additionally I use gridsearch, to determine the best settings for the SVC.
The program can be tested with the testing data of the trainings set, which are read in, in the same way as the training data.

In the future, I am interested in the optimization of the classification as well as knowledge extraction through clustering.
