
import os
from PIL import Image
import numpy as np
from sklearn import metrics, svm
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pylab as plt

#experimental because of slow laptop
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV


working_directory = os.getcwd()


#reading in pictures from training set

data = []
labels = []
names = ['pituitary/Tr-pi_', 'notumor/Tr-no_', 'meningioma/Tr-me_', 'glioma/Tr-gl_']

# 0 = pituitary , 1 = notumor , 2 = meningioma , 3 = glioma
label = 0
for name in names:
    for x in range(10,1300):
        if x < 100:
            y = '00' + str(x)
        elif x < 1000:
            y = '0' + str(x)
        else:
            y = str(x)
        path = working_directory + '/Downloads/archive/Training/' + name + y + '.jpg'
        picture = np.array(Image.open(path).convert('L').resize((512,512)))
        picture = np.reshape(picture, -1)
        maxVal = max(picture)
        flatImage = [p/maxVal for p in picture]
        data.append(flatImage)
        labels.append(label)
    label += 1


#estimator

clf = svm.SVC(gamma=0.001)
mod = HalvingGridSearchCV(estimator=clf, 
                   param_grid={'gamma': [0.001, 0.0006, 0.0002, 0.00008], 
                               'C':[1.0,2.0,3.0],
                               'kernel':['linear','rbf','poly']},
                   cv=3)

mod.fit(data, labels)
pd.DataFrame(mod.cv_results_)


#reading in pictures from testing set

testData = []
testLabels = []
testNames = ['pituitary/Te-pi_', 'notumor/Te-no_', 'meningioma/Te-me_', 'glioma/Te-gl_']

# 0 = pituitary , 1 = notumor , 2 = meningioma , 3 = glioma
label = 0
for name in testNames:
    for x in range(10,299):
        if x < 100:
            y = '00' + str(x)
        elif x < 1000:
            y = '0' + str(x)
        else:
            y = str(x)
        path = working_directory + '/Downloads/archive/Testing/' + name + y + '.jpg'
        picture = np.array(Image.open(path).convert('L').resize((512,512)))
        picture = np.reshape(picture, -1)
        maxVal = max(picture)
        flatImage = [p/maxVal for p in picture]
        testData.append(flatImage)
        testLabels.append(label)
    label += 1


predicted = mod.predict(testData)
print(
    f"Classification report for classifier {mod}:\n"
    f"{metrics.classification_report(testLabels, predicted)}\n"
)


