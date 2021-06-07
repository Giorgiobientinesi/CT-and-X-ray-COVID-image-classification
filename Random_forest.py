import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
import random
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV  #!pip install scikit-learn==0.24.2

path = "CT/COVID"   #Same for X-ray


data = []

i = 0
while i < len(os.listdir(path)):
  try:
    path1 = os.path.join(path, os.listdir(path)[i])
    SP500_img = cv2.imread(path1,0)
    SP500_img = cv2.resize(SP500_img, (150,150))
    image = np.array(SP500_img).flatten()
    data.append([image,0])
  except Exception:
    print("errore")
  i+=1

path = "CT/Non-COVID"

i = 0
while i < len(os.listdir(path)):
  try:
    path1 = os.path.join(path, os.listdir(path)[i])
    SP500_img = cv2.imread(path1,0)
    SP500_img = cv2.resize(SP500_img, (150,150))
    image = np.array(SP500_img).flatten()
    data.append([image,1])
  except Exception:
    print("errore")
  i+=1

random.shuffle(data)

Features = []
Labels = []

for features,label in data:
    Features.append(features)
    Labels.append(label)

X_train, X_test, y_train, y_test = train_test_split(Features, Labels, test_size = 0.2)

from sklearn.model_selection import RandomizedSearchCV
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
estimators = [300, 400]
# Method of selecting samples for training each tree
bootstrap = [True, False]
depth = [3,5]
# Create the random grid
random_grid = {'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'n_estimators' : estimators}
print(random_grid)

from sklearn.ensemble import RandomForestClassifier #RANDOM FOREST

model = RandomForestClassifier() #Run the model and find the best structure
grid_search = HalvingGridSearchCV(model, random_grid, max_resources=10,random_state=0, refit=True,verbose=3) #cv default = 5
grid_search.fit(X_train,y_train)


#Found the best

classifier = RandomForestClassifier(bootstrap=False, min_samples_leaf=2, min_samples_split=5,  n_estimators=400, random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred)))
print('Precision Score : ' + str(precision_score(y_test,y_pred)))
print('Recall Score : ' + str(recall_score(y_test,y_pred)))
print('F1 Score : ' + str(f1_score(y_test,y_pred)))

from sklearn.metrics import confusion_matrix
print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred)))

y_pred1 = classifier.predict(X_train)

print('Accuracy Score : ' + str(accuracy_score(y_train,y_pred1)))
print('Precision Score : ' + str(precision_score(y_train,y_pred1)))
print('Recall Score : ' + str(recall_score(y_train,y_pred1)))
print('F1 Score : ' + str(f1_score(y_train,y_pred1)))

print('Confusion Matrix : \n' + str(confusion_matrix(y_train,y_pred1)))