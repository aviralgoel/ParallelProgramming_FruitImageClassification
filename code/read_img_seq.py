# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 15:06:42 2020

@author: Nasser
"""

import os
import pandas as pd
import image_feature2
from datetime import datetime
import time
import pandas as pd
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler




image_list = [] #List which contain all images info (file name, class num, train/test)
df_train = pd.DataFrame() #store the training data
df_test = pd.DataFrame() #store the testing data
num_image=0
log_file = open("seq.txt", "a")

###+++++++++Creating the list of images+++++++#####
log_file.write("#########################################\n")
log_file.write("#########++++++++New Test++++++++########\n")
log_file.write("#########################################\n")
start_time = time.time()
#loop to get the list of testing images info
for root, dirnames, filenames in os.walk("train/"):
	class_num = 0
	for class_name in dirnames:
		class_num+=1
		for rt, classname, files in os.walk(root+class_name):
			for img_file in files:
				image_item = [rt+"/"+img_file, class_num, 0]
				
				image_list.append(image_item)
				num_image+=1

#loop to get the list of testing images info
for root, dirnames, filenames in os.walk("test/"):
	class_num = 0
	for class_name in dirnames:
		class_num+=1
		for rt, classname, files in os.walk(root+class_name):
			for img_file in files:
				image_item = [rt+"/"+img_file, class_num, 1]
				
				image_list.append(image_item)
				num_image+=1

log_file.write("Total Number of images = " + str(num_image) + "\n")
log_file.write("Time to create a list of images is:\n")
log_file.write(str(time.time() - start_time) + "\n")
log_file.write("#################################################\n")
###+++++++++Calculate the features+++++++#####

start_time = time.time()#time stamp
itr = 0


#loop to calculate the features
for i in range(num_image):
	image_item = image_list[i]
	
	features = image_feature2.get_image_dataframe(image_item[0],image_item[1])
	
	#construct the dataframe using the calculated features
	col = ["label","ur","ug","ub","stdr","stdg","stdb","skwr","skwg","skwb","kirtr","kirtg","kirtb",
       "u_mf","std_mf","skw_mf","kirt_mf",
       "Ct0","Ct45","Ct90","Ct135","Cn0","Cn45","Cn90","Cn135",
       "Ey0","Ey45","Ey90","Ey135","Hy0","Hy45","Hy90","Hy135"]

	df_item = pd.DataFrame(features,columns = col)
	
	#check if it is train or test image
	if(image_item[2] == 0):
		df_train = df_train.append(df_item,ignore_index = True,sort = False)
	else:
		df_test = df_test.append(df_item,ignore_index = True,sort = False)
	
	#print the progress of processing the images
	print(str(i) + "/" + str(num_image) , end = "\r")
	
print(str(num_image) + "/" + str(num_image))	
#print how much time it take to process the images
log_file.write("Time to process these images is:\n")
log_file.write(str(time.time() - start_time) + "\n")
log_file.write("#################################################\n")

###+++++++++Creating the Scalled features+++++++#####

start_time = time.time()
CF = [1,2,3,4,5,6,7,8,9,10,11,12]#Color Features
MF = [13,14,15,16]#Morphological Features
TF = [17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]#Texture Features

trainX = df_train.iloc[:, CF+MF+TF].values # train features
trainY = df_train.iloc[:, 0:1].values.flatten() # train labels

testX = df_test.iloc[:, CF+MF+TF].values # test features
testY = df_test.iloc[:, 0:1].values.flatten() # test labels

scaler = StandardScaler()
scaler.fit(trainX)

testX_Z_score=scaler.transform(testX)#scale the test data
trainX_Z_score=scaler.transform(trainX)#scale the train data

scalled_feature = [trainY, testY, trainX_Z_score, testX_Z_score]

log_file.write("Time to create the scalled features is:\n")
log_file.write(str(time.time() - start_time) + "\n")
log_file.write("#################################################\n")


###+++++++++Creating the list of Classifire+++++++#####

start_time = time.time()
classifier_list = []

#add KNN classifier to the list
knn = neighbors.KNeighborsClassifier(n_neighbors=1,algorithm='ball_tree')
classifier_list.append(knn)

#add GaussianNB classifier to the list
gau = GaussianNB()
classifier_list.append(gau)

#add SVM to the classifier list with different parameters
for C1 in [0.1,1, 10, 100]:
	for gamma1 in [1,0.1,0.01,0.001]:
		for kernal1 in ['linear','rbf', 'sigmoid']:
			classifier_list.append(svm.SVC(C=C1, gamma=gamma1, kernel=kernal1))

#add MLP to the classifier list with differnt parameters
'''for solver1 in ['lbfgs', 'sgd', 'adam']:
	for activation1 in ['logistic', 'relu', 'tanh']:
		for hidden_layer_sizes1 in [(50),(100),(50,50),(100,100),(1000)]:
			classifier_list.append(MLPClassifier(early_stopping=True, random_state=1, activation=activation1, hidden_layer_sizes=hidden_layer_sizes1,solver=solver1))
			
'''
log_file.write("Time to create the list of classifires is:\n")
log_file.write(str(time.time() - start_time) + "\n")
log_file.write("#################################################\n")

###+++++++++working on the classifiers+++++++#####

start_time = time.time()

print("processing classifiers ...")
class_file = open("seq_class.txt", "a")
for i in range(len(classifier_list)):
	print(str(i) + "/" + str(len(classifier_list)), end = "\r")
	classifier_list[i].fit(trainX_Z_score,trainY)
	predY = classifier_list[i].predict(testX_Z_score)
	accuracy=accuracy_score(testY, predY)
	
	class_file.write("#########################\n")
	class_file.write(str(classifier_list[i]) + "\n")
	class_file.write("Accurecy --> " + str(accuracy) + "\n")

class_file.close()

log_file.write("#################################################\n")		
log_file.write("Time to calculate the accurecy of classifires is:\n")
log_file.write(str(time.time() - start_time) + "\n")

log_file.write("\n")
log_file.close()

