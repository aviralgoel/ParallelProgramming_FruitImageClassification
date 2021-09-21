# -*- coding: utf-8 -*-
#use this command to run this parallel code:
# mpiexec -n numprocs python3 read_img_mpi.py

import os
import pandas as pd
import image_feature2
from datetime import datetime
import time
from mpi4py import MPI
import pandas as pd
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_proc = comm.Get_size()


if (rank==0): #in the master
	image_list = [] #List which contain all images info (file name, class num, train/test)
	log_file = open("mpi" + str(num_proc) + ".txt", "a")
	log_file.write("#########################################\n")
	log_file.write("#########++++++++New Test++++++++########\n")
	log_file.write("#########################################\n")
	df_train = pd.DataFrame() #store the training data
	df_test = pd.DataFrame() #store the testing data
	num_image=0

	###+++++++++Creating the list of images+++++++#####

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
	###+++++++++Sending image and getting features from workers+++++++#####

	start_time = time.time()#time stamp
	itr = 0

	#send one image to each process
	for i in range(num_proc-1):
		#Send using MPI. the tag = 10 for send image
		comm.send(image_list[i],dest=i+1,tag=10)
		itr += 1

	#loop to recieve the features from the workers and assign job to them
	for i in range(num_image):
		#get the feature from the worker
		status=MPI.Status()
		features = comm.recv(tag=MPI.ANY_TAG,status=status)

		#construct the dataframe using the recieved features
		col = ["label","ur","ug","ub","stdr","stdg","stdb","skwr","skwg","skwb","kirtr","kirtg","kirtb",
               "u_mf","std_mf","skw_mf","kirt_mf",
               "Ct0","Ct45","Ct90","Ct135","Cn0","Cn45","Cn90","Cn135",
               "Ey0","Ey45","Ey90","Ey135","Hy0","Hy45","Hy90","Hy135"]

		df_item = pd.DataFrame(features,columns = col)

		#check the tag. if it is 11 then this is trainging image. Else it testing
		if(status.Get_tag() == 11):
			df_train = df_train.append(df_item,ignore_index = True,sort = False)
		else:
			df_test = df_test.append(df_item,ignore_index = True,sort = False)

		#print the progress of processing the images
		print(str(itr) + "/" + str(num_image) , end = "\r")

		#if there is image to be processed send it to the worker. Else send termination tag 13
		if(itr<num_image):
			comm.send(image_list[itr],dest=status.Get_source(),tag=10)
			itr += 1
		else:
			comm.send("empty",dest=status.Get_source(),tag=13)

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

	###+++++++++send the Scalled features+++++++#####

	start_time = time.time()
	for i in range(num_proc-1):
		#Send scalled features to all workers. the tag = 14 for send scalled features
		comm.send(scalled_feature,dest=i+1,tag=14)

	#print how much time it take to send the scalled features
	log_file.write("Time to send the scalled features to all workers is:\n")
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


	log_file.write("Time to create the list of classifires is:\n")
	log_file.write(str(time.time() - start_time) + "\n")
	log_file.write("#################################################\n")
	###+++++++++Sending Classifires to the worker+++++++#####

	start_time = time.time()

	itr = 0
	for i in range(num_proc-1):
		if(i<len(classifier_list)):# if there is classifier to be sent
			classifier = [classifier_list[i], i]
			comm.send(classifier,dest=i+1,tag=15)
			itr += 1
		else:#if no classifier send termination to the rest of the process
			comm.send("empty",dest=i+1, tag=17)
	class_file = open("mpi" + str(num_proc) + "class.txt", "a")
	status = MPI.Status()
	for i in range(len(classifier_list)):
		print(str(i) + "/" + str(len(classifier_list)), end = "\r")
		result = comm.recv(tag=16, status=status)

		class_file.write("#########################\n")
		class_file.write("get result from process no.: " + str(status.Get_source()) + "\n")
		class_file.write(str(classifier_list[result[0]]) + "\n")
		class_file.write("Accurecy --> " + str(result[1]) + "\n")

		if(itr<len(classifier_list)):#if still there is a classifire in the list
			classifier = [classifier_list[itr], itr]
			comm.send(classifier,dest=status.Get_source(),tag=15)

			itr+=1
		else:#if no classifier send the termination
			comm.send("empty",dest=status.Get_source(), tag=17)
	class_file.close()
	log_file.write("#################################################\n")
	log_file.write("Time to calculate the accurecy of classifires is:\n")
	log_file.write(str(time.time() - start_time) + "\n")

	log_file.write("\n")
	log_file.close()



####+++++++++++++++++++Workers++++++++++++++++####
else: #in the worker
	#recieve the request from the master
	status = MPI.Status()
	image_item = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

	#keep reciving tasks to process the images until get termination from the master tag=13
	while(status.Get_tag() == 10):
		#process the image
		features = image_feature2.get_image_dataframe(image_item[0],image_item[1])

		#check if it is training image or test to send tag based on that
		if(image_item[2] == 0):
			comm.send(features,dest=0,tag=11)
		else:
			comm.send(features,dest=0,tag=12)

		#recive from the master
		image_item = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

	#Get the scaled features from the master
	scalled_feature = comm.recv(source=0, tag=14)
	trainY = scalled_feature[0]
	testY = scalled_feature[1]
	trainX_Z_score = scalled_feature[2]
	testX_Z_score = scalled_feature[3]

	#get the classifier from the master
	classifier = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

	#keep asking for classifier until get termination
	while(status.Get_tag() == 15):
		classifier[0].fit(trainX_Z_score,trainY)
		predY = classifier[0].predict(testX_Z_score)
		accuracy=accuracy_score(testY, predY)
		result = [classifier[1], accuracy]
		comm.send(result,dest=0,tag=16) #send the result to the classifier
		classifier = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
