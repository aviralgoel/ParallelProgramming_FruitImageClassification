# A Parallel Image (Fruit) Classification System using MPI4PY for very large dataset 
 A Fruit Image Classification System which uses various ML classifiers and fits them parallely on **High Performance System**.
 
 *Authors: Nasser Al Jabri, Aviral Goel*
 
 # Overview
 
 This image (fruit) classification system uses MPI4PY to distribute the image feature extraction work from master to slave workers.
 The master gathers the extracted features, scales them and distribute the work of classification (model fitting) to workers as well, where each worker gets a different instance of model to fit.
 
 The system is designed to deal with a very very large dataset and has been tested on Luban - High Performance Computer System (Supercomputer) at the Sultan Qaboos University. 

![Overview](https://i.imgur.com/JEUZO34.png)
![Communication Design](https://i.imgur.com/MISLVRR.png)

# Models used
* KNN
* Gausian 
* SVM

# Parallelization Results

![Achieved Speedup](https://i.imgur.com/Uk5370U.png)
![Achieved Efficiency](https://i.imgur.com/e2igAIL.png)

# Summary
In HPC - Luban, we notice that the speedup is keep increasing until it reach a limit where there will be no more speedup since that time is used for the communication for n images which is same for all number of processes since we will send the images one by one to the workers.

## Luban High Performance Computation Facility – Sultan Qaboos University
The authors acknowledge the Sultan Qaboos University HPC Luban supercomputing resources (http://squ.edu.om) made available for conducting the research reported in this report.



*Please find attached a report of the complete project in the files of the repository for further information about the work.*
