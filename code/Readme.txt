This folder contain 3 python files
1- image_feature2.py --> contain function of processing the images to calculate the features
2-read_img_mpi.py --> the parallel implementation
3-read_img_seq.py --> the sequential implementation

Also, it include a sample of dataset used in this expirement. full dataset can be downloaded from https://www.kaggle.com/moltean/fruits

to run the sequential implementation use this command:
python3 read_img_seq.py

to run the parallel implementation:
mpiexec -n numprocs python3 read_img_mpi.py