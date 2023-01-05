# How to Install a Fresh Instance

1- Install Python 3.8.10

2- (Optional) Create a virtual environment to install all dependencies inside it.

3- (Optional) If virtual environment created, activate it.

4- Navigate to the project root folder.

5- Install the required dependencies by running: `pip3 install -r requirements.txt'

6- Calculate the model similarity scores by running the following command: `python DNNCloneFinder.py MnistModels ArtifactDatasets random_data_mnist 2000`. This command calculate the similarity scores for each pair of MNIST shaped models inside the `MnistModels` folder by generating 2000 random inputs using the Spearman correlation coefficient. The generated random data is store under the name `random_data_mnist.csv` inside a folder named `ArtifactDatasets` (this folder is created if not already present).

6- Results will be in `correlation_results/results.txt`

# Parameters to DNNCloneFinder.py Python Script

1- First parameter (specified as `MnistModels` above) refers to the folder containing the DNN models which should be in `h5` format.

2- Second parameter (specified as `ArtifactDatasets` above) refers to the folder where the generated random datasets will be stored (will be created if not present).

3- Third parameter (specified as `random_data_mnist` above) refers to the file name under which the csv file containing the randomly dataset will be saved. If a file under this name is already present, the contents of that file will be used and therefore, no new file will be created.

4- Fourth parameter (specified as `2000` above), which is optional, refers to the number of random inputs the user wants to generate. If not provided, this parameter, will be set to a default value of 20,000.
