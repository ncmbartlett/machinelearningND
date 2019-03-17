# Udacity MLND Capstone: Comparison of sentiment analysis using traditional machine learning methods and recurrent neural networks

### Brief description

The accuracy of two benchmark models (Naive Bayes and Support Vector Machines) is compared to that of LSTM-based recurrent neural networks for the task of sentiment analysis using the Large Movie Review Dataset.

### Prerequisites

- python 3
- numpy
- nltk
- matplotlib
- seaborn
- pandas
- keras
- tensorflow
- scikit-learn

### Installation

Place the capstone.ipynb file in its own directory containing the following subfolders: 
- saved_models
- glove.6B

Please download and unzip the contents of the file glove.6B.zip (~2 GB) from <a href="https://nlp.stanford.edu/projects/glove/">https://nlp.stanford.edu/projects/glove/</a> into the glove.6B subfolder. <a href="http://nlp.stanford.edu/data/glove.6B.zip">Direct link</a>. 

The saved_models subfolder contains the weights of the various RNN models. Note that the weights for model 27 are not included due to the 500 MB upload limit for the project. The weights can be downloaded here: https://drive.google.com/open?id=1UuGBnTbXLiHmmchJI53GR6HgmldJuA0p

If you do not have the nltk "punkt" package installed, please uncomment the code in the second cell of the capstone.ipynb jupyter notebook and run it to install the package.

The LMRD dataset is built into Keras and will be downloaded in the capstone.ipynb file.

## Authors

* **Nate C.-M Bartlett**