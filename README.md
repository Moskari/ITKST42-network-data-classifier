# ITKST42-network-data-classifier

## A network data classifier for UNSW-NB15 data set. This is an university course work for "ITKST42 Information Security Technology". 

UNSW-NB15 is a network traffic data set with different categories for normal activities and synthetic attack behaviours.

Find the data set here: https://www.unsw.adfa.edu.au/australian-centre-for-cyber-security/cybersecurity/ADFA-NB15-Datasets/

This project includes a classification model for UNSW-NB15 data set, which I developed using a random forest and feed-forward neural network. The system uses the random forest that classifies data to normal or malicious data. This information is then used to train a neural network to further classify the attack data to different attack categories.

The results for attack detection were very good with approx. 0.88 precision for attacks and nearly 1.0 precision for normal data samples. Attack categorization had problems in differentiating between attack classes and could mostly classify the attacks to two different classes. However, it could accurately classify normal network data. See the report for more details.

### Requirements
Tested with Python 3.6.

[UNSW-NB15](https://www.unsw.adfa.edu.au/australian-centre-for-cyber-security/cybersecurity/ADFA-NB15-Datasets/) dataset.

Dependencies: Keras (earlier than Keras 2), Theano/Tensorflow, Numpy, Pandas, h5py, matplotlib

At least 8 GB RAM and plenty of HDD space.

### How to run

Run the scripts in this order:

-  	h2_preprocess.py
-  	h2_create_data_sets.py
-  	h2_fit_neural.py
