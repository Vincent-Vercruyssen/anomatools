# anomatools

`anomatools` is a small Python package containing recent **anomaly detection algorithms**.
Anomaly detection strives to detect *abnormal* or *anomalous* data points from a given (large) dataset.
The package contains two state-of-the-art (2018 and 2020) semi-supervised and two unsupervised anomaly detection algorithms.


## Installation

Install the package directly from PyPi with the following command:
```bash
pip install anomatools
```
OR install the package using the `setup.py` file:
```bash
python setup.py install
```
OR install it directly from GitHub itself:
```bash
pip install git+https://github.com/Vincent-Vercruyssen/anomatools.git@master
```


## Contents and usage

### Semi-supervised anomaly detection

Given a dataset with attributes **X** and labels *Y*, indicating whether a data point is *normal* or *anomalous*, semi-supervised anomaly detection algorithms are trained using all the instances **X** and some of the labels *Y*.
Semi-supervised approaches to anomaly detection generally outperform the unsupervised approaches, because they can use the label information to correct the assumptions on which the unsupervised detection process is based.
The `anomatools` package implements two recent semi-supervised anomaly detection algorithms:
1. The **SSDO** (*semi-supervised detection of outliers*) algorithm first computes an unsupervised prior anomaly score and then corrects this score with the known label information [1].
2. The **SSkNNO** (*semi-supervised k-nearest neighbor anomaly detection*) algorithm is a combination of the well-known *kNN* classifier and the *kNNO* (k-nearest neighbor outlier detection) method [2].

Given a training dataset **X_train** with labels *Y_train*, and a test dataset **X_test**, the algorithms are applied as follows:
```python
from anomatools.models import SSkNNO, SSDO

# train
detector = SSDO()
detector.fit(X_train, Y_train)

# predict
labels = detector.predict(X_test)
```

Similarly, the probability of each point in **X_test** being normal or anomalous can also be computed:
```python
probabilities = detector.predict_proba(X_test, method='squash')
```

Sometimes we are interested in detecting anomalies in the training data (e.g., when we are doing a post-mortem analysis):
```python
# train
detector = SSDO()
detector.fit(X_train, Y_train)

# predict
labels = detector.labels_

```

### Unsupervised anomaly detection:

Unsupervised anomaly detectors do not make use of label information (user feedback) when detecting anomalies in a dataset. Given a dataset with attributes **X** and labels *Y*, the unsupervised detectors are trained using only **X**.
The `anomatools` package implements two recent semi-supervised anomaly detection algorithms:
1. The **kNNO** (*k-nearest neighbor outlier detection*) algorithm computes for each data point the anomaly score as the distance to its k-nearest neighbor in the dataset [[3](https://dl.acm.org/citation.cfm?id=335437)].
2. The **iNNE** (*isolation nearest neighbor ensembles*) algorithm computes for each data point the anomaly score roughly based on how isolation the point is from the rest of the data [[4](https://onlinelibrary.wiley.com/doi/full/10.1111/coin.12156)].

Given a training dataset **X_train** with labels *Y_train*, and a test dataset **X_test**, the algorithms are applied as follows:
```python
from anomatools.models import kNNO, iNNE

# train
detector = kNNO()
detector.fit(X_train, Y_train)

# predict
labels = detector.predict(X_test)
```

## Package structure

The anomaly detection algorithms are located in: `anomatools/models/`

For further examples of how to use the algorithms see the notebooks: `anomatools/notebooks/`


## Dependencies

The `anomatools` package requires the following python packages to be installed:
- [Python 3](http://www.python.org)
- [Numpy](http://www.numpy.org)
- [Scipy](http://www.scipy.org)
- [Scikit-learn](https://scikit-learn.org/stable/)


## Contact

Contact the author of the package: [vincent.vercruyssen@kuleuven.be](mailto:vincent.vercruyssen@kuleuven.be)


## References

[1] Vercruyssen, V., Meert, W., Verbruggen, G., Maes, K., Bäumer, R., Davis, J. (2018) *Semi-Supervised Anomaly Detection with an Application to Water Analytics.* IEEE International Conference on Data Mining (ICDM), Singapore. p527--536.

[2] Vercruyssen, V., Meert, W., Davis, J. (2020) *Transfer Learning for Anomaly Detection through Localized and Unsupervised Instance Selection.* AAAI Conference on Artificial Intelligence, New York.
