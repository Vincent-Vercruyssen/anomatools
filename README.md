# anomatools

## What is anomatools?

The `anomatools` package is a collection of **anomaly detection tools**.
Anomaly detection strives to detect *abnormal* or *anomalous* data points from a given (large) dataset.
The package contains three anomaly detection algorithms and a clustering algorithm.


## Installation

1. Install the package directly from PyPi with the following command:
```bash
pip install anomatools
```
2. OR install the package using the `setup.py` file:
```bash
python setup.py install
```
3. OR install it directly from GitHub itself:
```bash
pip install git+https://github.com/Vincent-Vercruyssen/anomatools.git@master
```


## Contents and usage

### Unsupervised anomaly detection:

Unsupervised anomaly detectors do not make use of label information (user feedback) when detecting anomalies in a dataset. Given a dataset with attributes **X** and target *Y*, indicating whether a data point is normal or an anomaly, the unsupervised detectors only use **X** to compute an anomaly score for each data point in the dataset.
The `anomatools` package includes two unsupervised anomaly detection algorithms that can be initiated as follows:
```python
import anomatools
detector = anomatools.KNNO()
detector = anomatools.INNE()

# compute the anomaly scores:
scores = detector.fit_predict(X)
```
**KNNO** (k-nearest neighbor outlier detection) computes for each data point the anomaly score as the distance to its k-nearest neighbor in the dataset [[1](https://dl.acm.org/citation.cfm?id=335437)].
**INNE** (isolation nearest neighbor ensembles) computes for each data point the anomaly score roughly based on how isolation the point is from the rest of the data [[2](https://onlinelibrary.wiley.com/doi/full/10.1111/coin.12156)].


### Semi-supervised anomaly detection:

Unsupervised approaches are employed when label information is unavailable, a common condition in anomaly detection due to labels being expensive. However, they operate on some assumption about normal behavior to identify anomalies (e.g., normals are frequent). These assumptions are shaky and often violated in practice. Therefore, if *some* labels are available (some values of *Y* are known), we can use semi-supervised anomaly detection techniques:
```python
import anomatools
detector = anomatools.SSDO()

# compute the anomaly scores:
scores = detector.fit_predict(X, Y)
```
**SSDO** (semi-supervised detection of outliers) first computes an unsupervised prior anomaly score and then corrects with the known label information [3]. The prior can be computed beforehand using any unsupervised anomaly detection algorithm or using the clustering subroutine of **SSDO**.

### Constrained clustering:

Constrained clustering algorithms cluster a datasets **X** with the help of user-specified constraints. The constraints can be of two types: `must_link` constraints indicate that two data points should be in the same cluster, while `cannot_link` constraints prohibit them from being in the same cluster. The package contains an implementation of the **COPKMeans** algorithm [[4](https://dl.acm.org/citation.cfm?id=655669)]:
```python
import anomatools
detector = anomatools.clustering.COPKMeans()

# compute the anomaly scores:
centers, cluster_labels = detector.fit_predict(X, must_links, cannot_links)
```

## Package structure:

The anomaly detection algorithms are located in: `anomatools/anomaly_detection/*`

The clustering algorithms are located in: `anomatools/clustering/*`

For further examples of how to use the algorithms see the notebooks: `anomatools/notebooks/*`


## Dependencies

The `anomatools` package requires the following python packages to be installed:
- [Python 3](http://www.python.org)
- [Numpy](http://www.numpy.org)
- [Scipy](http://www.scipy.org)
- [Scikit-learn](https://scikit-learn.org/stable/)


## Contact

For any questions related to the code or the **SSDO** algorithm, contact the author of the package: [vincent.vercruyssen@kuleuven.be](mailto:vincent.vercruyssen@kuleuven.be)


## Citing the original SSDO paper

```
@inproceedings{vercruyssen2018semi,
    title       = {Semi-Supervised Anomaly Detection with an Application to Water Analytics},
    author      = {Vincent Vercruyssen and
                   Wannes Meert and
                   Gust Verbruggen and
                   Koen Maes and
                   Ruben B{\"a}umer and
                   Jesse Davis},
    booktitle   = {{IEEE} International Conference on Data Mining, {ICDM} 2018, Singapore, November 17-20, 2018},
    organization= {IEEE},
    pages       = {527--536},
    year        = {2018},
    doi         = {10.1109/ICDM.2018.00068},
}
```
