try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

def readme():
    with open('README.md') as f:
        return f.read()

description = 'Toolbox for anomaly detection.'

config = {
    'name': 'anomatools',
    'description': description,
    'long_description': readme(),
    'url': 'https://github.com/Vincent-Vercruyssen/anomatools',
    'author': 'Vincent Vercruyssen',
    'author_email': 'V.Vercruyssen@gmail.com',
    'version': '2.0',
    'install_requires': ['numpy',
                         'scipy',
                         'matplotlib',
                         'scikit-learn'],
    'packages': find_packages(),
    'package_dir' : {'anomatools': 'anomatools'},
    'keywords': 'anomaly detection',
    'include_package_data': True,
    'classifiers':['Intended Audience :: Science/Research',
                   'License :: OSI Approved :: Apache Software License',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence',
                   'Programming Language :: Python :: 3']
}

setup(**config)
