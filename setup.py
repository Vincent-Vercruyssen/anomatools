try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

import os

this_directory = os.path.abspath(os.path.dirname(__file__))

# get __version__ from _version.py
ver_file = os.path.join('anomatools', 'version.py')
with open(ver_file) as f:
    exec(f.read())

# read the contents of README.md
def readme():
    with open(os.path.join(this_directory, 'README.md')) as f:
        return f.read()

# read the contents of requirements.txt
with open(os.path.join(this_directory, 'requirements.txt')) as f:
    requirements = f.read().splitlines()

# setup configuration
config = {
    'name': 'anomatools',
    'version': __version__,
    'description':'A compact Python toolbox for anomaly detection.',
    'long_description': readme(),
    'long_description_content_type': 'text/markdown',
    'url': 'https://github.com/Vincent-Vercruyssen/anomatools',
    'author': 'Vincent Vercruyssen',
    'author_email': 'V.Vercruyssen@gmail.com',
    'keywords': [
        'outlier detection',
        'anomaly detection',
        'semi-supervised detection'
    ],
    'install_requires': requirements,
    'packages': find_packages(exclude=['test']),
    'package_dir' : {'anomatools': 'anomatools'},
    'include_package_data': True,
    'include_package_data': True,
    'classifiers':[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3'
    ]
}

setup(**config)
