import setuptools
import os
import sys

def get_version():
  version_path = os.path.join(os.path.dirname(__file__), 'molexpress')
  sys.path.insert(0, version_path)
  from _version import __version__ as version
  return version

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
    "tensorflow>=2.16.1", # Installs Keras 3
    "rdkit>=2023.9.5",
    "jupyter", # Optional, but needed for the notebooks
]

setuptools.setup(
    name='molexpress',
    version=get_version(),
    author="Alexander Kensert",
    author_email="alexander.kensert@gmail.com",
    description="Graph Neural Networks with Keras 3.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/compomics/molexpress",
    packages=setuptools.find_packages(include=["molexpress*"]),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.10.6",
    keywords=[
        'python',
        'keras-3',
        'machine-learning',
        'deep-learning',
        'graph-neural-networks',
        'graph-convolutional-networks',
        'graphs',
        'molecules',
        'chemistry',
        'cheminformatics',
        'bioinformatics',
    ]
)
