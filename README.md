# Scalable and Stable Surrogates for Flexible Classifiers with Fairness Constraints

<!---This repository is the official implementation of [My Paper Title](https://arxiv.org/abs/2030.12345). -->

## Requirements

This code runs on Python 3.8 with the following packages: `torch`, `torchvision`, and `tqdm`. Each folder contains instructions for downloading the data for that experiment.

## Training and Evaluation

For the Faces of the World and Yelp data sets, a single file trains and evaluates the model performance. For the CelebA data, there are separate training and evaluation files. The logistic regression models allow for interactive experimenting via Jupyter Notebook.

None of these methods require command-line arguments. Once the data is downloaded, experiments can be run via (e.g.)
```
python sim.py
```