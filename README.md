# ggcnn_pytorch
This repo is Pytorch implementation for the paper:
**Closing the Loop for Robotic Grasping: A Real-time, Generative Grasp Synthesis Approach**([arxiV](https://arxiv.org/abs/1804.05172))

The original code is implemented with Keras. [here](https://github.com/dougsm/ggcnn)

## Requirements
This code was tested with Python 3.6, Pytorch 1.0.0. Other required python packages can be found at the original repo. Of course Tensorflow and Keras is not neccessary. 

## Prepare dataset
The original code `generate_dataset.py` requires a large RAM to generate the dataset (mine is 32GB but still not enough). Memory error will occur due to lack of RAM. `generate_dataset.py` in this code is modified at the part of writing h5 file so that less RAM is required.

## Start training
1. Modify `DATA_PATH` `MODEL_PATH` `BATCH_SZIE` `NUM_EPOCHS` in `train.py` according to your need.
2. Run `python train.py` and the training will start.

## Evaluate
1. Modify `DATA_PATH` `MODEL_PATH` `BATCH_SZIE` `NO_GRASPS` in `evaluate.py`.
2. Run `evaluate.py` to get the accuracy of trained models.

## Visualization
Change the value of `VISUALISE_FAILURES` and `VISUALIZE_SUCCESSES` in `evaluate.py` to get visualization results.


