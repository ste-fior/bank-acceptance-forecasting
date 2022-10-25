# Bank Term Deposit Acceptance forecasting

The project aims to predict if a customer intercepted by a bank marketing campaign decides whether or not to take out a bank term deposit.

Each customer is desbribed by 16 features, which can be numerical, ordinal categorical or categorical.

- 8 features are personal data;
- 4 features are related to the last concact happened during this campaign;
- 4 features are related to the previous campaigns.

# Installing

The required libraries for the project are collected in ```requirements.txt``` file.

# Project structure

The project is implemented in the notebook file ```Fastweb-Data-Science-Assignment.ipynb```. The notebook is divided in sections, here briefly explained.

## Data Loader

Methods to load, clean and resample the dataset. The file ```data/features.yaml``` collects the featrues specification, such as the type, the classes for the categorial features and the max value to drop the outliers.

The method ```midsampling()``` munipulate the dataset in order to rebalance the classes. The majotiry classes are downsampled, while the minority classes are upsampled, trying to avoid the cons of both the cummon resampling methods, emphasized with highly unbalance datasets.

Downsampling leads to small dataset, losing most of the samples of the majority class. Upsampling leads to the overfitting of the minority class, due to the many clones created.

The datset has been analysed with the [pandas profiling tool](https://pandas-profiling.ydata.ai/docs/master/index.html) and the data distribution plotted with seaborn library, the results are in ```/data```.

![Image](/data/report.jpg)

## Model

The classification is done in three steps, executed by three different methods: preprocessing, classification and postprocessing.

Preprocessing encodes the input into numerical form, postprocess decode the output score into the predicted class.

The classifier is an istance of [XGBClassifier](https://xgboost.readthedocs.io/en/stable/index.html) and compute a score for each input data.

The trained model can be evaluated on a validation set with Accuracy, Precicion, Recall, F1-score, AUC metrics. Moreover ROC curve and confusion matrix can be plotted.

## Pipelines

To automatize the processes, the following pipelines have been implemented:

- **Loading train and test dataset**: automatically loads the dataset, cleans it and then splits it into train/test with a test size on 20%

- **Full training**: given the configuration dict, load the dataset, split into train and test, train the model and evaluate it, returning the trained model and the computed metrics

- **Cross validation**: given the configuration dict and the train dataset, run K different training process, each one evaluated on a different portion of the given dataset. The return is the same kind of the train pipeline, but with list of K elements

- **Inference**: given the trained model (or the exported path) and the input sample, it computes the output score. If a threshold is given, the predicted class is returned.

## Hyper-parameter optimization

To maximize the performance, the model's hyper-parameters must be tuned on the given dataset. 

Thanks to the [HyperOpt](http://hyperopt.github.io/hyperopt/) library, the user can define the configuration space, with the parameters for the model construction and an objective function to minimize, in this project set as the negative AUC (to obtain a miximization behave).

The algorithm tries multiple proposals, each one scored through the cross validation. The best configuration is returned and train pipeline is executed with it, trained on the whole train set and evaluated on the test set.

## Inference

The last cell is left to infer the trained model on custom inputs.