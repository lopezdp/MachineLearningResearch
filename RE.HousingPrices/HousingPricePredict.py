import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

# Load Data Set
california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")


# Randomize Data
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))

# median_house_value to unit of thousands
california_housing_dataframe["median_house_value"] /= 1000.0

# Examine the Data
california_housing_dataframe.describe()

# Build the model
# Define the input feature: total_rooms. pull data from
# cali_housing_df & define feature column as numeric
my_feature = california_housing_dataframe[["total_rooms"]]

# Configure a numeric feature column for total_rooms.
feature_columns = [tf.feature_column.numeric_column("total_rooms")]

# Define the label, the target to train for future predictions.
targets = california_housing_dataframe["median_house_value"]

california_housing_dataframe[["total_rooms"]]

# Configure the Linear Regressor
# Train model using GradientDescentOptimizer & implement
# Mini-Batch Stochastic Gradient Descent. The LearningRate 
# argument controls the size of the gradient step.
# Gradient Clipping also applied to optimizer with 
# clip_gradients_by_norm to ensure the magnitude of the
# gradients do not become too large during training, which
# can cause gradient descent to fail

# Use gradient descent as the optimizer for training the model.
my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

# Configure the linear regression model with our feature columns and optimizer.
# Set a learning rate of 0.0000001 for Gradient Descent.
linear_regressor = tf.estimator.LinearRegressor(
    feature_columns = feature_columns,
    optimizer = my_optimizer
)

# Define the input function
# Need to instruct TF how to preprocess data, batch, shuffle
# and repeat for model training

# First convert pandas feature df into a dict of numpy arrays
# Use TF Dataset API to construct dataset object from data,
# to then break data into batches of batch_size, to be repeated
# for a specific number of epochs (num_epochs).
# When default value of num_epochs = None is passed into
# repeat(), the input dataa will repeat indefinitely
# IF shuffle is set to True, shuffle data and pass to 
# model randomly during training. The buffer_size argument
# specifies size of dataset that shuffle will randomly sample.
# Input function must construct an iterator for dataset and returns 
# next batch of data to linear regressor.

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
  
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
 
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(buffer_size=10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


# Train the model
# can now call train() on our linear_regressor to train the model. 
# wrap my_input_fn in a lambda to pass in my_feature and target as 
# arguments, and to start, we'll train for 100 steps.

_ = linear_regressor.train(
    input_fn = lambda:my_input_fn(my_feature, targets),
    steps=100
)






