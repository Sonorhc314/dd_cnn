from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
#from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
import tensorflow as tf
from get_data import *


map_array = np.array(map_array)
processed_density = np.array(processed_density)

X_train = map_array[:610]
X_test = map_array[610:633]

y_train = processed_density[:610]
y_test = processed_density[610:633]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

latent_dim = 70