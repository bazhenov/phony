import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import metrics
from tensorflow.keras import callbacks
import numpy as np
import json
from optparse import OptionParser
import array
import fileinput
import h5py

def build_model():
  inputs = layers.Input(shape=(31,), name="input", dtype=tf.uint32)

  query_embedding = layers.Embedding(256, 4, input_length=31, name="query_embedding")(inputs)

  cnn = layers.Conv1D(
    filters=20,
    kernel_size=11,
    padding='same', activation="relu")

  x = cnn(query_embedding)
  x = layers.Flatten()(x)

  x = layers.Dropout(0.1)(x)
  x = layers.Dense(20, activation='relu')(x)
  x = layers.Dropout(0.1)(x)
  x = layers.Dense(1, name="output", activation='sigmoid')(x)

  model = keras.Model(inputs=inputs, outputs=x)
  choosen_metrics = [metrics.BinaryAccuracy(), metrics.Precision(), metrics.Recall()]
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=choosen_metrics)
  return model

def main(dataset_generator):
  parser = OptionParser()
  parser.add_option("-f", "--file", dest="filename", help="Input file for learning")
  parser.add_option("-o", "--out", dest="model", default="./model", help="Filename of the output model")
  parser.add_option("-e", "--epochs", dest="epochs", type="int", default=10, help="Number of epochs")
  parser.add_option("-v", "--validation", dest="validation", type="float", default=0.1, help="Validation split ratio")

  (options, args) = parser.parse_args()
  
  print("Building model...")

  model = build_model()
  model.summary()

  print("Reading samples...")
  X = []
  Y = []

  i = 0  
  for (input, output) in dataset_generator():
    X.append([input])
    Y.append([output])
    i += 1
  print("%d samples read" % i)

  X = np.vstack(X)
  Y = np.vstack(Y)
  print("X shape = %s, Y shape = %s" % (str(X.shape), str(Y.shape)))

  tensorboard_callback = callbacks.TensorBoard(log_dir='./tensorboard', histogram_freq=0,
          write_graph=True, write_grads=False, write_images=False, embeddings_freq=0,
          embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

  model.fit(np.array(X), np.array(Y), epochs=options.epochs, batch_size=64, validation_split=options.validation,
          shuffle=True, callbacks=[tensorboard_callback])

  print("Saving mode to: " + options.model)
  model.save(options.model, include_optimizer=False)

if __name__ == "__main__":
  main()