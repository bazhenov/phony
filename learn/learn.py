import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import metrics

from tensorflow.keras import backend as K
from tensorflow.keras import callbacks
import numpy as np
import json
from optparse import OptionParser
import array
import fileinput
import h5py

def stdin():
  for line in fileinput.input():
    yield line.strip()

def file(filename):
  for line in fileinput.input(filename, openhook=fileinput.hook_encoded("utf-8")):
    yield line.strip()

def read_json(x):
  for i in x:
    yield json.loads(i)

def onehot(idx, l=3):
  result = [0] * 3
  result[idx] = 1
  return result

def sample2vec(text, phone, l=64):
  x = [i for i in text.encode('cp1251')]
  x = x + [0] * (l - len(x))
  idx = text.index(phone)
  if idx < 0:
    raise ValueError
  y = ([0] * idx) + ([2]) + ([1] * (len(phone) - 1)) + ([0] * (l - idx - len(phone)))
  return (x, [onehot(i, l=3) for i in y])

def json2vec(sample, l=64, start=None):
  text = sample['message']
  
  chars = [i for i in text.encode('cp1251')]
  y = np.zeros(len(chars))
  for idx in sample['phone_indexes']:
    y[idx[0]:idx[1]] = 1
  
  if start == None:
    start = np.random.randint(0, len(text) - 1)
  end = start + l
  
  x = chars[start:end]
  y = y[start:end]
  
  padding = l - len(x)
  if padding > 0:
    x = x + ([0] * padding)
    y = np.pad(y, [(0, padding)], mode='constant')
  
  return (x, y)

def build_model():
  inputs = layers.Input(shape=(64,), name="input")

  emb = x = layers.Embedding(256, 4, input_length=64, name="embedding")(inputs)

  x = layers.Bidirectional(layers.LSTM(10, return_sequences=True))(x)
  x = layers.Dense(1, activation='sigmoid', name="final-dense")(x)
  predictions = layers.Flatten(name = 'output')(x)

  model = keras.Model(inputs=inputs, outputs=predictions)
  choosen_metrics = [metrics.Recall(), metrics.Precision()]
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=choosen_metrics)
  return model

if __name__ == "__main__":
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
  h5 = h5py.File(options.filename, "r")
  X = []
  Y = []
  i = 0
  while True:
    input_key = "input/" + str(i)
    output_key = "output/" + str(i)

    if input_key not in h5 or output_key not in h5:
      break
    
    X.append(h5[input_key])
    Y.append(h5[output_key])
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

