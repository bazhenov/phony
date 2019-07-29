import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Conv1D, Flatten, Dropout, Input,\
        LSTM, Bidirectional, Concatenate
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks
from keras_contrib.layers import CRF
from keras_contrib.metrics import crf_viterbi_accuracy
from keras_contrib.losses import crf_loss
import numpy as np
import json
from optparse import OptionParser
import array
import fileinput

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

  y_onehot = np.zeros((l, 2))
  y_onehot[np.arange(y.size), y.astype('int')] = 1
  
  return (x, y_onehot)

def build_model():
  inputs = Input(shape=(64,), name="input")

  emb = x = Embedding(256, 4, input_length=64, name="embedding")(inputs)

  x = Conv1D(32, 13, padding='same', activation='relu', name="conv1d-1")(x)
  x = Dropout(0.1, name="dropout-1")(x)
  aux = Concatenate(name="aux")([x, emb])
  x = Conv1D(64, 13, padding='same', activation='relu', name="conv1d-2")(aux)
  x = Dropout(0.1, name="dropout-2")(x)

  x = Dense(2, activation='sigmoid', name="final-dense")(x)
  #prediction = Flatten(name = 'output')(x)
  prediction = CRF(2)(x)
  #prediction = x

  model = Model(inputs=inputs, outputs = prediction)
  model.compile(loss=crf_loss, optimizer='adam', metrics=[crf_viterbi_accuracy])
  #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

if __name__ == "__main__":
  parser = OptionParser()
  parser.add_option("-f", "--file", dest="filename", help="Input file for learning")
  parser.add_option("-o", "--out", dest="model", default="./model", help="Filename of the output model")
  parser.add_option("-e", "--epochs", dest="epochs", type="int", default=10, help="Number of epochs")
  parser.add_option("-v", "--validation", dest="validation", type="float", default=0.1, help="Validation split ratio")

  (options, args) = parser.parse_args()

  inp = file(options.filename) if options.filename != None else stdin()

  print("Building model...")
  model = build_model()
  model.summary()

  print("Reading samples...")
  X = []
  Y = []
  for sample in read_json(inp):
    (x, y) = json2vec(sample)
    X.append(x)
    Y.append(y)

  model.fit(np.array(X), np.array(Y), epochs=options.epochs, batch_size=64, validation_split=options.validation,
          shuffle=True)

  print("Saving mode to: " + options.model)
  #tf.keras.experimental.export_saved_model(model, options.model + "/exp")

  tf.contrib.saved_model.save_keras_model(model, options.model)
