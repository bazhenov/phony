import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import json
from optparse import OptionParser
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Conv1D
from keras.layers import Flatten
from keras.layers import Dropout
import array
import fileinput

def stdin():
  for line in fileinput.input():
    yield line.strip()

def file(filename):
  for line in fileinput.input(filename):
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

if __name__ == "__main__":
  parser = OptionParser()
  parser.add_option("-f", "--file", dest="filename", help="Input file for learning")

  (options, args) = parser.parse_args()

  inp = file(options.filename) if options.filename != None else stdin()
  print("Reading samples...")
  X = []
  Y = []
  for sample in read_json(inp):
    for i in range(100):
      (x, y) = json2vec(sample)
      X.append(x)
      Y.append(y)
      
  print("Examples: " + str(len(X)))

  model = Sequential()
  model.add(Embedding(256, 2, input_length=64, name="embedding"))

  model.add(Conv1D(32, 5, padding='same', activation='relu', name="conv1d-1"))
  model.add(Dropout(0.1, name="dropout-1"))
  model.add(Conv1D(32, 9, padding='same', activation='relu', name="conv1d-2"))
  model.add(Dropout(0.1, name="dropout-2"))
  model.add(Conv1D(32, 13, padding='same', activation='relu', name="conv1d-3"))
  model.add(Dropout(0.1, name="dropout-3"))

  model.add(Dense(1, activation='relu', name="final-dense"))
  model.add(Flatten())

  model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])

  model.fit(np.array(X), np.array(Y), epochs=1, batch_size=64, validation_split=0.1, shuffle=True)