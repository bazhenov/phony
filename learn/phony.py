import learn
import numpy as np
import json

def sample2vec(text, phone_idx):
  chars = [i for i in text.encode('cp1251')]
  y = [0] * len(text)
  for (start, end) in phone_idx:
    for i in range(start, end):
      y[i] = 1
  
  return (chars, y)

def vec2tensor(vec, l=65, idx=0):
  assert l % 2 == 1
  (x, y) = vec
  assert len(x) == len(y)
  X = [0] * l

  half_l = int((l - 1) / 2)

  seq_start = max(0, idx - half_l)
  seq_end = min(len(x), idx + half_l + 1)

  padding_left = max(0, half_l - idx)

  X = [0] * padding_left + x[seq_start:idx] + [x[idx]] + x[idx+1:seq_end]
  Y = [0] * padding_left + y[seq_start:idx] + [y[idx]] + y[idx+1:seq_end]

  if len(X) - l:
    padding_right = l - len(X)
    X += [0] * padding_right
    Y += [0] * padding_right

  
  answer = Y[half_l]
  Y[half_l] = 0
  return (X, answer)

def learn_samples():
  for (input, output) in samples():
      for i in range(len(input)):
        yield vec2tensor((input, output), idx=i, l=31)

def samples():
  with open('./input') as f:
    for line in f:
      sample = json.loads(line)
      yield sample2vec(sample['sample'], sample['label'])

def evaluate(model, text, spans):
  (x, y) = sample2vec(text, spans)
  print(text)
  for i in range(len(text)):
    (X, _) = vec2tensor((x, y), l=31, idx=i)
    
    Y = model.predict([X])
    if Y[0][0] > 0.5:
      print("^", end='')
    else:
      print(" ", end='')
  print()

if __name__ == "__main__":
  learn.main(learn_samples)