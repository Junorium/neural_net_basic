import numpy as np
import pandas as pd

weight_A = np.array(
  [1,2,3],
  [4,5,6],
  [7,8,9]
)

input_A = np.array(
  [11,12,13]
)

bias_A = np.array(
  [14,15,16]
)

def sigmoid(x, W, b):
  '''
  weights W: n x n matrix
  input x: n size vector
  bias b: n size vector
  '''
  z = np.dot(W, x) + b
  return 1 / (1 + np.exp(-z))

sigmoid(input_A, weight_A, bias_A)
