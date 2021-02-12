#!/usr/bin/env python3
import argparse
import itertools
import numpy as np

from enum import Enum

# The 4 bases for mRNA.
Base = Enum('Base', ['A', 'C', 'G', 'U'])

# Creates 64 codons.
Codon = Enum('Codon', [
  ''.join(base.name for base in codon)
  for idx, codon in enumerate(itertools.product(Base, repeat=3))
])

# Creates 20 amino acids + a stop.
AminoAcid = Enum('AminoAcid', [
  'A', 'C', 'D', 'E', 'F', 'G', 'H',
  'I', 'K', 'L', 'M', 'N', 'P', 'Q',
  'R', 'S', 'T', 'V', 'W', 'Y', 'Stop',
])


def I(Q, W, V):
  '''
  Compute the mutual information between Q and W:

  I(Q, W) = \sum_x\sum_y Q(x)W(y|x)log(V(x|y)/Q(x))
          = \sum_x Q(x)\sum_y W(y|x)log(V(x|y)) - \sum_x Q(x)log(Q(x))
          = I1 - I2
  where:
    * I1 = diag(W.T @ log V.T) @ Q
    * I2 = Q @ log Q

  Ip = sum(
    Q[i]*W[j,i]*np.log(V[i,j]/Q[i])
    for i in range(x)
    for j in range(y)
  )
  '''
  # x = |X| and y = |Y|.
  x, y = len(Codon), len(AminoAcid)

  assert Q.shape == (x,)
  assert W.shape == (y, x)
  assert V.shape == (x, y)

  # Check distribution properties.
  assert np.isclose(Q.sum(), 1.)
  assert np.isclose(W.sum(axis=0), np.ones(x))
  assert np.isclose(V.sum(axis=0), np.ones(y))

  I1 = (W.T * np.log(V)).sum(axis=1) @ Q
  I2 = Q @ np.log(Q)
  mI = I1 - I2

  return I1 + I2


def blahut_arimoto(R, W, V, iterations=10):
  '''
  Computes the channel capacity C(W)= max_Q I(Q, W),
  where the mutual information is given by:

  I(Q,W) = I(Q) = \sum_x \sum_y Q(x)W(y|x)log(V(x|y)/Q(x)).

  Q(x)   = P(X=x)     <--- want to find this! in order to maximize I(Q, W).
  R(y)   = P(Y=y) = (QW)(y) = \sum_x W(y|x)Q(x)
  W(y|x) = P(Y=y|X=x)
  V(x|y) = P(X=x|Y=y)

  returns Q after some iterations.
  '''
  # x = |X| and y = |Y|.
  x, y = len(Codon), len(AminoAcid)

  assert R.shape == (y,)
  assert W.shape == (y, x)
  assert V.shape == (x, y)

  # Check distribution properties.
  assert np.isclose(R.sum(), 1.)
  assert np.isclose(W.sum(axis=0), np.ones(x))
  assert np.isclose(V.sum(axis=0), np.ones(y))

  # Initialize Q as the uniform distribution.
  Q = np.ones(x) / x

  for r in range(iterations):
    '''
    Want to compute T(x) = \sum_y W(y|x)log(Q(x)W(y|x)/R(y)).
    Break this up into T(x) = T1(x) + T2(x) - T3(x) where:
      * T1(x) = \sum_y W(y|x)log(Q(x)) = log(Q(x))\sum_y W(y|x) = log(Q(x)),
      * T2(x) = \sum_y W(y|x)log(W(y|x)) (i.e entries on the diagonal of W.T@W),
      * T3(x) = \sum_y W(y|x)log(R(y)).

    Tx = np.array([
      sum(
        W[j,i]*np.log(Q[i]*W[j,i]/sum(W[j,ip]*Q[ip] for ip in range(x)))
        for j in range(y)
      ) for i in range(x)
    ])
    '''
    T1 = np.log(Q)
    T2 = (W*np.log(W)).sum(axis=0)
    R = W@Q
    T3 = W.T@np.log(R)
    T = T1 + T2 - T3

    # Compute new probability distribution.
    eT = np.exp(T)
    Q = eT/eT.sum()

    diff = T-np.log(Q)
    m, M = min(diff), max(diff)
    print(r, m, M, diff)

  return Q






