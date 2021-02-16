#!/usr/bin/env python3
import Bio.Data.CodonTable
import argparse
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from enum import Enum

BITS = True
PLOT = True
log = np.log2 if BITS else np.log
exp = (lambda x: 2**x) if BITS else np.exp
p = 1e-4

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


def I(Q, W):
  '''
  Compute the mutual information between Q and W:

  I(Q, W) = \sum_x\sum_y Q(x)W(y|x)log(V(x|y)/Q(x))
          = \sum_x Q(x)\sum_y W(y|x)log(V(x|y)) - \sum_x Q(x)log(Q(x))
          = I1 - I2
  where:
    * I1 = diag(W.T @ log V.T) @ Q
    * I2 = Q @ log Q

  Ip = sum(
    Q[i]*W[j,i]*log(V[i,j]/Q[i])
    for i in range(x)
    for j in range(y)
  )
  '''
  # x = |X| and y = |Y|.
  x, y = len(Codon), len(AminoAcid)

  assert Q.shape == (x,)
  assert W.shape == (y, x)
  assert np.isclose(Q.sum(), 1.)
  assert np.all(np.isclose(W.sum(axis=0), 1.))

  R = W@Q
  V = (W*Q).T/R

  assert R.shape == (y, )
  assert V.shape == (x, y)
  assert np.isclose(R.sum(), 1.)
  assert np.all(np.isclose(V.sum(axis=0), 1.))

  I1 = (W.T * log(V)).sum(axis=1) @ Q
  I2 = Q @ log(Q)
  mI = I1 - I2

  return mI


def blahut_arimoto(W, iterations=10):
  '''
  Computes the channel capacity C(W)= max_Q I(Q, W),
  where the mutual information is given by:

  I(Q,W) = I(Q) = \sum_x \sum_y Q(x)W(y|x)log(V(x|y)/Q(x)).

  Q(x)   = P(X=x)     <--- want to find this! in order to maximize I(Q, W).
  R(y)   = P(Y=y) = (QW)(y) = \sum_x W(y|x)Q(x)
  W(y|x) = P(Y=y|X=x)
  V(x|y) = P(X=x|Y=y)

  returns (Q, its) after some iterations, where its is info from each iteration.
  '''
  # x = |X| and y = |Y|.
  x, y = len(Codon), len(AminoAcid)

  assert W.shape == (y, x)

  # Check distribution properties.
  assert np.all(np.isclose(W.sum(axis=0), 1.))

  # Initialize Q as the uniform distribution.
  pd = np.random.randn(x)
  Q = exp(pd) / np.sum(exp(pd)) # np.ones(x) / x

  print(f'iteration #0: I(Q, W)={I(Q, W)}')

  # List of (mutual info, lower bound, upper bound).
  its = []
  for r in range(iterations):
    '''
    Want to compute T(x) = \sum_y W(y|x)log(Q(x)W(y|x)/R(y)).
    Break this up into T(x) = T1(x) + T2(x) - T3(x) where:
      * T1(x) = \sum_y W(y|x)log(Q(x)) = log(Q(x))\sum_y W(y|x) = log(Q(x)),
      * T2(x) = \sum_y W(y|x)log(W(y|x)) (i.e entries on the diagonal of W.T@W),
      * T3(x) = \sum_y W(y|x)log(R(y)).

    Tx = np.array([
      sum(
        W[j,i]*log(Q[i]*W[j,i]/sum(W[j,ip]*Q[ip] for ip in range(x)))
        for j in range(y)
      ) for i in range(x)
    ])
    '''
    T1 = log(Q)
    T2 = (W*log(W)).sum(axis=0)
    R = W@Q
    T3 = W.T@log(R)
    T = T1 + T2 - T3

    # Compute new probability distribution.
    eT = exp(T)
    Q = eT/eT.sum()

    diff = T-log(Q)
    m, M = min(diff), max(diff)
    Ip = I(Q, W)
    its.append((Ip, m, M))
    print(f'iteration #{r+1}: {m} <= {Ip} <= {M}')

  return Q, its


if __name__ == '__main__':
  rna = Bio.Data.CodonTable.standard_rna_table
  Gx = {**rna.forward_table, **{codon: 'Stop' for codon in rna.stop_codons}}
  G = {Codon[codon]: AminoAcid[aa] for codon, aa in Gx.items()}
  W = np.array([[
      1-p if G[x] == y else p/20
      for x in Codon
    ]
    for y in AminoAcid
  ])

  Q, its = blahut_arimoto(W)

  if PLOT:
    Is, ms, Ms = zip(*its)
    plt.plot(range(1, len(Is)+1), Is, lw=2, marker='x', label='I(Q,W)')
    plt.plot(range(1, len(ms)+1), ms, lw=2, marker='o', label='min T-log Q')
    plt.plot(range(1, len(Ms)+1), Ms, lw=2, marker='o', label='max T-log Q')
    plt.xlabel('iteration')
    plt.ylabel('mutual info (bounds)')
    plt.legend(loc='lower right')
    plt.show()

