#!/usr/bin/env python3
import Bio.Data.CodonTable
import argparse
import decimal
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np

from decimal import Decimal
from enum import Enum


log = lambda x: Decimal(x).ln()
exp = lambda x: Decimal(x).exp()

def isclose(a, b):
  prec = decimal.getcontext().prec
  return abs(a-Decimal(b)) <= 10**(3-prec)

def dmap(A, f):
  s = A.shape
  ax = A.flatten()
  na = np.array([Decimal(f(x)) for x in ax])
  Ap = na.reshape(s)
  return Ap

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
  assert isclose(Q.sum(), 1.)
  assert np.all(isclose(W.sum(axis=0), 1.))

  R = W@Q
  V = (W*Q).T/R

  assert R.shape == (y, )
  assert V.shape == (x, y)
  assert isclose(R.sum(), 1.)
  assert np.all(isclose(V.sum(axis=0), 1.))

  I1 = (W.T * dmap(V, log)).sum(axis=1) @ Q
  I2 = Q @ dmap(Q, log)
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
  assert np.all(isclose(W.sum(axis=0), 1.))

  # Initialize Q as the uniform distribution.
  Q = np.array([Decimal(q) for q in np.ones(x)]) / x

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
    T1 = dmap(Q, log)
    T2 = (W*dmap(W, log)).sum(axis=0)
    R = W@Q
    T3 = W.T@dmap(R, log)
    T = T1 + T2 - T3

    diff = T-dmap(Q, log)
    m, M = min(diff), max(diff)
    Ip = I(Q, W)
    its.append((Ip, m, M))

    # Compute new probability distribution.
    eT = dmap(T, exp)
    Q = eT/eT.sum()

  return Q, its

def plot(Q, its):
  '''plot iterations of BA algorithm.'''
  Is, ms, Ms = zip(*its)
  plt.plot(range(1, len(Is)+1), Is, lw=2, marker='x', label='I(Q,W)')
  plt.plot(range(1, len(ms)+1), ms, lw=2, marker='o', label='min T-log Q')
  plt.plot(range(1, len(Ms)+1), Ms, lw=2, marker='o', label='max T-log Q')
  plt.xlabel('iteration')
  plt.ylabel('mutual info (bounds)')
  plt.legend(loc='lower right')
  plt.show()

def plot_distribution(Q, show: bool = True):
  '''plot Q.'''
  plt.title("Q distribution")
  plt.bar([codon.name for codon in Codon], Q)
  plt.xlabel('codons')
  plt.ylabel('probability')
  plt.xticks(rotation=-90)
  if show:
    plt.show()

def plot_hunch(G, show: bool = True):
  '''plot hunch.'''
  plt.title("Hunch Q distribution")
  plt.bar(
    x=[codon.name for codon in Codon],
    height=[
      (len(AminoAcid) * sum(bool(G[z] == G[x]) for z in Codon)) ** -1
      for x in Codon
    ]
  )
  plt.xlabel('codons')
  plt.ylabel('probability')
  plt.xticks(rotation=-90)
  if show:
    plt.show()


def main(args):
  global log, exp
  if args.bits:
    log, exp = (lambda x: Decimal(x).log10()/Decimal(2).log10(), (lambda x: 2**x))

  decimal.getcontext().prec = args.prec

  # Get codon table.
  rna = Bio.Data.CodonTable.unambiguous_rna_by_id[args.id]
  Gx = {**rna.forward_table, **{codon: 'Stop' for codon in rna.stop_codons}}
  G = {Codon[codon]: AminoAcid[aa] for codon, aa in Gx.items()}

  # Generate channel noise distribution.
  p = args.p
  W = np.array([[
      Decimal(1-p if G[x] == y else p/(len(AminoAcid)-1))
      for x in Codon
    ]
    for y in AminoAcid
  ])

  # Run BA using generated channel.
  Q, its = blahut_arimoto(W, iterations=args.its)
  for r, (Ip, m, M) in enumerate(its):
    print(f'iteration #{r+1}: {m} <= {Ip} <= {M}')

  # hcc = -len(AminoAcid) * (p/(len(AminoAcid)-1) + (1-len(AminoAcid)*p/(len(AminoAcid)-1))/len(AminoAcid) + len   )
  a = (p/20 + (1-21*p/20)/21)
  hcc = -21 * a * log(a) + 21*p/20*log(p/20) + p/20*log(p/20) - (1-p) * log(1-p)
  print(f"hunch channel capacity: {hcc}")
  if args.plot:
    plot(Q, its)

  if args.plot_distribution:
    plot_distribution(Q, show=False)
    plt.figure()
    plot_hunch(G)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('Blahut-Arimoto Algorithm for ribosome')
  parser.add_argument('--bits', action='store_true', help='use log base 2')
  parser.add_argument('--plot', action='store_true', help='plot convergence')
  parser.add_argument('--plot-distribution', action='store_true', help='plot distribution')
  parser.add_argument('--its', type=int, default=10, help='num iterations')
  parser.add_argument(
    '-p', type=Decimal, default=Decimal('0.0001'), help='channel noise'
  )
  parser.add_argument(
    '--prec', type=int, default=10, help='num digits of precision'
  )
  parser.add_argument(
    '--id', type=int, default=1, help='the id of the codon table from NCBI',
  )
  args = parser.parse_args()
  main(args)
