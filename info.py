#!/usr/bin/env python3
import Bio.Data.CodonTable
import argparse
import itertools
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import seaborn as sns

from enum import Enum

'''
A script to plot codon usage from CoCoPUTs.
Sources:
  * https://pubmed.ncbi.nlm.nih.gov/31029701/
  * https://hive.biochemistry.gwu.edu/cuts/about

Before running, you will need to install some packages:
  ```
  > python3 -m pip install biopython matplotlib numpy pandas seaborn
  ```

A typical invocation of the program looks like:
  ```
  > python3 info.py --bits --taxid=7227
  ```
'''

sns.set_theme()

BANK_FILE = 'o586358-genbank_species.tsv'
PICKLE_FILE = 'species.pkl'

log = np.log
exp = np.exp

# The 4 bases for DNA.
Base = Enum('Base', ['A', 'C', 'G', 'T'])

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

  # We should ignore zeros.
  R = W@Q
  with np.errstate(divide='ignore', invalid='ignore'):
    V = (W*Q).T/R
    V[~np.isfinite(V)] = 0  # -inf inf NaN

  assert R.shape == (y, )
  assert V.shape == (x, y)
  assert np.isclose(R.sum(), 1.)
  assert np.all(np.isclose(V.sum(axis=0), 1.))

  with np.errstate(divide='ignore', invalid='ignore'):
    lV = log(V)
    lV[~np.isfinite(lV)] = 0

  I1 = (W.T * lV).sum(axis=1) @ Q
  with np.errstate(divide='ignore', invalid='ignore'):
    lQ = log(Q)
    lQ[~np.isfinite(lQ)] = 0

  I2 = Q @ lQ
  mI = I1 - I2

  return mI


def plot(Q, W, p, species, taxid, bits):
  '''
  plot Q and list I(Q,W).

  :Q:       the pmf of the codons.
  :W:       the channel.
  :p:       the noise of the channel.
  :species: the name of the species.
  :taxid:   the unique taxonomy id.
  :bits:    a bool which is True iff we are using log2 to compute I(Q,W).
  '''
  # Create a polar graph.
  theta = np.linspace(0, 2*np.pi, len(Codon), endpoint=False)
  radii = Q
  radius = np.max(Q)
  cm = plt.cm.hot
  colors = cm(radii/(2*np.max(radii)))
  ax = plt.subplot(111, projection='polar')
  bars = ax.bar(
    theta, radii,
    width=theta[1], bottom=0, color=colors,
    alpha=0.5, align='edge',
  )

  # Set the outer ring to be invisible.
  ax.spines['polar'].set_visible(False)

  # Set the grid line locations but set the labels to be invisible.
  ax.grid(False)
  ax.set_thetagrids([], visible=False)
  ax.set_rgrids([3], visible=False)
  theta_shifted = theta + theta[1]/2

  # Add codon labels manually.
  for i, (bar, codon, ts) in enumerate(zip(bars, Codon, theta_shifted)):
    ax.text(
      ts, radius, codon.name, ha='center', va='center',
      rotation=np.rad2deg((i+1/2)*2*np.pi/len(Codon)),
      color=bar.get_facecolor(), family='monospace',
    )

  # TODO(davidb2): Add color legend.

  # Set title.
  info = I(Q, W)
  base = 'bits' if bits else 'base e'
  plt.title(f'Species: {species}, Taxid: {taxid}, I(Q,W)={info} {(base)} w/ p={p}')

  plt.show()


def get_codon_mass(df, taxid):
  '''
  :df:    the genbank_species pd.DataFrame.
  :taxid: the taxonomy id for the species.

  returns a 64 length numpy array for the codon mass (lexicographic order).
  '''
  # Extract correct row based on taxonomy id.
  tf = df[(df['Taxid'] == taxid) & (df['Organelle'] == 'genomic')]
  assert not tf.empty, f'could not find unique entry with taxid={taxid}.'
  tff = tf[[codon.name for codon in Codon]]
  ttf = tff.iloc[0].to_numpy()

  # Check that database is telling the truth.
  n = tf['# Codons'].squeeze()
  actual_n = ttf.sum()
  assert actual_n == n, (actual_n, n)

  return (ttf / n, tf['Species'].squeeze())


def main(args):
  global log, exp
  if args.bits:
    log, exp = np.log2, lambda x: 2**x

  if not args.no_cache and pathlib.Path(PICKLE_FILE).exists():
    df = pd.read_pickle(PICKLE_FILE)
  else:
    df = pd.read_csv(args.bank, sep='\\t+', engine='python')

  # Get codon usage from database.
  Q, species = get_codon_mass(df, args.taxid)

  # Get codon table.
  dna = Bio.Data.CodonTable.standard_dna_table
  Gx = {**dna.forward_table, **{codon: 'Stop' for codon in dna.stop_codons}}
  G = {Codon[codon]: AminoAcid[aa] for codon, aa in Gx.items()}

  # Generate channel noise distribution.
  p = args.p
  W = np.array([[
      1-p if G[x] == y else p/20
      for x in Codon
    ]
    for y in AminoAcid
  ])

  print(Q)
  # Finally, plot the info.
  plot(Q, W, p, species, args.taxid, args.bits)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('Map the pmf of Q.')
  parser.add_argument('--taxid', type=int, required=True, help='taxonomy id')
  parser.add_argument('--bits', action='store_true', help='use log base 2')
  parser.add_argument('--bank', type=pathlib.Path, default=BANK_FILE, help='species file')
  parser.add_argument('--no-cache', action='store_true', help='do not use pickle')
  parser.add_argument('-p', type=float, default=1e-4, help='channel noise')
  args = parser.parse_args()
  main(args)
