#!/usr/bin/env python3
import Bio.Data.CodonTable
import argparse
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import seaborn as sns

from ba import I, Codon, AminoAcid
from enum import Enum

sns.set_theme()

BANK_FILE = 'o586358-genbank_species.tsv'
PICKLE_FILE = 'species.pkl'

# The 4 bases for DNA.
Base = Enum('Base', ['A', 'C', 'G', 'T'])

# Creates 64 codons.
Codon = Enum('Codon', [
  ''.join(base.name for base in codon)
  for idx, codon in enumerate(itertools.product(Base, repeat=3))
])


def plot(Q, W, p):
  '''pmf of Q and calculate pmf'''

  # print(Q)
  # fQ = np.cumsum(Q)
  # plt.plot(range(len(Q)), Q, lw=2, marker='o', label='pmf')
  # plt.plot(range(len(Q)), fQ, lw=2, marker='x', label='cdf')
  # plt.xlabel('codons')
  # plt.ylabel('p')
  # plt.ylim(bottom=0)
  # plt.legend(loc='upper left')

  theta = np.linspace(0, 2*np.pi, len(Codon), endpoint=False)
  radii = Q
  radius = np.max(Q)
  colors = plt.cm.hot(radii/(2*np.max(radii)))
  ax = plt.subplot(111, projection='polar')
  bars = ax.bar(
    theta, radii,
    width=theta[1], bottom=0, color=colors,
    alpha=0.5, align='edge',
  )

  # Turn off polar labels
  # ax.axes.get_xaxis().set_visible(False)
  # ax.axes.get_yaxis().set_visible(False)
  # Set the outer ring to be invisible.
  ax.spines["polar"].set_visible(False)
  # # Set the grid line locations but set the labels to be invisible.
  ax.grid(False)
  ax.set_thetagrids([], visible=False)
  ax.set_rgrids([3], visible=False)
  theta_shifted=theta-theta[1]/2
  for i, (bar, codon, ts) in enumerate(zip(bars, Codon, theta_shifted)):
    ax.text(
      ts, radius, codon.name, ha='center', va='center',
      rotation=np.rad2deg((i)*2*np.pi/len(Codon)),
      color=bar.get_facecolor(), family='monospace',
    )
  # ax.xticks(range(len(Q)), [x.name for x in Codon], rotation=-90)
  plt.show()
  info = I(Q, W)
  # ax.text(f'Info: {info}')


def get_codon_mass(df, taxid):
  '''
  :df:    the genbank_species pd.DataFrame.
  :taxid: the taxonomy id for the species.

  returns a 64 length numpy array for the codon mass (lexicographic order).
  '''
  # Extract correct row based on taxonomy id.
  tf = df[df['Taxid'] == taxid]
  assert not tf.empty, len(tf)
  ttf = tf[[codon.name for codon in Codon]].loc[0].to_numpy()

  # Check that database is telling the truth.
  n = tf['# Codons'].squeeze()
  actual_n = ttf.sum()
  assert actual_n == n, (actual_n, n)

  return ttf / n


def main(args):
  global log, exp
  if args.bits:
    log, exp = np.log2, lambda x: 2**x

  if not args.no_cache and pathlib.Path(PICKLE_FILE).exists():
    df = pd.read_pickle(PICKLE_FILE)
  else:
    df = pd.read_csv(args.bank, sep='\\t+', engine='python')

  # Get codon usage from database.
  Q = get_codon_mass(df, args.taxid)

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

  # Finally, plot the info.
  plot(Q, W, p)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('Map the pmf of Q.')
  parser.add_argument('--taxid', type=int, required=True, help='taxonomy id')
  parser.add_argument('--bits', action='store_true', help='use log base 2')
  parser.add_argument('--bank', type=pathlib.Path, default=BANK_FILE, help='species file')
  parser.add_argument('--no-cache', action='store_true', help='do not use pickle')
  parser.add_argument('-p', type=float, default=1e-4, help='channel noise')
  args = parser.parse_args()
  main(args)
