#!/usr/bin/env python3
import Bio.Data.CodonTable
import argparse
import itertools
import json
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import pickle
import redis
import seaborn as sns

from Bio import Entrez
from Bio.Seq import Seq
from collections import defaultdict
from enum import Enum
from tqdm import tqdm

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
  > python3 info.py --bits disk --taxid=7227 -p=.001
  > python3 info.py --bits info --taxid=6239 -n=40
  ```
'''

sns.set_theme()

BANK_FILE = 'o586358-genbank_species.tsv'
PICKLE_FILE = 'species.pkl'
AUTHOR_EMAIL = 'davidb2@illinois.edu'
RANKS_PKL = 'ranks.pkl'
REDIS_HOST = 'localhost'
REDIS_PORT = 8000
REDIS_NAME = 'cluster'

redis_store = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
Entrez.email = AUTHOR_EMAIL

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

# Get codon table.
DNA = Bio.Data.CodonTable.standard_dna_table
Gx = {**DNA.forward_table, **{codon: 'Stop' for codon in DNA.stop_codons}}
G = {Codon[codon]: AminoAcid[aa] for codon, aa in Gx.items()}

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


def dna2rna(dna):
  '''Converts a string of dna to a string of rna.'''
  return str(Seq(dna).complement().transcribe())


def disk_plot(Q, W, p, species, taxid, bits):
  '''
  disk plot of Q and list I(Q,W).

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
  scodons = sorted(dna2rna(codon.name) for codon in Codon)
  for i, (bar, codon, ts) in enumerate(zip(bars, scodons, theta_shifted)):
    ax.text(
      ts, radius, codon, ha='center', va='center',
      rotation=np.rad2deg((i+1/2)*2*np.pi/len(Codon)),
      color=bar.get_facecolor(), family='monospace',
    )

  # TODO(davidb2): Add color legend.

  # Set title.
  IQW = I(Q, W)
  base = 'bits' if bits else 'base e'
  plt.title(f'Species: {species}, Taxid: {taxid}, I(Q,W)={IQW} ({base}) w/ p={p}')

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
  codons = sorted([codon.name for codon in Codon], key=dna2rna)
  tff = tf[codons]
  ttf = tff.iloc[0].to_numpy()

  # Check that database is telling the truth.
  n = tf['# Codons'].squeeze()
  actual_n = ttf.sum()
  assert actual_n == n, (actual_n, n)

  return (ttf / n, tf['Species'].squeeze())


def create_channel(p):
  '''Creates the noisy channel W w/ p as the noise.'''
  return np.array([[
      1-p if G[x] == y else p/20
      for x in Codon
    ]
    for y in AminoAcid
  ])


def cluster(args, df):
  '''Cluster species by I(Q; W)'''
  codons = sorted([codon.name for codon in Codon], key=dna2rna)
  p = args.p
  W = create_channel(p)

  # Extract correct row based on taxonomy id.
  tf = df[df['Organelle'] == 'genomic']
  tff = tf[codons]

  # Check that database is telling the truth.
  n = tf['# Codons']
  actual_n = tff.sum(axis=1)
  assert (actual_n == n).all(), (actual_n, n)
  Qs = tff.div(n, axis=0)
  tfs = tf['Taxid'].tolist()

  # Map from taxid to codon frequency/percentage.
  tid2Q = {}
  for tid, Qx in tqdm(zip(tfs, Qs.itertuples()), desc='tid2Q', total=len(tfs)):
    Q = np.array(Qx)[1:]
    tid2Q[tid] = Q

  # TaxIds we have not recorded yet.
  rtfs = [
    str(tid)
    for tid in tqdm(tfs, desc='rtfs')
    if not redis_store.hexists(name=REDIS_NAME, key=tid)
  ]
  print(f'# missing taxids: {len(rtfs)}')

  # Start fetches.
  tids = ','.join(rtfs)
  CHUNK_SIZE = 10000
  NUM_ITEMS = len(rtfs)

  for idx in tqdm(range(NUM_ITEMS // CHUNK_SIZE), desc='fetching handles'):
    handle = Entrez.efetch(
      db='Taxonomy',
      id=tids,
      retmode='xml',
      retstart=idx*CHUNK_SIZE,
      retmax=CHUNK_SIZE,
    )

    print('reading records ...')
    records = Entrez.read(handle)

    for record in tqdm(records, desc='records'):
      tid = int(record['TaxId'])
      if tid not in tid2Q:
        print(f'tid {tid} not found in tid2Q')
        continue

      Q = tid2Q[tid]
      redis_store.hset(REDIS_NAME, mapping={
        tid: json.dumps({
          'mutual_info': I(Q, W),
          'name': str(record['ScientificName']),
          'taxid': int(record['TaxId']),
          'ranks': {
            str(lineage['Rank']): {
              'name': str(lineage['ScientificName']),
              'taxid': int(lineage['TaxId']),
            }
            for lineage in itertools.chain(record, record['LineageEx'])
          }
        })
      })

  # Create a box plot for the mutual information by superkingdom.
  groups = defaultdict(list)
  n_entries = redis_store.hlen(name=REDIS_NAME)
  entries = redis_store.hscan_iter(name=REDIS_NAME)
  for k, vv in tqdm(entries, desc='HSCAN', total=n_entries):
    v = json.loads(vv)
    ranks = v['ranks']
    cls = None
    if 'superkingdom' in ranks:
      cls = ranks['superkingdom']
    elif 'no rank' in ranks:
      cls = ranks['no rank']
    else:
      print('class not found')
      print(f'taxid: {k}')
      print(json.dumps(v, indent=2, sort_keys=True))
      continue

    group = cls['name']
    mutual_info = v['mutual_info']
    groups[group].append(mutual_info)

  data = list(groups.values())
  labels = [
    f'{label} (n={len(datum)})'
    for label, datum in groups.items()
  ]

  plt.boxplot(x=data, labels=labels)
  plt.title(f'Mutual Info based on superkingdom')
  plt.xlabel('Superkingdom')
  plt.xticks(rotation=-25)
  plt.ylabel(f'Mutual Information (bits)')
  plt.show()


def info(args, df):
  '''Plot mutual info wrt p.'''
  # Get codon usage from database.
  Q, species = get_codon_mass(df, args.taxid)

  ps, iqs = list(zip(*(
    (p, I(Q, create_channel(p)))
    for p in np.linspace(start=0, stop=1, num=args.n)
  )))

  base = 'bits' if args.bits else 'base e'
  plt.title(f'Species: {species}, Taxid: {args.taxid}, p vs. I(Q,W)')
  plt.xlabel('p')
  plt.ylabel(f'Information ({base})')
  plt.plot(ps, iqs, linewidth=2, marker='x')
  plt.show()


def disk(args, df):
  '''Command if wanting to plot an individual species condon usage.'''
  # Get codon usage from database.
  Q, species = get_codon_mass(df, args.taxid)

  # Generate channel noise distribution.
  p = args.p
  W = create_channel(p)

  print(Q)
  # Finally, plot the info.
  disk_plot(Q, W, p, species, args.taxid, args.bits)


def main(args):
  global log, exp
  if args.bits:
    log, exp = np.log2, lambda x: 2**x

  if not args.no_cache and pathlib.Path(PICKLE_FILE).exists():
    df = pd.read_pickle(PICKLE_FILE)
  else:
    df = pd.read_csv(args.bank, sep='\\t+', engine='python')
    df.to_pickle(PICKLE_FILE)

  if args.command == 'disk':
    disk(args, df)
  elif args.command == 'info':
    info(args, df)
  elif args.command == 'cluster':
    cluster(args, df)

if __name__ == '__main__':
  parser = argparse.ArgumentParser('Utilities related to mutual information.')
  parser.add_argument('--bits', action='store_true', help='use log base 2')
  parser.add_argument('--bank', type=pathlib.Path, default=BANK_FILE, help='species file')
  parser.add_argument('--no-cache', action='store_true', help='do not use pickle')

  subparsers = parser.add_subparsers(dest='command', required=True)

  diskc = subparsers.add_parser('disk', help='plot codon usage as a disk')
  diskc.add_argument('-p', type=float, default=1e-4, help='channel noise')
  diskc.add_argument('--taxid', type=int, required=True, help='taxonomy id')

  infoc = subparsers.add_parser('info', help='plot p vs. I(Q,W)')
  infoc.add_argument('-n', type=int, default=10, help='number of intervals for p')
  infoc.add_argument('--taxid', type=int, required=True, help='taxonomy id')

  clusterc = subparsers.add_parser('cluster', help='cluster I(Q,W) for different species')
  clusterc.add_argument('-p', type=float, default=1e-4, help='channel noise')

  args = parser.parse_args()
  main(args)
