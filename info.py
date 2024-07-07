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
import ba_arbitrary

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
  > python3 -m pip install biopython matplotlib numpy pandas redis seaborn
  ```

A typical invocation of the program looks like:
  ```
  > python3 info.py --bits disk --taxid=7227 -p=.001
  > python3 info.py --bits info --taxid=6239 -n=40
  ```
'''

sns.set_theme(font_scale=2, rc={'text.usetex' : True})
sns.set_style("whitegrid", {'axes.grid' : False})

BANK_FILE = 'o586358-genbank_species.tsv'
PICKLE_FILE = 'species.pkl'
AUTHOR_EMAIL = 'davidb2@illinois.edu'
RANKS_PKL = 'ranks.pkl'
REDIS_HOST = 'localhost'
REDIS_PORT = 8000
REDIS_NAME = 'cluster'

Entrez.email = AUTHOR_EMAIL

log = np.log
exp = np.exp

# The 4 bases for DNA.
Nucleotide = Enum('Nucleotide', ['A', 'C', 'G', 'T'])

# Creates 64 codons.
CODON_LENGTH = 3
Codon = Enum('Codon', [
  ''.join(base.name for base in codon)
  for idx, codon in enumerate(itertools.product(Nucleotide, repeat=CODON_LENGTH))
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


def disk_plot(Q, W, p, species, taxid, base, id=None):
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
  SCALAR = 150
  radii = Q * SCALAR
  radius = np.max(Q) * SCALAR
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
  dist = compute_dist_from_capacity_achieving(Q)
  Qp = np.full(len(Codon), 1/len(Codon))
  distp = compute_dist_from_capacity_achieving(Qp)
  print(distp)
  plt.title(f'Species: {species}, Taxid: {taxid}, Id={id}, I(Q,W)={IQW:.3f} ({base}), dist(Q)={dist:.3f} w/ p={p}')

  plt.show()

def total_variation_distance(P, Q):
  return np.abs(P-Q).sum()/2

def kl_divergence_Q_from_P(P, Q):
  return sum(
    (
    Q[x] * log(Q[x] / P[x]) if not np.isclose(P[x], 0)
    else 0 if np.isclose(P[x], Q[x])
    else np.inf
    )
    for x in range(len(Q))
    if not np.isclose(Q[x], 0)
  )
  # return np.sum(Q * np.log(Q / P))
  # return np.sum(P * np.log(P / Q))

def constraints(subsets):
  cons = []
  for subset in subsets:
    cons.append({'type': 'eq', 'fun': lambda P, subset=subset: np.sum(P[subset]) - 1/len(AminoAcid)})
  return cons

def compute_dist_from_capacity_achieving(Q: np.array):
  assert np.isclose(np.sum(Q),1), np.sum(Q)
  from scipy.optimize import minimize


  # Define your subsets here. Each element in `subsets` should be a list of indices for a subset A_i.
  # Example for 21 subsets, each containing different indices
  subsets = [
    [
      idx
      for idx, codon in enumerate(sorted(map(lambda c: Seq(c.name).transcribe(), Codon)))
      if G[Codon[str(codon.back_transcribe())]] == aa
    ]
    for aa in AminoAcid
  ]

  # Constraints
  cons = constraints(subsets)

  # Bounds for P (P_j must be in [0, 1])
  bounds = [(0, 1) for _ in range(len(Q))]

  # Initial guess for P (must be positive and sum to 1)
  P0 = np.full(len(Q), 1/len(Q))

  # Optimization
  result = minimize(total_variation_distance, P0, args=(Q,), constraints=cons, bounds=bounds, method='SLSQP')

  # Result
  P_opt = result.x
  # print("Optimized P:", P_opt)
  return total_variation_distance(P_opt, Q)



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
  # Initialize our storage early.
  redis_store = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

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
  tns = n.tolist()

  # Map from taxid to codon frequency/percentage and number of codons.
  tid2Q = {}
  tid2n = {}
  triples = zip(tfs, Qs.itertuples(), tns)
  for tid, Qx, n_codons in tqdm(triples, desc='tid2Q', total=len(tfs)):
    Q = np.array(Qx)[1:]
    tid2Q[tid] = Q
    tid2n[tid] = n_codons

  # TaxIds we have not recorded yet.
  rtfs = [
    str(tid)
    for tid in tqdm(tfs, desc='rtfs')
    if not redis_store.hexists(name=REDIS_NAME, key=tid)
  ]
  print(f'# taxids missing from redis store: {len(rtfs)}')

  # Start fetches.
  tids = ','.join(rtfs)
  CHUNK_SIZE = 10000
  NUM_ITEMS = len(rtfs)
  missing_tids = []

  # TODO(davidb2): parallelize this.
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
        # print(f'tid {tid} not found in tid2Q')
        missing_tids.append(tid)
        continue

      Q = tid2Q[tid]
      n_codons = tid2n[tid]
      redis_store.hset(REDIS_NAME, mapping={
        tid: json.dumps({
          'mutual_info': I(Q, W),
          'n_codons': n_codons,
          'name': str(record['ScientificName']),
          'taxid': int(record['TaxId']),
          'ranks': {
            str(lineage['Rank']): {
              'name': str(lineage['ScientificName']),
              'taxid': int(lineage['TaxId']),
            }
            for lineage in itertools.chain([record], record['LineageEx'])
          }
        })
      })

  print(f'# taxids missing from cocoputs: {len(missing_tids)}')

  # Plot data.
  cluster_scatterplot(redis_store)
  # cluster_boxplot_by(redis_store, key='superkingdom')


def cluster_scatterplot(redis_store):
  '''Create a scatterplot: number of codons vs. mutual_info.'''
  xs, ys = [], []
  n_entries = redis_store.hlen(name=REDIS_NAME)
  entries = redis_store.hscan_iter(name=REDIS_NAME)
  for k, vv in tqdm(entries, desc='HSCAN', total=n_entries):
    v = json.loads(vv)
    if 'n_codons' not in v:
      print(f'tid {k} does not have n_codons')
      continue
    xs.append(v['n_codons'])
    ys.append(v['mutual_info'])

  from scipy.stats import pearsonr, spearmanr
  pr = pearsonr(xs, ys)
  sr = spearmanr(xs, ys)

  df = pd.DataFrame(list(zip(xs, ys)), columns=["Number of codons", "Mutual information"])
  plot = sns.relplot(
    data=df,
    x="Number of codons",
    y="Mutual information",
    # aspect="num edges",
    # markers="undirected",
    # scatter_kws={'alpha': 0.5, "s": 4, 'linewidths': 0.1},
    s=4,
    # facet_kws={'sharey': True, 'sharex': False},
    # hue="type",
    # hue_order=["undirected", "directed", "oriented"],
    # col="r",
    # col_wrap=2,
    # picker=4,
    # sharex=False,
    # sharey=True,
    # fit_reg=False,
  )
  # plt.scatter(xs, ys, mark)
  plt.title(f'Number of codons vs Mutual Info (pearson r={pr}, spearman r={sr})')
  # plt.xlabel('number of codons')
  # plt.ylabel(f'Mutual Information (bits)')
  plt.xscale('log')
  plt.show()


def cluster_boxplot_by(redis_store, key):
  '''Create a box plot for the mutual information by superkingdom.'''
  groups = defaultdict(list)
  n_entries = redis_store.hlen(name=REDIS_NAME)
  entries = redis_store.hscan_iter(name=REDIS_NAME)
  for k, vv in tqdm(entries, desc='HSCAN', total=n_entries):
    v = json.loads(vv)
    ranks = v['ranks']
    cls = None
    if key in ranks:
      cls = ranks[key]
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

  base = 'bits' if args.bits else 'nucleotides' if args.nucleotides else 'base e'
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
  base = 'bits' if args.bits else 'nucleotides' if args.nucleotides else 'base e'
  disk_plot(Q, W, p, species, args.taxid, base)

def disk_gene(args):
  '''Command if wanting to plot an individual species condon usage.'''
  # Get codon usage from database.
  # Q, species = get_codon_mass(df, args.taxid)

  from Bio import Entrez, SeqIO
  from Bio.Seq import Seq
  import textwrap
  from collections import Counter

  Entrez.email = "dbrewster@g.harvard.edu"

  handle = Entrez.efetch(db="nuccore",
                        id=args.id,
                        rettype="gb",
                        retmode="text")

  record = SeqIO.read(handle, "genbank")
  untranscribed_rna: Seq = record.seq
  # mrna: Seq = dna.transcribe()
  assert len(untranscribed_rna) % CODON_LENGTH == 0

  counts = Counter([
      Codon[codon_str]
      for codon_str in textwrap.wrap(str(untranscribed_rna), CODON_LENGTH)
  ])
  codons = {
    Seq(dna_codon.name).transcribe(): count
    for dna_codon, count in counts.items()
  }
  # print(codons)

  codon_count = len(untranscribed_rna) // CODON_LENGTH
  Q = np.array([
    codons.get(codon, 0) / codon_count
    for codon in sorted(map(lambda c: Seq(c.name).transcribe(), Codon))
  ])



  # Generate channel noise distribution.
  p = args.p
  W = create_channel(p)

  print(f"{Q=}")
  # print(I(Q, W))
  # subsets = {
  #   aa: {
  #     idx
  #     for idx, codon in enumerate(sorted(map(lambda c: Seq(c.name).transcribe(), Codon)))
  #     if G[Codon[str(codon.back_transcribe())]] == aa
  #   }
  #   for aa in AminoAcid
  # }
  # print(subsets.values())
  # Qp = np.zeros(len(Codon))
  # for idx in range(len(Codon)):
  #   for aa in AminoAcid:
  #     if idx in subsets[aa]:
  #       Qp[idx] = 1/(len(subsets[aa]) * len(AminoAcid))
  # print(Qp)
  # print(compute_dist_from_capacity_achieving(Qp))

  # Finally, plot the info.
  base = 'bits' if args.bits else 'nucleotides' if args.nucleotides else 'base e'
  disk_plot(Q, W, p, None, None, base, args.id)


def main(args):
  global log, exp
  assert not (args.bits and args.nucleotides), "pick only one base"
  if args.bits:
    log, exp = np.log2, lambda x: 2**x
  if args.nucleotides:
    log, exp = lambda x: np.emath.logn(len(Nucleotide), x), lambda x: len(Nucleotide)**x

  if not args.no_cache and pathlib.Path(PICKLE_FILE).exists():
    df = pd.read_pickle(PICKLE_FILE)
  else:
    df = pd.read_csv(args.bank, sep='\\t+', engine='python')
    df.to_pickle(PICKLE_FILE)

  if args.command == 'disk':
    disk(args, df)
  elif args.command == 'disk-gene':
    disk_gene(args)
  elif args.command == 'info':
    info(args, df)
  elif args.command == 'cluster':
    cluster(args, df)

if __name__ == '__main__':
  np.set_printoptions(suppress=True)
  parser = argparse.ArgumentParser('Utilities related to mutual information.')
  parser.add_argument('--bits', action='store_true', help='use log base 2')
  parser.add_argument('--nucleotides', action='store_true', help='use log base 4')
  parser.add_argument('--bank', type=pathlib.Path, default=BANK_FILE, help='species file')
  parser.add_argument('--no-cache', action='store_true', help='do not use pickle')

  subparsers = parser.add_subparsers(dest='command', required=True)

  diskc = subparsers.add_parser('disk', help='plot codon usage as a disk')
  diskc.add_argument('-p', type=float, default=1e-4, help='channel noise')
  diskc.add_argument('--taxid', type=int, required=True, help='taxonomy id')

  disk_genec = subparsers.add_parser('disk-gene', help='plot codon usage of gene as a disk')
  disk_genec.add_argument('-p', type=float, default=1e-4, help='channel noise')
  # nuccore db
  disk_genec.add_argument('--id', type=str, required=True, help='id')

  infoc = subparsers.add_parser('info', help='plot p vs. I(Q,W)')
  infoc.add_argument('-n', type=int, default=10, help='number of intervals for p')
  infoc.add_argument('--taxid', type=int, required=True, help='taxonomy id')

  # Need to run `redis-server --port 8000` to use this command.
  clusterc = subparsers.add_parser('cluster', help='cluster I(Q,W) for different species')
  clusterc.add_argument('-p', type=float, default=1e-4, help='channel noise')

  args = parser.parse_args()
  main(args)
