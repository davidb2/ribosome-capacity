import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from Bio import SeqIO
from Bio.Data import CodonTable
from Bio.Seq import Seq
from Bio.codonalign.codonseq import CodonSeq
from collections import Counter
from info import (
  CODON_LENGTH,
  Codon,
  AminoAcid,
  G,
  I,
  set_bits,
  create_channel,
  compute_dist_from_capacity_achieving,
)
from multiprocessing import Pool
from typing import *


NUM_PROCESSES = 32
FILENAME = '/Users/david/Downloads/ncbi/4932-genome-rna/ncbi_dataset/data/GCF_000146045.2/rna.fna'

set_bits()

def set_sns():
  sns.set_style("ticks")


def get_codons(rna: Seq):
  num_codons, remainder = divmod(len(rna), CODON_LENGTH)
  assert remainder == 0, len(rna)
  rna = rna.transcribe() # just to make sure we map T -> U.

  codon_seq = CodonSeq(str(rna))

  codons = [
    # codon_seq.get_codon(idx)
    # Codon(np.random.randint(0, 64)+1).name
    Codon(1).name
    for idx in range(num_codons)
  ]
  return codons


def codons_to_Q(codons: List[str | CodonSeq]):
  num_codons = len(codons)
  Q = np.zeros(shape=(64,))
  codon_counts = Counter(codons)
  for codon, count in sorted(codon_counts.items(), key=lambda t: t[0]):
    idx = Codon[Seq(codon).back_transcribe()].value-1
    Q[idx] = count / num_codons

  assert np.isclose(Q.sum(), 1), Q.sum()
  return Q


def get_info(args):
  record, W = args
  if 'mRNA' not in record.description: return None
  rna = record.seq.transcribe()

  if len(rna) % CODON_LENGTH != 0:
    print('bad mRNA length', 'skipping...')
    return None

  codons = get_codons(rna)
  # if codons[0] not in CodonTable.standard_rna_table.start_codons:
  #   print(f'bad mRNA start codon {codons[0]}', 'skipping...')
  #   return None

  # if codons[-1] not in CodonTable.standard_rna_table.stop_codons:
  #   print(f'bad mRNA stop codon {codons[-1]}', 'skipping...')
  #   return None


  print(record.description)
  Q = codons_to_Q(codons)
  return (I(Q,W), compute_dist_from_capacity_achieving(Q))


def compute():
  count = 0
  infos: List[Tuple[float, float]] = []
  W = create_channel(p=1e-4)
  with open(FILENAME, 'r') as handle:
    with Pool(processes=NUM_PROCESSES) as p:
      entries = ((record, W) for record in SeqIO.parse(handle, "fasta"))
      for result in p.imap_unordered(get_info, entries):
        if result is not None:
          infos.append(result)
          count += 1

  print(f"{count=}")
  return pd.DataFrame(data=infos, columns=['information', 'distance from capacity achieving'])


def store(df: pd.DataFrame):
  pd.to_pickle(df, 'Q.pkl')


def channel_capacity():
  subsets = [
    [
      idx
      for idx, codon in enumerate(sorted(map(lambda c: Seq(c.name).transcribe(), Codon)))
      if G[Codon[str(codon.back_transcribe())]] == aa
    ]
    for aa in AminoAcid
  ]
  Q = np.zeros(shape=(len(Codon),))
  for subset in subsets:
    Q[subset] = (1/len(AminoAcid))/len(subset)

  assert np.isclose(Q.sum(), 1), Q.sum()
  W = create_channel(p=1e-4)
  return I(Q, W)


def draw(df: pd.DataFrame):
  set_sns()

  fig, ax = plt.subplots(nrows=1, ncols=2)

  sns.histplot(data=df, x='information', ax=ax[0])
  ax[0].vlines(x=channel_capacity(), ymin=0, ymax=450, colors='r', linestyles='dashed')
  ax[0].set_xticks([2.5+.25*i for i in range(9)])

  sns.histplot(data=df, x='distance from capacity achieving', ax=ax[1])
  ax[1].set_xticks([.1 + .1*i for i in range(7)])

  plt.show()


if __name__ == '__main__':
  W = create_channel(p=1e-4)
  COMPUTE = True
  df: Optional[pd.DataFrame] = None
  if COMPUTE:
    df = compute()
    store(df)
  else:
    df = pd.read_pickle('Q.pkl')

  assert df is not None

  print('drawing')
  draw(df)
