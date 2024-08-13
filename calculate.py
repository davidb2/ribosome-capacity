import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
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
from customlogger import logger
from pathlib import Path


YEAST_TAXID = 4932
ECOLI_TAXID = 562
CRESS_TAXID = 3702

NUM_PROCESSES = 32
TAXID = YEAST_TAXID

INPUT_FILENAME = {
  # Saccharomyces cerevisiae (brewer/baker's yeast)
  YEAST_TAXID: '/Users/david/Dropbox/experiments/ribosome-capacity/raw-data/ncbi/4932-genome-rna/ncbi_dataset/data/GCF_000146045.2_R64_cds_from_genomic.fna', # '/Users/david/Dropbox/experiments/ribosome-capacity/raw-data/ncbi/4932-genome-rna/ncbi_dataset/data/GCF_000146045.2/rna.fna',
  # Escherichia coli
  ECOLI_TAXID: '/Users/david/Dropbox/experiments/ribosome-capacity/raw-data/ncbi/562-genome-rna/ncbi_dataset/data/GCF_000157115.2_Escherichia_sp_3_2_53FAA_V2_cds_from_genomic.fna',
  # Arabidopsis thaliana (thale cress)
  CRESS_TAXID: '/Users/david/Dropbox/experiments/ribosome-capacity/raw-data/ncbi/3702-genome-rna/ncbi_dataset/data/GCF_000001735.4_TAIR10.1_cds_from_genomic.fna', # '/Users/david/Dropbox/experiments/ribosome-capacity/raw-data/ncbi/3702-genome-rna/ncbi_dataset/data/GCF_000001735.4/rna.fna',
}[TAXID]

OUTPUT_FILENAME = f'/Users/david/Dropbox/experiments/ribosome-capacity/pretty-data/Q-{TAXID}.csv'

# Have to put it here due to how the multiprocessing package works.
set_bits()

def set_sns():
  sns.set_style("ticks")


def get_codons(rna: Seq):
  num_codons, remainder = divmod(len(rna), CODON_LENGTH)
  assert remainder == 0, len(rna)
  rna = rna.transcribe() # just to make sure we map T -> U.

  codon_seq = CodonSeq(str(rna))

  codons = [
    codon_seq.get_codon(idx)
    # Codon(np.random.randint(0, 64)+1).name
    # Codon(1).name
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

def is_valid(record: SeqRecord) -> bool:
  """The fna files have different description formats."""
  if 'pseudo=true' in record.description: return False
  if 'pseudogene' in record.description: return False
  return True

  # if TAXID == YEAST_TAXID:
  #   if 'pseudo=true' in record.description: return False
  #   # if 'mRNA' not in record.description: return False
  #   return True
  # elif TAXID == ECOLI_TAXID:
  #   if 'pseudo=true' in record.description: return False
  #   return True
  # elif TAXID == CRESS_TAXID:
  #   if 'pseudo=true' in record.description: return False
  #   if 'pseudogene' in record.description: return False
  #   # if 'mRNA' not in record.description: return False
  #   return True


def get_info(args: Tuple[SeqRecord, np.array]):
  record, W = args

  if not is_valid(record): return None
  rna = record.seq.transcribe()

  if len(rna) % CODON_LENGTH != 0:
    logger.info(('bad mRNA length', 'skipping...'))
    return None

  unique_bps = set(Counter(str(rna)).keys())
  if not unique_bps <= {'A', 'C', 'G', 'U'}:
    logger.info(('found bad bp in rna strand; unique_bps:', unique_bps))
    return None

  codons = get_codons(rna)
  # if codons[0] not in CodonTable.standard_rna_table.start_codons:
  #   print(f'bad mRNA start codon {codons[0]}', 'skipping...')
  #   return None

  # if codons[-1] not in CodonTable.standard_rna_table.stop_codons:
  #   print(f'bad mRNA stop codon {codons[-1]}', 'skipping...')
  #   return None

  logger.info(record.description)
  Q = codons_to_Q(codons)
  return (record.id, record.description, I(Q,W), compute_dist_from_capacity_achieving(Q))


def compute(input_filename: str):
  count = 0
  infos: List[Tuple[float, float]] = []
  W: np.array = create_channel(p=1e-4)
  with open(input_filename, 'r') as handle:
    with Pool(processes=NUM_PROCESSES) as p:
      entries = ((record, W) for record in SeqIO.parse(handle, "fasta"))
      for result in p.imap_unordered(get_info, entries):
        if result is not None:
          infos.append(result)
          count += 1

  logger.info(f"{count=}")
  return pd.DataFrame(data=infos, columns=['id', 'description', 'information', 'distance from capacity achieving'])


def store(df: pd.DataFrame, output_filename: str):
  df.to_csv(output_filename, index=False)


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


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input-filename', type=str, default=INPUT_FILENAME)
  parser.add_argument('--output-filename', type=str, default=OUTPUT_FILENAME)
  return parser.parse_args()

if __name__ == '__main__':
  args = get_args()
  W = create_channel(p=1e-4)
  df = compute(args.input_filename)
  store(df, args.output_filename)