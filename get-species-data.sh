#!/usr/bin/env bash
TAXID=3702
ACCESSION=GCF_000157115.2

datasets summary taxonomy taxon "Escherichia coli"
datasets summary taxonomy taxon "$TAXID" > "$TAXID-summary.json"
datasets download genome taxon "$TAXID" --include rna --filename "$TAXID-genome-rna.zip"
datasets download genome accession "$ACCESSION"