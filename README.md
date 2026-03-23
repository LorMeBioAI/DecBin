# DecBin

## Python dependencies
Essential libraries

* numpy
* pandas
* hyperopt
* biopython
* scikit-learn
* torch
* tabulate


## Quick Start
1. Data Preparation
   
   Place the Fastq sequencing file to be analyzed into the raw/ directory (example: raw/SRR13128012.fastq).

   Three PacBio HiFi human intestinal datasets, SRR13128012-SRR13128014, were obtained from the National Center for Bio technology Information (NCBI) (https://www.ncbi.nlm.nih.gov) under BioProject number PRJNA680590. NWC2 reads were obtained from the NCBI BioSample SAMN09580370 under the SRA accession codes SRX4451758 (Nanopore) and SRX4451757 (PacBio).

2. VAE Training
```
python vae-dec.py reads -r ./raw/SRR13128012.fastq -o ./SRR13128012
```

3. DecBin Training
```
python train_dec.py
```
