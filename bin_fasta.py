# Binning based on label_csv and indice.csv
import pandas as pd
from Bio import SeqIO
import os
import numpy as np

def bin2fasta(input,output, pred_label):
    """
    :param input: Original fasta file
    :param output: Output folder
    :param pred_label: Binning labels
    :param indice: Slicing order
    :return: None
    """
    bin_files = {}
    fmt ="fasta" if input.split(
        '.')[-1].lower() in ["fasta", "fna", "fa"] else "fastq"
    if not os.path.exists(f"{output}/binned_reads"):
        os.mkdir(f"{output}/binned_reads")
    for r,record in enumerate(SeqIO.parse(input, fmt)):
        if pred_label[r] not in bin_files:
            bin_files[pred_label[r]] = open(f"{output}/binned_reads/Bin-{pred_label[r]}.fasta", "w+")
        bin_files[pred_label[r]].write(f">read-{r}\n")
        bin_files[pred_label[r]].write(f"{record.seq}\n")

if __name__=="__main__":
    label_file =""
    label_indice_file=""
    input_fasta=""
    output=""
    label = np.array(pd.read_csv(label_file, index_col=0)).reshape(-1)
    label_indice = np.array(pd.read_csv(label_indice_file, index_col=0)).reshape(-1)
    index=np.argsort(label_indice)
    label = label[index]
    bin2fasta(input_fasta,output,label)