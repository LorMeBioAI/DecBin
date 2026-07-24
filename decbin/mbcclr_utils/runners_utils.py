import os
import sys
from collections import defaultdict
from Bio import SeqIO

#import importlib.resources

#bin_dir = importlib.resources.files("decbin.mbcclr_utils.bin")

def split_contigs(contigs, output):
    contig_groups = defaultdict(list)
    fragment_parent = {}
    frag_path = os.path.join(output, "fragments", "contigs.fasta")

    with open(frag_path, "w+") as scf:
        i = 0
        for n, record in enumerate(SeqIO.parse(contigs, "fasta")):
            if len(record.seq) >= 5000:
                sub_contigs = [record.seq[x:x+2500] for x in range(0, len(record.seq), 2500)]
                sub_contigs.append(record.seq[-2500:])
            else:
                sub_contigs = [record.seq]

            for sc in sub_contigs:
                rid = f">{n}_{i}"
                rec = str(sc)
                scf.write(f"{rid}\n{rec}\n")

                contig_groups[str(record.id)].append(i)
                fragment_parent[i] = str(record.id)
                i += 1

    return contig_groups, fragment_parent


def run_kmers(reads_path, output, k_size, threads):
    prof_dir = os.path.join(output, "profiles")
    if not os.path.isdir(prof_dir):
        os.makedirs(prof_dir)

    exe_path = os.path.join(os.path.dirname(__file__), "bin", "count-kmers")

    #exe_path = str(bin_dir / "count-kmers")

    out_file = os.path.join(prof_dir, "com_profs")
    cmd = f'"{exe_path}" "{reads_path}" "{out_file}" {k_size} {threads}'
    print("CMD::" + cmd)
    o = os.system(cmd)
    check_proc(o, "Counting Trimers")


def run_15mer_counts(reads_path, output, threads):
    prof_dir = os.path.join(output, "profiles")
    if not os.path.isdir(prof_dir):
        os.makedirs(prof_dir)

    exe_path = os.path.join(os.path.dirname(__file__), "bin", "count-15mers")

    #exe_path = str(bin_dir / "count-15mers")

    out_file = os.path.join(prof_dir, "15mers-counts")
    cmd = f'"{exe_path}" "{reads_path}" "{out_file}" {threads}'
    print("CMD::" + cmd)
    o = os.system(cmd)
    check_proc(o, "Counting 15-mers")


def run_15mer_vecs(reads_path, output, bin_size, bin_count, threads):
    prof_dir = os.path.join(output, "profiles")
    if not os.path.isdir(prof_dir):
        os.makedirs(prof_dir)

    #exe_path = str(bin_dir / "search-15mers")
    exe_path = os.path.join(os.path.dirname(__file__), "bin", "search-15mers")
    count_file = os.path.join(prof_dir, "15mers-counts")
    out_file = os.path.join(prof_dir, "cov_profs")
    cmd = f'"{exe_path}" "{count_file}" "{reads_path}" "{out_file}" {bin_size} {bin_count} {threads}'
    print("CMD::" + cmd)
    o = os.system(cmd)
    check_proc(o, "Counting 15-mer profiles")


def check_proc(ret, name=""):
    if ret != 0:
        if name != "":
            print(f"Error in step: {name}")
        print("Failed due to an error. Good Bye!")
        sys.exit(ret)
