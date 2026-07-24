import os
import numpy as np

from .mbcclr_utils.runners_utils import *
from .vae_dec import ae_utils


def run_reads_binning(args):
    reads_path = args.reads_path
    threads = args.threads
    bin_size = args.bin_size
    bin_count = args.bin_count
    k_size = args.k_size
    epochs = args.ae_epochs
    dims = args.ae_dims
    hidden = list(map(int, args.ae_hidden.split(",")))
    cuda = "cuda:0"
    output = args.output

    print("Counting k-mers")
    run_kmers(reads_path, output, k_size, threads)
    print("Counting k-mers complete")

    print("Counting 15-mers")
    run_15mer_counts(reads_path, output, threads)
    print("Counting 15-mers complete")

    print("Computing 15-mer profiles")
    run_15mer_vecs(reads_path, output, bin_size, bin_count, threads)
    print("Computing 15-mer profiles complete")

    print("Profiles saving as numpy arrays")
    com_profs = np.array([
        np.array(list(map(float, line.strip().split())))
        for line in open(os.path.join(output, "profiles", "com_profs"))
        if len(line.strip()) > 0
    ])
    cov_profs = np.array([
        np.array(list(map(float, line.strip().split())))
        for line in open(os.path.join(output, "profiles", "cov_profs"))
        if len(line.strip()) > 0
    ])

    np.save(os.path.join(output, "profiles", "com_profs"), com_profs)
    np.save(os.path.join(output, "profiles", "cov_profs"), cov_profs)

    del com_profs
    del cov_profs
    print("Profiles saving as numpy arrays complete")

    constraints = None
    print("VAE training information")
    print(f"\tDimensions {dims}")
    print(f"\tHidden Layers {hidden}")
    print(f"\tEpochs {epochs}")

    ae_utils.train_vae(
        output=output,
        latent_dims=dims,
        hidden_layers=hidden,
        epochs=epochs,
        constraints=constraints,
        cuda=cuda
    )
    print("VAE training complete")
