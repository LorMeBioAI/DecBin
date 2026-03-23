import pickle
import os
import logging
import sys
from collections import defaultdict
from Bio import SeqIO
import numpy as np
import random

from mbcclr_utils.runners_utils import *
from vae_dec import ae_utils

logger = logging.getLogger('LRBinner')

def run_reads_binning(args):
    # commong arguments for reads binning and contig binning
    reads_path = args.reads_path
    threads = args.threads
    bin_size = args.bin_size
    bin_count = args.bin_count
    k_size = args.k_size
    epochs = args.ae_epochs
    dims = args.ae_dims
    hidden = list(map(int, args.ae_hidden.split(",")))
    separate = args.separate
    cuda = args.cuda
    resume = args.resume
    min_cluster_size = max(args.min_bin_size, 1)
    iterations = max(args.bin_iterations, 0)
    output = args.output

    checkpoints_path = f"{output}/checkpoints"

    if not resume:
        checkpoint = Checkpointer(checkpoints_path)
    else:
        logger.info("Resuming the program from previous checkpoints")
        checkpoint = Checkpointer(checkpoints_path, True)
        logger.debug(checkpoint)

    # computing k-mer vectors
    stage = "1_1"
    stage_params = [reads_path, k_size]

    if checkpoint.should_run_step(stage, stage_params):
        logger.info("Counting k-mers")
        run_kmers(reads_path, output, k_size, threads)
        
        checkpoint.log(stage, stage_params)
        logger.info("Counting k-mers complete")
    else:
        logger.info("K-mer vectors already computed")

    # counting 15-mers
    stage = "1_2"
    stage_params = [reads_path]

    if checkpoint.should_run_step(stage, stage_params):
        logger.info("Counting 15-mers")
        run_15mer_counts(reads_path, output, threads)
        
        checkpoint.log(stage, stage_params)
        logger.info("Counting 15-mers complete")
    else:
        logger.info("15-mers already counted")

    # computing coverage vectors
    stage = "2_1"
    stage_params = [reads_path, bin_size, bin_count]

    if checkpoint.should_run_step(stage, stage_params):
        logger.info("Computing 15-mer profiles")
        run_15mer_vecs(
            reads_path, output, bin_size, bin_count, threads)
        
        checkpoint.log(stage, stage_params)
        logger.info("Computing 15-mer profiles complete")
    else:
        logger.info("Already computed 15-mer profiles complete")


    # numpy vectors
    stage = "3_1"
    stage_params = ['numpy']

    if checkpoint.should_run_step(stage, stage_params):
        logger.info("Profiles saving as numpy arrays")
        comp_profiles = np.array([np.array(list(map(float, line.strip().split()))) for line in open(
            f"{output}/profiles/com_profs") if len(line.strip()) > 0])
        cov_profiles = np.array([np.array(list(map(float, line.strip().split()))) for line in open(
            f"{output}/profiles/cov_profs") if len(line.strip()) > 0])

        np.save(f"{output}/profiles/com_profs", comp_profiles)
        np.save(f"{output}/profiles/cov_profs", cov_profiles)

        del comp_profiles
        del cov_profiles

        logger.info("Profiles saving as numpy arrays complete")
        
        checkpoint.log(stage, stage_params)
    else:
        logger.info("Numpy arrays already computed")

    # VAE encode
    # TODO 
    constraints = None
    stage = "4_1"
    stage_params = [output,
                    dims,
                    hidden,
                    epochs,
                    constraints]

    if checkpoint.should_run_step(stage, stage_params):

        logger.info(f"VAE training information")
        logger.info(f"\tDimensions {dims}")
        logger.info(f"\tHidden Layers {hidden}")
        logger.info(f"\tEpochs {epochs}")

        ae_utils.train_vae(
            output,
            dims,
            hidden,
            epochs,
            constraints,
            cuda)

        checkpoint.log(stage, stage_params)
        logger.info(f"VAE training complete")
    else:
        logger.info(f"VAE already trained")

    # DEC
