import argparse
import os
import sys
import time
import torch

from . import pipelines

SUPPORT_READ_EXT = {"fq", "fastq", "fa", "fasta"}
DEFAULT_THREADS = 8
DEFAULT_K_SIZE = 4
DEFAULT_BIN_SIZE = 12
DEFAULT_BIN_COUNT = 32
DEFAULT_AE_EPOCHS = 100
DEFAULT_AE_DIMS = 32
DEFAULT_AE_HIDDEN = "128,128"


def build_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""VAE Pretrain.""",
        add_help=True
    )
    parser.add_argument("--reads-path", "-r",
                       help="Reads path for binning",
                       type=str, required=True)
    parser.add_argument("--k-size", "-k",
                       help="k value for k-mer frequency vector. Choose between 3 and 5.",
                       type=int, required=False,
                       choices=[3, 4, 5], default=DEFAULT_K_SIZE)
    parser.add_argument("--bin-size", "-bs",
                       help="Bin size for the coverage histogram.",
                       type=int, required=False, default=DEFAULT_BIN_SIZE)
    parser.add_argument("--bin-count", "-bc",
                       help="Number of bins for the coverage histogram.",
                       type=int, required=False, default=DEFAULT_BIN_COUNT)
    parser.add_argument("--ae-epochs",
                       help="Epochs for the auto_encoder.",
                       type=int, required=False, default=DEFAULT_AE_EPOCHS)
    parser.add_argument("--ae-dims",
                       help="Size of the latent dimension.",
                       type=int, required=False, default=DEFAULT_AE_DIMS)
    parser.add_argument("--ae-hidden",
                       help="Hidden layer sizes eg: 128,128",
                       type=str, required=False, default=DEFAULT_AE_HIDDEN)
    parser.add_argument("--threads", "-t",
                       help="Thread count for computations",
                       type=int, default=DEFAULT_THREADS, required=False)
    parser.add_argument("--cuda",
                       action="store_true",
                       help="Whether to use CUDA if available.")
    parser.add_argument("--output", "-o", metavar="<DEST>",
                       help="Output directory", type=str, required=True)
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> None:
    reads_path = args.reads_path
    threads = args.threads
    output = args.output

    lower_path = reads_path.lower()
    valid = any(lower_path.endswith(ext) for ext in SUPPORT_READ_EXT)
    if not valid:
        print("Unable to detect file type of reads. Please use either FASTA or FASTQ.")
        sys.exit(1)

    if threads <= 0:
        print("Minimum number of threads is 1. Using thread count 1 and continue")
        args.threads = 1

    if not os.path.isfile(reads_path):
        print("Failed to open reads file")
        sys.exit(1)

    os.makedirs(os.path.join(output, "profiles"), exist_ok=True)


def check_cuda(enable_cuda_flag: bool) -> bool:
    if not enable_cuda_flag:
        return False
    if torch.cuda.is_available():
        print("CUDA found in system")
        return True
    print("CUDA not found in system")
    return False


def run_bin_pipeline(args: argparse.Namespace):
    pipelines.run_reads_binning(args)


def main():
    args = build_arg_parser()
    validate_inputs(args)
    start_time = time.time()
    print("Command " + " ".join(sys.argv))
    args.cuda = check_cuda(args.cuda)
    run_bin_pipeline(args)
    total_cost = time.time() - start_time
    print(f"Total time consumed = {total_cost:.2f} seconds")
    print("VAE Pretrain is finished")


if __name__ == "__main__":
    main()
