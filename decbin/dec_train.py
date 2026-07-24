import argparse
from time import time
from .vae_dec import dec_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--read-path", "-r", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--latent-dims", "-ld", type=int, default=32)
    parser.add_argument("--hidden-layers", "-hl", type=str, default="128,128")
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--batch-size", "-bs", type=int, default=30000)
    parser.add_argument("--device", "-d", type=str, default="cuda:0")
    parser.add_argument("--nsp", "-n", type=int, default=None)

    args = parser.parse_args()

    hidden = list(map(int, args.hidden_layers.split(",")))
    start = time()
    dec_utils.train_dec(
        args.read_path,
        args.output,
        args.latent_dims,
        hidden,
        args.epochs,
        args.batch_size,
        device=args.device,
        nsp=args.nsp
    )
    end = time()
    print(f"Total running time: {round(end - start, 2)}s")


if __name__ == "__main__":
    main()
