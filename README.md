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
* tqdm
* matplotlib


## Quick Start
1. Installation

   You can install DecBin from pip. After installing Anaconda (or miniconda), first, obtain DecBin:  
   git clone [https://github.com/LorMeBioAI/DecBin.git](https://github.com/LorMeBioAI/DecBin.git)
   Then create an environment to run DecBin.  
```
cd path_to_DecBin
conda create -n decbin python=3.9
conda activate decbin
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install dist/decbin-0.0.1-py3-none-any.whl
pip install numpy==1.24
```
   
2. Data Preparation

   Place the Fastq/Fasta sequencing file to be analyzed into the data/ directory (example: data/SRR13128012.fastq).
   Three PacBio HiFi human intestinal datasets, SRR13128012-SRR13128014, were obtained from the National Center for Bio technology Information ([NCBI](https://www.ncbi.nlm.nih.gov)) under BioProject number PRJNA680590. NWC2 reads were obtained from the NCBI BioSample SAMN09580370 under the SRA accession codes SRX4451758 (Nanopore) and SRX4451757 (PacBio).

4. VAE Pretraining

```
decbin-pretrain -r data/SRR13128012.fastq -o data/SRR13128012 -k 4 -t 32
```
```
usage: decbin-pretrain [-h] --reads-path READS_PATH [--k-size {3,4,5}] [--ae-epochs AE_EPOCHS] [--ae-dims AE_DIMS] [--ae-hidden AE_HIDDEN] [--threads THREADS] [--cuda] --output <DEST>

optional arguments:
  -h, --help            show this help message and exit
  --reads-path READS_PATH, -r READS_PATH
                        Reads path for binning
  --k-size {3,4,5}, -k {3,4,5}
                        k value for k-mer frequency vector. Choose between 3 and 5.
  --ae-epochs AE_EPOCHS
                        Epochs for the auto_encoder.
  --ae-dims AE_DIMS     Size of the latent dimension.
  --ae-hidden AE_HIDDEN
                        Hidden layer sizes eg: 128,128
  --threads THREADS, -t THREADS
                        Thread count for computations
  --cuda                Whether to use CUDA if available.
  --output <DEST>, -o <DEST>
                        Output directory
```
4. DecBin Training
```
decbin-run -r data/SRR13128012.fastq -o data/SRR13128012 -e 100
```
```
usage: decbin-run [-h] --read-path READ_PATH --output OUTPUT [--latent-dims LATENT_DIMS] [--hidden-layers HIDDEN_LAYERS] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--device DEVICE] [--nsp NSP]

optional arguments:
  -h, --help            show this help message and exit
  --read-path READ_PATH, -r READ_PATH
                        Reads path for binning
  --output OUTPUT, -o OUTPUT
                        Output directory
  --latent-dims LATENT_DIMS, -ld LATENT_DIMS
                        Size of the latent dimension
  --hidden-layers HIDDEN_LAYERS, -hl HIDDEN_LAYERS
                        Hidden layer sizes default: 128,128
  --epochs EPOCHS, -e EPOCHS
                        Epochs for training
  --batch-size BATCH_SIZE, -bs BATCH_SIZE
                        Batch size for training
  --device DEVICE, -d DEVICE
                        Whether to use CUDA if available
  --nsp NSP, -n NSP
                        Estimated number of species, default: None
```
