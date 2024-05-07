

# Privacy-Onion-Effect

This repository contains an unofficial implementation (CIFAR-10) of the "Privacy Onion Effect" described in the research paper by Nicholas Carlini et al. The implementation is intended to help researchers and enthusiasts understand and explore the privacy implications detailed in the study.

## Citation

If you use this implementation for academic or research purposes, please cite the original paper as follows:

```bibtex
@inproceedings{NEURIPS2022_564b5f82,
	author = {Carlini, Nicholas and Jagielski, Matthew and Zhang, Chiyuan and Papernot, Nicolas and Terzis, Andreas and Tramer, Florian},
	booktitle = {Advances in Neural Information Processing Systems},
	editor = {S. Koyejo and S. Mohamed and A. Agarwal and D. Belgrave and K. Cho and A. Oh},
	pages = {13263--13276},
	publisher = {Curran Associates, Inc.},
	title = {The Privacy Onion Effect: Memorization is Relative},
	url = {https://proceedings.neurips.cc/paper_files/paper/2022/file/564b5f8289ba846ebc498417e834c253-Paper.pdf},
	volume = {35},
	year = {2022}
}
```

## Installation

To install the necessary environment for running this implementation, use the following command:

```bash
conda env create -f cifar/environment.yaml
```

## Pre-Configuration

You need to adjust the absolute paths in the following files to your specific absolute paths:

- `cifar/MIA/configs/config_attack.yaml`
- `cifar/MIA/scripts/static.py`
- `cifar/MIA/scripts/privacy_score.py`

## Run Experiment

1. **Generate Experiment Models**: Run the `cifar/train_cifar.sh` script to generate the experiment models.

2. **Generate Results**:
   - Navigate to the `MIA` directory to ensure the data addresses are correct.
   - Execute the `./run_all.py` script to:
     - Generate shadow models.
     - Obtain observations from the shadow datasets.
     - Compute Gaussian distribution parameters.
     - Generate attack results on the experiment models.
     - Produce privacy scores for each sample.




