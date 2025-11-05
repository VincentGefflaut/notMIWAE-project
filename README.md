# not-MIWAE

Code accompanying the paper
*Niels Bruun Ipsen, Pierre-Alexandre Mattei, and Jes Frellsen.
not-MIWAE: Deep generative modelling with missing not at random data.
arXiv preprint [arXiv:2006.12871 (2020).](https://arxiv.org/abs/2006.12871)*

Shows how to learn deep generative models with missing data under the MNAR assumption.
The notebook `not-MIWAE-demo.ipynb` introduces the model and training step by step.
`task01.py` runs the not-MIWAE and competitors on a UCI dataset.

## Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate notmiwae
```

## Running the Notebook

Start Jupyter and open the demo notebook:

```bash
jupyter notebook not-MIWAE-demo.ipynb
```

**Important:** If you've previously opened the notebook, restart the kernel before running (`Kernel` → `Restart Kernel`).

## Requirements

- Python 3.9
- TensorFlow 2.16.2 (with TF 1.x compatibility mode)
- See `environment.yml` for full dependencies
