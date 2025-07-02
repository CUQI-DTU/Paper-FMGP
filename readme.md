# Fast Gaussian Processes under Monotonicity constraints

#### [ArXiv](https://google.com)

## Introduction

We propose a new virtual point-based method to build Gaussian processes (GPs) under monotonicity constraints, leveraging the Regularized Linear Randomize-then-Optimize (RLRTO) method.

## Local setup
#### Environments

The codes were developed with Python 3.12.0, and the core libraries being used are as follows:

- CUQIpy 1.3.0
- CUQIpy-PyTorch 0.4.0
- Pyro 1.9.1
- GPJax 0.11.0
- PyTorch 2.6.0
- nbconvert 7.16.6

See [requirements.txt](requirements.txt) for complete environment specifics.

## Commands

### Experiments in Section 4
- For `dim={1,2}`, `case={1,2,3}` and `ns={4,8,16,32,64,128}`, the following command will run the notebook and save a copy of it at `/out`:

```bash
NB_ARGS=' --ns {ns} ' jupyter nbconvert --execute --to notebook demo_{dim}d_{case}.ipynb --output out/demo_{dim}d_{case}_{ns}.ipynb
```
- Then run `postprocess.ipynb` to generate Figures 5-9.

### Experiments in Section 5
To run the experiment on SIR:
- `NB_ARGS=' --ns 64 ' jupyter nbconvert --execute --to notebook demo_2d_sir.ipynb --output out/demo_2d_sir_64.ipynb`

To run the experiment on the Convection diffusion equation:
- `NB_ARGS=' --ns 128 ' jupyter nbconvert --execute --to notebook demo_3d_heat.ipynb --output out/demo_3d_heat_128.ipynb`
