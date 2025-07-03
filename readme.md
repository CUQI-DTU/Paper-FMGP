# Fast Gaussian Processes under Monotonicity constraints

#### [ArXiv](https://google.com)

## Introduction

We propose a new virtual point-based method to build Gaussian processes (GPs) under monotonicity constraints. Our method builds on the Regularized Linear Randomize-then-Optimize (RLRTO) framework and offers improved computational efficiency compared to existing approaches.

## Environments

The codes were developed with Python 3.12.0, and the core libraries being used are as follows:

- CUQIpy 1.3.0
- CUQIpy-PyTorch 0.4.0
- Pyro 1.9.1
- GPJax 0.11.0
- PyTorch 2.6.0

See [requirements.txt](requirements.txt) for full environment specifications.

## Commands

### Experiments in Section 4
- For `dim={1,2}`, `case={1,2,3}` and `ns={4,8,16,32,64,128}`, the following command will execute the notebook and save a copy of it at `/hpc_output`:

```bash
NB_ARGS=' --ns {ns} ' jupyter nbconvert --execute --to notebook demo_{dim}d_{case}.ipynb --output hpc_output/demo_{dim}d_{case}_{ns}.ipynb
```
- Then run `postprocess.ipynb` to generate Figures 5-9.

### Experiments in Section 5
To run the SIR experiment:
```bash
NB_ARGS=' --ns 64 ' jupyter nbconvert --execute --to notebook demo_2d_sir.ipynb --output hpc_output/demo_2d_sir_64.ipynb
```

To run the the convection diffusion equation experiment:
```bash
NB_ARGS=' --ns 128 ' jupyter nbconvert --execute --to notebook demo_3d_heat.ipynb --output hpc_output/demo_3d_convection_diffusion_128.ipynb
```
