# Implicit Neural Spatial Representations for Time-dependent PDEs

### [Project Page](https://www.cs.columbia.edu/cg/INSR-PDE/)  | [Paper](https://arxiv.org/abs/2210.00124)

<img src="https://github.com/honglin-c/INSR-PDE/blob/main/.github/images/teaser.png" width="500">

Official implementation for the paper:
> **[Implicit Neural Spatial Representations for Time-dependent PDEs](https://www.cs.columbia.edu/cg/INSR-PDE/)**  
> [Honglin Chen*](https://www.cs.columbia.edu/~honglinchen/)<sup>1</sup>, [Rundi Wu*](https://www.cs.columbia.edu/~rundi/)<sup>1</sup>, [Eitan Grinspun](https://www.dgp.toronto.edu/~eitan/)<sup>2</sup>, [Changxi Zheng](http://www.cs.columbia.edu/~cxz/)<sup>1</sup>, [Peter Yichen Chen](https://peterchencyc.com/)<sup>3 </sup><sup>1</sup> <br>
> <sup>1</sup>Columbia University, <sup>2</sup>University of Toronto, <sup>3</sup>Massachusetts Institute of Technology <br>
> ICML 2023


## Installation
Prerequisites:
- python 3.9+
- An NVIDIA GPU

Install dependencies with pip:
```bash
pip install -r requirements.txt
# NOTE: check https://pytorch.org/ for pytorch installation command for your CUDA version
```


## Experiments
Run each shell script under `scripts/` for the examples shown in the paper:
```bash
bash scripts/xxx.sh
```

For instance,
```bash
bash scripts/advect1D.sh
```

## Citation
```
@inproceedings{chenwu2023insr-pde,
    title={Implicit Neural Spatial Representations for Time-dependent PDEs},
    author={Honglin Chen and Rundi Wu and Eitan Grinspun and Changxi Zheng and Peter Yichen Chen},
    booktitle={International Conference on Machine Learning},
    year={2023}
}
```
