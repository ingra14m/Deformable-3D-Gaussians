# Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction

## [Project page](https://ingra14m.github.io/Deformable-3D-Gaussians/) | [Paper](https://arxiv.org/abs/2309.13101)

![Teaser image](assets/teaser.png)

This repository contains the official implementation associated with the paper "Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction". We will release our code as soon as possible.



## Dataset

In our paper, we use synthetic dataset from [D-NeRF](https://www.albertpumarola.com/research/D-NeRF/index.html) and real dataset from [Hyper-NeRF](https://hypernerf.github.io/). 

> I have identified an **inconsistency in the D-NeRF's Lego dataset**. Specifically, the scenes corresponding to the training set differ from those in the test set. This discrepancy can be verified by observing the angle of the flipped Lego shovel. To meaningfully evaluate the performance of our method on this dataset, I recommend using the **validation set of the Lego dataset** as the test set.



## Pipeline

![Teaser image](assets/pipeline.png)



## Results

### D-NeRF Dataset

 <img src="assets/results/D-NeRF/bouncing.gif" alt="Image1" style="zoom:25%;" />  <img src="assets/results/D-NeRF/hell.gif" alt="Image1" style="zoom:25%;" />  <img src="assets/results/D-NeRF/hook.gif" alt="Image3" style="zoom:25%;" />  <img src="assets/results/D-NeRF/jump.gif" alt="Image4" style="zoom:25%;" /> 

 <img src="assets/results/D-NeRF/lego.gif" alt="Image5" style="zoom:25%;" />  <img src="assets/results/D-NeRF/mutant.gif" alt="Image6" style="zoom:25%;" />  <img src="assets/results/D-NeRF/stand.gif" alt="Image7" style="zoom:25%;" />  <img src="assets/results/D-NeRF/trex.gif" alt="Image8" style="zoom:25%;" /> 

### HyperNeRF Dataset

A demo for the HyperNeRF dataset will be released shortly. I would be immensely grateful if someone could **provide the code displaying the HyperNeRF dataset** as showcased on the project pages of HyperNeRF or TiNeuVox.



## BibTex

```
@article{yang2023deformable3dgs,
    title={Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction},
    author={Yang, Ziyi and Gao, Xinyu and Zhou, Wen and Jiao, Shaohui and Zhang, Yuqing and Jin, Xiaogang},
    journal={arXiv preprint arXiv:2309.13101},
    year={2023}
}
```