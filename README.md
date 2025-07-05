# Diffusion Models for Improved Reconstruction of DWI

# Introduction
The research of the diffusion models are basically based on the CVPR 2023 paper "[Solving 3D Inverse Problems using Pre-trained 2D Diffusion Models](https://arxiv.org/abs/2211.10655)" and the github repository https://github.com/hyungjin-chung/DiffusionMBIR.

# Contributions
The application of Diffusion Models for the reconstruction of DWI
Implementing the trained models in the ONNX format and integrating the ZSSSL model within a Docker container. 

# Docker container 
ADMM algorithm utilizing the ResNet2D ZSSSL model.
ISMRMRD format.
siemens_to_ismrmrd package. 
Additional preprocessing steps, including regridding and removing oversampling

An internal training mechanism tailored for specific matrix sizes or diffusion directions 
Dataset with 21 directions, 114 slices, 32 coils, 200 x 200 matrix size: Total time of 20 minutes (using 1 GPU-A100)
Faster than Compressed Sensing
<img width="400" alt="docker ckt reuse explaination" src="https://github.com/user-attachments/assets/d5a85c3e-c546-4330-af37-d4078cb77bba" width="400"/>


# Stochastic Differential Equations(SDEs)
<img src=https://github.com/user-attachments/assets/928a4e07-4546-4cb1-be42-588c4df05649 width="400"/>

# Noise Conditioned Score Networks (NCSN)  
<img src=https://github.com/user-attachments/assets/d37a671c-18c6-4c36-91da-138108b6028e width="400"/>

# Inverse Reconstruction Algorithm
<img src=https://github.com/user-attachments/assets/3109cbec-ab59-43ad-b1ce-3f10ec885ba3 width="400"/>

# Model Training
1. Shepp Logan dataset <img src="gif1.gif" width="150"/>

<img src=https://github.com/user-attachments/assets/76bff4d0-e597-4393-a5c6-f9937004a723 width="400"/>

2. DWI dataset <img src="assets/demo.gif" width="150"/>

# Reconstruction Using Diffusion Models
<img src=https://github.com/user-attachments/assets/300abeae-fdb4-4ab1-ab2b-6ba19e92f868 width="400"/>
<img src=https://github.com/user-attachments/assets/ebced77c-315b-4236-84d9-767bb92c8b04 width="400"/>
<img src=https://github.com/user-attachments/assets/356d7f78-465f-4c20-9974-d339a996aca7 width="400"/>

# Reconstruction Using Diffusion Models: fastMRI dataset
<img alt="ZSSSL v s DM fastmri" src="https://github.com/user-attachments/assets/742c2ada-aa2a-4e68-9fd3-7cc8ed675e43" width="400"/>

# Model generalization: cross subject 
<img alt="container model inference" src="https://github.com/user-attachments/assets/ff10337c-1dd5-4ae6-ab63-1ab3722608a0" width="400"/>

* Make a conda environment and install dependencies
```bash
conda env create --file environment.yml
```

## Diffusion models reconstruction
Once you have the pre-trained weights and the test data set up properly, you may run the following scripts.
```bash
python run_solve_inverse_problem_simple.py
```

## Training
You may train the diffusion model with your own data by using e.g.
```bash
bash train_AAPM256.sh
```
You can modify the training config with the ```--config``` flag.


