# Logistic regression

<p align="center">
  <img align="top" src="/convergence.gif" width="300" /> 
</p>

## Description

The goal is to learn about linear regression as a part of Machine Learning toolkit. I follow Stanford CS229 lectures. This project consists of three parts:

- fit 1D training data using Gradient Descent algortyhm and Newton's algorythm
- Newton's algorythm to cover multidimentional traning data for a rgression task
- classification task with a defferent dataset 
[Ising model][1]  


## Model parameters

Fixed parameters:

- System size: 256x256  
- Number of Monte-Carlo steps: 10^6  

Command line prompt:

- Temperature $T$: any positive value, in reduced units ($k_B = 1$)
  - Use $T = 1.69$ to trigger a special Jetstream Sam simulation
- One of the proposed initial configurations:   
  - "COLD" - all spins point in the same direction  
  - "WARM" - spin directions are chosen randomly  

## Brief file descriptions

`examples/` - a directory with a few prerecorded animations in .gif format

`image_processing/sam.py` - a python script that converts an image into a 256x256 array suitable for usage in Ising model

`image_processing/sam.jpg` - a black-and-white picture of Jetstream Sam

**`Ising_model.py`** - the main python script that does all the project related tasks  

`sam.txt` - contains a 256x256 array of $` \{ +1, -1 \} `$ values obtained from a processed .jpg picture  

[1]: <https://en.wikipedia.org/wiki/Ising_model> "Ising model"
[2]: <https://en.wikipedia.org/wiki/Metropolis–Hastings_algorithm> "Metropolis algorithm"
[3]: <https://en.wikipedia.org/wiki/Metal_Gear_Rising:_Revengeance> "Metal Gear: Revengeance"

