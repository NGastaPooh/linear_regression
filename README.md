# Ising model: Revengeance

<p align="center">
  <img align="top" src="/convergence.gif" width="300" /> 
</p>

## Description

The goal of this project is to visualize the evolution of 2D ferromagnetic [Ising model][1] at given temperature with no external field. Assuming that each site in a 2D system is labeled with a pair of indices $` \left(k, k'\right) `$ and has a value $` \sigma_{kk'} = \{ +1, -1 \} `$, then the corresponding Hamiltonian of the system is:  

$$
H(\sigma) = - \sum_{\langle ij, lm\rangle} \sigma_{ij} \sigma_{lm} , 
$$  

where the sum is over pairs of adjacent spins (every pair is counted once). The notation $\langle ij, lm\rangle$ indicates that the sites 
$(i, j)$ and $(l, m)$ are nearest neighbors.  

The system is propagated according to [Metropolis Monte-Carlo algorithm][2].

As an additional feature, this particular realization of Ising model can recreate an image of Jetstream Sam (use $T = 1.69$), a character from a video game called [Metal Gear Rising: Revengeance][3], as if it was a natural equilibrium state of Ising model, by using a processed image as a reference.

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

