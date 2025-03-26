# Accelerating the Computation of weights of Candidate Factors using C++, OpenMP and CUDA  

## Overview
A PLA is a two-level logic circuit--my program turns a PLA into an efficient and minimized multi-level logic circuit.

This project aims to accelerate the computation of weights of candidate factors in logic synthesis using C++, OpenMP, and CUDA. 
Parallelizing this computation is beneficial as it significantly reduces the runtime, especially in scenarios where timing goals are not as stringent, such as ASIC prototyping.
It is designed to handle large datasets efficiently.

Notable functions include makeClonesGpu(), fill_CountArray_Kernel(pla* plaGpup), and minimize() found in pla.cu.

I am not finished parallelzing everything yet.

## References  
1. Pla format: https://user.engineering.uiowa.edu/~switchin/OldSwitching/espresso.5.html

2. J. Vasudevamurthy and J. Rajski, "A method for concurrent decomposition and factorization of Boolean expressions," 1990 IEEE International Conference on Computer-Aided Design. Digest of Technical Papers, Santa Clara, CA, USA, 1990, pp. 510-513, doi: 10.1109/ICCAD.1990.129967. keywords: {Kernel;Network synthesis;Circuit synthesis;Logic functions;Laboratories;Automatic logic units;Logic circuits;Digital circuits;Microelectronics;Councils},
