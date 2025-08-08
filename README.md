# EWPT-GPU
## Single and Multi-GPU Julia based CUDA code to conduct 3+1 simulations of electroweak symmetry breaking

### The GPU kernel programs uses 6th-order finite differences to compute spatial derivatives, and Runge-Kutte 4th Order for temporal discretization.

## Multi-GPU: There are multi-GPU enabled scripts that use Julia MPI functions and have custom routines to enable domain division, boundary communications between GPU devices, and gathering global variables and outputs. 
The single GPU version has been functional on a wide range of CUDA-enabled GPUs.
Warning: The interplay between CUDA, MPI and Julia is a mess and needs to be properly built on the target cluster. Refer to notes.txt on various hacks and procedures. As of the latest commit, the multi gpu version of the code which once worked no longer does so on ASU Sol due to broken libraries.

## Axion coupling: Has axion coupling terms but have not been verified/tested.

## Initial conditions: Includes both bubble nucleation and thermalized initial conditions.
The results of the bubble nucleation initial conditions have been tested against results in
"Y. Zhang, T. Vachaspati and F. Ferrer, Phys. Rev. D 100, no.8, 083006 (2019)
doi:10.1103/PhysRevD.100.083006 [arXiv:1902.02751 [hep-ph]]."

The thermal initial conditions are based off
 "Z. G. Mou, P. M. Saffin and A. Tranberg, JHEP 06, 075 (2017)
doi:10.1007/JHEP06(2017)075 [arXiv:1704.08888 [hep-ph]]."

Both initialization modules are functional as of the latest commit.

### This numerical simulation codes leverage the high-level Julia coding language and CUDA kernels to conduct large lattice simulations at speeds much faster than traditional CPU based codes.

The advantages of Julia and GPU-based simulation codes have been highlighted in the talk: https://www.youtube.com/watch?v=Hz9IMJuW5hU

The GPU-based code here has been compared to an efficient single and multi-CPU Fortran code and the speedup is 10X-100X.
This, of course, depends on the number of CPU cores used and the GPU device. This speedup was observed using 125 CPU cores of modern AMD architecture for the CPU version and Nvidia A100 80 Gb GPUs on the SOL cluster at Arizona State University.
