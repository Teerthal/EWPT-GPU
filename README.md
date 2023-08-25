# EWPT-GPU
## Multi-GPU Julia code to conduct 3D lattice simulations of electroweak phase transitions

### These numerical simulation codes leverage the high-level Julia coding language and CUDA kernels to conduct large lattice simulations at speeds much faster than traditional CPU based codes.
### There are 2 sets of simulation codes:
1. (IGG_gpu) One using 2nd-order finite difference schemes for spatial derivatives based on ParallelStencil and ImplicitGlobalGrid packages that enable mutli-xPU simulations. (https://github.com/omlins/ParallelStencil.jl & https://github.com/eth-cscs/ImplicitGlobalGrid.jl)
2. (gpu_kernels) One using 6th-order finite difference schemes for spatial derivatives using CUDA kernel programming.

The advantages of Julia and GPU-based simulation codes have been highlighted in the talk: https://www.youtube.com/watch?v=Hz9IMJuW5hU

The GPU-based code here has been compared to an efficient single and multi-CPU Fortran code and the speedup is 10X-100X.
This, of course, depends on the number of CPU cores used and the GPU device. This speedup was observed using 125 CPU cores of modern AMD architecture for the CPU version and Nvidia A100 80 Gb GPUs on the SOL cluster at Arizona State University.

# IGG_gpu
### This code is based on the convenient macros of the packages ImplicitGlobalGrid and ParallelStencil that divide the physical domain into multiple subdomains and initiate parallel computation. 
This is an extremely useful and convenient approach to almost trivially parallelize the simulation and extend it to GPU devices. 
It allows for an extension to multiple GPU and CPU-enabled simulation codes from a serial version with the use of very few macros.
### Limitation: The packages currently only allow for second-order central differences.

# gpu_kernels

### These are custom kernels programmed to simulate EWPT by solving the electroweak equations of motion using 
1. Runge-Kutte 4th Order
2. Runge-Kutte Fehlberg (RKF45)
3. Crank-Nicolson

### The GPU kernel programs uses 6th-order finite differences to compute spatial derivatives but can be easily extended to the nth order.

##Multi-GPU: There are multi-GPU enabled scripts that use Julia MPI functions and have custom routines to enable domain division, boundary communications between GPU devices, and gathering global variables and outputs. 
