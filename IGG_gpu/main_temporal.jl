####---Multiple CPU/GPU evolution code
####RK4 evolution code for solving the SU(2)xU(1) equations of motion in the temporal gauge
####using the second order Crank-Nicolson recipie for time steps
####Code is built on the framework of ImplicitGlobalGrid and ParallelStencil julia modules
####https://github.com/eth-cscs/ImplicitGlobalGrid.jl
####https://github.com/omlins/ParallelStencil.jl
####Spatial derivatives are limited to second order
####Periodic Bounadry conditions are implemented

include("parameters.jl")
using .parameters

include("bubble_gen_ini.jl")
using .bubbles
import Random
const USE_GPU = true
using ImplicitGlobalGrid
import MPI
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
using StatsBase
using Random
# using CUDA
using Plots
using NPZ
# using WriteVTK
using HDF5
# using CUDA.CUFFT
using CUDA
using FFTW

include("random_ini_gen.jl")
using .randomizer

include("macros.jl")
using .compute_macros
using Distributions

include("spec_routines.jl")
using .spec_convolver

#using Plots
#pythonplot()

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3);
else
    @init_parallel_stencil(Threads, Float64, 3);
end

@parallel function diffusion3D_step!(T2, T, Ci, lam, dt, dx, dy, dz)
    @inn(T2) = @inn(T) + dt*(lam*@inn(Ci)*(@d2_xi(T)/dx^2 + @d2_yi(T)/dy^2 + @d2_zi(T)/dz^2));
    return
end

#Issues with too many variables
#one solution would be to break the system into 2 pairs: 1 updates the fields f and the other updates the fdot
#This was one can reduce the number of variables passed to the kernel
#works after removing the unused variables: max tested was 79 that worked. did not work for 105
#need to look into cleaning up and removing the time components of the gauge fields from the code in 
# its entirety for the choice of gauge

#Initial condition setup in serial code is painfull slow
#Worked out a parallel implementation
#Seems to be fine. Need to test against analytical

@parallel function evolve_dot!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    Γ_1,Γ_2,Γ_3,Σ,
    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    dϕ_1_dt_n,dϕ_2_dt_n,dϕ_3_dt_n,dϕ_4_dt_n,
    dW_1_2_dt_n,dW_1_3_dt_n,dW_1_4_dt_n,
    dW_2_2_dt_n,dW_2_3_dt_n,dW_2_4_dt_n,
    dW_3_2_dt_n,dW_3_3_dt_n,dW_3_4_dt_n,
    dY_2_dt_n,dY_3_dt_n,dY_4_dt_n,
    gw,gy,gp2,lambda,vev,dx,dt)
    
    # @inn(test_arr) = @inn(dϕ_1_dt).+gw*@D_1ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    # W_1_1,W_1_2,W_1_3,W_1_4,
    # W_2_1,W_2_2,W_2_3,W_2_4,
    # W_3_1,W_3_2,W_3_3,W_3_4,
    # Y_1,Y_2,Y_3,Y_4)
    # @inn(ϕ_1_n) = @inn(ϕ_1) .+ dt.*(@d2_xi(ϕ_1) .+@d2_yi(ϕ_1) .+@d2_zi(ϕ_1))
    @inn(dϕ_1_dt_n) = @inn(dϕ_1_dt) .+ dt.*( @d2_xi(ϕ_1)./(dx*dx) .+@d2_yi(ϕ_1)./(dx*dx) .+@d2_zi(ϕ_1)./(dx*dx) .-
    0.5*gw.*((-@inn(W_1_2).*@d_xi(ϕ_4)./dx.-@inn(W_1_3).*@d_yi(ϕ_4)./dx.-@inn(W_1_4).*@d_zi(ϕ_4)./dx).-
    (-@inn(W_2_2).*@d_xi(ϕ_3)./dx.-@inn(W_2_3).*@d_yi(ϕ_3)./dx.-@inn(W_2_4).*@d_zi(ϕ_3)./dx).+
    (-@inn(W_3_2).*@d_xi(ϕ_2)./dx.-@inn(W_3_3).*@d_yi(ϕ_2)./dx.-@inn(W_3_4).*@d_zi(ϕ_2)./dx)).-
    0.5*gy.*(-@inn(Y_2).*@d_xi(ϕ_2)./dx.-@inn(Y_3).*@d_yi(ϕ_2)./dx.-@inn(Y_4).*@d_zi(ϕ_2)./dx).-
    0.5*gw.*((-@inn(W_1_2).*(@d_xi(ϕ_4)./dx .+@D_2ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_1_3).*(@d_yi(ϕ_4)./dx .+@D_3ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_1_4).*(@d_zi(ϕ_4)./dx .+@D_4ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
    (-@inn(W_2_2).*(@d_xi(ϕ_3)./dx .+@D_2ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_2_3).*(@d_yi(ϕ_3)./dx .+@D_3ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_2_4).*(@d_zi(ϕ_3)./dx .+@D_4ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).+
    (-@inn(W_3_2).*(@d_xi(ϕ_2)./dx .+@D_2ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_3_3).*(@d_yi(ϕ_2)./dx .+@D_3ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_3_4).*(@d_zi(ϕ_2)./dx .+@D_4ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)))).-
    0.5*gy.*(-@inn(Y_2).*(@d_xi(ϕ_2)./dx .+@D_2ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(Y_3).*(@d_yi(ϕ_2)./dx .+@D_3ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(Y_4).*(@d_zi(ϕ_2)./dx .+@D_4ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
    2.0*lambda.*((@inn(ϕ_1).*@inn(ϕ_1)) .+(@inn(ϕ_2).*@inn(ϕ_2)) .+(@inn(ϕ_3).*@inn(ϕ_3)) .+(@inn(ϕ_4).*@inn(ϕ_4)) .-vev*vev).*@inn(ϕ_1).+
    0.5 .*((gw.*@inn(Γ_3) .+gy.*@inn(Σ)).*@inn(ϕ_2).-gw.*@inn(Γ_2) .*@inn(ϕ_3).+gw.*@inn(Γ_1).*@inn(ϕ_4)))

    @inn(dϕ_2_dt_n) =@inn(dϕ_2_dt) .+ dt.*(@d2_xi(ϕ_2)./(dx*dx) .+@d2_yi(ϕ_2)./(dx*dx) .+@d2_zi(ϕ_2)./(dx*dx) .+
    0.5*gw.*((-@inn(W_1_2).*@d_xi(ϕ_3)./dx.-@inn(W_1_3).*@d_yi(ϕ_3)./dx.-@inn(W_1_4).*@d_zi(ϕ_3)./dx).+
    (-@inn(W_2_2).*@d_xi(ϕ_4)./dx.-@inn(W_2_3).*@d_yi(ϕ_4)./dx.-@inn(W_2_4).*@d_zi(ϕ_4)./dx).+
    (-@inn(W_3_2).*@d_xi(ϕ_1)./dx.-@inn(W_3_3).*@d_yi(ϕ_1)./dx.-@inn(W_3_4).*@d_zi(ϕ_1)./dx)).+
    0.5.*gy.*(-@inn(Y_2).*@d_xi(ϕ_1)./dx.-@inn(Y_3).*@d_yi(ϕ_1)./dx.-@inn(Y_4).*@d_zi(ϕ_1)./dx).+
    0.5.*gw.*((-@inn(W_1_2).*(@d_xi(ϕ_3)./dx .+@D_2ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_1_3).*(@d_yi(ϕ_3)./dx .+@D_3ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_1_4).*(@d_zi(ϕ_3)./dx .+@D_4ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).+
    (-@inn(W_2_2).*(@d_xi(ϕ_4)./dx .+@D_2ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_2_3).*(@d_yi(ϕ_4)./dx .+@D_3ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_2_4).*(@d_zi(ϕ_4)./dx .+@D_4ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).+
    (-@inn(W_3_2).*(@d_xi(ϕ_1)./dx .+@D_2ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_3_3).*(@d_yi(ϕ_1)./dx .+@D_3ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_3_4).*(@d_zi(ϕ_1)./dx .+@D_4ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)))).+
    0.5.*gy.*(-@inn(Y_2).*(@d_xi(ϕ_1)./dx .+@D_2ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(Y_3).*(@d_yi(ϕ_1)./dx .+@D_3ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(Y_4).*(@d_zi(ϕ_1)./dx .+@D_4ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
    2.0.*lambda.*((@inn(ϕ_1).*@inn(ϕ_1)) .+(@inn(ϕ_2).*@inn(ϕ_2)) .+(@inn(ϕ_3).*@inn(ϕ_3)) .+(@inn(ϕ_4).*@inn(ϕ_4)) .-vev*vev)*@inn(ϕ_2).-
    0.5 .*((gw*@inn(Γ_3) .+gy.*@inn(Σ)) .*@inn(ϕ_1) .+gw .*@inn(Γ_1).*@inn(ϕ_3) .+gw.*@inn(Γ_2) .*@inn(ϕ_4)))

    @inn(dϕ_3_dt_n) =@inn(dϕ_3_dt) .+ dt.*(@d2_xi(ϕ_3)./(dx*dx) .+@d2_yi(ϕ_3)./(dx*dx) .+@d2_zi(ϕ_3)./(dx*dx) .-
    0.5*gw.*((-@inn(W_1_2).*@d_xi(ϕ_2)./dx.-@inn(W_1_3).*@d_yi(ϕ_2)./dx.-@inn(W_1_4).*@d_zi(ϕ_2)./dx).+
    (-@inn(W_2_2).*@d_xi(ϕ_1)./dx.-@inn(W_2_3).*@d_yi(ϕ_1)./dx.-@inn(W_2_4).*@d_zi(ϕ_1)./dx).-
    (-@inn(W_3_2).*@d_xi(ϕ_4)./dx.-@inn(W_3_3).*@d_yi(ϕ_4)./dx.-@inn(W_3_4).*@d_zi(ϕ_4)./dx)).-
    0.5.*gy.*(-@inn(Y_2).*@d_xi(ϕ_4)./dx.-@inn(Y_3).*@d_yi(ϕ_4)./dx.-@inn(Y_4).*@d_zi(ϕ_4)./dx).-
    0.5.*gw.*((-@inn(W_1_2).*(@d_xi(ϕ_2)./dx .+@D_2ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_1_3).*(@d_yi(ϕ_2)./dx .+@D_3ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_1_4).*(@d_zi(ϕ_2)./dx .+@D_4ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).+
    (-@inn(W_2_2).*(@d_xi(ϕ_1)./dx .+@D_2ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_2_3).*(@d_yi(ϕ_1)./dx .+@D_3ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_2_4).*(@d_zi(ϕ_1)./dx .+@D_4ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
    (-@inn(W_3_2).*(@d_xi(ϕ_4)./dx .+@D_2ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_3_3).*(@d_yi(ϕ_4)./dx .+@D_3ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_3_4).*(@d_zi(ϕ_4)./dx .+@D_4ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)))).-
    0.5.*gy.*(-@inn(Y_2).*(@d_xi(ϕ_4)./dx .+@D_2ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(Y_3).*(@d_yi(ϕ_4)./dx .+@D_3ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(Y_4).*(@d_zi(ϕ_4)./dx .+@D_4ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
    2.0.*lambda.*((@inn(ϕ_1).*@inn(ϕ_1)) .+(@inn(ϕ_2).*@inn(ϕ_2)) .+(@inn(ϕ_3).*@inn(ϕ_3)) .+(@inn(ϕ_4).*@inn(ϕ_4)) .-vev*vev).*@inn(ϕ_3).+
    0.5 .*((-gw.*@inn(Γ_3) .+gy*@inn(Σ))*@inn(ϕ_4).+gw .*@inn(Γ_2) .*@inn(ϕ_1).+gw .*@inn(Γ_1) .*@inn(ϕ_2)))

    @inn(dϕ_4_dt_n) =@inn(dϕ_4_dt) .+ dt.*(@d2_xi(ϕ_4)./(dx*dx) .+@d2_yi(ϕ_4)./(dx*dx) .+@d2_zi(ϕ_4)./(dx*dx) .+
    0.5.*gw.*((-@inn(W_1_2).*@d_xi(ϕ_1)./dx.-@inn(W_1_3).*@d_yi(ϕ_1)./dx.-@inn(W_1_4).*@d_zi(ϕ_1)./dx).-
    (-@inn(W_2_2).*@d_xi(ϕ_2)./dx.-@inn(W_2_3).*@d_yi(ϕ_2)./dx.-@inn(W_2_4).*@d_zi(ϕ_2)./dx).-
    (-@inn(W_3_2).*@d_xi(ϕ_3)./dx.-@inn(W_3_3).*@d_yi(ϕ_3)./dx.-@inn(W_3_4).*@d_zi(ϕ_3)./dx)).+
    0.5.*gy.*(-@inn(Y_2).*@d_xi(ϕ_3)./dx.-@inn(Y_3).*@d_yi(ϕ_3)./dx.-@inn(Y_4).*@d_zi(ϕ_3)./dx).+
    0.5.*gw.*((-@inn(W_1_2).*(@d_xi(ϕ_1)./dx .+@D_2ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_1_3).*(@d_yi(ϕ_1)./dx .+@D_3ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_1_4).*(@d_zi(ϕ_1)./dx .+@D_4ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
    (-@inn(W_2_2).*(@d_xi(ϕ_2)./dx .+@D_2ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_2_3).*(@d_yi(ϕ_2)./dx .+@D_3ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_2_4).*(@d_zi(ϕ_2)./dx .+@D_4ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
    (-@inn(W_3_2).*(@d_xi(ϕ_3)./dx .+@D_2ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_3_3).*(@d_yi(ϕ_3)./dx .+@D_3ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_3_4).*(@d_zi(ϕ_3)./dx .+@D_4ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)))).+
    0.5.*gy.*(-@inn(Y_2).*(@d_xi(ϕ_3)./dx .+@D_2ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(Y_3).*(@d_yi(ϕ_3)./dx .+@D_3ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(Y_4).*(@d_zi(ϕ_3)./dx .+@D_4ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
    2.0.*lambda.*((@inn(ϕ_1).*@inn(ϕ_1)) .+(@inn(ϕ_2).*@inn(ϕ_2)) .+(@inn(ϕ_3).*@inn(ϕ_3)) .+(@inn(ϕ_4).*@inn(ϕ_4)) .-vev.*vev).*@inn(ϕ_4).-
    0.5 .*((-gw .*@inn(Γ_3) .+gy.*@inn(Σ)).*@inn(ϕ_3).+gw.*@inn(Γ_1) .*@inn(ϕ_1).-gw.*@inn(Γ_2) .*@inn(ϕ_2)))

    # c W.-fluxes:
    # c
    # r(W_1_1)=0.

    # r(W_1_2)= 
    @inn(dW_1_2_dt_n) =@inn(dW_1_2_dt).+dt.*(@d2_xi(W_1_2)./(dx*dx) .+@d2_yi(W_1_2)./(dx*dx) .+@d2_zi(W_1_2)./(dx*dx) .+gw.*( 
        -(@d_xi(W_2_2)./dx.*@inn(W_3_2).-@d_xi(W_3_2)./dx.*@inn(W_2_2)).-
        (@d_yi(W_2_2)./dx.*@inn(W_3_3).-@d_yi(W_3_2)./dx.*@inn(W_2_3)).-
        (@d_zi(W_2_2)./dx.*@inn(W_3_4).-@d_zi(W_3_2)./dx.*@inn(W_2_4)).-
        (@inn(W_2_3).*( @d_xi(W_3_3)./dx .- @d_yi(W_3_2)./dx .+ gw.*( @inn(W_1_2) .*@inn(W_2_3) .- @inn(W_1_3).*@inn(W_2_2) )).-#fs(3,2,3)
        @inn(W_3_3).*( @d_xi(W_2_3)./dx .-@d_yi(W_2_2)./dx .+gw.*( @inn(W_3_2) .*@inn(W_1_3) .- @inn(W_3_3).*@inn(W_1_2) ))).-#fs(2,2,3)
        (@inn(W_2_4).*( @d_xi(W_3_4)./dx .-@d_zi(W_3_2)./dx .+gw.*( @inn(W_1_2) .*@inn(W_2_4) .- @inn(W_1_4).*@inn(W_2_2) )).-#fs(3,2,4)
        @inn(W_3_4).*( @d_xi(W_2_4)./dx .-@d_zi(W_2_2)./dx .+gw.*( @inn(W_3_2) .*@inn(W_1_4) .- @inn(W_3_4).*@inn(W_1_2) )))#fs(2,2,4)
        ).+
        gw.*(@inn(ϕ_1).*(@d_xi(ϕ_4)./dx .+@D_2ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
        @inn(ϕ_2).*(@d_xi(ϕ_3)./dx .+@D_2ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
        @inn(ϕ_3).*(@d_xi(ϕ_2)./dx .+@D_2ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
        @inn(ϕ_4).*(@d_xi(ϕ_1)./dx .+@D_2ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
        @d_xi(Γ_1)./dx .-gw.*(@inn(W_2_2).*@inn(Γ_3).-@inn(W_3_2).*@inn(Γ_2)))

    # r(W_1_3)=
    @inn(dW_1_3_dt_n) =@inn(dW_1_3_dt).+dt.*(@d2_xi(W_1_3)./(dx*dx) .+@d2_yi(W_1_3)./(dx*dx) .+@d2_zi(W_1_3)./(dx*dx) .+gw.*( 
        -(@d_xi(W_2_3)./dx.*@inn(W_3_2).-@d_xi(W_3_3)./dx.*@inn(W_2_2)).-
        (@d_yi(W_2_3)./dx.*@inn(W_3_3).-@d_yi(W_3_3)./dx.*@inn(W_2_3)).-
        (@d_zi(W_2_3)./dx.*@inn(W_3_4).-@d_zi(W_3_3)./dx.*@inn(W_2_4)).-
        (@inn(W_2_2).*-( @d_xi(W_3_3)./dx .- @d_yi(W_3_2)./dx .+ gw.*( @inn(W_1_2) .*@inn(W_2_3) .- @inn(W_1_3).*@inn(W_2_2) )).-#fs(3,3,2)
        @inn(W_3_2).*-( @d_xi(W_2_3)./dx .-@d_yi(W_2_2)./dx .+gw.*( @inn(W_3_2) .*@inn(W_1_3) .- @inn(W_3_3).*@inn(W_1_2) ))).-#fs(2,3,2)
        (@inn(W_2_4).*( @d_yi(W_3_4)./dx .-@d_zi(W_3_3)./dx .+gw.*( @inn(W_1_3) .*@inn(W_2_4) .- @inn(W_1_4).*@inn(W_2_3) )).-#fs(3,3,4)
        @inn(W_3_4).*( @d_yi(W_2_4)./dx .-@d_zi(W_2_3)./dx .+gw.*( @inn(W_3_3) .*@inn(W_1_4) .- @inn(W_3_4).*@inn(W_1_3) )))#fs(2,3,4)
        ).+
        gw.*(@inn(ϕ_1).*(@d_yi(ϕ_4)./dx .+@D_3ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
        @inn(ϕ_2).*(@d_yi(ϕ_3)./dx .+@D_3ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
        @inn(ϕ_3).*(@d_yi(ϕ_2)./dx .+@D_3ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
        @inn(ϕ_4).*(@d_yi(ϕ_1)./dx .+@D_3ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
        @d_yi(Γ_1)./dx .-gw.*(@inn(W_2_3).*@inn(Γ_3).-@inn(W_3_3).*@inn(Γ_2)))

    # r(W_1_4)=
    @inn(dW_1_4_dt_n) =@inn(dW_1_4_dt).+dt.*(@d2_xi(W_1_4)./(dx*dx) .+@d2_yi(W_1_4)./(dx*dx) .+@d2_zi(W_1_4)./(dx*dx) .+gw.*( 
        -(@d_xi(W_2_4)./dx.*@inn(W_3_2).-@d_xi(W_3_4)./dx.*@inn(W_2_2)).-
        (@d_yi(W_2_4)./dx.*@inn(W_3_3).-@d_yi(W_3_4)./dx.*@inn(W_2_3)).-
        (@d_zi(W_2_4)./dx.*@inn(W_3_4).-@d_zi(W_3_4)./dx.*@inn(W_2_4)).-
        (@inn(W_2_2).*-( @d_xi(W_3_4)./dx .-@d_zi(W_3_2)./dx .+gw.*( @inn(W_1_2) .*@inn(W_2_4) .- @inn(W_1_4).*@inn(W_2_2) )).-#fs(3,4,2)
        @inn(W_3_2).*-( @d_xi(W_2_4)./dx .-@d_zi(W_2_2)./dx .+gw.*( @inn(W_3_2) .*@inn(W_1_4) .- @inn(W_3_4).*@inn(W_1_2) ))).-#fs(2,4,2)
        (@inn(W_2_3).*-( @d_yi(W_3_4)./dx .-@d_zi(W_3_3)./dx .+gw.*( @inn(W_1_3) .*@inn(W_2_4) .- @inn(W_1_4).*@inn(W_2_3) )).-#fs(3,4,3)
        @inn(W_3_3).*-( @d_yi(W_2_4)./dx .-@d_zi(W_2_3)./dx .+gw.*( @inn(W_3_3) .*@inn(W_1_4) .- @inn(W_3_4).*@inn(W_1_3) )))#fs(2,4,3)
        ).+
        gw.*(@inn(ϕ_1).*(@d_zi(ϕ_4)./dx .+@D_4ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
        @inn(ϕ_2).*(@d_zi(ϕ_3)./dx .+@D_4ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
        @inn(ϕ_3).*(@d_zi(ϕ_2)./dx .+@D_4ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
        @inn(ϕ_4).*(@d_zi(ϕ_1)./dx .+@D_4ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
        @d_zi(Γ_1)./dx .-gw.*(@inn(W_2_4).*@inn(Γ_3).-@inn(W_3_4).*@inn(Γ_2)))#.-


    # r(W_2_1)=0.

    # r(W_2_2)=
    @inn(dW_2_2_dt_n) =@inn(dW_2_2_dt).+dt.*(@d2_xi(W_2_2)./(dx*dx) .+@d2_yi(W_2_2)./(dx*dx) .+@d2_zi(W_2_2)./(dx*dx) .+gw.*( 
        -(@d_xi(W_3_2)./dx.*@inn(W_1_2).-@d_xi(W_1_2)./dx.*@inn(W_3_2)).-
        (@d_yi(W_3_2)./dx.*@inn(W_1_3).-@d_yi(W_1_2)./dx.*@inn(W_3_3)).-
        (@d_zi(W_3_2)./dx.*@inn(W_1_4).-@d_zi(W_1_2)./dx.*@inn(W_3_4)).-
        (@inn(W_3_3).*( @d_xi(W_1_3)./dx .-@d_yi(W_1_2)./dx .+gw.*( @inn(W_2_2) .*@inn(W_3_3) .- @inn(W_2_3).*@inn(W_3_2) )).-#fs(1,2,3)
        @inn(W_1_3).*( @d_xi(W_3_3)./dx .-@d_yi(W_3_2)./dx .+gw.*( @inn(W_1_2) .*@inn(W_2_3) .- @inn(W_1_3).*@inn(W_2_2) ))).-#fs(3,2,3)
        (@inn(W_3_4).*( @d_xi(W_1_4)./dx .-@d_zi(W_1_2)./dx .+gw.*( @inn(W_2_2) .*@inn(W_3_4) .- @inn(W_2_4).*@inn(W_3_2) )).-#fs(1,2,4)
        @inn(W_1_4).*( @d_xi(W_3_4)./dx .-@d_zi(W_3_2)./dx .+gw.*( @inn(W_1_2) .*@inn(W_2_4) .- @inn(W_1_4).*@inn(W_2_2) )))#fs(3,2,4)
        ).+
        gw.*(-@inn(ϕ_1).*(@d_xi(ϕ_3)./dx .+@D_2ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
        @inn(ϕ_2).*(@d_xi(ϕ_4)./dx .+@D_2ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
        @inn(ϕ_3).*(@d_xi(ϕ_1)./dx .+@D_2ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
        @inn(ϕ_4).*(@d_xi(ϕ_2)./dx .+@D_2ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
        @d_xi(Γ_2)./dx .-gw.*(@inn(W_3_2).*@inn(Γ_1).-@inn(W_1_2).*@inn(Γ_3)))#.-


    # r(W_2_3)=
    @inn(dW_2_3_dt_n) =@inn(dW_2_3_dt).+dt.*(@d2_xi(W_2_3)./(dx*dx) .+@d2_yi(W_2_3)./(dx*dx) .+@d2_zi(W_2_3)./(dx*dx) .+gw.*( 
        -(@d_xi(W_3_3)./dx.*@inn(W_1_2).-@d_xi(W_1_3)./dx.*@inn(W_3_2)).-
        (@d_yi(W_3_3)./dx.*@inn(W_1_3).-@d_yi(W_1_3)./dx.*@inn(W_3_3)).-
        (@d_zi(W_3_3)./dx.*@inn(W_1_4).-@d_zi(W_1_3)./dx.*@inn(W_3_4)).-
        (@inn(W_3_2).*-( @d_xi(W_1_3)./dx .-@d_yi(W_1_2)./dx .+gw.*( @inn(W_2_2) .*@inn(W_3_3) .- @inn(W_2_3).*@inn(W_3_2) )).-#fs(1,3,2)
        @inn(W_1_2).*-( @d_xi(W_3_3)./dx .- @d_yi(W_3_2)./dx .+ gw.*( @inn(W_1_2) .*@inn(W_2_3) .- @inn(W_1_3).*@inn(W_2_2) ))).-#fs(3,3,2)
        (@inn(W_3_4).*( @d_yi(W_1_4)./dx .-@d_zi(W_1_3)./dx .+gw.*( @inn(W_2_3) .*@inn(W_3_4) .- @inn(W_2_4).*@inn(W_3_3) )).-#fs(1,3,4)
        @inn(W_1_4).*( @d_yi(W_3_4)./dx .-@d_zi(W_3_3)./dx .+gw.*( @inn(W_1_3) .*@inn(W_2_4) .- @inn(W_1_4).*@inn(W_2_3) )))#fs(3,3,4)
        ).+
        gw.*(-@inn(ϕ_1).*(@d_yi(ϕ_3)./dx .+@D_3ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
        @inn(ϕ_2).*(@d_yi(ϕ_4)./dx .+@D_3ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
        @inn(ϕ_3).*(@d_yi(ϕ_1)./dx .+@D_3ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
        @inn(ϕ_4).*(@d_yi(ϕ_2)./dx .+@D_3ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
        @d_yi(Γ_2)./dx .-gw.*(@inn(W_3_3).*@inn(Γ_1).-@inn(W_1_3).*@inn(Γ_3)))#.-

    # r(W_2_4)=
    @inn(dW_2_4_dt_n) =@inn(dW_2_4_dt).+dt.*(@d2_xi(W_2_4)./(dx*dx) .+@d2_yi(W_2_4)./(dx*dx) .+@d2_zi(W_2_4)./(dx*dx) .+gw.*( 
        -(@d_xi(W_3_4)./dx.*@inn(W_1_2).-@d_xi(W_1_4)./dx.*@inn(W_3_2)).-
        (@d_yi(W_3_4)./dx.*@inn(W_1_3).-@d_yi(W_1_4)./dx.*@inn(W_3_3)).-
        (@d_zi(W_3_4)./dx.*@inn(W_1_4).-@d_zi(W_1_4)./dx.*@inn(W_3_4)).-
        (@inn(W_3_2).*-( @d_xi(W_1_4)./dx .-@d_zi(W_1_2)./dx .+gw.*( @inn(W_2_2) .*@inn(W_3_4) .- @inn(W_2_4).*@inn(W_3_2) )).-#fs(1,4,2)
        @inn(W_1_2).*-( @d_xi(W_3_4)./dx .-@d_zi(W_3_2)./dx .+gw.*( @inn(W_1_2) .*@inn(W_2_4) .- @inn(W_1_4).*@inn(W_2_2) ))).-#fs(3,4,2)
        (@inn(W_3_3).*-( @d_yi(W_1_4)./dx .-@d_zi(W_1_3)./dx .+gw.*( @inn(W_2_3) .*@inn(W_3_4) .- @inn(W_2_4).*@inn(W_3_3) )).-#fs(1,4,3)
        @inn(W_1_3).*-( @d_yi(W_3_4)./dx .-@d_zi(W_3_3)./dx .+gw.*( @inn(W_1_3) .*@inn(W_2_4) .- @inn(W_1_4).*@inn(W_2_3) )))#fs(3,4,3)
        ).+
        gw.*(-@inn(ϕ_1).*(@d_zi(ϕ_3)./dx .+@D_4ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
        @inn(ϕ_2).*(@d_zi(ϕ_4)./dx .+@D_4ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
        @inn(ϕ_3).*(@d_zi(ϕ_1)./dx .+@D_4ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
        @inn(ϕ_4).*(@d_zi(ϕ_2)./dx .+@D_4ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
        @d_zi(Γ_2)./dx .-gw.*(@inn(W_3_4).*@inn(Γ_1).-@inn(W_1_4).*@inn(Γ_3)))#.-


    # r(W_3_1)=0.

    # r(W_3_2)=
    @inn(dW_3_2_dt_n) =@inn(dW_3_2_dt).+dt.*(@d2_xi(W_3_2)./(dx*dx) .+@d2_yi(W_3_2)./(dx*dx) .+@d2_zi(W_3_2)./(dx*dx) .+gw.*( 
        -(@d_xi(W_1_2).*@inn(W_2_2)./dx.-@d_xi(W_2_2)./dx.*@inn(W_1_2)).-
        (@d_yi(W_1_2).*@inn(W_2_3)./dx.-@d_yi(W_2_2)./dx.*@inn(W_1_3)).-
        (@d_zi(W_1_2).*@inn(W_2_4)./dx.-@d_zi(W_2_2)./dx.*@inn(W_1_4)).-
        (@inn(W_1_3).*( @d_xi(W_2_3)./dx .-@d_yi(W_2_2)./dx .+gw.*( @inn(W_3_2) .*@inn(W_1_3) .- @inn(W_3_3).*@inn(W_1_2) )).-#fs(2,2,3)
        @inn(W_2_3).*( @d_xi(W_1_3)./dx .-@d_yi(W_1_2)./dx .+gw.*( @inn(W_2_2) .*@inn(W_3_3) .- @inn(W_2_3).*@inn(W_3_2) ))).-#fs(1,2,3)
        (@inn(W_1_4).*( @d_xi(W_2_4)./dx .-@d_zi(W_2_2)./dx .+gw.*( @inn(W_3_2) .*@inn(W_1_4) .- @inn(W_3_4).*@inn(W_1_2) )).-#fs(2,2,4)
        @inn(W_2_4).*( @d_xi(W_1_4)./dx .-@d_zi(W_1_2)./dx .+gw.*( @inn(W_2_2) .*@inn(W_3_4) .- @inn(W_2_4).*@inn(W_3_2) )))#fs(3,2,4)
        ).+
        gw.*(@inn(ϕ_1).*(@d_xi(ϕ_2)./dx .+@D_2ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
        @inn(ϕ_2).*(@d_xi(ϕ_1)./dx .+@D_2ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
        @inn(ϕ_3).*(@d_xi(ϕ_4)./dx .+@D_2ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
        @inn(ϕ_4).*(@d_xi(ϕ_3)./dx .+@D_2ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
        @d_xi(Γ_3)./dx .-gw.*(@inn(W_1_2).*@inn(Γ_2).-@inn(W_2_2).*@inn(Γ_1)))#.-


    # r(W_3_3)=
    @inn(dW_3_3_dt_n) =@inn(dW_3_3_dt).+dt.*(@d2_xi(W_3_3)./(dx*dx) .+@d2_yi(W_3_3)./(dx*dx) .+@d2_zi(W_3_3)./(dx*dx) .+gw.*( 
        -(@d_xi(W_1_3)./dx.*@inn(W_2_2)./dx.-@d_xi(W_2_3)./dx.*@inn(W_1_2)).-
        (@d_yi(W_1_3)./dx.*@inn(W_2_3)./dx.-@d_yi(W_2_3)./dx.*@inn(W_1_3)).-
        (@d_zi(W_1_3)./dx.*@inn(W_2_4)./dx.-@d_zi(W_2_3)./dx.*@inn(W_1_4)).-
        (@inn(W_1_2).*-( @d_xi(W_2_3)./dx .-@d_yi(W_2_2)./dx .+gw.*( @inn(W_3_2) .*@inn(W_1_3) .- @inn(W_3_3).*@inn(W_1_2) )).-#fs(2,3,2)
        @inn(W_2_2).*-( @d_xi(W_1_3)./dx .-@d_yi(W_1_2)./dx .+gw.*( @inn(W_2_2) .*@inn(W_3_3) .- @inn(W_2_3).*@inn(W_3_2) ))).-#fs(1,3,2)
        (@inn(W_1_4).*( @d_yi(W_2_4)./dx .-@d_zi(W_2_3)./dx .+gw.*( @inn(W_3_3) .*@inn(W_1_4) .- @inn(W_3_4).*@inn(W_1_3) )).-#fs(2,3,4)
        @inn(W_2_4).*( @d_yi(W_1_4)./dx .-@d_zi(W_1_3)./dx .+gw.*( @inn(W_2_3) .*@inn(W_3_4) .- @inn(W_2_4).*@inn(W_3_3) )))#fs(1,3,4)
        ).+
        gw.*(@inn(ϕ_1).*(@d_yi(ϕ_2)./dx .+@D_3ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
        @inn(ϕ_2).*(@d_yi(ϕ_1)./dx .+@D_3ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
        @inn(ϕ_3).*(@d_yi(ϕ_4)./dx .+@D_3ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
        @inn(ϕ_4).*(@d_yi(ϕ_3)./dx .+@D_3ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
        @d_yi(Γ_3)./dx .-gw.*(@inn(W_1_3).*@inn(Γ_2).-@inn(W_2_3).*@inn(Γ_1)))#.-


    # r(W_3_4)=
    @inn(dW_3_4_dt_n) =@inn(dW_3_4_dt).+dt.*(@d2_xi(W_3_4)./(dx*dx) .+@d2_yi(W_3_4)./(dx*dx) .+@d2_zi(W_3_4)./(dx*dx) .+gw.*( 
        -(@d_xi(W_1_4)./dx.*@inn(W_2_2).-@d_xi(W_2_4)./dx.*@inn(W_1_2)).-
        (@d_yi(W_1_4)./dx.*@inn(W_2_3).-@d_yi(W_2_4)./dx.*@inn(W_1_3)).-
        (@d_zi(W_1_4)./dx.*@inn(W_2_4).-@d_zi(W_2_4)./dx.*@inn(W_1_4)).-
        (@inn(W_1_2).*-( @d_xi(W_2_4)./dx .-@d_zi(W_2_2)./dx .+gw.*( @inn(W_3_2) .*@inn(W_1_4) .- @inn(W_3_4).*@inn(W_1_2) )).-#fs(2,4,2)
        @inn(W_2_2).*-( @d_xi(W_1_4)./dx .-@d_zi(W_1_2)./dx .+gw.*( @inn(W_2_2) .*@inn(W_3_4) .- @inn(W_2_4).*@inn(W_3_2) ))).-#fs(1,4,2)
        (@inn(W_1_3).*-( @d_yi(W_2_4)./dx .-@d_zi(W_2_3)./dx .+gw.*( @inn(W_3_3) .*@inn(W_1_4) .- @inn(W_3_4).*@inn(W_1_3) )).-#fs(2,4,3)
        @inn(W_2_3).*-( @d_yi(W_1_4)./dx .-@d_zi(W_1_3)./dx .+gw.*( @inn(W_2_3) .*@inn(W_3_4) .- @inn(W_2_4).*@inn(W_3_3) )))#fs(1,4,3)
        ).+
        gw.*(@inn(ϕ_1).*(@d_zi(ϕ_2)./dx .+@D_4ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
        @inn(ϕ_2).*(@d_zi(ϕ_1)./dx .+@D_4ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
        @inn(ϕ_3).*(@d_zi(ϕ_4)./dx .+@D_4ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
        @inn(ϕ_4).*(@d_zi(ϕ_3)./dx .+@D_4ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
        @d_zi(Γ_3)./dx .-gw.*(@inn(W_1_4).*@inn(Γ_2).-@inn(W_2_4).*@inn(Γ_1)))#.-


    # c Y.-fluxes:

    # r(Y_1)=0.

    #r(Y_2)=
    @inn(dY_2_dt_n) =@inn(dY_2_dt).+dt.*(@d2_xi(Y_2)./(dx*dx) .+@d2_yi(Y_2)./(dx*dx) .+@d2_zi(Y_2)./(dx*dx) .+
    gy.*(@inn(ϕ_1).*(@d_xi(ϕ_2)./dx .+@D_2ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(ϕ_2).*(@d_xi(ϕ_1)./dx .+@D_2ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
    @inn(ϕ_3).*(@d_xi(ϕ_4)./dx .+@D_2ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(ϕ_4).*(@d_xi(ϕ_3)./dx .+@D_2ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
    @d_xi(Σ))

    # r[19]=
    @inn(dY_3_dt_n) =@inn(dY_3_dt).+dt.*(@d2_xi(Y_3)./(dx*dx) .+@d2_yi(Y_3)./(dx*dx) .+@d2_zi(Y_3)./(dx*dx) .+
    gy.*(@inn(ϕ_1).*(@d_yi(ϕ_2)./dx .+@D_3ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(ϕ_2).*(@d_yi(ϕ_1)./dx .+@D_3ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
    @inn(ϕ_3).*(@d_yi(ϕ_4)./dx .+@D_3ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(ϕ_4).*(@d_yi(ϕ_3)./dx .+@D_3ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
    @d_yi(Σ))

    # r[20]=
    @inn(dY_4_dt_n) =@inn(dY_4_dt).+dt.*(@d2_xi(Y_4)./(dx*dx) .+@d2_yi(Y_4)./(dx*dx) .+@d2_zi(Y_4)./(dx*dx) .+
    gy.*(@inn(ϕ_1).*(@d_zi(ϕ_2)./dx .+@D_4ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(ϕ_2).*(@d_zi(ϕ_1)./dx .+@D_4ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
    @inn(ϕ_3).*(@d_zi(ϕ_4)./dx .+@D_4ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(ϕ_4).*(@d_zi(ϕ_3)./dx .+@D_4ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
    @d_zi(Σ))

    return
end

@parallel function evolve!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    Γ_1,Γ_2,Γ_3,Σ,
    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    ϕ_1_n,ϕ_2_n,ϕ_3_n,ϕ_4_n,
    W_1_2_n,W_1_3_n,W_1_4_n,
    W_2_2_n,W_2_3_n,W_2_4_n,
    W_3_2_n,W_3_3_n,W_3_4_n,
    Y_2_n,Y_3_n,Y_4_n,
    Γ_1_n,Γ_2_n,Γ_3_n,Σ_n,
    gw,gy,gp2,lambda,vev,dx,dt)
    
    # s[1]=
    @inn(ϕ_1_n) =@inn(ϕ_1).+dt.*@inn(dϕ_1_dt)
    # s[2]=
    @inn(ϕ_2_n) =@inn(ϕ_2).+dt.*@inn(dϕ_2_dt)
    # s[3]=
    @inn(ϕ_3_n) =@inn(ϕ_3).+dt.*@inn(dϕ_3_dt)
    # s[4]=
    @inn(ϕ_4_n) =@inn(ϕ_4).+dt.*@inn(dϕ_4_dt)
    # c
    # c Eq. (2.11) of Baumgarte&Shapiro is $\partial_t A_i = -E_i -...$ so 
    # c we are taking fd(...)=+\partial_t A_i = -E_i (note the sign).
    # s[5]=0.
    # s[6]=
    @inn(W_1_2_n) =@inn(W_1_2).+dt.*(@inn(dW_1_2_dt))#.+
    # c in the gauge $W^a_0=0=Y_0$, f(5...)=0=f(9...)=f(13...) and the line
    # c below vanishes.
        # @d_xi(W_1_1)./dx .-gw.*(@inn(W_2_1).*@inn(W_3_2).-@inn(W_3_1).*@inn(W_2_2)))
    # s[7]=
    @inn(W_1_3_n) =@inn(W_1_3).+dt.*(@inn(dW_1_3_dt))#.+
        # @d_yi(W_1_1)./dx.-gw.*(@inn(W_2_1).*@inn(W_3_3).-@inn(W_3_1).*@inn(W_2_3)))
    # s[8]=
    @inn(W_1_4_n) =@inn(W_1_4).+dt.*(@inn(dW_1_4_dt))#.+
        # @d_zi(W_1_1)./dx.-gw.*(@inn(W_2_1).*@inn(W_3_4).-@inn(W_3_1).*@inn(W_2_4)))

    # s[9]=0.
    # s[10]=
    @inn(W_2_2_n) =@inn(W_2_2).+dt.*(@inn(dW_2_2_dt))#.+
        # @d_xi(W_2_1)./dx.-gw.*(@inn(W_3_1).*@inn(W_1_2).-@inn(W_1_1).*@inn(W_3_2)))
    # s[11]=
    @inn(W_2_3_n) =@inn(W_2_3).+dt.*(@inn(dW_2_3_dt))#.+
        # @d_yi(W_2_1)./dx.-gw.*(@inn(W_3_1).*@inn(W_1_3).-@inn(W_1_1).*@inn(W_3_3)))
    # s[12]=
    @inn(W_2_4_n) =@inn(W_2_4).+dt.*(@inn(dW_2_4_dt))#.+
        # @d_zi(W_2_1)./dx.-gw.*(@inn(W_3_1).*@inn(W_1_4).-@inn(W_1_1).*@inn(W_3_4)))

    # s[13]=0.
    # s[14]=
    @inn(W_3_2_n) =@inn(W_3_2).+dt.*(@inn(dW_3_2_dt))#.+
        # @d_xi(W_3_1)./dx.-gw.*(@inn(W_1_1).*@inn(W_2_2).-@inn(W_2_1).*@inn(W_1_2)))
    # s[15]=
    @inn(W_3_3_n) =@inn(W_3_3).+dt.*(@inn(dW_3_3_dt))#.+
        # @d_yi(W_3_1)./dx.-gw.*(@inn(W_1_1).*@inn(W_2_3).-@inn(W_2_1).*@inn(W_1_3)))
    # s[16]=
    @inn(W_3_4_n) =@inn(W_3_4).+dt.*(@inn(dW_3_4_dt))#.+
        # @d_zi(W_3_1)./dx.-gw.*(@inn(W_1_1).*@inn(W_2_4).-@inn(W_2_1).*@inn(W_1_4)))

    # s[17]=0.
    # s[18]=
    @inn(Y_2_n) =@inn(Y_2).+dt.*(@inn(dY_2_dt))#.+@d_xi(Y_1)./dx)
    # s[19]=
    @inn(Y_3_n) =@inn(Y_3).+dt.*(@inn(dY_3_dt))#.+@d_yi(Y_1)./dx)
    # s[20]=
    @inn(Y_4_n) =@inn(Y_4).+dt.*(@inn(dY_4_dt))#.+@d_zi(Y_1)./dx)

    # c fluxes for gauge functions:
    # cc if on boundaries:
    # c      if(abs(i).eq.latx.or.abs(j).eq.laty.or.abs(k).eq.latz) then
    # cc radial unit vector:
    # c        px=dfloat(i)/sqrt(dfloat(i**2+j**2+k**2))
    # c        py=dfloat(j)/sqrt(dfloat(i**2+j**2+k**2))
    # c        pz=dfloat(k)/sqrt(dfloat(i**2+j**2+k**2))
    # cc
    # c       s(21)=-(px*dfdx(21)+py*dfdy(21)+pz*dfdz(21))
    # c       s(22)=-(px*dfdx(22)+py*dfdy(22)+pz*dfdz(22))
    # c       s(23)=-(px*dfdx(23)+py*dfdy(23)+pz*dfdz(23))
    # c       s(24)=-(px*dfdx(24)+py*dfdy(24)+pz*dfdz(24))
    # c
    # cc if not on boundaries:
    # c      else
    # c
    # s(Γ_1)=
    @inn(Γ_1_n) =@inn(Γ_1).+dt.*((1.0.-gp2).*(@d_xi(dW_1_2_dt)./(dx) .+@d_yi(dW_1_3_dt)./(dx) .+@d_zi(dW_1_4_dt)./(dx)).+
    gp2 .*gw.*(
    -(@inn(W_2_2).*@inn(dW_3_2_dt).-
    @inn(W_3_2).*@inn(dW_2_2_dt)).-
    (@inn(W_2_3).*@inn(dW_3_3_dt).-
    @inn(W_3_3).*@inn(dW_2_3_dt)).-
    (@inn(W_2_4).*@inn(dW_3_4_dt).-
    @inn(W_3_4).*@inn(dW_2_4_dt))).+
# c charge from Higgs: 
    gp2 .*gw.*(@inn(ϕ_1).*(@inn(dϕ_4_dt) .+@D_1ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(ϕ_4).*(@inn(dϕ_1_dt) .+@D_1ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
    @inn(ϕ_3).*(@inn(dϕ_2_dt) .+@D_1ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(ϕ_2).*(@inn(dϕ_3_dt) .+@D_1ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))))

    # s(Γ_2)=
    @inn(Γ_2_n) =@inn(Γ_2).+dt.*((1.0.-gp2).*(@d_xi(dW_2_2_dt)./(dx) .+@d_yi(dW_2_3_dt)./(dx) .+@d_zi(dW_2_4_dt)./(dx)).+
    gp2 .*gw.*(
    -(@inn(W_3_2).*@inn(dW_1_2_dt).-
    @inn(W_1_2).*@inn(dW_3_2_dt)).-
    (@inn(W_3_3).*@inn(dW_1_3_dt).-
    @inn(W_1_3).*@inn(dW_3_3_dt)).-
    (@inn(W_3_4).*@inn(dW_1_4_dt).-
    @inn(W_1_4).*@inn(dW_3_4_dt))).+
# c charge from Higgs: 
    gp2 .*gw.*(@inn(ϕ_3).*(@inn(dϕ_1_dt) .+@D_1ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(ϕ_1).*(@inn(dϕ_3_dt) .+@D_1ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
    @inn(ϕ_4).*(@inn(dϕ_2_dt) .+@D_1ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(ϕ_2).*(@inn(dϕ_4_dt) .+@D_1ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))))

    # s(Γ_3)=
    @inn(Γ_3_n) =@inn(Γ_3).+dt.*((1.0.-gp2).*(@d_xi(dW_3_2_dt)./(dx) .+@d_yi(dW_3_3_dt)./(dx) .+@d_zi(dW_3_4_dt)./(dx)).+
    gp2 .*gw.*(
    -(@inn(W_1_2).*@inn(dW_2_2_dt).-
    @inn(W_2_2).*@inn(dW_1_2_dt)).-
    (@inn(W_1_3).*@inn(dW_2_3_dt).-
    @inn(W_2_3).*@inn(dW_1_3_dt)).-
    (@inn(W_1_4).*@inn(dW_2_4_dt).-
    @inn(W_2_4).*@inn(dW_1_4_dt))).+
# c current from Higgs: 
    gp2 .*gw.*(@inn(ϕ_1).*(@inn(dϕ_2_dt) .+@D_1ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(ϕ_2).*(@inn(dϕ_1_dt) .+@D_1ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
    @inn(ϕ_4).*(@inn(dϕ_3_dt) .+@D_1ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(ϕ_3).*(@inn(dϕ_4_dt) .+@D_1ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))))

    # s(Σ)=
    @inn(Σ_n) =@inn(Σ).+dt.*((1.0.-gp2).*(@d2_xi(Y_2)./(dx*dx) .+@d2_yi(Y_3)./(dx*dx) .+@d2_zi(Y_4)./(dx*dx)).+
    # c current from Higgs: 
        gp2 .*gy.*(@inn(ϕ_1)*(@inn(dϕ_2_dt) .-@D_1ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
        @inn(ϕ_2).*(@inn(dϕ_1_dt) .-@D_1ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
        @inn(ϕ_3).*(@inn(dϕ_4_dt) .-@D_1ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
        @inn(ϕ_4).*(@inn(dϕ_3_dt) .-@D_1ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))))

    return
end

@parallel function avg_half_step!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    Γ_1,Γ_2,Γ_3,Σ,
    ϕ_1_n,ϕ_2_n,ϕ_3_n,ϕ_4_n,
    W_1_2_n,W_1_3_n,W_1_4_n,
    W_2_2_n,W_2_3_n,W_2_4_n,
    W_3_2_n,W_3_3_n,W_3_4_n,
    Y_2_n,Y_3_n,Y_4_n,
    Γ_1_n,Γ_2_n,Γ_3_n,Σ_n)

    @inn(ϕ_1_n)=(@inn(ϕ_1_n)+@inn(ϕ_1)).*0.5
    @inn(ϕ_2_n)=(@inn(ϕ_2_n)+@inn(ϕ_2)).*0.5
    @inn(ϕ_3_n)=(@inn(ϕ_3_n)+@inn(ϕ_3)).*0.5
    @inn(ϕ_4_n)=(@inn(ϕ_4_n)+@inn(ϕ_4)).*0.5
    @inn(W_1_2_n)=(@inn(W_1_2)+@inn(W_1_2_n)).*0.5
    @inn(W_1_3_n)=(@inn(W_1_3)+@inn(W_1_3_n)).*0.5
    @inn(W_1_4_n)=(@inn(W_1_4)+@inn(W_1_4_n)).*0.5
    @inn(W_2_2_n)=(@inn(W_2_2)+@inn(W_2_2_n)).*0.5
    @inn(W_2_3_n)=(@inn(W_2_3)+@inn(W_2_3_n)).*0.5
    @inn(W_2_4_n)=(@inn(W_2_4)+@inn(W_2_4_n)).*0.5
    @inn(W_3_2_n)=(@inn(W_3_2)+@inn(W_3_2_n)).*0.5
    @inn(W_3_3_n)=(@inn(W_3_3)+@inn(W_3_3_n)).*0.5
    @inn(W_3_4_n)=(@inn(W_3_4)+@inn(W_3_4_n)).*0.5
    @inn(Y_2_n)=(@inn(Y_2)+@inn(Y_2_n)).*0.5
    @inn(Y_3_n)=(@inn(Y_3)+@inn(Y_3_n)).*0.5
    @inn(Y_4_n)=(@inn(Y_4)+@inn(Y_4_n)).*0.5
    @inn(Γ_1_n)=(@inn(Γ_1_n)+@inn(Γ_1)).*0.5
    @inn(Γ_2_n)=(@inn(Γ_2_n)+@inn(Γ_2)).*0.5
    @inn(Γ_3_n)=(@inn(Γ_3_n)+@inn(Γ_3)).*0.5
    @inn(Σ_n)=(@inn(Σ_n)+@inn(Σ)).*0.5

    return
end

@parallel function avg_half_step_dot!(dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    dϕ_1_dt_n,dϕ_2_dt_n,dϕ_3_dt_n,dϕ_4_dt_n,
    dW_1_2_dt_n,dW_1_3_dt_n,dW_1_4_dt_n,
    dW_2_2_dt_n,dW_2_3_dt_n,dW_2_4_dt_n,
    dW_3_2_dt_n,dW_3_3_dt_n,dW_3_4_dt_n,
    dY_2_dt_n,dY_3_dt_n,dY_4_dt_n)

    @inn(dϕ_1_dt_n)=(@inn(dϕ_1_dt_n)+@inn(dϕ_1_dt)).*0.5
    @inn(dϕ_2_dt_n)=(@inn(dϕ_2_dt_n)+@inn(dϕ_2_dt)).*0.5
    @inn(dϕ_3_dt_n)=(@inn(dϕ_3_dt_n)+@inn(dϕ_3_dt)).*0.5
    @inn(dϕ_4_dt_n)=(@inn(dϕ_4_dt_n)+@inn(dϕ_4_dt)).*0.5
    @inn(dW_1_2_dt_n)=(@inn(dW_1_2_dt)+@inn(dW_1_2_dt_n)).*0.5
    @inn(dW_1_3_dt_n)=(@inn(dW_1_3_dt)+@inn(dW_1_3_dt_n)).*0.5
    @inn(dW_1_4_dt_n)=(@inn(dW_1_4_dt)+@inn(dW_1_4_dt_n)).*0.5
    @inn(dW_2_2_dt_n)=(@inn(dW_2_2_dt)+@inn(dW_2_2_dt_n)).*0.5
    @inn(dW_2_3_dt_n)=(@inn(dW_2_3_dt)+@inn(dW_2_3_dt_n)).*0.5
    @inn(dW_2_4_dt_n)=(@inn(dW_2_4_dt)+@inn(dW_2_4_dt_n)).*0.5
    @inn(dW_3_2_dt_n)=(@inn(dW_3_2_dt)+@inn(dW_3_2_dt_n)).*0.5
    @inn(dW_3_3_dt_n)=(@inn(dW_3_3_dt)+@inn(dW_3_3_dt_n)).*0.5
    @inn(dW_3_4_dt_n)=(@inn(dW_3_4_dt)+@inn(dW_3_4_dt_n)).*0.5
    @inn(dY_2_dt_n)=(@inn(dY_2_dt)+@inn(dY_2_dt_n)).*0.5
    @inn(dY_3_dt_n)=(@inn(dY_3_dt)+@inn(dY_3_dt_n)).*0.5
    @inn(dY_4_dt_n)=(@inn(dY_4_dt)+@inn(dY_4_dt_n)).*0.5
    # @inn(dΓ_1_dt_n)=(@inn(dΓ_1_dt_n)+@inn(dΓ_1_dt)).*0.5
    # @inn(dΓ_2_dt_n)=(@inn(dΓ_2_dt_n)+@inn(dΓ_2_dt)).*0.5
    # @inn(dΓ_3_dt_n)=(@inn(dΓ_3_dt_n)+@inn(dΓ_3_dt)).*0.5
    # @inn(dΣ_dt_n)=(@inn(dΣ_dt_n)+@inn(dΣ_dt)).*0.5

    return
end

@parallel function leaforward_dot!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    Γ_1,Γ_2,Γ_3,Σ,
    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    dϕ_1_dt_n,dϕ_2_dt_n,dϕ_3_dt_n,dϕ_4_dt_n,
    dW_1_2_dt_n,dW_1_3_dt_n,dW_1_4_dt_n,
    dW_2_2_dt_n,dW_2_3_dt_n,dW_2_4_dt_n,
    dW_3_2_dt_n,dW_3_3_dt_n,dW_3_4_dt_n,
    dY_2_dt_n,dY_3_dt_n,dY_4_dt_n,
    gw,gy,gp2,lambda,vev,dx,dt)
    
    # @inn(test_arr) = @inn(dϕ_1_dt).+gw*@D_1ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    # W_1_1,W_1_2,W_1_3,W_1_4,
    # W_2_1,W_2_2,W_2_3,W_2_4,
    # W_3_1,W_3_2,W_3_3,W_3_4,
    # Y_1,Y_2,Y_3,Y_4)
    
    ##swapping namespaces when compared to evolve
    ##data input: _n is input from previous step
    ##without is the information from average half step

    #The expressions below calculate fluxes from the fields at half step
    #and assign to global _n arrays which is is non-_n arrays in this function

    @inn(dϕ_1_dt_n) = @inn(dϕ_1_dt_n) .+ dt.*( @d2_xi(ϕ_1)./(dx*dx) .+@d2_yi(ϕ_1)./(dx*dx) .+@d2_zi(ϕ_1)./(dx*dx) .-
    0.5*gw.*((-@inn(W_1_2).*@d_xi(ϕ_4)./dx.-@inn(W_1_3).*@d_yi(ϕ_4)./dx.-@inn(W_1_4).*@d_zi(ϕ_4)./dx).-
    (-@inn(W_2_2).*@d_xi(ϕ_3)./dx.-@inn(W_2_3).*@d_yi(ϕ_3)./dx.-@inn(W_2_4).*@d_zi(ϕ_3)./dx).+
    (-@inn(W_3_2).*@d_xi(ϕ_2)./dx.-@inn(W_3_3).*@d_yi(ϕ_2)./dx.-@inn(W_3_4).*@d_zi(ϕ_2)./dx)).-
    0.5*gy.*(-@inn(Y_2).*@d_xi(ϕ_2)./dx.-@inn(Y_3).*@d_yi(ϕ_2)./dx.-@inn(Y_4).*@d_zi(ϕ_2)./dx).-
    0.5*gw.*((-@inn(W_1_2).*(@d_xi(ϕ_4)./dx .+@D_2ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_1_3).*(@d_yi(ϕ_4)./dx .+@D_3ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_1_4).*(@d_zi(ϕ_4)./dx .+@D_4ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
    (-@inn(W_2_2).*(@d_xi(ϕ_3)./dx .+@D_2ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_2_3).*(@d_yi(ϕ_3)./dx .+@D_3ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_2_4).*(@d_zi(ϕ_3)./dx .+@D_4ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).+
    (-@inn(W_3_2).*(@d_xi(ϕ_2)./dx .+@D_2ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_3_3).*(@d_yi(ϕ_2)./dx .+@D_3ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_3_4).*(@d_zi(ϕ_2)./dx .+@D_4ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)))).-
    0.5*gy.*(-@inn(Y_2).*(@d_xi(ϕ_2)./dx .+@D_2ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(Y_3).*(@d_yi(ϕ_2)./dx .+@D_3ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(Y_4).*(@d_zi(ϕ_2)./dx .+@D_4ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
    2.0*lambda.*((@inn(ϕ_1).*@inn(ϕ_1)) .+(@inn(ϕ_2).*@inn(ϕ_2)) .+(@inn(ϕ_3).*@inn(ϕ_3)) .+(@inn(ϕ_4).*@inn(ϕ_4)) .-vev*vev).*@inn(ϕ_1).+
    0.5 .*((gw.*@inn(Γ_3) .+gy.*@inn(Σ)).*@inn(ϕ_2).-gw.*@inn(Γ_2) .*@inn(ϕ_3).+gw.*@inn(Γ_1).*@inn(ϕ_4)))

    @inn(dϕ_2_dt_n) =@inn(dϕ_2_dt_n) .+ dt.*(@d2_xi(ϕ_2)./(dx*dx) .+@d2_yi(ϕ_2)./(dx*dx) .+@d2_zi(ϕ_2)./(dx*dx) .+
    0.5*gw.*((-@inn(W_1_2).*@d_xi(ϕ_3)./dx.-@inn(W_1_3).*@d_yi(ϕ_3)./dx.-@inn(W_1_4).*@d_zi(ϕ_3)./dx).+
    (-@inn(W_2_2).*@d_xi(ϕ_4)./dx.-@inn(W_2_3).*@d_yi(ϕ_4)./dx.-@inn(W_2_4).*@d_zi(ϕ_4)./dx).+
    (-@inn(W_3_2).*@d_xi(ϕ_1)./dx.-@inn(W_3_3).*@d_yi(ϕ_1)./dx.-@inn(W_3_4).*@d_zi(ϕ_1)./dx)).+
    0.5.*gy.*(-@inn(Y_2).*@d_xi(ϕ_1)./dx.-@inn(Y_3).*@d_yi(ϕ_1)./dx.-@inn(Y_4).*@d_zi(ϕ_1)./dx).+
    0.5.*gw.*((-@inn(W_1_2).*(@d_xi(ϕ_3)./dx .+@D_2ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_1_3).*(@d_yi(ϕ_3)./dx .+@D_3ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_1_4).*(@d_zi(ϕ_3)./dx .+@D_4ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).+
    (-@inn(W_2_2).*(@d_xi(ϕ_4)./dx .+@D_2ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_2_3).*(@d_yi(ϕ_4)./dx .+@D_3ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_2_4).*(@d_zi(ϕ_4)./dx .+@D_4ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).+
    (-@inn(W_3_2).*(@d_xi(ϕ_1)./dx .+@D_2ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_3_3).*(@d_yi(ϕ_1)./dx .+@D_3ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_3_4).*(@d_zi(ϕ_1)./dx .+@D_4ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)))).+
    0.5.*gy.*(-@inn(Y_2).*(@d_xi(ϕ_1)./dx .+@D_2ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(Y_3).*(@d_yi(ϕ_1)./dx .+@D_3ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(Y_4).*(@d_zi(ϕ_1)./dx .+@D_4ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
    2.0.*lambda.*((@inn(ϕ_1).*@inn(ϕ_1)) .+(@inn(ϕ_2).*@inn(ϕ_2)) .+(@inn(ϕ_3).*@inn(ϕ_3)) .+(@inn(ϕ_4).*@inn(ϕ_4)) .-vev*vev)*@inn(ϕ_2).-
    0.5 .*((gw*@inn(Γ_3) .+gy.*@inn(Σ)) .*@inn(ϕ_1) .+gw .*@inn(Γ_1).*@inn(ϕ_3) .+gw.*@inn(Γ_2) .*@inn(ϕ_4)))

    @inn(dϕ_3_dt_n) =@inn(dϕ_3_dt_n) .+ dt.*(@d2_xi(ϕ_3)./(dx*dx) .+@d2_yi(ϕ_3)./(dx*dx) .+@d2_zi(ϕ_3)./(dx*dx) .-
    0.5*gw.*((-@inn(W_1_2).*@d_xi(ϕ_2)./dx.-@inn(W_1_3).*@d_yi(ϕ_2)./dx.-@inn(W_1_4).*@d_zi(ϕ_2)./dx).+
    (-@inn(W_2_2).*@d_xi(ϕ_1)./dx.-@inn(W_2_3).*@d_yi(ϕ_1)./dx.-@inn(W_2_4).*@d_zi(ϕ_1)./dx).-
    (-@inn(W_3_2).*@d_xi(ϕ_4)./dx.-@inn(W_3_3).*@d_yi(ϕ_4)./dx.-@inn(W_3_4).*@d_zi(ϕ_4)./dx)).-
    0.5.*gy.*(-@inn(Y_2).*@d_xi(ϕ_4)./dx.-@inn(Y_3).*@d_yi(ϕ_4)./dx.-@inn(Y_4).*@d_zi(ϕ_4)./dx).-
    0.5.*gw.*((-@inn(W_1_2).*(@d_xi(ϕ_2)./dx .+@D_2ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_1_3).*(@d_yi(ϕ_2)./dx .+@D_3ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_1_4).*(@d_zi(ϕ_2)./dx .+@D_4ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).+
    (-@inn(W_2_2).*(@d_xi(ϕ_1)./dx .+@D_2ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_2_3).*(@d_yi(ϕ_1)./dx .+@D_3ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_2_4).*(@d_zi(ϕ_1)./dx .+@D_4ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
    (-@inn(W_3_2).*(@d_xi(ϕ_4)./dx .+@D_2ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_3_3).*(@d_yi(ϕ_4)./dx .+@D_3ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_3_4).*(@d_zi(ϕ_4)./dx .+@D_4ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)))).-
    0.5.*gy.*(-@inn(Y_2).*(@d_xi(ϕ_4)./dx .+@D_2ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(Y_3).*(@d_yi(ϕ_4)./dx .+@D_3ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(Y_4).*(@d_zi(ϕ_4)./dx .+@D_4ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
    2.0.*lambda.*((@inn(ϕ_1).*@inn(ϕ_1)) .+(@inn(ϕ_2).*@inn(ϕ_2)) .+(@inn(ϕ_3).*@inn(ϕ_3)) .+(@inn(ϕ_4).*@inn(ϕ_4)) .-vev*vev).*@inn(ϕ_3).+
    0.5 .*((-gw.*@inn(Γ_3) .+gy*@inn(Σ))*@inn(ϕ_4).+gw .*@inn(Γ_2) .*@inn(ϕ_1).+gw .*@inn(Γ_1) .*@inn(ϕ_2)))

    @inn(dϕ_4_dt_n) =@inn(dϕ_4_dt_n) .+ dt.*(@d2_xi(ϕ_4)./(dx*dx) .+@d2_yi(ϕ_4)./(dx*dx) .+@d2_zi(ϕ_4)./(dx*dx) .+
    0.5.*gw.*((-@inn(W_1_2).*@d_xi(ϕ_1)./dx.-@inn(W_1_3).*@d_yi(ϕ_1)./dx.-@inn(W_1_4).*@d_zi(ϕ_1)./dx).-
    (-@inn(W_2_2).*@d_xi(ϕ_2)./dx.-@inn(W_2_3).*@d_yi(ϕ_2)./dx.-@inn(W_2_4).*@d_zi(ϕ_2)./dx).-
    (-@inn(W_3_2).*@d_xi(ϕ_3)./dx.-@inn(W_3_3).*@d_yi(ϕ_3)./dx.-@inn(W_3_4).*@d_zi(ϕ_3)./dx)).+
    0.5.*gy.*(-@inn(Y_2).*@d_xi(ϕ_3)./dx.-@inn(Y_3).*@d_yi(ϕ_3)./dx.-@inn(Y_4).*@d_zi(ϕ_3)./dx).+
    0.5.*gw.*((-@inn(W_1_2).*(@d_xi(ϕ_1)./dx .+@D_2ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_1_3).*(@d_yi(ϕ_1)./dx .+@D_3ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_1_4).*(@d_zi(ϕ_1)./dx .+@D_4ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
    (-@inn(W_2_2).*(@d_xi(ϕ_2)./dx .+@D_2ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_2_3).*(@d_yi(ϕ_2)./dx .+@D_3ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_2_4).*(@d_zi(ϕ_2)./dx .+@D_4ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
    (-@inn(W_3_2).*(@d_xi(ϕ_3)./dx .+@D_2ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_3_3).*(@d_yi(ϕ_3)./dx .+@D_3ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(W_3_4).*(@d_zi(ϕ_3)./dx .+@D_4ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)))).+
    0.5.*gy.*(-@inn(Y_2).*(@d_xi(ϕ_3)./dx .+@D_2ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(Y_3).*(@d_yi(ϕ_3)./dx .+@D_3ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(Y_4).*(@d_zi(ϕ_3)./dx .+@D_4ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
    2.0.*lambda.*((@inn(ϕ_1).*@inn(ϕ_1)) .+(@inn(ϕ_2).*@inn(ϕ_2)) .+(@inn(ϕ_3).*@inn(ϕ_3)) .+(@inn(ϕ_4).*@inn(ϕ_4)) .-vev.*vev).*@inn(ϕ_4).-
    0.5 .*((-gw .*@inn(Γ_3) .+gy.*@inn(Σ)).*@inn(ϕ_3).+gw.*@inn(Γ_1) .*@inn(ϕ_1).-gw.*@inn(Γ_2) .*@inn(ϕ_2)))

    # c W.-fluxes:
    # c
    # r(W_1_1)=0.

    # r(W_1_2)= 
    @inn(dW_1_2_dt_n) =@inn(dW_1_2_dt_n).+dt.*(@d2_xi(W_1_2)./(dx*dx) .+@d2_yi(W_1_2)./(dx*dx) .+@d2_zi(W_1_2)./(dx*dx) .+gw.*( 
        -(@d_xi(W_2_2)./dx.*@inn(W_3_2).-@d_xi(W_3_2)./dx.*@inn(W_2_2)).-
        (@d_yi(W_2_2)./dx.*@inn(W_3_3).-@d_yi(W_3_2)./dx.*@inn(W_2_3)).-
        (@d_zi(W_2_2)./dx.*@inn(W_3_4).-@d_zi(W_3_2)./dx.*@inn(W_2_4)).-
        (@inn(W_2_3).*( @d_xi(W_3_3)./dx .- @d_yi(W_3_2)./dx .+ gw.*( @inn(W_1_2) .*@inn(W_2_3) .- @inn(W_1_3).*@inn(W_2_2) )).-#fs(3,2,3)
        @inn(W_3_3).*( @d_xi(W_2_3)./dx .-@d_yi(W_2_2)./dx .+gw.*( @inn(W_3_2) .*@inn(W_1_3) .- @inn(W_3_3).*@inn(W_1_2) ))).-#fs(2,2,3)
        (@inn(W_2_4).*( @d_xi(W_3_4)./dx .-@d_zi(W_3_2)./dx .+gw.*( @inn(W_1_2) .*@inn(W_2_4) .- @inn(W_1_4).*@inn(W_2_2) )).-#fs(3,2,4)
        @inn(W_3_4).*( @d_xi(W_2_4)./dx .-@d_zi(W_2_2)./dx .+gw.*( @inn(W_3_2) .*@inn(W_1_4) .- @inn(W_3_4).*@inn(W_1_2) )))#fs(2,2,4)
        ).+
        gw.*(@inn(ϕ_1).*(@d_xi(ϕ_4)./dx .+@D_2ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
        @inn(ϕ_2).*(@d_xi(ϕ_3)./dx .+@D_2ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
        @inn(ϕ_3).*(@d_xi(ϕ_2)./dx .+@D_2ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
        @inn(ϕ_4).*(@d_xi(ϕ_1)./dx .+@D_2ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
        @d_xi(Γ_1)./dx .-gw.*(@inn(W_2_2).*@inn(Γ_3).-@inn(W_3_2).*@inn(Γ_2)))


    # r(W_1_3)=
    @inn(dW_1_3_dt_n) =@inn(dW_1_3_dt_n).+dt.*(@d2_xi(W_1_3)./(dx*dx) .+@d2_yi(W_1_3)./(dx*dx) .+@d2_zi(W_1_3)./(dx*dx) .+gw.*( 
        -(@d_xi(W_2_3)./dx.*@inn(W_3_2).-@d_xi(W_3_3)./dx.*@inn(W_2_2)).-
        (@d_yi(W_2_3)./dx.*@inn(W_3_3).-@d_yi(W_3_3)./dx.*@inn(W_2_3)).-
        (@d_zi(W_2_3)./dx.*@inn(W_3_4).-@d_zi(W_3_3)./dx.*@inn(W_2_4)).-
        (@inn(W_2_2).*-( @d_xi(W_3_3)./dx .- @d_yi(W_3_2)./dx .+ gw.*( @inn(W_1_2) .*@inn(W_2_3) .- @inn(W_1_3).*@inn(W_2_2) )).-#fs(3,3,2)
        @inn(W_3_2).*-( @d_xi(W_2_3)./dx .-@d_yi(W_2_2)./dx .+gw.*( @inn(W_3_2) .*@inn(W_1_3) .- @inn(W_3_3).*@inn(W_1_2) ))).-#fs(2,3,2)
        (@inn(W_2_4).*( @d_yi(W_3_4)./dx .-@d_zi(W_3_3)./dx .+gw.*( @inn(W_1_3) .*@inn(W_2_4) .- @inn(W_1_4).*@inn(W_2_3) )).-#fs(3,3,4)
        @inn(W_3_4).*( @d_yi(W_2_4)./dx .-@d_zi(W_2_3)./dx .+gw.*( @inn(W_3_3) .*@inn(W_1_4) .- @inn(W_3_4).*@inn(W_1_3) )))#fs(2,3,4)
        ).+
        gw.*(@inn(ϕ_1).*(@d_yi(ϕ_4)./dx .+@D_3ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
        @inn(ϕ_2).*(@d_yi(ϕ_3)./dx .+@D_3ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
        @inn(ϕ_3).*(@d_yi(ϕ_2)./dx .+@D_3ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
        @inn(ϕ_4).*(@d_yi(ϕ_1)./dx .+@D_3ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
        @d_yi(Γ_1)./dx .-gw.*(@inn(W_2_3).*@inn(Γ_3).-@inn(W_3_3).*@inn(Γ_2)))#.-

    # r(W_1_4)=
    @inn(dW_1_4_dt_n) =@inn(dW_1_4_dt_n).+dt.*(@d2_xi(W_1_4)./(dx*dx) .+@d2_yi(W_1_4)./(dx*dx) .+@d2_zi(W_1_4)./(dx*dx) .+gw.*( 
        -(@d_xi(W_2_4)./dx.*@inn(W_3_2).-@d_xi(W_3_4)./dx.*@inn(W_2_2)).-
        (@d_yi(W_2_4)./dx.*@inn(W_3_3).-@d_yi(W_3_4)./dx.*@inn(W_2_3)).-
        (@d_zi(W_2_4)./dx.*@inn(W_3_4).-@d_zi(W_3_4)./dx.*@inn(W_2_4)).-
        (@inn(W_2_2).*-( @d_xi(W_3_4)./dx .-@d_zi(W_3_2)./dx .+gw.*( @inn(W_1_2) .*@inn(W_2_4) .- @inn(W_1_4).*@inn(W_2_2) )).-#fs(3,4,2)
        @inn(W_3_2).*-( @d_xi(W_2_4)./dx .-@d_zi(W_2_2)./dx .+gw.*( @inn(W_3_2) .*@inn(W_1_4) .- @inn(W_3_4).*@inn(W_1_2) ))).-#fs(2,4,2)
        (@inn(W_2_3).*-( @d_yi(W_3_4)./dx .-@d_zi(W_3_3)./dx .+gw.*( @inn(W_1_3) .*@inn(W_2_4) .- @inn(W_1_4).*@inn(W_2_3) )).-#fs(3,4,3)
        @inn(W_3_3).*-( @d_yi(W_2_4)./dx .-@d_zi(W_2_3)./dx .+gw.*( @inn(W_3_3) .*@inn(W_1_4) .- @inn(W_3_4).*@inn(W_1_3) )))#fs(2,4,3)
        ).+
        gw.*(@inn(ϕ_1).*(@d_zi(ϕ_4)./dx .+@D_4ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
        @inn(ϕ_2).*(@d_zi(ϕ_3)./dx .+@D_4ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
        @inn(ϕ_3).*(@d_zi(ϕ_2)./dx .+@D_4ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
        @inn(ϕ_4).*(@d_zi(ϕ_1)./dx .+@D_4ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
        @d_zi(Γ_1)./dx .-gw.*(@inn(W_2_4).*@inn(Γ_3).-@inn(W_3_4).*@inn(Γ_2)))#.-


    # r(W_2_1)=0.

    # r(W_2_2)=
    @inn(dW_2_2_dt_n) =@inn(dW_2_2_dt_n).+dt.*(@d2_xi(W_2_2)./(dx*dx) .+@d2_yi(W_2_2)./(dx*dx) .+@d2_zi(W_2_2)./(dx*dx) .+gw.*( 
        -(@d_xi(W_3_2)./dx.*@inn(W_1_2).-@d_xi(W_1_2)./dx.*@inn(W_3_2)).-
        (@d_yi(W_3_2)./dx.*@inn(W_1_3).-@d_yi(W_1_2)./dx.*@inn(W_3_3)).-
        (@d_zi(W_3_2)./dx.*@inn(W_1_4).-@d_zi(W_1_2)./dx.*@inn(W_3_4)).-
        (@inn(W_3_3).*( @d_xi(W_1_3)./dx .-@d_yi(W_1_2)./dx .+gw.*( @inn(W_2_2) .*@inn(W_3_3) .- @inn(W_2_3).*@inn(W_3_2) )).-#fs(1,2,3)
        @inn(W_1_3).*( @d_xi(W_3_3)./dx .-@d_yi(W_3_2)./dx .+gw.*( @inn(W_1_2) .*@inn(W_2_3) .- @inn(W_1_3).*@inn(W_2_2) ))).-#fs(3,2,3)
        (@inn(W_3_4).*( @d_xi(W_1_4)./dx .-@d_zi(W_1_2)./dx .+gw.*( @inn(W_2_2) .*@inn(W_3_4) .- @inn(W_2_4).*@inn(W_3_2) )).-#fs(1,2,4)
        @inn(W_1_4).*( @d_xi(W_3_4)./dx .-@d_zi(W_3_2)./dx .+gw.*( @inn(W_1_2) .*@inn(W_2_4) .- @inn(W_1_4).*@inn(W_2_2) )))#fs(3,2,4)
        ).+
        gw.*(-@inn(ϕ_1).*(@d_xi(ϕ_3)./dx .+@D_2ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
        @inn(ϕ_2).*(@d_xi(ϕ_4)./dx .+@D_2ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
        @inn(ϕ_3).*(@d_xi(ϕ_1)./dx .+@D_2ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
        @inn(ϕ_4).*(@d_xi(ϕ_2)./dx .+@D_2ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
        @d_xi(Γ_2)./dx .-gw.*(@inn(W_3_2).*@inn(Γ_1).-@inn(W_1_2).*@inn(Γ_3)))#.-


    # r(W_2_3)=
    @inn(dW_2_3_dt_n) =@inn(dW_2_3_dt_n).+dt.*(@d2_xi(W_2_3)./(dx*dx) .+@d2_yi(W_2_3)./(dx*dx) .+@d2_zi(W_2_3)./(dx*dx) .+gw.*( 
        -(@d_xi(W_3_3)./dx.*@inn(W_1_2).-@d_xi(W_1_3)./dx.*@inn(W_3_2)).-
        (@d_yi(W_3_3)./dx.*@inn(W_1_3).-@d_yi(W_1_3)./dx.*@inn(W_3_3)).-
        (@d_zi(W_3_3)./dx.*@inn(W_1_4).-@d_zi(W_1_3)./dx.*@inn(W_3_4)).-
        (@inn(W_3_2).*-( @d_xi(W_1_3)./dx .-@d_yi(W_1_2)./dx .+gw.*( @inn(W_2_2) .*@inn(W_3_3) .- @inn(W_2_3).*@inn(W_3_2) )).-#fs(1,3,2)
        @inn(W_1_2).*-( @d_xi(W_3_3)./dx .- @d_yi(W_3_2)./dx .+ gw.*( @inn(W_1_2) .*@inn(W_2_3) .- @inn(W_1_3).*@inn(W_2_2) ))).-#fs(3,3,2)
        (@inn(W_3_4).*( @d_yi(W_1_4)./dx .-@d_zi(W_1_3)./dx .+gw.*( @inn(W_2_3) .*@inn(W_3_4) .- @inn(W_2_4).*@inn(W_3_3) )).-#fs(1,3,4)
        @inn(W_1_4).*( @d_yi(W_3_4)./dx .-@d_zi(W_3_3)./dx .+gw.*( @inn(W_1_3) .*@inn(W_2_4) .- @inn(W_1_4).*@inn(W_2_3) )))#fs(3,3,4)
        ).+
        gw.*(-@inn(ϕ_1).*(@d_yi(ϕ_3)./dx .+@D_3ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
        @inn(ϕ_2).*(@d_yi(ϕ_4)./dx .+@D_3ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
        @inn(ϕ_3).*(@d_yi(ϕ_1)./dx .+@D_3ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
        @inn(ϕ_4).*(@d_yi(ϕ_2)./dx .+@D_3ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
        @d_yi(Γ_2)./dx .-gw.*(@inn(W_3_3).*@inn(Γ_1).-@inn(W_1_3).*@inn(Γ_3)))#.-


    # r(W_2_4)=
    @inn(dW_2_4_dt_n) =@inn(dW_2_4_dt_n).+dt.*(@d2_xi(W_2_4)./(dx*dx) .+@d2_yi(W_2_4)./(dx*dx) .+@d2_zi(W_2_4)./(dx*dx) .+gw.*( 
        -(@d_xi(W_3_4)./dx.*@inn(W_1_2).-@d_xi(W_1_4)./dx.*@inn(W_3_2)).-
        (@d_yi(W_3_4)./dx.*@inn(W_1_3).-@d_yi(W_1_4)./dx.*@inn(W_3_3)).-
        (@d_zi(W_3_4)./dx.*@inn(W_1_4).-@d_zi(W_1_4)./dx.*@inn(W_3_4)).-
        (@inn(W_3_2).*-( @d_xi(W_1_4)./dx .-@d_zi(W_1_2)./dx .+gw.*( @inn(W_2_2) .*@inn(W_3_4) .- @inn(W_2_4).*@inn(W_3_2) )).-#fs(1,4,2)
        @inn(W_1_2).*-( @d_xi(W_3_4)./dx .-@d_zi(W_3_2)./dx .+gw.*( @inn(W_1_2) .*@inn(W_2_4) .- @inn(W_1_4).*@inn(W_2_2) ))).-#fs(3,4,2)
        (@inn(W_3_3).*-( @d_yi(W_1_4)./dx .-@d_zi(W_1_3)./dx .+gw.*( @inn(W_2_3) .*@inn(W_3_4) .- @inn(W_2_4).*@inn(W_3_3) )).-#fs(1,4,3)
        @inn(W_1_3).*-( @d_yi(W_3_4)./dx .-@d_zi(W_3_3)./dx .+gw.*( @inn(W_1_3) .*@inn(W_2_4) .- @inn(W_1_4).*@inn(W_2_3) )))#fs(3,4,3)
        ).+
        gw.*(-@inn(ϕ_1).*(@d_zi(ϕ_3)./dx .+@D_4ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
        @inn(ϕ_2).*(@d_zi(ϕ_4)./dx .+@D_4ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
        @inn(ϕ_3).*(@d_zi(ϕ_1)./dx .+@D_4ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
        @inn(ϕ_4).*(@d_zi(ϕ_2)./dx .+@D_4ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
        @d_zi(Γ_2)./dx .-gw.*(@inn(W_3_4).*@inn(Γ_1).-@inn(W_1_4).*@inn(Γ_3)))#.-


    # r(W_3_1)=0.

    # r(W_3_2)=
    @inn(dW_3_2_dt_n) =@inn(dW_3_2_dt_n).+dt.*(@d2_xi(W_3_2)./(dx*dx) .+@d2_yi(W_3_2)./(dx*dx) .+@d2_zi(W_3_2)./(dx*dx) .+gw.*( 
        -(@d_xi(W_1_2).*@inn(W_2_2)./dx.-@d_xi(W_2_2)./dx.*@inn(W_1_2)).-
        (@d_yi(W_1_2).*@inn(W_2_3)./dx.-@d_yi(W_2_2)./dx.*@inn(W_1_3)).-
        (@d_zi(W_1_2).*@inn(W_2_4)./dx.-@d_zi(W_2_2)./dx.*@inn(W_1_4)).-
        (@inn(W_1_3).*( @d_xi(W_2_3)./dx .-@d_yi(W_2_2)./dx .+gw.*( @inn(W_3_2) .*@inn(W_1_3) .- @inn(W_3_3).*@inn(W_1_2) )).-#fs(2,2,3)
        @inn(W_2_3).*( @d_xi(W_1_3)./dx .-@d_yi(W_1_2)./dx .+gw.*( @inn(W_2_2) .*@inn(W_3_3) .- @inn(W_2_3).*@inn(W_3_2) ))).-#fs(1,2,3)
        (@inn(W_1_4).*( @d_xi(W_2_4)./dx .-@d_zi(W_2_2)./dx .+gw.*( @inn(W_3_2) .*@inn(W_1_4) .- @inn(W_3_4).*@inn(W_1_2) )).-#fs(2,2,4)
        @inn(W_2_4).*( @d_xi(W_1_4)./dx .-@d_zi(W_1_2)./dx .+gw.*( @inn(W_2_2) .*@inn(W_3_4) .- @inn(W_2_4).*@inn(W_3_2) )))#fs(3,2,4)
        ).+
        gw.*(@inn(ϕ_1).*(@d_xi(ϕ_2)./dx .+@D_2ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
        @inn(ϕ_2).*(@d_xi(ϕ_1)./dx .+@D_2ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
        @inn(ϕ_3).*(@d_xi(ϕ_4)./dx .+@D_2ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
        @inn(ϕ_4).*(@d_xi(ϕ_3)./dx .+@D_2ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
        @d_xi(Γ_3)./dx .-gw.*(@inn(W_1_2).*@inn(Γ_2).-@inn(W_2_2).*@inn(Γ_1)))#.-


    # r(W_3_3)=
    @inn(dW_3_3_dt_n) =@inn(dW_3_3_dt_n).+dt.*(@d2_xi(W_3_3)./(dx*dx) .+@d2_yi(W_3_3)./(dx*dx) .+@d2_zi(W_3_3)./(dx*dx) .+gw.*( 
        -(@d_xi(W_1_3)./dx.*@inn(W_2_2)./dx.-@d_xi(W_2_3)./dx.*@inn(W_1_2)).-
        (@d_yi(W_1_3)./dx.*@inn(W_2_3)./dx.-@d_yi(W_2_3)./dx.*@inn(W_1_3)).-
        (@d_zi(W_1_3)./dx.*@inn(W_2_4)./dx.-@d_zi(W_2_3)./dx.*@inn(W_1_4)).-
        (@inn(W_1_2).*-( @d_xi(W_2_3)./dx .-@d_yi(W_2_2)./dx .+gw.*( @inn(W_3_2) .*@inn(W_1_3) .- @inn(W_3_3).*@inn(W_1_2) )).-#fs(2,3,2)
        @inn(W_2_2).*-( @d_xi(W_1_3)./dx .-@d_yi(W_1_2)./dx .+gw.*( @inn(W_2_2) .*@inn(W_3_3) .- @inn(W_2_3).*@inn(W_3_2) ))).-#fs(1,3,2)
        (@inn(W_1_4).*( @d_yi(W_2_4)./dx .-@d_zi(W_2_3)./dx .+gw.*( @inn(W_3_3) .*@inn(W_1_4) .- @inn(W_3_4).*@inn(W_1_3) )).-#fs(2,3,4)
        @inn(W_2_4).*( @d_yi(W_1_4)./dx .-@d_zi(W_1_3)./dx .+gw.*( @inn(W_2_3) .*@inn(W_3_4) .- @inn(W_2_4).*@inn(W_3_3) )))#fs(1,3,4)
        ).+
        gw.*(@inn(ϕ_1).*(@d_yi(ϕ_2)./dx .+@D_3ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
        @inn(ϕ_2).*(@d_yi(ϕ_1)./dx .+@D_3ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
        @inn(ϕ_3).*(@d_yi(ϕ_4)./dx .+@D_3ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
        @inn(ϕ_4).*(@d_yi(ϕ_3)./dx .+@D_3ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
        @d_yi(Γ_3)./dx .-gw.*(@inn(W_1_3).*@inn(Γ_2).-@inn(W_2_3).*@inn(Γ_1)))#.-


    # r(W_3_4)=
    @inn(dW_3_4_dt_n) =@inn(dW_3_4_dt_n).+dt.*(@d2_xi(W_3_4)./(dx*dx) .+@d2_yi(W_3_4)./(dx*dx) .+@d2_zi(W_3_4)./(dx*dx) .+gw.*( 
        -(@d_xi(W_1_4)./dx.*@inn(W_2_2).-@d_xi(W_2_4)./dx.*@inn(W_1_2)).-
        (@d_yi(W_1_4)./dx.*@inn(W_2_3).-@d_yi(W_2_4)./dx.*@inn(W_1_3)).-
        (@d_zi(W_1_4)./dx.*@inn(W_2_4).-@d_zi(W_2_4)./dx.*@inn(W_1_4)).-
        (@inn(W_1_2).*-( @d_xi(W_2_4)./dx .-@d_zi(W_2_2)./dx .+gw.*( @inn(W_3_2) .*@inn(W_1_4) .- @inn(W_3_4).*@inn(W_1_2) )).-#fs(2,4,2)
        @inn(W_2_2).*-( @d_xi(W_1_4)./dx .-@d_zi(W_1_2)./dx .+gw.*( @inn(W_2_2) .*@inn(W_3_4) .- @inn(W_2_4).*@inn(W_3_2) ))).-#fs(1,4,2)
        (@inn(W_1_3).*-( @d_yi(W_2_4)./dx .-@d_zi(W_2_3)./dx .+gw.*( @inn(W_3_3) .*@inn(W_1_4) .- @inn(W_3_4).*@inn(W_1_3) )).-#fs(2,4,3)
        @inn(W_2_3).*-( @d_yi(W_1_4)./dx .-@d_zi(W_1_3)./dx .+gw.*( @inn(W_2_3) .*@inn(W_3_4) .- @inn(W_2_4).*@inn(W_3_3) )))#fs(1,4,3)
        ).+
        gw.*(@inn(ϕ_1).*(@d_zi(ϕ_2)./dx .+@D_4ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
        @inn(ϕ_2).*(@d_zi(ϕ_1)./dx .+@D_4ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
        @inn(ϕ_3).*(@d_zi(ϕ_4)./dx .+@D_4ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
        @inn(ϕ_4).*(@d_zi(ϕ_3)./dx .+@D_4ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
        @d_zi(Γ_3)./dx .-gw.*(@inn(W_1_4).*@inn(Γ_2).-@inn(W_2_4).*@inn(Γ_1)))#.-


    # c Y.-fluxes:

    # r(Y_1)=0.

    #r(Y_2)=
    @inn(dY_2_dt_n) =@inn(dY_2_dt_n).+dt.*(@d2_xi(Y_2)./(dx*dx) .+@d2_yi(Y_2)./(dx*dx) .+@d2_zi(Y_2)./(dx*dx) .+
    gy.*(@inn(ϕ_1).*(@d_xi(ϕ_2)./dx .+@D_2ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(ϕ_2).*(@d_xi(ϕ_1)./dx .+@D_2ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
    @inn(ϕ_3).*(@d_xi(ϕ_4)./dx .+@D_2ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(ϕ_4).*(@d_xi(ϕ_3)./dx .+@D_2ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
    @d_xi(Σ))

    # r[19]=
    @inn(dY_3_dt_n) =@inn(dY_3_dt_n).+dt.*(@d2_xi(Y_3)./(dx*dx) .+@d2_yi(Y_3)./(dx*dx) .+@d2_zi(Y_3)./(dx*dx) .+
    gy.*(@inn(ϕ_1).*(@d_yi(ϕ_2)./dx .+@D_3ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(ϕ_2).*(@d_yi(ϕ_1)./dx .+@D_3ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
    @inn(ϕ_3).*(@d_yi(ϕ_4)./dx .+@D_3ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(ϕ_4).*(@d_yi(ϕ_3)./dx .+@D_3ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
    @d_yi(Σ))

    # r[20]=
    @inn(dY_4_dt_n) =@inn(dY_4_dt_n).+dt.*(@d2_xi(Y_4)./(dx*dx) .+@d2_yi(Y_4)./(dx*dx) .+@d2_zi(Y_4)./(dx*dx) .+
    gy.*(@inn(ϕ_1).*(@d_zi(ϕ_2)./dx .+@D_4ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(ϕ_2).*(@d_zi(ϕ_1)./dx .+@D_4ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
    @inn(ϕ_3).*(@d_zi(ϕ_4)./dx .+@D_4ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(ϕ_4).*(@d_zi(ϕ_3)./dx .+@D_4ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).-
    @d_zi(Σ))

    ##Need to drop these and just not bother evolving these. So no need
    ##to declare in the first place
    # @inn(Γ_1_n) = @inn(Γ_1_n).+dt.*@inn(Γ_1)
    # @inn(Γ_2_n) = @inn(Γ_2_n).+dt.*@inn(Γ_2)
    # @inn(Γ_3_n) = @inn(Γ_3_n).+dt.*@inn(Γ_3)
    # @inn(Σ_n) = @inn(Σ).+dt.*@inn(Σ)

    return
end

@parallel function leaforward!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    Γ_1,Γ_2,Γ_3,Σ,
    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    ϕ_1_n,ϕ_2_n,ϕ_3_n,ϕ_4_n,
    W_1_2_n,W_1_3_n,W_1_4_n,
    W_2_2_n,W_2_3_n,W_2_4_n,
    W_3_2_n,W_3_3_n,W_3_4_n,
    Y_2_n,Y_3_n,Y_4_n,
    Γ_1_n,Γ_2_n,Γ_3_n,Σ_n,
    gw,gy,gp2,lambda,vev,dx,dt)
    
    # @inn(test_arr) = @inn(dϕ_1_dt).+gw*@D_1ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    # W_1_1,W_1_2,W_1_3,W_1_4,
    # W_2_1,W_2_2,W_2_3,W_2_4,
    # W_3_1,W_3_2,W_3_3,W_3_4,
    # Y_1,Y_2,Y_3,Y_4)
    
    ##swapping namespaces when compared to evolve
    ##data input: _n is input from previous step
    ##without is the information from average half step

    #The expressions below calculate fluxes from the fields at half step
    #and assign to global _n arrays which is is non-_n arrays in this function

    # s[1]=
    @inn(ϕ_1_n) =@inn(ϕ_1_n).+dt.*@inn(dϕ_1_dt)
    # s[2]=
    @inn(ϕ_2_n) =@inn(ϕ_2_n).+dt.*@inn(dϕ_2_dt)
    # s[3]=
    @inn(ϕ_3_n) =@inn(ϕ_3_n).+dt.*@inn(dϕ_3_dt)
    # s[4]=
    @inn(ϕ_4_n) =@inn(ϕ_4_n).+dt.*@inn(dϕ_4_dt)
    # c
    # c Eq. (2.11) of Baumgarte&Shapiro is $\partial_t A_i = -E_i -...$ so 
    # c we are taking fd(...)=+\partial_t A_i = -E_i (note the sign).
    # s[5]=0.
    # s[6]=
    @inn(W_1_2_n) =@inn(W_1_2_n).+dt.*(@inn(dW_1_2_dt))#.+
    # c in the gauge $W^a_0=0=Y_0$, f(5...)=0=f(9...)=f(13...) and the line
    # c below vanishes.
        # @d_xi(W_1_1)./dx .-gw.*(@inn(W_2_1).*@inn(W_3_2).-@inn(W_3_1).*@inn(W_2_2)))
    # s[7]=
    @inn(W_1_3_n) =@inn(W_1_3_n).+dt.*(@inn(dW_1_3_dt))#.+
        # @d_yi(W_1_1)./dx.-gw.*(@inn(W_2_1).*@inn(W_3_3).-@inn(W_3_1).*@inn(W_2_3)))
    # s[8]=
    @inn(W_1_4_n) =@inn(W_1_4_n).+dt.*(@inn(dW_1_4_dt))#.+
        # @d_zi(W_1_1)./dx.-gw.*(@inn(W_2_1).*@inn(W_3_4).-@inn(W_3_1).*@inn(W_2_4)))

    # s[9]=0.
    # s[10]=
    @inn(W_2_2_n) =@inn(W_2_2_n).+dt.*(@inn(dW_2_2_dt))#.+
        # @d_xi(W_2_1)./dx.-gw.*(@inn(W_3_1).*@inn(W_1_2).-@inn(W_1_1).*@inn(W_3_2)))
    # s[11]=
    @inn(W_2_3_n) =@inn(W_2_3_n).+dt.*(@inn(dW_2_3_dt))#.+
        # @d_yi(W_2_1)./dx.-gw.*(@inn(W_3_1).*@inn(W_1_3).-@inn(W_1_1).*@inn(W_3_3)))
    # s[12]=
    @inn(W_2_4_n) =@inn(W_2_4_n).+dt.*(@inn(dW_2_4_dt))#.+
        # @d_zi(W_2_1)./dx.-gw.*(@inn(W_3_1).*@inn(W_1_4).-@inn(W_1_1).*@inn(W_3_4)))

    # s[13]=0.
    # s[14]=
    @inn(W_3_2_n) =@inn(W_3_2_n).+dt.*(@inn(dW_3_2_dt))#.+
        # @d_xi(W_3_1)./dx.-gw.*(@inn(W_1_1).*@inn(W_2_2).-@inn(W_2_1).*@inn(W_1_2)))
    # s[15]=
    @inn(W_3_3_n) =@inn(W_3_3_n).+dt.*(@inn(dW_3_3_dt))#.+
        # @d_yi(W_3_1)./dx.-gw.*(@inn(W_1_1).*@inn(W_2_3).-@inn(W_2_1).*@inn(W_1_3)))
    # s[16]=
    @inn(W_3_4_n) =@inn(W_3_4_n).+dt.*(@inn(dW_3_4_dt))#.+
        # @d_zi(W_3_1)./dx.-gw.*(@inn(W_1_1).*@inn(W_2_4).-@inn(W_2_1).*@inn(W_1_4)))

    # s[17]=0.
    # s[18]=
    @inn(Y_2_n) =@inn(Y_2_n).+dt.*(@inn(dY_2_dt))#.+@d_xi(Y_1)./dx)
    # s[19]=
    @inn(Y_3_n) =@inn(Y_3_n).+dt.*(@inn(dY_3_dt))#.+@d_yi(Y_1)./dx)
    # s[20]=
    @inn(Y_4_n) =@inn(Y_4_n).+dt.*(@inn(dY_4_dt))#.+@d_zi(Y_1)./dx)

    # c fluxes for gauge functions:
    # cc if on boundaries:
    # c      if(abs(i).eq.latx.or.abs(j).eq.laty.or.abs(k).eq.latz) then
    # cc radial unit vector:
    # c        px=dfloat(i)/sqrt(dfloat(i**2+j**2+k**2))
    # c        py=dfloat(j)/sqrt(dfloat(i**2+j**2+k**2))
    # c        pz=dfloat(k)/sqrt(dfloat(i**2+j**2+k**2))
    # cc
    # c       s(21)=-(px*dfdx(21)+py*dfdy(21)+pz*dfdz(21))
    # c       s(22)=-(px*dfdx(22)+py*dfdy(22)+pz*dfdz(22))
    # c       s(23)=-(px*dfdx(23)+py*dfdy(23)+pz*dfdz(23))
    # c       s(24)=-(px*dfdx(24)+py*dfdy(24)+pz*dfdz(24))
    # c
    # cc if not on boundaries:
    # c      else
    # c
    # s(Γ_1)=
    @inn(Γ_1_n) =@inn(Γ_1_n).+dt.*((1.0.-gp2).*(@d_xi(dW_1_2_dt)./(dx) .+@d_yi(dW_1_3_dt)./(dx) .+@d_zi(dW_1_4_dt)./(dx)).+
    gp2 .*gw.*(
    -(@inn(W_2_2).*@inn(dW_3_2_dt).-
    @inn(W_3_2).*@inn(dW_2_2_dt)).-
    (@inn(W_2_3).*@inn(dW_3_3_dt).-
    @inn(W_3_3).*@inn(dW_2_3_dt)).-
    (@inn(W_2_4).*@inn(dW_3_4_dt).-
    @inn(W_3_4).*@inn(dW_2_4_dt))).+
# c charge from Higgs: 
    gp2 .*gw.*(@inn(ϕ_1).*(@inn(dϕ_4_dt) .+@D_1ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(ϕ_4).*(@inn(dϕ_1_dt) .+@D_1ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
    @inn(ϕ_3).*(@inn(dϕ_2_dt) .+@D_1ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(ϕ_2).*(@inn(dϕ_3_dt) .+@D_1ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))))

    # s(Γ_2)=
    @inn(Γ_2_n) =@inn(Γ_2_n).+dt.*((1.0.-gp2).*(@d_xi(dW_2_2_dt)./(dx) .+@d_yi(dW_2_3_dt)./(dx) .+@d_zi(dW_2_4_dt)./(dx)).+
    gp2 .*gw.*(
    -(@inn(W_3_2).*@inn(dW_1_2_dt).-
    @inn(W_1_2).*@inn(dW_3_2_dt)).-
    (@inn(W_3_3).*@inn(dW_1_3_dt).-
    @inn(W_1_3).*@inn(dW_3_3_dt)).-
    (@inn(W_3_4).*@inn(dW_1_4_dt).-
    @inn(W_1_4).*@inn(dW_3_4_dt))).+
# c charge from Higgs: 
    gp2 .*gw.*(@inn(ϕ_3).*(@inn(dϕ_1_dt) .+@D_1ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(ϕ_1).*(@inn(dϕ_3_dt) .+@D_1ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
    @inn(ϕ_4).*(@inn(dϕ_2_dt) .+@D_1ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(ϕ_2).*(@inn(dϕ_4_dt) .+@D_1ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))))

    # s(Γ_3)=
    @inn(Γ_3_n) =@inn(Γ_3_n).+dt.*((1.0.-gp2).*(@d_xi(dW_3_2_dt)./(dx) .+@d_yi(dW_3_3_dt)./(dx) .+@d_zi(dW_3_4_dt)./(dx)).+
    gp2 .*gw.*(
    -(@inn(W_1_2).*@inn(dW_2_2_dt).-
    @inn(W_2_2).*@inn(dW_1_2_dt)).-
    (@inn(W_1_3).*@inn(dW_2_3_dt).-
    @inn(W_2_3).*@inn(dW_1_3_dt)).-
    (@inn(W_1_4).*@inn(dW_2_4_dt).-
    @inn(W_2_4).*@inn(dW_1_4_dt))).+
# c current from Higgs: 
    gp2 .*gw.*(@inn(ϕ_1).*(@inn(dϕ_2_dt) .+@D_1ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(ϕ_2).*(@inn(dϕ_1_dt) .+@D_1ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
    @inn(ϕ_4).*(@inn(dϕ_3_dt) .+@D_1ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    @inn(ϕ_3).*(@inn(dϕ_4_dt) .+@D_1ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))))

    # s(Σ)=
    @inn(Σ_n) =@inn(Σ_n).+dt.*((1.0.-gp2).*(@d_xi(dY_2_dt)./(dx) .+@d_yi(dY_3_dt)./(dx) .+@d_zi(dY_4_dt)./(dx)).+
    # c current from Higgs: 
        gp2 .*gy.*(@inn(ϕ_1)*(@inn(dϕ_2_dt) .+@D_1ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
        @inn(ϕ_2).*(@inn(dϕ_1_dt) .+@D_1ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
        @inn(ϕ_3).*(@inn(dϕ_4_dt) .+@D_1ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
        @inn(ϕ_4).*(@inn(dϕ_3_dt) .+@D_1ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))))

    return
end

@parallel function E_potential!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,E_V,
    lambda,vev)
    @inn(E_V) = lambda.*((@inn(ϕ_1).*@inn(ϕ_1).+@inn(ϕ_2).*@inn(ϕ_2)+@inn(ϕ_3).*@inn(ϕ_3)+@inn(ϕ_4).*@inn(ϕ_4)).-vev.^2).^2
    return
end

#Could lump in E_kin and E_grad to save on mem and compute

@parallel function E_kinetic!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt, E_kin)
    @inn(E_kin) = ((@inn(dϕ_1_dt) .-@D_1ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).*
        (@inn(dϕ_1_dt) .-@D_1ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).+
        ((@inn(dϕ_2_dt) .-@D_1ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).*
        (@inn(dϕ_2_dt) .-@D_1ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).+
        ((@inn(dϕ_3_dt) .-@D_1ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).*
        (@inn(dϕ_3_dt) .-@D_1ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).+
        ((@inn(dϕ_4_dt) .-@D_1ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).*
        (@inn(dϕ_4_dt) .-@D_1ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)))
    return
end

@parallel function E_gradient!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,E_grad,dx)
    @inn(E_grad) = ((((@d_xi(ϕ_1)./dx .-@D_2ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).*
    (@d_xi(ϕ_1)./dx .-@D_2ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).+
    ((@d_xi(ϕ_2)./dx .-@D_2ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).*
    (@d_xi(ϕ_2)./dx .-@D_2ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).+
    ((@d_xi(ϕ_3)./dx .-@D_2ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).*
    (@d_xi(ϕ_3)./dx .-@D_2ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).+
    ((@d_xi(ϕ_4)./dx .-@D_2ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).*
    (@d_xi(ϕ_4)./dx .-@D_2ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)))).+
    (((@d_yi(ϕ_1)./dx .-@D_3ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).*
    (@d_yi(ϕ_1)./dx .-@D_3ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).+
    ((@d_yi(ϕ_2)./dx .-@D_3ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).*
    (@d_yi(ϕ_2)./dx .-@D_3ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).+
    ((@d_yi(ϕ_3)./dx .-@D_3ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).*
    (@d_yi(ϕ_3)./dx .-@D_3ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).+
    ((@d_yi(ϕ_4)./dx .-@D_3ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).*
    (@d_yi(ϕ_4)./dx .-@D_3ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)))).+
    (((@d_zi(ϕ_1)./dx .-@D_4ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).*
    (@d_zi(ϕ_1)./dx .-@D_4ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).+
    ((@d_zi(ϕ_2)./dx .-@D_4ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).*
    (@d_zi(ϕ_2)./dx .-@D_4ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).+
    ((@d_zi(ϕ_3)./dx .-@D_4ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).*
    (@d_zi(ϕ_3)./dx .-@D_4ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))).+
    ((@d_zi(ϕ_4)./dx .-@D_4ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).*
    (@d_zi(ϕ_4)./dx .-@D_4ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)))))
    # @inn(E_grad) = ((@d_xi(ϕ_1)./dx .-@D_2ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).*
    # (@d_xi(ϕ_1)./dx .-@D_2ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)))
    return
end

#Could lump E_W_e, E_W_m, E_Y_e and E_Y_m to save on 
#memory and compilation time by a bit

@parallel function E_W_electric!(dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,E_W_e)
    @inn(E_W_e) = 0.5.*((@inn(dW_1_2_dt).*(@inn(dW_1_2_dt))).+
    (@inn(dW_1_3_dt).*(@inn(dW_1_3_dt))).+
    (@inn(dW_1_4_dt).*(@inn(dW_1_4_dt))).+
    (@inn(dW_2_2_dt).*(@inn(dW_2_2_dt))).+
    (@inn(dW_2_3_dt).*(@inn(dW_2_3_dt))).+
    (@inn(dW_2_4_dt).*(@inn(dW_2_4_dt))).+
    (@inn(dW_3_2_dt).*(@inn(dW_3_2_dt))).+
    (@inn(dW_3_3_dt).*(@inn(dW_3_3_dt))).+
    (@inn(dW_3_4_dt).*(@inn(dW_3_4_dt))))
    return
end

@parallel function E_W_magnetic!(W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,gw,dx,E_W_m)
    #Verified all field strength expressions
    @inn(E_W_m) = 0.5.*((( @d_xi(W_1_3)./dx .-@d_yi(W_1_2)./dx .+gw.*( @inn(W_2_2) .*@inn(W_3_3) .- @inn(W_2_3).*@inn(W_3_2) )).*
    ( @d_xi(W_1_3)./dx .-@d_yi(W_1_2)./dx .+gw.*( @inn(W_2_2) .*@inn(W_3_3) .- @inn(W_2_3).*@inn(W_3_2) ))).+
    (( @d_xi(W_1_4)./dx .-@d_zi(W_1_2)./dx .+gw.*( @inn(W_2_2) .*@inn(W_3_4) .- @inn(W_2_4).*@inn(W_3_2) )).*
    ( @d_xi(W_1_4)./dx .-@d_zi(W_1_2)./dx .+gw.*( @inn(W_2_2) .*@inn(W_3_4) .- @inn(W_2_4).*@inn(W_3_2) ))).+
    (( @d_yi(W_1_4)./dx .-@d_zi(W_1_3)./dx .+gw.*( @inn(W_2_3) .*@inn(W_3_4) .- @inn(W_2_4).*@inn(W_3_3) )).*
    ( @d_yi(W_1_4)./dx .-@d_zi(W_1_3)./dx .+gw.*( @inn(W_2_3) .*@inn(W_3_4) .- @inn(W_2_4).*@inn(W_3_3) ))).+
    (( @d_xi(W_2_3)./dx .-@d_yi(W_2_2)./dx .+gw.*( @inn(W_3_2) .*@inn(W_1_3) .- @inn(W_3_3).*@inn(W_1_2) )).*
    ( @d_xi(W_2_3)./dx .-@d_yi(W_2_2)./dx .+gw.*( @inn(W_3_2) .*@inn(W_1_3) .- @inn(W_3_3).*@inn(W_1_2) ))).+
    (( @d_xi(W_2_4)./dx .-@d_zi(W_2_2)./dx .+gw.*( @inn(W_3_2) .*@inn(W_1_4) .- @inn(W_3_4).*@inn(W_1_2) )).*
    ( @d_xi(W_2_4)./dx .-@d_zi(W_2_2)./dx .+gw.*( @inn(W_3_2) .*@inn(W_1_4) .- @inn(W_3_4).*@inn(W_1_2) ))).+
    (( @d_yi(W_2_4)./dx .-@d_zi(W_2_3)./dx .+gw.*( @inn(W_3_3) .*@inn(W_1_4) .- @inn(W_3_4).*@inn(W_1_3) )).*
    ( @d_yi(W_2_4)./dx .-@d_zi(W_2_3)./dx .+gw.*( @inn(W_3_3) .*@inn(W_1_4) .- @inn(W_3_4).*@inn(W_1_3) ))).+
    (( @d_xi(W_3_3)./dx .- @d_yi(W_3_2)./dx .+ gw.*( @inn(W_1_2) .*@inn(W_2_3) .- @inn(W_1_3).*@inn(W_2_2) )).*
    ( @d_xi(W_3_3)./dx .- @d_yi(W_3_2)./dx .+ gw.*( @inn(W_1_2) .*@inn(W_2_3) .- @inn(W_1_3).*@inn(W_2_2) ))).+
    (( @d_xi(W_3_4)./dx .-@d_zi(W_3_2)./dx .+gw.*( @inn(W_1_2) .*@inn(W_2_4) .- @inn(W_1_4).*@inn(W_2_2) )).*
    ( @d_xi(W_3_4)./dx .-@d_zi(W_3_2)./dx .+gw.*( @inn(W_1_2) .*@inn(W_2_4) .- @inn(W_1_4).*@inn(W_2_2) ))).+
    (( @d_xi(W_3_4)./dx .-@d_zi(W_3_3)./dx .+gw.*( @inn(W_1_3) .*@inn(W_2_4) .- @inn(W_1_4).*@inn(W_2_3) )).*
    ( @d_xi(W_3_4)./dx .-@d_zi(W_3_3)./dx .+gw.*( @inn(W_1_3) .*@inn(W_2_4) .- @inn(W_1_4).*@inn(W_2_3) ))))
    return
end

@parallel function E_Y_electric!(dY_2_dt,dY_3_dt,dY_4_dt,E_Y_e)
    @inn(E_Y_e) = 0.5.*(@inn(dY_2_dt).*@inn(dY_2_dt).+@inn(dY_3_dt).*@inn(dY_3_dt).+@inn(dY_4_dt).*@inn(dY_4_dt))
    return
end

@parallel function E_Y_magnetic!(Y_2,Y_3,Y_4,E_Y_m)
    @inn(E_Y_m) = 0.5.*((@d_xi(Y_3).-@d_yi(Y_2)).*(@d_xi(Y_3).-@d_yi(Y_2)).+
    (@d_xi(Y_4).-@d_zi(Y_2)).*(@d_xi(Y_4).-@d_zi(Y_2)).+
    (@d_yi(Y_4).-@d_zi(Y_3)).*(@d_yi(Y_4).-@d_zi(Y_3)))
    return
end

# Single device version
# @parallel_indices (ix,iy,iz) function initializer!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,ib,jb,kb,p1,p2,p3,p4,rk,dx,mH)
#     rb = (1.0/rk)*sqrt(sin(rk*((ix-1)-(ib-0.5))*dx)^2+sin(rk*((iy-1)-(jb-0.5))*dx)^2+sin(rk*((iz-1)-(kb-0.5))*dx)^2)
#     rmag = (1.0+(sqrt(2.0)-1.0)^2)*exp(-mH*rb/sqrt(2.0))/(1.0+((sqrt(2.0)-1.0)^2)*exp(-sqrt(2.0)*mH*rb))
#     ϕ_1[ix,iy,iz]=ϕ_1[ix,iy,iz]+rmag*p1
#     ϕ_2[ix,iy,iz]=ϕ_2[ix,iy,iz]+rmag*p2
#     ϕ_3[ix,iy,iz]=ϕ_3[ix,iy,iz]+rmag*p3
#     ϕ_4[ix,iy,iz]=ϕ_4[ix,iy,iz]+rmag*p4
#     return
# end

# Multi device version
# @parallel_indices (ix,iy,iz) function initializer!(ϕ_1_i,ϕ_2_i,ϕ_3_i,ϕ_4_i,ib,jb,kb,p1,p2,p3,p4,rkx,rky,rkz,dx,mH)
#     rb = sqrt((1.0/(rkx^2))*sin(rkx*((ix-1)-(ib-0.5))*dx)^2+(1.0/(rky^2))*sin(rky*((iy-1)-(jb-0.5))*dx)^2+(1.0/(rkz^2))*sin(rkz*((iz-1)-(kb-0.5))*dx)^2)
#     rmag = (1.0+(sqrt(2.0)-1.0)^2)*exp(-mH*rb/sqrt(2.0))/(1.0+((sqrt(2.0)-1.0)^2)*exp(-sqrt(2.0)*mH*rb))
#     ϕ_1_i[ix,iy,iz]=ϕ_1_i[ix,iy,iz]+rmag*p1
#     ϕ_2_i[ix,iy,iz]=ϕ_2_i[ix,iy,iz]+rmag*p2
#     ϕ_3_i[ix,iy,iz]=ϕ_3_i[ix,iy,iz]+rmag*p3
#     ϕ_4_i[ix,iy,iz]=ϕ_4_i[ix,iy,iz]+rmag*p4
#     return
# end

@parallel_indices (ix,iy,iz) function initializer!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,ib,jb,kb,p1,p2,p3,p4,rkx,rky,rkz,dx,mH,grid_x,grid_y,grid_z)
    x_g = (ix-2)*dx+(grid_x)*(size(ϕ_1,1)-2)*dx
    y_g = (iy-2)*dx+(grid_y)*(size(ϕ_1,2)-2)*dx
    z_g = (iz-2)*dx+(grid_z)*(size(ϕ_1,3)-2)*dx
    rb = sqrt((1.0/(rkx^2))*sin(rkx*(x_g-(ib-0.5)*dx))^2+(1.0/(rky^2))*sin(rky*(y_g-(jb-0.5)*dx))^2+(1.0/(rkz^2))*sin(rkz*(z_g-(kb-0.5)*dx))^2)
    rmag = (1.0+(sqrt(2.0)-1.0)^2)*exp(-mH*rb/sqrt(2.0))/(1.0+((sqrt(2.0)-1.0)^2)*exp(-sqrt(2.0)*mH*rb))
    ϕ_1[ix,iy,iz]=ϕ_1[ix,iy,iz]+rmag*p1
    ϕ_2[ix,iy,iz]=ϕ_2[ix,iy,iz]+rmag*p2
    ϕ_3[ix,iy,iz]=ϕ_3[ix,iy,iz]+rmag*p3
    ϕ_4[ix,iy,iz]=ϕ_4[ix,iy,iz]+rmag*p4
    return
end

@parallel function renormalize!(ϕ_1,ϕ_2,ϕ_3,ϕ_4)
    @all(ϕ_1)=@all(ϕ_1)/(sqrt(@all(ϕ_1)*@all(ϕ_1)+@all(ϕ_2)*@all(ϕ_2)+@all(ϕ_3)*@all(ϕ_3)+@all(ϕ_4)*@all(ϕ_4)))
    @all(ϕ_2)=@all(ϕ_2)/(sqrt(@all(ϕ_1)*@all(ϕ_1)+@all(ϕ_2)*@all(ϕ_2)+@all(ϕ_3)*@all(ϕ_3)+@all(ϕ_4)*@all(ϕ_4)))
    @all(ϕ_3)=@all(ϕ_3)/(sqrt(@all(ϕ_1)*@all(ϕ_1)+@all(ϕ_2)*@all(ϕ_2)+@all(ϕ_3)*@all(ϕ_3)+@all(ϕ_4)*@all(ϕ_4)))
    @all(ϕ_4)=@all(ϕ_4)/(sqrt(@all(ϕ_1)*@all(ϕ_1)+@all(ϕ_2)*@all(ϕ_2)+@all(ϕ_3)*@all(ϕ_3)+@all(ϕ_4)*@all(ϕ_4)))
    return
end

@parallel function Magnetic_fields!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    gw,sw,cw,vev,B_x,B_y,B_z,dx)
    # vn1 = ((-2 .*@inn(ϕ_1).*@inn(ϕ_3) -2 .*@inn(ϕ_2) .*@inn(ϕ_4))./(@inn(ϕ_1).*@inn(ϕ_1) .+@inn(ϕ_2).*@inn(ϕ_2) .+ @inn(ϕ_3).*@inn(ϕ_3) .+ @inn(ϕ_4).*@inn(ϕ_4)))
    # vn2 = ((2 .*@inn(ϕ_2) .*@inn(ϕ_3) -2 .*@inn(ϕ_1).*@inn(ϕ_4))./(@inn(ϕ_1).*@inn(ϕ_1) .+@inn(ϕ_2).*@inn(ϕ_2) .+ @inn(ϕ_3).*@inn(ϕ_3) .+ @inn(ϕ_4).*@inn(ϕ_4)))
    # vn3 = ((-@inn(ϕ_1).*@inn(ϕ_1) .-@inn(ϕ_2) .*@inn(ϕ_2) .+ @inn(ϕ_3).*@inn(ϕ_3) .+ @inn(ϕ_4).*@inn(ϕ_4))./(@inn(ϕ_1).*@inn(ϕ_1) .+@inn(ϕ_2).*@inn(ϕ_2) .+ @inn(ϕ_3).*@inn(ϕ_3) .+ @inn(ϕ_4).*@inn(ϕ_4)))
    @inn(B_x) = -(sw.*(((-2 .*@inn(ϕ_1).*@inn(ϕ_3) -2 .*@inn(ϕ_2) .*@inn(ϕ_4))./(@inn(ϕ_1).*@inn(ϕ_1) .+@inn(ϕ_2).*@inn(ϕ_2) .+ @inn(ϕ_3).*@inn(ϕ_3) .+ @inn(ϕ_4).*@inn(ϕ_4))).*
    ( @d_yi(W_1_4)./dx .-@d_zi(W_1_3)./dx .+gw.*( @inn(W_2_3) .*@inn(W_3_4) .- @inn(W_2_4).*@inn(W_3_3) )).+
    ((2 .*@inn(ϕ_2) .*@inn(ϕ_3) -2 .*@inn(ϕ_1).*@inn(ϕ_4))./(@inn(ϕ_1).*@inn(ϕ_1) .+@inn(ϕ_2).*@inn(ϕ_2) .+ @inn(ϕ_3).*@inn(ϕ_3) .+ @inn(ϕ_4).*@inn(ϕ_4))).*
    ( @d_yi(W_2_4)./dx .-@d_zi(W_2_3)./dx .+gw.*( @inn(W_3_3) .*@inn(W_1_4) .- @inn(W_3_4).*@inn(W_1_3) )).+
    ((-@inn(ϕ_1).*@inn(ϕ_1) .-@inn(ϕ_2) .*@inn(ϕ_2) .+ @inn(ϕ_3).*@inn(ϕ_3) .+ @inn(ϕ_4).*@inn(ϕ_4))./(@inn(ϕ_1).*@inn(ϕ_1) .+@inn(ϕ_2).*@inn(ϕ_2) .+ @inn(ϕ_3).*@inn(ϕ_3) .+ @inn(ϕ_4).*@inn(ϕ_4))).*
    ( @d_xi(W_3_4)./dx .-@d_zi(W_3_3)./dx .+gw.*( @inn(W_1_3) .*@inn(W_2_4) .- @inn(W_1_4).*@inn(W_2_3) ))).+
    cw.*(@d_yi(Y_4).-@d_zi(Y_3)).+#fs(4,3,4)
    (4*sw/(gw*vev^2)).*
    ((@d_yi(ϕ_1)./dx .-@D_3ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).*
    (@d_zi(ϕ_2)./dx .-@D_4ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    (@d_yi(ϕ_2)./dx .-@D_3ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).*
    (@d_zi(ϕ_1)./dx .-@D_4ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
    (@d_yi(ϕ_3)./dx .-@D_3ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).*
    (@d_zi(ϕ_3)./dx .-@D_4ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    (@d_yi(ϕ_4)./dx .-@D_3ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).*
    (@d_zi(ϕ_3)./dx .-@D_4ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))))
    
    @inn(B_y) = -(sw.*(((-2 .*@inn(ϕ_1).*@inn(ϕ_3) -2 .*@inn(ϕ_2) .*@inn(ϕ_4))./(@inn(ϕ_1).*@inn(ϕ_1) .+@inn(ϕ_2).*@inn(ϕ_2) .+ @inn(ϕ_3).*@inn(ϕ_3) .+ @inn(ϕ_4).*@inn(ϕ_4))).*
    ( @d_xi(W_1_4)./dx .-@d_zi(W_1_2)./dx .+gw.*( @inn(W_2_2) .*@inn(W_3_4) .- @inn(W_2_4).*@inn(W_3_2) )).+
    ((2 .*@inn(ϕ_2) .*@inn(ϕ_3) -2 .*@inn(ϕ_1).*@inn(ϕ_4))./(@inn(ϕ_1).*@inn(ϕ_1) .+@inn(ϕ_2).*@inn(ϕ_2) .+ @inn(ϕ_3).*@inn(ϕ_3) .+ @inn(ϕ_4).*@inn(ϕ_4))).*
    ( @d_xi(W_2_4)./dx .-@d_zi(W_2_2)./dx .+gw.*( @inn(W_3_2) .*@inn(W_1_4) .- @inn(W_3_4).*@inn(W_1_2) )).+
    ((-@inn(ϕ_1).*@inn(ϕ_1) .-@inn(ϕ_2) .*@inn(ϕ_2) .+ @inn(ϕ_3).*@inn(ϕ_3) .+ @inn(ϕ_4).*@inn(ϕ_4))./(@inn(ϕ_1).*@inn(ϕ_1) .+@inn(ϕ_2).*@inn(ϕ_2) .+ @inn(ϕ_3).*@inn(ϕ_3) .+ @inn(ϕ_4).*@inn(ϕ_4))).*
    ( @d_xi(W_3_4)./dx .-@d_zi(W_3_2)./dx .+gw.*( @inn(W_1_2) .*@inn(W_2_4) .- @inn(W_1_4).*@inn(W_2_2) ))).+
    cw.*(@d_zi(Y_2).-@d_xi(Y_4)).+
    (4*sw/(gw*vev^2)).*
    ((@d_zi(ϕ_1)./dx .-@D_4ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).*
    (@d_xi(ϕ_2)./dx .-@D_2ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    (@d_zi(ϕ_2)./dx .-@D_4ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).*
    (@d_xi(ϕ_1)./dx .-@D_2ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
    (@d_zi(ϕ_3)./dx .-@D_4ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).*
    (@d_xi(ϕ_4)./dx .-@D_2ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    (@d_zi(ϕ_4)./dx .-@D_4ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).*
    (@d_xi(ϕ_3)./dx .-@D_2ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))))

    @inn(B_z) = -(sw.*(((-2 .*@inn(ϕ_1).*@inn(ϕ_3) -2 .*@inn(ϕ_2) .*@inn(ϕ_4))./(@inn(ϕ_1).*@inn(ϕ_1) .+@inn(ϕ_2).*@inn(ϕ_2) .+ @inn(ϕ_3).*@inn(ϕ_3) .+ @inn(ϕ_4).*@inn(ϕ_4))).*
    ( @d_xi(W_1_3)./dx .-@d_yi(W_1_2)./dx .+gw.*( @inn(W_2_2) .*@inn(W_3_3) .- @inn(W_2_3).*@inn(W_3_2) )).+
    ((2 .*@inn(ϕ_2) .*@inn(ϕ_3) -2 .*@inn(ϕ_1).*@inn(ϕ_4))./(@inn(ϕ_1).*@inn(ϕ_1) .+@inn(ϕ_2).*@inn(ϕ_2) .+ @inn(ϕ_3).*@inn(ϕ_3) .+ @inn(ϕ_4).*@inn(ϕ_4))).*
    ( @d_xi(W_2_3)./dx .-@d_yi(W_2_2)./dx .+gw.*( @inn(W_3_2) .*@inn(W_1_3) .- @inn(W_3_3).*@inn(W_1_2) )).+
    ((-@inn(ϕ_1).*@inn(ϕ_1) .-@inn(ϕ_2) .*@inn(ϕ_2) .+ @inn(ϕ_3).*@inn(ϕ_3) .+ @inn(ϕ_4).*@inn(ϕ_4))./(@inn(ϕ_1).*@inn(ϕ_1) .+@inn(ϕ_2).*@inn(ϕ_2) .+ @inn(ϕ_3).*@inn(ϕ_3) .+ @inn(ϕ_4).*@inn(ϕ_4))).*
    ( @d_xi(W_3_3)./dx .- @d_yi(W_3_2)./dx .+ gw.*( @inn(W_1_2) .*@inn(W_2_3) .- @inn(W_1_3).*@inn(W_2_2) ))).+
    cw.*(@d_xi(Y_3)-@d_yi(Y_2)).+
    (4*sw/(gw*vev^2)).*
    ((@d_xi(ϕ_1)./dx .-@D_2ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).*
    (@d_yi(ϕ_2)./dx .-@D_3ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    (@d_xi(ϕ_2)./dx .-@D_2ϕ_2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).*
    (@d_yi(ϕ_1)./dx .-@D_3ϕ_1(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).+
    (@d_xi(ϕ_3)./dx .-@D_2ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).*
    (@d_yi(ϕ_4)./dx .-@D_3ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).-
    (@d_xi(ϕ_4)./dx .-@D_2ϕ_4(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4)).*
    (@d_yi(ϕ_3)./dx .-@D_3ϕ_3(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4))))
    return
end

@parallel function name_change!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    Γ_1,Γ_2,Γ_3,Σ,
    ϕ_1_n,ϕ_2_n,ϕ_3_n,ϕ_4_n,
    W_1_2_n,W_1_3_n,W_1_4_n,
    W_2_2_n,W_2_3_n,W_2_4_n,
    W_3_2_n,W_3_3_n,W_3_4_n,
    Y_2_n,Y_3_n,Y_4_n,
    Γ_1_n,Γ_2_n,Γ_3_n,Σ_n)

    @all(ϕ_1)=@all(ϕ_1_n)
    @all(ϕ_2)=@all(ϕ_2_n)
    @all(ϕ_3)=@all(ϕ_3_n)
    @all(ϕ_4)=@all(ϕ_4_n)
    # @all(W_1_1)=@all(W_1_1_n)
    @all(W_1_2)=@all(W_1_2_n)
    @all(W_1_3)=@all(W_1_3_n)
    @all(W_1_4)=@all(W_1_4_n)
    # @all(W_2_1)=@all(W_2_1_n)
    @all(W_2_2)=@all(W_2_2_n)
    @all(W_2_3)=@all(W_2_3_n)
    @all(W_2_4)=@all(W_2_4_n)
    # @all(W_3_1)=@all(W_3_1_n)
    @all(W_3_2)=@all(W_3_2_n)
    @all(W_3_3)=@all(W_3_3_n)
    @all(W_3_4)=@all(W_3_4_n)
    # @all(Y_1)=@all(Y_1_n)
    @all(Y_2)=@all(Y_2_n)
    @all(Y_3)=@all(Y_3_n)
    @all(Y_4)=@all(Y_4_n)
    @all(Γ_1)=@all(Γ_1_n)
    @all(Γ_2)=@all(Γ_2_n)
    @all(Γ_3)=@all(Γ_3_n)
    @all(Σ)=@all(Σ_n)

    @all(ϕ_1_n)=@all(ϕ_1)
    @all(ϕ_2_n)=@all(ϕ_2)
    @all(ϕ_3_n)=@all(ϕ_3)
    @all(ϕ_4_n)=@all(ϕ_4)
    # @all(W_1_1_n)=@all(W_1_1)
    @all(W_1_2_n)=@all(W_1_2)
    @all(W_1_3_n)=@all(W_1_3)
    @all(W_1_4_n)=@all(W_1_4)
    # @all(W_2_1_n)=@all(W_2_1)
    @all(W_2_2_n)=@all(W_2_2)
    @all(W_2_3_n)=@all(W_2_3)
    @all(W_2_4_n)=@all(W_2_4)
    # @all(W_3_1_n)=@all(W_3_1)
    @all(W_3_2_n)=@all(W_3_2)
    @all(W_3_3_n)=@all(W_3_3)
    @all(W_3_4_n)=@all(W_3_4)
    # @all(Y_1_n)=@all(Y_1)
    @all(Y_2_n)=@all(Y_2)
    @all(Y_3_n)=@all(Y_3)
    @all(Y_4_n)=@all(Y_4)
    @all(Γ_1_n)=@all(Γ_1)
    @all(Γ_2_n)=@all(Γ_2)
    @all(Γ_3_n)=@all(Γ_3)
    @all(Σ_n)=@all(Σ)

    return
end

@parallel function name_change_dot!(dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    dϕ_1_dt_n,dϕ_2_dt_n,dϕ_3_dt_n,dϕ_4_dt_n,
    dW_1_2_dt_n,dW_1_3_dt_n,dW_1_4_dt_n,
    dW_2_2_dt_n,dW_2_3_dt_n,dW_2_4_dt_n,
    dW_3_2_dt_n,dW_3_3_dt_n,dW_3_4_dt_n,
    dY_2_dt_n,dY_3_dt_n,dY_4_dt_n)
    @all(dϕ_1_dt)=@all(dϕ_1_dt_n)
    @all(dϕ_2_dt)=@all(dϕ_2_dt_n)
    @all(dϕ_3_dt)=@all(dϕ_3_dt_n)
    @all(dϕ_4_dt)=@all(dϕ_4_dt_n)
    # @all(dW_1_1_dt)=@all(dW_1_1_dt_n)
    @all(dW_1_2_dt)=@all(dW_1_2_dt_n)
    @all(dW_1_3_dt)=@all(dW_1_3_dt_n)
    @all(dW_1_4_dt)=@all(dW_1_4_dt_n)
    # @all(dW_2_1_dt)=@all(dW_2_1_dt_n)
    @all(dW_2_2_dt)=@all(dW_2_2_dt_n)
    @all(dW_2_3_dt)=@all(dW_2_3_dt_n)
    @all(dW_2_4_dt)=@all(dW_2_4_dt_n)
    # @all(dW_3_1_dt)=@all(dW_3_1_dt_n)
    @all(dW_3_2_dt)=@all(dW_3_2_dt_n)
    @all(dW_3_3_dt)=@all(dW_3_3_dt_n)
    @all(dW_3_4_dt)=@all(dW_3_4_dt_n)
    # @all(dY_1_dt)=@all(dY_1_dt_n)
    @all(dY_2_dt)=@all(dY_2_dt_n)
    @all(dY_3_dt)=@all(dY_3_dt_n)
    @all(dY_4_dt)=@all(dY_4_dt_n)

    @all(dϕ_1_dt_n)=@all(dϕ_1_dt)
    @all(dϕ_2_dt_n)=@all(dϕ_2_dt)
    @all(dϕ_3_dt_n)=@all(dϕ_3_dt)
    @all(dϕ_4_dt_n)=@all(dϕ_4_dt)
    # @all(dW_1_1_dt_n)=@all(dW_1_1_dt)
    @all(dW_1_2_dt_n)=@all(dW_1_2_dt)
    @all(dW_1_3_dt_n)=@all(dW_1_3_dt)
    @all(dW_1_4_dt_n)=@all(dW_1_4_dt)
    # @all(dW_2_1_dt_n)=@all(dW_2_1_dt)
    @all(dW_2_2_dt_n)=@all(dW_2_2_dt)
    @all(dW_2_3_dt_n)=@all(dW_2_3_dt)
    @all(dW_2_4_dt_n)=@all(dW_2_4_dt)
    # @all(dW_3_1_dt_n)=@all(dW_3_1_dt)
    @all(dW_3_2_dt_n)=@all(dW_3_2_dt)
    @all(dW_3_3_dt_n)=@all(dW_3_3_dt)
    @all(dW_3_4_dt_n)=@all(dW_3_4_dt)
    # @all(dY_1_dt_n)=@all(dY_1_dt)
    @all(dY_2_dt_n)=@all(dY_2_dt)
    @all(dY_3_dt_n)=@all(dY_3_dt)
    @all(dY_4_dt_n)=@all(dY_4_dt)

    return
end
@views inn(A) = A[2:end-1,2:end-1,2:end-1]

@views function gather(A,A_global,Nx,Ny,Nz,me,comm,nprocs)
	sendbuf=Array(A[2:end-1,2:end-1,1:end-2])
    # nprocs = MPI.Comm_size(comm)
	# println(sendbuf[1,1,1])
    if me!=0
        req=MPI.Isend(sendbuf,0,0,comm)
		MPI.Wait!(req)
    else
		# println(size(sendbuf))
		# println(size(A_global[1:Nx,1:Ny,1:Nz]))
		A_global[1:Nx,1:Ny,1:Nz]=sendbuf
        if nprocs>1
            for p in range(1,nprocs-1,step=1)
                cs = Cint[0,0,0]
                MPI.Cart_coords!(comm,p,cs)
                A_c = zeros(size(sendbuf))
                req = MPI.Irecv!(A_c,p,0,comm)
                MPI.Wait!(req)
                A_global[cs[1]*Nx+1:(cs[1]+1)*Nx,cs[2]*Ny+1:(cs[2]+1)*Ny,cs[3]*Nz+1:(cs[3]+1)*Nz]=A_c
                # println(A_c[1,1,1])
            end
        end
    end
    return
end

@views function gather_fft(A,A_global,Nx,Ny,Nz,me,comm,nprocs)
	sendbuf=Array(A[2:end-1,2:end-1,2:end-1])
    # nprocs = MPI.Comm_size(comm)
	# println(sendbuf[1,1,1])
    if me!=0
        req=MPI.Isend(sendbuf,0,0,comm)
		MPI.Wait!(req)
    else
		# println(size(sendbuf))
		# println(size(A_global[1:Nx,1:Ny,1:Nz]))
		A_global[1:Nx,1:Ny,1:Nz]=sendbuf
        if nprocs>1
            for p in range(1,nprocs-1,step=1)
                cs = Cint[0,0,0]
                MPI.Cart_coords!(comm,p,cs)
                A_c = zeros(ComplexF64,size(sendbuf))
                req = MPI.Irecv!(A_c,p,0,comm)
                MPI.Wait!(req)
                A_global[cs[1]*Nx+1:(cs[1]+1)*Nx,cs[2]*Ny+1:(cs[2]+1)*Ny,cs[3]*Nz+1:(cs[3]+1)*Nz]=A_c
                # println(A_c[1,1,1])
            end
        end
    end
    return
end

@views function run_ev()

# Numerics
nx, ny, nz = latx,laty,latz;                              # Number of gridpoints dimensions x, y and z.

me, dims, nprocs, coords, comm_cart = init_global_grid(nx, ny, nz; periodx=1, periody=1, periodz=1);
println("Device: ", me)
select_device()                                               # select one GPU per MPI local rank (if >1 GPU per node)
# println(device())

# Array initializations
ϕ_1 = @zeros(nx,ny,nz)
ϕ_2 = @zeros(nx,ny,nz)
ϕ_3 = @zeros(nx,ny,nz)
ϕ_4 = @zeros(nx,ny,nz)
# ϕ_1_i = @zeros(nx*dims[1],ny*dims[2],nz*dims[3])
# ϕ_2_i = @zeros(nx*dims[1],ny*dims[2],nz*dims[3])
# ϕ_3_i = @zeros(nx*dims[1],ny*dims[2],nz*dims[3])
# ϕ_4_i = @zeros(nx*dims[1],ny*dims[2],nz*dims[3])
# W_1_1 = @zeros(nx,ny,nz)
W_1_2 = @zeros(nx,ny,nz)
W_1_3 = @zeros(nx,ny,nz)
W_1_4 = @zeros(nx,ny,nz)
# W_2_1 = @zeros(nx,ny,nz)
W_2_2 = @zeros(nx,ny,nz)
W_2_3 = @zeros(nx,ny,nz)
W_2_4 = @zeros(nx,ny,nz)
# W_3_1 = @zeros(nx,ny,nz)
W_3_2 = @zeros(nx,ny,nz)
W_3_3 = @zeros(nx,ny,nz)
W_3_4 = @zeros(nx,ny,nz)
# Y_1 = @zeros(nx,ny,nz)
Y_2 = @zeros(nx,ny,nz)
Y_3 = @zeros(nx,ny,nz)
Y_4 = @zeros(nx,ny,nz)
Γ_1 = @zeros(nx,ny,nz)
Γ_2 = @zeros(nx,ny,nz)
Γ_3 = @zeros(nx,ny,nz)
Σ = @zeros(nx,ny,nz)
E_V = @zeros(nx,ny,nz)
E_kin = @zeros(nx,ny,nz)
E_grad = @zeros(nx,ny,nz)
E_W_e = @zeros(nx,ny,nz)
E_W_m = @zeros(nx,ny,nz)
E_Y_e = @zeros(nx,ny,nz)
E_Y_m = @zeros(nx,ny,nz)
B_x = @zeros(nx,ny,nz)
B_y = @zeros(nx,ny,nz)
B_z = @zeros(nx,ny,nz)

dϕ_1_dt = @zeros(nx,ny,nz)
dϕ_2_dt = @zeros(nx,ny,nz)
dϕ_3_dt = @zeros(nx,ny,nz)
dϕ_4_dt = @zeros(nx,ny,nz)
# dW_1_1_dt = @zeros(nx,ny,nz)
dW_1_2_dt = @zeros(nx,ny,nz)
dW_1_3_dt = @zeros(nx,ny,nz)
dW_1_4_dt = @zeros(nx,ny,nz)
# dW_2_1_dt = @zeros(nx,ny,nz)
dW_2_2_dt = @zeros(nx,ny,nz)
dW_2_3_dt = @zeros(nx,ny,nz)
dW_2_4_dt = @zeros(nx,ny,nz)
# dW_3_1_dt = @zeros(nx,ny,nz)
dW_3_2_dt = @zeros(nx,ny,nz)
dW_3_3_dt = @zeros(nx,ny,nz)
dW_3_4_dt = @zeros(nx,ny,nz)
# dY_1_dt = @zeros(nx,ny,nz)
dY_2_dt = @zeros(nx,ny,nz)
dY_3_dt = @zeros(nx,ny,nz)
dY_4_dt = @zeros(nx,ny,nz)
# dΓ_1_dt = @zeros(nx,ny,nz)
# dΓ_2_dt = @zeros(nx,ny,nz)
# dΓ_3_dt = @zeros(nx,ny,nz)
# dΣ_dt = @zeros(nx,ny,nz)

# Array initializations
ϕ_1_n = @zeros(nx,ny,nz)
ϕ_2_n = @zeros(nx,ny,nz)
ϕ_3_n = @zeros(nx,ny,nz)
ϕ_4_n = @zeros(nx,ny,nz)
# W_1_1_n = @zeros(nx,ny,nz)
W_1_2_n = @zeros(nx,ny,nz)
W_1_3_n = @zeros(nx,ny,nz)
W_1_4_n = @zeros(nx,ny,nz)
# W_2_1_n = @zeros(nx,ny,nz)
W_2_2_n = @zeros(nx,ny,nz)
W_2_3_n = @zeros(nx,ny,nz)
W_2_4_n = @zeros(nx,ny,nz)
# W_3_1_n = @zeros(nx,ny,nz)
W_3_2_n = @zeros(nx,ny,nz)
W_3_3_n = @zeros(nx,ny,nz)
W_3_4_n = @zeros(nx,ny,nz)
# Y_1_n = @zeros(nx,ny,nz)
Y_2_n = @zeros(nx,ny,nz)
Y_3_n = @zeros(nx,ny,nz)
Y_4_n = @zeros(nx,ny,nz)
Γ_1_n = @zeros(nx,ny,nz)
Γ_2_n = @zeros(nx,ny,nz)
Γ_3_n = @zeros(nx,ny,nz)
Σ_n = @zeros(nx,ny,nz)

dϕ_1_dt_n = @zeros(nx,ny,nz)
dϕ_2_dt_n = @zeros(nx,ny,nz)
dϕ_3_dt_n = @zeros(nx,ny,nz)
dϕ_4_dt_n = @zeros(nx,ny,nz)
# dW_1_1_dt_n = @zeros(nx,ny,nz)
dW_1_2_dt_n = @zeros(nx,ny,nz)
dW_1_3_dt_n = @zeros(nx,ny,nz)
dW_1_4_dt_n = @zeros(nx,ny,nz)
# dW_2_1_dt_n = @zeros(nx,ny,nz)
dW_2_2_dt_n = @zeros(nx,ny,nz)
dW_2_3_dt_n = @zeros(nx,ny,nz)
dW_2_4_dt_n = @zeros(nx,ny,nz)
# dW_3_1_dt_n = @zeros(nx,ny,nz)
dW_3_2_dt_n = @zeros(nx,ny,nz)
dW_3_3_dt_n = @zeros(nx,ny,nz)
dW_3_4_dt_n = @zeros(nx,ny,nz)
# dY_1_dt_n = @zeros(nx,ny,nz)
dY_2_dt_n = @zeros(nx,ny,nz)
dY_3_dt_n = @zeros(nx,ny,nz)
dY_4_dt_n = @zeros(nx,ny,nz)
# dΓ_1_dt_n = @zeros(nx,ny,nz)
# dΓ_2_dt_n = @zeros(nx,ny,nz)
# dΓ_3_dt_n = @zeros(nx,ny,nz)
# dΣ_dt_n = @zeros(nx,ny,nz)

#Output array declarations
#No halo local arrays
# E_V_local = zeros(nx-2,ny-2,nz-2)
# E_kin_local = zeros(nx-2,ny-2,nz-2)
# E_grad_local = zeros(nx-2,ny-2,nz-2)
# E_W_e_local = zeros(nx-2,ny-2,nz-2)
# E_W_m_local = zeros(nx-2,ny-2,nz-2)
# E_Y_e_local = zeros(nx-2,ny-2,nz-2)
# E_Y_m_local = zeros(nx-2,ny-2,nz-2)
# B_x_local = zeros(nx-2,ny-2,nz-2)
# B_y_local = zeros(nx-2,ny-2,nz-2)
# B_z_local = zeros(nx-2,ny-2,nz-2)
# E_total_local = zeros(nx-2,ny-2,nz-2)

#Global output array declaration
nx_g, ny_g, nz_g = (nx-2)*dims[1], (ny-2)*dims[2], (nz-2)*dims[3]
E_V_g = zeros(nx_g,ny_g,nz_g)
E_kin_g = zeros(nx_g,ny_g,nz_g)
E_grad_g = zeros(nx_g,ny_g,nz_g)
E_W_e_g = zeros(nx_g,ny_g,nz_g)
E_W_m_g = zeros(nx_g,ny_g,nz_g)
E_Y_e_g = zeros(nx_g,ny_g,nz_g)
E_Y_m_g = zeros(nx_g,ny_g,nz_g)
B_x_g = zeros(nx_g,ny_g,nz_g)
B_y_g = zeros(nx_g,ny_g,nz_g)
B_z_g = zeros(nx_g,ny_g,nz_g)
# E_total_g = zeros(nx_g,ny_g,nz_g)

B_x_fft_g=zeros(ComplexF64,(nx_g,ny_g,nz_g))
B_y_fft_g=zeros(ComplexF64,(nx_g,ny_g,nz_g))
B_z_fft_g=zeros(ComplexF64,(nx_g,ny_g,nz_g))

CUDA.memory_status()

spec_cut = [nx_g÷8,ny_g÷8,nz_g÷8]
N_bins = Kc_bin_nums(spec_cut[1],spec_cut[2],spec_cut[3])
B_fft = zeros((nsnaps+1,N_bins,2))

total_energies = zeros((nsnaps+1,7))

#START Single device initialization #works#

# bubs = []
# for kb in range(bub_diam,stop=latz,step=bub_diam)
#     for jb in range(bub_diam,stop=laty,step=bub_diam)
#         for ib in range(bub_diam,stop=latx,step=bub_diam)
#             phi=gen_phi()
#             push!(bubs,[ib,jb,kb,real(phi[1]),imag(phi[1]),real(phi[2]),imag(phi[2])])
#         end
#     end
# end


# println("# bubbles: ",size(bubs,1))
# Len = (latx-1)*dx
# rk = pi/Len

# @time for b in range(1,size(bubs,1),step=1)
#     ib,jb,kb,p1,p2,p3,p4 = bubs[b]
#     @parallel (1:nx,1:ny,1:nz) initializer!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,ib,jb,kb,p1,p2,p3,p4,rk,dx,mH)
# end

#END:Start single device implementation##########

#Multi device initialization : 052323
seed_value = 1223453134
# no_bubbles = 20

#Uniformlyl spaced bubbles
# xb_locs = range(bub_diam,stop=nx*dims[1],step=bub_diam)
# yb_locs = range(bub_diam,stop=ny*dims[2],step=bub_diam)
# zb_locs = range(bub_diam,stop=nz*dims[3],step=bub_diam)

# Random.seed!(seed_value)
# xb_locs= rand(2:nx_g,no_bubbles)
# Random.seed!(seed_value*2)
# yb_locs=rand(2:ny_g,no_bubbles)
# Random.seed!(seed_value*3)
# zb_locs=rand(2:nz_g,no_bubbles)
# bubble_locs = hcat(xb_locs,yb_locs,zb_locs)

Xb_sample = range(2,nx_g,step=bub_diam)
Yb_sample = range(2,ny_g,step=bub_diam)
Zb_sample = range(2,nz_g,step=bub_diam)

# Random.seed!(seed_value)
# xb_locs= rand(1:nx_g,no_bubbles)
# Random.seed!(seed_value*2)
# yb_locs=rand(2:ny_g,no_bubbles)
# Random.seed!(seed_value*3)
# zb_locs=rand(2:nz_g,no_bubbles)

Random.seed!(seed_value)
xb_loc_idxs= rand(1:size(Xb_sample,1),no_bubbles)
Random.seed!(seed_value*2)
yb_loc_idxs= rand(1:size(Yb_sample,1),no_bubbles)
Random.seed!(seed_value*3)
zb_loc_idxs= rand(1:size(Zb_sample,1),no_bubbles)

xb_locs = [Xb_sample[i] for i in xb_loc_idxs]
yb_locs = [Yb_sample[i] for i in yb_loc_idxs]
zb_locs = [Zb_sample[i] for i in zb_loc_idxs]


bubble_locs = hcat(xb_locs,yb_locs,zb_locs)

Random.seed!(seed_value)
# Hoft_arr = rand(Uniform(0,1),(size(xb_locs,1),size(yb_locs,1),size(zb_locs,1),3))
Hoft_arr = rand(Uniform(0,1),(no_bubbles,3))

# println(Hoft_arr[1,1,1,:])
# for (kb_idx,kb) in enumerate(zb_locs)
#     for (jb_idx,jb) in enumerate(yb_locs)
#         for (ib_idx,ib) in enumerate(xb_locs)
#             # println(string(ib_idx,jb_idx,kb_idx, Hoft_arr[ib_idx,jb_idx,kb_idx,:]));exit()
#             phi=gen_phi(Hoft_arr[ib_idx,jb_idx,kb_idx,:])
#             push!(bubs,[ib,jb,kb,real(phi[1]),imag(phi[1]),real(phi[2]),imag(phi[2])])
#         end
#     end
# end

bubs = []
for bub_idx in range(1,no_bubbles)
    phi=gen_phi(Hoft_arr[bub_idx,:])
    ib,jb,kb = bubble_locs[bub_idx,:]
    # println(string(ib," ",jb," ",kb," ", phi))
    push!(bubs,[ib,jb,kb,real(phi[1]),imag(phi[1]),real(phi[2]),imag(phi[2])])
end

println(string("bubble location matrix",size(bubble_locs)))

println(string("# bubbles: ",size(bubs,1)))
# Len = (latx-1)*dx
# rk = pi/Len
rkx=pi/(nx_g*dx)
rky=pi/(ny_g*dx)
rkz=pi/(nz_g*dx)

@time for b in range(1,size(bubs,1),step=1)
    ib,jb,kb,p1,p2,p3,p4 = bubs[b]
    # @parallel (1:nx*dims[1],1:ny*dims[2],1:nz*dims[3]) initializer!(ϕ_1_i,ϕ_2_i,ϕ_3_i,ϕ_4_i,ib,jb,kb,p1,p2,p3,p4,rkx,rky,rkz,dx,mH)
    @parallel (1:nx,1:ny,1:nz) initializer!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,ib,jb,kb,p1,p2,p3,p4,rkx,rky,rkz,dx,mH,coords[1],coords[2],coords[3])
end

# i_l = (coords[1])*nx + 1
# i_r = (coords[1])*nx + nx
# j_b = (coords[2])*ny + 1
# j_f = (coords[2])*ny + ny
# k_d = (coords[3])*nz + 1
# k_u = (coords[3])*nz + nz

# ϕ_1 .= ϕ_1_i[i_l:i_r,j_b:j_f,k_d:k_u]
# ϕ_2 .= ϕ_2_i[i_l:i_r,j_b:j_f,k_d:k_u]
# ϕ_3 .= ϕ_3_i[i_l:i_r,j_b:j_f,k_d:k_u]
# ϕ_4 .= ϕ_4_i[i_l:i_r,j_b:j_f,k_d:k_u]

# println(string(typeof(ϕ_1)," ",typeof(ϕ_2)," ",typeof(ϕ_3)," ",typeof(ϕ_4)," "))
# println(string(size(ϕ_1)," ",size(ϕ_2)," ",size(ϕ_3)," ",size(ϕ_4)," "));exit()
# @parallel renormalize!(ϕ_1,ϕ_2,ϕ_3,ϕ_4)

ϕ_1_n.=ϕ_1
ϕ_2_n.=ϕ_2
ϕ_3_n.=ϕ_3
ϕ_4_n.=ϕ_4

#Freeing up initializing array memory

# CUDA.unsafe_free!(ϕ_1_i)
# CUDA.unsafe_free!(ϕ_2_i)
# CUDA.unsafe_free!(ϕ_3_i)
# CUDA.unsafe_free!(ϕ_4_i)

CUDA.memory_status()

#Routines to compute initial energies#
@parallel E_potential!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,E_V,lambda,vev)

@parallel E_kinetic!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,
Y_2,Y_3,Y_4,
dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
E_kin)

@parallel E_gradient!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4,
E_grad,dx)

@parallel E_W_electric!(dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,E_W_e)

@parallel E_W_magnetic!(W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,gw,dx,E_W_m)

@parallel E_Y_electric!(dY_2_dt,dY_3_dt,dY_4_dt,E_Y_e)

@parallel E_Y_magnetic!(Y_2,Y_3,Y_4,E_Y_m)

@parallel Magnetic_fields!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
W_1_2,W_1_3,W_1_4,
W_2_2,W_2_3,W_2_4,
W_3_2,W_3_3,W_3_4,
Y_2,Y_3,Y_4,
gw,sin(Wangle),cos(Wangle),vev,B_x,B_y,B_z,dx)

# tic()
# E_V_local .= Array(inn(E_V)); gather!(E_V_local,E_V_g)
# E_kin_local .= Array(inn(E_kin)); gather!(E_kin_local,E_kin_g)
# E_grad_local .= Array(inn(E_grad)); gather!(E_grad_local,E_grad_g)
# E_W_e_local .= Array(inn(E_W_e)); gather!(E_W_e_local,E_W_e_g)
# E_W_m_local .= Array(inn(E_W_m)); gather!(E_W_m_local,E_W_m_g)
# E_Y_e_local .= Array(inn(E_Y_e)); gather!(E_Y_e_local,E_Y_e_g)
# E_Y_m_local .= Array(inn(E_Y_m)); gather!(E_Y_m_local, E_Y_m_g)
# B_x_local .= Array(inn(B_x)); gather!(B_x_local,B_x_g)
# B_y_local .= Array(inn(B_y)); gather!(B_y_local,B_y_g)
# B_z_local .= Array(inn(B_z)); gather!(B_z_local,B_z_g)
# E_total_local .= Array(inn(E_V))+Array(inn(E_kin))+Array(inn(E_grad))+Array(inn(E_W_e))+Array(inn(E_W_m))+Array(inn(E_Y_e))+Array(inn(E_Y_m))
# gather!(E_total_local,E_total_g)
# println("Initial Gather time: ", toc())

### Teerthal:May 1 23
### THe following output commands are causing race conditions on multiple gpus.
### in built synhronize funciton does not seem to work or i don;t know how to get it to work
### An alternative would be to output from individual devices and then merge them in post
### another would be to get each device to output deivce coded global array files. this would be 
### higly inefficient in terms of storage as well as having to use intermediatry gather statements
### that occupy significant compute times

# if (me==0)
# tic()
# println(string("--------Energies--t: 0 --process:",me,"----------"))
# println("Potentianl energy Higgs: ",sum(Array(inn(E_V))))
# println("Kinetic energy Higgs: ",sum(Array(inn(E_kin))))
# println("Gradient energy Higgs:",sum(Array(inn(E_grad))))
# println("Electric energy W: ",sum(Array(inn(E_W_e))))
# println("Magnetic energy W: ",sum(Array(inn(E_W_m))))
# println("Electric energy Y: ",sum(Array(inn(E_Y_e))))
# println("Magnetic energy Y: ",sum(Array(inn(E_Y_m))))
# println("Total energy: ", sum(Array(inn(E_V))+Array(inn(E_kin))+Array(inn(E_grad))+Array(inn(E_W_e))+Array(inn(E_W_m))+Array(inn(E_Y_e))+Array(inn(E_Y_m))))
# println("---------------------------------------")

# end

#Gather energies
MPI.Barrier(comm_cart)
gather(E_kin,E_kin_g,nx-2,ny-2,nz-2,me,comm_cart,nprocs)
gather(E_V,E_V_g,nx-2,ny-2,nz-2,me,comm_cart,nprocs)
gather(E_grad,E_grad_g,nx-2,ny-2,nz-2,me,comm_cart,nprocs)
gather(E_W_m,E_W_m_g,nx-2,ny-2,nz-2,me,comm_cart,nprocs)
gather(E_W_e,E_W_e_g,nx-2,ny-2,nz-2,me,comm_cart,nprocs)
gather(E_Y_m,E_Y_m_g,nx-2,ny-2,nz-2,me,comm_cart,nprocs)
gather(E_Y_e,E_Y_e_g,nx-2,ny-2,nz-2,me,comm_cart,nprocs)
MPI.Barrier(comm_cart)

total_energies[1,1] = sum(Array(E_V_g))
total_energies[1,2] = sum(Array(E_kin_g))
total_energies[1,3] = sum(Array(E_grad_g))
total_energies[1,4] = sum(Array(E_W_m_g))
total_energies[1,5] = sum(Array(E_W_e_g))
total_energies[1,6] = sum(Array(E_Y_m_g))
total_energies[1,7] = sum(Array(E_Y_e_g))

# Compute fft and convolve spectrum
@time begin
    B_x_fft = fft(Array(B_x))
    B_y_fft = fft(Array(B_y))
    B_z_fft = fft(Array(B_z))

    gather_fft(B_x_fft,B_x_fft_g,nx-2,ny-2,nz-2,me,comm_cart,nprocs)
    gather_fft(B_y_fft,B_y_fft_g,nx-2,ny-2,nz-2,me,comm_cart,nprocs)
    gather_fft(B_z_fft,B_z_fft_g,nx-2,ny-2,nz-2,me,comm_cart,nprocs)
    MPI.Barrier(comm_cart)

    B_fft[1,:,:] = convolve_1d((real(conj.(B_x_fft_g).*B_x_fft_g.+
    conj.(B_y_fft_g).*B_y_fft_g.+
    conj.(B_z_fft_g).*B_z_fft_g)),nx_g,ny_g,nz_g,spec_cut[1],spec_cut[2],spec_cut[3])
end
    

####Initialize and plot first frame####

if me==0

    println(string("--------Energies--t:",0,"--process:",me,"----------"))
    println("Potentianl energy Higgs: ",total_energies[1,1])
    println("Kinetic energy Higgs: ",total_energies[1,2])
    println("Gradient energy Higgs:",total_energies[1,3])
    println("Magnetic energy W: ",total_energies[1,4])
    println("Electric energy W: ",total_energies[1,5])
    println("Magnetic energy Y: ",total_energies[1,6])
    println("Electric energy Y: ",total_energies[1,7])
    println("Total energy: ", sum(total_energies[1,:]))
    println("---------------------------------------")

    ##PLOT##
    x=range(1,nx_g,step=1)
    y=range(1,ny_g,step=1)
    z=range(1,nz_g,step=1)
    println(size(x),size(y),size(z))
    println(size(Array(E_V)[:,ny_g÷2,:]))
    gr()
    ENV["GKSwstype"]="nul"
    anim = Animation();
    # plot_1=contourf(z,x,(Array(ϕ_1)[:,Ny÷2,:]).^2+(Array(ϕ_2)[:,Ny÷2,:]).^2+(Array(ϕ_3)[:,Ny÷2,:]).^2+(Array(ϕ_4)[:,Ny÷2,:]).^2)
    plot_1=contourf(z,x,(Array(E_V_g)[:,ny_g÷2,:]),title="PE")
    plot_2=contourf(z,x,(Array(E_kin_g)[:,ny_g÷2,:]),title="KE")
    plot_3=contourf(z,x,(Array(E_grad_g)[:,ny_g÷2,:]),title="GE")
    # plot_3=plot(B_fft[1,2:end,1],(((B_fft[1,2:end,1]).^2)./((2*pi)^3*nx_g^2)).*B_fft[1,2:end,2],xscale=:log10,yscale=:log10,minorgrid=true)
    # plot_4=contourf(z,x,(Array(E_W_e_g)[:,ny_g÷2,:]+Array(E_W_m_g)[:,ny_g÷2,:]+Array(E_Y_e_g)[:,ny_g÷2,:]+Array(E_Y_m_g)[:,ny_g÷2,:]),title="WY E")
    plot_4 = scatter([0],[total_energies[1,1] total_energies[1,2] total_energies[1,3] total_energies[1,4] total_energies[1,5] total_energies[1,6] total_energies[1,7]],
    label=["PE" "KE" "GE" "MEW" "EEW" "MEY" "EEY"],xlims=(0,nt))
    plot(plot_1,plot_2,plot_3,plot_4,layout=4,plot_title=string("it:",0),dpi=600)
    # plot(plot_1,title=string("it:",0),dpi=600)
    png(string("testini1",".png"))
    frame(anim)

end
MPI.Barrier(comm_cart)

##############End-first-frame_plot###

# vtk_grid("raw_0",x,y,z) do vtk
#     vtk["V"] = Float32.(Array(E_V))
# end

# h5open(string("raw_0_",me,".h5"), "w") do file
#     write(file, "E_V", Float32.(Array(inn(E_V))))  # alternatively, say "@write file A"
#     write(file, "E", Float32.(Array(inn(E_V))+Array(inn(E_kin))+Array(inn(E_grad))+Array(inn(E_W_e))+Array(inn(E_W_m))+Array(inn(E_Y_e))+Array(inn(E_Y_m))))
#     # write(file, "B_x_fft", ComplexF32.(Array(fft(B_x))))
#     # write(file, "B_y_fft", ComplexF32.(Array(fft(B_y))))
#     # write(file, "B_z_fft", ComplexF32.(Array(fft(B_z))))
#     write(file, "B_x", Float32.(Array(inn(B_x))))
#     write(file, "B_y", Float32.(Array(inn(B_x))))
#     write(file, "B_z", Float32.(Array(inn(B_x))))
# end
# println("Initial save time: ", toc())
# end

# @synchronize

println("Initialized")

#Counter for snaps
snp_idx = 1

for i in range(1,nt,step=1)
    
    # tic()

    @parallel evolve!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    Γ_1,Γ_2,Γ_3,Σ,
    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    ϕ_1_n,ϕ_2_n,ϕ_3_n,ϕ_4_n,
    W_1_2_n,W_1_3_n,W_1_4_n,
    W_2_2_n,W_2_3_n,W_2_4_n,
    W_3_2_n,W_3_3_n,W_3_4_n,
    Y_2_n,Y_3_n,Y_4_n,
    Γ_1_n,Γ_2_n,Γ_3_n,Σ_n,
    gw,gy,gp2,lambda,vev,dx,dt)

    @parallel evolve_dot!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    Γ_1,Γ_2,Γ_3,Σ,
    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    dϕ_1_dt_n,dϕ_2_dt_n,dϕ_3_dt_n,dϕ_4_dt_n,
    dW_1_2_dt_n,dW_1_3_dt_n,dW_1_4_dt_n,
    dW_2_2_dt_n,dW_2_3_dt_n,dW_2_4_dt_n,
    dW_3_2_dt_n,dW_3_3_dt_n,dW_3_4_dt_n,
    dY_2_dt_n,dY_3_dt_n,dY_4_dt_n,
    gw,gy,gp2,lambda,vev,dx,dt)

    @parallel avg_half_step!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    Γ_1,Γ_2,Γ_3,Σ,
    ϕ_1_n,ϕ_2_n,ϕ_3_n,ϕ_4_n,
    W_1_2_n,W_1_3_n,W_1_4_n,
    W_2_2_n,W_2_3_n,W_2_4_n,
    W_3_2_n,W_3_3_n,W_3_4_n,
    Y_2_n,Y_3_n,Y_4_n,
    Γ_1_n,Γ_2_n,Γ_3_n,Σ_n)

    @parallel avg_half_step_dot!(dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    dϕ_1_dt_n,dϕ_2_dt_n,dϕ_3_dt_n,dϕ_4_dt_n,
    dW_1_2_dt_n,dW_1_3_dt_n,dW_1_4_dt_n,
    dW_2_2_dt_n,dW_2_3_dt_n,dW_2_4_dt_n,
    dW_3_2_dt_n,dW_3_3_dt_n,dW_3_4_dt_n,
    dY_2_dt_n,dY_3_dt_n,dY_4_dt_n)

    @parallel leaforward!(ϕ_1_n,ϕ_2_n,ϕ_3_n,ϕ_4_n,
    W_1_2_n,W_1_3_n,W_1_4_n,
    W_2_2_n,W_2_3_n,W_2_4_n,
    W_3_2_n,W_3_3_n,W_3_4_n,
    Y_2_n,Y_3_n,Y_4_n,
    Γ_1_n,Γ_2_n,Γ_3_n,Σ_n,
    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    Γ_1,Γ_2,Γ_3,Σ,
    gw,gy,gp2,lambda,vev,dx,dt)

    @parallel leaforward_dot!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    Γ_1,Γ_2,Γ_3,Σ,
    dϕ_1_dt_n,dϕ_2_dt_n,dϕ_3_dt_n,dϕ_4_dt_n,
    dW_1_2_dt_n,dW_1_3_dt_n,dW_1_4_dt_n,
    dW_2_2_dt_n,dW_2_3_dt_n,dW_2_4_dt_n,
    dW_3_2_dt_n,dW_3_3_dt_n,dW_3_4_dt_n,
    dY_2_dt_n,dY_3_dt_n,dY_4_dt_n,
    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    gw,gy,gp2,lambda,vev,dx,dt)
    
    @parallel avg_half_step!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    Γ_1,Γ_2,Γ_3,Σ,
    ϕ_1_n,ϕ_2_n,ϕ_3_n,ϕ_4_n,
    W_1_2_n,W_1_3_n,W_1_4_n,
    W_2_2_n,W_2_3_n,W_2_4_n,
    W_3_2_n,W_3_3_n,W_3_4_n,
    Y_2_n,Y_3_n,Y_4_n,
    Γ_1_n,Γ_2_n,Γ_3_n,Σ_n)

    @parallel avg_half_step_dot!(dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    dϕ_1_dt_n,dϕ_2_dt_n,dϕ_3_dt_n,dϕ_4_dt_n,
    dW_1_2_dt_n,dW_1_3_dt_n,dW_1_4_dt_n,
    dW_2_2_dt_n,dW_2_3_dt_n,dW_2_4_dt_n,
    dW_3_2_dt_n,dW_3_3_dt_n,dW_3_4_dt_n,
    dY_2_dt_n,dY_3_dt_n,dY_4_dt_n)

    # @hide_communication (16, 8, 4) begin # communication/computation overlap

    @parallel leaforward!(ϕ_1_n,ϕ_2_n,ϕ_3_n,ϕ_4_n,
    W_1_2_n,W_1_3_n,W_1_4_n,
    W_2_2_n,W_2_3_n,W_2_4_n,
    W_3_2_n,W_3_3_n,W_3_4_n,
    Y_2_n,Y_3_n,Y_4_n,
    Γ_1_n,Γ_2_n,Γ_3_n,Σ_n,
    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    Γ_1,Γ_2,Γ_3,Σ,
    gw,gy,gp2,lambda,vev,dx,dt)

    # end

    # @hide_communication (16, 8, 4) begin # communication/computation overlap

    @parallel leaforward_dot!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    Γ_1,Γ_2,Γ_3,Σ,
    dϕ_1_dt_n,dϕ_2_dt_n,dϕ_3_dt_n,dϕ_4_dt_n,
    dW_1_2_dt_n,dW_1_3_dt_n,dW_1_4_dt_n,
    dW_2_2_dt_n,dW_2_3_dt_n,dW_2_4_dt_n,
    dW_3_2_dt_n,dW_3_3_dt_n,dW_3_4_dt_n,
    dY_2_dt_n,dY_3_dt_n,dY_4_dt_n,
    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    gw,gy,gp2,lambda,vev,dx,dt)

    # println("Evolution time: ", toc())

    # tic()
    # println(string("before,",me," ",ϕ_1_n[1:3,ny÷2,nz÷2], " ",ϕ_1_n[end-2:end,ny÷2,nz÷2]))
    update_halo!(dϕ_1_dt_n)
    update_halo!(dϕ_2_dt_n)
    update_halo!(dϕ_3_dt_n)
    update_halo!(dϕ_4_dt_n)
    # update_halo!(dW_1_1_dt_n)
    update_halo!(dW_1_2_dt_n)
    update_halo!(dW_1_3_dt_n)
    update_halo!(dW_1_4_dt_n)
    # update_halo!(dW_2_1_dt_n)
    update_halo!(dW_2_2_dt_n)
    update_halo!(dW_2_3_dt_n)
    update_halo!(dW_2_4_dt_n)
    # update_halo!(dW_3_1_dt_n)
    update_halo!(dW_3_2_dt_n)
    update_halo!(dW_3_3_dt_n)
    update_halo!(dW_3_4_dt_n)
    # update_halo!(dY_1_dt_n)
    update_halo!(dY_2_dt_n)
    update_halo!(dY_3_dt_n)
    update_halo!(dY_4_dt_n)
    # update_halo!(dΓ_1_dt_n)
    # update_halo!(dΓ_2_dt_n)
    # update_halo!(dΓ_3_dt_n)
    # update_halo!(dΣ_dt_n)

    update_halo!(ϕ_1_n)
    update_halo!(ϕ_2_n)
    update_halo!(ϕ_3_n)
    update_halo!(ϕ_4_n)
    # update_halo!(W_1_1_n)
    update_halo!(W_1_2_n)
    update_halo!(W_1_3_n)
    update_halo!(W_1_4_n)
    # update_halo!(W_2_1_n)
    update_halo!(W_2_2_n)
    update_halo!(W_2_3_n)
    update_halo!(W_2_4_n)
    # update_halo!(W_3_1_n)
    update_halo!(W_3_2_n)
    update_halo!(W_3_3_n)
    update_halo!(W_3_4_n)
    # update_halo!(Y_1_n)
    update_halo!(Y_2_n)
    update_halo!(Y_3_n)
    update_halo!(Y_4_n)
    update_halo!(Γ_1_n)
    update_halo!(Γ_2_n)
    update_halo!(Γ_3_n)
    update_halo!(Σ_n)
    # println(string("after,",me," ",ϕ_1_n[1:3,ny÷2,nz÷2], " ",ϕ_1_n[end-2:end,ny÷2,nz÷2]))
    # exit()
    # println("Halo updatetime: ",toc())

    @parallel name_change!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    Γ_1,Γ_2,Γ_3,Σ,
    ϕ_1_n,ϕ_2_n,ϕ_3_n,ϕ_4_n,
    W_1_2_n,W_1_3_n,W_1_4_n,
    W_2_2_n,W_2_3_n,W_2_4_n,
    W_3_2_n,W_3_3_n,W_3_4_n,
    Y_2_n,Y_3_n,Y_4_n,
    Γ_1_n,Γ_2_n,Γ_3_n,Σ_n)
    
    @parallel name_change_dot!(dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    dϕ_1_dt_n,dϕ_2_dt_n,dϕ_3_dt_n,dϕ_4_dt_n,
    dW_1_2_dt_n,dW_1_3_dt_n,dW_1_4_dt_n,
    dW_2_2_dt_n,dW_2_3_dt_n,dW_2_4_dt_n,
    dW_3_2_dt_n,dW_3_3_dt_n,dW_3_4_dt_n,
    dY_2_dt_n,dY_3_dt_n,dY_4_dt_n)

    # println("var switch time: ",toc())
    # exit()
    ##Write raw data##
    
    if mod(i,dsnaps)==0

        CUDA.memory_status()

        # tic()
        #Routines to compute energies#
        @parallel E_potential!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,E_V,lambda,vev)

        @parallel E_kinetic!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
        W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,
        Y_2,Y_3,Y_4,
        dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
        E_kin)

        @parallel E_gradient!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
        W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4,
        E_grad,dx)

        @parallel E_W_electric!(dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
        dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
        dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,E_W_e)

        @parallel E_W_magnetic!(W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,gw,dx,E_W_m)

        @parallel E_Y_electric!(dY_2_dt,dY_3_dt,dY_4_dt,E_Y_e)

        @parallel E_Y_magnetic!(Y_2,Y_3,Y_4,E_Y_m)
                
        #Compute Magnetic fields
        @parallel Magnetic_fields!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
        W_1_2,W_1_3,W_1_4,
        W_2_2,W_2_3,W_2_4,
        W_3_2,W_3_3,W_3_4,
        Y_2,Y_3,Y_4,
        gw,sin(Wangle),cos(Wangle),vev,B_x,B_y,B_z,dx)
        
        # println("Energy/Mag field compute time: ", toc())
        
        # tic()
        #Gather energies
        MPI.Barrier(comm_cart)
        gather(E_kin,E_kin_g,nx-2,ny-2,nz-2,me,comm_cart,nprocs)
        gather(E_V,E_V_g,nx-2,ny-2,nz-2,me,comm_cart,nprocs)
        gather(E_grad,E_grad_g,nx-2,ny-2,nz-2,me,comm_cart,nprocs)
        gather(E_W_m,E_W_m_g,nx-2,ny-2,nz-2,me,comm_cart,nprocs)
        gather(E_W_e,E_W_e_g,nx-2,ny-2,nz-2,me,comm_cart,nprocs)
        gather(E_Y_m,E_Y_m_g,nx-2,ny-2,nz-2,me,comm_cart,nprocs)
        gather(E_Y_e,E_Y_e_g,nx-2,ny-2,nz-2,me,comm_cart,nprocs)
        MPI.Barrier(comm_cart)
        # println("Gather time",toc())

        snp_idx = snp_idx+1

        total_energies[snp_idx,1] = sum(Array(E_V_g))
        total_energies[snp_idx,2] = sum(Array(E_kin_g))
        total_energies[snp_idx,3] = sum(Array(E_grad_g))
        total_energies[snp_idx,4] = sum(Array(E_W_m_g))
        total_energies[snp_idx,5] = sum(Array(E_W_e_g))
        total_energies[snp_idx,6] = sum(Array(E_Y_m_g))
        total_energies[snp_idx,7] = sum(Array(E_Y_e_g))

        ####Initialize and plot first frame####

        if me==0

            println(string("--------Energies--t:",i,"--process:",me,"----------"))
            println("Potentianl energy Higgs: ",total_energies[snp_idx,1])
            println("Kinetic energy Higgs: ",total_energies[snp_idx,2])
            println("Gradient energy Higgs:",total_energies[snp_idx,3])
            println("Magnetic energy W: ",total_energies[snp_idx,4])
            println("Electric energy W: ",total_energies[snp_idx,5])
            println("Magnetic energy Y: ",total_energies[snp_idx,6])
            println("Electric energy Y: ",total_energies[snp_idx,7])
            println("Total energy: ", sum(total_energies[snp_idx,:]))
            println("---------------------------------------")

            # plot_1=contourf(z,x,(Array(ϕ_1)[:,Ny÷2,:]).^2+(Array(ϕ_2)[:,Ny÷2,:]).^2+(Array(ϕ_3)[:,Ny÷2,:]).^2+(Array(ϕ_4)[:,Ny÷2,:]).^2)
            plot_1=contourf(z,x,(Array(E_V_g)[:,ny_g÷2,:]),title="PE")
            plot_2=contourf(z,x,(Array(E_kin_g)[:,ny_g÷2,:]),title="KE")
            plot_3=contourf(z,x,(Array(E_grad_g)[:,ny_g÷2,:]),title="GE")
            # plot_3=plot(B_fft[1,2:end,1],(((B_fft[1,2:end,1]).^2)./((2*pi)^3*Nx^2)).*B_fft[1,2:end,2],xscale=:log10,yscale=:log10,minorgrid=true)
            # plot_4=contourf(z,x,(Array(E_W_e_g)[:,ny_g÷2,:]+Array(E_W_m_g)[:,ny_g÷2,:]+Array(E_Y_e_g)[:,ny_g÷2,:]+Array(E_Y_m_g)[:,ny_g÷2,:]),title="WY E")
            plot_4 = plot(range(1,i+1,step=dsnaps),[total_energies[1:snp_idx,1] total_energies[1:snp_idx,2] total_energies[1:snp_idx,3] total_energies[1:snp_idx,4] total_energies[1:snp_idx,5] total_energies[1:snp_idx,6] total_energies[1:snp_idx,7]],
            label=["PE" "KE" "GE" "MEW" "EEW" "MEY" "EEY"],xlims=(0,nt))
            plot(plot_1,plot_2,plot_3,plot_4,layout=4,plot_title=string("it:",i),dpi=600)
            # plot(plot_1,title=string("it:",0),dpi=600)
            # png(string("testini1",".png"))
            frame(anim)

        end
        MPI.Barrier(comm_cart)

        # if (me==0)
            
        # tic()
        ###Loading the arrays from the GPU to the CPU 
        ###is a severe bottleneck.
        ###Prudent to decrease the frequency of the  energy addition operation
        ###or just dumping the energies onto files which is faster

        
        # println(string("--------Energies--t:",i,"--process:",me,"----------"))
        # println("Potentianl energy Higgs: ",sum(Array(inn(E_V))))
        # println("Kinetic energy Higgs: ",sum(Array(inn(E_kin))))
        # println("Gradient energy Higgs:",sum(Array(inn(E_grad))))
        # println("Electric energy W: ",sum(Array(inn(E_W_e))))
        # println("Magnetic energy W: ",sum(Array(inn(E_W_m))))
        # println("Electric energy Y: ",sum(Array(inn(E_Y_e))))
        # println("Magnetic energy Y: ",sum(Array(inn(E_Y_m))))
        # println("Total energy: ", sum(Array(inn(E_V))+Array(inn(E_kin))+Array(inn(E_grad))+Array(inn(E_W_e))+Array(inn(E_W_m))+Array(inn(E_Y_e))+Array(inn(E_Y_m))))
        # println("---------------------------------------")
        # println("------time to add energies---",toc(),"------")

        # tic()
        # h5open(string("raw_",i,".h5"), "w") do file
        # h5open(string("raw_",i,"_",me,".h5"), "w") do file
        #     write(file, "E_V", Float32.(Array(inn(E_V))))  # alternatively, say "@write file A"
        #     write(file, "E", Float32.(Array(inn(E_V))+Array(inn(E_kin))+Array(inn(E_grad))+Array(inn(E_W_e))+Array(inn(E_W_m))+Array(inn(E_Y_e))+Array(inn(E_Y_m))))
        #     # write(file, "B_x_fft", ComplexF32.(Array(fft(B_x))))
        #     # write(file, "B_y_fft", ComplexF32.(Array(fft(B_y))))
        #     # write(file, "B_z_fft", ComplexF32.(Array(fft(B_z))))
        #     write(file, "B_x", Float32.(Array(inn(B_x))))
        #     write(file, "B_y", Float32.(Array(inn(B_x))))
        #     write(file, "B_z", Float32.(Array(inn(B_x))))
        # end  
        
        #HDF5 saving is far faster than directly writing npz/vtk
        #Storage space is larger though
        
        # println("time to save: ",toc())
        # tic()
        # vtk_grid(string("raw_",i),x,y,z) do vtk
        #     vtk["V"] = Float32.(Array(E_V))
        # end
        # println("time to save: ",toc())            

        # end

        # @synchronize
            
    end
    
    ##End Write##

end

# if (me==0) gif(anim, "EW3d_test.gif", fps = 10) end


if me==0
    gif(anim, "EW3d_test.mp4", fps = FPS)

    # println("test:",Array(ϕ_1)[3,5,3])
    # CUDA.memory_status()

    gr()
    ENV["GKSwstype"]="nul"
    plot(range(0,nt,step=dsnaps),[total_energies[:,1] total_energies[:,2] total_energies[:,3] total_energies[:,4] total_energies[:,5] total_energies[:,6] total_energies[:,7]].+1.0,
    label=["PE" "KE" "GE" "EEW" "MEW" "EEY" "MEY"],yscale=:log10,dpi=600)
    png("energies.png")

end

MPI.Barrier(comm_cart)

# Compute fft and convolve spectrum
@time begin
    B_x_fft = fft(Array(B_x))
    B_y_fft = fft(Array(B_y))
    B_z_fft = fft(Array(B_z))

    gather_fft(B_x_fft,B_x_fft_g,nx-2,ny-2,nz-2,me,comm_cart,nprocs)
    gather_fft(B_y_fft,B_y_fft_g,nx-2,ny-2,nz-2,me,comm_cart,nprocs)
    gather_fft(B_z_fft,B_z_fft_g,nx-2,ny-2,nz-2,me,comm_cart,nprocs)
    MPI.Barrier(comm_cart)

    B_fft[end,:,:] = convolve_1d((real(conj.(B_x_fft_g).*B_x_fft_g.+
    conj.(B_y_fft_g).*B_y_fft_g.+
    conj.(B_z_fft_g).*B_z_fft_g)),nx_g,ny_g,nz_g,spec_cut[1],spec_cut[2],spec_cut[3])
end
    
if me==0
    gr()
    ENV["GKSwstype"]="nul"
    y1 = (((B_fft[1,2:end,1]).^2)./((2*pi)^3*nx_g^2)).*B_fft[1,2:end,2]
    y2 = (((B_fft[end,2:end,1]).^2)./((2*pi)^3*nx_g^2)).*B_fft[end,2:end,2]
    plot(B_fft[end,2:end,1],[y1 y2],label=[0 nt],xscale=:log10,yscale=:log10,minorgrid=true)
    png("spectra.png")
end

finalize_global_grid();
end

@time run_ev()

# using Profile
# @profile run_ev()
