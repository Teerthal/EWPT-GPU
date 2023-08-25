#Extending rk4 script to Runge-Kutte-Fehlberg(RKF45) to introduce adaptive stepsizes
#Butcher tableau coefficients can be adjusted to other 4th-5th order adapative recipies

using CUDA#, CuArrays
using Random
using StatsBase
using Distributions
using Plots
using CUDA.CUFFT
using Statistics
using MPI

# Initialize MPI
# MPI.Init()

# # Get the number of available CUDA devices and rank of the current process
# num_devices = CUDA.device_count()
# rank = MPI.Comm_rank(MPI.COMM_WORLD)

# # Set the active CUDA device based on the process rank
# CUDA.device!(rank % num_devices + 1)

# CUDA.memory_status()
# @views d_xa(A) = A[2:end  , :     , :     ] .- A[1:end-1, :     , :     ];
# @views d_xi(A) = A[2:end  ,2:end-1,2:end-1] .- A[1:end-1,2:end-1,2:end-1];
# @views d_ya(A) = A[ :     ,2:end  , :     ] .- A[ :     ,1:end-1, :     ];
# @views d_yi(A) = A[2:end-1,2:end  ,2:end-1] .- A[2:end-1,1:end-1,2:end-1];
# @views d_za(A) = A[ :     , :     ,2:end  ] .- A[ :     , :     ,1:end-1];
# @views d_zi(A) = A[2:end-1,2:end-1,2:end  ] .- A[2:end-1,2:end-1,1:end-1];
# @views  inn(A) = A[2:end-1,2:end-1,2:end-1]
# @inbounds @views macro d_xa(A) esc(:( ($A[2:end  , :     ] .- $A[1:end-1, :     ]) )) end

# @views function diffx(A,i,j,k)
#     return A[i,j,k]-A[i+1,j,k]
# end

# @views diffx_inn(A,i,j,k) = A[i+2,j,k]+A[i+1,j,k]-A[i-1,j,k]-A[i-2,j,k]
# @views diffx_2(A,i,j,k) = A[i+1,j,k]-A[i-1,j,k]
# @views diffx_bound(A,i,j,k) = A[i,j,k]-A[i-1,j,k]

include("diff_scheme.jl")
using .differentiations

include("coordinates.jl")
using .coords

include("cov_derivs.jl")
using .covariant_derivatives

include("field_strengths.jl")
using .f_strengths

include("convenients.jl")
using .convenience_functions

include("spec_routines.jl")
using .spec_convolver

# include("evolve_euler.jl")
# using .ev_spatial

# include("spatial_fluxes.jl")
# using .r_expressions

# diff_method = "abc"
# if diff_method == "abc"
#     diff = diff_abc
# end

# function rk4_step!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
#     W_1_1,W_1_2,W_1_3,W_1_4,
#     W_2_1,W_2_2,W_2_3,W_2_4,
#     W_3_1,W_3_2,W_3_3,W_3_4,
#     Y_1,Y_2,Y_3,Y_4,
#     Γ_1,Γ_2,Γ_3,Σ,
#     k_ϕ_1,k_ϕ_2,k_ϕ_3,k_ϕ_4,
#     k_W_1_1,k_W_1_2,k_W_1_3,k_W_1_4,
#     k_W_2_1,k_W_2_2,k_W_2_3,k_W_2_4,
#     k_W_3_1,k_W_3_2,k_W_3_3,k_W_3_4,
#     k_Y_1,k_Y_2,k_Y_3,k_Y_4,
#     k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,
#     dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
#     dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
#     dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
#     dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
#     dY_2_dt,dY_3_dt,dY_4_dt,
#     kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
#     kt_W_1_2,kt_W_1_3,kt_W_1_4,
#     kt_W_2_2,kt_W_2_3,kt_W_2_4,
#     kt_W_3_2,kt_W_3_3,kt_W_3_4,
#     kt_Y_2,kt_Y_3,kt_Y_4,
#     kt_Γ_1,kt_Γ_2,kt_Γ_3,kt_Σ,
#     gw,gy,gp2,vev,lambda,dx)
#     i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
#               (blockIdx().y - 1) * blockDim().y + threadIdx().y,
#               (blockIdx().z - 1) * blockDim().z + threadIdx().z
    
#     #Spatial Derivatives
#     dϕ_1_dx=dfdx(ϕ_1,i,j,k,0.,dx)
#     dϕ_2_dx=dfdx(ϕ_2,i,j,k,0.,dx)
#     dϕ_3_dx=dfdx(ϕ_3,i,j,k,0.,dx)
#     dϕ_4_dx=dfdx(ϕ_4,i,j,k,0.,dx)
#     # @cuprintln(dϕ_4_dx)
#     dϕ_1_dy=dfdy(ϕ_1,i,j,k,0.,dx)
#     dϕ_2_dy=dfdy(ϕ_2,i,j,k,0.,dx)
#     dϕ_3_dy=dfdy(ϕ_3,i,j,k,0.,dx)
#     dϕ_4_dy=dfdy(ϕ_4,i,j,k,0.,dx)

#     dϕ_1_dz=dfdz(ϕ_1,i,j,k,0.,dx)
#     dϕ_2_dz=dfdz(ϕ_2,i,j,k,0.,dx)
#     dϕ_3_dz=dfdz(ϕ_3,i,j,k,0.,dx)
#     dϕ_4_dz=dfdz(ϕ_4,i,j,k,0.,dx)

#     dW_1_1_dx = dfdx(W_1_1,i,j,k,0.,dx)
#     dW_1_2_dx = dfdx(W_1_2,i,j,k,0.,dx)
#     dW_1_3_dx = dfdx(W_1_3,i,j,k,0.,dx)
#     dW_1_4_dx = dfdx(W_1_4,i,j,k,0.,dx)

#     dW_1_1_dy = dfdy(W_1_1,i,j,k,0.,dx)
#     dW_1_2_dy = dfdy(W_1_2,i,j,k,0.,dx)
#     dW_1_3_dy = dfdy(W_1_3,i,j,k,0.,dx)
#     dW_1_4_dy = dfdy(W_1_4,i,j,k,0.,dx)

#     dW_1_1_dz = dfdz(W_1_1,i,j,k,0.,dx)
#     dW_1_2_dz = dfdz(W_1_2,i,j,k,0.,dx)
#     dW_1_3_dz = dfdz(W_1_3,i,j,k,0.,dx)
#     dW_1_4_dz = dfdz(W_1_4,i,j,k,0.,dx)

#     dW_2_1_dx = dfdx(W_2_1,i,j,k,0.,dx)
#     dW_2_2_dx = dfdx(W_2_2,i,j,k,0.,dx)
#     dW_2_3_dx = dfdx(W_2_3,i,j,k,0.,dx)
#     dW_2_4_dx = dfdx(W_2_4,i,j,k,0.,dx)

#     dW_2_1_dy = dfdy(W_2_1,i,j,k,0.,dx)
#     dW_2_2_dy = dfdy(W_2_2,i,j,k,0.,dx)
#     dW_2_3_dy = dfdy(W_2_3,i,j,k,0.,dx)
#     dW_2_4_dy = dfdy(W_2_4,i,j,k,0.,dx)

#     dW_2_1_dz = dfdz(W_2_1,i,j,k,0.,dx)
#     dW_2_2_dz = dfdz(W_2_2,i,j,k,0.,dx)
#     dW_2_3_dz = dfdz(W_2_3,i,j,k,0.,dx)
#     dW_2_4_dz = dfdz(W_2_4,i,j,k,0.,dx)

#     dW_3_1_dx = dfdx(W_3_1,i,j,k,0.,dx)
#     dW_3_2_dx = dfdx(W_3_2,i,j,k,0.,dx)
#     dW_3_3_dx = dfdx(W_3_3,i,j,k,0.,dx)
#     dW_3_4_dx = dfdx(W_3_4,i,j,k,0.,dx)

#     dW_3_1_dy = dfdy(W_3_1,i,j,k,0.,dx)
#     dW_3_2_dy = dfdy(W_3_2,i,j,k,0.,dx)
#     dW_3_3_dy = dfdy(W_3_3,i,j,k,0.,dx)
#     dW_3_4_dy = dfdy(W_3_4,i,j,k,0.,dx)

#     dW_3_1_dz = dfdz(W_3_1,i,j,k,0.,dx)
#     dW_3_2_dz = dfdz(W_3_2,i,j,k,0.,dx)
#     dW_3_3_dz = dfdz(W_3_3,i,j,k,0.,dx)
#     dW_3_4_dz = dfdz(W_3_4,i,j,k,0.,dx)

#     dY_1_dx = dfdx(Y_1,i,j,k,0.,dx)
#     dY_2_dx = dfdx(Y_2,i,j,k,0.,dx)
#     dY_3_dx = dfdx(Y_3,i,j,k,0.,dx)
#     dY_4_dx = dfdx(Y_4,i,j,k,0.,dx)

#     dY_1_dy = dfdy(Y_1,i,j,k,0.,dx)
#     dY_2_dy = dfdy(Y_2,i,j,k,0.,dx)
#     dY_3_dy = dfdy(Y_3,i,j,k,0.,dx)
#     dY_4_dy = dfdy(Y_4,i,j,k,0.,dx)

#     dY_1_dz = dfdz(Y_1,i,j,k,0.,dx)
#     dY_2_dz = dfdz(Y_2,i,j,k,0.,dx)
#     dY_3_dz = dfdz(Y_3,i,j,k,0.,dx)
#     dY_4_dz = dfdz(Y_4,i,j,k,0.,dx)

#     dΓ_1_dx = dfdx(Γ_1,i,j,k,0.,dx)
#     dΓ_1_dy = dfdy(Γ_1,i,j,k,0.,dx)
#     dΓ_1_dz = dfdz(Γ_1,i,j,k,0.,dx)

#     dΓ_2_dx = dfdx(Γ_2,i,j,k,0.,dx)
#     dΓ_2_dy = dfdy(Γ_2,i,j,k,0.,dx)
#     dΓ_2_dz = dfdz(Γ_2,i,j,k,0.,dx)

#     dΓ_3_dx = dfdx(Γ_3,i,j,k,0.,dx)
#     dΓ_3_dy = dfdy(Γ_3,i,j,k,0.,dx)
#     dΓ_3_dz = dfdz(Γ_3,i,j,k,0.,dx)
    
#     dΣ_dx = dfdx(Σ,i,j,k,0.,dx)
#     dΣ_dy = dfdy(Σ,i,j,k,0.,dx)
#     dΣ_dz = dfdz(Σ,i,j,k,0.,dx)

#     d2ϕ_1_dx2=d2fdx2(ϕ_1,i,j,k,0.,dx)
#     d2ϕ_2_dx2=d2fdx2(ϕ_2,i,j,k,0.,dx)
#     d2ϕ_3_dx2=d2fdx2(ϕ_2,i,j,k,0.,dx)
#     d2ϕ_4_dx2=d2fdx2(ϕ_2,i,j,k,0.,dx)

#     d2ϕ_1_dy2=d2fdy2(ϕ_1,i,j,k,0.,dx)
#     d2ϕ_2_dy2=d2fdy2(ϕ_2,i,j,k,0.,dx)
#     d2ϕ_3_dy2=d2fdy2(ϕ_2,i,j,k,0.,dx)
#     d2ϕ_4_dy2=d2fdy2(ϕ_2,i,j,k,0.,dx)

#     d2ϕ_1_dz2=d2fdz2(ϕ_1,i,j,k,0.,dx)
#     d2ϕ_2_dz2=d2fdz2(ϕ_2,i,j,k,0.,dx)
#     d2ϕ_3_dz2=d2fdz2(ϕ_2,i,j,k,0.,dx)
#     d2ϕ_4_dz2=d2fdz2(ϕ_2,i,j,k,0.,dx)

#     d2W_1_1_dx2 = d2fdx2(W_1_1,i,j,k,0.,dx)
#     d2W_1_2_dx2 = d2fdx2(W_1_2,i,j,k,0.,dx)
#     d2W_1_3_dx2 = d2fdx2(W_1_3,i,j,k,0.,dx)
#     d2W_1_4_dx2 = d2fdx2(W_1_4,i,j,k,0.,dx)

#     d2W_1_1_dy2 = d2fdy2(W_1_1,i,j,k,0.,dx)
#     d2W_1_2_dy2 = d2fdy2(W_1_2,i,j,k,0.,dx)
#     d2W_1_3_dy2 = d2fdy2(W_1_3,i,j,k,0.,dx)
#     d2W_1_4_dy2 = d2fdy2(W_1_4,i,j,k,0.,dx)

#     d2W_1_1_dz2 = d2fdz2(W_1_1,i,j,k,0.,dx)
#     d2W_1_2_dz2 = d2fdz2(W_1_2,i,j,k,0.,dx)
#     d2W_1_3_dz2 = d2fdz2(W_1_3,i,j,k,0.,dx)
#     d2W_1_4_dz2 = d2fdz2(W_1_4,i,j,k,0.,dx)

#     d2W_2_1_dx2 = d2fdx2(W_2_1,i,j,k,0.,dx)
#     d2W_2_2_dx2 = d2fdx2(W_2_2,i,j,k,0.,dx)
#     d2W_2_3_dx2 = d2fdx2(W_2_3,i,j,k,0.,dx)
#     d2W_2_4_dx2 = d2fdx2(W_2_4,i,j,k,0.,dx)

#     d2W_2_1_dy2 = d2fdy2(W_2_1,i,j,k,0.,dx)
#     d2W_2_2_dy2 = d2fdy2(W_2_2,i,j,k,0.,dx)
#     d2W_2_3_dy2 = d2fdy2(W_2_3,i,j,k,0.,dx)
#     d2W_2_4_dy2 = d2fdy2(W_2_4,i,j,k,0.,dx)

#     d2W_2_1_dz2 = d2fdz2(W_2_1,i,j,k,0.,dx)
#     d2W_2_2_dz2 = d2fdz2(W_2_2,i,j,k,0.,dx)
#     d2W_2_3_dz2 = d2fdz2(W_2_3,i,j,k,0.,dx)
#     d2W_2_4_dz2 = d2fdz2(W_2_4,i,j,k,0.,dx)

#     d2W_3_1_dx2 = d2fdx2(W_3_1,i,j,k,0.,dx)
#     d2W_3_2_dx2 = d2fdx2(W_3_2,i,j,k,0.,dx)
#     d2W_3_3_dx2 = d2fdx2(W_3_3,i,j,k,0.,dx)
#     d2W_3_4_dx2 = d2fdx2(W_3_4,i,j,k,0.,dx)

#     d2W_3_1_dy2 = d2fdy2(W_3_1,i,j,k,0.,dx)
#     d2W_3_2_dy2 = d2fdy2(W_3_2,i,j,k,0.,dx)
#     d2W_3_3_dy2 = d2fdy2(W_3_3,i,j,k,0.,dx)
#     d2W_3_4_dy2 = d2fdy2(W_3_4,i,j,k,0.,dx)

#     d2W_3_1_dz2 = d2fdz2(W_3_1,i,j,k,0.,dx)
#     d2W_3_2_dz2 = d2fdz2(W_3_2,i,j,k,0.,dx)
#     d2W_3_3_dz2 = d2fdz2(W_3_3,i,j,k,0.,dx)
#     d2W_3_4_dz2 = d2fdz2(W_3_4,i,j,k,0.,dx)

#     d2Y_1_dx2 = d2fdx2(Y_1,i,j,k,0.,dx)
#     d2Y_2_dx2 = d2fdx2(Y_2,i,j,k,0.,dx)
#     d2Y_3_dx2 = d2fdx2(Y_3,i,j,k,0.,dx)
#     d2Y_4_dx2 = d2fdx2(Y_4,i,j,k,0.,dx)

#     d2Y_1_dy2 = d2fdy2(Y_1,i,j,k,0.,dx)
#     d2Y_2_dy2 = d2fdy2(Y_2,i,j,k,0.,dx)
#     d2Y_3_dy2 = d2fdy2(Y_3,i,j,k,0.,dx)
#     d2Y_4_dy2 = d2fdy2(Y_4,i,j,k,0.,dx)

#     d2Y_1_dz2 = d2fdz2(Y_1,i,j,k,0.,dx)
#     d2Y_2_dz2 = d2fdz2(Y_2,i,j,k,0.,dx)
#     d2Y_3_dz2 = d2fdz2(Y_3,i,j,k,0.,dx)
#     d2Y_4_dz2 = d2fdz2(Y_4,i,j,k,0.,dx)

#     ##Covariant Derivatives##

#     Dt_ϕ_1 =D_1ϕ_1(dϕ_1_dt[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
#     W_1_1[i,j,k],W_2_1[i,j,k],W_3_1[i,j,k],Y_1[i,j,k],gw,gy)
#     Dt_ϕ_2 =D_1ϕ_2(dϕ_2_dt[i,j,k],ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
#     W_1_1[i,j,k],W_2_1[i,j,k],W_3_1[i,j,k],Y_1[i,j,k],gw,gy)
#     Dt_ϕ_3 =D_1ϕ_3(dϕ_3_dt[i,j,k],ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
#     W_1_1[i,j,k],W_2_1[i,j,k],W_3_1[i,j,k],Y_1[i,j,k],gw,gy)
#     Dt_ϕ_4 =D_1ϕ_4(dϕ_4_dt[i,j,k],ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
#     W_1_1[i,j,k],W_2_1[i,j,k],W_3_1[i,j,k],Y_1[i,j,k],gw,gy)
#     Dx_ϕ_1 =D_2ϕ_1(dϕ_1_dx,ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
#     W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy)
#     Dx_ϕ_2 =D_2ϕ_2(dϕ_2_dx,ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
#     W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy)
#     Dx_ϕ_3 =D_2ϕ_3(dϕ_3_dx,ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
#     W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy)
#     Dx_ϕ_4 =D_2ϕ_4(dϕ_4_dx,ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
#     W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy)
#     Dy_ϕ_1 =D_3ϕ_1(dϕ_1_dy,ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
#     W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy)
#     Dy_ϕ_2 =D_3ϕ_2(dϕ_2_dy,ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
#     W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy)
#     Dy_ϕ_3 =D_3ϕ_3(dϕ_3_dy,ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
#     W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy)
#     Dy_ϕ_4 =D_3ϕ_4(dϕ_4_dy,ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
#     W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy)
#     Dz_ϕ_1 =D_4ϕ_1(dϕ_1_dz,ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
#     W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)
#     Dz_ϕ_2 =D_4ϕ_2(dϕ_2_dz,ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
#     W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)
#     Dz_ϕ_3 =D_4ϕ_3(dϕ_3_dz,ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
#     W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)
#     Dz_ϕ_4 =D_4ϕ_4(dϕ_4_dz,ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
#     W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)

#     # Field Strengths #
#     # W_1_11()
#     # W_1_12(dW_1_2_dt)
#     # W_1_13(dW_1_3_dt)
#     # W_1_14(dW_1_4_dt)
#     # W_1_22(dW_2_2_dt)
#     W_1_23 =W_1_xy(dW_1_2_dx,dW_1_1_dy,W_2_2,W_3_3,W_2_3,W_3_2,gw,i,j,k)
#     W_1_24=W_1_xz(dW_1_4_dx,dW_1_2_dz,W_2_2,W_3_4,W_2_4,W_3_2,gw,i,j,k)
#     # W_1_33(dW_3_3_dt)
#     W_1_34=W_1_yz(dW_1_4_dy,dW_1_3_dz,W_2_3,W_3_4,W_2_4,W_3_3,gw,i,j,k)
#     # W_1_44()
#     # W_2_11()
#     # W_2_12(dW_2_2_dt)
#     # W_2_13(dW_2_3_dt)
#     # W_2_14(dW_2_4_dt)
#     # W_2_22()
#     W_2_23=W_2_xy(dW_2_3_dx,dW_2_2_dy,W_3_2,W_1_3,W_3_3,W_1_2,gw,i,j,k)
#     W_2_24=W_2_xz(dW_2_4_dx,dW_2_2_dz,W_3_2,W_1_4,W_3_4,W_1_2,gw,i,j,k)
#     # W_2_33()
#     W_2_34=W_2_yz(dW_2_4_dy,dW_2_3_dz,W_3_3,W_1_4,W_3_4,W_1_3,gw,i,j,k)
#     # W_2_44()
#     # W_3_11()
#     # W_3_12(dW_3_2_dt)
#     # W_3_13(dW_3_3_dt)
#     # W_3_14(dW_3_4_dt)
#     # W_3_22()
#     W_3_23=W_3_xy(dW_3_3_dx,dW_3_2_dy,W_1_2,W_2_3,W_1_3,W_2_2,gw,i,j,k)
#     W_3_24=W_3_xz(dW_3_4_dx,dW_3_2_dz,W_1_2,W_2_4,W_1_4,W_2_2,gw,i,j,k)
#     # W_3_33(dW_3_3_dt)
#     W_3_34=W_3_yz(dW_3_4_dy,dW_3_3_dz,W_1_3,W_2_4,W_1_4,W_2_3,gw,i,j,k)
#     # W_3_44()
#     # Y_1_1()
#     # Y_1_2(dY_2_dt)
#     # Y_1_3(dY_3_dt)
#     # Y_1_4(dY_4_dt)
#     # Y_2_2()
#     Y_2_3=Y_2_y(dY_3_dx,dY_2_dy)
#     Y_2_4=Y_2_z(dY_4_dx,dY_2_dz)
#     # Y_3_3()
#     Y_3_4=Y_3_z(dY_4_dy,dY_3_dz)
#     # Y_4_4()

#     # kt_1 expressions
#     @inbounds kt_ϕ_1[i,j,k] = ((d2ϕ_1_dx2+d2ϕ_1_dy2+d2ϕ_1_dz2-
#     0.5*gw*(((-W_1_2[i,j,k]*dϕ_4_dx)-(W_1_3[i,j,k]*dϕ_4_dy)-(W_1_4[i,j,k]*dϕ_4_dz))-
#     ((-W_2_2[i,j,k]*dϕ_3_dx)-(W_2_3[i,j,k]*dϕ_3_dy)-(W_2_4[i,j,k]*dϕ_3_dz))+
#     ((-W_3_2[i,j,k]*dϕ_2_dx)-(W_3_3[i,j,k]*dϕ_2_dy)-(W_3_4[i,j,k]*dϕ_2_dz)))-
#     0.5*gy*(-Y_2[i,j,k]*dϕ_2_dx-Y_3[i,j,k]*dϕ_2_dy-Y_4[i,j,k]*dϕ_2_dz)-
#     0.5*gw*((-W_1_2[i,j,k]*Dx_ϕ_4-W_1_3[i,j,k]*Dy_ϕ_4-W_1_4[i,j,k]*Dz_ϕ_4)-
#     (-W_2_2[i,j,k]*Dx_ϕ_3-W_2_3[i,j,k]*Dy_ϕ_3-W_2_4[i,j,k]*Dz_ϕ_3)+
#     (-W_3_2[i,j,k]*Dx_ϕ_2-W_3_3[i,j,k]*Dy_ϕ_2-W_3_4[i,j,k]*Dz_ϕ_2))-
#     0.5*gy*(-Y_2[i,j,k]*Dx_ϕ_2-Y_3[i,j,k]*Dy_ϕ_2-Y_4[i,j,k]*Dz_ϕ_2)-
#     2.0*lambda*(ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2-vev^2)*ϕ_1[i,j,k]+
#     0.5*((gw*Γ_3[i,j,k]+gy*Σ[i,j,k])*ϕ_2[i,j,k]-gw*Γ_2[i,j,k]*ϕ_3[i,j,k]+gw*Γ_1[i,j,k]*ϕ_4[i,j,k])))

#     @inbounds kt_ϕ_2[i,j,k] = ((d2ϕ_2_dx2+d2ϕ_2_dy2+d2ϕ_2_dz2+
#     0.5*gw*((-W_1_2[i,j,k]*dϕ_3_dx-W_1_3[i,j,k]*dϕ_3_dy-W_1_4[i,j,k]*dϕ_3_dz)+
#     (-W_2_2[i,j,k]*dϕ_4_dx-W_2_3[i,j,k]*dϕ_4_dy-W_2_4[i,j,k]*dϕ_4_dz)+
#     (-W_3_2[i,j,k]*dϕ_1_dx-W_3_3[i,j,k]*dϕ_1_dy-W_3_4[i,j,k]*dϕ_1_dz))+
#     0.5*gy*(-Y_2[i,j,k]*dϕ_1_dx-Y_3[i,j,k]*dϕ_1_dy-Y_4[i,j,k]*dϕ_1_dz)+
#     0.5*gw*((-W_1_2[i,j,k]*Dx_ϕ_3-W_1_3[i,j,k]*Dy_ϕ_3-W_1_4[i,j,k]*Dz_ϕ_3)+
#     (-W_2_2[i,j,k]*Dx_ϕ_4-W_2_3[i,j,k]*Dy_ϕ_4-W_2_4[i,j,k]*Dz_ϕ_4)+
#     (-W_3_2[i,j,k]*Dx_ϕ_1-W_3_3[i,j,k]*Dy_ϕ_1-W_3_4[i,j,k]*Dz_ϕ_1))+
#     0.5*gy*(-Y_2[i,j,k]*Dx_ϕ_1-Y_3[i,j,k]*Dy_ϕ_1-Y_4[i,j,k]*Dz_ϕ_1)-
#     2.0*lambda*(ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2-vev^2)*ϕ_2[i,j,k]-
#     0.5*((gw*Γ_3[i,j,k]+gy*Σ[i,j,k])*ϕ_1[i,j,k]+gw*Γ_1[i,j,k]*ϕ_3[i,j,k]+gw*Γ_2[i,j,k]*ϕ_4[i,j,k])))

#     @inbounds kt_ϕ_3[i,j,k] = ((d2ϕ_3_dx2+d2ϕ_3_dy2+d2ϕ_3_dz2-
#     0.5*gw*((-W_1_2[i,j,k]*dϕ_2_dx-W_1_3[i,j,k]*dϕ_2_dy-W_1_4[i,j,k]*dϕ_2_dz)+
#     (-W_2_2[i,j,k]*dϕ_1_dx-W_2_3[i,j,k]*dϕ_1_dy-W_2_4[i,j,k]*dϕ_1_dz)-
#     (-W_3_2[i,j,k]*dϕ_4_dx-W_3_3[i,j,k]*dϕ_4_dy-W_3_4[i,j,k]*dϕ_4_dz))-
#     0.5*gy*(-Y_2[i,j,k]*dϕ_4_dx-Y_3[i,j,k]*dϕ_4_dy-Y_4[i,j,k]*dϕ_4_dz)-
#     0.5*gw*((-W_1_2[i,j,k]*Dx_ϕ_2-W_1_3[i,j,k]*Dy_ϕ_2-W_1_4[i,j,k]*Dz_ϕ_2)+
#     (-W_2_2[i,j,k]*Dx_ϕ_1-W_2_3[i,j,k]*Dy_ϕ_1-W_2_4[i,j,k]*Dz_ϕ_1)-
#     (-W_3_2[i,j,k]*Dx_ϕ_4-W_3_3[i,j,k]*Dy_ϕ_4-W_3_4[i,j,k]*Dz_ϕ_4))-
#     0.5*gy*(-Y_2[i,j,k]*Dx_ϕ_4-Y_3[i,j,k]*Dy_ϕ_4-Y_4[i,j,k]*Dz_ϕ_4)-
#     2.0*lambda*(ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2-vev^2)*ϕ_3[i,j,k]+
#     0.5*((-gw*Γ_3[i,j,k]+gy*Σ[i,j,k])*ϕ_4[i,j,k]+gw*Γ_2[i,j,k]*ϕ_1[i,j,k]+gw*Γ_1[i,j,k]*ϕ_2[i,j,k])))

#     @inbounds kt_ϕ_4[i,j,k] = ((d2ϕ_4_dx2+d2ϕ_4_dy2+d2ϕ_4_dz2+
#     0.5*gw*((-W_1_2[i,j,k]*dϕ_1_dx-W_1_3[i,j,k]*dϕ_1_dy-W_1_4[i,j,k]*dϕ_1_dz)-
#     (-W_2_2[i,j,k]*dϕ_2_dx-W_2_3[i,j,k]*dϕ_2_dy-W_2_4[i,j,k]*dϕ_2_dz)-
#     (-W_3_2[i,j,k]*dϕ_3_dx-W_3_3[i,j,k]*dϕ_3_dy-W_3_4[i,j,k]*dϕ_3_dz))+
#     0.5*gy*(-Y_2[i,j,k]*dϕ_3_dx-Y_3[i,j,k]*dϕ_3_dy-Y_4[i,j,k]*dϕ_3_dz)+
#     0.5*gw*((-W_1_2[i,j,k]*Dx_ϕ_1-W_1_3[i,j,k]*Dy_ϕ_1-W_1_4[i,j,k]*Dz_ϕ_1)-
#     (-W_2_2[i,j,k]*Dx_ϕ_2-W_2_3[i,j,k]*Dy_ϕ_2-W_2_4[i,j,k]*Dz_ϕ_2)-
#     (-W_3_2[i,j,k]*Dx_ϕ_3-W_3_3[i,j,k]*Dy_ϕ_3-W_3_4[i,j,k]*Dz_ϕ_3))+
#     0.5*gy*(-Y_2[i,j,k]*Dx_ϕ_3-Y_3[i,j,k]*Dy_ϕ_3-Y_4[i,j,k]*Dz_ϕ_3)-
#     2.0*lambda*(ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2-vev^2)*ϕ_4[i,j,k]-
#     0.5*((-gw*Γ_3[i,j,k]+gy*Σ[i,j,k])*ϕ_3[i,j,k]+gw*Γ_1[i,j,k]*ϕ_1[i,j,k]-gw*Γ_2[i,j,k]*ϕ_2[i,j,k])))

#     @inbounds kt_W_1_2[i,j,k] = ((d2W_1_2_dx2+d2W_1_2_dy2+d2W_1_2_dz2+
#     gw*(-(dW_2_2_dx*W_3_2[i,j,k]-dW_3_2_dx*W_2_2[i,j,k])-
#     (dW_2_2_dy*W_3_3[i,j,k]-dW_3_2_dy*W_2_3[i,j,k])-
#     (dW_2_2_dz*W_3_4[i,j,k]-dW_3_2_dz*W_2_4[i,j,k])-
#     (W_2_3[i,j,k]*W_3_23-W_3_3[i,j,k]*W_2_23)-
#     (W_2_4[i,j,k]*W_3_24-W_3_4[i,j,k]*W_2_24))+
#     gw*(ϕ_1[i,j,k]*Dx_ϕ_4-ϕ_2[i,j,k]*Dx_ϕ_3+ϕ_3[i,j,k]*Dx_ϕ_2-ϕ_4[i,j,k]*Dx_ϕ_1)-
#     dΓ_1_dx-gw*(W_2_2[i,j,k]*Γ_3[i,j,k]-W_3_2[i,j,k]*Γ_2[i,j,k])-
#     gw*(W_2_1[i,j,k]*dW_3_2_dt[i,j,k]-W_3_1[i,j,k]*dW_2_2_dt[i,j,k])))

#     @inbounds kt_W_1_3[i,j,k] = ((d2W_1_3_dx2+d2W_1_3_dy2+d2W_1_3_dz2+
#     gw*(-(dW_2_3_dx*W_3_2[i,j,k]-dW_3_3_dx*W_2_2[i,j,k])-
#     (dW_2_3_dy*W_3_3[i,j,k]-dW_3_3_dy*W_2_3[i,j,k])-
#     (dW_2_3_dz*W_3_4[i,j,k]-dW_3_3_dz*W_2_4[i,j,k])-
#     (W_2_2[i,j,k]*(-W_3_23)-W_3_2[i,j,k]*(-W_2_23))-
#     (W_2_4[i,j,k]*W_3_34-W_3_4[i,j,k]*W_2_34))+
#     gw*(ϕ_1[i,j,k]*Dy_ϕ_4-ϕ_2[i,j,k]*Dy_ϕ_3+ϕ_3[i,j,k]*Dy_ϕ_2-ϕ_4[i,j,k]*Dy_ϕ_1)-
#     dΓ_1_dy-gw*(W_2_3[i,j,k]*Γ_3[i,j,k]-W_3_3[i,j,k]*Γ_2[i,j,k])-
#     gw*(W_2_1[i,j,k]*dW_3_3_dt[i,j,k]-W_3_1[i,j,k]*dW_2_3_dt[i,j,k])))

#     @inbounds kt_W_1_4[i,j,k] = ((d2W_1_4_dx2+d2W_1_4_dy2+d2W_1_4_dz2+
#     gw*(-(dW_2_4_dx*W_3_2[i,j,k]-dW_3_4_dx*W_2_2[i,j,k])-
#     (dW_2_4_dy*W_3_3[i,j,k]-dW_3_4_dy*W_2_3[i,j,k])-
#     (dW_2_4_dz*W_3_4[i,j,k]-dW_3_4_dz*W_2_4[i,j,k])-
#     (W_2_2[i,j,k]*(-W_3_24)-W_3_2[i,j,k]*(-W_2_24))-
#     (W_2_3[i,j,k]*(-W_3_34)-W_3_3[i,j,k]*(-W_2_34)))+
#     gw*(ϕ_1[i,j,k]*Dz_ϕ_4-ϕ_2[i,j,k]*Dz_ϕ_3+ϕ_3[i,j,k]*Dz_ϕ_2-ϕ_4[i,j,k]*Dz_ϕ_1)-
#     dΓ_1_dz-gw*(W_2_4[i,j,k]*Γ_3[i,j,k]-W_3_4[i,j,k]*Γ_2[i,j,k])-
#     gw*(W_2_1[i,j,k]*dW_3_4_dt[i,j,k]-W_3_1[i,j,k]*dW_2_4_dt[i,j,k])))

#     @inbounds kt_W_2_2[i,j,k] = ((d2W_2_2_dx2+d2W_2_2_dy2+d2W_2_2_dz2+
#     gw*(-(dW_3_2_dx*W_1_2[i,j,k]-dW_1_2_dx*W_3_2[i,j,k])-
#     (dW_3_2_dy*W_1_3[i,j,k]-dW_1_2_dy*W_3_3[i,j,k])-
#     (dW_3_2_dz*W_1_4[i,j,k]-dW_1_2_dz*W_3_4[i,j,k])-
#     (W_3_3[i,j,k]*W_1_23-W_1_3[i,j,k]*W_3_23)-
#     (W_3_4[i,j,k]*W_1_24-W_1_4[i,j,k]*W_3_24))+
#     gw*(-ϕ_1[i,j,k]*Dx_ϕ_3-ϕ_2[i,j,k]*Dx_ϕ_4+ϕ_3[i,j,k]*Dx_ϕ_1+ϕ_4[i,j,k]*Dx_ϕ_2)-
#     dΓ_2_dx-gw*(W_3_2[i,j,k]*Γ_1[i,j,k]-W_1_2[i,j,k]*Γ_3[i,j,k])-
#     gw*(W_3_1[i,j,k]*dW_1_2_dt[i,j,k]-W_1_1[i,j,k]*dW_3_2_dt[i,j,k])))

#     @inbounds kt_W_2_3[i,j,k] = ((d2W_2_3_dx2+d2W_2_3_dy2+d2W_2_3_dz2+
#     gw*(-(dW_3_3_dx*W_1_2[i,j,k]-dW_1_3_dx*W_3_2[i,j,k])-
#     (dW_3_3_dy*W_1_3[i,j,k]-dW_1_3_dy*W_3_3[i,j,k])-
#     (dW_3_3_dz*W_1_4[i,j,k]-dW_1_3_dz*W_3_4[i,j,k])-
#     (W_3_2[i,j,k]*(-W_1_23)-W_1_2[i,j,k]*(-W_3_23))-
#     (W_3_4[i,j,k]*(W_1_34)-W_1_4[i,j,k]*W_3_34))+
#     gw*(-ϕ_1[i,j,k]*Dy_ϕ_3-ϕ_2[i,j,k]*Dy_ϕ_4+ϕ_3[i,j,k]*Dy_ϕ_1+ϕ_4[i,j,k]*Dy_ϕ_2)-
#     dΓ_2_dy-gw*(W_3_3[i,j,k]*Γ_1[i,j,k]-W_1_3[i,j,k]*Γ_3[i,j,k])-
#     gw*(W_3_1[i,j,k]*dW_1_3_dt[i,j,k]-W_1_1[i,j,k]*dW_3_3_dt[i,j,k])))

#     @inbounds kt_W_2_4[i,j,k] = ((d2W_2_4_dx2+d2W_2_4_dy2+d2W_2_4_dz2+
#     gw*(-(dW_3_4_dx*W_1_2[i,j,k]-dW_1_4_dx*W_3_2[i,j,k])-
#     (dW_3_4_dy*W_1_3[i,j,k]-dW_1_4_dy*W_3_3[i,j,k])-
#     (dW_3_4_dz*W_1_4[i,j,k]-dW_1_4_dz*W_3_4[i,j,k])-
#     (W_3_2[i,j,k]*(-W_1_24)-W_1_2[i,j,k]*(-W_3_24))-
#     (W_3_3[i,j,k]*(-W_1_34)-W_1_3[i,j,k]*(-W_3_34)))+
#     gw*(-ϕ_1[i,j,k]*Dz_ϕ_3-ϕ_2[i,j,k]*Dz_ϕ_4+ϕ_3[i,j,k]*Dz_ϕ_1+ϕ_4[i,j,k]*Dz_ϕ_2)-
#     dΓ_2_dz-gw*(W_3_4[i,j,k]*Γ_1[i,j,k]-W_1_4[i,j,k]*Γ_3[i,j,k])-
#     gw*(W_3_1[i,j,k]*dW_1_4_dt[i,j,k]-W_1_1[i,j,k]*dW_3_4_dt[i,j,k])))

#     @inbounds kt_W_3_2[i,j,k] = ((d2W_3_2_dx2+d2W_3_2_dy2+d2W_3_2_dz2+
#     gw*(-(dW_1_2_dx*W_2_2[i,j,k]-dW_2_2_dx*W_1_2[i,j,k])-
#     (dW_1_2_dy*W_2_3[i,j,k]-dW_2_2_dy*W_1_3[i,j,k])-
#     (dW_1_2_dz*W_2_4[i,j,k]-dW_2_2_dz*W_1_4[i,j,k])-
#     (W_1_3[i,j,k]*W_2_23-W_2_3[i,j,k]*W_1_23)-
#     (W_1_4[i,j,k]*W_2_24-W_2_4[i,j,k]*W_1_24))+
#     gw*(ϕ_1[i,j,k]*Dx_ϕ_2-ϕ_2[i,j,k]*Dx_ϕ_1-ϕ_3[i,j,k]*Dx_ϕ_4+ϕ_4[i,j,k]*Dx_ϕ_3)-
#     dΓ_3_dx-gw*(W_1_2[i,j,k]*Γ_2[i,j,k]-W_2_2[i,j,k]*Γ_1[i,j,k])-
#     gw*(W_1_1[i,j,k]*dW_2_2_dt[i,j,k]-W_2_1[i,j,k]*dW_1_2_dt[i,j,k])))

#     @inbounds kt_W_3_3[i,j,k] = ((d2W_3_3_dx2+d2W_3_3_dy2+d2W_3_3_dz2+
#     gw*(-(dW_1_3_dx*W_2_2[i,j,k]-dW_2_3_dx*W_1_2[i,j,k])-
#         (dW_1_3_dy*W_2_3[i,j,k]-dW_2_3_dy*W_1_3[i,j,k])-
#         (dW_1_3_dz*W_2_4[i,j,k]-dW_2_3_dz*W_1_4[i,j,k])-
#         (W_1_2[i,j,k]*(-W_2_23)-W_2_2[i,j,k]*(-W_1_23))-
#         (W_1_4[i,j,k]*W_2_34-W_2_4[i,j,k]*(W_1_34)))+
#     gw*(ϕ_1[i,j,k]*Dy_ϕ_2-ϕ_2[i,j,k]*Dy_ϕ_1-ϕ_3[i,j,k]*Dy_ϕ_4+ϕ_4[i,j,k]*Dy_ϕ_3)-
#     dΓ_3_dy-gw*(W_1_3[i,j,k]*Γ_2[i,j,k]-W_2_3[i,j,k]*Γ_1[i,j,k])-
#     gw*(W_1_1[i,j,k]*dW_2_3_dt[i,j,k]-W_2_1[i,j,k]*dW_1_3_dt[i,j,k])))

#     @inbounds kt_W_3_4[i,j,k] = ((d2W_3_4_dx2+d2W_3_4_dy2+d2W_3_4_dz2+
#     gw*(-(dW_1_4_dx*W_2_2[i,j,k]-dW_2_4_dx*W_1_2[i,j,k])-
#         (dW_1_4_dy*W_2_3[i,j,k]-dW_2_4_dy*W_1_3[i,j,k])-
#         (dW_1_4_dz*W_2_4[i,j,k]-dW_2_4_dz*W_1_4[i,j,k])-
#         (W_1_2[i,j,k]*(-W_2_24)-W_2_2[i,j,k]*(-W_1_24))-
#         (W_1_3[i,j,k]*(-W_2_34)-W_2_3[i,j,k]*(-W_1_34)))+
#     gw*(ϕ_1[i,j,k]*Dz_ϕ_2-ϕ_2[i,j,k]*Dz_ϕ_1-ϕ_3[i,j,k]*Dz_ϕ_4+ϕ_4[i,j,k]*Dz_ϕ_3)-
#     dΓ_3_dz-gw*(W_1_4[i,j,k]*Γ_2[i,j,k]-W_2_4[i,j,k]*Γ_1[i,j,k])-
#     gw*(W_1_1[i,j,k]*dW_2_4_dt[i,j,k]-W_2_1[i,j,k]*dW_1_4_dt[i,j,k])))

#     @inbounds kt_Y_2[i,j,k] = ((d2Y_2_dx2+d2Y_2_dy2+d2Y_2_dz2+
#     gy*(ϕ_1[i,j,k]*Dx_ϕ_2-ϕ_2[i,j,k]*Dx_ϕ_1+ϕ_3[i,j,k]*Dx_ϕ_4-ϕ_4[i,j,k]*Dx_ϕ_3)-dΣ_dx))

#     @inbounds kt_Y_3[i,j,k] = ((d2Y_3_dx2+d2Y_3_dy2+d2Y_3_dz2+
#     gy*(ϕ_1[i,j,k]*Dy_ϕ_2-ϕ_2[i,j,k]*Dy_ϕ_1+ϕ_3[i,j,k]*Dy_ϕ_4-ϕ_4[i,j,k]*Dy_ϕ_3)-dΣ_dy))

#     @inbounds kt_Y_4[i,j,k] = ((d2Y_4_dx2+d2Y_4_dy2+d2Y_4_dz2+
#     gy*(ϕ_1[i,j,k]*Dz_ϕ_2-ϕ_2[i,j,k]*Dz_ϕ_1+ϕ_3[i,j,k]*Dz_ϕ_4-ϕ_4[i,j,k]*Dz_ϕ_3)-dΣ_dz))

#     # k_1 expressions
#     @inbounds k_ϕ_1[i,j,k] =(dϕ_1_dt[i,j,k])
#     @inbounds k_ϕ_2[i,j,k] =(dϕ_2_dt[i,j,k])
#     @inbounds k_ϕ_3[i,j,k] =(dϕ_3_dt[i,j,k])
#     @inbounds k_ϕ_4[i,j,k] =(dϕ_4_dt[i,j,k])
#     # c
#     # c Eq. (2.11) of Baumgarte&Shapiro is $\partial_t A_i = -E_i -...$ so 
#     # c we are taking fd(...)=+\partial_t A_i = -E_i (note the sign).
#     # s[5]=0.
#     # s[6]=
#     @inbounds k_W_1_2[i,j,k] =((dW_1_2_dt[i,j,k]).+
#     # c in the gauge $W^a_0=0=Y_0$, f(5...)=0=f(9...)=f(13...) and the line
#     # c below vanishes.
#         dW_1_1_dx .-gw.*((W_2_1[i,j,k]).*(W_3_2[i,j,k]).-(W_3_1[i,j,k]).*(W_2_2[i,j,k])))
#     # s[7]=
#     @inbounds k_W_1_3[i,j,k] =((dW_1_3_dt[i,j,k]).+
#         dW_1_1_dy.-gw.*((W_2_1[i,j,k]).*(W_3_3[i,j,k]).-(W_3_1[i,j,k]).*(W_2_3[i,j,k])))
#     # s[8]=
#     @inbounds k_W_1_4[i,j,k] =((dW_1_4_dt[i,j,k]).+
#         dW_1_1_dz.-gw.*((W_2_1[i,j,k]).*(W_3_4[i,j,k]).-(W_3_1[i,j,k]).*(W_2_4[i,j,k])))

#     # s[9]=0.
#     # s[10]=
#     @inbounds k_W_2_2[i,j,k] =((dW_2_2_dt[i,j,k]).+
#         dW_2_1_dx.-gw.*((W_3_1[i,j,k]).*(W_1_2[i,j,k]).-(W_1_1[i,j,k]).*(W_3_2[i,j,k])))
#     # s[11]=
#     @inbounds k_W_2_3[i,j,k] =((dW_2_3_dt[i,j,k]).+
#         dW_2_1_dy.-gw.*((W_3_1[i,j,k]).*(W_1_3[i,j,k]).-(W_1_1[i,j,k]).*(W_3_3[i,j,k])))
#     # s[12]=
#     @inbounds k_W_2_4[i,j,k] =((dW_2_4_dt[i,j,k]).+
#         dW_2_1_dz.-gw.*((W_3_1[i,j,k]).*(W_1_4[i,j,k]).-(W_1_1[i,j,k]).*(W_3_4[i,j,k])))

#     # s[13]=0.
#     # s[14]=
#     @inbounds k_W_3_2[i,j,k] =((dW_3_2_dt[i,j,k]).+
#         dW_3_1_dx.-gw.*((W_1_1[i,j,k]).*(W_2_2[i,j,k]).-(W_2_1[i,j,k]).*(W_1_2[i,j,k])))
#     # s[15]=
#     @inbounds k_W_3_3[i,j,k] =((dW_3_3_dt[i,j,k]).+
#         dW_3_1_dy.-gw.*((W_1_1[i,j,k]).*(W_2_3[i,j,k]).-(W_2_1[i,j,k]).*(W_1_3[i,j,k])))
#     # s[16]=
#     @inbounds k_W_3_4[i,j,k] =((dW_3_4_dt[i,j,k]).+
#         dW_3_1_dz.-gw.*((W_1_1[i,j,k]).*(W_2_4[i,j,k]).-(W_2_1[i,j,k]).*(W_1_4[i,j,k])))

#     # s[17]=0.
#     # s[18]=
#     @inbounds k_Y_2[i,j,k] =((dY_2_dt[i,j,k]).+dY_1_dx)
#     # s[19]=
#     @inbounds k_Y_3[i,j,k] =((dY_3_dt[i,j,k]).+dY_1_dy)
#     # s[20]=
#     @inbounds k_Y_4[i,j,k] =((dY_4_dt[i,j,k]).+dY_1_dz)

#     # c fluxes for gauge functions:
#     # cc if on boundaries:
#     # c      if(abs(i).eq.latx.or.abs(j).eq.laty.or.abs(k).eq.latz) then
#     # cc radial unit vector:
#     # c        px=dfloat(i)/sqrt(dfloat(i**2+j**2+k**2))
#     # c        py=dfloat(j)/sqrt(dfloat(i**2+j**2+k**2))
#     # c        pz=dfloat(k)/sqrt(dfloat(i**2+j**2+k**2))
#     # cc
#     # c       s(21)=-(px*dfdx(21)+py*dfdy(21)+pz*dfdz(21))
#     # c       s(22)=-(px*dfdx(22)+py*dfdy(22)+pz*dfdz(22))
#     # c       s(23)=-(px*dfdx(23)+py*dfdy(23)+pz*dfdz(23))
#     # c       s(24)=-(px*dfdx(24)+py*dfdy(24)+pz*dfdz(24))
#     # c
#     # cc if not on boundaries:
#     # c      else
#     # c
#     # s(Γ_1)=
#     @inbounds k_Γ_1[i,j,k] =((1.0.-gp2).*(d2W_1_2_dx2 .+d2W_1_3_dy2 .+d2W_1_4_dz2).+
#         gp2 .*gw.*(
#         -((W_2_2[i,j,k]).*(dW_3_2_dt[i,j,k]).-
#         (W_3_2[i,j,k]).*(dW_2_2_dt[i,j,k])).-
#         ((W_2_3[i,j,k]).*(dW_3_3_dt[i,j,k]).-
#         (W_3_3[i,j,k]).*(dW_2_3_dt[i,j,k])).-
#         ((W_2_4[i,j,k]).*(dW_3_4_dt[i,j,k]).-
#         (W_3_4[i,j,k]).*(dW_2_4_dt[i,j,k]))).+
#     # c charge from Higgs: 
#         gp2 .*gw.*((ϕ_1[i,j,k]).*((dϕ_4_dt[i,j,k]) .-Dt_ϕ_4).-
#         (ϕ_4[i,j,k]).*((dϕ_1_dt[i,j,k]) .-Dt_ϕ_1).+
#         (ϕ_3[i,j,k]).*((dϕ_2_dt[i,j,k]) .-Dt_ϕ_2).-
#         (ϕ_2[i,j,k]).*((dϕ_3_dt[i,j,k]) .-Dt_ϕ_3)))

#     # s(Γ_2)=
#     @inbounds k_Γ_2[i,j,k] =((1.0.-gp2).*(d2W_2_2_dx2 .+d2W_2_3_dy2 .+d2W_2_4_dz2).+
#         gp2 .*gw.*(
#         -((W_3_2[i,j,k]).*(dW_1_2_dt[i,j,k]).-
#         (W_1_2[i,j,k]).*(dW_3_2_dt[i,j,k])).-
#         ((W_3_3[i,j,k]).*(dW_1_3_dt[i,j,k]).-
#         (W_1_3[i,j,k]).*(dW_3_3_dt[i,j,k])).-
#         ((W_3_4[i,j,k]).*(dW_1_4_dt[i,j,k]).-
#         (W_1_4[i,j,k]).*(dW_3_4_dt[i,j,k]))).+
#     # c charge from Higgs: 
#         gp2 .*gw.*((ϕ_3[i,j,k]).*((dϕ_1_dt[i,j,k]) .-Dt_ϕ_1).-
#         (ϕ_1[i,j,k]).*((dϕ_3_dt[i,j,k]) .-Dt_ϕ_3).+
#         (ϕ_4[i,j,k]).*((dϕ_2_dt[i,j,k]) .-Dt_ϕ_2).-
#         (ϕ_2[i,j,k]).*((dϕ_4_dt[i,j,k]) .-Dt_ϕ_4)))

#     # s(Γ_3)=
#     @inbounds k_Γ_3[i,j,k] =((1.0.-gp2).*(d2W_3_2_dx2 .+d2W_3_3_dy2 .+d2W_3_4_dz2).+
#         gp2 .*gw.*(
#         -((W_1_2[i,j,k]).*(dW_2_2_dt[i,j,k]).-
#         (W_2_2[i,j,k]).*(dW_1_2_dt[i,j,k])).-
#         ((W_1_3[i,j,k]).*(dW_2_3_dt[i,j,k]).-
#         (W_2_3[i,j,k]).*(dW_1_3_dt[i,j,k])).-
#         ((W_1_4[i,j,k]).*(dW_2_4_dt[i,j,k]).-
#         (W_2_4[i,j,k]).*(dW_1_4_dt[i,j,k]))).+
#     # c current from Higgs: 
#         gp2 .*gw.*((ϕ_1[i,j,k]).*((dϕ_2_dt[i,j,k]) .-Dt_ϕ_2).-
#         (ϕ_2[i,j,k]).*((dϕ_1_dt[i,j,k]) .-Dt_ϕ_1).+
#         (ϕ_4[i,j,k]).*((dϕ_3_dt[i,j,k]) .-Dt_ϕ_3).-
#         (ϕ_3[i,j,k]).*((dϕ_4_dt[i,j,k]) .-Dt_ϕ_4)))

#     # s(Σ)=
#     @inbounds k_Σ[i,j,k] =((1.0.-gp2).*(d2Y_2_dx2 .+d2Y_3_dy2 .+d2Y_4_dz2).+
#     # c current from Higgs: 
#         gp2 .*gy.*((ϕ_1[i,j,k])*((dϕ_2_dt[i,j,k]) .-Dt_ϕ_2).-
#         (ϕ_2[i,j,k]).*((dϕ_1_dt[i,j,k]) .-Dt_ϕ_1).+
#         (ϕ_3[i,j,k]).*((dϕ_4_dt[i,j,k]) .-Dt_ϕ_4).-
#         (ϕ_4[i,j,k]).*((dϕ_3_dt[i,j,k]) .-Dt_ϕ_3)))

#     return
# end

@views function rk4_step!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    Γ_1,Γ_2,Γ_3,Σ,
    k_ϕ_1,k_ϕ_2,k_ϕ_3,k_ϕ_4,
    k_W_1_2,k_W_1_3,k_W_1_4,
    k_W_2_2,k_W_2_3,k_W_2_4,
    k_W_3_2,k_W_3_3,k_W_3_4,
    k_Y_2,k_Y_3,k_Y_4,
    k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,
    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
    kt_W_1_2,kt_W_1_3,kt_W_1_4,
    kt_W_2_2,kt_W_2_3,kt_W_2_4,
    kt_W_3_2,kt_W_3_3,kt_W_3_4,
    kt_Y_2,kt_Y_3,kt_Y_4,
    gw,gy,gp2,vev,lambda,dx)
    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
              (blockIdx().y - 1) * blockDim().y + threadIdx().y,
              (blockIdx().z - 1) * blockDim().z + threadIdx().z
    # if (i>3 && i<size(size(ϕ_1,1)-3) && j>3 && j<size(size(ϕ_1,2)-3) && k>3 && k<size(size(ϕ_1,3)-3))
    #Spatial Derivatives
    dϕ_1_dx=dfdx(ϕ_1,i,j,k,0.,dx)
    dϕ_2_dx=dfdx(ϕ_2,i,j,k,0.,dx)
    dϕ_3_dx=dfdx(ϕ_3,i,j,k,0.,dx)
    dϕ_4_dx=dfdx(ϕ_4,i,j,k,0.,dx)
    # @cuprintln(dϕ_4_dx)
    dϕ_1_dy=dfdy(ϕ_1,i,j,k,0.,dx)
    dϕ_2_dy=dfdy(ϕ_2,i,j,k,0.,dx)
    dϕ_3_dy=dfdy(ϕ_3,i,j,k,0.,dx)
    dϕ_4_dy=dfdy(ϕ_4,i,j,k,0.,dx)

    dϕ_1_dz=dfdz(ϕ_1,i,j,k,0.,dx)
    dϕ_2_dz=dfdz(ϕ_2,i,j,k,0.,dx)
    dϕ_3_dz=dfdz(ϕ_3,i,j,k,0.,dx)
    dϕ_4_dz=dfdz(ϕ_4,i,j,k,0.,dx)

    # dW_1_1_dx = dfdx(W_1_1,i,j,k,0.,dx)
    dW_1_2_dx = dfdx(W_1_2,i,j,k,0.,dx)
    dW_1_3_dx = dfdx(W_1_3,i,j,k,0.,dx)
    dW_1_4_dx = dfdx(W_1_4,i,j,k,0.,dx)

    # dW_1_1_dy = dfdy(W_1_1,i,j,k,0.,dx)
    dW_1_2_dy = dfdy(W_1_2,i,j,k,0.,dx)
    dW_1_3_dy = dfdy(W_1_3,i,j,k,0.,dx)
    dW_1_4_dy = dfdy(W_1_4,i,j,k,0.,dx)

    # dW_1_1_dz = dfdz(W_1_1,i,j,k,0.,dx)
    dW_1_2_dz = dfdz(W_1_2,i,j,k,0.,dx)
    dW_1_3_dz = dfdz(W_1_3,i,j,k,0.,dx)
    dW_1_4_dz = dfdz(W_1_4,i,j,k,0.,dx)

    # dW_2_1_dx = dfdx(W_2_1,i,j,k,0.,dx)
    dW_2_2_dx = dfdx(W_2_2,i,j,k,0.,dx)
    dW_2_3_dx = dfdx(W_2_3,i,j,k,0.,dx)
    dW_2_4_dx = dfdx(W_2_4,i,j,k,0.,dx)

    # dW_2_1_dy = dfdy(W_2_1,i,j,k,0.,dx)
    dW_2_2_dy = dfdy(W_2_2,i,j,k,0.,dx)
    dW_2_3_dy = dfdy(W_2_3,i,j,k,0.,dx)
    dW_2_4_dy = dfdy(W_2_4,i,j,k,0.,dx)

    # dW_2_1_dz = dfdz(W_2_1,i,j,k,0.,dx)
    dW_2_2_dz = dfdz(W_2_2,i,j,k,0.,dx)
    dW_2_3_dz = dfdz(W_2_3,i,j,k,0.,dx)
    dW_2_4_dz = dfdz(W_2_4,i,j,k,0.,dx)

    # dW_3_1_dx = dfdx(W_3_1,i,j,k,0.,dx)
    dW_3_2_dx = dfdx(W_3_2,i,j,k,0.,dx)
    dW_3_3_dx = dfdx(W_3_3,i,j,k,0.,dx)
    dW_3_4_dx = dfdx(W_3_4,i,j,k,0.,dx)

    # dW_3_1_dy = dfdy(W_3_1,i,j,k,0.,dx)
    dW_3_2_dy = dfdy(W_3_2,i,j,k,0.,dx)
    dW_3_3_dy = dfdy(W_3_3,i,j,k,0.,dx)
    dW_3_4_dy = dfdy(W_3_4,i,j,k,0.,dx)

    # dW_3_1_dz = dfdz(W_3_1,i,j,k,0.,dx)
    dW_3_2_dz = dfdz(W_3_2,i,j,k,0.,dx)
    dW_3_3_dz = dfdz(W_3_3,i,j,k,0.,dx)
    dW_3_4_dz = dfdz(W_3_4,i,j,k,0.,dx)

    # dY_1_dx = dfdx(Y_1,i,j,k,0.,dx)
    dY_2_dx = dfdx(Y_2,i,j,k,0.,dx)
    dY_3_dx = dfdx(Y_3,i,j,k,0.,dx)
    dY_4_dx = dfdx(Y_4,i,j,k,0.,dx)

    # dY_1_dy = dfdy(Y_1,i,j,k,0.,dx)
    dY_2_dy = dfdy(Y_2,i,j,k,0.,dx)
    dY_3_dy = dfdy(Y_3,i,j,k,0.,dx)
    dY_4_dy = dfdy(Y_4,i,j,k,0.,dx)

    # dY_1_dz = dfdz(Y_1,i,j,k,0.,dx)
    dY_2_dz = dfdz(Y_2,i,j,k,0.,dx)
    dY_3_dz = dfdz(Y_3,i,j,k,0.,dx)
    dY_4_dz = dfdz(Y_4,i,j,k,0.,dx)

    dΓ_1_dx = dfdx(Γ_1,i,j,k,0.,dx)
    dΓ_1_dy = dfdy(Γ_1,i,j,k,0.,dx)
    dΓ_1_dz = dfdz(Γ_1,i,j,k,0.,dx)

    dΓ_2_dx = dfdx(Γ_2,i,j,k,0.,dx)
    dΓ_2_dy = dfdy(Γ_2,i,j,k,0.,dx)
    dΓ_2_dz = dfdz(Γ_2,i,j,k,0.,dx)

    dΓ_3_dx = dfdx(Γ_3,i,j,k,0.,dx)
    dΓ_3_dy = dfdy(Γ_3,i,j,k,0.,dx)
    dΓ_3_dz = dfdz(Γ_3,i,j,k,0.,dx)
    
    dΣ_dx = dfdx(Σ,i,j,k,0.,dx)
    dΣ_dy = dfdy(Σ,i,j,k,0.,dx)
    dΣ_dz = dfdz(Σ,i,j,k,0.,dx)

    d2ϕ_1_dx2=d2fdx2(ϕ_1,i,j,k,0.,dx)
    d2ϕ_2_dx2=d2fdx2(ϕ_2,i,j,k,0.,dx)
    d2ϕ_3_dx2=d2fdx2(ϕ_2,i,j,k,0.,dx)
    d2ϕ_4_dx2=d2fdx2(ϕ_2,i,j,k,0.,dx)

    d2ϕ_1_dy2=d2fdy2(ϕ_1,i,j,k,0.,dx)
    d2ϕ_2_dy2=d2fdy2(ϕ_2,i,j,k,0.,dx)
    d2ϕ_3_dy2=d2fdy2(ϕ_2,i,j,k,0.,dx)
    d2ϕ_4_dy2=d2fdy2(ϕ_2,i,j,k,0.,dx)

    d2ϕ_1_dz2=d2fdz2(ϕ_1,i,j,k,0.,dx)
    d2ϕ_2_dz2=d2fdz2(ϕ_2,i,j,k,0.,dx)
    d2ϕ_3_dz2=d2fdz2(ϕ_2,i,j,k,0.,dx)
    d2ϕ_4_dz2=d2fdz2(ϕ_2,i,j,k,0.,dx)

    # d2W_1_1_dx2 = d2fdx2(W_1_1,i,j,k,0.,dx)
    d2W_1_2_dx2 = d2fdx2(W_1_2,i,j,k,0.,dx)
    d2W_1_3_dx2 = d2fdx2(W_1_3,i,j,k,0.,dx)
    d2W_1_4_dx2 = d2fdx2(W_1_4,i,j,k,0.,dx)

    # d2W_1_1_dy2 = d2fdy2(W_1_1,i,j,k,0.,dx)
    d2W_1_2_dy2 = d2fdy2(W_1_2,i,j,k,0.,dx)
    d2W_1_3_dy2 = d2fdy2(W_1_3,i,j,k,0.,dx)
    d2W_1_4_dy2 = d2fdy2(W_1_4,i,j,k,0.,dx)

    # d2W_1_1_dz2 = d2fdz2(W_1_1,i,j,k,0.,dx)
    d2W_1_2_dz2 = d2fdz2(W_1_2,i,j,k,0.,dx)
    d2W_1_3_dz2 = d2fdz2(W_1_3,i,j,k,0.,dx)
    d2W_1_4_dz2 = d2fdz2(W_1_4,i,j,k,0.,dx)

    # d2W_2_1_dx2 = d2fdx2(W_2_1,i,j,k,0.,dx)
    d2W_2_2_dx2 = d2fdx2(W_2_2,i,j,k,0.,dx)
    d2W_2_3_dx2 = d2fdx2(W_2_3,i,j,k,0.,dx)
    d2W_2_4_dx2 = d2fdx2(W_2_4,i,j,k,0.,dx)

    # d2W_2_1_dy2 = d2fdy2(W_2_1,i,j,k,0.,dx)
    d2W_2_2_dy2 = d2fdy2(W_2_2,i,j,k,0.,dx)
    d2W_2_3_dy2 = d2fdy2(W_2_3,i,j,k,0.,dx)
    d2W_2_4_dy2 = d2fdy2(W_2_4,i,j,k,0.,dx)

    # d2W_2_1_dz2 = d2fdz2(W_2_1,i,j,k,0.,dx)
    d2W_2_2_dz2 = d2fdz2(W_2_2,i,j,k,0.,dx)
    d2W_2_3_dz2 = d2fdz2(W_2_3,i,j,k,0.,dx)
    d2W_2_4_dz2 = d2fdz2(W_2_4,i,j,k,0.,dx)

    # d2W_3_1_dx2 = d2fdx2(W_3_1,i,j,k,0.,dx)
    d2W_3_2_dx2 = d2fdx2(W_3_2,i,j,k,0.,dx)
    d2W_3_3_dx2 = d2fdx2(W_3_3,i,j,k,0.,dx)
    d2W_3_4_dx2 = d2fdx2(W_3_4,i,j,k,0.,dx)

    # d2W_3_1_dy2 = d2fdy2(W_3_1,i,j,k,0.,dx)
    d2W_3_2_dy2 = d2fdy2(W_3_2,i,j,k,0.,dx)
    d2W_3_3_dy2 = d2fdy2(W_3_3,i,j,k,0.,dx)
    d2W_3_4_dy2 = d2fdy2(W_3_4,i,j,k,0.,dx)

    # d2W_3_1_dz2 = d2fdz2(W_3_1,i,j,k,0.,dx)
    d2W_3_2_dz2 = d2fdz2(W_3_2,i,j,k,0.,dx)
    d2W_3_3_dz2 = d2fdz2(W_3_3,i,j,k,0.,dx)
    d2W_3_4_dz2 = d2fdz2(W_3_4,i,j,k,0.,dx)

    # d2Y_1_dx2 = d2fdx2(Y_1,i,j,k,0.,dx)
    d2Y_2_dx2 = d2fdx2(Y_2,i,j,k,0.,dx)
    d2Y_3_dx2 = d2fdx2(Y_3,i,j,k,0.,dx)
    d2Y_4_dx2 = d2fdx2(Y_4,i,j,k,0.,dx)

    # d2Y_1_dy2 = d2fdy2(Y_1,i,j,k,0.,dx)
    d2Y_2_dy2 = d2fdy2(Y_2,i,j,k,0.,dx)
    d2Y_3_dy2 = d2fdy2(Y_3,i,j,k,0.,dx)
    d2Y_4_dy2 = d2fdy2(Y_4,i,j,k,0.,dx)

    # d2Y_1_dz2 = d2fdz2(Y_1,i,j,k,0.,dx)
    d2Y_2_dz2 = d2fdz2(Y_2,i,j,k,0.,dx)
    d2Y_3_dz2 = d2fdz2(Y_3,i,j,k,0.,dx)
    d2Y_4_dz2 = d2fdz2(Y_4,i,j,k,0.,dx)

    # ##Covariant Derivatives##

    # Dt_ϕ_1 =D_1ϕ_1(dϕ_1_dt[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
    # W_1_1[i,j,k],W_2_1[i,j,k],W_3_1[i,j,k],Y_1[i,j,k],gw,gy)
    # Dt_ϕ_2 =D_1ϕ_2(dϕ_2_dt[i,j,k],ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
    # W_1_1[i,j,k],W_2_1[i,j,k],W_3_1[i,j,k],Y_1[i,j,k],gw,gy)
    # Dt_ϕ_3 =D_1ϕ_3(dϕ_3_dt[i,j,k],ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
    # W_1_1[i,j,k],W_2_1[i,j,k],W_3_1[i,j,k],Y_1[i,j,k],gw,gy)
    # Dt_ϕ_4 =D_1ϕ_4(dϕ_4_dt[i,j,k],ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
    # W_1_1[i,j,k],W_2_1[i,j,k],W_3_1[i,j,k],Y_1[i,j,k],gw,gy)
    # Dx_ϕ_1 =D_2ϕ_1(dϕ_1_dx,ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
    # W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy)
    # Dx_ϕ_2 =D_2ϕ_2(dϕ_2_dx,ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
    # W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy)
    # Dx_ϕ_3 =D_2ϕ_3(dϕ_3_dx,ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
    # W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy)
    # Dx_ϕ_4 =D_2ϕ_4(dϕ_4_dx,ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
    # W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy)
    # Dy_ϕ_1 =D_3ϕ_1(dϕ_1_dy,ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
    # W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy)
    # Dy_ϕ_2 =D_3ϕ_2(dϕ_2_dy,ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
    # W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy)
    # Dy_ϕ_3 =D_3ϕ_3(dϕ_3_dy,ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
    # W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy)
    # Dy_ϕ_4 =D_3ϕ_4(dϕ_4_dy,ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
    # W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy)
    # Dz_ϕ_1 =D_4ϕ_1(dϕ_1_dz,ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
    # W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)
    # Dz_ϕ_2 =D_4ϕ_2(dϕ_2_dz,ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
    # W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)
    # Dz_ϕ_3 =D_4ϕ_3(dϕ_3_dz,ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
    # W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)
    # Dz_ϕ_4 =D_4ϕ_4(dϕ_4_dz,ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
    # W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)

    ##Covariant Derivatives##Temporal gauge

    Dt_ϕ_1 =dϕ_1_dt[i,j,k]
    Dt_ϕ_2 =dϕ_2_dt[i,j,k]
    Dt_ϕ_3 =dϕ_3_dt[i,j,k]
    Dt_ϕ_4 =dϕ_4_dt[i,j,k]
    Dx_ϕ_1 =D_2ϕ_1(dϕ_1_dx,ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
    W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy)
    Dx_ϕ_2 =D_2ϕ_2(dϕ_2_dx,ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
    W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy)
    Dx_ϕ_3 =D_2ϕ_3(dϕ_3_dx,ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
    W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy)
    Dx_ϕ_4 =D_2ϕ_4(dϕ_4_dx,ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
    W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy)
    Dy_ϕ_1 =D_3ϕ_1(dϕ_1_dy,ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
    W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy)
    Dy_ϕ_2 =D_3ϕ_2(dϕ_2_dy,ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
    W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy)
    Dy_ϕ_3 =D_3ϕ_3(dϕ_3_dy,ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
    W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy)
    Dy_ϕ_4 =D_3ϕ_4(dϕ_4_dy,ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
    W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy)
    Dz_ϕ_1 =D_4ϕ_1(dϕ_1_dz,ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
    W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)
    Dz_ϕ_2 =D_4ϕ_2(dϕ_2_dz,ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
    W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)
    Dz_ϕ_3 =D_4ϕ_3(dϕ_3_dz,ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
    W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)
    Dz_ϕ_4 =D_4ϕ_4(dϕ_4_dz,ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
    W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)

    # Field Strengths #
    # W_1_11()
    # W_1_12(dW_1_2_dt)
    # W_1_13(dW_1_3_dt)
    # W_1_14(dW_1_4_dt)
    # W_1_22(dW_2_2_dt)
    W_1_23 =W_1_xy(dW_1_3_dx,dW_1_2_dy,W_2_2,W_3_3,W_2_3,W_3_2,gw,i,j,k)
    W_1_24=W_1_xz(dW_1_4_dx,dW_1_2_dz,W_2_2,W_3_4,W_2_4,W_3_2,gw,i,j,k)
    # W_1_33(dW_3_3_dt)
    W_1_34=W_1_yz(dW_1_4_dy,dW_1_3_dz,W_2_3,W_3_4,W_2_4,W_3_3,gw,i,j,k)
    # W_1_44()
    # W_2_11()
    # W_2_12(dW_2_2_dt)
    # W_2_13(dW_2_3_dt)
    # W_2_14(dW_2_4_dt)
    # W_2_22()
    W_2_23=W_2_xy(dW_2_3_dx,dW_2_2_dy,W_3_2,W_1_3,W_3_3,W_1_2,gw,i,j,k)
    W_2_24=W_2_xz(dW_2_4_dx,dW_2_2_dz,W_3_2,W_1_4,W_3_4,W_1_2,gw,i,j,k)
    # W_2_33()
    W_2_34=W_2_yz(dW_2_4_dy,dW_2_3_dz,W_3_3,W_1_4,W_3_4,W_1_3,gw,i,j,k)
    # W_2_44()
    # W_3_11()
    # W_3_12(dW_3_2_dt)
    # W_3_13(dW_3_3_dt)
    # W_3_14(dW_3_4_dt)
    # W_3_22()
    W_3_23=W_3_xy(dW_3_3_dx,dW_3_2_dy,W_1_2,W_2_3,W_1_3,W_2_2,gw,i,j,k)
    W_3_24=W_3_xz(dW_3_4_dx,dW_3_2_dz,W_1_2,W_2_4,W_1_4,W_2_2,gw,i,j,k)
    # W_3_33(dW_3_3_dt)
    W_3_34=W_3_yz(dW_3_4_dy,dW_3_3_dz,W_1_3,W_2_4,W_1_4,W_2_3,gw,i,j,k)
    # W_3_44()
    # Y_1_1()
    # Y_1_2(dY_2_dt)
    # Y_1_3(dY_3_dt)
    # Y_1_4(dY_4_dt)
    # Y_2_2()
    Y_2_3=Y_2_y(dY_3_dx,dY_2_dy)
    Y_2_4=Y_2_z(dY_4_dx,dY_2_dz)
    # Y_3_3()
    Y_3_4=Y_3_z(dY_4_dy,dY_3_dz)
    # Y_4_4()

    # kt_1 expressions
    @inbounds kt_ϕ_1[i,j,k] = ((d2ϕ_1_dx2+d2ϕ_1_dy2+d2ϕ_1_dz2-
    0.5*gw*(((-W_1_2[i,j,k]*dϕ_4_dx)-(W_1_3[i,j,k]*dϕ_4_dy)-(W_1_4[i,j,k]*dϕ_4_dz))-
    ((-W_2_2[i,j,k]*dϕ_3_dx)-(W_2_3[i,j,k]*dϕ_3_dy)-(W_2_4[i,j,k]*dϕ_3_dz))+
    ((-W_3_2[i,j,k]*dϕ_2_dx)-(W_3_3[i,j,k]*dϕ_2_dy)-(W_3_4[i,j,k]*dϕ_2_dz)))-
    0.5*gy*(-Y_2[i,j,k]*dϕ_2_dx-Y_3[i,j,k]*dϕ_2_dy-Y_4[i,j,k]*dϕ_2_dz)-
    0.5*gw*((-W_1_2[i,j,k]*Dx_ϕ_4-W_1_3[i,j,k]*Dy_ϕ_4-W_1_4[i,j,k]*Dz_ϕ_4)-
    (-W_2_2[i,j,k]*Dx_ϕ_3-W_2_3[i,j,k]*Dy_ϕ_3-W_2_4[i,j,k]*Dz_ϕ_3)+
    (-W_3_2[i,j,k]*Dx_ϕ_2-W_3_3[i,j,k]*Dy_ϕ_2-W_3_4[i,j,k]*Dz_ϕ_2))-
    0.5*gy*(-Y_2[i,j,k]*Dx_ϕ_2-Y_3[i,j,k]*Dy_ϕ_2-Y_4[i,j,k]*Dz_ϕ_2)-
    2.0*lambda*(ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2-vev^2)*ϕ_1[i,j,k]+
    0.5*((gw*Γ_3[i,j,k]+gy*Σ[i,j,k])*ϕ_2[i,j,k]-gw*Γ_2[i,j,k]*ϕ_3[i,j,k]+gw*Γ_1[i,j,k]*ϕ_4[i,j,k])))

    @inbounds kt_ϕ_2[i,j,k] = ((d2ϕ_2_dx2+d2ϕ_2_dy2+d2ϕ_2_dz2+
    0.5*gw*((-W_1_2[i,j,k]*dϕ_3_dx-W_1_3[i,j,k]*dϕ_3_dy-W_1_4[i,j,k]*dϕ_3_dz)+
    (-W_2_2[i,j,k]*dϕ_4_dx-W_2_3[i,j,k]*dϕ_4_dy-W_2_4[i,j,k]*dϕ_4_dz)+
    (-W_3_2[i,j,k]*dϕ_1_dx-W_3_3[i,j,k]*dϕ_1_dy-W_3_4[i,j,k]*dϕ_1_dz))+
    0.5*gy*(-Y_2[i,j,k]*dϕ_1_dx-Y_3[i,j,k]*dϕ_1_dy-Y_4[i,j,k]*dϕ_1_dz)+
    0.5*gw*((-W_1_2[i,j,k]*Dx_ϕ_3-W_1_3[i,j,k]*Dy_ϕ_3-W_1_4[i,j,k]*Dz_ϕ_3)+
    (-W_2_2[i,j,k]*Dx_ϕ_4-W_2_3[i,j,k]*Dy_ϕ_4-W_2_4[i,j,k]*Dz_ϕ_4)+
    (-W_3_2[i,j,k]*Dx_ϕ_1-W_3_3[i,j,k]*Dy_ϕ_1-W_3_4[i,j,k]*Dz_ϕ_1))+
    0.5*gy*(-Y_2[i,j,k]*Dx_ϕ_1-Y_3[i,j,k]*Dy_ϕ_1-Y_4[i,j,k]*Dz_ϕ_1)-
    2.0*lambda*(ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2-vev^2)*ϕ_2[i,j,k]-
    0.5*((gw*Γ_3[i,j,k]+gy*Σ[i,j,k])*ϕ_1[i,j,k]+gw*Γ_1[i,j,k]*ϕ_3[i,j,k]+gw*Γ_2[i,j,k]*ϕ_4[i,j,k])))

    @inbounds kt_ϕ_3[i,j,k] = ((d2ϕ_3_dx2+d2ϕ_3_dy2+d2ϕ_3_dz2-
    0.5*gw*((-W_1_2[i,j,k]*dϕ_2_dx-W_1_3[i,j,k]*dϕ_2_dy-W_1_4[i,j,k]*dϕ_2_dz)+
    (-W_2_2[i,j,k]*dϕ_1_dx-W_2_3[i,j,k]*dϕ_1_dy-W_2_4[i,j,k]*dϕ_1_dz)-
    (-W_3_2[i,j,k]*dϕ_4_dx-W_3_3[i,j,k]*dϕ_4_dy-W_3_4[i,j,k]*dϕ_4_dz))-
    0.5*gy*(-Y_2[i,j,k]*dϕ_4_dx-Y_3[i,j,k]*dϕ_4_dy-Y_4[i,j,k]*dϕ_4_dz)-
    0.5*gw*((-W_1_2[i,j,k]*Dx_ϕ_2-W_1_3[i,j,k]*Dy_ϕ_2-W_1_4[i,j,k]*Dz_ϕ_2)+
    (-W_2_2[i,j,k]*Dx_ϕ_1-W_2_3[i,j,k]*Dy_ϕ_1-W_2_4[i,j,k]*Dz_ϕ_1)-
    (-W_3_2[i,j,k]*Dx_ϕ_4-W_3_3[i,j,k]*Dy_ϕ_4-W_3_4[i,j,k]*Dz_ϕ_4))-
    0.5*gy*(-Y_2[i,j,k]*Dx_ϕ_4-Y_3[i,j,k]*Dy_ϕ_4-Y_4[i,j,k]*Dz_ϕ_4)-
    2.0*lambda*(ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2-vev^2)*ϕ_3[i,j,k]+
    0.5*((-gw*Γ_3[i,j,k]+gy*Σ[i,j,k])*ϕ_4[i,j,k]+gw*Γ_2[i,j,k]*ϕ_1[i,j,k]+gw*Γ_1[i,j,k]*ϕ_2[i,j,k])))

    @inbounds kt_ϕ_4[i,j,k] = ((d2ϕ_4_dx2+d2ϕ_4_dy2+d2ϕ_4_dz2+
    0.5*gw*((-W_1_2[i,j,k]*dϕ_1_dx-W_1_3[i,j,k]*dϕ_1_dy-W_1_4[i,j,k]*dϕ_1_dz)-
    (-W_2_2[i,j,k]*dϕ_2_dx-W_2_3[i,j,k]*dϕ_2_dy-W_2_4[i,j,k]*dϕ_2_dz)-
    (-W_3_2[i,j,k]*dϕ_3_dx-W_3_3[i,j,k]*dϕ_3_dy-W_3_4[i,j,k]*dϕ_3_dz))+
    0.5*gy*(-Y_2[i,j,k]*dϕ_3_dx-Y_3[i,j,k]*dϕ_3_dy-Y_4[i,j,k]*dϕ_3_dz)+
    0.5*gw*((-W_1_2[i,j,k]*Dx_ϕ_1-W_1_3[i,j,k]*Dy_ϕ_1-W_1_4[i,j,k]*Dz_ϕ_1)-
    (-W_2_2[i,j,k]*Dx_ϕ_2-W_2_3[i,j,k]*Dy_ϕ_2-W_2_4[i,j,k]*Dz_ϕ_2)-
    (-W_3_2[i,j,k]*Dx_ϕ_3-W_3_3[i,j,k]*Dy_ϕ_3-W_3_4[i,j,k]*Dz_ϕ_3))+
    0.5*gy*(-Y_2[i,j,k]*Dx_ϕ_3-Y_3[i,j,k]*Dy_ϕ_3-Y_4[i,j,k]*Dz_ϕ_3)-
    2.0*lambda*(ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2-vev^2)*ϕ_4[i,j,k]-
    0.5*((-gw*Γ_3[i,j,k]+gy*Σ[i,j,k])*ϕ_3[i,j,k]+gw*Γ_1[i,j,k]*ϕ_1[i,j,k]-gw*Γ_2[i,j,k]*ϕ_2[i,j,k])))

    @inbounds kt_W_1_2[i,j,k] = (d2W_1_2_dx2+d2W_1_2_dy2+d2W_1_2_dz2+
    gw*(-(dW_2_2_dx*W_3_2[i,j,k]-dW_3_2_dx*W_2_2[i,j,k])-
    (dW_2_2_dy*W_3_3[i,j,k]-dW_3_2_dy*W_2_3[i,j,k])-
    (dW_2_2_dz*W_3_4[i,j,k]-dW_3_2_dz*W_2_4[i,j,k])-
    (W_2_3[i,j,k]*W_3_23-W_3_3[i,j,k]*W_2_23)-
    (W_2_4[i,j,k]*W_3_24-W_3_4[i,j,k]*W_2_24))+
    gw*(ϕ_1[i,j,k]*Dx_ϕ_4-ϕ_2[i,j,k]*Dx_ϕ_3+ϕ_3[i,j,k]*Dx_ϕ_2-ϕ_4[i,j,k]*Dx_ϕ_1)-
    dΓ_1_dx-gw*(W_2_2[i,j,k]*Γ_3[i,j,k]-W_3_2[i,j,k]*Γ_2[i,j,k]))

    @inbounds kt_W_1_3[i,j,k] = (d2W_1_3_dx2+d2W_1_3_dy2+d2W_1_3_dz2+
    gw*(-(dW_2_3_dx*W_3_2[i,j,k]-dW_3_3_dx*W_2_2[i,j,k])-
    (dW_2_3_dy*W_3_3[i,j,k]-dW_3_3_dy*W_2_3[i,j,k])-
    (dW_2_3_dz*W_3_4[i,j,k]-dW_3_3_dz*W_2_4[i,j,k])-
    (W_2_2[i,j,k]*(-W_3_23)-W_3_2[i,j,k]*(-W_2_23))-
    (W_2_4[i,j,k]*W_3_34-W_3_4[i,j,k]*W_2_34))+
    gw*(ϕ_1[i,j,k]*Dy_ϕ_4-ϕ_2[i,j,k]*Dy_ϕ_3+ϕ_3[i,j,k]*Dy_ϕ_2-ϕ_4[i,j,k]*Dy_ϕ_1)-
    dΓ_1_dy-gw*(W_2_3[i,j,k]*Γ_3[i,j,k]-W_3_3[i,j,k]*Γ_2[i,j,k]))

    @inbounds kt_W_1_4[i,j,k] = (d2W_1_4_dx2+d2W_1_4_dy2+d2W_1_4_dz2+
    gw*(-(dW_2_4_dx*W_3_2[i,j,k]-dW_3_4_dx*W_2_2[i,j,k])-
    (dW_2_4_dy*W_3_3[i,j,k]-dW_3_4_dy*W_2_3[i,j,k])-
    (dW_2_4_dz*W_3_4[i,j,k]-dW_3_4_dz*W_2_4[i,j,k])-
    (W_2_2[i,j,k]*(-W_3_24)-W_3_2[i,j,k]*(-W_2_24))-
    (W_2_3[i,j,k]*(-W_3_34)-W_3_3[i,j,k]*(-W_2_34)))+
    gw*(ϕ_1[i,j,k]*Dz_ϕ_4-ϕ_2[i,j,k]*Dz_ϕ_3+ϕ_3[i,j,k]*Dz_ϕ_2-ϕ_4[i,j,k]*Dz_ϕ_1)-
    dΓ_1_dz-gw*(W_2_4[i,j,k]*Γ_3[i,j,k]-W_3_4[i,j,k]*Γ_2[i,j,k]))

    @inbounds kt_W_2_2[i,j,k] = (d2W_2_2_dx2+d2W_2_2_dy2+d2W_2_2_dz2+
    gw*(-(dW_3_2_dx*W_1_2[i,j,k]-dW_1_2_dx*W_3_2[i,j,k])-
    (dW_3_2_dy*W_1_3[i,j,k]-dW_1_2_dy*W_3_3[i,j,k])-
    (dW_3_2_dz*W_1_4[i,j,k]-dW_1_2_dz*W_3_4[i,j,k])-
    (W_3_3[i,j,k]*W_1_23-W_1_3[i,j,k]*W_3_23)-
    (W_3_4[i,j,k]*W_1_24-W_1_4[i,j,k]*W_3_24))+
    gw*(-ϕ_1[i,j,k]*Dx_ϕ_3-ϕ_2[i,j,k]*Dx_ϕ_4+ϕ_3[i,j,k]*Dx_ϕ_1+ϕ_4[i,j,k]*Dx_ϕ_2)-
    dΓ_2_dx-gw*(W_3_2[i,j,k]*Γ_1[i,j,k]-W_1_2[i,j,k]*Γ_3[i,j,k]))

    @inbounds kt_W_2_3[i,j,k] = (d2W_2_3_dx2+d2W_2_3_dy2+d2W_2_3_dz2+
    gw*(-(dW_3_3_dx*W_1_2[i,j,k]-dW_1_3_dx*W_3_2[i,j,k])-
    (dW_3_3_dy*W_1_3[i,j,k]-dW_1_3_dy*W_3_3[i,j,k])-
    (dW_3_3_dz*W_1_4[i,j,k]-dW_1_3_dz*W_3_4[i,j,k])-
    (W_3_2[i,j,k]*(-W_1_23)-W_1_2[i,j,k]*(-W_3_23))-
    (W_3_4[i,j,k]*(W_1_34)-W_1_4[i,j,k]*W_3_34))+
    gw*(-ϕ_1[i,j,k]*Dy_ϕ_3-ϕ_2[i,j,k]*Dy_ϕ_4+ϕ_3[i,j,k]*Dy_ϕ_1+ϕ_4[i,j,k]*Dy_ϕ_2)-
    dΓ_2_dy-gw*(W_3_3[i,j,k]*Γ_1[i,j,k]-W_1_3[i,j,k]*Γ_3[i,j,k]))

    @inbounds kt_W_2_4[i,j,k] = (d2W_2_4_dx2+d2W_2_4_dy2+d2W_2_4_dz2+
    gw*(-(dW_3_4_dx*W_1_2[i,j,k]-dW_1_4_dx*W_3_2[i,j,k])-
    (dW_3_4_dy*W_1_3[i,j,k]-dW_1_4_dy*W_3_3[i,j,k])-
    (dW_3_4_dz*W_1_4[i,j,k]-dW_1_4_dz*W_3_4[i,j,k])-
    (W_3_2[i,j,k]*(-W_1_24)-W_1_2[i,j,k]*(-W_3_24))-
    (W_3_3[i,j,k]*(-W_1_34)-W_1_3[i,j,k]*(-W_3_34)))+
    gw*(-ϕ_1[i,j,k]*Dz_ϕ_3-ϕ_2[i,j,k]*Dz_ϕ_4+ϕ_3[i,j,k]*Dz_ϕ_1+ϕ_4[i,j,k]*Dz_ϕ_2)-
    dΓ_2_dz-gw*(W_3_4[i,j,k]*Γ_1[i,j,k]-W_1_4[i,j,k]*Γ_3[i,j,k]))

    @inbounds kt_W_3_2[i,j,k] = (d2W_3_2_dx2+d2W_3_2_dy2+d2W_3_2_dz2+
    gw*(-(dW_1_2_dx*W_2_2[i,j,k]-dW_2_2_dx*W_1_2[i,j,k])-
    (dW_1_2_dy*W_2_3[i,j,k]-dW_2_2_dy*W_1_3[i,j,k])-
    (dW_1_2_dz*W_2_4[i,j,k]-dW_2_2_dz*W_1_4[i,j,k])-
    (W_1_3[i,j,k]*W_2_23-W_2_3[i,j,k]*W_1_23)-
    (W_1_4[i,j,k]*W_2_24-W_2_4[i,j,k]*W_1_24))+
    gw*(ϕ_1[i,j,k]*Dx_ϕ_2-ϕ_2[i,j,k]*Dx_ϕ_1-ϕ_3[i,j,k]*Dx_ϕ_4+ϕ_4[i,j,k]*Dx_ϕ_3)-
    dΓ_3_dx-gw*(W_1_2[i,j,k]*Γ_2[i,j,k]-W_2_2[i,j,k]*Γ_1[i,j,k]))

    @inbounds kt_W_3_3[i,j,k] = (d2W_3_3_dx2+d2W_3_3_dy2+d2W_3_3_dz2+
    gw*(-(dW_1_3_dx*W_2_2[i,j,k]-dW_2_3_dx*W_1_2[i,j,k])-
        (dW_1_3_dy*W_2_3[i,j,k]-dW_2_3_dy*W_1_3[i,j,k])-
        (dW_1_3_dz*W_2_4[i,j,k]-dW_2_3_dz*W_1_4[i,j,k])-
        (W_1_2[i,j,k]*(-W_2_23)-W_2_2[i,j,k]*(-W_1_23))-
        (W_1_4[i,j,k]*W_2_34-W_2_4[i,j,k]*(W_1_34)))+
    gw*(ϕ_1[i,j,k]*Dy_ϕ_2-ϕ_2[i,j,k]*Dy_ϕ_1-ϕ_3[i,j,k]*Dy_ϕ_4+ϕ_4[i,j,k]*Dy_ϕ_3)-
    dΓ_3_dy-gw*(W_1_3[i,j,k]*Γ_2[i,j,k]-W_2_3[i,j,k]*Γ_1[i,j,k]))

    @inbounds kt_W_3_4[i,j,k] = (d2W_3_4_dx2+d2W_3_4_dy2+d2W_3_4_dz2+
    gw*(-(dW_1_4_dx*W_2_2[i,j,k]-dW_2_4_dx*W_1_2[i,j,k])-
        (dW_1_4_dy*W_2_3[i,j,k]-dW_2_4_dy*W_1_3[i,j,k])-
        (dW_1_4_dz*W_2_4[i,j,k]-dW_2_4_dz*W_1_4[i,j,k])-
        (W_1_2[i,j,k]*(-W_2_24)-W_2_2[i,j,k]*(-W_1_24))-
        (W_1_3[i,j,k]*(-W_2_34)-W_2_3[i,j,k]*(-W_1_34)))+
    gw*(ϕ_1[i,j,k]*Dz_ϕ_2-ϕ_2[i,j,k]*Dz_ϕ_1-ϕ_3[i,j,k]*Dz_ϕ_4+ϕ_4[i,j,k]*Dz_ϕ_3)-
    dΓ_3_dz-gw*(W_1_4[i,j,k]*Γ_2[i,j,k]-W_2_4[i,j,k]*Γ_1[i,j,k]))

    @inbounds kt_Y_2[i,j,k] = ((d2Y_2_dx2+d2Y_2_dy2+d2Y_2_dz2+
    gy*(ϕ_1[i,j,k]*Dx_ϕ_2-ϕ_2[i,j,k]*Dx_ϕ_1+ϕ_3[i,j,k]*Dx_ϕ_4-ϕ_4[i,j,k]*Dx_ϕ_3)-dΣ_dx))

    @inbounds kt_Y_3[i,j,k] = ((d2Y_3_dx2+d2Y_3_dy2+d2Y_3_dz2+
    gy*(ϕ_1[i,j,k]*Dy_ϕ_2-ϕ_2[i,j,k]*Dy_ϕ_1+ϕ_3[i,j,k]*Dy_ϕ_4-ϕ_4[i,j,k]*Dy_ϕ_3)-dΣ_dy))

    @inbounds kt_Y_4[i,j,k] = ((d2Y_4_dx2+d2Y_4_dy2+d2Y_4_dz2+
    gy*(ϕ_1[i,j,k]*Dz_ϕ_2-ϕ_2[i,j,k]*Dz_ϕ_1+ϕ_3[i,j,k]*Dz_ϕ_4-ϕ_4[i,j,k]*Dz_ϕ_3)-dΣ_dz))

    # k_1 expressions
    @inbounds k_ϕ_1[i,j,k] =(dϕ_1_dt[i,j,k])
    @inbounds k_ϕ_2[i,j,k] =(dϕ_2_dt[i,j,k])
    @inbounds k_ϕ_3[i,j,k] =(dϕ_3_dt[i,j,k])
    @inbounds k_ϕ_4[i,j,k] =(dϕ_4_dt[i,j,k])
    # c
    # c Eq. (2.11) of Baumgarte&Shapiro is $\partial_t A_i = -E_i -...$ so 
    # c we are taking fd(...)=+\partial_t A_i = -E_i (note the sign).
    # s[5]=0.
    # s[6]=
    @inbounds k_W_1_2[i,j,k] =dW_1_2_dt[i,j,k]
    # s[7]=
    @inbounds k_W_1_3[i,j,k] =dW_1_3_dt[i,j,k]
    # s[8]=
    @inbounds k_W_1_4[i,j,k] =dW_1_4_dt[i,j,k]

    # s[9]=0.
    # s[10]=
    @inbounds k_W_2_2[i,j,k] =dW_2_2_dt[i,j,k]
    # s[11]=
    @inbounds k_W_2_3[i,j,k] =dW_2_3_dt[i,j,k]
    # s[12]=
    @inbounds k_W_2_4[i,j,k] =dW_2_4_dt[i,j,k]

    # s[13]=0.
    # s[14]=
    @inbounds k_W_3_2[i,j,k] =dW_3_2_dt[i,j,k]
    # s[15]=
    @inbounds k_W_3_3[i,j,k] =dW_3_3_dt[i,j,k]
    # s[16]=
    @inbounds k_W_3_4[i,j,k] =dW_3_4_dt[i,j,k]

    # s[17]=0.
    # s[18]=
    @inbounds k_Y_2[i,j,k] =dY_2_dt[i,j,k]
    # s[19]=
    @inbounds k_Y_3[i,j,k] =dY_3_dt[i,j,k]
    # s[20]=
    @inbounds k_Y_4[i,j,k] =dY_4_dt[i,j,k]

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
    @inbounds k_Γ_1[i,j,k] =((1.0.-gp2).*(d2W_1_2_dx2 .+d2W_1_3_dy2 .+d2W_1_4_dz2).+
        gp2 .*gw.*(
        -((W_2_2[i,j,k]).*(dW_3_2_dt[i,j,k]).-
        (W_3_2[i,j,k]).*(dW_2_2_dt[i,j,k])).-
        ((W_2_3[i,j,k]).*(dW_3_3_dt[i,j,k]).-
        (W_3_3[i,j,k]).*(dW_2_3_dt[i,j,k])).-
        ((W_2_4[i,j,k]).*(dW_3_4_dt[i,j,k]).-
        (W_3_4[i,j,k]).*(dW_2_4_dt[i,j,k]))).+
    # c charge from Higgs: 
        gp2 .*gw.*((ϕ_1[i,j,k]).*((dϕ_4_dt[i,j,k]) .-Dt_ϕ_4).-
        (ϕ_4[i,j,k]).*((dϕ_1_dt[i,j,k]) .-Dt_ϕ_1).+
        (ϕ_3[i,j,k]).*((dϕ_2_dt[i,j,k]) .-Dt_ϕ_2).-
        (ϕ_2[i,j,k]).*((dϕ_3_dt[i,j,k]) .-Dt_ϕ_3)))

    # s(Γ_2)=
    @inbounds k_Γ_2[i,j,k] =((1.0.-gp2).*(d2W_2_2_dx2 .+d2W_2_3_dy2 .+d2W_2_4_dz2).+
        gp2 .*gw.*(
        -((W_3_2[i,j,k]).*(dW_1_2_dt[i,j,k]).-
        (W_1_2[i,j,k]).*(dW_3_2_dt[i,j,k])).-
        ((W_3_3[i,j,k]).*(dW_1_3_dt[i,j,k]).-
        (W_1_3[i,j,k]).*(dW_3_3_dt[i,j,k])).-
        ((W_3_4[i,j,k]).*(dW_1_4_dt[i,j,k]).-
        (W_1_4[i,j,k]).*(dW_3_4_dt[i,j,k]))).+
    # c charge from Higgs: 
        gp2 .*gw.*((ϕ_3[i,j,k]).*((dϕ_1_dt[i,j,k]) .-Dt_ϕ_1).-
        (ϕ_1[i,j,k]).*((dϕ_3_dt[i,j,k]) .-Dt_ϕ_3).+
        (ϕ_4[i,j,k]).*((dϕ_2_dt[i,j,k]) .-Dt_ϕ_2).-
        (ϕ_2[i,j,k]).*((dϕ_4_dt[i,j,k]) .-Dt_ϕ_4)))

    # s(Γ_3)=
    @inbounds k_Γ_3[i,j,k] =((1.0.-gp2).*(d2W_3_2_dx2 .+d2W_3_3_dy2 .+d2W_3_4_dz2).+
        gp2 .*gw.*(
        -((W_1_2[i,j,k]).*(dW_2_2_dt[i,j,k]).-
        (W_2_2[i,j,k]).*(dW_1_2_dt[i,j,k])).-
        ((W_1_3[i,j,k]).*(dW_2_3_dt[i,j,k]).-
        (W_2_3[i,j,k]).*(dW_1_3_dt[i,j,k])).-
        ((W_1_4[i,j,k]).*(dW_2_4_dt[i,j,k]).-
        (W_2_4[i,j,k]).*(dW_1_4_dt[i,j,k]))).+
    # c current from Higgs: 
        gp2 .*gw.*((ϕ_1[i,j,k]).*((dϕ_2_dt[i,j,k]) .-Dt_ϕ_2).-
        (ϕ_2[i,j,k]).*((dϕ_1_dt[i,j,k]) .-Dt_ϕ_1).+
        (ϕ_4[i,j,k]).*((dϕ_3_dt[i,j,k]) .-Dt_ϕ_3).-
        (ϕ_3[i,j,k]).*((dϕ_4_dt[i,j,k]) .-Dt_ϕ_4)))

    # s(Σ)=
    @inbounds k_Σ[i,j,k] =((1.0.-gp2).*(d2Y_2_dx2 .+d2Y_3_dy2 .+d2Y_4_dz2).+
    # c current from Higgs: 
        gp2 .*gy.*((ϕ_1[i,j,k])*((dϕ_2_dt[i,j,k]) .-Dt_ϕ_2).-
        (ϕ_2[i,j,k]).*((dϕ_1_dt[i,j,k]) .-Dt_ϕ_1).+
        (ϕ_3[i,j,k]).*((dϕ_4_dt[i,j,k]) .-Dt_ϕ_4).-
        (ϕ_4[i,j,k]).*((dϕ_3_dt[i,j,k]) .-Dt_ϕ_3)))

    return
end

# function compute_aux!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
#     W_1_1,W_1_2,W_1_3,W_1_4,
#     W_2_1,W_2_2,W_2_3,W_2_4,
#     W_3_1,W_3_2,W_3_3,W_3_4,
#     Y_1,Y_2,Y_3,Y_4,
#     Γ_1,Γ_2,Γ_3,Σ,
#     dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
#     dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
#     dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
#     dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
#     dY_2_dt,dY_3_dt,dY_4_dt,
#     KE_Phi,GE_Phi,PE_Phi,
#     ElectricE_W,MagneticE_W,ElectricE_Y,MagneticE_Y,
#     B_x,B_y,B_z,
#     gw,gy,gp2,vev,lambda,θ_w,dx)
#     i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
#               (blockIdx().y - 1) * blockDim().y + threadIdx().y,
#               (blockIdx().z - 1) * blockDim().z + threadIdx().z
    
#     #Spatial Derivatives
#     dϕ_1_dx=dfdx(ϕ_1,i,j,k,0.,dx)
#     dϕ_2_dx=dfdx(ϕ_2,i,j,k,0.,dx)
#     dϕ_3_dx=dfdx(ϕ_3,i,j,k,0.,dx)
#     dϕ_4_dx=dfdx(ϕ_4,i,j,k,0.,dx)
#     # @cuprintln(dϕ_4_dx)
#     dϕ_1_dy=dfdy(ϕ_1,i,j,k,0.,dx)
#     dϕ_2_dy=dfdy(ϕ_2,i,j,k,0.,dx)
#     dϕ_3_dy=dfdy(ϕ_3,i,j,k,0.,dx)
#     dϕ_4_dy=dfdy(ϕ_4,i,j,k,0.,dx)

#     dϕ_1_dz=dfdz(ϕ_1,i,j,k,0.,dx)
#     dϕ_2_dz=dfdz(ϕ_2,i,j,k,0.,dx)
#     dϕ_3_dz=dfdz(ϕ_3,i,j,k,0.,dx)
#     dϕ_4_dz=dfdz(ϕ_4,i,j,k,0.,dx)

#     dW_1_1_dx = dfdx(W_1_1,i,j,k,0.,dx)
#     dW_1_2_dx = dfdx(W_1_2,i,j,k,0.,dx)
#     dW_1_3_dx = dfdx(W_1_3,i,j,k,0.,dx)
#     dW_1_4_dx = dfdx(W_1_4,i,j,k,0.,dx)

#     dW_1_1_dy = dfdy(W_1_1,i,j,k,0.,dx)
#     dW_1_2_dy = dfdy(W_1_2,i,j,k,0.,dx)
#     dW_1_3_dy = dfdy(W_1_3,i,j,k,0.,dx)
#     dW_1_4_dy = dfdy(W_1_4,i,j,k,0.,dx)

#     dW_1_1_dz = dfdz(W_1_1,i,j,k,0.,dx)
#     dW_1_2_dz = dfdz(W_1_2,i,j,k,0.,dx)
#     dW_1_3_dz = dfdz(W_1_3,i,j,k,0.,dx)
#     dW_1_4_dz = dfdz(W_1_4,i,j,k,0.,dx)

#     dW_2_1_dx = dfdx(W_2_1,i,j,k,0.,dx)
#     dW_2_2_dx = dfdx(W_2_2,i,j,k,0.,dx)
#     dW_2_3_dx = dfdx(W_2_3,i,j,k,0.,dx)
#     dW_2_4_dx = dfdx(W_2_4,i,j,k,0.,dx)

#     dW_2_1_dy = dfdy(W_2_1,i,j,k,0.,dx)
#     dW_2_2_dy = dfdy(W_2_2,i,j,k,0.,dx)
#     dW_2_3_dy = dfdy(W_2_3,i,j,k,0.,dx)
#     dW_2_4_dy = dfdy(W_2_4,i,j,k,0.,dx)

#     dW_2_1_dz = dfdz(W_2_1,i,j,k,0.,dx)
#     dW_2_2_dz = dfdz(W_2_2,i,j,k,0.,dx)
#     dW_2_3_dz = dfdz(W_2_3,i,j,k,0.,dx)
#     dW_2_4_dz = dfdz(W_2_4,i,j,k,0.,dx)

#     dW_3_1_dx = dfdx(W_3_1,i,j,k,0.,dx)
#     dW_3_2_dx = dfdx(W_3_2,i,j,k,0.,dx)
#     dW_3_3_dx = dfdx(W_3_3,i,j,k,0.,dx)
#     dW_3_4_dx = dfdx(W_3_4,i,j,k,0.,dx)

#     dW_3_1_dy = dfdy(W_3_1,i,j,k,0.,dx)
#     dW_3_2_dy = dfdy(W_3_2,i,j,k,0.,dx)
#     dW_3_3_dy = dfdy(W_3_3,i,j,k,0.,dx)
#     dW_3_4_dy = dfdy(W_3_4,i,j,k,0.,dx)

#     dW_3_1_dz = dfdz(W_3_1,i,j,k,0.,dx)
#     dW_3_2_dz = dfdz(W_3_2,i,j,k,0.,dx)
#     dW_3_3_dz = dfdz(W_3_3,i,j,k,0.,dx)
#     dW_3_4_dz = dfdz(W_3_4,i,j,k,0.,dx)

#     dY_1_dx = dfdx(Y_1,i,j,k,0.,dx)
#     dY_2_dx = dfdx(Y_2,i,j,k,0.,dx)
#     dY_3_dx = dfdx(Y_3,i,j,k,0.,dx)
#     dY_4_dx = dfdx(Y_4,i,j,k,0.,dx)

#     dY_1_dy = dfdy(Y_1,i,j,k,0.,dx)
#     dY_2_dy = dfdy(Y_2,i,j,k,0.,dx)
#     dY_3_dy = dfdy(Y_3,i,j,k,0.,dx)
#     dY_4_dy = dfdy(Y_4,i,j,k,0.,dx)

#     dY_1_dz = dfdz(Y_1,i,j,k,0.,dx)
#     dY_2_dz = dfdz(Y_2,i,j,k,0.,dx)
#     dY_3_dz = dfdz(Y_3,i,j,k,0.,dx)
#     dY_4_dz = dfdz(Y_4,i,j,k,0.,dx)

#     dΓ_1_dx = dfdx(Γ_1,i,j,k,0.,dx)
#     dΓ_1_dy = dfdy(Γ_1,i,j,k,0.,dx)
#     dΓ_1_dz = dfdz(Γ_1,i,j,k,0.,dx)

#     dΓ_2_dx = dfdx(Γ_2,i,j,k,0.,dx)
#     dΓ_2_dy = dfdy(Γ_2,i,j,k,0.,dx)
#     dΓ_2_dz = dfdz(Γ_2,i,j,k,0.,dx)

#     dΓ_3_dx = dfdx(Γ_3,i,j,k,0.,dx)
#     dΓ_3_dy = dfdy(Γ_3,i,j,k,0.,dx)
#     dΓ_3_dz = dfdz(Γ_3,i,j,k,0.,dx)
    
#     dΣ_dx = dfdx(Σ,i,j,k,0.,dx)
#     dΣ_dy = dfdy(Σ,i,j,k,0.,dx)
#     dΣ_dz = dfdz(Σ,i,j,k,0.,dx)

#     d2ϕ_1_dx2=d2fdx2(ϕ_1,i,j,k,0.,dx)
#     d2ϕ_2_dx2=d2fdx2(ϕ_2,i,j,k,0.,dx)
#     d2ϕ_3_dx2=d2fdx2(ϕ_2,i,j,k,0.,dx)
#     d2ϕ_4_dx2=d2fdx2(ϕ_2,i,j,k,0.,dx)

#     d2ϕ_1_dy2=d2fdy2(ϕ_1,i,j,k,0.,dx)
#     d2ϕ_2_dy2=d2fdy2(ϕ_2,i,j,k,0.,dx)
#     d2ϕ_3_dy2=d2fdy2(ϕ_2,i,j,k,0.,dx)
#     d2ϕ_4_dy2=d2fdy2(ϕ_2,i,j,k,0.,dx)

#     d2ϕ_1_dz2=d2fdz2(ϕ_1,i,j,k,0.,dx)
#     d2ϕ_2_dz2=d2fdz2(ϕ_2,i,j,k,0.,dx)
#     d2ϕ_3_dz2=d2fdz2(ϕ_2,i,j,k,0.,dx)
#     d2ϕ_4_dz2=d2fdz2(ϕ_2,i,j,k,0.,dx)

#     d2W_1_1_dx2 = d2fdx2(W_1_1,i,j,k,0.,dx)
#     d2W_1_2_dx2 = d2fdx2(W_1_2,i,j,k,0.,dx)
#     d2W_1_3_dx2 = d2fdx2(W_1_3,i,j,k,0.,dx)
#     d2W_1_4_dx2 = d2fdx2(W_1_4,i,j,k,0.,dx)

#     d2W_1_1_dy2 = d2fdy2(W_1_1,i,j,k,0.,dx)
#     d2W_1_2_dy2 = d2fdy2(W_1_2,i,j,k,0.,dx)
#     d2W_1_3_dy2 = d2fdy2(W_1_3,i,j,k,0.,dx)
#     d2W_1_4_dy2 = d2fdy2(W_1_4,i,j,k,0.,dx)

#     d2W_1_1_dz2 = d2fdz2(W_1_1,i,j,k,0.,dx)
#     d2W_1_2_dz2 = d2fdz2(W_1_2,i,j,k,0.,dx)
#     d2W_1_3_dz2 = d2fdz2(W_1_3,i,j,k,0.,dx)
#     d2W_1_4_dz2 = d2fdz2(W_1_4,i,j,k,0.,dx)

#     d2W_2_1_dx2 = d2fdx2(W_2_1,i,j,k,0.,dx)
#     d2W_2_2_dx2 = d2fdx2(W_2_2,i,j,k,0.,dx)
#     d2W_2_3_dx2 = d2fdx2(W_2_3,i,j,k,0.,dx)
#     d2W_2_4_dx2 = d2fdx2(W_2_4,i,j,k,0.,dx)

#     d2W_2_1_dy2 = d2fdy2(W_2_1,i,j,k,0.,dx)
#     d2W_2_2_dy2 = d2fdy2(W_2_2,i,j,k,0.,dx)
#     d2W_2_3_dy2 = d2fdy2(W_2_3,i,j,k,0.,dx)
#     d2W_2_4_dy2 = d2fdy2(W_2_4,i,j,k,0.,dx)

#     d2W_2_1_dz2 = d2fdz2(W_2_1,i,j,k,0.,dx)
#     d2W_2_2_dz2 = d2fdz2(W_2_2,i,j,k,0.,dx)
#     d2W_2_3_dz2 = d2fdz2(W_2_3,i,j,k,0.,dx)
#     d2W_2_4_dz2 = d2fdz2(W_2_4,i,j,k,0.,dx)

#     d2W_3_1_dx2 = d2fdx2(W_3_1,i,j,k,0.,dx)
#     d2W_3_2_dx2 = d2fdx2(W_3_2,i,j,k,0.,dx)
#     d2W_3_3_dx2 = d2fdx2(W_3_3,i,j,k,0.,dx)
#     d2W_3_4_dx2 = d2fdx2(W_3_4,i,j,k,0.,dx)

#     d2W_3_1_dy2 = d2fdy2(W_3_1,i,j,k,0.,dx)
#     d2W_3_2_dy2 = d2fdy2(W_3_2,i,j,k,0.,dx)
#     d2W_3_3_dy2 = d2fdy2(W_3_3,i,j,k,0.,dx)
#     d2W_3_4_dy2 = d2fdy2(W_3_4,i,j,k,0.,dx)

#     d2W_3_1_dz2 = d2fdz2(W_3_1,i,j,k,0.,dx)
#     d2W_3_2_dz2 = d2fdz2(W_3_2,i,j,k,0.,dx)
#     d2W_3_3_dz2 = d2fdz2(W_3_3,i,j,k,0.,dx)
#     d2W_3_4_dz2 = d2fdz2(W_3_4,i,j,k,0.,dx)

#     d2Y_1_dx2 = d2fdx2(Y_1,i,j,k,0.,dx)
#     d2Y_2_dx2 = d2fdx2(Y_2,i,j,k,0.,dx)
#     d2Y_3_dx2 = d2fdx2(Y_3,i,j,k,0.,dx)
#     d2Y_4_dx2 = d2fdx2(Y_4,i,j,k,0.,dx)

#     d2Y_1_dy2 = d2fdy2(Y_1,i,j,k,0.,dx)
#     d2Y_2_dy2 = d2fdy2(Y_2,i,j,k,0.,dx)
#     d2Y_3_dy2 = d2fdy2(Y_3,i,j,k,0.,dx)
#     d2Y_4_dy2 = d2fdy2(Y_4,i,j,k,0.,dx)

#     d2Y_1_dz2 = d2fdz2(Y_1,i,j,k,0.,dx)
#     d2Y_2_dz2 = d2fdz2(Y_2,i,j,k,0.,dx)
#     d2Y_3_dz2 = d2fdz2(Y_3,i,j,k,0.,dx)
#     d2Y_4_dz2 = d2fdz2(Y_4,i,j,k,0.,dx)
    
#     ##Covariant Derivatives##

#     Dt_ϕ_1 =D_1ϕ_1(dϕ_1_dt[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
#     W_1_1[i,j,k],W_2_1[i,j,k],W_3_1[i,j,k],Y_1[i,j,k],gw,gy)
#     Dt_ϕ_2 =D_1ϕ_2(dϕ_2_dt[i,j,k],ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
#     W_1_1[i,j,k],W_2_1[i,j,k],W_3_1[i,j,k],Y_1[i,j,k],gw,gy)
#     Dt_ϕ_3 =D_1ϕ_3(dϕ_3_dt[i,j,k],ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
#     W_1_1[i,j,k],W_2_1[i,j,k],W_3_1[i,j,k],Y_1[i,j,k],gw,gy)
#     Dt_ϕ_4 =D_1ϕ_4(dϕ_4_dt[i,j,k],ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
#     W_1_1[i,j,k],W_2_1[i,j,k],W_3_1[i,j,k],Y_1[i,j,k],gw,gy)
#     Dx_ϕ_1 =D_2ϕ_1(dϕ_1_dx,ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
#     W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy)
#     Dx_ϕ_2 =D_2ϕ_2(dϕ_2_dx,ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
#     W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy)
#     Dx_ϕ_3 =D_2ϕ_3(dϕ_3_dx,ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
#     W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy)
#     Dx_ϕ_4 =D_2ϕ_4(dϕ_4_dx,ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
#     W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy)
#     Dy_ϕ_1 =D_3ϕ_1(dϕ_1_dy,ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
#     W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy)
#     Dy_ϕ_2 =D_3ϕ_2(dϕ_2_dy,ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
#     W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy)
#     Dy_ϕ_3 =D_3ϕ_3(dϕ_3_dy,ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
#     W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy)
#     Dy_ϕ_4 =D_3ϕ_4(dϕ_4_dy,ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
#     W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy)
#     Dz_ϕ_1 =D_4ϕ_1(dϕ_1_dz,ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
#     W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)
#     Dz_ϕ_2 =D_4ϕ_2(dϕ_2_dz,ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
#     W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)
#     Dz_ϕ_3 =D_4ϕ_3(dϕ_3_dz,ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
#     W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)
#     Dz_ϕ_4 =D_4ϕ_4(dϕ_4_dz,ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
#     W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)

#     # Field Strengths #
#     # W_1_11()
#     # W_1_12(dW_1_2_dt)
#     # W_1_13(dW_1_3_dt)
#     # W_1_14(dW_1_4_dt)
#     # W_1_22(dW_2_2_dt)
#     W_1_23 =W_1_xy(dW_1_2_dx,dW_1_1_dy,W_2_2,W_3_3,W_2_3,W_3_2,gw,i,j,k)
#     W_1_24=W_1_xz(dW_1_4_dx,dW_1_2_dz,W_2_2,W_3_4,W_2_4,W_3_2,gw,i,j,k)
#     # W_1_33(dW_3_3_dt)
#     W_1_34=W_1_yz(dW_1_4_dy,dW_1_3_dz,W_2_3,W_3_4,W_2_4,W_3_3,gw,i,j,k)
#     # W_1_44()
#     # W_2_11()
#     # W_2_12(dW_2_2_dt)
#     # W_2_13(dW_2_3_dt)
#     # W_2_14(dW_2_4_dt)
#     # W_2_22()
#     W_2_23=W_2_xy(dW_2_3_dx,dW_2_2_dy,W_3_2,W_1_3,W_3_3,W_1_2,gw,i,j,k)
#     W_2_24=W_2_xz(dW_2_4_dx,dW_2_2_dz,W_3_2,W_1_4,W_3_4,W_1_2,gw,i,j,k)
#     # W_2_33()
#     W_2_34=W_2_yz(dW_2_4_dy,dW_2_3_dz,W_3_3,W_1_4,W_3_4,W_1_3,gw,i,j,k)
#     # W_2_44()
#     # W_3_11()
#     # W_3_12(dW_3_2_dt)
#     # W_3_13(dW_3_3_dt)
#     # W_3_14(dW_3_4_dt)
#     # W_3_22()
#     W_3_23=W_3_xy(dW_3_3_dx,dW_3_2_dy,W_1_2,W_2_3,W_1_3,W_2_2,gw,i,j,k)
#     W_3_24=W_3_xz(dW_3_4_dx,dW_3_2_dz,W_1_2,W_2_4,W_1_4,W_2_2,gw,i,j,k)
#     # W_3_33(dW_3_3_dt)
#     W_3_34=W_3_yz(dW_3_4_dy,dW_3_3_dz,W_1_3,W_2_4,W_1_4,W_2_3,gw,i,j,k)
#     # W_3_44()
#     # Y_1_1()
#     # Y_1_2(dY_2_dt)
#     # Y_1_3(dY_3_dt)
#     # Y_1_4(dY_4_dt)
#     # Y_2_2()
#     Y_2_3=Y_2_y(dY_3_dx,dY_2_dy)
#     Y_2_4=Y_2_z(dY_4_dx,dY_2_dz)
#     # Y_3_3()
#     Y_3_4=Y_3_z(dY_4_dy,dY_3_dz)
#     # Y_4_4()
    
#     # Energy expressions
#     @inbounds PE_Phi[i,j,k] = (lambda*((ϕ_1[i,j,k])^2+
#     (ϕ_2[i,j,k])^2+(ϕ_3[i,j,k])^2+(ϕ_4[i,j,k])^2-vev^2)^2)
    
#     @inbounds GE_Phi[i,j,k] = (Dx_ϕ_1^2+Dy_ϕ_1^2+Dz_ϕ_1^2+
#     Dx_ϕ_2^2+Dy_ϕ_2^2+Dz_ϕ_2^2+
#     Dx_ϕ_3^2+Dy_ϕ_3^2+Dz_ϕ_3^2+
#     Dx_ϕ_4^2+Dy_ϕ_4^2+Dz_ϕ_4^2)

#     @inbounds KE_Phi[i,j,k] = (Dt_ϕ_1^2+Dt_ϕ_2^2+Dt_ϕ_3^2+Dt_ϕ_4^2)

#     @inbounds ElectricE_W[i,j,k] =(0.5*
#     ((dW_1_2_dt[i,j,k])^2+(dW_1_3_dt[i,j,k])^2+(dW_1_4_dt[i,j,k])^2+
#     (dW_2_2_dt[i,j,k])^2+(dW_2_3_dt[i,j,k])^2+(dW_2_4_dt[i,j,k])^2+
#     (dW_3_2_dt[i,j,k])^2+(dW_3_3_dt[i,j,k])^2+(dW_3_4_dt[i,j,k])^2))

#     @inbounds MagneticE_W[i,j,k] = (0.5*
#     (W_1_23^2+W_1_24^2+W_1_34^2+
#     W_2_23^2+W_2_24^2+W_2_34^2+
#     W_3_23^2+W_3_24^2+W_3_34^2))

#     @inbounds ElectricE_Y[i,j,k] = (0.5*
#     ((dY_2_dt[i,j,k])^2+(dY_3_dt[i,j,k])^2+(dY_4_dt[i,j,k])^2))

#     @inbounds MagneticE_Y[i,j,k] = (0.5*(Y_2_3^2+Y_2_4^2+Y_3_4^2))

#     # Higgs n definitions
#     n_1=-2. *(ϕ_1[i,j,k]*ϕ_3[i,j,k]+ϕ_2[i,j,k]*ϕ_4[i,j,k])/(ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2)
#     n_2=+2. *(ϕ_2[i,j,k]*ϕ_3[i,j,k]-ϕ_1[i,j,k]*ϕ_4[i,j,k])/(ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2)
#     n_3=(-ϕ_1[i,j,k]^2-ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2)/(ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2)

#     # Magnetic field defintions
#     # $A_{ij} = stw*na*W^a_{ij}+ctw*Y_{ij}
#     #      -i*(2*stw/(gw*vev^2))*((D_i\Phi)^\dag D_j\Phi-(D_j\Phi)^\dag D_i\Phi)
#     # and,            
#     # B_x= -A_{yx}, B_y= -A_{zx}, B_z= -A_{xy} 

#     @inbounds B_x[i,j,k] = -(sin(θ_w)*(n_1*W_1_34+n_2*W_1_34+n_3*W_3_34)+cos(θ_w)*Y_3_4
#     +(4. *sin(θ_w)/(gw*vev^2))*(Dy_ϕ_1*Dz_ϕ_2-Dy_ϕ_2*Dz_ϕ_1
#     +Dy_ϕ_3*Dz_ϕ_4-Dy_ϕ_4*Dz_ϕ_3))

#     @inbounds B_y[i,j,k] = -(sin(θ_w)*(n_1*(-W_1_24)+n_2*(-W_2_24)+n_3*(-W_3_24))+cos(θ_w)*(-Y_2_4)
#     +(4. *sin(θ_w)/(gw*vev^2))*(Dz_ϕ_1*Dx_ϕ_2-Dz_ϕ_2*Dx_ϕ_1
#     +Dz_ϕ_3*Dx_ϕ_4-Dz_ϕ_4*Dx_ϕ_3))

#     @inbounds B_z[i,j,k] = -(sin(θ_w)*(n_1*(W_1_23)+n_2*W_2_23+n_3*W_3_23)+cos(θ_w)*Y_2_3
#     +(4. *sin(θ_w)/(gw*vev^2))*(Dx_ϕ_1*Dy_ϕ_2-Dx_ϕ_2*Dy_ϕ_1
#     +Dx_ϕ_3*Dy_ϕ_4-Dx_ϕ_4*Dy_ϕ_3))

#     return
# end

@views function compute_aux!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
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
    KE_Phi,GE_Phi,PE_Phi,
    ElectricE_W,MagneticE_W,ElectricE_Y,MagneticE_Y,
    B_x,B_y,B_z,
    gw,gy,gp2,vev,lambda,θ_w,dx)
    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
              (blockIdx().y - 1) * blockDim().y + threadIdx().y,
              (blockIdx().z - 1) * blockDim().z + threadIdx().z
    
    #Spatial Derivatives
    dϕ_1_dx=dfdx(ϕ_1,i,j,k,0.,dx)
    dϕ_2_dx=dfdx(ϕ_2,i,j,k,0.,dx)
    dϕ_3_dx=dfdx(ϕ_3,i,j,k,0.,dx)
    dϕ_4_dx=dfdx(ϕ_4,i,j,k,0.,dx)
    # @cuprintln(dϕ_4_dx)
    dϕ_1_dy=dfdy(ϕ_1,i,j,k,0.,dx)
    dϕ_2_dy=dfdy(ϕ_2,i,j,k,0.,dx)
    dϕ_3_dy=dfdy(ϕ_3,i,j,k,0.,dx)
    dϕ_4_dy=dfdy(ϕ_4,i,j,k,0.,dx)

    dϕ_1_dz=dfdz(ϕ_1,i,j,k,0.,dx)
    dϕ_2_dz=dfdz(ϕ_2,i,j,k,0.,dx)
    dϕ_3_dz=dfdz(ϕ_3,i,j,k,0.,dx)
    dϕ_4_dz=dfdz(ϕ_4,i,j,k,0.,dx)

    # dW_1_1_dx = dfdx(W_1_1,i,j,k,0.,dx)
    dW_1_2_dx = dfdx(W_1_2,i,j,k,0.,dx)
    dW_1_3_dx = dfdx(W_1_3,i,j,k,0.,dx)
    dW_1_4_dx = dfdx(W_1_4,i,j,k,0.,dx)

    # dW_1_1_dy = dfdy(W_1_1,i,j,k,0.,dx)
    dW_1_2_dy = dfdy(W_1_2,i,j,k,0.,dx)
    dW_1_3_dy = dfdy(W_1_3,i,j,k,0.,dx)
    dW_1_4_dy = dfdy(W_1_4,i,j,k,0.,dx)

    # dW_1_1_dz = dfdz(W_1_1,i,j,k,0.,dx)
    dW_1_2_dz = dfdz(W_1_2,i,j,k,0.,dx)
    dW_1_3_dz = dfdz(W_1_3,i,j,k,0.,dx)
    dW_1_4_dz = dfdz(W_1_4,i,j,k,0.,dx)

    # dW_2_1_dx = dfdx(W_2_1,i,j,k,0.,dx)
    dW_2_2_dx = dfdx(W_2_2,i,j,k,0.,dx)
    dW_2_3_dx = dfdx(W_2_3,i,j,k,0.,dx)
    dW_2_4_dx = dfdx(W_2_4,i,j,k,0.,dx)

    # dW_2_1_dy = dfdy(W_2_1,i,j,k,0.,dx)
    dW_2_2_dy = dfdy(W_2_2,i,j,k,0.,dx)
    dW_2_3_dy = dfdy(W_2_3,i,j,k,0.,dx)
    dW_2_4_dy = dfdy(W_2_4,i,j,k,0.,dx)

    # dW_2_1_dz = dfdz(W_2_1,i,j,k,0.,dx)
    dW_2_2_dz = dfdz(W_2_2,i,j,k,0.,dx)
    dW_2_3_dz = dfdz(W_2_3,i,j,k,0.,dx)
    dW_2_4_dz = dfdz(W_2_4,i,j,k,0.,dx)

    # dW_3_1_dx = dfdx(W_3_1,i,j,k,0.,dx)
    dW_3_2_dx = dfdx(W_3_2,i,j,k,0.,dx)
    dW_3_3_dx = dfdx(W_3_3,i,j,k,0.,dx)
    dW_3_4_dx = dfdx(W_3_4,i,j,k,0.,dx)

    # dW_3_1_dy = dfdy(W_3_1,i,j,k,0.,dx)
    dW_3_2_dy = dfdy(W_3_2,i,j,k,0.,dx)
    dW_3_3_dy = dfdy(W_3_3,i,j,k,0.,dx)
    dW_3_4_dy = dfdy(W_3_4,i,j,k,0.,dx)

    # dW_3_1_dz = dfdz(W_3_1,i,j,k,0.,dx)
    dW_3_2_dz = dfdz(W_3_2,i,j,k,0.,dx)
    dW_3_3_dz = dfdz(W_3_3,i,j,k,0.,dx)
    dW_3_4_dz = dfdz(W_3_4,i,j,k,0.,dx)

    # dY_1_dx = dfdx(Y_1,i,j,k,0.,dx)
    dY_2_dx = dfdx(Y_2,i,j,k,0.,dx)
    dY_3_dx = dfdx(Y_3,i,j,k,0.,dx)
    dY_4_dx = dfdx(Y_4,i,j,k,0.,dx)

    # dY_1_dy = dfdy(Y_1,i,j,k,0.,dx)
    dY_2_dy = dfdy(Y_2,i,j,k,0.,dx)
    dY_3_dy = dfdy(Y_3,i,j,k,0.,dx)
    dY_4_dy = dfdy(Y_4,i,j,k,0.,dx)

    # dY_1_dz = dfdz(Y_1,i,j,k,0.,dx)
    dY_2_dz = dfdz(Y_2,i,j,k,0.,dx)
    dY_3_dz = dfdz(Y_3,i,j,k,0.,dx)
    dY_4_dz = dfdz(Y_4,i,j,k,0.,dx)

    dΓ_1_dx = dfdx(Γ_1,i,j,k,0.,dx)
    dΓ_1_dy = dfdy(Γ_1,i,j,k,0.,dx)
    dΓ_1_dz = dfdz(Γ_1,i,j,k,0.,dx)

    dΓ_2_dx = dfdx(Γ_2,i,j,k,0.,dx)
    dΓ_2_dy = dfdy(Γ_2,i,j,k,0.,dx)
    dΓ_2_dz = dfdz(Γ_2,i,j,k,0.,dx)

    dΓ_3_dx = dfdx(Γ_3,i,j,k,0.,dx)
    dΓ_3_dy = dfdy(Γ_3,i,j,k,0.,dx)
    dΓ_3_dz = dfdz(Γ_3,i,j,k,0.,dx)
    
    dΣ_dx = dfdx(Σ,i,j,k,0.,dx)
    dΣ_dy = dfdy(Σ,i,j,k,0.,dx)
    dΣ_dz = dfdz(Σ,i,j,k,0.,dx)

    d2ϕ_1_dx2=d2fdx2(ϕ_1,i,j,k,0.,dx)
    d2ϕ_2_dx2=d2fdx2(ϕ_2,i,j,k,0.,dx)
    d2ϕ_3_dx2=d2fdx2(ϕ_2,i,j,k,0.,dx)
    d2ϕ_4_dx2=d2fdx2(ϕ_2,i,j,k,0.,dx)

    d2ϕ_1_dy2=d2fdy2(ϕ_1,i,j,k,0.,dx)
    d2ϕ_2_dy2=d2fdy2(ϕ_2,i,j,k,0.,dx)
    d2ϕ_3_dy2=d2fdy2(ϕ_2,i,j,k,0.,dx)
    d2ϕ_4_dy2=d2fdy2(ϕ_2,i,j,k,0.,dx)

    d2ϕ_1_dz2=d2fdz2(ϕ_1,i,j,k,0.,dx)
    d2ϕ_2_dz2=d2fdz2(ϕ_2,i,j,k,0.,dx)
    d2ϕ_3_dz2=d2fdz2(ϕ_2,i,j,k,0.,dx)
    d2ϕ_4_dz2=d2fdz2(ϕ_2,i,j,k,0.,dx)

    # d2W_1_1_dx2 = d2fdx2(W_1_1,i,j,k,0.,dx)
    d2W_1_2_dx2 = d2fdx2(W_1_2,i,j,k,0.,dx)
    d2W_1_3_dx2 = d2fdx2(W_1_3,i,j,k,0.,dx)
    d2W_1_4_dx2 = d2fdx2(W_1_4,i,j,k,0.,dx)

    # d2W_1_1_dy2 = d2fdy2(W_1_1,i,j,k,0.,dx)
    d2W_1_2_dy2 = d2fdy2(W_1_2,i,j,k,0.,dx)
    d2W_1_3_dy2 = d2fdy2(W_1_3,i,j,k,0.,dx)
    d2W_1_4_dy2 = d2fdy2(W_1_4,i,j,k,0.,dx)

    # d2W_1_1_dz2 = d2fdz2(W_1_1,i,j,k,0.,dx)
    d2W_1_2_dz2 = d2fdz2(W_1_2,i,j,k,0.,dx)
    d2W_1_3_dz2 = d2fdz2(W_1_3,i,j,k,0.,dx)
    d2W_1_4_dz2 = d2fdz2(W_1_4,i,j,k,0.,dx)

    # d2W_2_1_dx2 = d2fdx2(W_2_1,i,j,k,0.,dx)
    d2W_2_2_dx2 = d2fdx2(W_2_2,i,j,k,0.,dx)
    d2W_2_3_dx2 = d2fdx2(W_2_3,i,j,k,0.,dx)
    d2W_2_4_dx2 = d2fdx2(W_2_4,i,j,k,0.,dx)

    # d2W_2_1_dy2 = d2fdy2(W_2_1,i,j,k,0.,dx)
    d2W_2_2_dy2 = d2fdy2(W_2_2,i,j,k,0.,dx)
    d2W_2_3_dy2 = d2fdy2(W_2_3,i,j,k,0.,dx)
    d2W_2_4_dy2 = d2fdy2(W_2_4,i,j,k,0.,dx)

    # d2W_2_1_dz2 = d2fdz2(W_2_1,i,j,k,0.,dx)
    d2W_2_2_dz2 = d2fdz2(W_2_2,i,j,k,0.,dx)
    d2W_2_3_dz2 = d2fdz2(W_2_3,i,j,k,0.,dx)
    d2W_2_4_dz2 = d2fdz2(W_2_4,i,j,k,0.,dx)

    # d2W_3_1_dx2 = d2fdx2(W_3_1,i,j,k,0.,dx)
    d2W_3_2_dx2 = d2fdx2(W_3_2,i,j,k,0.,dx)
    d2W_3_3_dx2 = d2fdx2(W_3_3,i,j,k,0.,dx)
    d2W_3_4_dx2 = d2fdx2(W_3_4,i,j,k,0.,dx)

    # d2W_3_1_dy2 = d2fdy2(W_3_1,i,j,k,0.,dx)
    d2W_3_2_dy2 = d2fdy2(W_3_2,i,j,k,0.,dx)
    d2W_3_3_dy2 = d2fdy2(W_3_3,i,j,k,0.,dx)
    d2W_3_4_dy2 = d2fdy2(W_3_4,i,j,k,0.,dx)

    # d2W_3_1_dz2 = d2fdz2(W_3_1,i,j,k,0.,dx)
    d2W_3_2_dz2 = d2fdz2(W_3_2,i,j,k,0.,dx)
    d2W_3_3_dz2 = d2fdz2(W_3_3,i,j,k,0.,dx)
    d2W_3_4_dz2 = d2fdz2(W_3_4,i,j,k,0.,dx)

    # d2Y_1_dx2 = d2fdx2(Y_1,i,j,k,0.,dx)
    d2Y_2_dx2 = d2fdx2(Y_2,i,j,k,0.,dx)
    d2Y_3_dx2 = d2fdx2(Y_3,i,j,k,0.,dx)
    d2Y_4_dx2 = d2fdx2(Y_4,i,j,k,0.,dx)

    # d2Y_1_dy2 = d2fdy2(Y_1,i,j,k,0.,dx)
    d2Y_2_dy2 = d2fdy2(Y_2,i,j,k,0.,dx)
    d2Y_3_dy2 = d2fdy2(Y_3,i,j,k,0.,dx)
    d2Y_4_dy2 = d2fdy2(Y_4,i,j,k,0.,dx)

    # d2Y_1_dz2 = d2fdz2(Y_1,i,j,k,0.,dx)
    d2Y_2_dz2 = d2fdz2(Y_2,i,j,k,0.,dx)
    d2Y_3_dz2 = d2fdz2(Y_3,i,j,k,0.,dx)
    d2Y_4_dz2 = d2fdz2(Y_4,i,j,k,0.,dx)

    ##Covariant Derivatives##Temporal gauge

    Dt_ϕ_1 =dϕ_1_dt[i,j,k]
    Dt_ϕ_2 =dϕ_2_dt[i,j,k]
    Dt_ϕ_3 =dϕ_3_dt[i,j,k]
    Dt_ϕ_4 =dϕ_4_dt[i,j,k]
    Dx_ϕ_1 =D_2ϕ_1(dϕ_1_dx,ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
    W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy)
    Dx_ϕ_2 =D_2ϕ_2(dϕ_2_dx,ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
    W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy)
    Dx_ϕ_3 =D_2ϕ_3(dϕ_3_dx,ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
    W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy)
    Dx_ϕ_4 =D_2ϕ_4(dϕ_4_dx,ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
    W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy)
    Dy_ϕ_1 =D_3ϕ_1(dϕ_1_dy,ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
    W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy)
    Dy_ϕ_2 =D_3ϕ_2(dϕ_2_dy,ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
    W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy)
    Dy_ϕ_3 =D_3ϕ_3(dϕ_3_dy,ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
    W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy)
    Dy_ϕ_4 =D_3ϕ_4(dϕ_4_dy,ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
    W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy)
    Dz_ϕ_1 =D_4ϕ_1(dϕ_1_dz,ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
    W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)
    Dz_ϕ_2 =D_4ϕ_2(dϕ_2_dz,ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
    W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)
    Dz_ϕ_3 =D_4ϕ_3(dϕ_3_dz,ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
    W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)
    Dz_ϕ_4 =D_4ϕ_4(dϕ_4_dz,ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
    W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)
    
    # Field Strengths #
    # W_1_11()
    # W_1_12(dW_1_2_dt)
    # W_1_13(dW_1_3_dt)
    # W_1_14(dW_1_4_dt)
    # W_1_22(dW_2_2_dt)
    W_1_23 =W_1_xy(dW_1_3_dx,dW_1_2_dy,W_2_2,W_3_3,W_2_3,W_3_2,gw,i,j,k)
    W_1_24=W_1_xz(dW_1_4_dx,dW_1_2_dz,W_2_2,W_3_4,W_2_4,W_3_2,gw,i,j,k)
    # W_1_33(dW_3_3_dt)
    W_1_34=W_1_yz(dW_1_4_dy,dW_1_3_dz,W_2_3,W_3_4,W_2_4,W_3_3,gw,i,j,k)
    # W_1_44()
    # W_2_11()
    # W_2_12(dW_2_2_dt)
    # W_2_13(dW_2_3_dt)
    # W_2_14(dW_2_4_dt)
    # W_2_22()
    W_2_23=W_2_xy(dW_2_3_dx,dW_2_2_dy,W_3_2,W_1_3,W_3_3,W_1_2,gw,i,j,k)
    W_2_24=W_2_xz(dW_2_4_dx,dW_2_2_dz,W_3_2,W_1_4,W_3_4,W_1_2,gw,i,j,k)
    # W_2_33()
    W_2_34=W_2_yz(dW_2_4_dy,dW_2_3_dz,W_3_3,W_1_4,W_3_4,W_1_3,gw,i,j,k)
    # W_2_44()
    # W_3_11()
    # W_3_12(dW_3_2_dt)
    # W_3_13(dW_3_3_dt)
    # W_3_14(dW_3_4_dt)
    # W_3_22()
    W_3_23=W_3_xy(dW_3_3_dx,dW_3_2_dy,W_1_2,W_2_3,W_1_3,W_2_2,gw,i,j,k)
    W_3_24=W_3_xz(dW_3_4_dx,dW_3_2_dz,W_1_2,W_2_4,W_1_4,W_2_2,gw,i,j,k)
    # W_3_33(dW_3_3_dt)
    W_3_34=W_3_yz(dW_3_4_dy,dW_3_3_dz,W_1_3,W_2_4,W_1_4,W_2_3,gw,i,j,k)
    # W_3_44()
    # Y_1_1()
    # Y_1_2(dY_2_dt)
    # Y_1_3(dY_3_dt)
    # Y_1_4(dY_4_dt)
    # Y_2_2()
    Y_2_3=Y_2_y(dY_3_dx,dY_2_dy)
    Y_2_4=Y_2_z(dY_4_dx,dY_2_dz)
    # Y_3_3()
    Y_3_4=Y_3_z(dY_4_dy,dY_3_dz)
    # Y_4_4()

    # Energy expressions
    @inbounds PE_Phi[i,j,k] = (lambda*((ϕ_1[i,j,k])^2+
    (ϕ_2[i,j,k])^2+(ϕ_3[i,j,k])^2+(ϕ_4[i,j,k])^2-vev^2)^2)
    
    @inbounds GE_Phi[i,j,k] = (Dx_ϕ_1^2+Dy_ϕ_1^2+Dz_ϕ_1^2+
    Dx_ϕ_2^2+Dy_ϕ_2^2+Dz_ϕ_2^2+
    Dx_ϕ_3^2+Dy_ϕ_3^2+Dz_ϕ_3^2+
    Dx_ϕ_4^2+Dy_ϕ_4^2+Dz_ϕ_4^2)

    @inbounds KE_Phi[i,j,k] = (Dt_ϕ_1^2+Dt_ϕ_2^2+Dt_ϕ_3^2+Dt_ϕ_4^2)

    @inbounds ElectricE_W[i,j,k] =(0.5*
    ((dW_1_2_dt[i,j,k])^2+(dW_1_3_dt[i,j,k])^2+(dW_1_4_dt[i,j,k])^2+
    (dW_2_2_dt[i,j,k])^2+(dW_2_3_dt[i,j,k])^2+(dW_2_4_dt[i,j,k])^2+
    (dW_3_2_dt[i,j,k])^2+(dW_3_3_dt[i,j,k])^2+(dW_3_4_dt[i,j,k])^2))

    @inbounds MagneticE_W[i,j,k] = (0.5*
    (W_1_23^2+W_1_24^2+W_1_34^2+
    W_2_23^2+W_2_24^2+W_2_34^2+
    W_3_23^2+W_3_24^2+W_3_34^2))

    @inbounds ElectricE_Y[i,j,k] = (0.5*
    ((dY_2_dt[i,j,k])^2+(dY_3_dt[i,j,k])^2+(dY_4_dt[i,j,k])^2))

    @inbounds MagneticE_Y[i,j,k] = (0.5*(Y_2_3^2+Y_2_4^2+Y_3_4^2))

    # Higgs n definitions
    n_1=-2. *(ϕ_1[i,j,k]*ϕ_3[i,j,k]+ϕ_2[i,j,k]*ϕ_4[i,j,k])/(ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2)
    n_2=+2. *(ϕ_2[i,j,k]*ϕ_3[i,j,k]-ϕ_1[i,j,k]*ϕ_4[i,j,k])/(ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2)
    n_3=(-ϕ_1[i,j,k]^2-ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2)/(ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2)

    # Magnetic field defintions
    # $A_{ij} = stw*na*W^a_{ij}+ctw*Y_{ij}
    #      -i*(2*stw/(gw*vev^2))*((D_i\Phi)^\dag D_j\Phi-(D_j\Phi)^\dag D_i\Phi)
    # and,            
    # B_x= -A_{yx}, B_y= -A_{zx}, B_z= -A_{xy} 

    @inbounds B_x[i,j,k] = -(sin(θ_w)*(n_1*W_1_34+n_2*W_1_34+n_3*W_3_34)+cos(θ_w)*Y_3_4
    +(4. *sin(θ_w)/(gw*vev^2))*(Dy_ϕ_1*Dz_ϕ_2-Dy_ϕ_2*Dz_ϕ_1
    +Dy_ϕ_3*Dz_ϕ_4-Dy_ϕ_4*Dz_ϕ_3))

    @inbounds B_y[i,j,k] = -(sin(θ_w)*(n_1*(-W_1_24)+n_2*(-W_2_24)+n_3*(-W_3_24))+cos(θ_w)*(-Y_2_4)
    +(4. *sin(θ_w)/(gw*vev^2))*(Dz_ϕ_1*Dx_ϕ_2-Dz_ϕ_2*Dx_ϕ_1
    +Dz_ϕ_3*Dx_ϕ_4-Dz_ϕ_4*Dx_ϕ_3))

    @inbounds B_z[i,j,k] = -(sin(θ_w)*(n_1*(W_1_23)+n_2*W_2_23+n_3*W_3_23)+cos(θ_w)*Y_2_3
    +(4. *sin(θ_w)/(gw*vev^2))*(Dx_ϕ_1*Dy_ϕ_2-Dx_ϕ_2*Dy_ϕ_1
    +Dx_ϕ_3*Dy_ϕ_4-Dx_ϕ_4*Dy_ϕ_3))

    return
end

# function initializer!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,ib,jb,kb,p1,p2,p3,p4,rkx,rky,rkz,dx,mH)
#     i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
#     (blockIdx().y - 1) * blockDim().y + threadIdx().y,
#     (blockIdx().z - 1) * blockDim().z + threadIdx().z
#     # @cuprintln(i," ",j," ",k)
#     rb = sqrt((1.0/(rkx^2))*sin(rkx*((i-1)-(ib-0.5))*dx)^2+(1.0/(rky^2))*sin(rky*((j-1)-(jb-0.5))*dx)^2+(1.0/(rkz^2))*sin(rkz*((k-1)-(kb-0.5))*dx)^2)
#     rmag = (1.0+(sqrt(2.0)-1.0)^2)*exp(-mH*rb/sqrt(2.0))/(1.0+((sqrt(2.0)-1.0)^2)*exp(-sqrt(2.0)*mH*rb))
#     @inbounds ϕ_1[i,j,k]=ϕ_1[i,j,k]+rmag*p1
#     @inbounds ϕ_2[i,j,k]=ϕ_2[i,j,k]+rmag*p2
#     @inbounds ϕ_3[i,j,k]=ϕ_3[i,j,k]+rmag*p3
#     @inbounds ϕ_4[i,j,k]=ϕ_4[i,j,k]+rmag*p4
#     # if j==16 && k==16
#     #     @cuprintln(ϕ_1[i,j,k],rmag,p1)
#     # end

#     return
# end

function x_g(i,Nx,grid_x,dx)
    return (i-4)*dx+(grid_x+1)*(Nx-6)*dx
end

function y_g(j,Ny,grid_y,dx)
    return (j-4)*dx+(grid_y+1)*(Ny-6)*dx
end

function z_g(k,Nz,grid_z,dx)
    return (k-4)*dx+(grid_z+1)*(Nz-6)*dx
end

@views function initializer!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,ib,jb,kb,p1,p2,p3,p4,rkx,rky,rkz,dx,mH,Nx,Ny,Nz,grid_x,grid_y,grid_z)
    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
    (blockIdx().y - 1) * blockDim().y + threadIdx().y,
    (blockIdx().z - 1) * blockDim().z + threadIdx().z
    # cs = Cint[0,0,0]
    # MPI.Cart_coords!(comm,p,cs)
    # @cuprintln(i," ",j," ",k)
    rb = sqrt((1.0/(rkx^2))*sin(rkx*(x_g(i,Nx,grid_x,dx)-(ib-0.5)*dx))^2+
    (1.0/(rky^2))*sin(rky*(y_g(j,Ny,grid_y,dx)-(jb-0.5)*dx))^2+
    (1.0/(rkz^2))*sin(rkz*(z_g(k,Nz,grid_z,dx)-(kb-0.5)*dx))^2)
    rmag = (1.0+(sqrt(2.0)-1.0)^2)*exp(-mH*rb/sqrt(2.0))/(1.0+((sqrt(2.0)-1.0)^2)*exp(-sqrt(2.0)*mH*rb))
    @inbounds ϕ_1[i,j,k]=ϕ_1[i,j,k]+rmag*p1
    @inbounds ϕ_2[i,j,k]=ϕ_2[i,j,k]+rmag*p2
    @inbounds ϕ_3[i,j,k]=ϕ_3[i,j,k]+rmag*p3
    @inbounds ϕ_4[i,j,k]=ϕ_4[i,j,k]+rmag*p4
    # if j==16 && k==16
    #     @cuprintln(ϕ_1[i,j,k],rmag,p1)
    # end

    return
end

@views function update_halo!(A,neighbors_x,neighbors_y,neighbors_z,comm)
	#x direction
	#Left
    # println(neighbors_x,neighbors_y,neighbors_z)
	if neighbors_x[1] != -1
        # buf_size = 3*(size(A,2)*size(A,3))
        # sendbuf = reshape(Array(A[4:6 ,:,:]),buf_size)
		sendbuf = Array(A[4:6 ,:,:])
        # println(size(sendbuf))
        # recvbuf = zeros(buf_size)
		recvbuf = zeros(size(sendbuf))
        # println(size(recvbuf))
		# r=MPI.Isend(sendbuf, neighbors_x[1], 0, comm)
		# MPI.Wait!(r)
		# r=MPI.Irecv!(recvbuf, neighbors_x[1], 1, comm)
		# MPI.Wait!(r)
		MPI.Sendrecv!(sendbuf,neighbors_x[1],0,
					  recvbuf,neighbors_x[1],1, comm)
        # copyto!(A[1:3 ,:,:],reshape(recvbuf,size(A[1:3 ,:,:])))
		copyto!(A[1:3 ,:,:],recvbuf)

        # sendbuf = zeros(size(A[4:6 ,:,:]))
        # recvbuf = zeros(size(A[1:3 ,:,:]))
        # copyto!(sendbuf,A[4:6 ,:,:])
		# MPI.Isend(sendbuf, neighbors_x[1], 0, comm)
		# MPI.Irecv!(recvbuf, neighbors_x[1], 1, comm)
        # copyto!(A[1:3 ,:,:],recvbuf)

	end
	#Right
	if neighbors_x[2] != -1
        # buf_size = 3*(size(A,2)*size(A,3))
        # sendbuf = reshape(Array(A[end-5:end-3 ,:,:]),buf_size)
		sendbuf = Array(A[end-5:end-3 ,:,:])
        # recvbuf = zeros(buf_size)
		recvbuf = zeros(size(sendbuf))
		# r=MPI.Isend(sendbuf, neighbors_x[2], 1, comm)
		# MPI.Wait!(r)
		# r=MPI.Irecv!(recvbuf, neighbors_x[2], 0, comm)
		# MPI.Wait!(r)
		MPI.Sendrecv!(sendbuf,neighbors_x[2],1,
		recvbuf,neighbors_x[2],0, comm)
        # copyto!(A[end-2:end ,:,:],reshape(recvbuf,size(A[end-2:end ,:,:])))
		copyto!(A[end-2:end ,:,:],recvbuf)

        # sendbuf = zeros(size(A[end-5:end-3 ,:, :]))
        # recvbuf = zeros(size(A[end-2:end , :, :]))
        # copyto!(sendbuf,A[end-5:end-3 ,:, :])
		# MPI.Isend(sendbuf,  neighbors_x[2], 1, comm)
		# MPI.Irecv!(recvbuf, neighbors_x[2], 0, comm)
        # copyto!(A[end-2:end , :, :],recvbuf)
	end
	# #y direction
	# #Back
	
	if neighbors_y[1] != -1
        # buf_size = 3*(size(A,1)*size(A,3))
        # sendbuf = reshape(Array(A[:,4:6 ,:]),buf_size)
		sendbuf = Array(A[:,4:6 ,:])
        # recvbuf = zeros(buf_size)
		recvbuf = zeros(size(sendbuf))
		# r=MPI.Isend(sendbuf, neighbors_y[1], 0, comm)
		# MPI.Wait!(r)
		# r=MPI.Irecv!(recvbuf, neighbors_y[1], 1, comm)
		# MPI.Wait!(r)
		MPI.Sendrecv!(sendbuf,neighbors_y[1],0,
		recvbuf,neighbors_y[1],1, comm)
        # copyto!(A[:,1:3 ,:],reshape(recvbuf,size(A[:,1:3 ,:])))
		copyto!(A[:,1:3 ,:],recvbuf)

        # sendbuf = zeros(size(A[:,4:6 ,:]))
        # recvbuf = zeros(size(A[:,1:3 ,:]))
        # copyto!(sendbuf,A[:,4:6 ,:])
		# MPI.Isend(sendbuf, neighbors_y[1], 0, comm)
		# MPI.Irecv!(recvbuf, neighbors_y[1], 1, comm)
        # copyto!(A[:,1:3,:],recvbuf)
	end
	#Forward
	if neighbors_y[2] != -1
        # buf_size = 3*(size(A,1)*size(A,3))
        # sendbuf = reshape(Array(A[:,end-5:end-3 , :]),buf_size)
		sendbuf = Array(A[:,end-5:end-3 , :])
        # recvbuf = zeros(buf_size)
		recvbuf = zeros(size(sendbuf))
		# r=MPI.Isend(sendbuf, neighbors_y[2], 1, comm)
		# MPI.Wait!(r)
		# r=MPI.Irecv!(recvbuf, neighbors_y[2], 0, comm)
		# MPI.Wait!(r)
		MPI.Sendrecv!(sendbuf,neighbors_y[2],1,
		recvbuf,neighbors_y[2],0, comm)
        # copyto!(A[:,end-2:end , :],reshape(recvbuf,size(A[:,end-2:end , :])))
		copyto!(A[:,end-2:end , :],recvbuf)

        # sendbuf = zeros(size(A[:,end-5:end-3 , :]))
        # recvbuf = zeros(size(A[:,end-2:end , :]))
        # copyto!(sendbuf,A[:,end-5:end-3 , :])
		# MPI.Isend(sendbuf,  neighbors_y[2], 1, comm)
		# MPI.Irecv!(recvbuf, neighbors_y[2], 0, comm)
        # copyto!(A[:,end-2:end , :],recvbuf)
	end
	
	#z direction
	#Up
	if neighbors_z[1] != -1
        # buf_size = 3*(size(A,1)*size(A,2))
        # sendbuf = reshape(Array(A[:,:,4:6]),buf_size)
		sendbuf = Array(A[:,:,4:6])
        # recvbuf = zeros(buf_size)
		recvbuf = zeros(size(sendbuf))
		# r=MPI.Isend(sendbuf, neighbors_z[1], 0, comm)
		# MPI.Wait!(r)
		# r=MPI.Irecv!(recvbuf, neighbors_z[1], 1, comm)
		# MPI.Wait!(r)
		MPI.Sendrecv!(sendbuf,neighbors_z[1],0,
		recvbuf,neighbors_z[1],1, comm)
        # copyto!(A[:,:,1:3],reshape(recvbuf,size(A[:,:,1:3])))
		copyto!(A[:,:,1:3],recvbuf)

        # sendbuf = zeros(size(A[:,:,4:6]))
        # recvbuf = zeros(size((A[:,:,1:3]))
        # copyto!(sendbuf,A[:,:,4:6])
		# MPI.Isend(sendbuf,  neighbors_z[1], 1, comm)
		# MPI.Irecv!(recvbuf, neighbors_z[1], 0, comm)
        # copyto!((A[:,:,1:3],recvbuf)
	end
	#Down
	if neighbors_z[2] != -1
        # buf_size = 3*(size(A,1)*size(A,2))
        # sendbuf = reshape(Array(A[:,:,end-5:end-3]),buf_size)
		sendbuf = Array(A[:,:,end-5:end-3])
        # recvbuf = zeros(buf_size)
		recvbuf = zeros(size(sendbuf))
		# r=MPI.Isend(sendbuf, neighbors_z[2], 1, comm)
		# MPI.Wait!(r)
		# r=MPI.Irecv!(recvbuf, neighbors_z[2], 0, comm)
		# MPI.Wait!(r)
		MPI.Sendrecv!(sendbuf,neighbors_z[2],1,
		recvbuf,neighbors_z[2],0, comm)
        # copyto!(A[:,:,end-2:end],reshape(recvbuf,size(A[:,:,end-2:end])))
		copyto!(A[:,:,end-2:end],recvbuf)

        # sendbuf = zeros(size(A[:, :,end-5:end-3 ]))
        # recvbuf = zeros(size(A[:, :,end-2:end ]))
        # copyto!(sendbuf,A[:, :,end-5:end-3 ])
		# MPI.Isend(sendbuf,  neighbors_z[2], 1, comm)
		# MPI.Irecv!(recvbuf, neighbors_z[2], 0, comm)
        # copyto!(A[:, :,end-2:end],recvbuf)
	end
	return
end

@views function boundary_x(A,dims,comm,coords)
    # left_rank = MPI.Cart_rank(comm,[0,0,0])
    # right_rank = MPI.Cart_rank(comm,[dims[1]-1,0,0])
    if dims[1]>1
        if (coords[1]==(dims[1]-1))
        # if me == right_rank
            sendbuf=Array(A[end-6:end-3,:,:])
            recvbuf=zeros(size(A[end-2:end,:,:]))
            send_rank = MPI.Cart_rank(comm,[0,coords[2],coords[3]])
            # req=MPI.Isend(sendbuf,send_rank,0,comm)
            # MPI.Wait!(req)
            MPI.Sendrecv!(sendbuf,send_rank,0,recvbuf,send_rank,1,comm)
            copyto!(A[end-2:end,:,:],recvbuf)
        end
        if (coords[1]==0)
            recv_rank = MPI.Cart_rank(comm,[dims[1]-1,coords[2],coords[3]])
            sendbuf=Array(A[4:6,:,:])
            recvbuf=zeros(size(Array(A[end-6:end-3,:,:])))
            # req=MPI.Irecv!(recvbuf,recv_rank,0,comm)
            # MPI.Wait!(req)
            MPI.Sendrecv!(sendbuf,recv_rank,1,recvbuf,recv_rank,0,comm)
            copyto!(A[1:4,:,:],recvbuf)
        end
    else
        # A[1:6,:,:]=A[end-5:end,:,:]
        A[1:4,:,:]=A[end-6:end-3,:,:]
        A[end-2:end,:,:]=A[4:6,:,:]

    end

    return
end

@views function boundary_y(A,dims,comm,coords)
    # left_rank = MPI.Cart_rank(comm,[0,0,0])
    # right_rank = MPI.Cart_rank(comm,[dims[1]-1,0,0])
    if dims[2]>1
        if (coords[2]==(dims[2]-1))
        # if me == right_rank
            sendbuf=Array(A[:,end-6:end-3,:])
            recvbuf=zeros(size(A[:,end-2:end,:]))
            send_rank = MPI.Cart_rank(comm,[coords[1],0,coords[3]])
            # req=MPI.Isend(sendbuf,send_rank,0,comm)
            # MPI.Wait!(req)
            MPI.Sendrecv!(sendbuf,send_rank,0,recvbuf,send_rank,1,comm)
            copyto!(A[:,end-2:end,:],recvbuf)
        end
        if (coords[2]==0)
            recv_rank = MPI.Cart_rank(comm,[coords[1],dims[2]-1,coords[3]])
            sendbuf=Array(A[:,4:6,:])
            recvbuf=zeros(size(Array(A[:,end-6:end-3,:])))
            # req=MPI.Irecv!(recvbuf,recv_rank,0,comm)
            # MPI.Wait!(req)
            # copyto!(A[:,1:6,:],recvbuf)
            MPI.Sendrecv!(sendbuf,recv_rank,1,recvbuf,recv_rank,0,comm)
            copyto!(A[:,1:4,:],recvbuf)
        end
    else
        # A[:,1:6,:]=A[:,end-5:end,:]
        A[:,1:4,:]=A[:,end-6:end-3,:]
        A[:,end-2:end,:]=A[:,4:6,:]

    end

    return
end

@views function boundary_z(A,dims,comm,coords)
    # left_rank = MPI.Cart_rank(comm,[0,0,0])
    # right_rank = MPI.Cart_rank(comm,[dims[1]-1,0,0])
    if dims[3]>1
        if (coords[3]==(dims[3]-1))
        # if me == right_rank
            sendbuf=Array(A[:,:,end-6:end-3])
            recvbuf=zeros(size(A[:,:,end-2:end]))
            send_rank = MPI.Cart_rank(comm,[coords[1],coords[2],0])
            # req=MPI.Isend(sendbuf,send_rank,0,comm)
            # MPI.Wait!(req)
            MPI.Sendrecv!(sendbuf,send_rank,0,recvbuf,send_rank,1,comm)
            copyto!(A[:,:,end-2:end],recvbuf)
        end
        if (coords[3]==0)
            recv_rank = MPI.Cart_rank(comm,[coords[1],coords[2],dims[3]-1])
            sendbuf=Array(A[:,:,4:6])
            recvbuf=zeros(size(Array(A[:,:,end-6:end-3])))
            # req=MPI.Irecv!(recvbuf,recv_rank,0,comm)
            # MPI.Wait!(req)
            MPI.Sendrecv!(sendbuf,recv_rank,1,recvbuf,recv_rank,0,comm)
            # copyto!(A[:,:,1:6],recvbuf)
            copyto!(A[:,:,1:4],recvbuf)
        end
    else
        A[:,:,1:4]=A[:,:,end-6:end-3]
        A[:,:,end-2:end]=A[:,:,4:6]
    end

    return
end

@views function gather_fft(A,A_global,Nx,Ny,Nz,me,comm,nprocs)
	sendbuf=Array(A[4:end-3,4:end-3,4:end-3])
    # nprocs = MPI.Comm_size(comm)
	# println(sendbuf[1,1,1])
    if me!=0
        req=MPI.Isend(sendbuf,0,0,comm)
		MPI.Wait!(req)
    else
		# println(size(sendbuf))
		# println(size(A_global[1:Nx,1:Ny,1:Nz]))
		A_global[1:Nx,1:Ny,1:Nz]=sendbuf
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
    return
end

@views function gather(A,A_global,Nx,Ny,Nz,me,comm,nprocs)
	sendbuf=Array(A[4:end-3,4:end-3,4:end-3])
    # nprocs = MPI.Comm_size(comm)
	# println(sendbuf[1,1,1])
    if me!=0
        req=MPI.Isend(sendbuf,0,0,comm)
		MPI.Wait!(req)
    else
		# println(size(sendbuf))
		# println(size(A_global[1:Nx,1:Ny,1:Nz]))
		A_global[1:Nx,1:Ny,1:Nz]=sendbuf
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
    return
end

@views function gather_metrics(A,A_global,me,comm,nprocs)
	sendbuf=A
    recvbuf=zeros(size(sendbuf))
    for p in range(0,nprocs-1,step=1)
        if p != me
            MPI.Sendrecv!(sendbuf,p,0,recvbuf,p,0,comm)
            A_global[(1+p*36):(p+1)*36] = recvbuf
        else
            A_global[(1+me*36):(me+1)*36]=sendbuf
        end
    end
    return
end

@views function updater(A,k1_A,k3_A,k4_A,k5_A,b1,b3,b4,b5,dt)
    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
    (blockIdx().y - 1) * blockDim().y + threadIdx().y,
    (blockIdx().z - 1) * blockDim().z + threadIdx().z
    
    @inbounds A[i,j,k]=A[i,j,k]+dt*b1*k1_A[i,j,k]+dt*b3*k3_A[i,j,k]+dt*b4*k4_A[i,j,k]+dt*b5*k5_A[i,j,k]

    return
end

function run()

    MPI.Init()
    dims        = [0,0,0]
    comm        = MPI.COMM_WORLD
    # comm        = MPI.CUDA_COMM_WORLD
    nprocs      = MPI.Comm_size(comm)
    MPI.Dims_create!(nprocs, dims)
    comm_cart   = MPI.Cart_create(comm, dims, [0,0,0], 1)
    me          = MPI.Comm_rank(comm_cart)
    coords      = MPI.Cart_coords(comm_cart)
    neighbors_x = MPI.Cart_shift(comm_cart, 0, 1)
    neighbors_y = MPI.Cart_shift(comm_cart, 1, 1)
    neighbors_z = MPI.Cart_shift(comm_cart, 2, 1)

    Nx=Ny=Nz=32*6
    println(Nx,",",Ny,",",Nz)
    gw = 0.65
    gy = 0.34521
    gp2 = 0.75
    vev = 1.0
    lambda = 1.0/8.0
    dx=0.2
    dt=dx/20
    mH = 2.0*sqrt(lambda)*vev
    nte = 5000
    θ_w = asin(sqrt(0.22))
    tol = 1e-4
    a_tol=tol
    r_tol=tol
    sfty = 0.9

    Nx_g =(Nx-6)*dims[1]
    Ny_g =(Ny-6)*dims[2]
    Nz_g =(Nz-6)*dims[3]

    nsnaps=100
    dsnaps = floor(Int,nte/nsnaps)

    seed_value = 1223453134
    no_bubbles = 20

    # Array initializations
    ϕ_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    ϕ_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    ϕ_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    ϕ_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))

    # W_1_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    W_1_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    W_1_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    W_1_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # W_2_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    W_2_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    W_2_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    W_2_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # W_3_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    W_3_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    W_3_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    W_3_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # Y_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    Y_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    Y_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    Y_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    Γ_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    Γ_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    Γ_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    Σ = CUDA.zeros(Float64,(Nx,Ny,Nz))
    dϕ_1_dt = CUDA.zeros(Float64,(Nx,Ny,Nz))
    dϕ_2_dt = CUDA.zeros(Float64,(Nx,Ny,Nz))
    dϕ_3_dt = CUDA.zeros(Float64,(Nx,Ny,Nz))
    dϕ_4_dt = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     dW_1_1_dt = CUDA.zeros(Float64,(Nx,Ny,Nz))
    dW_1_2_dt = CUDA.zeros(Float64,(Nx,Ny,Nz))
    dW_1_3_dt = CUDA.zeros(Float64,(Nx,Ny,Nz))
    dW_1_4_dt = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     dW_2_1_dt = CUDA.zeros(Float64,(Nx,Ny,Nz))
    dW_2_2_dt = CUDA.zeros(Float64,(Nx,Ny,Nz))
    dW_2_3_dt = CUDA.zeros(Float64,(Nx,Ny,Nz))
    dW_2_4_dt = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     dW_3_1_dt = CUDA.zeros(Float64,(Nx,Ny,Nz))
    dW_3_2_dt = CUDA.zeros(Float64,(Nx,Ny,Nz))
    dW_3_3_dt = CUDA.zeros(Float64,(Nx,Ny,Nz))
    dW_3_4_dt = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     dY_1_dt = CUDA.zeros(Float64,(Nx,Ny,Nz))
    dY_2_dt = CUDA.zeros(Float64,(Nx,Ny,Nz))
    dY_3_dt = CUDA.zeros(Float64,(Nx,Ny,Nz))
    dY_4_dt = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # dΓ_1_dt = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # dΓ_2_dt = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # dΓ_3_dt = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # dΣ_dt = CUDA.zeros(Float64,(Nx,Ny,Nz))

    #Flux arrays
    k1_ϕ_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k1_ϕ_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k1_ϕ_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k1_ϕ_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))

    # k1_W_1_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k1_W_1_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k1_W_1_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k1_W_1_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # k1_W_2_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k1_W_2_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k1_W_2_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k1_W_2_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # k1_W_3_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k1_W_3_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k1_W_3_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k1_W_3_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # k1_Y_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k1_Y_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k1_Y_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k1_Y_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k1_Γ_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k1_Γ_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k1_Γ_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k1_Σ = CUDA.zeros(Float64,(Nx,Ny,Nz))

    #Flux arrays
    kt1_ϕ_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt1_ϕ_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt1_ϕ_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt1_ϕ_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))

    # kt1_W_1_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt1_W_1_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt1_W_1_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt1_W_1_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt1_W_2_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt1_W_2_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt1_W_2_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt1_W_2_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt1_W_3_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt1_W_3_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt1_W_3_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt1_W_3_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt1_Y_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt1_Y_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt1_Y_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt1_Y_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt1_Γ_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt1_Γ_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt1_Γ_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt1_Σ = CUDA.zeros(Float64,(Nx,Ny,Nz))

    #Flux arrays 2
    k2_ϕ_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k2_ϕ_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k2_ϕ_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k2_ϕ_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))

    # k2_W_1_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k2_W_1_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k2_W_1_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k2_W_1_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # k2_W_2_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k2_W_2_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k2_W_2_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k2_W_2_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # k2_W_3_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k2_W_3_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k2_W_3_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k2_W_3_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # k2_Y_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k2_Y_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k2_Y_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k2_Y_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k2_Γ_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k2_Γ_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k2_Γ_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k2_Σ = CUDA.zeros(Float64,(Nx,Ny,Nz))
    
    #Flux arrays 2
    kt2_ϕ_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt2_ϕ_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt2_ϕ_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt2_ϕ_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))

    # kt2_W_1_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt2_W_1_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt2_W_1_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt2_W_1_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt2_W_2_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt2_W_2_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt2_W_2_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt2_W_2_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt2_W_3_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt2_W_3_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt2_W_3_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt2_W_3_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt2_Y_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt2_Y_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt2_Y_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt2_Y_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt2_Γ_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt2_Γ_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt2_Γ_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt2_Σ = CUDA.zeros(Float64,(Nx,Ny,Nz))
    
    #Flux arrays 3
    k3_ϕ_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k3_ϕ_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k3_ϕ_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k3_ϕ_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))

    # k3_W_1_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k3_W_1_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k3_W_1_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k3_W_1_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # k3_W_2_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k3_W_2_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k3_W_2_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k3_W_2_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # k3_W_3_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k3_W_3_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k3_W_3_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k3_W_3_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # k3_Y_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k3_Y_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k3_Y_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k3_Y_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k3_Γ_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k3_Γ_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k3_Γ_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k3_Σ = CUDA.zeros(Float64,(Nx,Ny,Nz))
    
    #Flux arrays 3
    kt3_ϕ_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt3_ϕ_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt3_ϕ_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt3_ϕ_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))

    # kt3_W_1_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt3_W_1_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt3_W_1_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt3_W_1_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt3_W_2_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt3_W_2_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt3_W_2_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt3_W_2_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt3_W_3_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt3_W_3_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt3_W_3_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt3_W_3_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt3_Y_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt3_Y_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt3_Y_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt3_Y_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt3_Γ_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt3_Γ_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt3_Γ_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt3_Σ = CUDA.zeros(Float64,(Nx,Ny,Nz))
    
    #Flux arrays 4
    k4_ϕ_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k4_ϕ_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k4_ϕ_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k4_ϕ_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))

    # k4_W_1_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k4_W_1_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k4_W_1_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k4_W_1_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # k4_W_2_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k4_W_2_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k4_W_2_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k4_W_2_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # k4_W_3_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k4_W_3_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k4_W_3_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k4_W_3_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # k4_Y_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k4_Y_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k4_Y_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k4_Y_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k4_Γ_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k4_Γ_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k4_Γ_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k4_Σ = CUDA.zeros(Float64,(Nx,Ny,Nz))
    
    #Flux arrays 4
    kt4_ϕ_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt4_ϕ_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt4_ϕ_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt4_ϕ_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))

    # kt4_W_1_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt4_W_1_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt4_W_1_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt4_W_1_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt4_W_2_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt4_W_2_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt4_W_2_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt4_W_2_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt4_W_3_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt4_W_3_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt4_W_3_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt4_W_3_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt4_Y_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt4_Y_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt4_Y_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt4_Y_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt4_Γ_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt4_Γ_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt4_Γ_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt4_Σ = CUDA.zeros(Float64,(Nx,Ny,Nz))
    
    #Flux arrays 5
    k5_ϕ_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k5_ϕ_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k5_ϕ_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k5_ϕ_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))

    # k5_W_1_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k5_W_1_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k5_W_1_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k5_W_1_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # k5_W_2_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k5_W_2_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k5_W_2_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k5_W_2_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # k5_W_3_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k5_W_3_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k5_W_3_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k5_W_3_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # k5_Y_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k5_Y_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k5_Y_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k5_Y_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k5_Γ_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k5_Γ_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k5_Γ_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k5_Σ = CUDA.zeros(Float64,(Nx,Ny,Nz))
    
    #Flux arrays 5
    kt5_ϕ_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt5_ϕ_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt5_ϕ_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt5_ϕ_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))

    # kt5_W_1_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt5_W_1_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt5_W_1_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt5_W_1_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt5_W_2_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt5_W_2_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt5_W_2_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt5_W_2_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt5_W_3_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt5_W_3_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt5_W_3_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt5_W_3_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt5_Y_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt5_Y_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt5_Y_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt5_Y_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt5_Γ_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt5_Γ_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt5_Γ_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt5_Σ = CUDA.zeros(Float64,(Nx,Ny,Nz))
    
    #Flux arrays 6
    k6_ϕ_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k6_ϕ_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k6_ϕ_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k6_ϕ_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))

    # k6_W_1_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k6_W_1_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k6_W_1_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k6_W_1_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # k6_W_2_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k6_W_2_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k6_W_2_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k6_W_2_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # k6_W_3_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k6_W_3_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k6_W_3_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k6_W_3_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # k6_Y_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k6_Y_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k6_Y_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k6_Y_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k6_Γ_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k6_Γ_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k6_Γ_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k6_Σ = CUDA.zeros(Float64,(Nx,Ny,Nz))
    
    #Flux arrays 6
    kt6_ϕ_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt6_ϕ_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt6_ϕ_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt6_ϕ_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))

    # kt6_W_1_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt6_W_1_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt6_W_1_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt6_W_1_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt6_W_2_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt6_W_2_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt6_W_2_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt6_W_2_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt6_W_3_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt6_W_3_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt6_W_3_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt6_W_3_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt6_Y_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt6_Y_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt6_Y_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt6_Y_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt6_Γ_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt6_Γ_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt6_Γ_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt6_Σ = CUDA.zeros(Float64,(Nx,Ny,Nz))
    
#     #Updated field arrays
#     ϕ_1_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     ϕ_2_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     ϕ_3_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     ϕ_4_n = CUDA.zeros(Float64,(Nx,Ny,Nz))

#     W_1_1_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     W_1_2_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     W_1_3_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     W_1_4_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     W_2_1_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     W_2_2_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     W_2_3_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     W_2_4_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     W_3_1_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     W_3_2_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     W_3_3_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     W_3_4_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     Y_1_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     Y_2_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     Y_3_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     Y_4_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     Γ_1_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     Γ_2_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     Γ_3_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     Σ_n = CUDA.zeros(Float64,(Nx,Ny,Nz))

#     dϕ_1_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     dϕ_2_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     dϕ_3_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     dϕ_4_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
# #     dW_1_1_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     dW_1_2_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     dW_1_3_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     dW_1_4_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
# #     dW_2_1_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     dW_2_2_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     dW_2_3_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     dW_2_4_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
# #     dW_3_1_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     dW_3_2_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     dW_3_3_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     dW_3_4_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
# #     dY_1_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     dY_2_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     dY_3_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     dY_4_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     dΓ_1_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     dΓ_2_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     dΓ_3_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     dΣ_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))

    ##Energy arrays##
    KE_Phi = CUDA.zeros(Float64,(Nx,Ny,Nz))
    GE_Phi = CUDA.zeros(Float64,(Nx,Ny,Nz))
    PE_Phi = CUDA.zeros(Float64,(Nx,Ny,Nz))
    ElectricE_W = CUDA.zeros(Float64,(Nx,Ny,Nz))
    MagneticE_W = CUDA.zeros(Float64,(Nx,Ny,Nz))
    ElectricE_Y = CUDA.zeros(Float64,(Nx,Ny,Nz))
    MagneticE_Y = CUDA.zeros(Float64,(Nx,Ny,Nz))
    total_energies = zeros((nsnaps+1,7))

    B_x = CUDA.zeros(Float64,(Nx,Ny,Nz))
    B_y = CUDA.zeros(Float64,(Nx,Ny,Nz))
    B_z = CUDA.zeros(Float64,(Nx,Ny,Nz))

    # spec_cut = [Nx÷4,Ny÷4,Nz÷4]
    spec_cut = [Nx_g÷8,Ny_g÷8,Nz_g÷8]
    N_bins = Kc_bin_nums(spec_cut[1],spec_cut[2],spec_cut[3])
    B_fft = zeros((nsnaps+1,N_bins,2))
    
    #Error arrays
    err = zeros((36))
    err_g = zeros((36*nprocs))
    max_mags_g = zeros((36*nprocs))

    #Global Arrays
    KE_Phi_g =zeros(Float64,(Nx_g,Ny_g,Nz_g))
    GE_Phi_g =zeros(Float64,(Nx_g,Ny_g,Nz_g))
    PE_Phi_g =zeros(Float64,(Nx_g,Ny_g,Nz_g))
    ElectricE_W_g =zeros(Float64,(Nx_g,Ny_g,Nz_g))
    MagneticE_W_g =zeros(Float64,(Nx_g,Ny_g,Nz_g))
    ElectricE_Y_g =zeros(Float64,(Nx_g,Ny_g,Nz_g))
    MagneticE_Y_g =zeros(Float64,(Nx_g,Ny_g,Nz_g))
    B_x_fft_g=zeros(ComplexF64,(Nx_g,Ny_g,Nz_g))
    B_y_fft_g=zeros(ComplexF64,(Nx_g,Ny_g,Nz_g))
    B_z_fft_g=zeros(ComplexF64,(Nx_g,Ny_g,Nz_g))

    CUDA.memory_status()

    ##########Configuring thread block grid###########

    thrds = (32,1,1)
    blks = (Nx÷thrds[1],Ny÷thrds[2],Nz÷thrds[3])
    println(string("#threads:",thrds," #blocks:",blks))

    ##########END Configuring thread block grid###########

    #Initializing random bubbles

    # Random.seed!(seed_value)
    # bubble_locs = rand(1:Nx,(no_bubbles,3))
    Random.seed!(seed_value)
    bubble_locs_x = rand(4:(Nx-3)*dims[1],no_bubbles)
    Random.seed!(seed_value*2)
    bubble_locs_y = rand(4:(Ny-3)*dims[2],no_bubbles)
    Random.seed!(seed_value*3)
    bubble_locs_z = rand(4:(Nz-3)*dims[3],no_bubbles)
    bubble_locs = hcat(bubble_locs_x,bubble_locs_y,bubble_locs_z)

    println(string("bubble location matrix",size(bubble_locs)))
    
    Random.seed!(seed_value)
    Hoft_arr = rand(Uniform(0,1),(no_bubbles,3))

    bubs = []
    for bub_idx in range(1,no_bubbles)
        phi=gen_phi(Hoft_arr[bub_idx,:])
        ib,jb,kb = bubble_locs[bub_idx,:]
        # println(string(ib," ",jb," ",kb," ", phi))
        push!(bubs,[ib,jb,kb,real(phi[1]),imag(phi[1]),real(phi[2]),imag(phi[2])])
    end

    # rkx=pi/(Nx*dx)
    # rky=pi/(Ny*dx)
    # rkz=pi/(Nz*dx)
    rkx=pi/(Nx_g*dx)
    rky=pi/(Ny_g*dx)
    rkz=pi/(Nz_g*dx)

    @time for b in range(1,size(bubs,1),step=1)
        ib,jb,kb,p1,p2,p3,p4 = bubs[b]
        # @cuda threads=thrds blocks=blks initializer!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,ib,jb,kb,p1,p2,p3,p4,rkx,rky,rkz,dx,mH)
        # @cuda threads=thrds blocks=blks initializer!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,ib,jb,kb,p1,p2,p3,p4,rkx,rky,rkz,dx,mH)
        @cuda threads=thrds blocks=blks initializer!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,ib,jb,kb,p1,p2,p3,p4,rkx,rky,rkz,dx,mH,Nx,Ny,Nz,coords[1],coords[2],coords[3])
        synchronize()
    end

    #compute energies and magnetic fields at initial time step
    @cuda threads=thrds blocks=blks compute_aux!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
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
    KE_Phi,GE_Phi,PE_Phi,
    ElectricE_W,MagneticE_W,ElectricE_Y,MagneticE_Y,
    B_x,B_y,B_z,
    gw,gy,gp2,vev,lambda,θ_w,dx)
    
    synchronize()

    # Compute fft and convolve spectrum
    @time begin
    B_x_fft = Array(fft(B_x))
    B_y_fft = Array(fft(B_y))
    B_z_fft = Array(fft(B_z))
    
    gather_fft(B_x_fft,B_x_fft_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
    gather_fft(B_y_fft,B_y_fft_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
    gather_fft(B_z_fft,B_z_fft_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
    MPI.Barrier(comm_cart)

    B_fft[1,:,:] = convolve_1d((real(conj.(B_x_fft_g).*B_x_fft_g.+
    conj.(B_y_fft_g).*B_y_fft_g.+
    conj.(B_z_fft_g).*B_z_fft_g)),Nx_g,Ny_g,Nz_g,spec_cut[1],spec_cut[2],spec_cut[3])
    end

    #Gather energies
    gather(KE_Phi,KE_Phi_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
    gather(PE_Phi,PE_Phi_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
    gather(GE_Phi,GE_Phi_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
    gather(MagneticE_W,MagneticE_W_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
    gather(ElectricE_W,ElectricE_W_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
    gather(MagneticE_Y,MagneticE_Y_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
    gather(ElectricE_Y,ElectricE_Y_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
    MPI.Barrier(comm_cart)

    # #Add energies
    # total_energies[1,1] = sum(Array(PE_Phi))
    # total_energies[1,2] = sum(Array(KE_Phi))
    # total_energies[1,3] = sum(Array(GE_Phi))
    # total_energies[1,4] = sum(Array(ElectricE_W))
    # total_energies[1,5] = sum(Array(MagneticE_W))
    # total_energies[1,6] = sum(Array(ElectricE_Y))
    # total_energies[1,7] = sum(Array(MagneticE_Y))
    total_energies[1,1] = sum(Array(PE_Phi_g))
    total_energies[1,2] = sum(Array(KE_Phi_g))
    total_energies[1,3] = sum(Array(GE_Phi_g))
    total_energies[1,4] = sum(Array(ElectricE_W_g))
    total_energies[1,5] = sum(Array(MagneticE_W_g))
    total_energies[1,6] = sum(Array(ElectricE_Y_g))
    total_energies[1,7] = sum(Array(MagneticE_Y_g))


    ##PLOT initial frame##
    if me==0
        ##PLOT##
        x=range(1,Nx_g,step=1)
        y=range(1,Ny_g,step=1)
        z=range(1,Nz_g,step=1)

        gr()
        ENV["GKSwstype"]="nul"
        anim = Animation();
        # plot_1=contourf(z,x,(Array(ϕ_1)[:,Ny÷2,:]).^2+(Array(ϕ_2)[:,Ny÷2,:]).^2+(Array(ϕ_3)[:,Ny÷2,:]).^2+(Array(ϕ_4)[:,Ny÷2,:]).^2)
        plot_1=contourf(z,x,(Array(PE_Phi_g)[:,Ny_g÷2,:]),title="PE")
        plot_2=contourf(z,x,(Array(KE_Phi_g)[:,Ny_g÷2,:]),title="KE")
        # plot_3=contourf(z,x,(Array(GE_Phi_g)[:,Ny_g÷2,:]),title="GE")
        plot_3=plot(B_fft[1,2:end,1],(((B_fft[1,2:end,1]).^2)./((2*pi)^3*Nx^2)).*B_fft[1,2:end,2],xscale=:log10,yscale=:log10,minorgrid=true)
        # plot_4=contourf(z,x,(Array(ElectricE_W)[:,Ny÷2,:]+Array(MagneticE_W)[:,Ny÷2,:]+Array(ElectricE_Y)[:,Ny÷2,:]+Array(MagneticE_Y)[:,Ny÷2,:]),title="WY E")
        plot_4 = scatter([0],[total_energies[1,1] total_energies[1,2] total_energies[1,3] total_energies[1,4] total_energies[1,5] total_energies[1,6] total_energies[1,7]],
        label=["PE" "KE" "GE" "EEW" "MEW" "EEY" "MEY"],xlims=(0,nte))
        plot(plot_1,plot_2,plot_3,plot_4,layout=4,plot_title=string("it:",0),dpi=600)
        # plot(plot_1,title=string("it:",0),dpi=600)
        png(string("testini1_rk45",".png"))
        frame(anim)
        # exit()
    end

    MPI.Barrier(comm_cart)

    #RKF45 coefficients
    a21 = 1.0/4.0
    a31 = 3.0/32.0
    a32 = 9.0/32.0
    a41 = 1932.0/2197.0
    a42 = -7200.0/2197.0
    a43 = 7296.0/2197.0
    a51 = 439.0/216.0
    a52 = -8.0
    a53 = 3680.0/513.0
    a54 = -845.0/4104.0
    a61 = -8.0/27.0
    a62 = -2.0
    a63 = -3544.0/2565.0
    a64 = 1859.0/4104.0
    a65 = -11.0/40.0
    b1 = 5.0/216.0
    b2 = 0.0
    b3 = 1408.0/2565.0
    b4 = 2197.0/1404.0
    b5 = -1.0/5.0
    c1 = 16.0/135.0
    c2 = 0.0
    c3 = 6656.0/12825.0
    c4 = 28561.0/56430.0
    c5 = -9.0/50.0
    c6 = 2.0/55.0
    
    ###END Initializing###
    #Counter for snaps
    snp_idx = 1
    it=1
    # @time for it in range(1,nte,step=1)
    while it < nte
        # @time begin
        #Compute k_1 arrays
        @cuda threads=thrds blocks=blks rk4_step!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
        W_1_2,W_1_3,W_1_4,
        W_2_2,W_2_3,W_2_4,
        W_3_2,W_3_3,W_3_4,
        Y_2,Y_3,Y_4,
        Γ_1,Γ_2,Γ_3,Σ,
        k1_ϕ_1,k1_ϕ_2,k1_ϕ_3,k1_ϕ_4,
        k1_W_1_2,k1_W_1_3,k1_W_1_4,
        k1_W_2_2,k1_W_2_3,k1_W_2_4,
        k1_W_3_2,k1_W_3_3,k1_W_3_4,
        k1_Y_2,k1_Y_3,k1_Y_4,
        k1_Γ_1,k1_Γ_2,k1_Γ_3,k1_Σ,
        dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
        dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
        dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
        dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
        dY_2_dt,dY_3_dt,dY_4_dt,
        kt1_ϕ_1,kt1_ϕ_2,kt1_ϕ_3,kt1_ϕ_4,
        kt1_W_1_2,kt1_W_1_3,kt1_W_1_4,
        kt1_W_2_2,kt1_W_2_3,kt1_W_2_4,
        kt1_W_3_2,kt1_W_3_3,kt1_W_3_4,
        kt1_Y_2,kt1_Y_3,kt1_Y_4,
        gw,gy,gp2,vev,lambda,dx)
        synchronize()

        #Compute k_2 arrays
        @cuda threads=thrds blocks=blks rk4_step!(ϕ_1+k1_ϕ_1 .*dt*a21,ϕ_2+k1_ϕ_2 .*dt*a21,ϕ_3+k1_ϕ_3 .*dt*a21,ϕ_4+k1_ϕ_4 .*dt*a21,
        W_1_2+k1_W_1_2 .*dt*a21,W_1_3+k1_W_1_3 .*dt*a21,W_1_4+k1_W_1_4 .*dt*a21,
        W_2_2+k1_W_2_2 .*dt*a21,W_2_3+k1_W_2_3 .*dt*a21,W_2_4+k1_W_2_4 .*dt*a21,
        W_3_2+k1_W_3_2 .*dt*a21,W_3_3+k1_W_3_3 .*dt*a21,W_3_4+k1_W_3_4 .*dt*a21,
        Y_2+k1_Y_2 .*dt*a21,Y_3+k1_Y_3 .*dt*a21,Y_4+k1_Y_4 .*dt*a21,
        Γ_1+k1_Γ_1 .*dt*a21,Γ_2+k1_Γ_2 .*dt*a21,Γ_3+k1_Γ_3 .*dt*a21,Σ+k1_Σ .*dt*a21,
        k2_ϕ_1,k2_ϕ_2,k2_ϕ_3,k2_ϕ_4,
        k2_W_1_2,k2_W_1_3,k2_W_1_4,
        k2_W_2_2,k2_W_2_3,k2_W_2_4,
        k2_W_3_2,k2_W_3_3,k2_W_3_4,
        k2_Y_2,k2_Y_3,k2_Y_4,
        k2_Γ_1,k2_Γ_2,k2_Γ_3,k2_Σ,
        dϕ_1_dt+kt1_ϕ_1 .*dt*a21,dϕ_2_dt+kt1_ϕ_2 .*dt*a21,dϕ_3_dt+kt1_ϕ_3 .*dt*a21,dϕ_4_dt+kt1_ϕ_4 .*dt*a21,
        dW_1_2_dt+kt1_W_1_2 .*dt*a21,dW_1_3_dt+kt1_W_1_3 .*dt*a21,dW_1_4_dt+kt1_W_1_4 .*dt*a21,
        dW_2_2_dt+kt1_W_2_2 .*dt*a21,dW_2_3_dt+kt1_W_2_3 .*dt*a21,dW_2_4_dt+kt1_W_2_4 .*dt*a21,
        dW_3_2_dt+kt1_W_3_2 .*dt*a21,dW_3_3_dt+kt1_W_3_3 .*dt*a21,dW_3_4_dt+kt1_W_3_4 .*dt*a21,
        dY_2_dt+kt1_Y_2 .*dt*a21,dY_3_dt+kt1_Y_3 .*dt*a21,dY_4_dt+kt1_Y_4 .*dt*a21,
        kt2_ϕ_1,kt2_ϕ_2,kt2_ϕ_3,kt2_ϕ_4,
        kt2_W_1_2,kt2_W_1_3,kt2_W_1_4,
        kt2_W_2_2,kt2_W_2_3,kt2_W_2_4,
        kt2_W_3_2,kt2_W_3_3,kt2_W_3_4,
        kt2_Y_2,kt2_Y_3,kt2_Y_4,
        gw,gy,gp2,vev,lambda,dx)
        synchronize()

        #Compute k_3 arrays
        @cuda threads=thrds blocks=blks rk4_step!(ϕ_1+k1_ϕ_1 .*dt*a31+k2_ϕ_1 .*dt*a32,ϕ_2+k1_ϕ_2 .*dt*a31+k2_ϕ_2 .*dt*a32,ϕ_3+k1_ϕ_3 .*dt*a31+k2_ϕ_3 .*dt*a32,ϕ_4+k1_ϕ_4 .*dt*a31+k2_ϕ_4 .*dt*a32,
        W_1_2+k1_W_1_2 .*dt*a31+k2_W_1_2 .*dt*a32,W_1_3+k1_W_1_3 .*dt*a31+k2_W_1_3 .*dt*a32,W_1_4+k1_W_1_4 .*dt*a31+k2_W_1_4 .*dt*a32,
        W_2_2+k1_W_2_2 .*dt*a31+k2_W_2_2 .*dt*a32,W_2_3+k1_W_2_3 .*dt*a31+k2_W_2_3 .*dt*a32,W_2_4+k1_W_2_4 .*dt*a31+k2_W_2_4 .*dt*a32,
        W_3_2+k1_W_3_2 .*dt*a31+k2_W_3_2 .*dt*a32,W_3_3+k1_W_3_3 .*dt*a31+k2_W_3_3 .*dt*a32,W_3_4+k1_W_3_4 .*dt*a31+k2_W_3_4 .*dt*a32,
        Y_2+k1_Y_2 .*dt*a31+k2_Y_2 .*dt*a32,Y_3+k1_Y_3 .*dt*a31+k2_Y_3 .*dt*a32,Y_4+k1_Y_4 .*dt*a31+k2_Y_4 .*dt*a32,
        Γ_1+k1_Γ_1 .*dt*a31+k2_Γ_1 .*dt*a32,Γ_2+k1_Γ_2 .*dt*a31+k2_Γ_2 .*dt*a32,Γ_3+k1_Γ_3 .*dt*a31+k2_Γ_3 .*dt*a32,Σ+k1_Σ .*dt*a31+k2_Σ .*dt*a32,
        k3_ϕ_1,k3_ϕ_2,k3_ϕ_3,k3_ϕ_4,
        k3_W_1_2,k3_W_1_3,k3_W_1_4,
        k3_W_2_2,k3_W_2_3,k3_W_2_4,
        k3_W_3_2,k3_W_3_3,k3_W_3_4,
        k3_Y_2,k3_Y_3,k3_Y_4,
        k3_Γ_1,k3_Γ_2,k3_Γ_3,k3_Σ,
        dϕ_1_dt+kt1_ϕ_1 .*dt*a31+kt2_ϕ_1 .*dt*a32,dϕ_2_dt+kt1_ϕ_2 .*dt*a31+kt2_ϕ_2 .*dt*a32,dϕ_3_dt+kt1_ϕ_3 .*dt*a31+kt2_ϕ_3 .*dt*a32,dϕ_4_dt+kt1_ϕ_4 .*dt*a31+kt2_ϕ_4 .*dt*a32,
        dW_1_2_dt+kt1_W_1_2 .*dt*a31+kt2_W_1_2 .*dt*a32,dW_1_3_dt+kt1_W_1_3 .*dt*a31+kt2_W_1_3 .*dt*a32,dW_1_4_dt+kt1_W_1_4 .*dt*a31+kt2_W_1_4 .*dt*a32,
        dW_2_2_dt+kt1_W_2_2 .*dt*a31+kt2_W_2_2 .*dt*a32,dW_2_3_dt+kt1_W_2_3 .*dt*a31+kt2_W_2_3 .*dt*a32,dW_2_4_dt+kt1_W_2_4 .*dt*a31+kt2_W_2_4 .*dt*a32,
        dW_3_2_dt+kt1_W_3_2 .*dt*a31+kt2_W_3_2 .*dt*a32,dW_3_3_dt+kt1_W_3_3 .*dt*a31+kt2_W_3_3 .*dt*a32,dW_3_4_dt+kt1_W_3_4 .*dt*a31+kt2_W_3_4 .*dt*a32,
        dY_2_dt+kt1_Y_2 .*dt*a31+kt2_Y_2 .*dt*a32,dY_3_dt+kt1_Y_3 .*dt*a31+kt2_Y_3 .*dt*a32,dY_4_dt+kt1_Y_4 .*dt*a31+kt2_Y_4 .*dt*a32,
        kt3_ϕ_1,kt3_ϕ_2,kt3_ϕ_3,kt3_ϕ_4,
        kt3_W_1_2,kt3_W_1_3,kt3_W_1_4,
        kt3_W_2_2,kt3_W_2_3,kt3_W_2_4,
        kt3_W_3_2,kt3_W_3_3,kt3_W_3_4,
        kt3_Y_2,kt3_Y_3,kt3_Y_4,
        gw,gy,gp2,vev,lambda,dx)
        synchronize()

        #Compute k_4 arrays
        @cuda threads=thrds blocks=blks rk4_step!(ϕ_1+k1_ϕ_1 .*dt*a41+k2_ϕ_1 .*dt*a42+k3_ϕ_1 .*dt*a43,ϕ_2+k1_ϕ_2 .*dt*a41+k2_ϕ_2 .*dt*a42+k3_ϕ_2 .*dt*a43,ϕ_3+k1_ϕ_3 .*dt*a41+k2_ϕ_3 .*dt*a42+k3_ϕ_3 .*dt*a43,ϕ_4+k1_ϕ_4 .*dt*a41+k2_ϕ_4 .*dt*a42+k3_ϕ_4 .*dt*a43,
        W_1_2+k1_W_1_2 .*dt*a41+k2_W_1_2 .*dt*a42+k3_W_1_2 .*dt*a43,W_1_3+k1_W_1_3 .*dt*a41+k2_W_1_3 .*dt*a42+k3_W_1_3 .*dt*a43,W_1_4+k1_W_1_4 .*dt*a41+k2_W_1_4 .*dt*a42+k3_W_1_4 .*dt*a43,
        W_2_2+k1_W_2_2 .*dt*a41+k2_W_2_2 .*dt*a42+k3_W_2_2 .*dt*a43,W_2_3+k1_W_2_3 .*dt*a41+k2_W_2_3 .*dt*a42+k3_W_2_3 .*dt*a43,W_2_4+k1_W_2_4 .*dt*a41+k2_W_2_4 .*dt*a42+k3_W_2_4 .*dt*a43,
        W_3_2+k1_W_3_2 .*dt*a41+k2_W_3_2 .*dt*a42+k3_W_3_2 .*dt*a43,W_3_3+k1_W_3_3 .*dt*a41+k2_W_3_3 .*dt*a42+k3_W_3_3 .*dt*a43,W_3_4+k1_W_3_4 .*dt*a41+k2_W_3_4 .*dt*a42+k3_W_3_4 .*dt*a43,
        Y_2+k1_Y_2 .*dt*a41+k2_Y_2 .*dt*a42+k3_Y_2 .*dt*a43,Y_3+k1_Y_3 .*dt*a41+k2_Y_3 .*dt*a42+k3_Y_3 .*dt*a43,Y_4+k1_Y_4 .*dt*a41+k2_Y_4 .*dt*a42+k3_Y_4 .*dt*a43,
        Γ_1+k1_Γ_1 .*dt*a41+k2_Γ_1 .*dt*a42+k3_Γ_1 .*dt*a43,Γ_2+k1_Γ_2 .*dt*a41+k2_Γ_2 .*dt*a42+k3_Γ_2 .*dt*a43,Γ_3+k1_Γ_3 .*dt*a41+k2_Γ_3 .*dt*a42+k3_Γ_3 .*dt*a43,Σ+k1_Σ .*dt*a41+k2_Σ .*dt*a42+k3_Σ .*dt*a43,
        k4_ϕ_1,k4_ϕ_2,k4_ϕ_3,k4_ϕ_4,
        k4_W_1_2,k4_W_1_3,k4_W_1_4,
        k4_W_2_2,k4_W_2_3,k4_W_2_4,
        k4_W_3_2,k4_W_3_3,k4_W_3_4,
        k4_Y_2,k4_Y_3,k4_Y_4,
        k4_Γ_1,k4_Γ_2,k4_Γ_3,k4_Σ,
        dϕ_1_dt+kt1_ϕ_1 .*dt*a41+kt2_ϕ_1 .*dt*a42+kt3_ϕ_1 .*dt*a43,dϕ_2_dt+kt1_ϕ_2 .*dt*a41+kt2_ϕ_2 .*dt*a42+kt3_ϕ_2 .*dt*a43,dϕ_3_dt+kt1_ϕ_3 .*dt*a41+kt2_ϕ_3 .*dt*a42+kt3_ϕ_3 .*dt*a43,dϕ_4_dt+kt1_ϕ_4 .*dt*a41+kt2_ϕ_4 .*dt*a42+kt3_ϕ_4 .*dt*a43,
        dW_1_2_dt+kt1_W_1_2 .*dt*a41+kt2_W_1_2 .*dt*a42+kt3_W_1_2 .*dt*a43,dW_1_3_dt+kt1_W_1_3 .*dt*a41+kt2_W_1_3 .*dt*a42+kt3_W_1_3 .*dt*a43,dW_1_4_dt+kt1_W_1_4 .*dt*a41+kt2_W_1_4 .*dt*a42+kt3_W_1_4 .*dt*a43,
        dW_2_2_dt+kt1_W_2_2 .*dt*a41+kt2_W_2_2 .*dt*a42+kt3_W_2_2 .*dt*a43,dW_2_3_dt+kt1_W_2_3 .*dt*a41+kt2_W_2_3 .*dt*a42+kt3_W_2_3 .*dt*a43,dW_2_4_dt+kt1_W_2_4 .*dt*a41+kt2_W_2_4 .*dt*a42+kt3_W_2_4 .*dt*a43,
        dW_3_2_dt+kt1_W_3_2 .*dt*a41+kt2_W_3_2 .*dt*a42+kt3_W_3_2 .*dt*a43,dW_3_3_dt+kt1_W_3_3 .*dt*a41+kt2_W_3_3 .*dt*a42+kt3_W_3_3 .*dt*a43,dW_3_4_dt+kt1_W_3_4 .*dt*a41+kt2_W_3_4 .*dt*a42+kt3_W_3_4 .*dt*a43,
        dY_2_dt+kt1_Y_2 .*dt*a41+kt2_Y_2 .*dt*a42+kt3_Y_2 .*dt*a43,dY_3_dt+kt1_Y_3 .*dt*a41+kt2_Y_3 .*dt*a42+kt3_Y_3 .*dt*a43,dY_4_dt+kt1_Y_4 .*dt*a41+kt2_Y_4 .*dt*a42+kt3_Y_4 .*dt*a43,
        kt4_ϕ_1,kt4_ϕ_2,kt4_ϕ_3,kt4_ϕ_4,
        kt4_W_1_2,kt4_W_1_3,kt4_W_1_4,
        kt4_W_2_2,kt4_W_2_3,kt4_W_2_4,
        kt4_W_3_2,kt4_W_3_3,kt4_W_3_4,
        kt4_Y_2,kt4_Y_3,kt4_Y_4,
        gw,gy,gp2,vev,lambda,dx)
        synchronize()

        #Compute k_5 arrays
        @cuda threads=thrds blocks=blks rk4_step!(ϕ_1+k1_ϕ_1 .*dt*a51+k2_ϕ_1 .*dt*a52+k3_ϕ_1 .*dt*a53+k4_ϕ_1 .*dt*a54,ϕ_2+k1_ϕ_2 .*dt*a51+k2_ϕ_2 .*dt*a52+k3_ϕ_2 .*dt*a53+k4_ϕ_2 .*dt*a54,ϕ_3+k1_ϕ_3 .*dt*a51+k2_ϕ_3 .*dt*a52+k3_ϕ_3 .*dt*a53+k4_ϕ_3 .*dt*a54,ϕ_4+k1_ϕ_4 .*dt*a51+k2_ϕ_4 .*dt*a52+k3_ϕ_4 .*dt*a53+k4_ϕ_4 .*dt*a54,
        W_1_2+k1_W_1_2 .*dt*a51+k2_W_1_2 .*dt*a52+k3_W_1_2 .*dt*a53+k4_W_1_2 .*dt*a54,W_1_3+k1_W_1_3 .*dt*a51+k2_W_1_3 .*dt*a52+k3_W_1_3 .*dt*a53+k4_W_1_3 .*dt*a54,W_1_4+k1_W_1_4 .*dt*a51+k2_W_1_4 .*dt*a52+k3_W_1_4 .*dt*a53+k4_W_1_4 .*dt*a54,
        W_2_2+k1_W_2_2 .*dt*a51+k2_W_2_2 .*dt*a52+k3_W_2_2 .*dt*a53+k4_W_2_2 .*dt*a54,W_2_3+k1_W_2_3 .*dt*a51+k2_W_2_3 .*dt*a52+k3_W_2_3 .*dt*a53+k4_W_2_3 .*dt*a54,W_2_4+k1_W_2_4 .*dt*a51+k2_W_2_4 .*dt*a52+k3_W_2_4 .*dt*a53+k4_W_2_4 .*dt*a54,
        W_3_2+k1_W_3_2 .*dt*a51+k2_W_3_2 .*dt*a52+k3_W_3_2 .*dt*a53+k4_W_3_2 .*dt*a54,W_3_3+k1_W_3_3 .*dt*a51+k2_W_3_3 .*dt*a52+k3_W_3_3 .*dt*a53+k4_W_3_3 .*dt*a54,W_3_4+k1_W_3_4 .*dt*a51+k2_W_3_4 .*dt*a52+k3_W_3_4 .*dt*a53+k4_W_3_4 .*dt*a54,
        Y_2+k1_Y_2 .*dt*a51+k2_Y_2 .*dt*a52+k3_Y_2 .*dt*a53+k4_Y_2 .*dt*a54,Y_3+k1_Y_3 .*dt*a51+k2_Y_3 .*dt*a52+k3_Y_3 .*dt*a53+k4_Y_3 .*dt*a54,Y_4+k1_Y_4 .*dt*a51+k2_Y_4 .*dt*a52+k3_Y_4 .*dt*a53+k4_Y_4 .*dt*a54,
        Γ_1+k1_Γ_1 .*dt*a51+k2_Γ_1 .*dt*a52+k3_Γ_1 .*dt*a53+k4_Γ_1 .*dt*a54,Γ_2+k1_Γ_2 .*dt*a51+k2_Γ_2 .*dt*a52+k3_Γ_2 .*dt*a53+k4_Γ_2 .*dt*a54,Γ_3+k1_Γ_3 .*dt*a51+k2_Γ_3 .*dt*a52+k3_Γ_3 .*dt*a53+k4_Γ_3 .*dt*a54,Σ+k1_Σ .*dt*a51+k2_Σ .*dt*a52+k3_Σ .*dt*a53+k4_Σ .*dt*a54,
        k5_ϕ_1,k5_ϕ_2,k5_ϕ_3,k5_ϕ_4,
        k5_W_1_2,k5_W_1_3,k5_W_1_4,
        k5_W_2_2,k5_W_2_3,k5_W_2_4,
        k5_W_3_2,k5_W_3_3,k5_W_3_4,
        k5_Y_2,k5_Y_3,k5_Y_4,
        k5_Γ_1,k5_Γ_2,k5_Γ_3,k5_Σ,
        dϕ_1_dt+kt1_ϕ_1 .*dt*a51+kt2_ϕ_1 .*dt*a52+kt3_ϕ_1 .*dt*a53+kt4_ϕ_1 .*dt*a54,dϕ_2_dt+kt1_ϕ_2 .*dt*a51+kt2_ϕ_2 .*dt*a52+kt3_ϕ_2 .*dt*a53+kt4_ϕ_2 .*dt*a54,dϕ_3_dt+kt1_ϕ_3 .*dt*a51+kt2_ϕ_3 .*dt*a52+kt3_ϕ_3 .*dt*a53+kt4_ϕ_3 .*dt*a54,dϕ_4_dt+kt1_ϕ_4 .*dt*a51+kt2_ϕ_4 .*dt*a52+kt3_ϕ_4 .*dt*a53+kt4_ϕ_4 .*dt*a54,
        dW_1_2_dt+kt1_W_1_2 .*dt*a51+kt2_W_1_2 .*dt*a52+kt3_W_1_2 .*dt*a53+kt4_W_1_2 .*dt*a54,dW_1_3_dt+kt1_W_1_3 .*dt*a51+kt2_W_1_3 .*dt*a52+kt3_W_1_3 .*dt*a53+kt4_W_1_3 .*dt*a54,dW_1_4_dt+kt1_W_1_4 .*dt*a51+kt2_W_1_4 .*dt*a52+kt3_W_1_4 .*dt*a53+kt4_W_1_4 .*dt*a54,
        dW_2_2_dt+kt1_W_2_2 .*dt*a51+kt2_W_2_2 .*dt*a52+kt3_W_2_2 .*dt*a53+kt4_W_2_2 .*dt*a54,dW_2_3_dt+kt1_W_2_3 .*dt*a51+kt2_W_2_3 .*dt*a52+kt3_W_2_3 .*dt*a53+kt4_W_2_3 .*dt*a54,dW_2_4_dt+kt1_W_2_4 .*dt*a51+kt2_W_2_4 .*dt*a52+kt3_W_2_4 .*dt*a53+kt4_W_2_4 .*dt*a54,
        dW_3_2_dt+kt1_W_3_2 .*dt*a51+kt2_W_3_2 .*dt*a52+kt3_W_3_2 .*dt*a53+kt4_W_3_2 .*dt*a54,dW_3_3_dt+kt1_W_3_3 .*dt*a51+kt2_W_3_3 .*dt*a52+kt3_W_3_3 .*dt*a53+kt4_W_3_3 .*dt*a54,dW_3_4_dt+kt1_W_3_4 .*dt*a51+kt2_W_3_4 .*dt*a52+kt3_W_3_4 .*dt*a53+kt4_W_3_4 .*dt*a54,
        dY_2_dt+kt1_Y_2 .*dt*a51+kt2_Y_2 .*dt*a52+kt3_Y_2 .*dt*a53+kt4_Y_2 .*dt*a54,dY_3_dt+kt1_Y_3 .*dt*a51+kt2_Y_3 .*dt*a52+kt3_Y_3 .*dt*a53+kt4_Y_3 .*dt*a54,dY_4_dt+kt1_Y_4 .*dt*a51+kt2_Y_4 .*dt*a52+kt3_Y_4 .*dt*a53+kt4_Y_4 .*dt*a54,
        kt5_ϕ_1,kt5_ϕ_2,kt5_ϕ_3,kt5_ϕ_4,
        kt5_W_1_2,kt5_W_1_3,kt5_W_1_4,
        kt5_W_2_2,kt5_W_2_3,kt5_W_2_4,
        kt5_W_3_2,kt5_W_3_3,kt5_W_3_4,
        kt5_Y_2,kt5_Y_3,kt5_Y_4,
        gw,gy,gp2,vev,lambda,dx)
        synchronize()

        #Compute k_6 arrays
        @cuda threads=thrds blocks=blks rk4_step!(ϕ_1+k1_ϕ_1 .*dt*a61+k2_ϕ_1 .*dt*a62+k3_ϕ_1 .*dt*a63+k4_ϕ_1 .*dt*a64+k5_ϕ_1 .*dt*a65,ϕ_2+k1_ϕ_2 .*dt*a61+k2_ϕ_2 .*dt*a62+k3_ϕ_2 .*dt*a63+k4_ϕ_2 .*dt*a64+k5_ϕ_2 .*dt*a65,ϕ_3+k1_ϕ_3 .*dt*a61+k2_ϕ_3 .*dt*a62+k3_ϕ_3 .*dt*a63+k4_ϕ_3 .*dt*a64+k5_ϕ_3 .*dt*a65,ϕ_4+k1_ϕ_4 .*dt*a61+k2_ϕ_4 .*dt*a62+k3_ϕ_4 .*dt*a63+k4_ϕ_4 .*dt*a64+k5_ϕ_4 .*dt*a65,
        W_1_2+k1_W_1_2 .*dt*a61+k2_W_1_2 .*dt*a62+k3_W_1_2 .*dt*a63+k4_W_1_2 .*dt*a64+k5_W_1_2 .*dt*a65,W_1_3+k1_W_1_3 .*dt*a61+k2_W_1_3 .*dt*a62+k3_W_1_3 .*dt*a63+k4_W_1_3 .*dt*a64+k5_W_1_3 .*dt*a65,W_1_4+k1_W_1_4 .*dt*a61+k2_W_1_4 .*dt*a62+k3_W_1_4 .*dt*a63+k4_W_1_4 .*dt*a64+k5_W_1_4 .*dt*a65,
        W_2_2+k1_W_2_2 .*dt*a61+k2_W_2_2 .*dt*a62+k3_W_2_2 .*dt*a63+k4_W_2_2 .*dt*a64+k5_W_2_2 .*dt*a65,W_2_3+k1_W_2_3 .*dt*a61+k2_W_2_3 .*dt*a62+k3_W_2_3 .*dt*a63+k4_W_2_3 .*dt*a64+k5_W_2_3 .*dt*a65,W_2_4+k1_W_2_4 .*dt*a61+k2_W_2_4 .*dt*a62+k3_W_2_4 .*dt*a63+k4_W_2_4 .*dt*a64+k5_W_2_4 .*dt*a65,
        W_3_2+k1_W_3_2 .*dt*a61+k2_W_3_2 .*dt*a62+k3_W_3_2 .*dt*a63+k4_W_3_2 .*dt*a64+k5_W_3_2 .*dt*a65,W_3_3+k1_W_3_3 .*dt*a61+k2_W_3_3 .*dt*a62+k3_W_3_3 .*dt*a63+k4_W_3_3 .*dt*a64+k5_W_3_3 .*dt*a65,W_3_4+k1_W_3_4 .*dt*a61+k2_W_3_4 .*dt*a62+k3_W_3_4 .*dt*a63+k4_W_3_4 .*dt*a64+k5_W_3_4 .*dt*a65,
        Y_2+k1_Y_2 .*dt*a61+k2_Y_2 .*dt*a62+k3_Y_2 .*dt*a63+k4_Y_2 .*dt*a64+k4_Y_2 .*dt*a65,Y_3+k1_Y_3 .*dt*a61+k2_Y_3 .*dt*a62+k3_Y_3 .*dt*a63+k4_Y_3 .*dt*a64+k4_Y_3 .*dt*a65,Y_4+k1_Y_4 .*dt*a61+k2_Y_4 .*dt*a62+k3_Y_4 .*dt*a63+k4_Y_4 .*dt*a64+k5_Y_4 .*dt*a65,
        Γ_1+k1_Γ_1 .*dt*a61+k2_Γ_1 .*dt*a62+k3_Γ_1 .*dt*a63+k4_Γ_1 .*dt*a64+k4_Γ_1 .*dt*a65,Γ_2+k1_Γ_2 .*dt*a61+k2_Γ_2 .*dt*a62+k3_Γ_2 .*dt*a63+k4_Γ_2 .*dt*a64+k4_Γ_2 .*dt*a65,Γ_3+k1_Γ_3 .*dt*a61+k2_Γ_3 .*dt*a62+k3_Γ_3 .*dt*a63+k4_Γ_3 .*dt*a64+k5_Γ_3 .*dt*a65,Σ+k1_Σ .*dt*a61+k2_Σ .*dt*a62+k3_Σ .*dt*a63+k4_Σ .*dt*a64+k5_Σ .*dt*a65,
        k6_ϕ_1,k6_ϕ_2,k6_ϕ_3,k6_ϕ_4,
        k6_W_1_2,k6_W_1_3,k6_W_1_4,
        k6_W_2_2,k6_W_2_3,k6_W_2_4,
        k6_W_3_2,k6_W_3_3,k6_W_3_4,
        k6_Y_2,k6_Y_3,k6_Y_4,
        k6_Γ_1,k6_Γ_2,k6_Γ_3,k6_Σ,
        dϕ_1_dt+kt1_ϕ_1 .*dt*a61+kt2_ϕ_1 .*dt*a62+kt3_ϕ_1 .*dt*a63+kt4_ϕ_1 .*dt*a64+kt5_ϕ_1 .*dt*a65,dϕ_2_dt+kt1_ϕ_2 .*dt*a61+kt2_ϕ_2 .*dt*a62+kt3_ϕ_2 .*dt*a63+kt4_ϕ_2 .*dt*a64+kt5_ϕ_2 .*dt*a65,dϕ_3_dt+kt1_ϕ_3 .*dt*a61+kt2_ϕ_3 .*dt*a62+kt3_ϕ_3 .*dt*a63+kt4_ϕ_3 .*dt*a64+kt5_ϕ_3 .*dt*a65,dϕ_4_dt+kt1_ϕ_4 .*dt*a61+kt2_ϕ_4 .*dt*a62+kt3_ϕ_4 .*dt*a63+kt4_ϕ_4 .*dt*a64+kt5_ϕ_4 .*dt*a65,
        dW_1_2_dt+kt1_W_1_2 .*dt*a61+kt2_W_1_2 .*dt*a62+kt3_W_1_2 .*dt*a63+kt4_W_1_2 .*dt*a64+kt5_W_1_2 .*dt*a65,dW_1_3_dt+kt1_W_1_3 .*dt*a61+kt2_W_1_3 .*dt*a62+kt3_W_1_3 .*dt*a63+kt4_W_1_3 .*dt*a64+kt5_W_1_3 .*dt*a65,dW_1_4_dt+kt1_W_1_4 .*dt*a61+kt2_W_1_4 .*dt*a62+kt3_W_1_4 .*dt*a63+kt4_W_1_4 .*dt*a64+kt5_W_1_4 .*dt*a65,
        dW_2_2_dt+kt1_W_2_2 .*dt*a61+kt2_W_2_2 .*dt*a62+kt3_W_2_2 .*dt*a63+kt4_W_2_2 .*dt*a64+kt5_W_2_2 .*dt*a65,dW_2_3_dt+kt1_W_2_3 .*dt*a61+kt2_W_2_3 .*dt*a62+kt3_W_2_3 .*dt*a63+kt4_W_2_3 .*dt*a64+kt5_W_2_3 .*dt*a65,dW_2_4_dt+kt1_W_2_4 .*dt*a61+kt2_W_2_4 .*dt*a62+kt3_W_2_4 .*dt*a63+kt4_W_2_4 .*dt*a64+kt5_W_2_4 .*dt*a65,
        dW_3_2_dt+kt1_W_3_2 .*dt*a61+kt2_W_3_2 .*dt*a62+kt3_W_3_2 .*dt*a63+kt4_W_3_2 .*dt*a64+kt5_W_3_2 .*dt*a65,dW_3_3_dt+kt1_W_3_3 .*dt*a61+kt2_W_3_3 .*dt*a62+kt3_W_3_3 .*dt*a63+kt4_W_3_3 .*dt*a64+kt5_W_3_3 .*dt*a65,dW_3_4_dt+kt1_W_3_4 .*dt*a61+kt2_W_3_4 .*dt*a62+kt3_W_3_4 .*dt*a63+kt4_W_3_4 .*dt*a64+kt5_W_3_4 .*dt*a65,
        dY_2_dt+kt1_Y_2 .*dt*a61+kt2_Y_2 .*dt*a62+kt3_Y_2 .*dt*a63+kt4_Y_2 .*dt*a64+kt4_Y_2 .*dt*a65,dY_3_dt+kt1_Y_3 .*dt*a61+kt2_Y_3 .*dt*a62+kt3_Y_3 .*dt*a63+kt4_Y_3 .*dt*a64+kt5_Y_3 .*dt*a65,dY_4_dt+kt1_Y_4 .*dt*a61+kt2_Y_4 .*dt*a62+kt3_Y_4 .*dt*a63+kt4_Y_4 .*dt*a64+kt5_Y_4 .*dt*a65,
        kt6_ϕ_1,kt6_ϕ_2,kt6_ϕ_3,kt6_ϕ_4,
        kt6_W_1_2,kt6_W_1_3,kt6_W_1_4,
        kt6_W_2_2,kt6_W_2_3,kt6_W_2_4,
        kt6_W_3_2,kt6_W_3_3,kt6_W_3_4,
        kt6_Y_2,kt6_Y_3,kt6_Y_4,
        gw,gy,gp2,vev,lambda,dx)
        synchronize()
        
        # end

        # Error compute redundant expressions
        
        #Update expressions
        # Array(ϕ_1+dt*b1*k1_ϕ_1+dt*b3*k3_ϕ_1+dt*b4*k4_ϕ_1+dt*b5*k5_ϕ_1)
        # Array(ϕ_2+dt*b1*k1_ϕ_2+dt*b3*k3_ϕ_2+dt*b4*k4_ϕ_2+dt*b5*k5_ϕ_2) 
        # Array(ϕ_3+dt*b1*k1_ϕ_3+dt*b3*k3_ϕ_3+dt*b4*k4_ϕ_3+dt*b5*k5_ϕ_3) 
        # Array(ϕ_4+dt*b1*k1_ϕ_4+dt*b3*k3_ϕ_4+dt*b4*k4_ϕ_4+dt*b5*k5_ϕ_4) 

        # Array(W_1_1+dt*b1*k1_W_1_1+dt*b3*k3_W_1_1+dt*b4*k4_W_1_1+dt*b5*k5_W_1_1) 
        # Array(W_1_2+dt*b1*k1_W_1_2+dt*b3*k3_W_1_2+dt*b4*k4_W_1_2+dt*b5*k5_W_1_2) 
        # Array(W_1_3+dt*b1*k1_W_1_3+dt*b3*k3_W_1_3+dt*b4*k4_W_1_3+dt*b5*k5_W_1_3) 
        # Array(W_1_4+dt*b1*k1_W_1_4+dt*b3*k3_W_1_4+dt*b4*k4_W_1_4+dt*b5*k5_W_1_4) 
        # Array(W_2_1+dt*b1*k1_W_2_1+dt*b3*k3_W_2_1+dt*b4*k4_W_2_1+dt*b5*k5_W_2_1) 
        # Array(W_2_2+dt*b1*k1_W_2_2+dt*b3*k3_W_2_2+dt*b4*k4_W_2_2+dt*b5*k5_W_2_2) 
        # Array(W_2_3+dt*b1*k1_W_2_3+dt*b3*k3_W_2_3+dt*b4*k4_W_2_3+dt*b5*k5_W_2_3) 
        # Array(W_2_4+dt*b1*k1_W_2_4+dt*b3*k3_W_2_4+dt*b4*k4_W_2_4+dt*b5*k5_W_2_4) 
        # Array(W_3_1+dt*b1*k1_W_3_1+dt*b3*k3_W_3_1+dt*b4*k4_W_3_1+dt*b5*k5_W_3_1) 
        # Array(W_3_2+dt*b1*k1_W_3_2+dt*b3*k3_W_3_2+dt*b4*k4_W_3_2+dt*b5*k5_W_3_2) 
        # Array(W_3_3+dt*b1*k1_W_3_3+dt*b3*k3_W_3_3+dt*b4*k4_W_3_3+dt*b5*k5_W_3_3) 
        # Array(W_3_4+dt*b1*k1_W_3_4+dt*b3*k3_W_3_4+dt*b4*k4_W_3_4+dt*b5*k5_W_3_4) 
        # Array(Y_1+dt*b1*k1_Y_1+dt*b3*k3_Y_1+dt*b4*k4_Y_1+dt*b5*k5_Y_1) 
        # Array(Y_2+dt*b1*k1_Y_2+dt*b3*k3_Y_2+dt*b4*k4_Y_2+dt*b5*k5_Y_2) 
        # Array(Y_3+dt*b1*k1_Y_3+dt*b3*k3_Y_3+dt*b4*k4_Y_3+dt*b5*k5_Y_3) 
        # Array(Y_4+dt*b1*k1_Y_4+dt*b3*k3_Y_4+dt*b4*k4_Y_4+dt*b5*k5_Y_4) 
        # Array(Γ_1+dt*b1*k1_Γ_1+dt*b3*k3_Γ_1+dt*b4*k4_Γ_1+dt*b5*k5_Γ_1) 
        # Array(Γ_2+dt*b1*k1_Γ_2+dt*b3*k3_Γ_2+dt*b4*k4_Γ_2+dt*b5*k5_Γ_2) 
        # Array(Γ_3+dt*b1*k1_Γ_3+dt*b3*k3_Γ_3+dt*b4*k4_Γ_3+dt*b5*k5_Γ_3) 
        # Array(Σ+dt*b1*k1_Σ+dt*b3*k3_Σ+dt*b4*k4_Σ+dt*b5*k5_Σ)

        # Array(dϕ_1_dt+dt*b1*kt1_ϕ_1+dt*b3*kt3_ϕ_1+dt*b4*kt4_ϕ_1+dt*b5*kt5_ϕ_1)
        # Array(dϕ_2_dt+dt*b1*kt1_ϕ_2+dt*b3*kt3_ϕ_2+dt*b4*kt4_ϕ_2+dt*b5*kt5_ϕ_2) 
        # Array(dϕ_3_dt+dt*b1*kt1_ϕ_3+dt*b3*kt3_ϕ_3+dt*b4*kt4_ϕ_3+dt*b5*kt5_ϕ_3) 
        # Array(dϕ_4_dt+dt*b1*kt1_ϕ_4+dt*b3*kt3_ϕ_4+dt*b4*kt4_ϕ_4+dt*b5*kt5_ϕ_4) 

        # # Array(W_1_1+dt*b1*kt1_W_1_1+dt*b3*kt3_W_1_1+dt*b4*kt4_W_1_1+dt*b5*kt5_W_1_1) 
        # Array(dW_1_2_dt+dt*b1*kt1_W_1_2+dt*b3*kt3_W_1_2+dt*b4*kt4_W_1_2+dt*b5*kt5_W_1_2) 
        # Array(dW_1_3_dt+dt*b1*kt1_W_1_3+dt*b3*kt3_W_1_3+dt*b4*kt4_W_1_3+dt*b5*kt5_W_1_3) 
        # Array(dW_1_4_dt+dt*b1*kt1_W_1_4+dt*b3*kt3_W_1_4+dt*b4*kt4_W_1_4+dt*b5*kt5_W_1_4) 
        # # Array(W_2_1+dt*b1*kt1_W_2_1+dt*b3*kt3_W_2_1+dt*b4*kt4_W_2_1+dt*b5*kt5_W_1_1) 
        # Array(dW_2_2_dt+dt*b1*kt1_W_2_2+dt*b3*kt3_W_2_2+dt*b4*kt4_W_2_2+dt*b5*kt5_W_2_2) 
        # Array(dW_2_3_dt+dt*b1*kt1_W_2_3+dt*b3*kt3_W_2_3+dt*b4*kt4_W_2_3+dt*b5*kt5_W_2_3) 
        # Array(dW_2_4_dt+dt*b1*kt1_W_2_4+dt*b3*kt3_W_2_4+dt*b4*kt4_W_2_4+dt*b5*kt5_W_2_4) 
        # # Array(W_3_1+dt*b1*kt1_W_3_1+dt*b3*kt3_W_3_1+dt*b4*kt4_W_3_1+dt*b5*kt5_W_1_1) 
        # Array(dW_3_2_dt+dt*b1*kt1_W_3_2+dt*b3*kt3_W_3_2+dt*b4*kt4_W_3_2+dt*b5*kt5_W_3_2) 
        # Array(dW_3_3_dt+dt*b1*kt1_W_3_3+dt*b3*kt3_W_3_3+dt*b4*kt4_W_3_3+dt*b5*kt5_W_3_3) 
        # Array(dW_3_4_dt+dt*b1*kt1_W_3_4+dt*b3*kt3_W_3_4+dt*b4*kt4_W_3_4+dt*b5*kt5_W_3_4) 
        # # Array(Y_1+dt*b1*kt1_Y_1+dt*b3*kt3_Y_1+dt*b4*kt4_Y_1+dt*b5*kt5_Y_1) 
        # Array(dY_2_dt+dt*b1*kt1_Y_2+dt*b3*kt3_Y_2+dt*b4*kt4_Y_2+dt*b5*kt5_Y_2) 
        # Array(dY_3_dt+dt*b1*kt1_Y_3+dt*b3*kt3_Y_3+dt*b4*kt4_Y_3+dt*b5*kt5_Y_3) 
        # Array(dY_4_dt+dt*b1*kt1_Y_4+dt*b3*kt3_Y_4+dt*b4*kt4_Y_4+dt*b5*kt5_Y_4) 
        # # Array(Γ_1+dt*b1*kt1_Γ_1+dt*b3*kt3_Γ_1+dt*b4*kt4_Γ_1+dt*b5*kt5_Γ_1) 
        # # Array(Γ_2+dt*b1*kt1_Γ_2+dt*b3*kt3_Γ_2+dt*b4*kt4_Γ_2+dt*b5*kt5_Γ_2) 
        # # Array(Γ_3+dt*b1*kt1_Γ_3+dt*b3*kt3_Γ_3+dt*b4*kt4_Γ_3+dt*b5*kt5_Γ_3) 
        # # Array(Σ+dt*b1*kt1_Σ+dt*b3*kt3_Σ+dt*b4*kt4_Σ+dt*b5*kt5_Σ)

        # Array(ϕ_1+dt*b1*k1_ϕ_1+dt*b3*k3_ϕ_1+dt*b4*k4_ϕ_1+dt*b5*k5_ϕ_1+dt*b6*k6_ϕ_1)
        # Array(ϕ_2+dt*b1*k1_ϕ_2+dt*b3*k3_ϕ_2+dt*b4*k4_ϕ_2+dt*b5*k5_ϕ_2+dt*b6*k6_ϕ_2) 
        # Array(ϕ_3+dt*b1*k1_ϕ_3+dt*b3*k3_ϕ_3+dt*b4*k4_ϕ_3+dt*b5*k5_ϕ_3+dt*b6*k6_ϕ_3) 
        # Array(ϕ_4+dt*b1*k1_ϕ_4+dt*b3*k3_ϕ_4+dt*b4*k4_ϕ_4+dt*b5*k5_ϕ_4+dt*b6*k6_ϕ_4) 

        # Array(W_1_1+dt*b1*k1_W_1_1+dt*b3*k3_W_1_1+dt*b4*k4_W_1_1+dt*b5*k5_W_1_1+dt*b6*k6_W_1_1) 
        # Array(W_1_2+dt*b1*k1_W_1_2+dt*b3*k3_W_1_2+dt*b4*k4_W_1_2+dt*b5*k5_W_1_2+dt*b6*k6_W_1_2) 
        # Array(W_1_3+dt*b1*k1_W_1_3+dt*b3*k3_W_1_3+dt*b4*k4_W_1_3+dt*b5*k5_W_1_3+dt*b6*k6_W_1_3) 
        # Array(W_1_4+dt*b1*k1_W_1_4+dt*b3*k3_W_1_4+dt*b4*k4_W_1_4+dt*b5*k5_W_1_4+dt*b6*k6_W_2_4) 
        # Array(W_2_1+dt*b1*k1_W_2_1+dt*b3*k3_W_2_1+dt*b4*k4_W_2_1+dt*b5*k5_W_2_1+dt*b6*k6_W_2_1) 
        # Array(W_2_2+dt*b1*k1_W_2_2+dt*b3*k3_W_2_2+dt*b4*k4_W_2_2+dt*b5*k5_W_2_2+dt*b6*k6_W_2_2) 
        # Array(W_2_3+dt*b1*k1_W_2_3+dt*b3*k3_W_2_3+dt*b4*k4_W_2_3+dt*b5*k5_W_2_3+dt*b6*k6_W_2_3) 
        # Array(W_2_4+dt*b1*k1_W_2_4+dt*b3*k3_W_2_4+dt*b4*k4_W_2_4+dt*b5*k5_W_2_4+dt*b6*k6_W_2_4) 
        # Array(W_3_1+dt*b1*k1_W_3_1+dt*b3*k3_W_3_1+dt*b4*k4_W_3_1+dt*b5*k5_W_3_1+dt*b6*k6_W_3_1) 
        # Array(W_3_2+dt*b1*k1_W_3_2+dt*b3*k3_W_3_2+dt*b4*k4_W_3_2+dt*b5*k5_W_3_2+dt*b6*k6_W_3_2) 
        # Array(W_3_3+dt*b1*k1_W_3_3+dt*b3*k3_W_3_3+dt*b4*k4_W_3_3+dt*b5*k5_W_3_3+dt*b6*k6_W_3_3) 
        # Array(W_3_4+dt*b1*k1_W_3_4+dt*b3*k3_W_3_4+dt*b4*k4_W_3_4+dt*b5*k5_W_3_4+dt*b6*k6_W_3_4) 
        # Array(Y_1+dt*b1*k1_Y_1+dt*b3*k3_Y_1+dt*b4*k4_Y_1+dt*b5*k5_Y_1+dt*b6*k6_Y_1) 
        # Array(Y_2+dt*b1*k1_Y_2+dt*b3*k3_Y_2+dt*b4*k4_Y_2+dt*b5*k5_Y_2+dt*b6*k6_Y_2) 
        # Array(Y_3+dt*b1*k1_Y_3+dt*b3*k3_Y_3+dt*b4*k4_Y_3+dt*b5*k5_Y_3+dt*b6*k6_Y_3) 
        # Array(Y_4+dt*b1*k1_Y_4+dt*b3*k3_Y_4+dt*b4*k4_Y_4+dt*b5*k5_Y_4+dt*b6*k6_Y_4) 
        # Array(Γ_1+dt*b1*k1_Γ_1+dt*b3*k3_Γ_1+dt*b4*k4_Γ_1+dt*b5*k5_Γ_1+dt*b6*k6_Γ_1) 
        # Array(Γ_2+dt*b1*k1_Γ_2+dt*b3*k3_Γ_2+dt*b4*k4_Γ_2+dt*b5*k5_Γ_2+dt*b6*k6_Γ_2) 
        # Array(Γ_3+dt*b1*k1_Γ_3+dt*b3*k3_Γ_3+dt*b4*k4_Γ_3+dt*b5*k5_Γ_3+dt*b6*k6_Γ_3) 
        # Array(Σ+dt*b1*k1_Σ+dt*b3*k3_Σ+dt*b4*k4_Σ+dt*b5*k5_Σ+dt*b6*k6_Σ)

        # Array(dϕ_1_dt+dt*b1*kt1_ϕ_1+dt*b3*kt3_ϕ_1+dt*b4*kt4_ϕ_1+dt*b5*kt5_ϕ_1+dt*b6*kt6_ϕ_1)
        # Array(dϕ_2_dt+dt*b1*kt1_ϕ_2+dt*b3*kt3_ϕ_2+dt*b4*kt4_ϕ_2+dt*b5*kt5_ϕ_2+dt*b6*kt6_ϕ_2) 
        # Array(dϕ_3_dt+dt*b1*kt1_ϕ_3+dt*b3*kt3_ϕ_3+dt*b4*kt4_ϕ_3+dt*b5*kt5_ϕ_3+dt*b6*kt6_ϕ_3) 
        # Array(dϕ_4_dt+dt*b1*kt1_ϕ_4+dt*b3*kt3_ϕ_4+dt*b4*kt4_ϕ_4+dt*b5*kt5_ϕ_4+dt*b6*kt6_ϕ_4) 

        # # Array(W_1_1+dt*b1*kt1_W_1_1+dt*b3*kt3_W_1_1+dt*b4*kt4_W_1_1+dt*b5*kt5_W_1_1) 
        # Array(dW_1_2_dt+dt*b1*kt1_W_1_2+dt*b3*kt3_W_1_2+dt*b4*kt4_W_1_2+dt*b5*kt5_W_1_2+dt*b6*kt6_W_1_2) 
        # Array(dW_1_3_dt+dt*b1*kt1_W_1_3+dt*b3*kt3_W_1_3+dt*b4*kt4_W_1_3+dt*b5*kt5_W_1_3+dt*b6*kt6_W_1_3) 
        # Array(dW_1_4_dt+dt*b1*kt1_W_1_4+dt*b3*kt3_W_1_4+dt*b4*kt4_W_1_4+dt*b5*kt5_W_1_4+dt*b6*kt6_W_1_4) 
        # # Array(W_2_1+dt*b1*kt1_W_2_1+dt*b3*kt3_W_2_1+dt*b4*kt4_W_2_1+dt*b5*kt5_W_1_1) 
        # Array(dW_2_2_dt+dt*b1*kt1_W_2_2+dt*b3*kt3_W_2_2+dt*b4*kt4_W_2_2+dt*b5*kt5_W_2_2+dt*b6*kt6_W_2_2) 
        # Array(dW_2_3_dt+dt*b1*kt1_W_2_3+dt*b3*kt3_W_2_3+dt*b4*kt4_W_2_3+dt*b5*kt5_W_2_3+dt*b6*kt6_W_2_3) 
        # Array(dW_2_4_dt+dt*b1*kt1_W_2_4+dt*b3*kt3_W_2_4+dt*b4*kt4_W_2_4+dt*b5*kt5_W_2_4+dt*b6*kt6_W_2_4) 
        # # Array(W_3_1+dt*b1*kt1_W_3_1+dt*b3*kt3_W_3_1+dt*b4*kt4_W_3_1+dt*b5*kt5_W_1_1) 
        # Array(dW_3_2_dt+dt*b1*kt1_W_3_2+dt*b3*kt3_W_3_2+dt*b4*kt4_W_3_2+dt*b5*kt5_W_3_2+dt*b6*kt6_W_3_2) 
        # Array(dW_3_3_dt+dt*b1*kt1_W_3_3+dt*b3*kt3_W_3_3+dt*b4*kt4_W_3_3+dt*b5*kt5_W_3_3+dt*b6*kt6_W_3_3) 
        # Array(dW_3_4_dt+dt*b1*kt1_W_3_4+dt*b3*kt3_W_3_4+dt*b4*kt4_W_3_4+dt*b5*kt5_W_3_4+dt*b6*kt6_W_3_4) 
        # # Array(Y_1+dt*b1*kt1_Y_1+dt*b3*kt3_Y_1+dt*b4*kt4_Y_1+dt*b5*kt5_Y_1) 
        # Array(dY_2_dt+dt*b1*kt1_Y_2+dt*b3*kt3_Y_2+dt*b4*kt4_Y_2+dt*b5*kt5_Y_2+dt*b6*kt6_Y_2) 
        # Array(dY_3_dt+dt*b1*kt1_Y_3+dt*b3*kt3_Y_3+dt*b4*kt4_Y_3+dt*b5*kt5_Y_3+dt*b6*kt6_Y_3) 
        # Array(dY_4_dt+dt*b1*kt1_Y_4+dt*b3*kt3_Y_4+dt*b4*kt4_Y_4+dt*b5*kt5_Y_4+dt*b6*kt6_Y_4) 
        # # Array(Γ_1+dt*b1*kt1_Γ_1+dt*b3*kt3_Γ_1+dt*b4*kt4_Γ_1+dt*b5*kt5_Γ_1) 
        # # Array(Γ_2+dt*b1*kt1_Γ_2+dt*b3*kt3_Γ_2+dt*b4*kt4_Γ_2+dt*b5*kt5_Γ_2) 
        # # Array(Γ_3+dt*b1*kt1_Γ_3+dt*b3*kt3_Γ_3+dt*b4*kt4_Γ_3+dt*b5*kt5_Γ_3) 
        # # Array(Σ+dt*b1*kt1_Σ+dt*b3*kt3_Σ+dt*b4*kt4_Σ+dt*b5*kt5_Σ)

        @time begin
        err[1]=maximum(abs.(Array((ϕ_1+dt*c1*k1_ϕ_1+dt*c3*k3_ϕ_1+dt*c4*k4_ϕ_1+dt*c5*k5_ϕ_1+dt*c6*k6_ϕ_1) -(ϕ_1+dt*b1*k1_ϕ_1+dt*b3*k3_ϕ_1+dt*b4*k4_ϕ_1+dt*b5*k5_ϕ_1))))
        err[2]=maximum(abs.(Array((ϕ_2+dt*c1*k1_ϕ_2+dt*c3*k3_ϕ_2+dt*c4*k4_ϕ_2+dt*c5*k5_ϕ_2+dt*c6*k6_ϕ_2) -(ϕ_2+dt*b1*k1_ϕ_2+dt*b3*k3_ϕ_2+dt*b4*k4_ϕ_2+dt*b5*k5_ϕ_2))))
        err[3]=maximum(abs.(Array((ϕ_3+dt*c1*k1_ϕ_3+dt*c3*k3_ϕ_3+dt*c4*k4_ϕ_3+dt*c5*k5_ϕ_3+dt*c6*k6_ϕ_3) -(ϕ_3+dt*b1*k1_ϕ_3+dt*b3*k3_ϕ_3+dt*b4*k4_ϕ_3+dt*b5*k5_ϕ_3)))) 
        err[4]=maximum(abs.(Array((ϕ_4+dt*c1*k1_ϕ_4+dt*c3*k3_ϕ_4+dt*c4*k4_ϕ_4+dt*c5*k5_ϕ_4+dt*c6*k6_ϕ_4) -(ϕ_4+dt*b1*k1_ϕ_4+dt*b3*k3_ϕ_4+dt*b4*k4_ϕ_4+dt*b5*k5_ϕ_4)))) 

        # err[5]=maximum(abs.(Array((W_1_1+dt*c1*k1_W_1_1+dt*c3*k3_W_1_1+dt*c4*k4_W_1_1+dt*c5*k5_W_1_1+dt*c6*k6_W_1_1) -(W_1_1+dt*b1*k1_W_1_1+dt*b3*k3_W_1_1+dt*b4*k4_W_1_1+dt*b5*k5_W_1_1)))) 
        err[5]=maximum(abs.(Array((W_1_2+dt*c1*k1_W_1_2+dt*c3*k3_W_1_2+dt*c4*k4_W_1_2+dt*c5*k5_W_1_2+dt*c6*k6_W_1_2) -(W_1_2+dt*b1*k1_W_1_2+dt*b3*k3_W_1_2+dt*b4*k4_W_1_2+dt*b5*k5_W_1_2)))) 
        err[6]=maximum(abs.(Array((W_1_3+dt*c1*k1_W_1_3+dt*c3*k3_W_1_3+dt*c4*k4_W_1_3+dt*c5*k5_W_1_3+dt*c6*k6_W_1_3) -(W_1_3+dt*b1*k1_W_1_3+dt*b3*k3_W_1_3+dt*b4*k4_W_1_3+dt*b5*k5_W_1_3)))) 
        err[7]=maximum(abs.(Array((W_1_4+dt*c1*k1_W_1_4+dt*c3*k3_W_1_4+dt*c4*k4_W_1_4+dt*c5*k5_W_1_4+dt*c6*k6_W_1_4) -(W_1_4+dt*b1*k1_W_1_4+dt*b3*k3_W_1_4+dt*b4*k4_W_1_4+dt*b5*k5_W_1_4)))) 
        # err[9]=maximum(abs.(Array((W_2_1+dt*c1*k1_W_2_1+dt*c3*k3_W_2_1+dt*c4*k4_W_2_1+dt*c5*k5_W_2_1+dt*c6*k6_W_2_1) -(W_2_1+dt*b1*k1_W_2_1+dt*b3*k3_W_2_1+dt*b4*k4_W_2_1+dt*b5*k5_W_2_1)))) 
        err[8]=maximum(abs.(Array((W_2_2+dt*c1*k1_W_2_2+dt*c3*k3_W_2_2+dt*c4*k4_W_2_2+dt*c5*k5_W_2_2+dt*c6*k6_W_2_2) -(W_2_2+dt*b1*k1_W_2_2+dt*b3*k3_W_2_2+dt*b4*k4_W_2_2+dt*b5*k5_W_2_2)))) 
        err[9]=maximum(abs.(Array((W_2_3+dt*c1*k1_W_2_3+dt*c3*k3_W_2_3+dt*c4*k4_W_2_3+dt*c5*k5_W_2_3+dt*c6*k6_W_2_3) -(W_2_3+dt*b1*k1_W_2_3+dt*b3*k3_W_2_3+dt*b4*k4_W_2_3+dt*b5*k5_W_2_3)))) 
        err[10]=maximum(abs.(Array((W_2_4+dt*c1*k1_W_2_4+dt*c3*k3_W_2_4+dt*c4*k4_W_2_4+dt*c5*k5_W_2_4+dt*c6*k6_W_2_4) -(W_2_4+dt*b1*k1_W_2_4+dt*b3*k3_W_2_4+dt*b4*k4_W_2_4+dt*b5*k5_W_2_4)))) 
        # err[13]=maximum(abs.(Array((W_3_1+dt*c1*k1_W_3_1+dt*c3*k3_W_3_1+dt*c4*k4_W_3_1+dt*c5*k5_W_3_1+dt*c6*k6_W_3_1) -(W_3_1+dt*b1*k1_W_3_1+dt*b3*k3_W_3_1+dt*b4*k4_W_3_1+dt*b5*k5_W_3_1)))) 
        err[11]=maximum(abs.(Array((W_3_2+dt*c1*k1_W_3_2+dt*c3*k3_W_3_2+dt*c4*k4_W_3_2+dt*c5*k5_W_3_2+dt*c6*k6_W_3_2) -(W_3_2+dt*b1*k1_W_3_2+dt*b3*k3_W_3_2+dt*b4*k4_W_3_2+dt*b5*k5_W_3_2)))) 
        err[12]=maximum(abs.(Array((W_3_3+dt*c1*k1_W_3_3+dt*c3*k3_W_3_3+dt*c4*k4_W_3_3+dt*c5*k5_W_3_3+dt*c6*k6_W_3_3) -(W_3_3+dt*b1*k1_W_3_3+dt*b3*k3_W_3_3+dt*b4*k4_W_3_3+dt*b5*k5_W_3_3)))) 
        err[13]=maximum(abs.(Array((W_3_4+dt*c1*k1_W_3_4+dt*c3*k3_W_3_4+dt*c4*k4_W_3_4+dt*c5*k5_W_3_4+dt*c6*k6_W_3_4) -(W_3_4+dt*b1*k1_W_3_4+dt*b3*k3_W_3_4+dt*b4*k4_W_3_4+dt*b5*k5_W_3_4)))) 
        # err[17]=maximum(abs.(Array((Y_1+dt*c1*k1_Y_1+dt*c3*k3_Y_1+dt*c4*k4_Y_1+dt*c5*k5_Y_1+dt*c6*k6_Y_1) -(Y_1+dt*b1*k1_Y_1+dt*b3*k3_Y_1+dt*b4*k4_Y_1+dt*b5*k5_Y_1)))) 
        err[14]=maximum(abs.(Array((Y_2+dt*c1*k1_Y_2+dt*c3*k3_Y_2+dt*c4*k4_Y_2+dt*c5*k5_Y_2+dt*c6*k6_Y_2) -(Y_2+dt*b1*k1_Y_2+dt*b3*k3_Y_2+dt*b4*k4_Y_2+dt*b5*k5_Y_2)))) 
        err[15]=maximum(abs.(Array((Y_3+dt*c1*k1_Y_3+dt*c3*k3_Y_3+dt*c4*k4_Y_3+dt*c5*k5_Y_3+dt*c6*k6_Y_3) -(Y_3+dt*b1*k1_Y_3+dt*b3*k3_Y_3+dt*b4*k4_Y_3+dt*b5*k5_Y_3)))) 
        err[16]=maximum(abs.(Array((Y_4+dt*c1*k1_Y_4+dt*c3*k3_Y_4+dt*c4*k4_Y_4+dt*c5*k5_Y_4+dt*c6*k6_Y_4) -(Y_4+dt*b1*k1_Y_4+dt*b3*k3_Y_4+dt*b4*k4_Y_4+dt*b5*k5_Y_4)))) 
        err[17]=maximum(abs.(Array((Γ_1+dt*c1*k1_Γ_1+dt*c3*k3_Γ_1+dt*c4*k4_Γ_1+dt*c5*k5_Γ_1+dt*c6*k6_Γ_1) -(Γ_1+dt*b1*k1_Γ_1+dt*b3*k3_Γ_1+dt*b4*k4_Γ_1+dt*b5*k5_Γ_1)))) 
        err[18]=maximum(abs.(Array((Γ_2+dt*c1*k1_Γ_2+dt*c3*k3_Γ_2+dt*c4*k4_Γ_2+dt*c5*k5_Γ_2+dt*c6*k6_Γ_2) -(Γ_2+dt*b1*k1_Γ_2+dt*b3*k3_Γ_2+dt*b4*k4_Γ_2+dt*b5*k5_Γ_2)))) 
        err[19]=maximum(abs.(Array((Γ_3+dt*c1*k1_Γ_3+dt*c3*k3_Γ_3+dt*c4*k4_Γ_3+dt*c5*k5_Γ_3+dt*c6*k6_Γ_3) -(Γ_3+dt*b1*k1_Γ_3+dt*b3*k3_Γ_3+dt*b4*k4_Γ_3+dt*b5*k5_Γ_3)))) 
        err[20]=maximum(abs.(Array((Σ+dt*c1*k1_Σ+dt*c3*k3_Σ+dt*c4*k4_Σ+dt*c5*k5_Σ+dt*c6*k6_Σ)-(Σ+dt*b1*k1_Σ+dt*b3*k3_Σ+dt*b4*k4_Σ+dt*b5*k5_Σ))))

        err[21]=maximum(abs.(Array((dϕ_1_dt+dt*c1*kt1_ϕ_1+dt*c3*kt3_ϕ_1+dt*c4*kt4_ϕ_1+dt*c5*kt5_ϕ_1+dt*c6*kt6_ϕ_1)-(dϕ_1_dt+dt*b1*kt1_ϕ_1+dt*b3*kt3_ϕ_1+dt*b4*kt4_ϕ_1+dt*b5*kt5_ϕ_1))))
        err[22]=maximum(abs.(Array((dϕ_2_dt+dt*c1*kt1_ϕ_2+dt*c3*kt3_ϕ_2+dt*c4*kt4_ϕ_2+dt*c5*kt5_ϕ_2+dt*c6*kt6_ϕ_2) -(dϕ_2_dt+dt*b1*kt1_ϕ_2+dt*b3*kt3_ϕ_2+dt*b4*kt4_ϕ_2+dt*b5*kt5_ϕ_2)))) 
        err[23]=maximum(abs.(Array((dϕ_3_dt+dt*c1*kt1_ϕ_3+dt*c3*kt3_ϕ_3+dt*c4*kt4_ϕ_3+dt*c5*kt5_ϕ_3+dt*c6*kt6_ϕ_3) -(dϕ_3_dt+dt*b1*kt1_ϕ_3+dt*b3*kt3_ϕ_3+dt*b4*kt4_ϕ_3+dt*b5*kt5_ϕ_3)))) 
        err[24]=maximum(abs.(Array((dϕ_4_dt+dt*c1*kt1_ϕ_4+dt*c3*kt3_ϕ_4+dt*c4*kt4_ϕ_4+dt*c5*kt5_ϕ_4+dt*c6*kt6_ϕ_4) -(dϕ_4_dt+dt*b1*kt1_ϕ_4+dt*b3*kt3_ϕ_4+dt*b4*kt4_ϕ_4+dt*b5*kt5_ϕ_4)))) 

        # -(W_1_1+dt*b1*kt1_W_1_1+dt*b3*kt3_W_1_1+dt*b4*kt4_W_1_1+dt*b5*kt5_W_1_1) 
        err[25]=maximum(abs.(Array((dW_1_2_dt+dt*c1*kt1_W_1_2+dt*c3*kt3_W_1_2+dt*c4*kt4_W_1_2+dt*c5*kt5_W_1_2+dt*c6*kt6_W_1_2) -(dW_1_2_dt+dt*b1*kt1_W_1_2+dt*b3*kt3_W_1_2+dt*b4*kt4_W_1_2+dt*b5*kt5_W_1_2)))) 
        err[26]=maximum(abs.(Array((dW_1_3_dt+dt*c1*kt1_W_1_3+dt*c3*kt3_W_1_3+dt*c4*kt4_W_1_3+dt*c5*kt5_W_1_3+dt*c6*kt6_W_1_3) -(dW_1_3_dt+dt*b1*kt1_W_1_3+dt*b3*kt3_W_1_3+dt*b4*kt4_W_1_3+dt*b5*kt5_W_1_3)))) 
        err[27]=maximum(abs.(Array((dW_1_4_dt+dt*c1*kt1_W_1_4+dt*c3*kt3_W_1_4+dt*c4*kt4_W_1_4+dt*c5*kt5_W_1_4+dt*c6*kt6_W_1_4) -(dW_1_4_dt+dt*b1*kt1_W_1_4+dt*b3*kt3_W_1_4+dt*b4*kt4_W_1_4+dt*b5*kt5_W_1_4)))) 
        # -(W_2_1+dt*b1*kt1_W_2_1+dt*b3*kt3_W_2_1+dt*b4*kt4_W_2_1+dt*b5*kt5_W_1_1) 
        err[28]=maximum(abs.(Array((dW_2_2_dt+dt*c1*kt1_W_2_2+dt*c3*kt3_W_2_2+dt*c4*kt4_W_2_2+dt*c5*kt5_W_2_2+dt*c6*kt6_W_2_2) -(dW_2_2_dt+dt*b1*kt1_W_2_2+dt*b3*kt3_W_2_2+dt*b4*kt4_W_2_2+dt*b5*kt5_W_2_2)))) 
        err[29]=maximum(abs.(Array((dW_2_3_dt+dt*c1*kt1_W_2_3+dt*c3*kt3_W_2_3+dt*c4*kt4_W_2_3+dt*c5*kt5_W_2_3+dt*c6*kt6_W_2_3) -(dW_2_3_dt+dt*b1*kt1_W_2_3+dt*b3*kt3_W_2_3+dt*b4*kt4_W_2_3+dt*b5*kt5_W_2_3)))) 
        err[30]=maximum(abs.(Array((dW_2_4_dt+dt*c1*kt1_W_2_4+dt*c3*kt3_W_2_4+dt*c4*kt4_W_2_4+dt*c5*kt5_W_2_4+dt*c6*kt6_W_2_4) -(dW_2_4_dt+dt*b1*kt1_W_2_4+dt*b3*kt3_W_2_4+dt*b4*kt4_W_2_4+dt*b5*kt5_W_2_4)))) 
        # -(W_3_1+dt*b1*kt1_W_3_1+dt*b3*kt3_W_3_1+dt*b4*kt4_W_3_1+dt*b5*kt5_W_2_1) 
        err[31]=maximum(abs.(Array((dW_3_2_dt+dt*c1*kt1_W_3_2+dt*c3*kt3_W_3_2+dt*c4*kt4_W_3_2+dt*c5*kt5_W_3_2+dt*c6*kt6_W_3_2) -(dW_3_2_dt+dt*b1*kt1_W_3_2+dt*b3*kt3_W_3_2+dt*b4*kt4_W_3_2+dt*b5*kt5_W_3_2)))) 
        err[32]=maximum(abs.(Array((dW_3_3_dt+dt*c1*kt1_W_3_3+dt*c3*kt3_W_3_3+dt*c4*kt4_W_3_3+dt*c5*kt5_W_3_3+dt*c6*kt6_W_3_3) -(dW_3_3_dt+dt*b1*kt1_W_3_3+dt*b3*kt3_W_3_3+dt*b4*kt4_W_3_3+dt*b5*kt5_W_3_3)))) 
        err[33]=maximum(abs.(Array((dW_3_4_dt+dt*c1*kt1_W_3_4+dt*c3*kt3_W_3_4+dt*c4*kt4_W_3_4+dt*c5*kt5_W_3_4+dt*c6*kt6_W_3_4) -(dW_3_4_dt+dt*b1*kt1_W_3_4+dt*b3*kt3_W_3_4+dt*b4*kt4_W_3_4+dt*b5*kt5_W_3_4)))) 
        # -(Y_1+dt*b1*kt1_Y_1+dt*b3*kt3_Y_1+dt*b4*kt4_Y_1+dt*b5*kt5_Y_1) 
        err[34]=maximum(abs.(Array((dY_2_dt+dt*c1*kt1_Y_2+dt*c3*kt3_Y_2+dt*c4*kt4_Y_2+dt*c5*kt5_Y_2+dt*c6*kt6_Y_2) -(dY_2_dt+dt*b1*kt1_Y_2+dt*b3*kt3_Y_2+dt*b4*kt4_Y_2+dt*b5*kt5_Y_2)))) 
        err[35]=maximum(abs.(Array((dY_3_dt+dt*c1*kt1_Y_3+dt*c3*kt3_Y_3+dt*c4*kt4_Y_3+dt*c5*kt5_Y_3+dt*c6*kt6_Y_3) -(dY_3_dt+dt*b1*kt1_Y_3+dt*b3*kt3_Y_3+dt*b4*kt4_Y_3+dt*b5*kt5_Y_3)))) 
        err[36]=maximum(abs.(Array((dY_4_dt+dt*c1*kt1_Y_4+dt*c3*kt3_Y_4+dt*c4*kt4_Y_4+dt*c5*kt5_Y_4+dt*c6*kt6_Y_4) -(dY_4_dt+dt*b1*kt1_Y_4+dt*b3*kt3_Y_4+dt*b4*kt4_Y_4+dt*b5*kt5_Y_4)))) 
        # -(Γ_1+dt*b1*kt1_Γ_1+dt*b3*kt3_Γ_1+dt*b4*kt4_Γ_1+dt*b5*kt5_Γ_1) 
        # -(Γ_2+dt*b1*kt1_Γ_2+dt*b3*kt3_Γ_2+dt*b4*kt4_Γ_2+dt*b5*kt5_Γ_2) 
        # -(Γ_3+dt*b1*kt1_Γ_3+dt*b3*kt3_Γ_3+dt*b4*kt4_Γ_3+dt*b5*kt5_Γ_3) 
        # -(Σ+dt*b1*kt1_Σ+dt*b3*kt3_Σ+dt*b4*kt4_Σ+dt*b5*kt5_Σ)
        end
        MPI.Barrier(comm_cart)
        gather_metrics(err,err_g,me,comm_cart,nprocs)
        MPI.Barrier(comm_cart)

        max_err = maximum(abs.(err_g))
        max_mags = [maximum(abs.(Array(ϕ_1))),maximum(abs.(Array(ϕ_2))),maximum(abs.(Array(ϕ_3))),maximum(abs.(Array(ϕ_4))),
                    maximum(abs.(Array(W_1_2))),maximum(abs.(Array(W_1_3))),maximum(abs.(Array(W_1_4))),
                    maximum(abs.(Array(W_2_2))),maximum(abs.(Array(W_2_3))),maximum(abs.(Array(W_2_4))),
                    maximum(abs.(Array(W_3_2))),maximum(abs.(Array(W_3_3))),maximum(abs.(Array(W_3_4))),
                    maximum(abs.(Array(Y_2))),maximum(abs.(Array(Y_3))),maximum(abs.(Array(Y_4))),
                    maximum(abs.(Array(Γ_1))),maximum(abs.(Array(Γ_2))),maximum(abs.(Array(Γ_3))),maximum(abs.(Array(Σ))),
                    maximum(abs.(Array(dϕ_1_dt))),maximum(abs.(Array(dϕ_2_dt))),maximum(abs.(Array(dϕ_3_dt))),maximum(abs.(Array(dϕ_4_dt))),
                    maximum(abs.(Array(dW_1_2_dt))),maximum(abs.(Array(dW_1_3_dt))),maximum(abs.(Array(dW_1_4_dt))),
                    maximum(abs.(Array(dW_2_2_dt))),maximum(abs.(Array(dW_2_3_dt))),maximum(abs.(Array(dW_2_4_dt))),
                    maximum(abs.(Array(dW_3_2_dt))),maximum(abs.(Array(dW_3_3_dt))),maximum(abs.(Array(dW_3_4_dt))),
                    maximum(abs.(Array(dY_2_dt))),maximum(abs.(Array(dY_3_dt))),maximum(abs.(Array(dY_4_dt)))]
        
        MPI.Barrier(comm_cart)
        gather_metrics(max_mags,max_mags_g,me,comm_cart,nprocs)
        MPI.Barrier(comm_cart)

        # max_scale = a_tol+r_tol*(maximum(max_mags))
        max_scales = a_tol.+r_tol*max_mags_g
        err_ecl = sqrt((1/36)*sum((err_g/max_scales).^2))
        # if (max_err <= (tol*dt))
        if (err_ecl>1.0)
            # dt = dt*(tol*dt/(2.0*max_err))^(1/4)
            dt=sfty*dt*(1/err_ecl)^(1/5)
            println(string("step too large" ,dt))
        else
            println(string("step too small", it, dt, max_err, tol*dt))
            it= it+1
            # dt = dt*(tol*dt/(2.0*max_err))^(1/4)
            dt=sfty*dt*(1/err_ecl)^(1/5)
            
            #Update values
            # ϕ_1=ϕ_1+dt*b1*k1_ϕ_1+dt*b3*k3_ϕ_1+dt*b4*k4_ϕ_1+dt*b5*k5_ϕ_1
            # ϕ_2=ϕ_2+dt*b1*k1_ϕ_2+dt*b3*k3_ϕ_2+dt*b4*k4_ϕ_2+dt*b5*k5_ϕ_2 
            # ϕ_3=ϕ_3+dt*b1*k1_ϕ_3+dt*b3*k3_ϕ_3+dt*b4*k4_ϕ_3+dt*b5*k5_ϕ_3 
            # ϕ_4=ϕ_4+dt*b1*k1_ϕ_4+dt*b3*k3_ϕ_4+dt*b4*k4_ϕ_4+dt*b5*k5_ϕ_4 
    
            # # W_1_1=W_1_1+dt*b1*k1_W_1_1+dt*b3*k3_W_1_1+dt*b4*k4_W_1_1+dt*b5*k5_W_1_1 
            # W_1_2=W_1_2+dt*b1*k1_W_1_2+dt*b3*k3_W_1_2+dt*b4*k4_W_1_2+dt*b5*k5_W_1_2 
            # W_1_3=W_1_3+dt*b1*k1_W_1_3+dt*b3*k3_W_1_3+dt*b4*k4_W_1_3+dt*b5*k5_W_1_3 
            # W_1_4=W_1_4+dt*b1*k1_W_1_4+dt*b3*k3_W_1_4+dt*b4*k4_W_1_4+dt*b5*k5_W_1_4 
            # # W_2_1=W_2_1+dt*b1*k1_W_2_1+dt*b3*k3_W_2_1+dt*b4*k4_W_2_1+dt*b5*k5_W_2_1 
            # W_2_2=W_2_2+dt*b1*k1_W_2_2+dt*b3*k3_W_2_2+dt*b4*k4_W_2_2+dt*b5*k5_W_2_2 
            # W_2_3=W_2_3+dt*b1*k1_W_2_3+dt*b3*k3_W_2_3+dt*b4*k4_W_2_3+dt*b5*k5_W_2_3 
            # W_2_4=W_2_4+dt*b1*k1_W_2_4+dt*b3*k3_W_2_4+dt*b4*k4_W_2_4+dt*b5*k5_W_2_4 
            # # W_3_1=W_3_1+dt*b1*k1_W_3_1+dt*b3*k3_W_3_1+dt*b4*k4_W_3_1+dt*b5*k5_W_3_1 
            # W_3_2=W_3_2+dt*b1*k1_W_3_2+dt*b3*k3_W_3_2+dt*b4*k4_W_3_2+dt*b5*k5_W_3_2 
            # W_3_3=W_3_3+dt*b1*k1_W_3_3+dt*b3*k3_W_3_3+dt*b4*k4_W_3_3+dt*b5*k5_W_3_3 
            # W_3_4=W_3_4+dt*b1*k1_W_3_4+dt*b3*k3_W_3_4+dt*b4*k4_W_3_4+dt*b5*k5_W_3_4 
            # # Y_1=Y_1+dt*b1*k1_Y_1+dt*b3*k3_Y_1+dt*b4*k4_Y_1+dt*b5*k5_Y_1 
            # Y_2=Y_2+dt*b1*k1_Y_2+dt*b3*k3_Y_2+dt*b4*k4_Y_2+dt*b5*k5_Y_2 
            # Y_3=Y_3+dt*b1*k1_Y_3+dt*b3*k3_Y_3+dt*b4*k4_Y_3+dt*b5*k5_Y_3 
            # Y_4=Y_4+dt*b1*k1_Y_4+dt*b3*k3_Y_4+dt*b4*k4_Y_4+dt*b5*k5_Y_4 
            # Γ_1=Γ_1+dt*b1*k1_Γ_1+dt*b3*k3_Γ_1+dt*b4*k4_Γ_1+dt*b5*k5_Γ_1 
            # Γ_2=Γ_2+dt*b1*k1_Γ_2+dt*b3*k3_Γ_2+dt*b4*k4_Γ_2+dt*b5*k5_Γ_2 
            # Γ_3=Γ_3+dt*b1*k1_Γ_3+dt*b3*k3_Γ_3+dt*b4*k4_Γ_3+dt*b5*k5_Γ_3 
            # Σ=Σ+dt*b1*k1_Σ+dt*b3*k3_Σ+dt*b4*k4_Σ+dt*b5*k5_Σ
    
            # dϕ_1_dt=dϕ_1_dt+dt*b1*kt1_ϕ_1+dt*b3*kt3_ϕ_1+dt*b4*kt4_ϕ_1+dt*b5*kt5_ϕ_1
            # dϕ_2_dt=dϕ_2_dt+dt*b1*kt1_ϕ_2+dt*b3*kt3_ϕ_2+dt*b4*kt4_ϕ_2+dt*b5*kt5_ϕ_2 
            # dϕ_3_dt=dϕ_3_dt+dt*b1*kt1_ϕ_3+dt*b3*kt3_ϕ_3+dt*b4*kt4_ϕ_3+dt*b5*kt5_ϕ_3 
            # dϕ_4_dt=dϕ_4_dt+dt*b1*kt1_ϕ_4+dt*b3*kt3_ϕ_4+dt*b4*kt4_ϕ_4+dt*b5*kt5_ϕ_4 
    
            # # W_1_1+dt*b1*kt1_W_1_1+dt*b3*kt3_W_1_1+dt*b4*kt4_W_1_1+dt*b5*kt5_W_1_1 
            # dW_1_2_dt=dW_1_2_dt+dt*b1*kt1_W_1_2+dt*b3*kt3_W_1_2+dt*b4*kt4_W_1_2+dt*b5*kt5_W_1_2 
            # dW_1_3_dt=dW_1_3_dt+dt*b1*kt1_W_1_3+dt*b3*kt3_W_1_3+dt*b4*kt4_W_1_3+dt*b5*kt5_W_1_3 
            # dW_1_4_dt=dW_1_4_dt+dt*b1*kt1_W_1_4+dt*b3*kt3_W_1_4+dt*b4*kt4_W_1_4+dt*b5*kt5_W_1_4 
            # # W_2_1+dt*b1*kt1_W_2_1+dt*b3*kt3_W_2_1+dt*b4*kt4_W_2_1+dt*b5*kt5_W_1_1 
            # dW_2_2_dt=dW_2_2_dt+dt*b1*kt1_W_2_2+dt*b3*kt3_W_2_2+dt*b4*kt4_W_2_2+dt*b5*kt5_W_2_2 
            # dW_2_3_dt=dW_2_3_dt+dt*b1*kt1_W_2_3+dt*b3*kt3_W_2_3+dt*b4*kt4_W_2_3+dt*b5*kt5_W_2_3 
            # dW_2_4_dt=dW_2_4_dt+dt*b1*kt1_W_2_4+dt*b3*kt3_W_2_4+dt*b4*kt4_W_2_4+dt*b5*kt5_W_2_4 
            # # W_3_1+dt*b1*kt1_W_3_1+dt*b3*kt3_W_3_1+dt*b4*kt4_W_3_1+dt*b5*kt5_W_1_1 
            # dW_3_2_dt=dW_3_2_dt+dt*b1*kt1_W_3_2+dt*b3*kt3_W_3_2+dt*b4*kt4_W_3_2+dt*b5*kt5_W_3_2 
            # dW_3_3_dt=dW_3_3_dt+dt*b1*kt1_W_3_3+dt*b3*kt3_W_3_3+dt*b4*kt4_W_3_3+dt*b5*kt5_W_3_3 
            # dW_3_4_dt=dW_3_4_dt+dt*b1*kt1_W_3_4+dt*b3*kt3_W_3_4+dt*b4*kt4_W_3_4+dt*b5*kt5_W_3_4 
            # # Y_1+dt*b1*kt1_Y_1+dt*b3*kt3_Y_1+dt*b4*kt4_Y_1+dt*b5*kt5_Y_1 
            # dY_2_dt=dY_2_dt+dt*b1*kt1_Y_2+dt*b3*kt3_Y_2+dt*b4*kt4_Y_2+dt*b5*kt5_Y_2 
            # dY_3_dt=dY_3_dt+dt*b1*kt1_Y_3+dt*b3*kt3_Y_3+dt*b4*kt4_Y_3+dt*b5*kt5_Y_3 
            # dY_4_dt=dY_4_dt+dt*b1*kt1_Y_4+dt*b3*kt3_Y_4+dt*b4*kt4_Y_4+dt*b5*kt5_Y_4 
            # # Γ_1+dt*b1*kt1_Γ_1+dt*b3*kt3_Γ_1+dt*b4*kt4_Γ_1+dt*b5*kt5_Γ_1 
            # # Γ_2+dt*b1*kt1_Γ_2+dt*b3*kt3_Γ_2+dt*b4*kt4_Γ_2+dt*b5*kt5_Γ_2 
            # # Γ_3+dt*b1*kt1_Γ_3+dt*b3*kt3_Γ_3+dt*b4*kt4_Γ_3+dt*b5*kt5_Γ_3 
            # # Σ+dt*b1*kt1_Σ+dt*b3*kt3_Σ+dt*b4*kt4_Σ+dt*b5*kt5_Σ
      
            # # synchronize()

            @cuda threads=thrds blocks=blks updater(ϕ_1,k1_ϕ_1,k3_ϕ_1,k4_ϕ_1,k5_ϕ_1,b1,b3,b4,b5,dt)
            @cuda threads=thrds blocks=blks updater(ϕ_2,k1_ϕ_2,k3_ϕ_2,k4_ϕ_2,k5_ϕ_2,b1,b3,b4,b5,dt)
            @cuda threads=thrds blocks=blks updater(ϕ_3,k1_ϕ_3,k3_ϕ_3,k4_ϕ_3,k5_ϕ_3,b1,b3,b4,b5,dt)
            @cuda threads=thrds blocks=blks updater(ϕ_4,k1_ϕ_4,k3_ϕ_4,k4_ϕ_4,k5_ϕ_4,b1,b3,b4,b5,dt)
            @cuda threads=thrds blocks=blks updater(W_1_2,k1_W_1_2,k3_W_1_2,k4_W_1_2,k5_W_1_2,b1,b3,b4,b5,dt)
            @cuda threads=thrds blocks=blks updater(W_1_3,k1_W_1_3,k3_W_1_3,k4_W_1_3,k5_W_1_3,b1,b3,b4,b5,dt)
            @cuda threads=thrds blocks=blks updater(W_1_4,k1_W_1_4,k3_W_1_4,k4_W_1_4,k5_W_1_4,b1,b3,b4,b5,dt)
            @cuda threads=thrds blocks=blks updater(W_2_2,k1_W_2_2,k3_W_2_2,k4_W_2_2,k5_W_2_2,b1,b3,b4,b5,dt)
            @cuda threads=thrds blocks=blks updater(W_2_3,k1_W_2_3,k3_W_2_3,k4_W_2_3,k5_W_2_3,b1,b3,b4,b5,dt)
            @cuda threads=thrds blocks=blks updater(W_2_4,k1_W_2_4,k3_W_2_4,k4_W_2_4,k5_W_2_4,b1,b3,b4,b5,dt)
            @cuda threads=thrds blocks=blks updater(W_3_2,k1_W_3_2,k3_W_3_2,k4_W_3_2,k5_W_3_2,b1,b3,b4,b5,dt)
            @cuda threads=thrds blocks=blks updater(W_3_3,k1_W_3_3,k3_W_3_3,k4_W_3_3,k5_W_3_3,b1,b3,b4,b5,dt)
            @cuda threads=thrds blocks=blks updater(W_3_4,k1_W_3_4,k3_W_3_4,k4_W_3_4,k5_W_3_4,b1,b3,b4,b5,dt)
            @cuda threads=thrds blocks=blks updater(Y_2,k1_Y_2,k3_Y_2,k4_Y_2,k5_Y_2,b1,b3,b4,b5,dt)
            @cuda threads=thrds blocks=blks updater(Y_3,k1_Y_3,k3_Y_3,k4_Y_3,k5_Y_3,b1,b3,b4,b5,dt)
            @cuda threads=thrds blocks=blks updater(Y_4,k1_Y_4,k3_Y_4,k4_Y_4,k5_Y_4,b1,b3,b4,b5,dt)
            @cuda threads=thrds blocks=blks updater(Γ_1,k1_Γ_1,k3_Γ_1,k4_Γ_1,k5_Γ_1,b1,b3,b4,b5,dt)
            @cuda threads=thrds blocks=blks updater(Γ_2,k1_Γ_2,k3_Γ_2,k4_Γ_2,k5_Γ_2,b1,b3,b4,b5,dt)
            @cuda threads=thrds blocks=blks updater(Γ_3,k1_Γ_3,k3_Γ_3,k4_Γ_3,k5_Γ_3,b1,b3,b4,b5,dt)
            @cuda threads=thrds blocks=blks updater(Σ,k1_Σ,k3_Σ,k4_Σ,k5_Σ,b1,b3,b4,b5,dt)
            @cuda threads=thrds blocks=blks updater(dϕ_1_dt,kt1_ϕ_1,kt3_ϕ_1,kt4_ϕ_1,kt5_ϕ_1,b1,b3,b4,b5,dt)
            @cuda threads=thrds blocks=blks updater(dϕ_2_dt,kt1_ϕ_2,kt3_ϕ_2,kt4_ϕ_2,kt5_ϕ_2,b1,b3,b4,b5,dt)
            @cuda threads=thrds blocks=blks updater(dϕ_3_dt,kt1_ϕ_3,kt3_ϕ_3,kt4_ϕ_3,kt5_ϕ_3,b1,b3,b4,b5,dt)
            @cuda threads=thrds blocks=blks updater(dϕ_4_dt,kt1_ϕ_4,kt3_ϕ_4,kt4_ϕ_4,kt5_ϕ_4,b1,b3,b4,b5,dt)
            @cuda threads=thrds blocks=blks updater(dW_1_2_dt,kt1_W_1_2,kt3_W_1_2,kt4_W_1_2,kt5_W_1_2,b1,b3,b4,b5,dt)
            @cuda threads=thrds blocks=blks updater(dW_1_3_dt,kt1_W_1_3,kt3_W_1_3,kt4_W_1_3,kt5_W_1_3,b1,b3,b4,b5,dt)
            @cuda threads=thrds blocks=blks updater(dW_1_4_dt,kt1_W_1_4,kt3_W_1_4,kt4_W_1_4,kt5_W_1_4,b1,b3,b4,b5,dt)
            @cuda threads=thrds blocks=blks updater(dW_2_2_dt,kt1_W_2_2,kt3_W_2_2,kt4_W_2_2,kt5_W_2_2,b1,b3,b4,b5,dt)
            @cuda threads=thrds blocks=blks updater(dW_2_3_dt,kt1_W_2_3,kt3_W_2_3,kt4_W_2_3,kt5_W_2_3,b1,b3,b4,b5,dt)
            @cuda threads=thrds blocks=blks updater(dW_2_4_dt,kt1_W_2_4,kt3_W_2_4,kt4_W_2_4,kt5_W_2_4,b1,b3,b4,b5,dt)
            @cuda threads=thrds blocks=blks updater(dW_3_2_dt,kt1_W_3_2,kt3_W_3_2,kt4_W_3_2,kt5_W_3_2,b1,b3,b4,b5,dt)
            @cuda threads=thrds blocks=blks updater(dW_3_3_dt,kt1_W_3_3,kt3_W_3_3,kt4_W_3_3,kt5_W_3_3,b1,b3,b4,b5,dt)
            @cuda threads=thrds blocks=blks updater(dW_3_4_dt,kt1_W_3_4,kt3_W_3_4,kt4_W_3_4,kt5_W_3_4,b1,b3,b4,b5,dt)
            @cuda threads=thrds blocks=blks updater(dY_2_dt,kt1_Y_2,kt3_Y_2,kt4_Y_2,kt5_Y_2,b1,b3,b4,b5,dt)
            @cuda threads=thrds blocks=blks updater(dY_3_dt,kt1_Y_3,kt3_Y_3,kt4_Y_3,kt5_Y_3,b1,b3,b4,b5,dt)
            @cuda threads=thrds blocks=blks updater(dY_4_dt,kt1_Y_4,kt3_Y_4,kt4_Y_4,kt5_Y_4,b1,b3,b4,b5,dt)
            synchronize()

            MPI.Barrier(comm_cart)
            #Update halo
            # println("updating halos ",it)
            update_halo!(ϕ_1,neighbors_x,neighbors_y,neighbors_z,comm_cart)
            MPI.Barrier(comm_cart)
            update_halo!(ϕ_2,neighbors_x,neighbors_y,neighbors_z,comm_cart)
            MPI.Barrier(comm_cart)
            update_halo!(ϕ_3,neighbors_x,neighbors_y,neighbors_z,comm_cart)
            MPI.Barrier(comm_cart)
            update_halo!(ϕ_4,neighbors_x,neighbors_y,neighbors_z,comm_cart)
            MPI.Barrier(comm_cart)
            update_halo!(W_1_2,neighbors_x,neighbors_y,neighbors_z,comm_cart)
            MPI.Barrier(comm_cart)
            update_halo!(W_1_3,neighbors_x,neighbors_y,neighbors_z,comm_cart)
            MPI.Barrier(comm_cart)
            update_halo!(W_1_4,neighbors_x,neighbors_y,neighbors_z,comm_cart)
            MPI.Barrier(comm_cart)
            update_halo!(W_2_2,neighbors_x,neighbors_y,neighbors_z,comm_cart)
            MPI.Barrier(comm_cart)
            update_halo!(W_2_3,neighbors_x,neighbors_y,neighbors_z,comm_cart)
            MPI.Barrier(comm_cart)
            update_halo!(W_2_4,neighbors_x,neighbors_y,neighbors_z,comm_cart)
            MPI.Barrier(comm_cart)
            update_halo!(W_3_2,neighbors_x,neighbors_y,neighbors_z,comm_cart)
            MPI.Barrier(comm_cart)
            update_halo!(W_3_3,neighbors_x,neighbors_y,neighbors_z,comm_cart)
            MPI.Barrier(comm_cart)
            update_halo!(W_3_4,neighbors_x,neighbors_y,neighbors_z,comm_cart)
            MPI.Barrier(comm_cart)
            update_halo!(Y_2,neighbors_x,neighbors_y,neighbors_z,comm_cart)
            MPI.Barrier(comm_cart)
            update_halo!(Y_3,neighbors_x,neighbors_y,neighbors_z,comm_cart)
            MPI.Barrier(comm_cart)
            update_halo!(Y_4,neighbors_x,neighbors_y,neighbors_z,comm_cart)
            MPI.Barrier(comm_cart)
            update_halo!(Γ_1,neighbors_x,neighbors_y,neighbors_z,comm_cart)
            MPI.Barrier(comm_cart)
            update_halo!(Γ_2,neighbors_x,neighbors_y,neighbors_z,comm_cart)
            MPI.Barrier(comm_cart)
            update_halo!(Γ_3,neighbors_x,neighbors_y,neighbors_z,comm_cart)
            MPI.Barrier(comm_cart)
            update_halo!(Σ,neighbors_x,neighbors_y,neighbors_z,comm_cart)
    
            update_halo!(dϕ_1_dt,neighbors_x,neighbors_y,neighbors_z,comm_cart)
            MPI.Barrier(comm_cart)
            update_halo!(dϕ_2_dt,neighbors_x,neighbors_y,neighbors_z,comm_cart)
            MPI.Barrier(comm_cart)
            update_halo!(dϕ_3_dt,neighbors_x,neighbors_y,neighbors_z,comm_cart)
            MPI.Barrier(comm_cart)
            update_halo!(dϕ_4_dt,neighbors_x,neighbors_y,neighbors_z,comm_cart)
            MPI.Barrier(comm_cart)
            update_halo!(dW_1_2_dt,neighbors_x,neighbors_y,neighbors_z,comm_cart)
            MPI.Barrier(comm_cart)
            update_halo!(dW_1_3_dt,neighbors_x,neighbors_y,neighbors_z,comm_cart)
            MPI.Barrier(comm_cart)
            update_halo!(dW_1_4_dt,neighbors_x,neighbors_y,neighbors_z,comm_cart)
            MPI.Barrier(comm_cart)
            update_halo!(dW_2_2_dt,neighbors_x,neighbors_y,neighbors_z,comm_cart)
            MPI.Barrier(comm_cart)
            update_halo!(dW_2_3_dt,neighbors_x,neighbors_y,neighbors_z,comm_cart)
            MPI.Barrier(comm_cart)
            update_halo!(dW_2_4_dt,neighbors_x,neighbors_y,neighbors_z,comm_cart)
            MPI.Barrier(comm_cart)
            update_halo!(dW_3_2_dt,neighbors_x,neighbors_y,neighbors_z,comm_cart)
            MPI.Barrier(comm_cart)
            update_halo!(dW_3_3_dt,neighbors_x,neighbors_y,neighbors_z,comm_cart)
            MPI.Barrier(comm_cart)
            update_halo!(dW_3_4_dt,neighbors_x,neighbors_y,neighbors_z,comm_cart)
            MPI.Barrier(comm_cart)
            update_halo!(dY_2_dt,neighbors_x,neighbors_y,neighbors_z,comm_cart)
            MPI.Barrier(comm_cart)
            update_halo!(dY_3_dt,neighbors_x,neighbors_y,neighbors_z,comm_cart)
            MPI.Barrier(comm_cart)
            update_halo!(dY_4_dt,neighbors_x,neighbors_y,neighbors_z,comm_cart)

            #Impose boundary conditions
            # println("Updating boundaries")
            ##X boundaries##
            MPI.Barrier(comm_cart)
            boundary_x(ϕ_1,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_x(ϕ_2,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_x(ϕ_3,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_x(ϕ_4,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_x(W_1_2,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_x(W_1_3,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_x(W_1_4,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_x(W_2_2,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_x(W_2_3,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_x(W_2_4,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_x(W_3_2,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_x(W_3_3,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_x(W_3_4,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_x(Y_2,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_x(Y_3,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_x(Y_4,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_x(Γ_1,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_x(Γ_2,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_x(Γ_3,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_x(Σ,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
    
            boundary_x(dϕ_1_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_x(dϕ_2_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_x(dϕ_3_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_x(dϕ_4_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_x(dW_1_2_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_x(dW_1_3_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_x(dW_1_4_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_x(dW_2_2_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_x(dW_2_3_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_x(dW_2_4_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_x(dW_3_2_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_x(dW_3_3_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_x(dW_3_4_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_x(dY_2_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_x(dY_3_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_x(dY_4_dt,dims,comm_cart,coords)
            # MPI.Barrier(comm_cart)
    
            ##Y boundaries##
            MPI.Barrier(comm_cart)
            boundary_y(ϕ_1,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_y(ϕ_2,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_y(ϕ_3,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_y(ϕ_4,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_y(W_1_2,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_y(W_1_3,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_y(W_1_4,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_y(W_2_2,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_y(W_2_3,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_y(W_2_4,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_y(W_3_2,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_y(W_3_3,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_y(W_3_4,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_y(Y_2,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_y(Y_3,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_y(Y_4,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_y(Γ_1,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_y(Γ_2,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_y(Γ_3,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_y(Σ,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
    
            boundary_y(dϕ_1_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_y(dϕ_2_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_y(dϕ_3_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_y(dϕ_4_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_y(dW_1_2_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_y(dW_1_3_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_y(dW_1_4_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_y(dW_2_2_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_y(dW_2_3_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_y(dW_2_4_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_y(dW_3_2_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_y(dW_3_3_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_y(dW_3_4_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_y(dY_2_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_y(dY_3_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_y(dY_4_dt,dims,comm_cart,coords)
            # MPI.Barrier(comm_cart)
    
            ##Z boundaries##
            MPI.Barrier(comm_cart)
            boundary_z(ϕ_1,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_z(ϕ_2,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_z(ϕ_3,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_z(ϕ_4,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_z(W_1_2,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_z(W_1_3,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_z(W_1_4,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_z(W_2_2,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_z(W_2_3,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_z(W_2_4,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_z(W_3_2,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_z(W_3_3,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_z(W_3_4,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_z(Y_2,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_z(Y_3,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_z(Y_4,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_z(Γ_1,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_z(Γ_2,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_z(Γ_3,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_z(Σ,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
    
            boundary_z(dϕ_1_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_z(dϕ_2_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_z(dϕ_3_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_z(dϕ_4_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_z(dW_1_2_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_z(dW_1_3_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_z(dW_1_4_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_z(dW_2_2_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_z(dW_2_3_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_z(dW_2_4_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_z(dW_3_2_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_z(dW_3_3_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_z(dW_3_4_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_z(dY_2_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_z(dY_3_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)
            boundary_z(dY_4_dt,dims,comm_cart,coords)
            MPI.Barrier(comm_cart)

            #Compute B fft and 1d convolution
            if mod(it,dsnaps)==0

                #Compute energies and magnetic fields
                @cuda threads=thrds blocks=blks compute_aux!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
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
                KE_Phi,GE_Phi,PE_Phi,
                ElectricE_W,MagneticE_W,ElectricE_Y,MagneticE_Y,
                B_x,B_y,B_z,
                gw,gy,gp2,vev,lambda,θ_w,dx)
                
                synchronize()

                #Gather energies
                gather(KE_Phi,KE_Phi_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
                gather(PE_Phi,PE_Phi_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
                gather(GE_Phi,GE_Phi_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
                gather(MagneticE_W,MagneticE_W_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
                gather(ElectricE_W,ElectricE_W_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
                gather(MagneticE_Y,MagneticE_Y_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
                gather(ElectricE_Y,ElectricE_Y_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
                MPI.Barrier(comm_cart)
                
                snp_idx = snp_idx+1

                # #FFT compute and convolve
                # B_x_fft = Array(fft(B_x))
                # B_y_fft = Array(fft(B_y))
                # B_z_fft = Array(fft(B_z))

                # gather_fft(B_x_fft,B_x_fft_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
                # gather_fft(B_y_fft,B_y_fft_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
                # gather_fft(B_z_fft,B_z_fft_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
                # MPI.Barrier(comm_cart)

                # B_fft[snp_idx,:,:] = convolve_1d((real(conj.(B_x_fft_g).*B_x_fft_g.+
                # conj.(B_y_fft_g).*B_y_fft_g.+
                # conj.(B_z_fft_g).*B_z_fft_g)),Nx_g,Ny_g,Nz_g,spec_cut[1],spec_cut[2],spec_cut[3])
                
                #Sum energies
                # total_energies[snp_idx,1] = sum(Array(PE_Phi))
                # total_energies[snp_idx,2] = sum(Array(KE_Phi))
                # total_energies[snp_idx,3] = sum(Array(GE_Phi))
                # total_energies[snp_idx,4] = sum(Array(ElectricE_W))
                # total_energies[snp_idx,5] = sum(Array(MagneticE_W))
                # total_energies[snp_idx,6] = sum(Array(ElectricE_Y))
                # total_energies[snp_idx,7] = sum(Array(MagneticE_Y))

                #Total energies
                total_energies[snp_idx,1] = sum(Array(PE_Phi_g))
                total_energies[snp_idx,2] = sum(Array(KE_Phi_g))
                total_energies[snp_idx,3] = sum(Array(GE_Phi_g))
                total_energies[snp_idx,4] = sum(Array(ElectricE_W_g))
                total_energies[snp_idx,5] = sum(Array(MagneticE_W_g))
                total_energies[snp_idx,6] = sum(Array(ElectricE_Y_g))
                total_energies[snp_idx,7] = sum(Array(MagneticE_Y_g))
                println(string(it,total_energies[snp_idx,:]))
                if me==0
                    # plot_1=contourf(z,x,(Array(ϕ_1)[:,Ny÷2,:]).^2+(Array(ϕ_2)[:,Ny÷2,:]).^2+(Array(ϕ_3)[:,Ny÷2,:]).^2+(Array(ϕ_4)[:,Ny÷2,:]).^2)
                    plot_1=contourf(z,x,(Array(PE_Phi_g)[:,Ny_g÷2,:]),title="PE")
                    plot_2=contourf(z,x,(Array(KE_Phi_g)[:,Ny_g÷2,:]),title="KE")
                    # plot_3=contourf(z,x,(Array(GE_Phi_g)[:,Ny_g÷2,:]),title="GE")
    
                    # plot_3=plot(B_fft[snp_idx,2:end,1],(((B_fft[snp_idx,2:end,1]).^2)./((2*pi)^3*Nx^2)).*B_fft[snp_idx,2:end,2],xscale=:log10,yscale=:log10,minorgrid=true)
                    plot_4=contourf(z,x,(Array(ElectricE_W_g)[:,Ny_g÷2,:]+Array(MagneticE_W_g)[:,Ny_g÷2,:]+Array(ElectricE_Y_g)[:,Ny_g÷2,:]+Array(MagneticE_Y_g)[:,Ny_g÷2,:]),title="WY E")
                    plot_4 = plot(range(1,snp_idx,step=1).*dsnaps,[total_energies[1:snp_idx,1] total_energies[1:snp_idx,2] total_energies[1:snp_idx,3] total_energies[1:snp_idx,4] total_energies[1:snp_idx,5] total_energies[1:snp_idx,6] total_energies[1:snp_idx,7]],
                    label=["PE" "KE" "GE" "EEW" "MEW" "EEY" "MEY"],xlims=(0,nte))
                    plot(plot_1,plot_2,plot_3,plot_4,layout=4,plot_title=string("it:",it),dpi=600)
                    frame(anim)
                end

                MPI.Barrier(comm_cart)

            end

        end

    end
    if me==0
        gif(anim, "EW3d_test_rk45.mp4", fps = 10)

        # println("test:",Array(ϕ_1)[3,5,3])
        # CUDA.memory_status()

        gr()
        ENV["GKSwstype"]="nul"
        plot(range(0,nte,step=dsnaps),[total_energies[:,1] total_energies[:,2] total_energies[:,3] total_energies[:,4] total_energies[:,5] total_energies[:,6] total_energies[:,7]].+1.0,
        label=["PE" "KE" "GE" "EEW" "MEW" "EEY" "MEY"],yscale=:log10,dpi=600)
        png("energies_rk45.png")

    end

    MPI.Barrier(comm_cart)

    #FFT compute and convolve
    B_x_fft = Array(fft(B_x))
    B_y_fft = Array(fft(B_y))
    B_z_fft = Array(fft(B_z))

    gather_fft(B_x_fft,B_x_fft_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
    gather_fft(B_y_fft,B_y_fft_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
    gather_fft(B_z_fft,B_z_fft_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
    MPI.Barrier(comm_cart)

    B_fft[end,:,:] = convolve_1d((real(conj.(B_x_fft_g).*B_x_fft_g.+
    conj.(B_y_fft_g).*B_y_fft_g.+
    conj.(B_z_fft_g).*B_z_fft_g)),Nx_g,Ny_g,Nz_g,spec_cut[1],spec_cut[2],spec_cut[3])
    ##END FFT##

    if me==0
        gr()
        ENV["GKSwstype"]="nul"
        y1 = (((B_fft[1,2:end,1]).^2)./((2*pi)^3*Nx^2)).*B_fft[1,2:end,2]
        y2 = (((B_fft[end,2:end,1]).^2)./((2*pi)^3*Nx^2)).*B_fft[end,2:end,2]
        plot(B_fft[end,2:end,1],[y1,y2],label=[0,nte],xscale=:log10,yscale=:log10,minorgrid=true)
        png("spectra_rk45.png")
    end


    MPI.Barrier(comm_cart)

    MPI.Finalize()

    return
    
end
run()
CUDA.memory_status()
