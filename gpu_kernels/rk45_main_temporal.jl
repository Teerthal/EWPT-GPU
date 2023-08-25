#Extending rk4 tempral gauge solvers routines to Runge-Kutte-Fehlberg(RKF45) to introduce adaptive stepsizes
#Butcher tableau coefficients can be adjusted to other 4th-5th order adapative recipies

using CUDA#, CuArrays
using Random
using StatsBase
using Distributions
using Plots
using CUDA.CUFFT
using Statistics

CUDA.memory_status()
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

function rk4_step!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
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

function compute_aux!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
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

function initializer!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,ib,jb,kb,p1,p2,p3,p4,rkx,rky,rkz,dx,mH)
    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
    (blockIdx().y - 1) * blockDim().y + threadIdx().y,
    (blockIdx().z - 1) * blockDim().z + threadIdx().z
    # @cuprintln(i," ",j," ",k)
    rb = sqrt((1.0/(rkx^2))*sin(rkx*((i-1)-(ib-0.5))*dx)^2+(1.0/(rky^2))*sin(rky*((j-1)-(jb-0.5))*dx)^2+(1.0/(rkz^2))*sin(rkz*((k-1)-(kb-0.5))*dx)^2)
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

function run()

    Nx=Ny=Nz=16*10
    println(Nx,",",Ny,",",Nz)
    gw = 0.65
    gy = 0.34521
    gp2 = 0.75
    vev = 1.0
    lambda = 1.0/8.0
    dx=0.2
    dt=dx/75
    mH = 2.0*sqrt(lambda)*vev
    nte = 20000
    θ_w = asin(sqrt(0.22))
    tol = 1e-4

    nsnaps=50
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

    k2_W_1_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k2_W_1_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k2_W_1_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k2_W_1_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k2_W_2_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k2_W_2_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k2_W_2_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k2_W_2_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k2_W_3_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k2_W_3_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k2_W_3_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k2_W_3_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k2_Y_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
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

    kt2_W_1_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt2_W_1_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt2_W_1_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt2_W_1_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt2_W_2_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt2_W_2_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt2_W_2_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt2_W_2_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt2_W_3_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt2_W_3_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt2_W_3_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt2_W_3_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt2_Y_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
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

    k3_W_1_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k3_W_1_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k3_W_1_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k3_W_1_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k3_W_2_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k3_W_2_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k3_W_2_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k3_W_2_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k3_W_3_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k3_W_3_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k3_W_3_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k3_W_3_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k3_Y_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
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

    kt3_W_1_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt3_W_1_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt3_W_1_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt3_W_1_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt3_W_2_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt3_W_2_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt3_W_2_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt3_W_2_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt3_W_3_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt3_W_3_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt3_W_3_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt3_W_3_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt3_Y_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
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

    k4_W_1_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k4_W_1_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k4_W_1_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k4_W_1_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k4_W_2_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k4_W_2_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k4_W_2_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k4_W_2_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k4_W_3_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k4_W_3_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k4_W_3_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k4_W_3_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k4_Y_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
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

    kt4_W_1_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt4_W_1_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt4_W_1_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt4_W_1_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt4_W_2_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt4_W_2_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt4_W_2_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt4_W_2_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt4_W_3_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt4_W_3_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt4_W_3_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt4_W_3_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt4_Y_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
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

    k5_W_1_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k5_W_1_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k5_W_1_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k5_W_1_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k5_W_2_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k5_W_2_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k5_W_2_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k5_W_2_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k5_W_3_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k5_W_3_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k5_W_3_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k5_W_3_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k5_Y_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
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

    kt5_W_1_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt5_W_1_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt5_W_1_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt5_W_1_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt5_W_2_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt5_W_2_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt5_W_2_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt5_W_2_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt5_W_3_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt5_W_3_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt5_W_3_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt5_W_3_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt5_Y_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
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

    k6_W_1_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k6_W_1_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k6_W_1_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k6_W_1_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k6_W_2_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k6_W_2_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k6_W_2_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k6_W_2_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k6_W_3_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k6_W_3_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k6_W_3_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k6_W_3_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    k6_Y_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
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

    kt6_W_1_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt6_W_1_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt6_W_1_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt6_W_1_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt6_W_2_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt6_W_2_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt6_W_2_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt6_W_2_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt6_W_3_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt6_W_3_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt6_W_3_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt6_W_3_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt6_Y_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt6_Y_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt6_Y_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    kt6_Y_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt6_Γ_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt6_Γ_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt6_Γ_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    # kt6_Σ = CUDA.zeros(Float64,(Nx,Ny,Nz))

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

    spec_cut = [Nx÷4,Ny÷4,Nz÷4]
    N_bins = Kc_bin_nums(spec_cut[1],spec_cut[2],spec_cut[3])
    B_fft = zeros((nsnaps+1,N_bins,2))
    
    #Error arrays
    err = zeros((48))

    CUDA.memory_status()

    ##########Configuring thread block grid###########

    thrds = (32,1,1)
    blks = (Nx÷thrds[1],Ny÷thrds[2],Nz÷thrds[3])
    println(string("#threads:",thrds," #blocks:",blks))

    ##########END Configuring thread block grid###########

    #Initializing random bubbles

    Random.seed!(seed_value)
    bubble_locs = rand(1:Nx,(no_bubbles,3))

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

    rkx=pi/(Nx*dx)
    rky=pi/(Ny*dx)
    rkz=pi/(Nz*dx)

    @time for b in range(1,size(bubs,1),step=1)
        ib,jb,kb,p1,p2,p3,p4 = bubs[b]
        # @cuda threads=thrds blocks=blks initializer!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,ib,jb,kb,p1,p2,p3,p4,rkx,rky,rkz,dx,mH)
        @cuda threads=thrds blocks=blks initializer!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,ib,jb,kb,p1,p2,p3,p4,rkx,rky,rkz,dx,mH)
        synchronize()
    end

    #compute energies and magnetic fields at initial time step
    @cuda threads=thrds blocks=blks compute_aux!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_1,W_1_2,W_1_3,W_1_4,
    W_2_1,W_2_2,W_2_3,W_2_4,
    W_3_1,W_3_2,W_3_3,W_3_4,
    Y_1,Y_2,Y_3,Y_4,
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
    
    B_fft[1,:,:] = convolve_1d((real(conj.(B_x_fft).*B_x_fft.+
    conj.(B_y_fft).*B_y_fft.+
    conj.(B_z_fft).*B_z_fft)),Nx,Ny,Nz,spec_cut[1],spec_cut[2],spec_cut[3])
    end

    #Add energies
    total_energies[1,1] = sum(Array(PE_Phi))
    total_energies[1,2] = sum(Array(KE_Phi))
    total_energies[1,3] = sum(Array(GE_Phi))
    total_energies[1,4] = sum(Array(ElectricE_W))
    total_energies[1,5] = sum(Array(MagneticE_W))
    total_energies[1,6] = sum(Array(ElectricE_Y))
    total_energies[1,7] = sum(Array(MagneticE_Y))

    ##PLOT##
    x=range(1,Nx,step=1)
    y=range(1,Ny,step=1)
    z=range(1,Nz,step=1)
    # println(size(x),size(y),size(z))
    gr()
    ENV["GKSwstype"]="nul"
    anim = Animation();
    # plot_1=contourf(z,x,(Array(ϕ_1)[:,Ny÷2,:]).^2+(Array(ϕ_2)[:,Ny÷2,:]).^2+(Array(ϕ_3)[:,Ny÷2,:]).^2+(Array(ϕ_4)[:,Ny÷2,:]).^2)
    plot_1=contourf(z,x,(Array(PE_Phi)[:,Ny÷2,:]),title="PE")
    plot_2=contourf(z,x,(Array(KE_Phi)[:,Ny÷2,:]),title="KE")
    # plot_3=contourf(z,x,(Array(GE_Phi)[:,Ny÷2,:]),title="GE")
    plot_3=plot(B_fft[1,2:end,1],(((B_fft[1,2:end,1]).^2)./((2*pi)^3*Nx^2)).*B_fft[1,2:end,2],xscale=:log10,yscale=:log10,minorgrid=true)
    # plot_4=contourf(z,x,(Array(ElectricE_W)[:,Ny÷2,:]+Array(MagneticE_W)[:,Ny÷2,:]+Array(ElectricE_Y)[:,Ny÷2,:]+Array(MagneticE_Y)[:,Ny÷2,:]),title="WY E")
    plot_4 = scatter([0],[total_energies[1,1] total_energies[1,2] total_energies[1,3] total_energies[1,4] total_energies[1,5] total_energies[1,6] total_energies[1,7]],
    label=["PE" "KE" "GE" "EEW" "MEW" "EEY" "MEY"],xlims=(0,nte))
    plot(plot_1,plot_2,plot_3,plot_4,layout=4,plot_title=string("it:",0),dpi=600)
    # plot(plot_1,title=string("it:",0),dpi=600)
    png(string("testini1",".png"))
    frame(anim)

    # exit()

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
        @time begin
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
        k3_W_1_1,k3_W_1_2,k3_W_1_3,k3_W_1_4,
        k3_W_2_1,k3_W_2_2,k3_W_2_3,k3_W_2_4,
        k3_W_3_1,k3_W_3_2,k3_W_3_3,k3_W_3_4,
        k3_Y_1,k3_Y_2,k3_Y_3,k3_Y_4,
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
        k4_W_1_1,k4_W_1_2,k4_W_1_3,k4_W_1_4,
        k4_W_2_1,k4_W_2_2,k4_W_2_3,k4_W_2_4,
        k4_W_3_1,k4_W_3_2,k4_W_3_3,k4_W_3_4,
        k4_Y_1,k4_Y_2,k4_Y_3,k4_Y_4,
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
        k5_W_1_1,k5_W_1_2,k5_W_1_3,k5_W_1_4,
        k5_W_2_1,k5_W_2_2,k5_W_2_3,k5_W_2_4,
        k5_W_3_1,k5_W_3_2,k5_W_3_3,k5_W_3_4,
        k5_Y_1,k5_Y_2,k5_Y_3,k5_Y_4,
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
        k6_W_1_1,k6_W_1_2,k6_W_1_3,k6_W_1_4,
        k6_W_2_1,k6_W_2_2,k6_W_2_3,k6_W_2_4,
        k6_W_3_1,k6_W_3_2,k6_W_3_3,k6_W_3_4,
        k6_Y_1,k6_Y_2,k6_Y_3,k6_Y_4,
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
        
        end

        @time begin
        err[1]=maximum(abs.(Array((ϕ_1+dt*c1*k1_ϕ_1+dt*c3*k3_ϕ_1+dt*c4*k4_ϕ_1+dt*c5*k5_ϕ_1+dt*c6*k6_ϕ_1) -(ϕ_1+dt*b1*k1_ϕ_1+dt*b3*k3_ϕ_1+dt*b4*k4_ϕ_1+dt*b5*k5_ϕ_1))))
        err[2]=maximum(abs.(Array((ϕ_2+dt*c1*k1_ϕ_2+dt*c3*k3_ϕ_2+dt*c4*k4_ϕ_2+dt*c5*k5_ϕ_2+dt*c6*k6_ϕ_2) -(ϕ_2+dt*b1*k1_ϕ_2+dt*b3*k3_ϕ_2+dt*b4*k4_ϕ_2+dt*b5*k5_ϕ_2))))
        err[3]=maximum(abs.(Array((ϕ_3+dt*c1*k1_ϕ_3+dt*c3*k3_ϕ_3+dt*c4*k4_ϕ_3+dt*c5*k5_ϕ_3+dt*c6*k6_ϕ_3) -(ϕ_3+dt*b1*k1_ϕ_3+dt*b3*k3_ϕ_3+dt*b4*k4_ϕ_3+dt*b5*k5_ϕ_3)))) 
        err[4]=maximum(abs.(Array((ϕ_4+dt*c1*k1_ϕ_4+dt*c3*k3_ϕ_4+dt*c4*k4_ϕ_4+dt*c5*k5_ϕ_4+dt*c6*k6_ϕ_4) -(ϕ_4+dt*b1*k1_ϕ_4+dt*b3*k3_ϕ_4+dt*b4*k4_ϕ_4+dt*b5*k5_ϕ_4)))) 

        # err[5]=maximum(abs.(Array((W_1_1+dt*c1*k1_W_1_1+dt*c3*k3_W_1_1+dt*c4*k4_W_1_1+dt*c5*k5_W_1_1+dt*c6*k6_W_1_1) -(W_1_1+dt*b1*k1_W_1_1+dt*b3*k3_W_1_1+dt*b4*k4_W_1_1+dt*b5*k5_W_1_1)))) 
        err[6]=maximum(abs.(Array((W_1_2+dt*c1*k1_W_1_2+dt*c3*k3_W_1_2+dt*c4*k4_W_1_2+dt*c5*k5_W_1_2+dt*c6*k6_W_1_2) -(W_1_2+dt*b1*k1_W_1_2+dt*b3*k3_W_1_2+dt*b4*k4_W_1_2+dt*b5*k5_W_1_2)))) 
        err[7]=maximum(abs.(Array((W_1_3+dt*c1*k1_W_1_3+dt*c3*k3_W_1_3+dt*c4*k4_W_1_3+dt*c5*k5_W_1_3+dt*c6*k6_W_1_3) -(W_1_3+dt*b1*k1_W_1_3+dt*b3*k3_W_1_3+dt*b4*k4_W_1_3+dt*b5*k5_W_1_3)))) 
        err[8]=maximum(abs.(Array((W_1_4+dt*c1*k1_W_1_4+dt*c3*k3_W_1_4+dt*c4*k4_W_1_4+dt*c5*k5_W_1_4+dt*c6*k6_W_1_4) -(W_1_4+dt*b1*k1_W_1_4+dt*b3*k3_W_1_4+dt*b4*k4_W_1_4+dt*b5*k5_W_1_4)))) 
        # err[9]=maximum(abs.(Array((W_2_1+dt*c1*k1_W_2_1+dt*c3*k3_W_2_1+dt*c4*k4_W_2_1+dt*c5*k5_W_2_1+dt*c6*k6_W_2_1) -(W_2_1+dt*b1*k1_W_2_1+dt*b3*k3_W_2_1+dt*b4*k4_W_2_1+dt*b5*k5_W_2_1)))) 
        err[10]=maximum(abs.(Array((W_2_2+dt*c1*k1_W_2_2+dt*c3*k3_W_2_2+dt*c4*k4_W_2_2+dt*c5*k5_W_2_2+dt*c6*k6_W_2_2) -(W_2_2+dt*b1*k1_W_2_2+dt*b3*k3_W_2_2+dt*b4*k4_W_2_2+dt*b5*k5_W_2_2)))) 
        err[11]=maximum(abs.(Array((W_2_3+dt*c1*k1_W_2_3+dt*c3*k3_W_2_3+dt*c4*k4_W_2_3+dt*c5*k5_W_2_3+dt*c6*k6_W_2_3) -(W_2_3+dt*b1*k1_W_2_3+dt*b3*k3_W_2_3+dt*b4*k4_W_2_3+dt*b5*k5_W_2_3)))) 
        err[12]=maximum(abs.(Array((W_2_4+dt*c1*k1_W_2_4+dt*c3*k3_W_2_4+dt*c4*k4_W_2_4+dt*c5*k5_W_2_4+dt*c6*k6_W_2_4) -(W_2_4+dt*b1*k1_W_2_4+dt*b3*k3_W_2_4+dt*b4*k4_W_2_4+dt*b5*k5_W_2_4)))) 
        # err[13]=maximum(abs.(Array((W_3_1+dt*c1*k1_W_3_1+dt*c3*k3_W_3_1+dt*c4*k4_W_3_1+dt*c5*k5_W_3_1+dt*c6*k6_W_3_1) -(W_3_1+dt*b1*k1_W_3_1+dt*b3*k3_W_3_1+dt*b4*k4_W_3_1+dt*b5*k5_W_3_1)))) 
        err[14]=maximum(abs.(Array((W_3_2+dt*c1*k1_W_3_2+dt*c3*k3_W_3_2+dt*c4*k4_W_3_2+dt*c5*k5_W_3_2+dt*c6*k6_W_3_2) -(W_3_2+dt*b1*k1_W_3_2+dt*b3*k3_W_3_2+dt*b4*k4_W_3_2+dt*b5*k5_W_3_2)))) 
        err[15]=maximum(abs.(Array((W_3_3+dt*c1*k1_W_3_3+dt*c3*k3_W_3_3+dt*c4*k4_W_3_3+dt*c5*k5_W_3_3+dt*c6*k6_W_3_3) -(W_3_3+dt*b1*k1_W_3_3+dt*b3*k3_W_3_3+dt*b4*k4_W_3_3+dt*b5*k5_W_3_3)))) 
        err[16]=maximum(abs.(Array((W_3_4+dt*c1*k1_W_3_4+dt*c3*k3_W_3_4+dt*c4*k4_W_3_4+dt*c5*k5_W_3_4+dt*c6*k6_W_3_4) -(W_3_4+dt*b1*k1_W_3_4+dt*b3*k3_W_3_4+dt*b4*k4_W_3_4+dt*b5*k5_W_3_4)))) 
        # err[17]=maximum(abs.(Array((Y_1+dt*c1*k1_Y_1+dt*c3*k3_Y_1+dt*c4*k4_Y_1+dt*c5*k5_Y_1+dt*c6*k6_Y_1) -(Y_1+dt*b1*k1_Y_1+dt*b3*k3_Y_1+dt*b4*k4_Y_1+dt*b5*k5_Y_1)))) 
        err[18]=maximum(abs.(Array((Y_2+dt*c1*k1_Y_2+dt*c3*k3_Y_2+dt*c4*k4_Y_2+dt*c5*k5_Y_2+dt*c6*k6_Y_2) -(Y_2+dt*b1*k1_Y_2+dt*b3*k3_Y_2+dt*b4*k4_Y_2+dt*b5*k5_Y_2)))) 
        err[19]=maximum(abs.(Array((Y_3+dt*c1*k1_Y_3+dt*c3*k3_Y_3+dt*c4*k4_Y_3+dt*c5*k5_Y_3+dt*c6*k6_Y_3) -(Y_3+dt*b1*k1_Y_3+dt*b3*k3_Y_3+dt*b4*k4_Y_3+dt*b5*k5_Y_3)))) 
        err[20]=maximum(abs.(Array((Y_4+dt*c1*k1_Y_4+dt*c3*k3_Y_4+dt*c4*k4_Y_4+dt*c5*k5_Y_4+dt*c6*k6_Y_4) -(Y_4+dt*b1*k1_Y_4+dt*b3*k3_Y_4+dt*b4*k4_Y_4+dt*b5*k5_Y_4)))) 
        err[21]=maximum(abs.(Array((Γ_1+dt*c1*k1_Γ_1+dt*c3*k3_Γ_1+dt*c4*k4_Γ_1+dt*c5*k5_Γ_1+dt*c6*k6_Γ_1) -(Γ_1+dt*b1*k1_Γ_1+dt*b3*k3_Γ_1+dt*b4*k4_Γ_1+dt*b5*k5_Γ_1)))) 
        err[22]=maximum(abs.(Array((Γ_2+dt*c1*k1_Γ_2+dt*c3*k3_Γ_2+dt*c4*k4_Γ_2+dt*c5*k5_Γ_2+dt*c6*k6_Γ_2) -(Γ_2+dt*b1*k1_Γ_2+dt*b3*k3_Γ_2+dt*b4*k4_Γ_2+dt*b5*k5_Γ_2)))) 
        err[23]=maximum(abs.(Array((Γ_3+dt*c1*k1_Γ_3+dt*c3*k3_Γ_3+dt*c4*k4_Γ_3+dt*c5*k5_Γ_3+dt*c6*k6_Γ_3) -(Γ_3+dt*b1*k1_Γ_3+dt*b3*k3_Γ_3+dt*b4*k4_Γ_3+dt*b5*k5_Γ_3)))) 
        err[24]=maximum(abs.(Array((Σ+dt*c1*k1_Σ+dt*c3*k3_Σ+dt*c4*k4_Σ+dt*c5*k5_Σ+dt*c6*k6_Σ)-(Σ+dt*b1*k1_Σ+dt*b3*k3_Σ+dt*b4*k4_Σ+dt*b5*k5_Σ))))

        err[25]=maximum(abs.(Array((dϕ_1_dt+dt*c1*kt1_ϕ_1+dt*c3*kt3_ϕ_1+dt*c4*kt4_ϕ_1+dt*c5*kt5_ϕ_1+dt*c6*kt6_ϕ_1)-(dϕ_1_dt+dt*b1*kt1_ϕ_1+dt*b3*kt3_ϕ_1+dt*b4*kt4_ϕ_1+dt*b5*kt5_ϕ_1))))
        err[26]=maximum(abs.(Array((dϕ_2_dt+dt*c1*kt1_ϕ_2+dt*c3*kt3_ϕ_2+dt*c4*kt4_ϕ_2+dt*c5*kt5_ϕ_2+dt*c6*kt6_ϕ_2) -(dϕ_2_dt+dt*b1*kt1_ϕ_2+dt*b3*kt3_ϕ_2+dt*b4*kt4_ϕ_2+dt*b5*kt5_ϕ_2)))) 
        err[27]=maximum(abs.(Array((dϕ_3_dt+dt*c1*kt1_ϕ_3+dt*c3*kt3_ϕ_3+dt*c4*kt4_ϕ_3+dt*c5*kt5_ϕ_3+dt*c6*kt6_ϕ_3) -(dϕ_3_dt+dt*b1*kt1_ϕ_3+dt*b3*kt3_ϕ_3+dt*b4*kt4_ϕ_3+dt*b5*kt5_ϕ_3)))) 
        err[28]=maximum(abs.(Array((dϕ_4_dt+dt*c1*kt1_ϕ_4+dt*c3*kt3_ϕ_4+dt*c4*kt4_ϕ_4+dt*c5*kt5_ϕ_4+dt*c6*kt6_ϕ_4) -(dϕ_4_dt+dt*b1*kt1_ϕ_4+dt*b3*kt3_ϕ_4+dt*b4*kt4_ϕ_4+dt*b5*kt5_ϕ_4)))) 

        # -(W_1_1+dt*b1*kt1_W_1_1+dt*b3*kt3_W_1_1+dt*b4*kt4_W_1_1+dt*b5*kt5_W_1_1) 
        err[30]=maximum(abs.(Array((dW_1_2_dt+dt*c1*kt1_W_1_2+dt*c3*kt3_W_1_2+dt*c4*kt4_W_1_2+dt*c5*kt5_W_1_2+dt*c6*kt6_W_1_2) -(dW_1_2_dt+dt*b1*kt1_W_1_2+dt*b3*kt3_W_1_2+dt*b4*kt4_W_1_2+dt*b5*kt5_W_1_2)))) 
        err[31]=maximum(abs.(Array((dW_1_3_dt+dt*c1*kt1_W_1_3+dt*c3*kt3_W_1_3+dt*c4*kt4_W_1_3+dt*c5*kt5_W_1_3+dt*c6*kt6_W_1_3) -(dW_1_3_dt+dt*b1*kt1_W_1_3+dt*b3*kt3_W_1_3+dt*b4*kt4_W_1_3+dt*b5*kt5_W_1_3)))) 
        err[32]=maximum(abs.(Array((dW_1_4_dt+dt*c1*kt1_W_1_4+dt*c3*kt3_W_1_4+dt*c4*kt4_W_1_4+dt*c5*kt5_W_1_4+dt*c6*kt6_W_1_4) -(dW_1_4_dt+dt*b1*kt1_W_1_4+dt*b3*kt3_W_1_4+dt*b4*kt4_W_1_4+dt*b5*kt5_W_1_4)))) 
        # -(W_2_1+dt*b1*kt1_W_2_1+dt*b3*kt3_W_2_1+dt*b4*kt4_W_2_1+dt*b5*kt5_W_1_1) 
        err[34]=maximum(abs.(Array((dW_2_2_dt+dt*c1*kt1_W_2_2+dt*c3*kt3_W_2_2+dt*c4*kt4_W_2_2+dt*c5*kt5_W_2_2+dt*c6*kt6_W_2_2) -(dW_2_2_dt+dt*b1*kt1_W_2_2+dt*b3*kt3_W_2_2+dt*b4*kt4_W_2_2+dt*b5*kt5_W_2_2)))) 
        err[35]=maximum(abs.(Array((dW_2_3_dt+dt*c1*kt1_W_2_3+dt*c3*kt3_W_2_3+dt*c4*kt4_W_2_3+dt*c5*kt5_W_2_3+dt*c6*kt6_W_2_3) -(dW_2_3_dt+dt*b1*kt1_W_2_3+dt*b3*kt3_W_2_3+dt*b4*kt4_W_2_3+dt*b5*kt5_W_2_3)))) 
        err[36]=maximum(abs.(Array((dW_2_4_dt+dt*c1*kt1_W_2_4+dt*c3*kt3_W_2_4+dt*c4*kt4_W_2_4+dt*c5*kt5_W_2_4+dt*c6*kt6_W_2_4) -(dW_2_4_dt+dt*b1*kt1_W_2_4+dt*b3*kt3_W_2_4+dt*b4*kt4_W_2_4+dt*b5*kt5_W_2_4)))) 
        # -(W_3_1+dt*b1*kt1_W_3_1+dt*b3*kt3_W_3_1+dt*b4*kt4_W_3_1+dt*b5*kt5_W_2_1) 
        err[38]=maximum(abs.(Array((dW_3_2_dt+dt*c1*kt1_W_3_2+dt*c3*kt3_W_3_2+dt*c4*kt4_W_3_2+dt*c5*kt5_W_3_2+dt*c6*kt6_W_3_2) -(dW_3_2_dt+dt*b1*kt1_W_3_2+dt*b3*kt3_W_3_2+dt*b4*kt4_W_3_2+dt*b5*kt5_W_3_2)))) 
        err[39]=maximum(abs.(Array((dW_3_3_dt+dt*c1*kt1_W_3_3+dt*c3*kt3_W_3_3+dt*c4*kt4_W_3_3+dt*c5*kt5_W_3_3+dt*c6*kt6_W_3_3) -(dW_3_3_dt+dt*b1*kt1_W_3_3+dt*b3*kt3_W_3_3+dt*b4*kt4_W_3_3+dt*b5*kt5_W_3_3)))) 
        err[40]=maximum(abs.(Array((dW_3_4_dt+dt*c1*kt1_W_3_4+dt*c3*kt3_W_3_4+dt*c4*kt4_W_3_4+dt*c5*kt5_W_3_4+dt*c6*kt6_W_3_4) -(dW_3_4_dt+dt*b1*kt1_W_3_4+dt*b3*kt3_W_3_4+dt*b4*kt4_W_3_4+dt*b5*kt5_W_3_4)))) 
        # -(Y_1+dt*b1*kt1_Y_1+dt*b3*kt3_Y_1+dt*b4*kt4_Y_1+dt*b5*kt5_Y_1) 
        err[42]=maximum(abs.(Array((dY_2_dt+dt*c1*kt1_Y_2+dt*c3*kt3_Y_2+dt*c4*kt4_Y_2+dt*c5*kt5_Y_2+dt*c6*kt6_Y_2) -(dY_2_dt+dt*b1*kt1_Y_2+dt*b3*kt3_Y_2+dt*b4*kt4_Y_2+dt*b5*kt5_Y_2)))) 
        err[43]=maximum(abs.(Array((dY_3_dt+dt*c1*kt1_Y_3+dt*c3*kt3_Y_3+dt*c4*kt4_Y_3+dt*c5*kt5_Y_3+dt*c6*kt6_Y_3) -(dY_3_dt+dt*b1*kt1_Y_3+dt*b3*kt3_Y_3+dt*b4*kt4_Y_3+dt*b5*kt5_Y_3)))) 
        err[44]=maximum(abs.(Array((dY_4_dt+dt*c1*kt1_Y_4+dt*c3*kt3_Y_4+dt*c4*kt4_Y_4+dt*c5*kt5_Y_4+dt*c6*kt6_Y_4) -(dY_4_dt+dt*b1*kt1_Y_4+dt*b3*kt3_Y_4+dt*b4*kt4_Y_4+dt*b5*kt5_Y_4)))) 
        # -(Γ_1+dt*b1*kt1_Γ_1+dt*b3*kt3_Γ_1+dt*b4*kt4_Γ_1+dt*b5*kt5_Γ_1) 
        # -(Γ_2+dt*b1*kt1_Γ_2+dt*b3*kt3_Γ_2+dt*b4*kt4_Γ_2+dt*b5*kt5_Γ_2) 
        # -(Γ_3+dt*b1*kt1_Γ_3+dt*b3*kt3_Γ_3+dt*b4*kt4_Γ_3+dt*b5*kt5_Γ_3) 
        # -(Σ+dt*b1*kt1_Σ+dt*b3*kt3_Σ+dt*b4*kt4_Σ+dt*b5*kt5_Σ)
        end

        max_err = maximum(err)

        if (max_err <= (tol*dt))
            dt = dt*(tol*dt/(max_err))
            println(string("step too large"))
        else
            println(string("step too small"))
            it= it+1
            dt = dt*(tol*dt/(2.0*max_err))^(1/4)

            #Update values
            ϕ_1=ϕ_1+dt*b1*k1_ϕ_1+dt*b3*k3_ϕ_1+dt*b4*k4_ϕ_1+dt*b5*k5_ϕ_1
            ϕ_2=ϕ_2+dt*b1*k1_ϕ_2+dt*b3*k3_ϕ_2+dt*b4*k4_ϕ_2+dt*b5*k5_ϕ_2 
            ϕ_3=ϕ_3+dt*b1*k1_ϕ_3+dt*b3*k3_ϕ_3+dt*b4*k4_ϕ_3+dt*b5*k5_ϕ_3 
            ϕ_4=ϕ_4+dt*b1*k1_ϕ_4+dt*b3*k3_ϕ_4+dt*b4*k4_ϕ_4+dt*b5*k5_ϕ_4 
    
            # W_1_1=W_1_1+dt*b1*k1_W_1_1+dt*b3*k3_W_1_1+dt*b4*k4_W_1_1+dt*b5*k5_W_1_1 
            W_1_2=W_1_2+dt*b1*k1_W_1_2+dt*b3*k3_W_1_2+dt*b4*k4_W_1_2+dt*b5*k5_W_1_2 
            W_1_3=W_1_3+dt*b1*k1_W_1_3+dt*b3*k3_W_1_3+dt*b4*k4_W_1_3+dt*b5*k5_W_1_3 
            W_1_4=W_1_4+dt*b1*k1_W_1_4+dt*b3*k3_W_1_4+dt*b4*k4_W_1_4+dt*b5*k5_W_1_4 
            # W_2_1=W_2_1+dt*b1*k1_W_2_1+dt*b3*k3_W_2_1+dt*b4*k4_W_2_1+dt*b5*k5_W_2_1 
            W_2_2=W_2_2+dt*b1*k1_W_2_2+dt*b3*k3_W_2_2+dt*b4*k4_W_2_2+dt*b5*k5_W_2_2 
            W_2_3=W_2_3+dt*b1*k1_W_2_3+dt*b3*k3_W_2_3+dt*b4*k4_W_2_3+dt*b5*k5_W_2_3 
            W_2_4=W_2_4+dt*b1*k1_W_2_4+dt*b3*k3_W_2_4+dt*b4*k4_W_2_4+dt*b5*k5_W_2_4 
            # W_3_1=W_3_1+dt*b1*k1_W_3_1+dt*b3*k3_W_3_1+dt*b4*k4_W_3_1+dt*b5*k5_W_3_1 
            W_3_2=W_3_2+dt*b1*k1_W_3_2+dt*b3*k3_W_3_2+dt*b4*k4_W_3_2+dt*b5*k5_W_3_2 
            W_3_3=W_3_3+dt*b1*k1_W_3_3+dt*b3*k3_W_3_3+dt*b4*k4_W_3_3+dt*b5*k5_W_3_3 
            W_3_4=W_3_4+dt*b1*k1_W_3_4+dt*b3*k3_W_3_4+dt*b4*k4_W_3_4+dt*b5*k5_W_3_4 
            # Y_1=Y_1+dt*b1*k1_Y_1+dt*b3*k3_Y_1+dt*b4*k4_Y_1+dt*b5*k5_Y_1 
            Y_2=Y_2+dt*b1*k1_Y_2+dt*b3*k3_Y_2+dt*b4*k4_Y_2+dt*b5*k5_Y_2 
            Y_3=Y_3+dt*b1*k1_Y_3+dt*b3*k3_Y_3+dt*b4*k4_Y_3+dt*b5*k5_Y_3 
            Y_4=Y_4+dt*b1*k1_Y_4+dt*b3*k3_Y_4+dt*b4*k4_Y_4+dt*b5*k5_Y_4 
            Γ_1=Γ_1+dt*b1*k1_Γ_1+dt*b3*k3_Γ_1+dt*b4*k4_Γ_1+dt*b5*k5_Γ_1 
            Γ_2=Γ_2+dt*b1*k1_Γ_2+dt*b3*k3_Γ_2+dt*b4*k4_Γ_2+dt*b5*k5_Γ_2 
            Γ_3=Γ_3+dt*b1*k1_Γ_3+dt*b3*k3_Γ_3+dt*b4*k4_Γ_3+dt*b5*k5_Γ_3 
            Σ=Σ+dt*b1*k1_Σ+dt*b3*k3_Σ+dt*b4*k4_Σ+dt*b5*k5_Σ
    
            dϕ_1_dt=dϕ_1_dt+dt*b1*kt1_ϕ_1+dt*b3*kt3_ϕ_1+dt*b4*kt4_ϕ_1+dt*b5*kt5_ϕ_1
            dϕ_2_dt=dϕ_2_dt+dt*b1*kt1_ϕ_2+dt*b3*kt3_ϕ_2+dt*b4*kt4_ϕ_2+dt*b5*kt5_ϕ_2 
            dϕ_3_dt=dϕ_3_dt+dt*b1*kt1_ϕ_3+dt*b3*kt3_ϕ_3+dt*b4*kt4_ϕ_3+dt*b5*kt5_ϕ_3 
            dϕ_4_dt=dϕ_4_dt+dt*b1*kt1_ϕ_4+dt*b3*kt3_ϕ_4+dt*b4*kt4_ϕ_4+dt*b5*kt5_ϕ_4 
    
            # W_1_1+dt*b1*kt1_W_1_1+dt*b3*kt3_W_1_1+dt*b4*kt4_W_1_1+dt*b5*kt5_W_1_1 
            dW_1_2_dt=dW_1_2_dt+dt*b1*kt1_W_1_2+dt*b3*kt3_W_1_2+dt*b4*kt4_W_1_2+dt*b5*kt5_W_1_2 
            dW_1_3_dt=dW_1_3_dt+dt*b1*kt1_W_1_3+dt*b3*kt3_W_1_3+dt*b4*kt4_W_1_3+dt*b5*kt5_W_1_3 
            dW_1_4_dt=dW_1_4_dt+dt*b1*kt1_W_1_4+dt*b3*kt3_W_1_4+dt*b4*kt4_W_1_4+dt*b5*kt5_W_1_4 
            # W_2_1+dt*b1*kt1_W_2_1+dt*b3*kt3_W_2_1+dt*b4*kt4_W_2_1+dt*b5*kt5_W_1_1 
            dW_2_2_dt=dW_2_2_dt+dt*b1*kt1_W_2_2+dt*b3*kt3_W_2_2+dt*b4*kt4_W_2_2+dt*b5*kt5_W_2_2 
            dW_2_3_dt=dW_2_3_dt+dt*b1*kt1_W_2_3+dt*b3*kt3_W_2_3+dt*b4*kt4_W_2_3+dt*b5*kt5_W_2_3 
            dW_2_4_dt=dW_2_4_dt+dt*b1*kt1_W_2_4+dt*b3*kt3_W_2_4+dt*b4*kt4_W_2_4+dt*b5*kt5_W_2_4 
            # W_3_1+dt*b1*kt1_W_3_1+dt*b3*kt3_W_3_1+dt*b4*kt4_W_3_1+dt*b5*kt5_W_1_1 
            dW_3_2_dt=dW_3_2_dt+dt*b1*kt1_W_3_2+dt*b3*kt3_W_3_2+dt*b4*kt4_W_3_2+dt*b5*kt5_W_3_2 
            dW_3_3_dt=dW_3_3_dt+dt*b1*kt1_W_3_3+dt*b3*kt3_W_3_3+dt*b4*kt4_W_3_3+dt*b5*kt5_W_3_3 
            dW_3_4_dt=dW_3_4_dt+dt*b1*kt1_W_3_4+dt*b3*kt3_W_3_4+dt*b4*kt4_W_3_4+dt*b5*kt5_W_3_4 
            # Y_1+dt*b1*kt1_Y_1+dt*b3*kt3_Y_1+dt*b4*kt4_Y_1+dt*b5*kt5_Y_1 
            dY_2_dt=dY_2_dt+dt*b1*kt1_Y_2+dt*b3*kt3_Y_2+dt*b4*kt4_Y_2+dt*b5*kt5_Y_2 
            dY_3_dt=dY_3_dt+dt*b1*kt1_Y_3+dt*b3*kt3_Y_3+dt*b4*kt4_Y_3+dt*b5*kt5_Y_3 
            dY_4_dt=dY_4_dt+dt*b1*kt1_Y_4+dt*b3*kt3_Y_4+dt*b4*kt4_Y_4+dt*b5*kt5_Y_4 
            # Γ_1+dt*b1*kt1_Γ_1+dt*b3*kt3_Γ_1+dt*b4*kt4_Γ_1+dt*b5*kt5_Γ_1 
            # Γ_2+dt*b1*kt1_Γ_2+dt*b3*kt3_Γ_2+dt*b4*kt4_Γ_2+dt*b5*kt5_Γ_2 
            # Γ_3+dt*b1*kt1_Γ_3+dt*b3*kt3_Γ_3+dt*b4*kt4_Γ_3+dt*b5*kt5_Γ_3 
            # Σ+dt*b1*kt1_Σ+dt*b3*kt3_Σ+dt*b4*kt4_Σ+dt*b5*kt5_Σ
      
            # synchronize()

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
                
                B_x_fft = Array(fft(B_x))
                B_y_fft = Array(fft(B_y))
                B_z_fft = Array(fft(B_z))
                
                snp_idx = snp_idx+1
                B_fft[snp_idx,:,:] = convolve_1d((real(conj.(B_x_fft).*B_x_fft.+
                conj.(B_y_fft).*B_y_fft.+
                conj.(B_z_fft).*B_z_fft)),Nx,Ny,Nz,spec_cut[1],spec_cut[2],spec_cut[3])

                #Sum energies
                total_energies[snp_idx,1] = sum(Array(PE_Phi))
                total_energies[snp_idx,2] = sum(Array(KE_Phi))
                total_energies[snp_idx,3] = sum(Array(GE_Phi))
                total_energies[snp_idx,4] = sum(Array(ElectricE_W))
                total_energies[snp_idx,5] = sum(Array(MagneticE_W))
                total_energies[snp_idx,6] = sum(Array(ElectricE_Y))
                total_energies[snp_idx,7] = sum(Array(MagneticE_Y))

                # println("test:",Array(ϕ_1_n)[3,5,3])
                # exit()
                # plot_1=contourf(z,x,(Array(ϕ_1)[:,Ny÷2,:]).^2+(Array(ϕ_2)[:,Ny÷2,:]).^2+(Array(ϕ_3)[:,Ny÷2,:]).^2+(Array(ϕ_4)[:,Ny÷2,:]).^2)
                # plot(plot_1,title=string("it:",it),dpi=600)
                # plot_1=contourf(z,x,(Array(ϕ_1)[:,Ny÷2,:]).^2+(Array(ϕ_2)[:,Ny÷2,:]).^2+(Array(ϕ_3)[:,Ny÷2,:]).^2+(Array(ϕ_4)[:,Ny÷2,:]).^2)
                plot_1=contourf(z,x,(Array(PE_Phi)[:,Ny÷2,:]),title="PE")
                plot_2=contourf(z,x,(Array(KE_Phi)[:,Ny÷2,:]),title="KE")
                # plot_3=contourf(z,x,(Array(GE_Phi)[:,Ny÷2,:]),title="GE")

                plot_3=plot(B_fft[snp_idx,2:end,1],(((B_fft[snp_idx,2:end,1]).^2)./((2*pi)^3*Nx^2)).*B_fft[snp_idx,2:end,2],xscale=:log10,yscale=:log10,minorgrid=true)
                # plot_4=contourf(z,x,(Array(ElectricE_W)[:,Ny÷2,:]+Array(MagneticE_W)[:,Ny÷2,:]+Array(ElectricE_Y)[:,Ny÷2,:]+Array(MagneticE_Y)[:,Ny÷2,:]),title="WY E")
                plot_4 = plot(range(1,snp_idx,step=1),[total_energies[1:snp_idx,1] total_energies[1:snp_idx,2] total_energies[1:snp_idx,3] total_energies[1:snp_idx,4] total_energies[1:snp_idx,5] total_energies[1:snp_idx,6] total_energies[1:snp_idx,7]],
                label=["PE" "KE" "GE" "EEW" "MEW" "EEY" "MEY"],xlims=(0,nte))
                plot(plot_1,plot_2,plot_3,plot_4,layout=4,plot_title=string("it:",it),dpi=600)
                frame(anim)
            end

        end

    end
    gif(anim, "EW3d_rkf45.gif", fps = 60)
    println("test:",Array(ϕ_1)[3,5,3])
    CUDA.memory_status()
    # println(size(ϕ_2))
    # println(size(findall(Array(ϕ_1_n).!=0)))
    # println(Array(out)[5,1,1])
    return
    
end
run()
CUDA.memory_status()
