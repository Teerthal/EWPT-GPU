using CUDA#, CuArrays
using Random
using StatsBase
using Distributions
using Plots
using CUDA.CUFFT
using Statistics
import FFTW.bfft
using HDF5

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

include("parameters.jl")
using .parameters

include("diff_scheme.jl")
using .differentiations

include("coordinates.jl")
using .coords

include("cov_derivs_temporal.jl")
using .covariant_derivatives

include("field_strengths.jl")
using .f_strengths

include("convenients.jl")
using .convenience_functions

include("spec_routines.jl")
using .spec_convolver

@views function rk4_step!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    Γ_1,Γ_2,Γ_3,Σ,
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
    gw,gy,gp2,vev,lambda,dx,Jex,γ)
    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
              (blockIdx().y - 1) * blockDim().y + threadIdx().y,
              (blockIdx().z - 1) * blockDim().z + threadIdx().z
    
    # 2-24-24: Checked and correct sp derivative calls below:
    # found and resolved incorrect calls to second order spatial
    # derivatives below

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
        d2ϕ_3_dx2=d2fdx2(ϕ_3,i,j,k,0.,dx)
        d2ϕ_4_dx2=d2fdx2(ϕ_4,i,j,k,0.,dx)

        d2ϕ_1_dy2=d2fdy2(ϕ_1,i,j,k,0.,dx)
        d2ϕ_2_dy2=d2fdy2(ϕ_2,i,j,k,0.,dx)
        d2ϕ_3_dy2=d2fdy2(ϕ_3,i,j,k,0.,dx)
        d2ϕ_4_dy2=d2fdy2(ϕ_4,i,j,k,0.,dx)

        d2ϕ_1_dz2=d2fdz2(ϕ_1,i,j,k,0.,dx)
        d2ϕ_2_dz2=d2fdz2(ϕ_2,i,j,k,0.,dx)
        d2ϕ_3_dz2=d2fdz2(ϕ_3,i,j,k,0.,dx)
        d2ϕ_4_dz2=d2fdz2(ϕ_4,i,j,k,0.,dx)

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

    #End spatial derivatives#

    # 2-24-24 : Checked all calls to cov derivatives #
    ##Covariant Derivatives##Temporal gauge
    #In temporal gauge, can drop a alot of terms
    
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

    #END covarian derivates#

    # 2-24-24 : Checked all calls to field strengths #

    # Field Strengths # Temporal gauge: can drop a lot of the terms 
    #or enter them in expressions explicity#

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
    #END field strengths#

    # 2-24-24: Checked all flux expressions#

    # kt_1 expressions
    @inbounds kt_ϕ_1[i,j,k] = (d2ϕ_1_dx2+d2ϕ_1_dy2+d2ϕ_1_dz2-
        0.5*gw*(((-W_1_2[i,j,k]*dϕ_4_dx)-(W_1_3[i,j,k]*dϕ_4_dy)-(W_1_4[i,j,k]*dϕ_4_dz))-
        ((-W_2_2[i,j,k]*dϕ_3_dx)-(W_2_3[i,j,k]*dϕ_3_dy)-(W_2_4[i,j,k]*dϕ_3_dz))+
        ((-W_3_2[i,j,k]*dϕ_2_dx)-(W_3_3[i,j,k]*dϕ_2_dy)-(W_3_4[i,j,k]*dϕ_2_dz)))-
        0.5*gy*(-Y_2[i,j,k]*dϕ_2_dx-Y_3[i,j,k]*dϕ_2_dy-Y_4[i,j,k]*dϕ_2_dz)-
        0.5*gw*((-W_1_2[i,j,k]*Dx_ϕ_4-W_1_3[i,j,k]*Dy_ϕ_4-W_1_4[i,j,k]*Dz_ϕ_4)-
        (-W_2_2[i,j,k]*Dx_ϕ_3-W_2_3[i,j,k]*Dy_ϕ_3-W_2_4[i,j,k]*Dz_ϕ_3)+
        (-W_3_2[i,j,k]*Dx_ϕ_2-W_3_3[i,j,k]*Dy_ϕ_2-W_3_4[i,j,k]*Dz_ϕ_2))-
        0.5*gy*(-Y_2[i,j,k]*Dx_ϕ_2-Y_3[i,j,k]*Dy_ϕ_2-Y_4[i,j,k]*Dz_ϕ_2)-
        2.0*lambda*(ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2-vev^2)*ϕ_1[i,j,k]+
        0.5*((gw*Γ_3[i,j,k]+gy*Σ[i,j,k])*ϕ_2[i,j,k]-gw*Γ_2[i,j,k]*ϕ_3[i,j,k]+gw*Γ_1[i,j,k]*ϕ_4[i,j,k])-
        0.5*Jex*ϕ_1[i,j,k]-γ*dϕ_1_dt[i,j,k]^2/(ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2))


    @inbounds kt_ϕ_2[i,j,k] = (d2ϕ_2_dx2+d2ϕ_2_dy2+d2ϕ_2_dz2+
        0.5*gw*((-W_1_2[i,j,k]*dϕ_3_dx-W_1_3[i,j,k]*dϕ_3_dy-W_1_4[i,j,k]*dϕ_3_dz)+
        (-W_2_2[i,j,k]*dϕ_4_dx-W_2_3[i,j,k]*dϕ_4_dy-W_2_4[i,j,k]*dϕ_4_dz)+
        (-W_3_2[i,j,k]*dϕ_1_dx-W_3_3[i,j,k]*dϕ_1_dy-W_3_4[i,j,k]*dϕ_1_dz))+
        0.5*gy*(-Y_2[i,j,k]*dϕ_1_dx-Y_3[i,j,k]*dϕ_1_dy-Y_4[i,j,k]*dϕ_1_dz)+
        0.5*gw*((-W_1_2[i,j,k]*Dx_ϕ_3-W_1_3[i,j,k]*Dy_ϕ_3-W_1_4[i,j,k]*Dz_ϕ_3)+
        (-W_2_2[i,j,k]*Dx_ϕ_4-W_2_3[i,j,k]*Dy_ϕ_4-W_2_4[i,j,k]*Dz_ϕ_4)+
        (-W_3_2[i,j,k]*Dx_ϕ_1-W_3_3[i,j,k]*Dy_ϕ_1-W_3_4[i,j,k]*Dz_ϕ_1))+
        0.5*gy*(-Y_2[i,j,k]*Dx_ϕ_1-Y_3[i,j,k]*Dy_ϕ_1-Y_4[i,j,k]*Dz_ϕ_1)-
        2.0*lambda*(ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2-vev^2)*ϕ_2[i,j,k]-
        0.5*((gw*Γ_3[i,j,k]+gy*Σ[i,j,k])*ϕ_1[i,j,k]+gw*Γ_1[i,j,k]*ϕ_3[i,j,k]+gw*Γ_2[i,j,k]*ϕ_4[i,j,k])-
        0.5*Jex*ϕ_2[i,j,k]-γ*dϕ_2_dt[i,j,k]^2/(ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2))

    @inbounds kt_ϕ_3[i,j,k] = (d2ϕ_3_dx2+d2ϕ_3_dy2+d2ϕ_3_dz2-
        0.5*gw*((-W_1_2[i,j,k]*dϕ_2_dx-W_1_3[i,j,k]*dϕ_2_dy-W_1_4[i,j,k]*dϕ_2_dz)+
        (-W_2_2[i,j,k]*dϕ_1_dx-W_2_3[i,j,k]*dϕ_1_dy-W_2_4[i,j,k]*dϕ_1_dz)-
        (-W_3_2[i,j,k]*dϕ_4_dx-W_3_3[i,j,k]*dϕ_4_dy-W_3_4[i,j,k]*dϕ_4_dz))-
        0.5*gy*(-Y_2[i,j,k]*dϕ_4_dx-Y_3[i,j,k]*dϕ_4_dy-Y_4[i,j,k]*dϕ_4_dz)-
        0.5*gw*((-W_1_2[i,j,k]*Dx_ϕ_2-W_1_3[i,j,k]*Dy_ϕ_2-W_1_4[i,j,k]*Dz_ϕ_2)+
        (-W_2_2[i,j,k]*Dx_ϕ_1-W_2_3[i,j,k]*Dy_ϕ_1-W_2_4[i,j,k]*Dz_ϕ_1)-
        (-W_3_2[i,j,k]*Dx_ϕ_4-W_3_3[i,j,k]*Dy_ϕ_4-W_3_4[i,j,k]*Dz_ϕ_4))-
        0.5*gy*(-Y_2[i,j,k]*Dx_ϕ_4-Y_3[i,j,k]*Dy_ϕ_4-Y_4[i,j,k]*Dz_ϕ_4)-
        2.0*lambda*(ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2-vev^2)*ϕ_3[i,j,k]+
        0.5*((-gw*Γ_3[i,j,k]+gy*Σ[i,j,k])*ϕ_4[i,j,k]+gw*Γ_2[i,j,k]*ϕ_1[i,j,k]+gw*Γ_1[i,j,k]*ϕ_2[i,j,k])-
        0.5*Jex*ϕ_3[i,j,k]-γ*dϕ_3_dt[i,j,k]^2/(ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2))

    @inbounds kt_ϕ_4[i,j,k] = (d2ϕ_4_dx2+d2ϕ_4_dy2+d2ϕ_4_dz2+
        0.5*gw*((-W_1_2[i,j,k]*dϕ_1_dx-W_1_3[i,j,k]*dϕ_1_dy-W_1_4[i,j,k]*dϕ_1_dz)-
        (-W_2_2[i,j,k]*dϕ_2_dx-W_2_3[i,j,k]*dϕ_2_dy-W_2_4[i,j,k]*dϕ_2_dz)-
        (-W_3_2[i,j,k]*dϕ_3_dx-W_3_3[i,j,k]*dϕ_3_dy-W_3_4[i,j,k]*dϕ_3_dz))+
        0.5*gy*(-Y_2[i,j,k]*dϕ_3_dx-Y_3[i,j,k]*dϕ_3_dy-Y_4[i,j,k]*dϕ_3_dz)+
        0.5*gw*((-W_1_2[i,j,k]*Dx_ϕ_1-W_1_3[i,j,k]*Dy_ϕ_1-W_1_4[i,j,k]*Dz_ϕ_1)-
        (-W_2_2[i,j,k]*Dx_ϕ_2-W_2_3[i,j,k]*Dy_ϕ_2-W_2_4[i,j,k]*Dz_ϕ_2)-
        (-W_3_2[i,j,k]*Dx_ϕ_3-W_3_3[i,j,k]*Dy_ϕ_3-W_3_4[i,j,k]*Dz_ϕ_3))+
        0.5*gy*(-Y_2[i,j,k]*Dx_ϕ_3-Y_3[i,j,k]*Dy_ϕ_3-Y_4[i,j,k]*Dz_ϕ_3)-
        2.0*lambda*(ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2-vev^2)*ϕ_4[i,j,k]-
        0.5*((-gw*Γ_3[i,j,k]+gy*Σ[i,j,k])*ϕ_3[i,j,k]+gw*Γ_1[i,j,k]*ϕ_1[i,j,k]-gw*Γ_2[i,j,k]*ϕ_2[i,j,k])-
        0.5*Jex*ϕ_4[i,j,k]-γ*dϕ_4_dt[i,j,k]^2/(ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2))

    @inbounds kt_W_1_2[i,j,k] = (d2W_1_2_dx2+d2W_1_2_dy2+d2W_1_2_dz2+
        gw*(-(dW_2_2_dx*W_3_2[i,j,k]-dW_3_2_dx*W_2_2[i,j,k])-
        (dW_2_2_dy*W_3_3[i,j,k]-dW_3_2_dy*W_2_3[i,j,k])-
        (dW_2_2_dz*W_3_4[i,j,k]-dW_3_2_dz*W_2_4[i,j,k])-
        (W_2_3[i,j,k]*W_3_23-W_3_3[i,j,k]*W_2_23)-
        (W_2_4[i,j,k]*W_3_24-W_3_4[i,j,k]*W_2_24))+
        gw*(ϕ_1[i,j,k]*Dx_ϕ_4-ϕ_2[i,j,k]*Dx_ϕ_3+ϕ_3[i,j,k]*Dx_ϕ_2-ϕ_4[i,j,k]*Dx_ϕ_1)-
        dΓ_1_dx-gw*(W_2_2[i,j,k]*Γ_3[i,j,k]-W_3_2[i,j,k]*Γ_2[i,j,k])-γ*dW_1_2_dt[i,j,k])

    @inbounds kt_W_1_3[i,j,k] = (d2W_1_3_dx2+d2W_1_3_dy2+d2W_1_3_dz2+
        gw*(-(dW_2_3_dx*W_3_2[i,j,k]-dW_3_3_dx*W_2_2[i,j,k])-
        (dW_2_3_dy*W_3_3[i,j,k]-dW_3_3_dy*W_2_3[i,j,k])-
        (dW_2_3_dz*W_3_4[i,j,k]-dW_3_3_dz*W_2_4[i,j,k])-
        (W_2_2[i,j,k]*(-W_3_23)-W_3_2[i,j,k]*(-W_2_23))-
        (W_2_4[i,j,k]*W_3_34-W_3_4[i,j,k]*W_2_34))+
        gw*(ϕ_1[i,j,k]*Dy_ϕ_4-ϕ_2[i,j,k]*Dy_ϕ_3+ϕ_3[i,j,k]*Dy_ϕ_2-ϕ_4[i,j,k]*Dy_ϕ_1)-
        dΓ_1_dy-gw*(W_2_3[i,j,k]*Γ_3[i,j,k]-W_3_3[i,j,k]*Γ_2[i,j,k])-γ*dW_1_3_dt[i,j,k])

    @inbounds kt_W_1_4[i,j,k] = (d2W_1_4_dx2+d2W_1_4_dy2+d2W_1_4_dz2+
        gw*(-(dW_2_4_dx*W_3_2[i,j,k]-dW_3_4_dx*W_2_2[i,j,k])-
        (dW_2_4_dy*W_3_3[i,j,k]-dW_3_4_dy*W_2_3[i,j,k])-
        (dW_2_4_dz*W_3_4[i,j,k]-dW_3_4_dz*W_2_4[i,j,k])-
        (W_2_2[i,j,k]*(-W_3_24)-W_3_2[i,j,k]*(-W_2_24))-
        (W_2_3[i,j,k]*(-W_3_34)-W_3_3[i,j,k]*(-W_2_34)))+
        gw*(ϕ_1[i,j,k]*Dz_ϕ_4-ϕ_2[i,j,k]*Dz_ϕ_3+ϕ_3[i,j,k]*Dz_ϕ_2-ϕ_4[i,j,k]*Dz_ϕ_1)-
        dΓ_1_dz-gw*(W_2_4[i,j,k]*Γ_3[i,j,k]-W_3_4[i,j,k]*Γ_2[i,j,k])-γ*dW_1_4_dt[i,j,k])

    @inbounds kt_W_2_2[i,j,k] = (d2W_2_2_dx2+d2W_2_2_dy2+d2W_2_2_dz2+
        gw*(-(dW_3_2_dx*W_1_2[i,j,k]-dW_1_2_dx*W_3_2[i,j,k])-
        (dW_3_2_dy*W_1_3[i,j,k]-dW_1_2_dy*W_3_3[i,j,k])-
        (dW_3_2_dz*W_1_4[i,j,k]-dW_1_2_dz*W_3_4[i,j,k])-
        (W_3_3[i,j,k]*W_1_23-W_1_3[i,j,k]*W_3_23)-
        (W_3_4[i,j,k]*W_1_24-W_1_4[i,j,k]*W_3_24))+
        gw*(-ϕ_1[i,j,k]*Dx_ϕ_3-ϕ_2[i,j,k]*Dx_ϕ_4+ϕ_3[i,j,k]*Dx_ϕ_1+ϕ_4[i,j,k]*Dx_ϕ_2)-
        dΓ_2_dx-gw*(W_3_2[i,j,k]*Γ_1[i,j,k]-W_1_2[i,j,k]*Γ_3[i,j,k])-γ*dW_2_2_dt[i,j,k])

    @inbounds kt_W_2_3[i,j,k] = (d2W_2_3_dx2+d2W_2_3_dy2+d2W_2_3_dz2+
        gw*(-(dW_3_3_dx*W_1_2[i,j,k]-dW_1_3_dx*W_3_2[i,j,k])-
        (dW_3_3_dy*W_1_3[i,j,k]-dW_1_3_dy*W_3_3[i,j,k])-
        (dW_3_3_dz*W_1_4[i,j,k]-dW_1_3_dz*W_3_4[i,j,k])-
        (W_3_2[i,j,k]*(-W_1_23)-W_1_2[i,j,k]*(-W_3_23))-
        (W_3_4[i,j,k]*(W_1_34)-W_1_4[i,j,k]*W_3_34))+
        gw*(-ϕ_1[i,j,k]*Dy_ϕ_3-ϕ_2[i,j,k]*Dy_ϕ_4+ϕ_3[i,j,k]*Dy_ϕ_1+ϕ_4[i,j,k]*Dy_ϕ_2)-
        dΓ_2_dy-gw*(W_3_3[i,j,k]*Γ_1[i,j,k]-W_1_3[i,j,k]*Γ_3[i,j,k])-γ*dW_2_3_dt[i,j,k])

    @inbounds kt_W_2_4[i,j,k] = (d2W_2_4_dx2+d2W_2_4_dy2+d2W_2_4_dz2+
        gw*(-(dW_3_4_dx*W_1_2[i,j,k]-dW_1_4_dx*W_3_2[i,j,k])-
        (dW_3_4_dy*W_1_3[i,j,k]-dW_1_4_dy*W_3_3[i,j,k])-
        (dW_3_4_dz*W_1_4[i,j,k]-dW_1_4_dz*W_3_4[i,j,k])-
        (W_3_2[i,j,k]*(-W_1_24)-W_1_2[i,j,k]*(-W_3_24))-
        (W_3_3[i,j,k]*(-W_1_34)-W_1_3[i,j,k]*(-W_3_34)))+
        gw*(-ϕ_1[i,j,k]*Dz_ϕ_3-ϕ_2[i,j,k]*Dz_ϕ_4+ϕ_3[i,j,k]*Dz_ϕ_1+ϕ_4[i,j,k]*Dz_ϕ_2)-
        dΓ_2_dz-gw*(W_3_4[i,j,k]*Γ_1[i,j,k]-W_1_4[i,j,k]*Γ_3[i,j,k])-γ*dW_2_4_dt[i,j,k])

    @inbounds kt_W_3_2[i,j,k] = (d2W_3_2_dx2+d2W_3_2_dy2+d2W_3_2_dz2+
        gw*(-(dW_1_2_dx*W_2_2[i,j,k]-dW_2_2_dx*W_1_2[i,j,k])-
        (dW_1_2_dy*W_2_3[i,j,k]-dW_2_2_dy*W_1_3[i,j,k])-
        (dW_1_2_dz*W_2_4[i,j,k]-dW_2_2_dz*W_1_4[i,j,k])-
        (W_1_3[i,j,k]*W_2_23-W_2_3[i,j,k]*W_1_23)-
        (W_1_4[i,j,k]*W_2_24-W_2_4[i,j,k]*W_1_24))+
        gw*(ϕ_1[i,j,k]*Dx_ϕ_2-ϕ_2[i,j,k]*Dx_ϕ_1-ϕ_3[i,j,k]*Dx_ϕ_4+ϕ_4[i,j,k]*Dx_ϕ_3)-
        dΓ_3_dx-gw*(W_1_2[i,j,k]*Γ_2[i,j,k]-W_2_2[i,j,k]*Γ_1[i,j,k])-γ*dW_3_2_dt[i,j,k])

    @inbounds kt_W_3_3[i,j,k] = (d2W_3_3_dx2+d2W_3_3_dy2+d2W_3_3_dz2+
        gw*(-(dW_1_3_dx*W_2_2[i,j,k]-dW_2_3_dx*W_1_2[i,j,k])-
            (dW_1_3_dy*W_2_3[i,j,k]-dW_2_3_dy*W_1_3[i,j,k])-
            (dW_1_3_dz*W_2_4[i,j,k]-dW_2_3_dz*W_1_4[i,j,k])-
            (W_1_2[i,j,k]*(-W_2_23)-W_2_2[i,j,k]*(-W_1_23))-
            (W_1_4[i,j,k]*W_2_34-W_2_4[i,j,k]*(W_1_34)))+
        gw*(ϕ_1[i,j,k]*Dy_ϕ_2-ϕ_2[i,j,k]*Dy_ϕ_1-ϕ_3[i,j,k]*Dy_ϕ_4+ϕ_4[i,j,k]*Dy_ϕ_3)-
        dΓ_3_dy-gw*(W_1_3[i,j,k]*Γ_2[i,j,k]-W_2_3[i,j,k]*Γ_1[i,j,k])-γ*dW_3_3_dt[i,j,k])

    @inbounds kt_W_3_4[i,j,k] = (d2W_3_4_dx2+d2W_3_4_dy2+d2W_3_4_dz2+
        gw*(-(dW_1_4_dx*W_2_2[i,j,k]-dW_2_4_dx*W_1_2[i,j,k])-
            (dW_1_4_dy*W_2_3[i,j,k]-dW_2_4_dy*W_1_3[i,j,k])-
            (dW_1_4_dz*W_2_4[i,j,k]-dW_2_4_dz*W_1_4[i,j,k])-
            (W_1_2[i,j,k]*(-W_2_24)-W_2_2[i,j,k]*(-W_1_24))-
            (W_1_3[i,j,k]*(-W_2_34)-W_2_3[i,j,k]*(-W_1_34)))+
        gw*(ϕ_1[i,j,k]*Dz_ϕ_2-ϕ_2[i,j,k]*Dz_ϕ_1-ϕ_3[i,j,k]*Dz_ϕ_4+ϕ_4[i,j,k]*Dz_ϕ_3)-
        dΓ_3_dz-gw*(W_1_4[i,j,k]*Γ_2[i,j,k]-W_2_4[i,j,k]*Γ_1[i,j,k])-γ*dW_3_4_dt[i,j,k])

    @inbounds kt_Y_2[i,j,k] = ((d2Y_2_dx2+d2Y_2_dy2+d2Y_2_dz2+
        gy*(ϕ_1[i,j,k]*Dx_ϕ_2-ϕ_2[i,j,k]*Dx_ϕ_1+ϕ_3[i,j,k]*Dx_ϕ_4-ϕ_4[i,j,k]*Dx_ϕ_3)-dΣ_dx-γ*dY_2_dt[i,j,k]))

    @inbounds kt_Y_3[i,j,k] = ((d2Y_3_dx2+d2Y_3_dy2+d2Y_3_dz2+
        gy*(ϕ_1[i,j,k]*Dy_ϕ_2-ϕ_2[i,j,k]*Dy_ϕ_1+ϕ_3[i,j,k]*Dy_ϕ_4-ϕ_4[i,j,k]*Dy_ϕ_3)-dΣ_dy-γ*dY_3_dt[i,j,k]))

    @inbounds kt_Y_4[i,j,k] = ((d2Y_4_dx2+d2Y_4_dy2+d2Y_4_dz2+
        gy*(ϕ_1[i,j,k]*Dz_ϕ_2-ϕ_2[i,j,k]*Dz_ϕ_1+ϕ_3[i,j,k]*Dz_ϕ_4-ϕ_4[i,j,k]*Dz_ϕ_3)-dΣ_dz-γ*dY_4_dt[i,j,k]))


    # s(Γ_1)=
    @inbounds k_Γ_1[i,j,k] =((1.0.-gp2).*(dfdx(dW_1_2_dt,i,j,k,0.,dx) .+
    dfdy(dW_1_3_dt,i,j,k,0.,dx) .+ dfdz(dW_1_4_dt,i,j,k,0.,dx)).+
        gp2 .*gw.*(
        -((W_2_2[i,j,k]).*(dW_3_2_dt[i,j,k]).-
        (W_3_2[i,j,k]).*(dW_2_2_dt[i,j,k])).-
        ((W_2_3[i,j,k]).*(dW_3_3_dt[i,j,k]).-
        (W_3_3[i,j,k]).*(dW_2_3_dt[i,j,k])).-
        ((W_2_4[i,j,k]).*(dW_3_4_dt[i,j,k]).-
        (W_3_4[i,j,k]).*(dW_2_4_dt[i,j,k]))).+
    # c charge from Higgs: 
        gp2 .*gw.*((ϕ_1[i,j,k]).*((dϕ_4_dt[i,j,k])).-
        (ϕ_4[i,j,k]).*((dϕ_1_dt[i,j,k])).+
        (ϕ_3[i,j,k]).*((dϕ_2_dt[i,j,k])).-
        (ϕ_2[i,j,k]).*((dϕ_3_dt[i,j,k]))))

    # s(Γ_2)=
    @inbounds k_Γ_2[i,j,k] =((1.0.-gp2).*(dfdx(dW_2_2_dt,i,j,k,0.,dx) .+
    dfdy(dW_2_3_dt,i,j,k,0.,dx) .+ dfdz(dW_2_4_dt,i,j,k,0.,dx)).+
        gp2 .*gw.*(
        -((W_3_2[i,j,k]).*(dW_1_2_dt[i,j,k]).-
        (W_1_2[i,j,k]).*(dW_3_2_dt[i,j,k])).-
        ((W_3_3[i,j,k]).*(dW_1_3_dt[i,j,k]).-
        (W_1_3[i,j,k]).*(dW_3_3_dt[i,j,k])).-
        ((W_3_4[i,j,k]).*(dW_1_4_dt[i,j,k]).-
        (W_1_4[i,j,k]).*(dW_3_4_dt[i,j,k]))).+
    # c charge from Higgs: 
        gp2 .*gw.*((ϕ_3[i,j,k]).*((dϕ_1_dt[i,j,k])).-
        (ϕ_1[i,j,k]).*((dϕ_3_dt[i,j,k])).+
        (ϕ_4[i,j,k]).*((dϕ_2_dt[i,j,k])).-
        (ϕ_2[i,j,k]).*((dϕ_4_dt[i,j,k]))))

    # s(Γ_3)=
    @inbounds k_Γ_3[i,j,k] =((1.0.-gp2).*(dfdx(dW_3_2_dt,i,j,k,0.,dx) .+
    dfdy(dW_3_3_dt,i,j,k,0.,dx) .+ dfdz(dW_3_4_dt,i,j,k,0.,dx)).+
        gp2 .*gw.*(
        -((W_1_2[i,j,k]).*(dW_2_2_dt[i,j,k]).-
        (W_2_2[i,j,k]).*(dW_1_2_dt[i,j,k])).-
        ((W_1_3[i,j,k]).*(dW_2_3_dt[i,j,k]).-
        (W_2_3[i,j,k]).*(dW_1_3_dt[i,j,k])).-
        ((W_1_4[i,j,k]).*(dW_2_4_dt[i,j,k]).-
        (W_2_4[i,j,k]).*(dW_1_4_dt[i,j,k]))).+
    # c current from Higgs: 
        gp2 .*gw.*((ϕ_1[i,j,k]).*((dϕ_2_dt[i,j,k])).-
        (ϕ_2[i,j,k]).*((dϕ_1_dt[i,j,k])).+
        (ϕ_4[i,j,k]).*((dϕ_3_dt[i,j,k])).-
        (ϕ_3[i,j,k]).*((dϕ_4_dt[i,j,k]))))

    # s(Σ)=
    @inbounds k_Σ[i,j,k] =((1.0.-gp2).*(dfdx(dY_2_dt,i,j,k,0.,dx) .+
    dfdy(dY_3_dt,i,j,k,0.,dx) .+ dfdz(dY_4_dt,i,j,k,0.,dx)).+
    # c current from Higgs: 
        gp2 .*gy.*((ϕ_1[i,j,k])*((dϕ_2_dt[i,j,k])).-
        (ϕ_2[i,j,k]).*((dϕ_1_dt[i,j,k])).+
        (ϕ_3[i,j,k]).*((dϕ_4_dt[i,j,k])).-
        (ϕ_4[i,j,k]).*((dϕ_3_dt[i,j,k]))))

    return
end

# @views function rk4_step!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
#     W_1_2,W_1_3,W_1_4,
#     W_2_2,W_2_3,W_2_4,
#     W_3_2,W_3_3,W_3_4,
#     Y_2,Y_3,Y_4,
#     Γ_1,Γ_2,Γ_3,Σ,
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
#     gw,gy,gp2,vev,lambda,dx,Jex,γ)
#     i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
#               (blockIdx().y - 1) * blockDim().y + threadIdx().y,
#               (blockIdx().z - 1) * blockDim().z + threadIdx().z
    
#     # 2-24-24: Checked and correct sp derivative calls below:
#     # found and resolved incorrect calls to second order spatial
#     # derivatives below

#     # #Spatial Derivatives
#     #     (dfdx(ϕ_1,i,j,k,0.,dx))=dfdx(ϕ_1,i,j,k,0.,dx)
#     #     (dfdx(ϕ_2,i,j,k,0.,dx))=dfdx(ϕ_2,i,j,k,0.,dx)
#     #     (dfdx(ϕ_3,i,j,k,0.,dx))=dfdx(ϕ_3,i,j,k,0.,dx)
#     #     (dfdx(ϕ_4,i,j,k,0.,dx))=dfdx(ϕ_4,i,j,k,0.,dx)
#     #     # @cuprintln((dfdx(ϕ_4,i,j,k,0.,dx)))
#     #     (dfdy(ϕ_1,i,j,k,0.,dx))=dfdy(ϕ_1,i,j,k,0.,dx)
#     #     (dfdy(ϕ_2,i,j,k,0.,dx))=dfdy(ϕ_2,i,j,k,0.,dx)
#     #     (dfdy(ϕ_3,i,j,k,0.,dx))=dfdy(ϕ_3,i,j,k,0.,dx)
#     #     (dfdy(ϕ_4,i,j,k,0.,dx))=dfdy(ϕ_4,i,j,k,0.,dx)

#     #     (dfdz(ϕ_1,i,j,k,0.,dx))=dfdz(ϕ_1,i,j,k,0.,dx)
#     #     (dfdz(ϕ_2,i,j,k,0.,dx))=dfdz(ϕ_2,i,j,k,0.,dx)
#     #     (dfdz(ϕ_3,i,j,k,0.,dx))=dfdz(ϕ_3,i,j,k,0.,dx)
#     #     (dfdz(ϕ_4,i,j,k,0.,dx))=dfdz(ϕ_4,i,j,k,0.,dx)

#     #     # dW_1_1_dx = dfdx(W_1_1,i,j,k,0.,dx)
#     #     (dfdx(W_1_2,i,j,k,0.,dx)) = dfdx(W_1_2,i,j,k,0.,dx)
#     #     (dfdx(W_1_3,i,j,k,0.,dx)) = dfdx(W_1_3,i,j,k,0.,dx)
#     #     (dfdx(W_1_4,i,j,k,0.,dx)) = dfdx(W_1_4,i,j,k,0.,dx)

#     #     # dW_1_1_dy = dfdy(W_1_1,i,j,k,0.,dx)
#     #     (dfdy(W_1_2,i,j,k,0.,dx)) = dfdy(W_1_2,i,j,k,0.,dx)
#     #     (dfdy(W_1_3,i,j,k,0.,dx)) = dfdy(W_1_3,i,j,k,0.,dx)
#     #     (dfdy(W_1_4,i,j,k,0.,dx)) = dfdy(W_1_4,i,j,k,0.,dx)

#     #     # dW_1_1_dz = dfdz(W_1_1,i,j,k,0.,dx)
#     #     (dfdz(W_1_2,i,j,k,0.,dx)) = dfdz(W_1_2,i,j,k,0.,dx)
#     #     (dfdz(W_1_3,i,j,k,0.,dx)) = dfdz(W_1_3,i,j,k,0.,dx)
#     #     (dfdz(W_1_4,i,j,k,0.,dx)) = dfdz(W_1_4,i,j,k,0.,dx)

#     #     # dW_2_1_dx = dfdx(W_2_1,i,j,k,0.,dx)
#     #     (dfdx(W_2_2,i,j,k,0.,dx)) = dfdx(W_2_2,i,j,k,0.,dx)
#     #     (dfdx(W_2_3,i,j,k,0.,dx)) = dfdx(W_2_3,i,j,k,0.,dx)
#     #     (dfdx(W_2_4,i,j,k,0.,dx)) = dfdx(W_2_4,i,j,k,0.,dx)

#     #     # dW_2_1_dy = dfdy(W_2_1,i,j,k,0.,dx)
#     #     (dfdy(W_2_2,i,j,k,0.,dx)) = dfdy(W_2_2,i,j,k,0.,dx)
#     #     (dfdy(W_2_3,i,j,k,0.,dx)) = dfdy(W_2_3,i,j,k,0.,dx)
#     #     (dfdy(W_2_4,i,j,k,0.,dx)) = dfdy(W_2_4,i,j,k,0.,dx)

#     #     # dW_2_1_dz = dfdz(W_2_1,i,j,k,0.,dx)
#     #     (dfdz(W_2_2,i,j,k,0.,dx)) = dfdz(W_2_2,i,j,k,0.,dx)
#     #     (dfdz(W_2_3,i,j,k,0.,dx)) = dfdz(W_2_3,i,j,k,0.,dx)
#     #     (dfdz(W_2_4,i,j,k,0.,dx)) = dfdz(W_2_4,i,j,k,0.,dx)

#     #     # dW_3_1_dx = dfdx(W_3_1,i,j,k,0.,dx)
#     #     (dfdx(W_3_2,i,j,k,0.,dx)) = dfdx(W_3_2,i,j,k,0.,dx)
#     #     (dfdx(W_3_3,i,j,k,0.,dx)) = dfdx(W_3_3,i,j,k,0.,dx)
#     #     (dfdx(W_3_4,i,j,k,0.,dx)) = dfdx(W_3_4,i,j,k,0.,dx)

#     #     # dW_3_1_dy = dfdy(W_3_1,i,j,k,0.,dx)
#     #     dfdy(W_3_2,i,j,k,0.,dx) = dfdy(W_3_2,i,j,k,0.,dx)
#     #     dfdy(W_3_3,i,j,k,0.,dx) = dfdy(W_3_3,i,j,k,0.,dx)
#     #     dfdy(W_3_4,i,j,k,0.,dx) = dfdy(W_3_4,i,j,k,0.,dx)

#     #     # dW_3_1_dz = dfdz(W_3_1,i,j,k,0.,dx)
#     #     (dfdz(W_3_2,i,j,k,0.,dx)) = dfdz(W_3_2,i,j,k,0.,dx)
#     #     (dfdz(W_3_3,i,j,k,0.,dx)) = dfdz(W_3_3,i,j,k,0.,dx)
#     #     (dfdz(W_3_4,i,j,k,0.,dx)) = dfdz(W_3_4,i,j,k,0.,dx)

#     #     # dY_1_dx = dfdx(Y_1,i,j,k,0.,dx)
#     #     (dfdx(Y_2,i,j,k,0.,dx)) = dfdx(Y_2,i,j,k,0.,dx)
#     #     (dfdx(Y_3,i,j,k,0.,dx)) = dfdx(Y_3,i,j,k,0.,dx)
#     #     (dfdx(Y_4,i,j,k,0.,dx)) = dfdx(Y_4,i,j,k,0.,dx)

#     #     # dY_1_dy = dfdy(Y_1,i,j,k,0.,dx)
#     #     (dfdy(Y_2,i,j,k,0.,dx)) = dfdy(Y_2,i,j,k,0.,dx)
#     #     (dfdy(Y_3,i,j,k,0.,dx)) = dfdy(Y_3,i,j,k,0.,dx)
#     #     (dfdy(Y_4,i,j,k,0.,dx)) = dfdy(Y_4,i,j,k,0.,dx)

#     #     # dY_1_dz = dfdz(Y_1,i,j,k,0.,dx)
#     #     (dfdz(Y_2,i,j,k,0.,dx)) = dfdz(Y_2,i,j,k,0.,dx)
#     #     (dfdz(Y_3,i,j,k,0.,dx)) = dfdz(Y_3,i,j,k,0.,dx)
#     #     (dfdz(Y_4,i,j,k,0.,dx)) = dfdz(Y_4,i,j,k,0.,dx)

#     #     (dfdx(Γ_1,i,j,k,0.,dx)) = dfdx(Γ_1,i,j,k,0.,dx)
#     #     (dfdy(Γ_1,i,j,k,0.,dx)) = dfdy(Γ_1,i,j,k,0.,dx)
#     #     (dfdz(Γ_1,i,j,k,0.,dx)) = dfdz(Γ_1,i,j,k,0.,dx)

#     #     (dfdx(Γ_2,i,j,k,0.,dx)) = dfdx(Γ_2,i,j,k,0.,dx)
#     #     (dfdy(Γ_2,i,j,k,0.,dx)) = dfdy(Γ_2,i,j,k,0.,dx)
#     #     (dfdz(Γ_2,i,j,k,0.,dx)) = dfdz(Γ_2,i,j,k,0.,dx)

#     #     (dfdx(Γ_3,i,j,k,0.,dx)) = dfdx(Γ_3,i,j,k,0.,dx)
#     #     (dfdy(Γ_3,i,j,k,0.,dx)) = dfdy(Γ_3,i,j,k,0.,dx)
#     #     (dfdz(Γ_3,i,j,k,0.,dx)) = dfdz(Γ_3,i,j,k,0.,dx)
        
#     #     (dfdx(Σ,i,j,k,0.,dx)) = dfdx(Σ,i,j,k,0.,dx)
#     #     (dfdy(Σ,i,j,k,0.,dx)) = dfdy(Σ,i,j,k,0.,dx)
#     #     (dfdz(Σ,i,j,k,0.,dx)) = dfdz(Σ,i,j,k,0.,dx)

#     #     (d2fdx2(ϕ_1,i,j,k,0.,dx))=d2fdx2(ϕ_1,i,j,k,0.,dx)
#     #     (d2fdx2(ϕ_2,i,j,k,0.,dx))=d2fdx2(ϕ_2,i,j,k,0.,dx)
#     #     (d2fdx2(ϕ_3,i,j,k,0.,dx))=d2fdx2(ϕ_3,i,j,k,0.,dx)
#     #     (d2fdx2(ϕ_4,i,j,k,0.,dx))=d2fdx2(ϕ_4,i,j,k,0.,dx)

#     #     (d2fdy2(ϕ_1,i,j,k,0.,dx))=d2fdy2(ϕ_1,i,j,k,0.,dx)
#     #     (d2fdy2(ϕ_2,i,j,k,0.,dx))=d2fdy2(ϕ_2,i,j,k,0.,dx)
#     #     (d2fdy2(ϕ_3,i,j,k,0.,dx))=d2fdy2(ϕ_3,i,j,k,0.,dx)
#     #     (d2fdy2(ϕ_4,i,j,k,0.,dx))=d2fdy2(ϕ_4,i,j,k,0.,dx)

#     #     (d2fdz2(ϕ_1,i,j,k,0.,dx))=d2fdz2(ϕ_1,i,j,k,0.,dx)
#     #     (d2fdz2(ϕ_2,i,j,k,0.,dx))=d2fdz2(ϕ_2,i,j,k,0.,dx)
#     #     (d2fdz2(ϕ_3,i,j,k,0.,dx))=d2fdz2(ϕ_3,i,j,k,0.,dx)
#     #     (d2fdz2(ϕ_4,i,j,k,0.,dx))=d2fdz2(ϕ_4,i,j,k,0.,dx)

#     #     # d2W_1_1_dx2 = d2fdx2(W_1_1,i,j,k,0.,dx)
#     #     (d2fdx2(W_1_2,i,j,k,0.,dx)) = d2fdx2(W_1_2,i,j,k,0.,dx)
#     #     (d2fdx2(W_1_3,i,j,k,0.,dx)) = d2fdx2(W_1_3,i,j,k,0.,dx)
#     #     (d2fdx2(W_1_4,i,j,k,0.,dx)) = d2fdx2(W_1_4,i,j,k,0.,dx)

#     #     # d2W_1_1_dy2 = d2fdy2(W_1_1,i,j,k,0.,dx)
#     #     (d2fdy2(W_1_2,i,j,k,0.,dx)) = d2fdy2(W_1_2,i,j,k,0.,dx)
#     #     (d2fdy2(W_1_3,i,j,k,0.,dx)) = d2fdy2(W_1_3,i,j,k,0.,dx)
#     #     (d2fdy2(W_1_4,i,j,k,0.,dx)) = d2fdy2(W_1_4,i,j,k,0.,dx)

#     #     # d2W_1_1_dz2 = d2fdz2(W_1_1,i,j,k,0.,dx)
#     #     (d2fdz2(W_1_2,i,j,k,0.,dx)) = d2fdz2(W_1_2,i,j,k,0.,dx)
#     #     (d2fdz2(W_1_3,i,j,k,0.,dx)) = d2fdz2(W_1_3,i,j,k,0.,dx)
#     #     (d2fdz2(W_1_4,i,j,k,0.,dx)) = d2fdz2(W_1_4,i,j,k,0.,dx)

#     #     # d2W_2_1_dx2 = d2fdx2(W_2_1,i,j,k,0.,dx)
#     #     (d2fdx2(W_2_2,i,j,k,0.,dx)) = d2fdx2(W_2_2,i,j,k,0.,dx)
#     #     (d2fdx2(W_2_3,i,j,k,0.,dx)) = d2fdx2(W_2_3,i,j,k,0.,dx)
#     #     (d2fdx2(W_2_4,i,j,k,0.,dx)) = d2fdx2(W_2_4,i,j,k,0.,dx)

#     #     # d2W_2_1_dy2 = d2fdy2(W_2_1,i,j,k,0.,dx)
#     #     (d2fdy2(W_2_2,i,j,k,0.,dx)) = d2fdy2(W_2_2,i,j,k,0.,dx)
#     #     (d2fdy2(W_2_3,i,j,k,0.,dx)) = d2fdy2(W_2_3,i,j,k,0.,dx)
#     #     (d2fdy2(W_2_4,i,j,k,0.,dx)) = d2fdy2(W_2_4,i,j,k,0.,dx)

#     #     # d2W_2_1_dz2 = d2fdz2(W_2_1,i,j,k,0.,dx)
#     #     (d2fdz2(W_2_2,i,j,k,0.,dx)) = d2fdz2(W_2_2,i,j,k,0.,dx)
#     #     (d2fdz2(W_2_3,i,j,k,0.,dx)) = d2fdz2(W_2_3,i,j,k,0.,dx)
#     #     (d2fdz2(W_2_4,i,j,k,0.,dx)) = d2fdz2(W_2_4,i,j,k,0.,dx)

#     #     # d2W_3_1_dx2 = d2fdx2(W_3_1,i,j,k,0.,dx)
#     #     (d2fdx2(W_3_2,i,j,k,0.,dx)) = d2fdx2(W_3_2,i,j,k,0.,dx)
#     #     (d2fdx2(W_3_3,i,j,k,0.,dx)) = d2fdx2(W_3_3,i,j,k,0.,dx)
#     #     (d2fdx2(W_3_4,i,j,k,0.,dx)) = d2fdx2(W_3_4,i,j,k,0.,dx)

#     #     # d2W_3_1_dy2 = d2fdy2(W_3_1,i,j,k,0.,dx)
#     #     (d2fdy2(W_3_2,i,j,k,0.,dx)) = d2fdy2(W_3_2,i,j,k,0.,dx)
#     #     (d2fdy2(W_3_3,i,j,k,0.,dx)) = d2fdy2(W_3_3,i,j,k,0.,dx)
#     #     (d2fdy2(W_3_4,i,j,k,0.,dx)) = d2fdy2(W_3_4,i,j,k,0.,dx)

#     #     # d2W_3_1_dz2 = d2fdz2(W_3_1,i,j,k,0.,dx)
#     #     (d2fdz2(W_3_2,i,j,k,0.,dx)) = d2fdz2(W_3_2,i,j,k,0.,dx)
#     #     (d2fdz2(W_3_3,i,j,k,0.,dx)) = d2fdz2(W_3_3,i,j,k,0.,dx)
#     #     (d2fdz2(W_3_4,i,j,k,0.,dx)) = d2fdz2(W_3_4,i,j,k,0.,dx)

#     #     # d2Y_1_dx2 = d2fdx2(Y_1,i,j,k,0.,dx)
#     #     (d2fdx2(Y_2,i,j,k,0.,dx)) = d2fdx2(Y_2,i,j,k,0.,dx)
#     #     (d2fdx2(Y_3,i,j,k,0.,dx)) = d2fdx2(Y_3,i,j,k,0.,dx)
#     #     (d2fdx2(Y_4,i,j,k,0.,dx)) = d2fdx2(Y_4,i,j,k,0.,dx)

#     #     # d2Y_1_dy2 = d2fdy2(Y_1,i,j,k,0.,dx)
#     #     (d2fdy2(Y_2,i,j,k,0.,dx)) = d2fdy2(Y_2,i,j,k,0.,dx)
#     #     (d2fdy2(Y_3,i,j,k,0.,dx)) = d2fdy2(Y_3,i,j,k,0.,dx)
#     #     (d2fdy2(Y_4,i,j,k,0.,dx)) = d2fdy2(Y_4,i,j,k,0.,dx)

#     #     # d2Y_1_dz2 = d2fdz2(Y_1,i,j,k,0.,dx)
#     #     (d2fdz2(Y_2,i,j,k,0.,dx)) = d2fdz2(Y_2,i,j,k,0.,dx)
#     #     (d2fdz2(Y_3,i,j,k,0.,dx)) = d2fdz2(Y_3,i,j,k,0.,dx)
#     #     (d2fdz2(Y_4,i,j,k,0.,dx)) = d2fdz2(Y_4,i,j,k,0.,dx)

#     # #End spatial derivatives#

#     # # 2-24-24 : Checked all calls to cov derivatives #
#     # ##Covariant Derivatives##Temporal gauge
#     # #In temporal gauge, can drop a alot of terms
    
#     #     (dϕ_1_dt[i,j,k]) =dϕ_1_dt[i,j,k]
#     #     (dϕ_2_dt[i,j,k]) =dϕ_2_dt[i,j,k]
#     #     (dϕ_3_dt[i,j,k]) =dϕ_3_dt[i,j,k]
#     #     (dϕ_4_dt[i,j,k]) =dϕ_4_dt[i,j,k]
#     #     (D_2ϕ_1((dfdx(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
#     #     W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy)) =D_2ϕ_1((dfdx(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
#     #     W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy)
#     #     (D_2ϕ_2((dfdx(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
#     #     W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy)) =D_2ϕ_2((dfdx(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
#     #     W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy)
#     #     (D_2ϕ_3((dfdx(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
#     #     W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy)) =D_2ϕ_3((dfdx(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
#     #     W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy)
#     #     (D_2ϕ_4((dfdx(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
#     #     W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy)) =D_2ϕ_4((dfdx(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
#     #     W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy)
#     #     (D_3ϕ_1((dfdy(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
#     #     W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy)) =D_3ϕ_1((dfdy(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
#     #     W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy)
#     #     (D_3ϕ_2((dfdy(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
#     #     W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy)) =D_3ϕ_2((dfdy(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
#     #     W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy)
#     #     (D_3ϕ_3((dfdy(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
#     #     W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy)) =D_3ϕ_3((dfdy(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
#     #     W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy)
#     #     (D_3ϕ_4((dfdy(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
#     #     W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy)) =D_3ϕ_4((dfdy(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
#     #     W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy)
#     #     (D_4ϕ_1((dfdz(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
#     #     W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)) =D_4ϕ_1((dfdz(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
#     #     W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)
#     #     (D_4ϕ_2((dfdz(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
#     #     W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)) =D_4ϕ_2((dfdz(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
#     #     W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)
#     #     (D_4ϕ_3((dfdz(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
#     #     W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)) =D_4ϕ_3((dfdz(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
#     #     W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)
#     #     (D_4ϕ_4((dfdz(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
#     #     W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)) =D_4ϕ_4((dfdz(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
#     #     W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)

#     # #END covarian derivates#

#     # 2-24-24 : Checked all calls to field strengths #

#     # # Field Strengths # Temporal gauge: can drop a lot of the terms 
#     # #or enter them in expressions explicity#

#     #     # W_1_11()
#     #     # W_1_12(dW_1_2_dt)
#     #     # W_1_13(dW_1_3_dt)
#     #     # W_1_14(dW_1_4_dt)
#     #     # W_1_22(dW_2_2_dt)
#     #     (W_1_xy((dfdx(W_1_3,i,j,k,0.,dx)),(dfdy(W_1_2,i,j,k,0.,dx)),W_2_2,W_3_3,W_2_3,W_3_2,gw,i,j,k)) =W_1_xy((dfdx(W_1_3,i,j,k,0.,dx)),(dfdy(W_1_2,i,j,k,0.,dx)),W_2_2,W_3_3,W_2_3,W_3_2,gw,i,j,k)
#     #     (W_1_xz((dfdx(W_1_4,i,j,k,0.,dx)),(dfdz(W_1_2,i,j,k,0.,dx)),W_2_2,W_3_4,W_2_4,W_3_2,gw,i,j,k))=W_1_xz((dfdx(W_1_4,i,j,k,0.,dx)),(dfdz(W_1_2,i,j,k,0.,dx)),W_2_2,W_3_4,W_2_4,W_3_2,gw,i,j,k)
#     #     # W_1_33(dW_3_3_dt)
#     #     (W_1_yz((dfdy(W_1_4,i,j,k,0.,dx)),(dfdz(W_1_3,i,j,k,0.,dx)),W_2_3,W_3_4,W_2_4,W_3_3,gw,i,j,k))=W_1_yz((dfdy(W_1_4,i,j,k,0.,dx)),(dfdz(W_1_3,i,j,k,0.,dx)),W_2_3,W_3_4,W_2_4,W_3_3,gw,i,j,k)
#     #     # W_1_44()
#     #     # W_2_11()
#     #     # W_2_12(dW_2_2_dt)
#     #     # W_2_13(dW_2_3_dt)
#     #     # W_2_14(dW_2_4_dt)
#     #     # W_2_22()
#     #     (W_2_xy((dfdx(W_2_3,i,j,k,0.,dx)),(dfdy(W_2_2,i,j,k,0.,dx)),W_3_2,W_1_3,W_3_3,W_1_2,gw,i,j,k))=W_2_xy((dfdx(W_2_3,i,j,k,0.,dx)),(dfdy(W_2_2,i,j,k,0.,dx)),W_3_2,W_1_3,W_3_3,W_1_2,gw,i,j,k)
#     #     (W_2_xz((dfdx(W_2_4,i,j,k,0.,dx)),(dfdz(W_2_2,i,j,k,0.,dx)),W_3_2,W_1_4,W_3_4,W_1_2,gw,i,j,k))=W_2_xz((dfdx(W_2_4,i,j,k,0.,dx)),(dfdz(W_2_2,i,j,k,0.,dx)),W_3_2,W_1_4,W_3_4,W_1_2,gw,i,j,k)
#     #     # W_2_33()
#     #     (W_2_yz((dfdy(W_2_4,i,j,k,0.,dx)),(dfdz(W_2_3,i,j,k,0.,dx)),W_3_3,W_1_4,W_3_4,W_1_3,gw,i,j,k))=W_2_yz((dfdy(W_2_4,i,j,k,0.,dx)),(dfdz(W_2_3,i,j,k,0.,dx)),W_3_3,W_1_4,W_3_4,W_1_3,gw,i,j,k)
#     #     # W_2_44()
#     #     # W_3_11()
#     #     # W_3_12(dW_3_2_dt)
#     #     # W_3_13(dW_3_3_dt)
#     #     # W_3_14(dW_3_4_dt)
#     #     # W_3_22()
#     #     (W_3_xy((dfdx(W_3_3,i,j,k,0.,dx)),dfdy(W_3_2,i,j,k,0.,dx),W_1_2,W_2_3,W_1_3,W_2_2,gw,i,j,k))=W_3_xy((dfdx(W_3_3,i,j,k,0.,dx)),dfdy(W_3_2,i,j,k,0.,dx),W_1_2,W_2_3,W_1_3,W_2_2,gw,i,j,k)
#     #     (W_3_xz((dfdx(W_3_4,i,j,k,0.,dx)),(dfdz(W_3_2,i,j,k,0.,dx)),W_1_2,W_2_4,W_1_4,W_2_2,gw,i,j,k))=W_3_xz((dfdx(W_3_4,i,j,k,0.,dx)),(dfdz(W_3_2,i,j,k,0.,dx)),W_1_2,W_2_4,W_1_4,W_2_2,gw,i,j,k)
#     #     # W_3_33(dW_3_3_dt)
#     #     (W_3_yz(dfdy(W_3_4,i,j,k,0.,dx),(dfdz(W_3_3,i,j,k,0.,dx)),W_1_3,W_2_4,W_1_4,W_2_3,gw,i,j,k))=W_3_yz(dfdy(W_3_4,i,j,k,0.,dx),(dfdz(W_3_3,i,j,k,0.,dx)),W_1_3,W_2_4,W_1_4,W_2_3,gw,i,j,k)
#     #     # W_3_44()
#     #     # Y_1_1()
#     #     # Y_1_2(dY_2_dt)
#     #     # Y_1_3(dY_3_dt)
#     #     # Y_1_4(dY_4_dt)
#     #     # Y_2_2()
#     #     (Y_2_y((dfdx(Y_3,i,j,k,0.,dx)),(dfdy(Y_2,i,j,k,0.,dx))))=Y_2_y((dfdx(Y_3,i,j,k,0.,dx)),(dfdy(Y_2,i,j,k,0.,dx)))
#     #     (Y_2_z((dfdx(Y_4,i,j,k,0.,dx)),(dfdz(Y_2,i,j,k,0.,dx))))=Y_2_z((dfdx(Y_4,i,j,k,0.,dx)),(dfdz(Y_2,i,j,k,0.,dx)))
#     #     # Y_3_3()
#     #     (Y_3_z((dfdy(Y_4,i,j,k,0.,dx)),(dfdz(Y_3,i,j,k,0.,dx))))=Y_3_z((dfdy(Y_4,i,j,k,0.,dx)),(dfdz(Y_3,i,j,k,0.,dx)))
#     #     # Y_4_4()
#     # #END field strengths#

#     # 2-24-24: Checked all flux expressions#

#     # kt_1 expressions
#     @inbounds kt_ϕ_1[i,j,k] = ((d2fdx2(ϕ_1,i,j,k,0.,dx))+(d2fdy2(ϕ_1,i,j,k,0.,dx))+(d2fdz2(ϕ_1,i,j,k,0.,dx))-
#         0.5*gw*(((-W_1_2[i,j,k]*(dfdx(ϕ_4,i,j,k,0.,dx)))-(W_1_3[i,j,k]*(dfdy(ϕ_4,i,j,k,0.,dx)))-(W_1_4[i,j,k]*(dfdz(ϕ_4,i,j,k,0.,dx))))-
#         ((-W_2_2[i,j,k]*(dfdx(ϕ_3,i,j,k,0.,dx)))-(W_2_3[i,j,k]*(dfdy(ϕ_3,i,j,k,0.,dx)))-(W_2_4[i,j,k]*(dfdz(ϕ_3,i,j,k,0.,dx))))+
#         ((-W_3_2[i,j,k]*(dfdx(ϕ_2,i,j,k,0.,dx)))-(W_3_3[i,j,k]*(dfdy(ϕ_2,i,j,k,0.,dx)))-(W_3_4[i,j,k]*(dfdz(ϕ_2,i,j,k,0.,dx)))))-
#         0.5*gy*(-Y_2[i,j,k]*(dfdx(ϕ_2,i,j,k,0.,dx))-Y_3[i,j,k]*(dfdy(ϕ_2,i,j,k,0.,dx))-Y_4[i,j,k]*(dfdz(ϕ_2,i,j,k,0.,dx)))-
#         0.5*gw*((-W_1_2[i,j,k]*(D_2ϕ_4((dfdx(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))-
#         W_1_3[i,j,k]*(D_3ϕ_4((dfdy(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))-
#         W_1_4[i,j,k]*(D_4ϕ_4((dfdz(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)))-
#         (-W_2_2[i,j,k]*(D_2ϕ_3((dfdx(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))-
#         W_2_3[i,j,k]*(D_3ϕ_3((dfdy(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))-
#         W_2_4[i,j,k]*(D_4ϕ_3((dfdz(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)))+
#         (-W_3_2[i,j,k]*(D_2ϕ_2((dfdx(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))-
#         W_3_3[i,j,k]*(D_3ϕ_2((dfdy(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))-
#         W_3_4[i,j,k]*(D_4ϕ_2((dfdz(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy))))-
#         0.5*gy*(-Y_2[i,j,k]*(D_2ϕ_2((dfdx(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))-
#         Y_3[i,j,k]*(D_3ϕ_2((dfdy(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))-
#         Y_4[i,j,k]*(D_4ϕ_2((dfdz(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)))-
#         2.0*lambda*(ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2-vev^2)*ϕ_1[i,j,k]+
#         0.5*((gw*Γ_3[i,j,k]+gy*Σ[i,j,k])*ϕ_2[i,j,k]-gw*Γ_2[i,j,k]*ϕ_3[i,j,k]+gw*Γ_1[i,j,k]*ϕ_4[i,j,k])-
#         0.5*Jex*ϕ_1[i,j,k]-γ*dϕ_1_dt[i,j,k]^2/(ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2))

    
#     @inbounds kt_ϕ_2[i,j,k] = ((d2fdx2(ϕ_2,i,j,k,0.,dx))+(d2fdy2(ϕ_2,i,j,k,0.,dx))+(d2fdz2(ϕ_2,i,j,k,0.,dx))+
#         0.5*gw*((-W_1_2[i,j,k]*(dfdx(ϕ_3,i,j,k,0.,dx))-W_1_3[i,j,k]*(dfdy(ϕ_3,i,j,k,0.,dx))-W_1_4[i,j,k]*(dfdz(ϕ_3,i,j,k,0.,dx)))+
#         (-W_2_2[i,j,k]*(dfdx(ϕ_4,i,j,k,0.,dx))-W_2_3[i,j,k]*(dfdy(ϕ_4,i,j,k,0.,dx))-W_2_4[i,j,k]*(dfdz(ϕ_4,i,j,k,0.,dx)))+
#         (-W_3_2[i,j,k]*(dfdx(ϕ_1,i,j,k,0.,dx))-W_3_3[i,j,k]*(dfdy(ϕ_1,i,j,k,0.,dx))-W_3_4[i,j,k]*(dfdz(ϕ_1,i,j,k,0.,dx))))+
#         0.5*gy*(-Y_2[i,j,k]*(dfdx(ϕ_1,i,j,k,0.,dx))-Y_3[i,j,k]*(dfdy(ϕ_1,i,j,k,0.,dx))-Y_4[i,j,k]*(dfdz(ϕ_1,i,j,k,0.,dx)))+
#         0.5*gw*((-W_1_2[i,j,k]*(D_2ϕ_3((dfdx(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))-W_1_3[i,j,k]*(D_3ϕ_3((dfdy(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))-W_1_4[i,j,k]*(D_4ϕ_3((dfdz(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)))+
#         (-W_2_2[i,j,k]*(D_2ϕ_4((dfdx(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))-W_2_3[i,j,k]*(D_3ϕ_4((dfdy(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))-W_2_4[i,j,k]*(D_4ϕ_4((dfdz(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)))+
#         (-W_3_2[i,j,k]*(D_2ϕ_1((dfdx(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))-W_3_3[i,j,k]*(D_3ϕ_1((dfdy(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))-W_3_4[i,j,k]*(D_4ϕ_1((dfdz(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy))))+
#         0.5*gy*(-Y_2[i,j,k]*(D_2ϕ_1((dfdx(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))-Y_3[i,j,k]*(D_3ϕ_1((dfdy(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))-Y_4[i,j,k]*(D_4ϕ_1((dfdz(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)))-
#         2.0*lambda*(ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2-vev^2)*ϕ_2[i,j,k]-
#         0.5*((gw*Γ_3[i,j,k]+gy*Σ[i,j,k])*ϕ_1[i,j,k]+gw*Γ_1[i,j,k]*ϕ_3[i,j,k]+gw*Γ_2[i,j,k]*ϕ_4[i,j,k])-
#         0.5*Jex*ϕ_2[i,j,k]-γ*dϕ_2_dt[i,j,k]^2/(ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2))

#     @inbounds kt_ϕ_3[i,j,k] = ((d2fdx2(ϕ_3,i,j,k,0.,dx))+(d2fdy2(ϕ_3,i,j,k,0.,dx))+(d2fdz2(ϕ_3,i,j,k,0.,dx))-
#         0.5*gw*((-W_1_2[i,j,k]*(dfdx(ϕ_2,i,j,k,0.,dx))-W_1_3[i,j,k]*(dfdy(ϕ_2,i,j,k,0.,dx))-W_1_4[i,j,k]*(dfdz(ϕ_2,i,j,k,0.,dx)))+
#         (-W_2_2[i,j,k]*(dfdx(ϕ_1,i,j,k,0.,dx))-W_2_3[i,j,k]*(dfdy(ϕ_1,i,j,k,0.,dx))-W_2_4[i,j,k]*(dfdz(ϕ_1,i,j,k,0.,dx)))-
#         (-W_3_2[i,j,k]*(dfdx(ϕ_4,i,j,k,0.,dx))-W_3_3[i,j,k]*(dfdy(ϕ_4,i,j,k,0.,dx))-W_3_4[i,j,k]*(dfdz(ϕ_4,i,j,k,0.,dx))))-
#         0.5*gy*(-Y_2[i,j,k]*(dfdx(ϕ_4,i,j,k,0.,dx))-Y_3[i,j,k]*(dfdy(ϕ_4,i,j,k,0.,dx))-Y_4[i,j,k]*(dfdz(ϕ_4,i,j,k,0.,dx)))-
#         0.5*gw*((-W_1_2[i,j,k]*(D_2ϕ_2((dfdx(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))-
#         W_1_3[i,j,k]*(D_3ϕ_2((dfdy(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))-
#         W_1_4[i,j,k]*(D_4ϕ_2((dfdz(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)))+
#         (-W_2_2[i,j,k]*(D_2ϕ_1((dfdx(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))-
#         W_2_3[i,j,k]*(D_3ϕ_1((dfdy(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))-
#         W_2_4[i,j,k]*(D_4ϕ_1((dfdz(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)))-
#         (-W_3_2[i,j,k]*(D_2ϕ_4((dfdx(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))-
#         W_3_3[i,j,k]*(D_3ϕ_4((dfdy(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))-
#         W_3_4[i,j,k]*(D_4ϕ_4((dfdz(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy))))-
#         0.5*gy*(-Y_2[i,j,k]*(D_2ϕ_4((dfdx(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))-
#         Y_3[i,j,k]*(D_3ϕ_4((dfdy(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))-
#         Y_4[i,j,k]*(D_4ϕ_4((dfdz(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)))-
#         2.0*lambda*(ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2-vev^2)*ϕ_3[i,j,k]+
#         0.5*((-gw*Γ_3[i,j,k]+gy*Σ[i,j,k])*ϕ_4[i,j,k]+gw*Γ_2[i,j,k]*ϕ_1[i,j,k]+gw*Γ_1[i,j,k]*ϕ_2[i,j,k])-
#         0.5*Jex*ϕ_3[i,j,k]-γ*dϕ_3_dt[i,j,k]^2/(ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2))

#     @inbounds kt_ϕ_4[i,j,k] = ((d2fdx2(ϕ_4,i,j,k,0.,dx))+(d2fdy2(ϕ_4,i,j,k,0.,dx))+(d2fdz2(ϕ_4,i,j,k,0.,dx))+
#         0.5*gw*((-W_1_2[i,j,k]*(dfdx(ϕ_1,i,j,k,0.,dx))-W_1_3[i,j,k]*(dfdy(ϕ_1,i,j,k,0.,dx))-W_1_4[i,j,k]*(dfdz(ϕ_1,i,j,k,0.,dx)))-
#         (-W_2_2[i,j,k]*(dfdx(ϕ_2,i,j,k,0.,dx))-W_2_3[i,j,k]*(dfdy(ϕ_2,i,j,k,0.,dx))-W_2_4[i,j,k]*(dfdz(ϕ_2,i,j,k,0.,dx)))-
#         (-W_3_2[i,j,k]*(dfdx(ϕ_3,i,j,k,0.,dx))-W_3_3[i,j,k]*(dfdy(ϕ_3,i,j,k,0.,dx))-W_3_4[i,j,k]*(dfdz(ϕ_3,i,j,k,0.,dx))))+
#         0.5*gy*(-Y_2[i,j,k]*(dfdx(ϕ_3,i,j,k,0.,dx))-Y_3[i,j,k]*(dfdy(ϕ_3,i,j,k,0.,dx))-Y_4[i,j,k]*(dfdz(ϕ_3,i,j,k,0.,dx)))+
#         0.5*gw*((-W_1_2[i,j,k]*(D_2ϕ_1((dfdx(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))-W_1_3[i,j,k]*(D_3ϕ_1((dfdy(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))-W_1_4[i,j,k]*(D_4ϕ_1((dfdz(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)))-
#         (-W_2_2[i,j,k]*(D_2ϕ_2((dfdx(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))-W_2_3[i,j,k]*(D_3ϕ_2((dfdy(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))-W_2_4[i,j,k]*(D_4ϕ_2((dfdz(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)))-
#         (-W_3_2[i,j,k]*(D_2ϕ_3((dfdx(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))-W_3_3[i,j,k]*(D_3ϕ_3((dfdy(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))-W_3_4[i,j,k]*(D_4ϕ_3((dfdz(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy))))+
#         0.5*gy*(-Y_2[i,j,k]*(D_2ϕ_3((dfdx(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))-Y_3[i,j,k]*(D_3ϕ_3((dfdy(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))-Y_4[i,j,k]*(D_4ϕ_3((dfdz(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)))-
#         2.0*lambda*(ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2-vev^2)*ϕ_4[i,j,k]-
#         0.5*((-gw*Γ_3[i,j,k]+gy*Σ[i,j,k])*ϕ_3[i,j,k]+gw*Γ_1[i,j,k]*ϕ_1[i,j,k]-gw*Γ_2[i,j,k]*ϕ_2[i,j,k])-
#         0.5*Jex*ϕ_4[i,j,k]-γ*dϕ_4_dt[i,j,k]^2/(ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2))

#     @inbounds kt_W_1_2[i,j,k] = ((d2fdx2(W_1_2,i,j,k,0.,dx))+(d2fdy2(W_1_2,i,j,k,0.,dx))+(d2fdz2(W_1_2,i,j,k,0.,dx))+
#         gw*(-((dfdx(W_2_2,i,j,k,0.,dx))*W_3_2[i,j,k]-(dfdx(W_3_2,i,j,k,0.,dx))*W_2_2[i,j,k])-
#         ((dfdy(W_2_2,i,j,k,0.,dx))*W_3_3[i,j,k]-dfdy(W_3_2,i,j,k,0.,dx)*W_2_3[i,j,k])-
#         ((dfdz(W_2_2,i,j,k,0.,dx))*W_3_4[i,j,k]-(dfdz(W_3_2,i,j,k,0.,dx))*W_2_4[i,j,k])-
#         (W_2_3[i,j,k]*(W_3_xy((dfdx(W_3_3,i,j,k,0.,dx)),dfdy(W_3_2,i,j,k,0.,dx),W_1_2,W_2_3,W_1_3,W_2_2,gw,i,j,k))-W_3_3[i,j,k]*(W_2_xy((dfdx(W_2_3,i,j,k,0.,dx)),(dfdy(W_2_2,i,j,k,0.,dx)),W_3_2,W_1_3,W_3_3,W_1_2,gw,i,j,k)))-
#         (W_2_4[i,j,k]*(W_3_xz((dfdx(W_3_4,i,j,k,0.,dx)),(dfdz(W_3_2,i,j,k,0.,dx)),W_1_2,W_2_4,W_1_4,W_2_2,gw,i,j,k))-W_3_4[i,j,k]*(W_2_xz((dfdx(W_2_4,i,j,k,0.,dx)),(dfdz(W_2_2,i,j,k,0.,dx)),W_3_2,W_1_4,W_3_4,W_1_2,gw,i,j,k))))+
#         gw*(ϕ_1[i,j,k]*(D_2ϕ_4((dfdx(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))-ϕ_2[i,j,k]*(D_2ϕ_3((dfdx(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))+ϕ_3[i,j,k]*(D_2ϕ_2((dfdx(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))-ϕ_4[i,j,k]*(D_2ϕ_1((dfdx(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy)))-
#         (dfdx(Γ_1,i,j,k,0.,dx))-gw*(W_2_2[i,j,k]*Γ_3[i,j,k]-W_3_2[i,j,k]*Γ_2[i,j,k])-γ*dW_1_2_dt[i,j,k])

#     @inbounds kt_W_1_3[i,j,k] = ((d2fdx2(W_1_3,i,j,k,0.,dx))+(d2fdy2(W_1_3,i,j,k,0.,dx))+(d2fdz2(W_1_3,i,j,k,0.,dx))+
#         gw*(-((dfdx(W_2_3,i,j,k,0.,dx))*W_3_2[i,j,k]-(dfdx(W_3_3,i,j,k,0.,dx))*W_2_2[i,j,k])-
#         ((dfdy(W_2_3,i,j,k,0.,dx))*W_3_3[i,j,k]-dfdy(W_3_3,i,j,k,0.,dx)*W_2_3[i,j,k])-
#         ((dfdz(W_2_3,i,j,k,0.,dx))*W_3_4[i,j,k]-(dfdz(W_3_3,i,j,k,0.,dx))*W_2_4[i,j,k])-
#         (W_2_2[i,j,k]*(-(W_3_xy((dfdx(W_3_3,i,j,k,0.,dx)),dfdy(W_3_2,i,j,k,0.,dx),W_1_2,W_2_3,W_1_3,W_2_2,gw,i,j,k)))-W_3_2[i,j,k]*(-(W_2_xy((dfdx(W_2_3,i,j,k,0.,dx)),(dfdy(W_2_2,i,j,k,0.,dx)),W_3_2,W_1_3,W_3_3,W_1_2,gw,i,j,k))))-
#         (W_2_4[i,j,k]*(W_3_yz(dfdy(W_3_4,i,j,k,0.,dx),(dfdz(W_3_3,i,j,k,0.,dx)),W_1_3,W_2_4,W_1_4,W_2_3,gw,i,j,k))-W_3_4[i,j,k]*(W_2_yz((dfdy(W_2_4,i,j,k,0.,dx)),(dfdz(W_2_3,i,j,k,0.,dx)),W_3_3,W_1_4,W_3_4,W_1_3,gw,i,j,k))))+
#         gw*(ϕ_1[i,j,k]*(D_3ϕ_4((dfdy(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))-ϕ_2[i,j,k]*(D_3ϕ_3((dfdy(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))+ϕ_3[i,j,k]*(D_3ϕ_2((dfdy(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))-ϕ_4[i,j,k]*(D_3ϕ_1((dfdy(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy)))-
#         (dfdy(Γ_1,i,j,k,0.,dx))-gw*(W_2_3[i,j,k]*Γ_3[i,j,k]-W_3_3[i,j,k]*Γ_2[i,j,k])-γ*dW_1_3_dt[i,j,k])

#     @inbounds kt_W_1_4[i,j,k] = ((d2fdx2(W_1_4,i,j,k,0.,dx))+(d2fdy2(W_1_4,i,j,k,0.,dx))+(d2fdz2(W_1_4,i,j,k,0.,dx))+
#         gw*(-((dfdx(W_2_4,i,j,k,0.,dx))*W_3_2[i,j,k]-(dfdx(W_3_4,i,j,k,0.,dx))*W_2_2[i,j,k])-
#         ((dfdy(W_2_4,i,j,k,0.,dx))*W_3_3[i,j,k]-dfdy(W_3_4,i,j,k,0.,dx)*W_2_3[i,j,k])-
#         ((dfdz(W_2_4,i,j,k,0.,dx))*W_3_4[i,j,k]-(dfdz(W_3_4,i,j,k,0.,dx))*W_2_4[i,j,k])-
#         (W_2_2[i,j,k]*(-(W_3_xz((dfdx(W_3_4,i,j,k,0.,dx)),(dfdz(W_3_2,i,j,k,0.,dx)),W_1_2,W_2_4,W_1_4,W_2_2,gw,i,j,k)))-W_3_2[i,j,k]*(-(W_2_xz((dfdx(W_2_4,i,j,k,0.,dx)),(dfdz(W_2_2,i,j,k,0.,dx)),W_3_2,W_1_4,W_3_4,W_1_2,gw,i,j,k))))-
#         (W_2_3[i,j,k]*(-(W_3_yz(dfdy(W_3_4,i,j,k,0.,dx),(dfdz(W_3_3,i,j,k,0.,dx)),W_1_3,W_2_4,W_1_4,W_2_3,gw,i,j,k)))-W_3_3[i,j,k]*(-(W_2_yz((dfdy(W_2_4,i,j,k,0.,dx)),(dfdz(W_2_3,i,j,k,0.,dx)),W_3_3,W_1_4,W_3_4,W_1_3,gw,i,j,k)))))+
#         gw*(ϕ_1[i,j,k]*(D_4ϕ_4((dfdz(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy))-ϕ_2[i,j,k]*(D_4ϕ_3((dfdz(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy))+ϕ_3[i,j,k]*(D_4ϕ_2((dfdz(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy))-ϕ_4[i,j,k]*(D_4ϕ_1((dfdz(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)))-
#         (dfdz(Γ_1,i,j,k,0.,dx))-gw*(W_2_4[i,j,k]*Γ_3[i,j,k]-W_3_4[i,j,k]*Γ_2[i,j,k])-γ*dW_1_4_dt[i,j,k])

#     @inbounds kt_W_2_2[i,j,k] = ((d2fdx2(W_2_2,i,j,k,0.,dx))+(d2fdy2(W_2_2,i,j,k,0.,dx))+(d2fdz2(W_2_2,i,j,k,0.,dx))+
#         gw*(-((dfdx(W_3_2,i,j,k,0.,dx))*W_1_2[i,j,k]-(dfdx(W_1_2,i,j,k,0.,dx))*W_3_2[i,j,k])-
#         (dfdy(W_3_2,i,j,k,0.,dx)*W_1_3[i,j,k]-(dfdy(W_1_2,i,j,k,0.,dx))*W_3_3[i,j,k])-
#         ((dfdz(W_3_2,i,j,k,0.,dx))*W_1_4[i,j,k]-(dfdz(W_1_2,i,j,k,0.,dx))*W_3_4[i,j,k])-
#         (W_3_3[i,j,k]*(W_1_xy((dfdx(W_1_3,i,j,k,0.,dx)),(dfdy(W_1_2,i,j,k,0.,dx)),W_2_2,W_3_3,W_2_3,W_3_2,gw,i,j,k))-W_1_3[i,j,k]*(W_3_xy((dfdx(W_3_3,i,j,k,0.,dx)),dfdy(W_3_2,i,j,k,0.,dx),W_1_2,W_2_3,W_1_3,W_2_2,gw,i,j,k)))-
#         (W_3_4[i,j,k]*(W_1_xz((dfdx(W_1_4,i,j,k,0.,dx)),(dfdz(W_1_2,i,j,k,0.,dx)),W_2_2,W_3_4,W_2_4,W_3_2,gw,i,j,k))-W_1_4[i,j,k]*(W_3_xz((dfdx(W_3_4,i,j,k,0.,dx)),(dfdz(W_3_2,i,j,k,0.,dx)),W_1_2,W_2_4,W_1_4,W_2_2,gw,i,j,k))))+
#         gw*(-ϕ_1[i,j,k]*(D_2ϕ_3((dfdx(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))-ϕ_2[i,j,k]*(D_2ϕ_4((dfdx(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))+ϕ_3[i,j,k]*(D_2ϕ_1((dfdx(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))+ϕ_4[i,j,k]*(D_2ϕ_2((dfdx(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy)))-
#         (dfdx(Γ_2,i,j,k,0.,dx))-gw*(W_3_2[i,j,k]*Γ_1[i,j,k]-W_1_2[i,j,k]*Γ_3[i,j,k])-γ*dW_2_2_dt[i,j,k])

#     @inbounds kt_W_2_3[i,j,k] = ((d2fdx2(W_2_3,i,j,k,0.,dx))+(d2fdy2(W_2_3,i,j,k,0.,dx))+(d2fdz2(W_2_3,i,j,k,0.,dx))+
#         gw*(-((dfdx(W_3_3,i,j,k,0.,dx))*W_1_2[i,j,k]-(dfdx(W_1_3,i,j,k,0.,dx))*W_3_2[i,j,k])-
#         (dfdy(W_3_3,i,j,k,0.,dx)*W_1_3[i,j,k]-(dfdy(W_1_3,i,j,k,0.,dx))*W_3_3[i,j,k])-
#         ((dfdz(W_3_3,i,j,k,0.,dx))*W_1_4[i,j,k]-(dfdz(W_1_3,i,j,k,0.,dx))*W_3_4[i,j,k])-
#         (W_3_2[i,j,k]*(-(W_1_xy((dfdx(W_1_3,i,j,k,0.,dx)),(dfdy(W_1_2,i,j,k,0.,dx)),W_2_2,W_3_3,W_2_3,W_3_2,gw,i,j,k)))-W_1_2[i,j,k]*(-(W_3_xy((dfdx(W_3_3,i,j,k,0.,dx)),dfdy(W_3_2,i,j,k,0.,dx),W_1_2,W_2_3,W_1_3,W_2_2,gw,i,j,k))))-
#         (W_3_4[i,j,k]*((W_1_yz((dfdy(W_1_4,i,j,k,0.,dx)),(dfdz(W_1_3,i,j,k,0.,dx)),W_2_3,W_3_4,W_2_4,W_3_3,gw,i,j,k)))-W_1_4[i,j,k]*(W_3_yz(dfdy(W_3_4,i,j,k,0.,dx),(dfdz(W_3_3,i,j,k,0.,dx)),W_1_3,W_2_4,W_1_4,W_2_3,gw,i,j,k))))+
#         gw*(-ϕ_1[i,j,k]*(D_3ϕ_3((dfdy(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))-ϕ_2[i,j,k]*(D_3ϕ_4((dfdy(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))+ϕ_3[i,j,k]*(D_3ϕ_1((dfdy(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))+ϕ_4[i,j,k]*(D_3ϕ_2((dfdy(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy)))-
#         (dfdy(Γ_2,i,j,k,0.,dx))-gw*(W_3_3[i,j,k]*Γ_1[i,j,k]-W_1_3[i,j,k]*Γ_3[i,j,k])-γ*dW_2_3_dt[i,j,k])

#     @inbounds kt_W_2_4[i,j,k] = ((d2fdx2(W_2_4,i,j,k,0.,dx))+(d2fdy2(W_2_4,i,j,k,0.,dx))+(d2fdz2(W_2_4,i,j,k,0.,dx))+
#         gw*(-((dfdx(W_3_4,i,j,k,0.,dx))*W_1_2[i,j,k]-(dfdx(W_1_4,i,j,k,0.,dx))*W_3_2[i,j,k])-
#         (dfdy(W_3_4,i,j,k,0.,dx)*W_1_3[i,j,k]-(dfdy(W_1_4,i,j,k,0.,dx))*W_3_3[i,j,k])-
#         ((dfdz(W_3_4,i,j,k,0.,dx))*W_1_4[i,j,k]-(dfdz(W_1_4,i,j,k,0.,dx))*W_3_4[i,j,k])-
#         (W_3_2[i,j,k]*(-(W_1_xz((dfdx(W_1_4,i,j,k,0.,dx)),(dfdz(W_1_2,i,j,k,0.,dx)),W_2_2,W_3_4,W_2_4,W_3_2,gw,i,j,k)))-W_1_2[i,j,k]*(-(W_3_xz((dfdx(W_3_4,i,j,k,0.,dx)),(dfdz(W_3_2,i,j,k,0.,dx)),W_1_2,W_2_4,W_1_4,W_2_2,gw,i,j,k))))-
#         (W_3_3[i,j,k]*(-(W_1_yz((dfdy(W_1_4,i,j,k,0.,dx)),(dfdz(W_1_3,i,j,k,0.,dx)),W_2_3,W_3_4,W_2_4,W_3_3,gw,i,j,k)))-W_1_3[i,j,k]*(-(W_3_yz(dfdy(W_3_4,i,j,k,0.,dx),(dfdz(W_3_3,i,j,k,0.,dx)),W_1_3,W_2_4,W_1_4,W_2_3,gw,i,j,k)))))+
#         gw*(-ϕ_1[i,j,k]*(D_4ϕ_3((dfdz(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy))-ϕ_2[i,j,k]*(D_4ϕ_4((dfdz(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy))+ϕ_3[i,j,k]*(D_4ϕ_1((dfdz(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy))+ϕ_4[i,j,k]*(D_4ϕ_2((dfdz(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)))-
#         (dfdz(Γ_2,i,j,k,0.,dx))-gw*(W_3_4[i,j,k]*Γ_1[i,j,k]-W_1_4[i,j,k]*Γ_3[i,j,k])-γ*dW_2_4_dt[i,j,k])

#     @inbounds kt_W_3_2[i,j,k] = ((d2fdx2(W_3_2,i,j,k,0.,dx))+(d2fdy2(W_3_2,i,j,k,0.,dx))+(d2fdz2(W_3_2,i,j,k,0.,dx))+
#         gw*(-((dfdx(W_1_2,i,j,k,0.,dx))*W_2_2[i,j,k]-(dfdx(W_2_2,i,j,k,0.,dx))*W_1_2[i,j,k])-
#         ((dfdy(W_1_2,i,j,k,0.,dx))*W_2_3[i,j,k]-(dfdy(W_2_2,i,j,k,0.,dx))*W_1_3[i,j,k])-
#         ((dfdz(W_1_2,i,j,k,0.,dx))*W_2_4[i,j,k]-(dfdz(W_2_2,i,j,k,0.,dx))*W_1_4[i,j,k])-
#         (W_1_3[i,j,k]*(W_2_xy((dfdx(W_2_3,i,j,k,0.,dx)),(dfdy(W_2_2,i,j,k,0.,dx)),W_3_2,W_1_3,W_3_3,W_1_2,gw,i,j,k))-W_2_3[i,j,k]*(W_1_xy((dfdx(W_1_3,i,j,k,0.,dx)),(dfdy(W_1_2,i,j,k,0.,dx)),W_2_2,W_3_3,W_2_3,W_3_2,gw,i,j,k)))-
#         (W_1_4[i,j,k]*(W_2_xz((dfdx(W_2_4,i,j,k,0.,dx)),(dfdz(W_2_2,i,j,k,0.,dx)),W_3_2,W_1_4,W_3_4,W_1_2,gw,i,j,k))-W_2_4[i,j,k]*(W_1_xz((dfdx(W_1_4,i,j,k,0.,dx)),(dfdz(W_1_2,i,j,k,0.,dx)),W_2_2,W_3_4,W_2_4,W_3_2,gw,i,j,k))))+
#         gw*(ϕ_1[i,j,k]*(D_2ϕ_2((dfdx(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))-ϕ_2[i,j,k]*(D_2ϕ_1((dfdx(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))-ϕ_3[i,j,k]*(D_2ϕ_4((dfdx(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))+ϕ_4[i,j,k]*(D_2ϕ_3((dfdx(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy)))-
#         (dfdx(Γ_3,i,j,k,0.,dx))-gw*(W_1_2[i,j,k]*Γ_2[i,j,k]-W_2_2[i,j,k]*Γ_1[i,j,k])-γ*dW_3_2_dt[i,j,k])

#     @inbounds kt_W_3_3[i,j,k] = ((d2fdx2(W_3_3,i,j,k,0.,dx))+(d2fdy2(W_3_3,i,j,k,0.,dx))+(d2fdz2(W_3_3,i,j,k,0.,dx))+
#         gw*(-((dfdx(W_1_3,i,j,k,0.,dx))*W_2_2[i,j,k]-(dfdx(W_2_3,i,j,k,0.,dx))*W_1_2[i,j,k])-
#             ((dfdy(W_1_3,i,j,k,0.,dx))*W_2_3[i,j,k]-(dfdy(W_2_3,i,j,k,0.,dx))*W_1_3[i,j,k])-
#             ((dfdz(W_1_3,i,j,k,0.,dx))*W_2_4[i,j,k]-(dfdz(W_2_3,i,j,k,0.,dx))*W_1_4[i,j,k])-
#             (W_1_2[i,j,k]*(-(W_2_xy((dfdx(W_2_3,i,j,k,0.,dx)),(dfdy(W_2_2,i,j,k,0.,dx)),W_3_2,W_1_3,W_3_3,W_1_2,gw,i,j,k)))-W_2_2[i,j,k]*(-(W_1_xy((dfdx(W_1_3,i,j,k,0.,dx)),(dfdy(W_1_2,i,j,k,0.,dx)),W_2_2,W_3_3,W_2_3,W_3_2,gw,i,j,k))))-
#             (W_1_4[i,j,k]*(W_2_yz((dfdy(W_2_4,i,j,k,0.,dx)),(dfdz(W_2_3,i,j,k,0.,dx)),W_3_3,W_1_4,W_3_4,W_1_3,gw,i,j,k))-W_2_4[i,j,k]*((W_1_yz((dfdy(W_1_4,i,j,k,0.,dx)),(dfdz(W_1_3,i,j,k,0.,dx)),W_2_3,W_3_4,W_2_4,W_3_3,gw,i,j,k)))))+
#         gw*(ϕ_1[i,j,k]*(D_3ϕ_2((dfdy(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))-ϕ_2[i,j,k]*(D_3ϕ_1((dfdy(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))-ϕ_3[i,j,k]*(D_3ϕ_4((dfdy(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))+ϕ_4[i,j,k]*(D_3ϕ_3((dfdy(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy)))-
#         (dfdy(Γ_3,i,j,k,0.,dx))-gw*(W_1_3[i,j,k]*Γ_2[i,j,k]-W_2_3[i,j,k]*Γ_1[i,j,k])-γ*dW_3_3_dt[i,j,k])

#     @inbounds kt_W_3_4[i,j,k] = ((d2fdx2(W_3_4,i,j,k,0.,dx))+(d2fdy2(W_3_4,i,j,k,0.,dx))+(d2fdz2(W_3_4,i,j,k,0.,dx))+
#         gw*(-((dfdx(W_1_4,i,j,k,0.,dx))*W_2_2[i,j,k]-(dfdx(W_2_4,i,j,k,0.,dx))*W_1_2[i,j,k])-
#             ((dfdy(W_1_4,i,j,k,0.,dx))*W_2_3[i,j,k]-(dfdy(W_2_4,i,j,k,0.,dx))*W_1_3[i,j,k])-
#             ((dfdz(W_1_4,i,j,k,0.,dx))*W_2_4[i,j,k]-(dfdz(W_2_4,i,j,k,0.,dx))*W_1_4[i,j,k])-
#             (W_1_2[i,j,k]*(-(W_2_xz((dfdx(W_2_4,i,j,k,0.,dx)),(dfdz(W_2_2,i,j,k,0.,dx)),W_3_2,W_1_4,W_3_4,W_1_2,gw,i,j,k)))-W_2_2[i,j,k]*(-(W_1_xz((dfdx(W_1_4,i,j,k,0.,dx)),(dfdz(W_1_2,i,j,k,0.,dx)),W_2_2,W_3_4,W_2_4,W_3_2,gw,i,j,k))))-
#             (W_1_3[i,j,k]*(-(W_2_yz((dfdy(W_2_4,i,j,k,0.,dx)),(dfdz(W_2_3,i,j,k,0.,dx)),W_3_3,W_1_4,W_3_4,W_1_3,gw,i,j,k)))-W_2_3[i,j,k]*(-(W_1_yz((dfdy(W_1_4,i,j,k,0.,dx)),(dfdz(W_1_3,i,j,k,0.,dx)),W_2_3,W_3_4,W_2_4,W_3_3,gw,i,j,k)))))+
#         gw*(ϕ_1[i,j,k]*(D_4ϕ_2((dfdz(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy))-ϕ_2[i,j,k]*(D_4ϕ_1((dfdz(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy))-ϕ_3[i,j,k]*(D_4ϕ_4((dfdz(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy))+ϕ_4[i,j,k]*(D_4ϕ_3((dfdz(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)))-
#         (dfdz(Γ_3,i,j,k,0.,dx))-gw*(W_1_4[i,j,k]*Γ_2[i,j,k]-W_2_4[i,j,k]*Γ_1[i,j,k])-γ*dW_3_4_dt[i,j,k])

#     @inbounds kt_Y_2[i,j,k] = (((d2fdx2(Y_2,i,j,k,0.,dx))+(d2fdy2(Y_2,i,j,k,0.,dx))+(d2fdz2(Y_2,i,j,k,0.,dx))+
#         gy*(ϕ_1[i,j,k]*(D_2ϕ_2((dfdx(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))-ϕ_2[i,j,k]*(D_2ϕ_1((dfdx(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))+ϕ_3[i,j,k]*(D_2ϕ_4((dfdx(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))-ϕ_4[i,j,k]*(D_2ϕ_3((dfdx(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy)))-(dfdx(Σ,i,j,k,0.,dx))-γ*dY_2_dt[i,j,k]))

#     @inbounds kt_Y_3[i,j,k] = (((d2fdx2(Y_3,i,j,k,0.,dx))+(d2fdy2(Y_3,i,j,k,0.,dx))+(d2fdz2(Y_3,i,j,k,0.,dx))+
#         gy*(ϕ_1[i,j,k]*(D_3ϕ_2((dfdy(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
#         W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))-ϕ_2[i,j,k]*(D_3ϕ_1((dfdy(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
#         W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))+ϕ_3[i,j,k]*(D_3ϕ_4((dfdy(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
#         W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))-ϕ_4[i,j,k]*(D_3ϕ_3((dfdy(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
#         W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy)))-(dfdy(Σ,i,j,k,0.,dx))-γ*dY_3_dt[i,j,k]))

#     @inbounds kt_Y_4[i,j,k] = (((d2fdx2(Y_4,i,j,k,0.,dx))+(d2fdy2(Y_4,i,j,k,0.,dx))+(d2fdz2(Y_4,i,j,k,0.,dx))+
#         gy*(ϕ_1[i,j,k]*(D_4ϕ_2((dfdz(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy))-ϕ_2[i,j,k]*(D_4ϕ_1((dfdz(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy))+ϕ_3[i,j,k]*(D_4ϕ_4((dfdz(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy))-ϕ_4[i,j,k]*(D_4ϕ_3((dfdz(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy)))-(dfdz(Σ,i,j,k,0.,dx))-γ*dY_4_dt[i,j,k]))


#     # s(Γ_1)=
#     @inbounds k_Γ_1[i,j,k] =((1.0.-gp2).*(dfdx(dW_1_2_dt,i,j,k,0.,dx) .+
#     dfdy(dW_1_3_dt,i,j,k,0.,dx) .+ dfdz(dW_1_4_dt,i,j,k,0.,dx)).+
#         gp2 .*gw.*(
#         -((W_2_2[i,j,k]).*(dW_3_2_dt[i,j,k]).-
#         (W_3_2[i,j,k]).*(dW_2_2_dt[i,j,k])).-
#         ((W_2_3[i,j,k]).*(dW_3_3_dt[i,j,k]).-
#         (W_3_3[i,j,k]).*(dW_2_3_dt[i,j,k])).-
#         ((W_2_4[i,j,k]).*(dW_3_4_dt[i,j,k]).-
#         (W_3_4[i,j,k]).*(dW_2_4_dt[i,j,k]))).+
#         # c charge from Higgs: 
#         gp2 .*gw.*((ϕ_1[i,j,k]).*((dϕ_4_dt[i,j,k])).-
#         (ϕ_4[i,j,k]).*((dϕ_1_dt[i,j,k])).+
#         (ϕ_3[i,j,k]).*((dϕ_2_dt[i,j,k])).-
#         (ϕ_2[i,j,k]).*((dϕ_3_dt[i,j,k]))))

#     # s(Γ_2)=
#     @inbounds k_Γ_2[i,j,k] =((1.0.-gp2).*(dfdx(dW_2_2_dt,i,j,k,0.,dx) .+
#     dfdy(dW_2_3_dt,i,j,k,0.,dx) .+ dfdz(dW_2_4_dt,i,j,k,0.,dx)).+
#         gp2 .*gw.*(
#         -((W_3_2[i,j,k]).*(dW_1_2_dt[i,j,k]).-
#         (W_1_2[i,j,k]).*(dW_3_2_dt[i,j,k])).-
#         ((W_3_3[i,j,k]).*(dW_1_3_dt[i,j,k]).-
#         (W_1_3[i,j,k]).*(dW_3_3_dt[i,j,k])).-
#         ((W_3_4[i,j,k]).*(dW_1_4_dt[i,j,k]).-
#         (W_1_4[i,j,k]).*(dW_3_4_dt[i,j,k]))).+
#         # c charge from Higgs: 
#         gp2 .*gw.*((ϕ_3[i,j,k]).*((dϕ_1_dt[i,j,k])).-
#         (ϕ_1[i,j,k]).*((dϕ_3_dt[i,j,k])).+
#         (ϕ_4[i,j,k]).*((dϕ_2_dt[i,j,k])).-
#         (ϕ_2[i,j,k]).*((dϕ_4_dt[i,j,k]))))

#     # s(Γ_3)=
#     @inbounds k_Γ_3[i,j,k] =((1.0.-gp2).*(dfdx(dW_3_2_dt,i,j,k,0.,dx) .+
#     dfdy(dW_3_3_dt,i,j,k,0.,dx) .+ dfdz(dW_3_4_dt,i,j,k,0.,dx)).+
#         gp2 .*gw.*(
#         -((W_1_2[i,j,k]).*(dW_2_2_dt[i,j,k]).-
#         (W_2_2[i,j,k]).*(dW_1_2_dt[i,j,k])).-
#         ((W_1_3[i,j,k]).*(dW_2_3_dt[i,j,k]).-
#         (W_2_3[i,j,k]).*(dW_1_3_dt[i,j,k])).-
#         ((W_1_4[i,j,k]).*(dW_2_4_dt[i,j,k]).-
#         (W_2_4[i,j,k]).*(dW_1_4_dt[i,j,k]))).+
#         # c current from Higgs: 
#         gp2 .*gw.*((ϕ_1[i,j,k]).*((dϕ_2_dt[i,j,k])).-
#         (ϕ_2[i,j,k]).*((dϕ_1_dt[i,j,k])).+
#         (ϕ_4[i,j,k]).*((dϕ_3_dt[i,j,k])).-
#         (ϕ_3[i,j,k]).*((dϕ_4_dt[i,j,k]))))

#     # s(Σ)=
#     @inbounds k_Σ[i,j,k] =((1.0.-gp2).*(dfdx(dY_2_dt,i,j,k,0.,dx) .+
#     dfdy(dY_3_dt,i,j,k,0.,dx) .+ dfdz(dY_4_dt,i,j,k,0.,dx)).+
#         # c current from Higgs: 
#         gp2 .*gy.*((ϕ_1[i,j,k])*((dϕ_2_dt[i,j,k])).-
#         (ϕ_2[i,j,k]).*((dϕ_1_dt[i,j,k])).+
#         (ϕ_3[i,j,k]).*((dϕ_4_dt[i,j,k])).-
#         (ϕ_4[i,j,k]).*((dϕ_3_dt[i,j,k]))))
#     ###
    
#     return
# end

@views function temp_updater_2!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    Γ_1,Γ_2,Γ_3,Σ,
    ϕ_1_t,ϕ_2_t,ϕ_3_t,ϕ_4_t,
    W_1_2_t,W_1_3_t,W_1_4_t,
    W_2_2_t,W_2_3_t,W_2_4_t,
    W_3_2_t,W_3_3_t,W_3_4_t,
    Y_2_t,Y_3_t,Y_4_t,
    Γ_1_t,Γ_2_t,Γ_3_t,Σ_t,
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
    k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,dt)

    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
    (blockIdx().y - 1) * blockDim().y + threadIdx().y,
    (blockIdx().z - 1) * blockDim().z + threadIdx().z

    @inbounds ϕ_1_t[i,j,k] = ϕ_1[i,j,k]+(dϕ_1_dt[i,j,k]).*dt/2.0
    @inbounds ϕ_2_t[i,j,k] = ϕ_2[i,j,k]+(dϕ_2_dt[i,j,k]).*dt/2.0
    @inbounds ϕ_3_t[i,j,k] = ϕ_3[i,j,k]+(dϕ_3_dt[i,j,k]).*dt/2.0
    @inbounds ϕ_4_t[i,j,k] = ϕ_4[i,j,k]+(dϕ_4_dt[i,j,k]).*dt/2.0

    @inbounds W_1_2_t[i,j,k] = W_1_2[i,j,k]+(dW_1_2_dt[i,j,k]).*dt/2.0
    @inbounds W_1_3_t[i,j,k] = W_1_3[i,j,k]+(dW_1_3_dt[i,j,k]).*dt/2.0
    @inbounds W_1_4_t[i,j,k] = W_1_4[i,j,k]+(dW_1_4_dt[i,j,k]).*dt/2.0

    @inbounds W_2_2_t[i,j,k] = W_2_2[i,j,k]+(dW_2_2_dt[i,j,k]).*dt/2.0
    @inbounds W_2_3_t[i,j,k] = W_2_3[i,j,k]+(dW_2_3_dt[i,j,k]).*dt/2.0
    @inbounds W_2_4_t[i,j,k] = W_2_4[i,j,k]+(dW_2_4_dt[i,j,k]).*dt/2.0

    @inbounds W_3_2_t[i,j,k] = W_3_2[i,j,k]+(dW_3_2_dt[i,j,k]).*dt/2.0
    @inbounds W_3_3_t[i,j,k] = W_3_3[i,j,k]+(dW_3_3_dt[i,j,k]).*dt/2.0
    @inbounds W_3_4_t[i,j,k] = W_3_4[i,j,k]+(dW_3_4_dt[i,j,k]).*dt/2.0

    @inbounds Y_2_t[i,j,k] = Y_2[i,j,k]+(dY_2_dt[i,j,k]).*dt/2.0
    @inbounds Y_3_t[i,j,k] = Y_3[i,j,k]+(dY_3_dt[i,j,k]).*dt/2.0
    @inbounds Y_4_t[i,j,k] = Y_4[i,j,k]+(dY_4_dt[i,j,k]).*dt/2.0

    @inbounds Γ_1_t[i,j,k] = Γ_1[i,j,k]+k_Γ_1[i,j,k].*dt/2.0
    @inbounds Γ_2_t[i,j,k] = Γ_2[i,j,k]+k_Γ_2[i,j,k].*dt/2.0
    @inbounds Γ_3_t[i,j,k] = Γ_3[i,j,k]+k_Γ_3[i,j,k].*dt/2.0
    @inbounds Σ_t[i,j,k] = Σ[i,j,k]+k_Σ[i,j,k].*dt/2.0

    return
end

@views function temp_updater_t_2!(dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    dϕ_1_dt_t,dϕ_2_dt_t,dϕ_3_dt_t,dϕ_4_dt_t,
    dW_1_2_dt_t,dW_1_3_dt_t,dW_1_4_dt_t,
    dW_2_2_dt_t,dW_2_3_dt_t,dW_2_4_dt_t,
    dW_3_2_dt_t,dW_3_3_dt_t,dW_3_4_dt_t,
    dY_2_dt_t,dY_3_dt_t,dY_4_dt_t,
    kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
    kt_W_1_2,kt_W_1_3,kt_W_1_4,
    kt_W_2_2,kt_W_2_3,kt_W_2_4,
    kt_W_3_2,kt_W_3_3,kt_W_3_4,
    kt_Y_2,kt_Y_3,kt_Y_4,dt)

    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
    (blockIdx().y - 1) * blockDim().y + threadIdx().y,
    (blockIdx().z - 1) * blockDim().z + threadIdx().z

    @inbounds dϕ_1_dt_t[i,j,k] = dϕ_1_dt[i,j,k]+kt_ϕ_1[i,j,k].*dt/2.0
    @inbounds dϕ_2_dt_t[i,j,k] = dϕ_2_dt[i,j,k]+kt_ϕ_2[i,j,k].*dt/2.0
    @inbounds dϕ_3_dt_t[i,j,k] = dϕ_3_dt[i,j,k]+kt_ϕ_3[i,j,k].*dt/2.0
    @inbounds dϕ_4_dt_t[i,j,k] = dϕ_4_dt[i,j,k]+kt_ϕ_4[i,j,k].*dt/2.0
    @inbounds dW_1_2_dt_t[i,j,k] = dW_1_2_dt[i,j,k]+kt_W_1_2[i,j,k].*dt/2.0
    @inbounds dW_1_3_dt_t[i,j,k] = dW_1_3_dt[i,j,k]+kt_W_1_3[i,j,k].*dt/2.0
    @inbounds dW_1_4_dt_t[i,j,k] = dW_1_4_dt[i,j,k]+kt_W_1_4[i,j,k].*dt/2.0
    @inbounds dW_2_2_dt_t[i,j,k] = dW_2_2_dt[i,j,k]+kt_W_2_2[i,j,k].*dt/2.0
    @inbounds dW_2_3_dt_t[i,j,k] = dW_2_3_dt[i,j,k]+kt_W_2_3[i,j,k].*dt/2.0
    @inbounds dW_2_4_dt_t[i,j,k] = dW_2_4_dt[i,j,k]+kt_W_2_4[i,j,k].*dt/2.0
    @inbounds dW_3_2_dt_t[i,j,k] = dW_3_2_dt[i,j,k]+kt_W_3_2[i,j,k].*dt/2.0
    @inbounds dW_3_3_dt_t[i,j,k] = dW_3_3_dt[i,j,k]+kt_W_3_3[i,j,k].*dt/2.0
    @inbounds dW_3_4_dt_t[i,j,k] = dW_3_4_dt[i,j,k]+kt_W_3_4[i,j,k].*dt/2.0
    @inbounds dY_2_dt_t[i,j,k] = dY_2_dt[i,j,k]+kt_Y_2[i,j,k].*dt/2.0
    @inbounds dY_3_dt_t[i,j,k] = dY_3_dt[i,j,k]+kt_Y_3[i,j,k].*dt/2.0
    @inbounds dY_4_dt_t[i,j,k] = dY_4_dt[i,j,k]+kt_Y_4[i,j,k].*dt/2.0

    return
end

@views function temp_updater_3!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    Γ_1,Γ_2,Γ_3,Σ,
    ϕ_1_t,ϕ_2_t,ϕ_3_t,ϕ_4_t,
    W_1_2_t,W_1_3_t,W_1_4_t,
    W_2_2_t,W_2_3_t,W_2_4_t,
    W_3_2_t,W_3_3_t,W_3_4_t,
    Y_2_t,Y_3_t,Y_4_t,
    Γ_1_t,Γ_2_t,Γ_3_t,Σ_t,
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
    k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,dt)

    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
    (blockIdx().y - 1) * blockDim().y + threadIdx().y,
    (blockIdx().z - 1) * blockDim().z + threadIdx().z

    @inbounds ϕ_1_t[i,j,k] = ϕ_1[i,j,k]+(dϕ_1_dt[i,j,k]+kt_ϕ_1[i,j,k].*dt/2.0).*dt/2.0
    @inbounds ϕ_2_t[i,j,k] = ϕ_2[i,j,k]+(dϕ_2_dt[i,j,k]+kt_ϕ_2[i,j,k].*dt/2.0).*dt/2.0
    @inbounds ϕ_3_t[i,j,k] = ϕ_3[i,j,k]+(dϕ_3_dt[i,j,k]+kt_ϕ_3[i,j,k].*dt/2.0).*dt/2.0
    @inbounds ϕ_4_t[i,j,k] = ϕ_4[i,j,k]+(dϕ_4_dt[i,j,k]+kt_ϕ_4[i,j,k].*dt/2.0).*dt/2.0

    @inbounds W_1_2_t[i,j,k] = W_1_2[i,j,k]+(dW_1_2_dt[i,j,k]+kt_W_1_2[i,j,k].*dt/2.0).*dt/2.0
    @inbounds W_1_3_t[i,j,k] = W_1_3[i,j,k]+(dW_1_3_dt[i,j,k]+kt_W_1_3[i,j,k].*dt/2.0).*dt/2.0
    @inbounds W_1_4_t[i,j,k] = W_1_4[i,j,k]+(dW_1_4_dt[i,j,k]+kt_W_1_4[i,j,k].*dt/2.0).*dt/2.0

    @inbounds W_2_2_t[i,j,k] = W_2_2[i,j,k]+(dW_2_2_dt[i,j,k]+kt_W_2_2[i,j,k].*dt/2.0).*dt/2.0
    @inbounds W_2_3_t[i,j,k] = W_2_3[i,j,k]+(dW_2_3_dt[i,j,k]+kt_W_2_3[i,j,k].*dt/2.0).*dt/2.0
    @inbounds W_2_4_t[i,j,k] = W_2_4[i,j,k]+(dW_2_4_dt[i,j,k]+kt_W_2_4[i,j,k].*dt/2.0).*dt/2.0

    @inbounds W_3_2_t[i,j,k] = W_3_2[i,j,k]+(dW_3_2_dt[i,j,k]+kt_W_3_2[i,j,k].*dt/2.0).*dt/2.0
    @inbounds W_3_3_t[i,j,k] = W_3_3[i,j,k]+(dW_3_3_dt[i,j,k]+kt_W_3_3[i,j,k].*dt/2.0).*dt/2.0
    @inbounds W_3_4_t[i,j,k] = W_3_4[i,j,k]+(dW_3_4_dt[i,j,k]+kt_W_3_4[i,j,k].*dt/2.0).*dt/2.0

    @inbounds Y_2_t[i,j,k] = Y_2[i,j,k]+(dY_2_dt[i,j,k]+kt_Y_2[i,j,k].*dt/2.0).*dt/2.0
    @inbounds Y_3_t[i,j,k] = Y_3[i,j,k]+(dY_3_dt[i,j,k]+kt_Y_3[i,j,k].*dt/2.0).*dt/2.0
    @inbounds Y_4_t[i,j,k] = Y_4[i,j,k]+(dY_4_dt[i,j,k]+kt_Y_4[i,j,k].*dt/2.0).*dt/2.0

    @inbounds Γ_1_t[i,j,k] = Γ_1[i,j,k]+k_Γ_1[i,j,k].*dt/2.0
    @inbounds Γ_2_t[i,j,k] = Γ_2[i,j,k]+k_Γ_2[i,j,k].*dt/2.0
    @inbounds Γ_3_t[i,j,k] = Γ_3[i,j,k]+k_Γ_3[i,j,k].*dt/2.0
    @inbounds Σ_t[i,j,k] = Σ[i,j,k]+k_Σ[i,j,k].*dt/2.0
    return
end

@views function temp_updater_t_3!(dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    dϕ_1_dt_t,dϕ_2_dt_t,dϕ_3_dt_t,dϕ_4_dt_t,
    dW_1_2_dt_t,dW_1_3_dt_t,dW_1_4_dt_t,
    dW_2_2_dt_t,dW_2_3_dt_t,dW_2_4_dt_t,
    dW_3_2_dt_t,dW_3_3_dt_t,dW_3_4_dt_t,
    dY_2_dt_t,dY_3_dt_t,dY_4_dt_t,
    kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
    kt_W_1_2,kt_W_1_3,kt_W_1_4,
    kt_W_2_2,kt_W_2_3,kt_W_2_4,
    kt_W_3_2,kt_W_3_3,kt_W_3_4,
    kt_Y_2,kt_Y_3,kt_Y_4,
    k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,dt)

    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
    (blockIdx().y - 1) * blockDim().y + threadIdx().y,
    (blockIdx().z - 1) * blockDim().z + threadIdx().z

    @inbounds dϕ_1_dt_t[i,j,k] = dϕ_1_dt[i,j,k]+kt_ϕ_1[i,j,k].*dt/2.0
    @inbounds dϕ_2_dt_t[i,j,k] = dϕ_2_dt[i,j,k]+kt_ϕ_2[i,j,k].*dt/2.0
    @inbounds dϕ_3_dt_t[i,j,k] = dϕ_3_dt[i,j,k]+kt_ϕ_3[i,j,k].*dt/2.0
    @inbounds dϕ_4_dt_t[i,j,k] = dϕ_4_dt[i,j,k]+kt_ϕ_4[i,j,k].*dt/2.0
    @inbounds dW_1_2_dt_t[i,j,k] = dW_1_2_dt[i,j,k]+kt_W_1_2[i,j,k].*dt/2.0
    @inbounds dW_1_3_dt_t[i,j,k] = dW_1_3_dt[i,j,k]+kt_W_1_3[i,j,k].*dt/2.0
    @inbounds dW_1_4_dt_t[i,j,k] = dW_1_4_dt[i,j,k]+kt_W_1_4[i,j,k].*dt/2.0
    @inbounds dW_2_2_dt_t[i,j,k] = dW_2_2_dt[i,j,k]+kt_W_2_2[i,j,k].*dt/2.0
    @inbounds dW_2_3_dt_t[i,j,k] = dW_2_3_dt[i,j,k]+kt_W_2_3[i,j,k].*dt/2.0
    @inbounds dW_2_4_dt_t[i,j,k] = dW_2_4_dt[i,j,k]+kt_W_2_4[i,j,k].*dt/2.0
    @inbounds dW_3_2_dt_t[i,j,k] = dW_3_2_dt[i,j,k]+kt_W_3_2[i,j,k].*dt/2.0
    @inbounds dW_3_3_dt_t[i,j,k] = dW_3_3_dt[i,j,k]+kt_W_3_3[i,j,k].*dt/2.0
    @inbounds dW_3_4_dt_t[i,j,k] = dW_3_4_dt[i,j,k]+kt_W_3_4[i,j,k].*dt/2.0
    @inbounds dY_2_dt_t[i,j,k] = dY_2_dt[i,j,k]+kt_Y_2[i,j,k].*dt/2.0
    @inbounds dY_3_dt_t[i,j,k] = dY_3_dt[i,j,k]+kt_Y_3[i,j,k].*dt/2.0
    @inbounds dY_4_dt_t[i,j,k] = dY_4_dt[i,j,k]+kt_Y_4[i,j,k].*dt/2.0

    return
end

@views function temp_updater_4!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    Γ_1,Γ_2,Γ_3,Σ,
    ϕ_1_t,ϕ_2_t,ϕ_3_t,ϕ_4_t,
    W_1_2_t,W_1_3_t,W_1_4_t,
    W_2_2_t,W_2_3_t,W_2_4_t,
    W_3_2_t,W_3_3_t,W_3_4_t,
    Y_2_t,Y_3_t,Y_4_t,
    Γ_1_t,Γ_2_t,Γ_3_t,Σ_t,
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
    k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,dt)

    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
    (blockIdx().y - 1) * blockDim().y + threadIdx().y,
    (blockIdx().z - 1) * blockDim().z + threadIdx().z

    @inbounds ϕ_1_t[i,j,k] = ϕ_1[i,j,k]+(dϕ_1_dt[i,j,k]+kt_ϕ_1[i,j,k].*dt/2.0).*dt
    @inbounds ϕ_2_t[i,j,k] = ϕ_2[i,j,k]+(dϕ_2_dt[i,j,k]+kt_ϕ_2[i,j,k].*dt/2.0).*dt
    @inbounds ϕ_3_t[i,j,k] = ϕ_3[i,j,k]+(dϕ_3_dt[i,j,k]+kt_ϕ_3[i,j,k].*dt/2.0).*dt
    @inbounds ϕ_4_t[i,j,k] = ϕ_4[i,j,k]+(dϕ_4_dt[i,j,k]+kt_ϕ_4[i,j,k].*dt/2.0).*dt

    @inbounds W_1_2_t[i,j,k] = W_1_2[i,j,k]+(dW_1_2_dt[i,j,k]+kt_W_1_2[i,j,k].*dt/2.0).*dt
    @inbounds W_1_3_t[i,j,k] = W_1_3[i,j,k]+(dW_1_3_dt[i,j,k]+kt_W_1_3[i,j,k].*dt/2.0).*dt
    @inbounds W_1_4_t[i,j,k] = W_1_4[i,j,k]+(dW_1_4_dt[i,j,k]+kt_W_1_4[i,j,k].*dt/2.0).*dt

    @inbounds W_2_2_t[i,j,k] = W_2_2[i,j,k]+(dW_2_2_dt[i,j,k]+kt_W_2_2[i,j,k].*dt/2.0).*dt
    @inbounds W_2_3_t[i,j,k] = W_2_3[i,j,k]+(dW_2_3_dt[i,j,k]+kt_W_2_3[i,j,k].*dt/2.0).*dt
    @inbounds W_2_4_t[i,j,k] = W_2_4[i,j,k]+(dW_2_4_dt[i,j,k]+kt_W_2_4[i,j,k].*dt/2.0).*dt

    @inbounds W_3_2_t[i,j,k] = W_3_2[i,j,k]+(dW_3_2_dt[i,j,k]+kt_W_3_2[i,j,k].*dt/2.0).*dt
    @inbounds W_3_3_t[i,j,k] = W_3_3[i,j,k]+(dW_3_3_dt[i,j,k]+kt_W_3_3[i,j,k].*dt/2.0).*dt
    @inbounds W_3_4_t[i,j,k] = W_3_4[i,j,k]+(dW_3_4_dt[i,j,k]+kt_W_3_4[i,j,k].*dt/2.0).*dt

    @inbounds Y_2_t[i,j,k] = Y_2[i,j,k]+(dY_2_dt[i,j,k]+kt_Y_2[i,j,k].*dt/2.0).*dt
    @inbounds Y_3_t[i,j,k] = Y_3[i,j,k]+(dY_3_dt[i,j,k]+kt_Y_3[i,j,k].*dt/2.0).*dt
    @inbounds Y_4_t[i,j,k] = Y_4[i,j,k]+(dY_4_dt[i,j,k]+kt_Y_4[i,j,k].*dt/2.0).*dt

    @inbounds Γ_1_t[i,j,k] = Γ_1[i,j,k]+k_Γ_1[i,j,k].*dt
    @inbounds Γ_2_t[i,j,k] = Γ_2[i,j,k]+k_Γ_2[i,j,k].*dt
    @inbounds Γ_3_t[i,j,k] = Γ_3[i,j,k]+k_Γ_3[i,j,k].*dt
    @inbounds Σ_t[i,j,k] = Σ[i,j,k]+k_Σ[i,j,k].*dt

    return
end

@views function temp_updater_t_4!(dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    dϕ_1_dt_t,dϕ_2_dt_t,dϕ_3_dt_t,dϕ_4_dt_t,
    dW_1_2_dt_t,dW_1_3_dt_t,dW_1_4_dt_t,
    dW_2_2_dt_t,dW_2_3_dt_t,dW_2_4_dt_t,
    dW_3_2_dt_t,dW_3_3_dt_t,dW_3_4_dt_t,
    dY_2_dt_t,dY_3_dt_t,dY_4_dt_t,
    kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
    kt_W_1_2,kt_W_1_3,kt_W_1_4,
    kt_W_2_2,kt_W_2_3,kt_W_2_4,
    kt_W_3_2,kt_W_3_3,kt_W_3_4,
    kt_Y_2,kt_Y_3,kt_Y_4,
    k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,dt)

    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
    (blockIdx().y - 1) * blockDim().y + threadIdx().y,
    (blockIdx().z - 1) * blockDim().z + threadIdx().z

    @inbounds dϕ_1_dt_t[i,j,k] = dϕ_1_dt[i,j,k]+kt_ϕ_1[i,j,k].*dt
    @inbounds dϕ_2_dt_t[i,j,k] = dϕ_2_dt[i,j,k]+kt_ϕ_2[i,j,k].*dt
    @inbounds dϕ_3_dt_t[i,j,k] = dϕ_3_dt[i,j,k]+kt_ϕ_3[i,j,k].*dt
    @inbounds dϕ_4_dt_t[i,j,k] = dϕ_4_dt[i,j,k]+kt_ϕ_4[i,j,k].*dt
    @inbounds dW_1_2_dt_t[i,j,k] = dW_1_2_dt[i,j,k]+kt_W_1_2[i,j,k].*dt
    @inbounds dW_1_3_dt_t[i,j,k] = dW_1_3_dt[i,j,k]+kt_W_1_3[i,j,k].*dt
    @inbounds dW_1_4_dt_t[i,j,k] = dW_1_4_dt[i,j,k]+kt_W_1_4[i,j,k].*dt
    @inbounds dW_2_2_dt_t[i,j,k] = dW_2_2_dt[i,j,k]+kt_W_2_2[i,j,k].*dt
    @inbounds dW_2_3_dt_t[i,j,k] = dW_2_3_dt[i,j,k]+kt_W_2_3[i,j,k].*dt
    @inbounds dW_2_4_dt_t[i,j,k] = dW_2_4_dt[i,j,k]+kt_W_2_4[i,j,k].*dt
    @inbounds dW_3_2_dt_t[i,j,k] = dW_3_2_dt[i,j,k]+kt_W_3_2[i,j,k].*dt
    @inbounds dW_3_3_dt_t[i,j,k] = dW_3_3_dt[i,j,k]+kt_W_3_3[i,j,k].*dt
    @inbounds dW_3_4_dt_t[i,j,k] = dW_3_4_dt[i,j,k]+kt_W_3_4[i,j,k].*dt
    @inbounds dY_2_dt_t[i,j,k] = dY_2_dt[i,j,k]+kt_Y_2[i,j,k].*dt
    @inbounds dY_3_dt_t[i,j,k] = dY_3_dt[i,j,k]+kt_Y_3[i,j,k].*dt
    @inbounds dY_4_dt_t[i,j,k] = dY_4_dt[i,j,k]+kt_Y_4[i,j,k].*dt

    return
end

@views function compute_aux!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    KE_Phi,GE_Phi,PE_Phi,
    ElectricE_W,MagneticE_W,ElectricE_Y,MagneticE_Y,
    B_x,B_y,B_z,
    B_x_2,B_y_2,B_z_2,
    gw,gy,gp2,vev,lambda,θ_w,dx)
    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
              (blockIdx().y - 1) * blockDim().y + threadIdx().y,
              (blockIdx().z - 1) * blockDim().z + threadIdx().z
    
    # 2-24-24: Checked and correct sp derivative calls below:
    # found and resolved incorrect calls to second order spatial
    # derivatives below

    ##Verified energy expressions: 2-25-24##

    # Energy expressions
    @inbounds PE_Phi[i,j,k] = (lambda*((ϕ_1[i,j,k])^2+
    (ϕ_2[i,j,k])^2+(ϕ_3[i,j,k])^2+(ϕ_4[i,j,k])^2-vev^2)^2)
    
    @inbounds GE_Phi[i,j,k] = ((D_2ϕ_1((dfdx(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))^2+(D_3ϕ_1((dfdy(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))^2+(D_4ϕ_1((dfdz(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy))^2+
    (D_2ϕ_2((dfdx(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))^2+(D_3ϕ_2((dfdy(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))^2+(D_4ϕ_2((dfdz(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy))^2+
    (D_2ϕ_3((dfdx(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))^2+(D_3ϕ_3((dfdy(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))^2+(D_4ϕ_3((dfdz(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy))^2+
    (D_2ϕ_4((dfdx(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))^2+(D_3ϕ_4((dfdy(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))^2+(D_4ϕ_4((dfdz(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy))^2)

    @inbounds KE_Phi[i,j,k] = ((dϕ_1_dt[i,j,k])^2+(dϕ_2_dt[i,j,k])^2+(dϕ_3_dt[i,j,k])^2+(dϕ_4_dt[i,j,k])^2)

    @inbounds ElectricE_W[i,j,k] =(0.5*
    ((dW_1_2_dt[i,j,k])^2+(dW_1_3_dt[i,j,k])^2+(dW_1_4_dt[i,j,k])^2+
    (dW_2_2_dt[i,j,k])^2+(dW_2_3_dt[i,j,k])^2+(dW_2_4_dt[i,j,k])^2+
    (dW_3_2_dt[i,j,k])^2+(dW_3_3_dt[i,j,k])^2+(dW_3_4_dt[i,j,k])^2))

    @inbounds MagneticE_W[i,j,k] = (0.5*
    ((W_1_xy((dfdx(W_1_3,i,j,k,0.,dx)),(dfdy(W_1_2,i,j,k,0.,dx)),W_2_2,W_3_3,W_2_3,W_3_2,gw,i,j,k))^2+(W_1_xz((dfdx(W_1_4,i,j,k,0.,dx)),(dfdz(W_1_2,i,j,k,0.,dx)),W_2_2,W_3_4,W_2_4,W_3_2,gw,i,j,k))^2+(W_1_yz((dfdy(W_1_4,i,j,k,0.,dx)),(dfdz(W_1_3,i,j,k,0.,dx)),W_2_3,W_3_4,W_2_4,W_3_3,gw,i,j,k))^2+
    (W_2_xy((dfdx(W_2_3,i,j,k,0.,dx)),(dfdy(W_2_2,i,j,k,0.,dx)),W_3_2,W_1_3,W_3_3,W_1_2,gw,i,j,k))^2+(W_2_xz((dfdx(W_2_4,i,j,k,0.,dx)),(dfdz(W_2_2,i,j,k,0.,dx)),W_3_2,W_1_4,W_3_4,W_1_2,gw,i,j,k))^2+(W_2_yz((dfdy(W_2_4,i,j,k,0.,dx)),(dfdz(W_2_3,i,j,k,0.,dx)),W_3_3,W_1_4,W_3_4,W_1_3,gw,i,j,k))^2+
    (W_3_xy((dfdx(W_3_3,i,j,k,0.,dx)),dfdy(W_3_2,i,j,k,0.,dx),W_1_2,W_2_3,W_1_3,W_2_2,gw,i,j,k))^2+(W_3_xz((dfdx(W_3_4,i,j,k,0.,dx)),(dfdz(W_3_2,i,j,k,0.,dx)),W_1_2,W_2_4,W_1_4,W_2_2,gw,i,j,k))^2+(W_3_yz(dfdy(W_3_4,i,j,k,0.,dx),(dfdz(W_3_3,i,j,k,0.,dx)),W_1_3,W_2_4,W_1_4,W_2_3,gw,i,j,k))^2))

    @inbounds ElectricE_Y[i,j,k] = (0.5*
    ((dY_2_dt[i,j,k])^2+(dY_3_dt[i,j,k])^2+(dY_4_dt[i,j,k])^2))

    @inbounds MagneticE_Y[i,j,k] = (0.5*((Y_2_y((dfdx(Y_3,i,j,k,0.,dx)),(dfdy(Y_2,i,j,k,0.,dx))))^2+(Y_2_z((dfdx(Y_4,i,j,k,0.,dx)),(dfdz(Y_2,i,j,k,0.,dx))))^2+(Y_3_z((dfdy(Y_4,i,j,k,0.,dx)),(dfdz(Y_3,i,j,k,0.,dx))))^2))

    ϕ_mag = (ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2)

    # Higgs n definitions

    if ϕ_mag == 0.0
        n_1= 0.0
        n_2= 0.0
        n_3= 0.0
    else
        n_1=-2. *(ϕ_1[i,j,k]*ϕ_3[i,j,k]+ϕ_2[i,j,k]*ϕ_4[i,j,k])/ϕ_mag
        n_2=+2. *(ϕ_2[i,j,k]*ϕ_3[i,j,k]-ϕ_1[i,j,k]*ϕ_4[i,j,k])/ϕ_mag
        n_3=(-ϕ_1[i,j,k]^2-ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2)/ϕ_mag
    end
    
    # Magnetic field defintions
    # $A_{ij} = stw*na*W^a_{ij}+ctw*Y_{ij}
    #      -i*(2*stw/(gw*vev^2))*((D_i\Phi)^\dag D_j\Phi-(D_j\Phi)^\dag D_i\Phi)
    # and,            
    # B_x= -A_{yx}, B_y= -A_{zx}, B_z= -A_{xy} 

    @inbounds B_x[i,j,k] = -(sin(θ_w)*(n_1*(W_1_yz((dfdy(W_1_4,i,j,k,0.,dx)),(dfdz(W_1_3,i,j,k,0.,dx)),W_2_3,W_3_4,W_2_4,W_3_3,gw,i,j,k))+n_2*(W_2_yz((dfdy(W_2_4,i,j,k,0.,dx)),(dfdz(W_2_3,i,j,k,0.,dx)),W_3_3,W_1_4,W_3_4,W_1_3,gw,i,j,k))+n_3*(W_3_yz(dfdy(W_3_4,i,j,k,0.,dx),(dfdz(W_3_3,i,j,k,0.,dx)),W_1_3,W_2_4,W_1_4,W_2_3,gw,i,j,k)))+cos(θ_w)*(Y_3_z((dfdy(Y_4,i,j,k,0.,dx)),(dfdz(Y_3,i,j,k,0.,dx))))
    +(4. *sin(θ_w)/(gw*vev^2))*((D_3ϕ_1((dfdy(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
        W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))*(D_4ϕ_2((dfdz(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
        W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy))-(D_3ϕ_2((dfdy(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
        W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))*(D_4ϕ_1((dfdz(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
        W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy))
    +(D_3ϕ_3((dfdy(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
        W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))*(D_4ϕ_4((dfdz(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
        W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy))-(D_3ϕ_4((dfdy(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
        W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))*(D_4ϕ_3((dfdz(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
        W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy))))

    @inbounds B_y[i,j,k] = -(sin(θ_w)*(n_1*(-(W_1_xz((dfdx(W_1_4,i,j,k,0.,dx)),(dfdz(W_1_2,i,j,k,0.,dx)),W_2_2,W_3_4,W_2_4,W_3_2,gw,i,j,k)))+n_2*(-(W_2_xz((dfdx(W_2_4,i,j,k,0.,dx)),(dfdz(W_2_2,i,j,k,0.,dx)),W_3_2,W_1_4,W_3_4,W_1_2,gw,i,j,k)))+n_3*(-(W_3_xz((dfdx(W_3_4,i,j,k,0.,dx)),(dfdz(W_3_2,i,j,k,0.,dx)),W_1_2,W_2_4,W_1_4,W_2_2,gw,i,j,k))))+cos(θ_w)*(-(Y_2_z((dfdx(Y_4,i,j,k,0.,dx)),(dfdz(Y_2,i,j,k,0.,dx)))))
    +(4. *sin(θ_w)/(gw*vev^2))*((D_4ϕ_1((dfdz(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
        W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy))*(D_2ϕ_2((dfdx(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
        W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))-(D_4ϕ_2((dfdz(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
        W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy))*(D_2ϕ_1((dfdx(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
        W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))
    +(D_4ϕ_3((dfdz(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
        W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy))*(D_2ϕ_4((dfdx(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
        W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))-(D_4ϕ_4((dfdz(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
        W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy))*(D_2ϕ_3((dfdx(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
        W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))))

    @inbounds B_z[i,j,k] = -(sin(θ_w)*(n_1*((W_1_xy((dfdx(W_1_3,i,j,k,0.,dx)),(dfdy(W_1_2,i,j,k,0.,dx)),W_2_2,W_3_3,W_2_3,W_3_2,gw,i,j,k)))+n_2*(W_2_xy((dfdx(W_2_3,i,j,k,0.,dx)),(dfdy(W_2_2,i,j,k,0.,dx)),W_3_2,W_1_3,W_3_3,W_1_2,gw,i,j,k))+n_3*(W_3_xy((dfdx(W_3_3,i,j,k,0.,dx)),dfdy(W_3_2,i,j,k,0.,dx),W_1_2,W_2_3,W_1_3,W_2_2,gw,i,j,k)))+cos(θ_w)*(Y_2_y((dfdx(Y_3,i,j,k,0.,dx)),(dfdy(Y_2,i,j,k,0.,dx))))
    +(4. *sin(θ_w)/(gw*vev^2))*((D_2ϕ_1((dfdx(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
        W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))*(D_3ϕ_2((dfdy(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
        W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))-(D_2ϕ_2((dfdx(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
        W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))*(D_3ϕ_1((dfdy(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
        W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))
    +(D_2ϕ_3((dfdx(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
        W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))*(D_3ϕ_4((dfdy(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
        W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))-(D_2ϕ_4((dfdx(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
        W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))*(D_3ϕ_3((dfdy(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
        W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))))

    if ϕ_mag == 0.0
        @inbounds B_x_2[i,j,k] = 0.0
        @inbounds B_y_2[i,j,k] = 0.0
        @inbounds B_z_2[i,j,k] = 0.0
    else
        @inbounds B_x_2[i,j,k] = -(sin(θ_w)*(n_1*(W_1_yz((dfdy(W_1_4,i,j,k,0.,dx)),(dfdz(W_1_3,i,j,k,0.,dx)),W_2_3,W_3_4,W_2_4,W_3_3,gw,i,j,k))+n_2*(W_2_yz((dfdy(W_2_4,i,j,k,0.,dx)),(dfdz(W_2_3,i,j,k,0.,dx)),W_3_3,W_1_4,W_3_4,W_1_3,gw,i,j,k))+n_3*(W_3_yz(dfdy(W_3_4,i,j,k,0.,dx),(dfdz(W_3_3,i,j,k,0.,dx)),W_1_3,W_2_4,W_1_4,W_2_3,gw,i,j,k)))+cos(θ_w)*(Y_3_z((dfdy(Y_4,i,j,k,0.,dx)),(dfdz(Y_3,i,j,k,0.,dx))))
        +(4. *sin(θ_w)/(gw*ϕ_mag^2))*((D_3ϕ_1((dfdy(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
        W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))*(D_4ϕ_2((dfdz(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
        W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy))-(D_3ϕ_2((dfdy(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
        W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))*(D_4ϕ_1((dfdz(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
        W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy))
        +(D_3ϕ_3((dfdy(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
        W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))*(D_4ϕ_4((dfdz(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
        W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy))-(D_3ϕ_4((dfdy(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
        W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))*(D_4ϕ_3((dfdz(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
        W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy))))

        @inbounds B_y_2[i,j,k] = -(sin(θ_w)*(n_1*(-(W_1_xz((dfdx(W_1_4,i,j,k,0.,dx)),(dfdz(W_1_2,i,j,k,0.,dx)),W_2_2,W_3_4,W_2_4,W_3_2,gw,i,j,k)))+n_2*(-(W_2_xz((dfdx(W_2_4,i,j,k,0.,dx)),(dfdz(W_2_2,i,j,k,0.,dx)),W_3_2,W_1_4,W_3_4,W_1_2,gw,i,j,k)))+n_3*(-(W_3_xz((dfdx(W_3_4,i,j,k,0.,dx)),(dfdz(W_3_2,i,j,k,0.,dx)),W_1_2,W_2_4,W_1_4,W_2_2,gw,i,j,k))))+cos(θ_w)*(-(Y_2_z((dfdx(Y_4,i,j,k,0.,dx)),(dfdz(Y_2,i,j,k,0.,dx)))))
        +(4. *sin(θ_w)/(gw*ϕ_mag^2))*((D_4ϕ_1((dfdz(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
        W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy))*(D_2ϕ_2((dfdx(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
        W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))-(D_4ϕ_2((dfdz(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
        W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy))*(D_2ϕ_1((dfdx(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
        W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))
        +(D_4ϕ_3((dfdz(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
        W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy))*(D_2ϕ_4((dfdx(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
        W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))-(D_4ϕ_4((dfdz(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
        W_1_4[i,j,k],W_2_4[i,j,k],W_3_4[i,j,k],Y_4[i,j,k],gw,gy))*(D_2ϕ_3((dfdx(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
        W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))))

        @inbounds B_z_2[i,j,k] = -(sin(θ_w)*(n_1*((W_1_xy((dfdx(W_1_3,i,j,k,0.,dx)),(dfdy(W_1_2,i,j,k,0.,dx)),W_2_2,W_3_3,W_2_3,W_3_2,gw,i,j,k)))+n_2*(W_2_xy((dfdx(W_2_3,i,j,k,0.,dx)),(dfdy(W_2_2,i,j,k,0.,dx)),W_3_2,W_1_3,W_3_3,W_1_2,gw,i,j,k))+n_3*(W_3_xy((dfdx(W_3_3,i,j,k,0.,dx)),dfdy(W_3_2,i,j,k,0.,dx),W_1_2,W_2_3,W_1_3,W_2_2,gw,i,j,k)))+cos(θ_w)*(Y_2_y((dfdx(Y_3,i,j,k,0.,dx)),(dfdy(Y_2,i,j,k,0.,dx))))
        +(4. *sin(θ_w)/(gw*ϕ_mag^2))*((D_2ϕ_1((dfdx(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
        W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))*(D_3ϕ_2((dfdy(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
        W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))-(D_2ϕ_2((dfdx(ϕ_2,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
        W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))*(D_3ϕ_1((dfdy(ϕ_1,i,j,k,0.,dx)),ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
        W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))
        +(D_2ϕ_3((dfdx(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
        W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))*(D_3ϕ_4((dfdy(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
        W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))-(D_2ϕ_4((dfdx(ϕ_4,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
        W_1_2[i,j,k],W_2_2[i,j,k],W_3_2[i,j,k],Y_2[i,j,k],gw,gy))*(D_3ϕ_3((dfdy(ϕ_3,i,j,k,0.,dx)),ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
        W_1_3[i,j,k],W_2_3[i,j,k],W_3_3[i,j,k],Y_3[i,j,k],gw,gy))))
    end

    return
end

@views function initializer!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,ib,jb,kb,p1,p2,p3,p4,rkx,rky,rkz,dx,mH)
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

function k_vec(x,y,z,Nx,Ny,Nz)
    x=x#-1
    y=y#-1
    z=z#-1
    if x <= floor(Int,Nx/2)
        K_x = x
    else
        K_x = x - Nx
    end

    if y <= floor(Int,Ny/2)
        K_y = y
    else
        K_y = y - Ny
    end

    if z <= floor(Int,Nz/2)
        K_z = z
    else
        K_z = z - Nz
    end

    return [K_x,K_y,K_z]
end

function k_mag(x,y,z,Nx,Ny,Nz,dx)
    #Physical k magnitude#
    k=k_vec(x,y,z,Nx,Ny,Nz)
    kx = 2.0*pi*k[1]/(dx*Nx)
    ky = 2.0*pi*k[2]/(dx*Ny)
    kz = 2.0*pi*k[3]/(dx*Nz)

    return sqrt(kx^2+ky^2+kz^2)
end

function thermal_initializer!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,gw,gy,gp2,vev,dx,T,Nx,Ny,Nz)
    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
    (blockIdx().y - 1) * blockDim().y + threadIdx().y,
    (blockIdx().z - 1) * blockDim().z + threadIdx().z
    #Momentum space arrays
    ϕ_1_k = zeros(prec,(Nx,Ny,Nz))
    ϕ_2_k = zeros(prec,(Nx,Ny,Nz))
    ϕ_3_k = zeros(prec,(Nx,Ny,Nz))
    ϕ_4_k = zeros(prec,(Nx,Ny,Nz))
    Random.seed!(123456)
    mag_arr = rand(Normal(0.0,2.0),(Nx,Ny,Nz,4))
    
    for i in range(1,Nx,step=1)
        for j in range(1,Ny,step=1)
            for k in range(1,Nz,step=1)
                ω_p = sqrt(k_mag(i,j,k,Nx,Ny,Nz,dx)^2+32.0)
                n_p = 1.0/(exp(ω_p/T)-1.0)
                # println(i," ",j," ",k," ",ω_p," ",n_p)
                ϕ_1_k[i,j,k] = sqrt(n_p/(2.0*ω_p))*mag_arr[i,j,k,1]
                ϕ_2_k[i,j,k] = sqrt(n_p/(2.0*ω_p))*mag_arr[i,j,k,2]
                ϕ_3_k[i,j,k] = sqrt(n_p/(2.0*ω_p))*mag_arr[i,j,k,3]
                ϕ_4_k[i,j,k] = sqrt(n_p/(2.0*ω_p))*mag_arr[i,j,k,4]
            end
        end
    end
    
    copyto!(ϕ_1,real.(bfft(ϕ_1_k))/(Nx*Ny*Nz*dx^3))
    copyto!(ϕ_2,real.(bfft(ϕ_2_k))/(Nx*Ny*Nz*dx^3))
    copyto!(ϕ_3,real.(bfft(ϕ_3_k))/(Nx*Ny*Nz*dx^3))
    copyto!(ϕ_4,real.(bfft(ϕ_4_k))/(Nx*Ny*Nz*dx^3))

    mod_ϕ = sqrt.(ϕ_1.^2 .+ϕ_2.^2 .+ϕ_3.^2 .+ϕ_4.^2)

    # max_ϕ = max(maximum(ϕ_1),maximum(ϕ_2),maximum(ϕ_3),maximum(ϕ_4))
    max_ϕ = maximum(mod_ϕ)

    if max_ϕ>1
        ϕ_1 = ϕ_1 ./max_ϕ
        ϕ_2 = ϕ_2 ./max_ϕ
        ϕ_3 = ϕ_3 ./max_ϕ
        ϕ_4 = ϕ_4 ./max_ϕ
    end
    
    return
end

function thermal_initializer_ϕ!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    gw,gy,gp2,vev,dx,T,meff_sq,
    Nx,Ny,Nz)
    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
    (blockIdx().y - 1) * blockDim().y + threadIdx().y,
    (blockIdx().z - 1) * blockDim().z + threadIdx().z

    # i = (ix-2)+(grid_x)*(size(ϕ_1,1)-2)
    # j = (iy-2)+(grid_y)*(size(ϕ_1,2)-2)
    # k = (iz-2)+(grid_z)*(size(ϕ_1,3)-2)

    kx = 2.0*sin(0.5*i*2.0*pi/Nx)/dx
    ky = 2.0*sin(0.5*j*2.0*pi/Ny)/dx
    kz = 2.0*sin(0.5*k*2.0*pi/Nz)/dx

    mod_k = sqrt(2.0*((1.0-cos(i*2.0*pi/Nx))/(dx^2)+
                 (1.0-cos(j*2.0*pi/Ny))/(dx^2)+
                 (1.0-cos(k*2.0*pi/Nz))/(dx^2)))

    # ω_p = sqrt(k_mag(i,j,k,Nx_g,Ny_g,Nz_g,dx)^2+32.0)
    ω_p = sqrt(mod_k^2+meff_sq)
    n_p = 1.0/(exp(ω_p/T)-1.0)
    vol = Nx*Ny*Nz*dx^3
    # println(i," ",j," ",k," ",ω_p," ",n_p)
    ϕ_1[i,j,k] = sqrt(n_p/(ω_p*vol))*ϕ_1[i,j,k]
    ϕ_2[i,j,k] = sqrt(n_p/(ω_p*vol))*ϕ_2[i,j,k]
    ϕ_3[i,j,k] = sqrt(n_p/(ω_p*vol))*ϕ_3[i,j,k]
    ϕ_4[i,j,k] = sqrt(n_p/(ω_p*vol))*ϕ_4[i,j,k]
    #         end
    #     end
    # end
    
    # mod_ϕ = sqrt.(ϕ_1.^2 .+ϕ_2.^2 .+ϕ_3.^2 .+ϕ_4.^2)

    # # max_ϕ = max(maximum(ϕ_1),maximum(ϕ_2),maximum(ϕ_3),maximum(ϕ_4))
    # max_ϕ = maximum(mod_ϕ)

    # if max_ϕ>1
    #     ϕ_1 = ϕ_1 ./max_ϕ
    #     ϕ_2 = ϕ_2 ./max_ϕ
    #     ϕ_3 = ϕ_3 ./max_ϕ
    #     ϕ_4 = ϕ_4 ./max_ϕ
    # end

    return
end

function thermal_initializer_gauge!(gw,gy,gp2,vev,dx,T,
    Nx,Ny,Nz,
    r_1,r_2,r_3,
    r_1_i,r_2_i,r_3_i,
    x_1,x_2,x_3,x_4)

    # i_g = (ix-2)+(grid_x)*(size(r_1,1)-2)
    # j_g = (iy-2)+(grid_y)*(size(r_1,2)-2)
    # k_g = (iz-2)+(grid_z)*(size(r_1,3)-2)

    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
    (blockIdx().y - 1) * blockDim().y + threadIdx().y,
    (blockIdx().z - 1) * blockDim().z + threadIdx().z

    kx = 2.0*sin(0.5*i*2.0*pi/Nx)/dx
    ky = 2.0*sin(0.5*j*2.0*pi/Ny)/dx
    kz = 2.0*sin(0.5*k*2.0*pi/Nz)/dx

    mod_k = sqrt(2.0*((1.0-cos(i*2.0*pi/Nx))/(dx^2)+
                 (1.0-cos(j*2.0*pi/Ny))/(dx^2)+
                 (1.0-cos(k*2.0*pi/Nz))/(dx^2)))

    ϵ_1_x = r_2[i,j,k]*kz - r_3[i,j,k]*ky
    ϵ_1_y = r_3[i,j,k]*kx - r_1[i,j,k]*kz
    ϵ_1_z = r_1[i,j,k]*ky - r_2[i,j,k]*kx
    len_ϵ_1 = sqrt(ϵ_1_x^2+ϵ_1_y^2+ϵ_1_z^2)
    ϵ_1_x = ϵ_1_x/len_ϵ_1
    ϵ_1_y = ϵ_1_y/len_ϵ_1
    ϵ_1_z = ϵ_1_z/len_ϵ_1

    ϵ_2_x = ϵ_1_y*kz - ϵ_1_z*ky
    ϵ_2_y = ϵ_1_z*kx - ϵ_1_x*kz
    ϵ_2_z = ϵ_1_x*ky - ϵ_1_y*kx
    len_ϵ_2 = sqrt(ϵ_2_x^2+ϵ_2_y^2+ϵ_2_z^2)
    ϵ_2_x = ϵ_2_x/len_ϵ_2
    ϵ_2_y = ϵ_2_y/len_ϵ_2
    ϵ_2_z = ϵ_2_z/len_ϵ_2

    ω_p = sqrt(mod_k^2)
    n_p = 1.0/(exp(ω_p/T)-1.0)
    vol = Nx*Ny*Nz*dx^3
    #If loop to deal with the singularity in k=0 modes
    if ω_p!=0.0
        r_1[i,j,k] = (sqrt(n_p/(ω_p*vol))*
                    (ϵ_1_x*x_1[i,j,k]+ϵ_2_x*x_2[i,j,k]))
        r_2[i,j,k] = (sqrt(n_p/(ω_p*vol))*
                    (ϵ_1_y*x_1[i,j,k]+ϵ_2_y*x_2[i,j,k]))
        r_3[i,j,k] = (sqrt(n_p/(ω_p*vol))*
                    (ϵ_1_z*x_1[i,j,k]+ϵ_2_z*x_2[i,j,k]))

        r_1_i[i,j,k] = (sqrt(n_p/(ω_p*vol))*
                    (ϵ_1_x*x_3[i,j,k]+ϵ_2_x*x_4[i,j,k]))
        r_2_i[i,j,k] = (sqrt(n_p/(ω_p*vol))*
                    (ϵ_1_y*x_3[i,j,k]+ϵ_2_y*x_4[i,j,k]))
        r_3_i[i,j,k] = (sqrt(n_p/(ω_p*vol))*
                    (ϵ_1_z*x_3[i,j,k]+ϵ_2_z*x_4[i,j,k]))            
    else
        r_1[i,j,k] = 0.0
        r_2[i,j,k] = 0.0
        r_3[i,j,k] = 0.0
        r_1_i[i,j,k] = 0.0
        r_2_i[i,j,k] = 0.0
        r_3_i[i,j,k] = 0.0
    end
    # if ix==10&&iy==10&&iz==10
    #     @cuprintln(ϵ_1_x," ",ϵ_2_x," ",x_1[ix,iy,iz]," ",
    #     x_2[ix,iy,iz]," ",ω_p/T," ",n_p," ",r_1[ix,iy,iz])
    # end
    # Ax_i[ix,iy,iz] = (sqrt(n_p/(2.0*ω_p))*
    #                (ϵ_1_x*x_3[ix,iy,iz]+ϵ_2_x*x_4[ix,iy,iz]))
    # Ay_i[ix,iy,iz] = (sqrt(n_p/(2.0*ω_p))*
    #                (ϵ_1_y*x_3[ix,iy,iz]+ϵ_2_y*x_4[ix,iy,iz]))
    # Az_i[ix,iy,iz] = (sqrt(n_p/(2.0*ω_p))*
    #                (ϵ_1_z*x_3[ix,iy,iz]+ϵ_2_z*x_4[ix,iy,iz]))

    return
end

@views function updater_1!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
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
    Γ_1_n,Γ_2_n,Γ_3_n,Σ_n,
    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,dt,fac)

    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
    (blockIdx().y - 1) * blockDim().y + threadIdx().y,
    (blockIdx().z - 1) * blockDim().z + threadIdx().z

    @inbounds ϕ_1_n[i,j,k] = dϕ_1_dt[i,j,k]*dt*fac + ϕ_1[i,j,k]
    @inbounds ϕ_2_n[i,j,k] = dϕ_2_dt[i,j,k]*dt*fac + ϕ_2[i,j,k]
    @inbounds ϕ_3_n[i,j,k] = dϕ_3_dt[i,j,k]*dt*fac + ϕ_3[i,j,k]
    @inbounds ϕ_4_n[i,j,k] = dϕ_4_dt[i,j,k]*dt*fac + ϕ_4[i,j,k]
    @inbounds W_1_2_n[i,j,k] = dW_1_2_dt[i,j,k]*dt*fac + W_1_2[i,j,k]
    @inbounds W_1_3_n[i,j,k] = dW_1_3_dt[i,j,k]*dt*fac + W_1_3[i,j,k]
    @inbounds W_1_4_n[i,j,k] = dW_1_4_dt[i,j,k]*dt*fac + W_1_4[i,j,k]
    @inbounds W_2_2_n[i,j,k] = dW_2_2_dt[i,j,k]*dt*fac + W_2_2[i,j,k]
    @inbounds W_2_3_n[i,j,k] = dW_2_3_dt[i,j,k]*dt*fac + W_2_3[i,j,k]
    @inbounds W_2_4_n[i,j,k] = dW_2_4_dt[i,j,k]*dt*fac + W_2_4[i,j,k]
    @inbounds W_3_2_n[i,j,k] = dW_3_2_dt[i,j,k]*dt*fac + W_3_2[i,j,k]
    @inbounds W_3_3_n[i,j,k] = dW_3_3_dt[i,j,k]*dt*fac + W_3_3[i,j,k]
    @inbounds W_3_4_n[i,j,k] = dW_3_4_dt[i,j,k]*dt*fac + W_3_4[i,j,k]
    @inbounds Y_2_n[i,j,k] = dY_2_dt[i,j,k]*dt*fac + Y_2[i,j,k]
    @inbounds Y_3_n[i,j,k] = dY_3_dt[i,j,k]*dt*fac + Y_3[i,j,k]
    @inbounds Y_4_n[i,j,k] = dY_4_dt[i,j,k]*dt*fac + Y_4[i,j,k]
    @inbounds Γ_1_n[i,j,k] = k_Γ_1[i,j,k]*dt*fac + Γ_1[i,j,k]
    @inbounds Γ_2_n[i,j,k] = k_Γ_2[i,j,k]*dt*fac + Γ_2[i,j,k]
    @inbounds Γ_3_n[i,j,k] = k_Γ_3[i,j,k]*dt*fac + Γ_3[i,j,k]
    @inbounds Σ_n[i,j,k] = k_Σ[i,j,k]*dt*fac + Σ[i,j,k]

    return
end

@views function updater_t_1!(dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    dϕ_1_dt_n,dϕ_2_dt_n,dϕ_3_dt_n,dϕ_4_dt_n,
    dW_1_2_dt_n,dW_1_3_dt_n,dW_1_4_dt_n,
    dW_2_2_dt_n,dW_2_3_dt_n,dW_2_4_dt_n,
    dW_3_2_dt_n,dW_3_3_dt_n,dW_3_4_dt_n,
    dY_2_dt_n,dY_3_dt_n,dY_4_dt_n,
    kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
    kt_W_1_2,kt_W_1_3,kt_W_1_4,
    kt_W_2_2,kt_W_2_3,kt_W_2_4,
    kt_W_3_2,kt_W_3_3,kt_W_3_4,
    kt_Y_2,kt_Y_3,kt_Y_4,dt,fac)

    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
    (blockIdx().y - 1) * blockDim().y + threadIdx().y,
    (blockIdx().z - 1) * blockDim().z + threadIdx().z

    @inbounds dϕ_1_dt_n[i,j,k] = kt_ϕ_1[i,j,k]*dt*fac + dϕ_1_dt[i,j,k]
    @inbounds dϕ_2_dt_n[i,j,k] = kt_ϕ_2[i,j,k]*dt*fac + dϕ_2_dt[i,j,k]
    @inbounds dϕ_3_dt_n[i,j,k] = kt_ϕ_3[i,j,k]*dt*fac + dϕ_3_dt[i,j,k]
    @inbounds dϕ_4_dt_n[i,j,k] = kt_ϕ_4[i,j,k]*dt*fac + dϕ_4_dt[i,j,k]
    @inbounds dW_1_2_dt_n[i,j,k] = kt_W_1_2[i,j,k]*dt*fac + dW_1_2_dt[i,j,k]
    @inbounds dW_1_3_dt_n[i,j,k] = kt_W_1_3[i,j,k]*dt*fac + dW_1_3_dt[i,j,k]
    @inbounds dW_1_4_dt_n[i,j,k] = kt_W_1_4[i,j,k]*dt*fac + dW_1_4_dt[i,j,k]
    @inbounds dW_2_2_dt_n[i,j,k] = kt_W_2_2[i,j,k]*dt*fac + dW_2_2_dt[i,j,k]
    @inbounds dW_2_3_dt_n[i,j,k] = kt_W_2_3[i,j,k]*dt*fac + dW_2_3_dt[i,j,k]
    @inbounds dW_2_4_dt_n[i,j,k] = kt_W_2_4[i,j,k]*dt*fac + dW_2_4_dt[i,j,k]
    @inbounds dW_3_2_dt_n[i,j,k] = kt_W_3_2[i,j,k]*dt*fac + dW_3_2_dt[i,j,k]
    @inbounds dW_3_3_dt_n[i,j,k] = kt_W_3_3[i,j,k]*dt*fac + dW_3_3_dt[i,j,k]
    @inbounds dW_3_4_dt_n[i,j,k] = kt_W_3_4[i,j,k]*dt*fac + dW_3_4_dt[i,j,k]
    @inbounds dY_2_dt_n[i,j,k] = kt_Y_2[i,j,k]*dt*fac + dY_2_dt[i,j,k]
    @inbounds dY_3_dt_n[i,j,k] = kt_Y_3[i,j,k]*dt*fac + dY_3_dt[i,j,k]
    @inbounds dY_4_dt_n[i,j,k] = kt_Y_4[i,j,k]*dt*fac + dY_4_dt[i,j,k]
    # dΓ_1_dt_n = kt_Γ_1*dt*fac + dΓ_1_dt
    # dΓ_2_dt_n = kt_Γ_2*dt*fac + dΓ_2_dt
    # dΓ_3_dt_n = kt_Γ_3*dt*fac + dΓ_3_dt
    # dΣ_dt_n = kt_Σ*dt*fac + dΣ_dt

    return
end

@views function updater!(Γ_1_n,Γ_2_n,Γ_3_n,Σ_n,
    k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,fac)

    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
    (blockIdx().y - 1) * blockDim().y + threadIdx().y,
    (blockIdx().z - 1) * blockDim().z + threadIdx().z

    @inbounds Γ_1_n[i,j,k] = k_Γ_1[i,j,k] *fac + Γ_1_n[i,j,k]
    @inbounds Γ_2_n[i,j,k] = k_Γ_2[i,j,k] *fac + Γ_2_n[i,j,k]
    @inbounds Γ_3_n[i,j,k] = k_Γ_3[i,j,k] *fac + Γ_3_n[i,j,k]
    @inbounds Σ_n[i,j,k] = k_Σ[i,j,k] *fac + Σ_n[i,j,k]

    return
end

@views function updater_t!(dϕ_1_dt_n,dϕ_2_dt_n,dϕ_3_dt_n,dϕ_4_dt_n,
    dW_1_2_dt_n,dW_1_3_dt_n,dW_1_4_dt_n,
    dW_2_2_dt_n,dW_2_3_dt_n,dW_2_4_dt_n,
    dW_3_2_dt_n,dW_3_3_dt_n,dW_3_4_dt_n,
    dY_2_dt_n,dY_3_dt_n,dY_4_dt_n,
    kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
    kt_W_1_2,kt_W_1_3,kt_W_1_4,
    kt_W_2_2,kt_W_2_3,kt_W_2_4,
    kt_W_3_2,kt_W_3_3,kt_W_3_4,
    kt_Y_2,kt_Y_3,kt_Y_4,
    Γ_1_n,Γ_2_n,Γ_3_n,Σ_n,
    k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,dt,fac)

    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
    (blockIdx().y - 1) * blockDim().y + threadIdx().y,
    (blockIdx().z - 1) * blockDim().z + threadIdx().z
    
    @inbounds Γ_1_n[i,j,k] = k_Γ_1[i,j,k]*dt*fac + Γ_1_n[i,j,k]
    @inbounds Γ_2_n[i,j,k] = k_Γ_2[i,j,k]*dt*fac + Γ_2_n[i,j,k]
    @inbounds Γ_3_n[i,j,k] = k_Γ_3[i,j,k]*dt*fac + Γ_3_n[i,j,k]
    @inbounds Σ_n[i,j,k] = k_Σ[i,j,k]*dt*fac + Σ_n[i,j,k]

    @inbounds dϕ_1_dt_n[i,j,k] = kt_ϕ_1[i,j,k]*dt*fac + dϕ_1_dt_n[i,j,k]
    @inbounds dϕ_2_dt_n[i,j,k] = kt_ϕ_2[i,j,k]*dt*fac + dϕ_2_dt_n[i,j,k]
    @inbounds dϕ_3_dt_n[i,j,k] = kt_ϕ_3[i,j,k]*dt*fac + dϕ_3_dt_n[i,j,k]
    @inbounds dϕ_4_dt_n[i,j,k] = kt_ϕ_4[i,j,k]*dt*fac + dϕ_4_dt_n[i,j,k]
    @inbounds dW_1_2_dt_n[i,j,k] = kt_W_1_2[i,j,k]*dt*fac + dW_1_2_dt_n[i,j,k]
    @inbounds dW_1_3_dt_n[i,j,k] = kt_W_1_3[i,j,k]*dt*fac + dW_1_3_dt_n[i,j,k]
    @inbounds dW_1_4_dt_n[i,j,k] = kt_W_1_4[i,j,k]*dt*fac + dW_1_4_dt_n[i,j,k]
    @inbounds dW_2_2_dt_n[i,j,k] = kt_W_2_2[i,j,k]*dt*fac + dW_2_2_dt_n[i,j,k]
    @inbounds dW_2_3_dt_n[i,j,k] = kt_W_2_3[i,j,k]*dt*fac + dW_2_3_dt_n[i,j,k]
    @inbounds dW_2_4_dt_n[i,j,k] = kt_W_2_4[i,j,k]*dt*fac + dW_2_4_dt_n[i,j,k]
    @inbounds dW_3_2_dt_n[i,j,k] = kt_W_3_2[i,j,k]*dt*fac + dW_3_2_dt_n[i,j,k]
    @inbounds dW_3_3_dt_n[i,j,k] = kt_W_3_3[i,j,k]*dt*fac + dW_3_3_dt_n[i,j,k]
    @inbounds dW_3_4_dt_n[i,j,k] = kt_W_3_4[i,j,k]*dt*fac + dW_3_4_dt_n[i,j,k]
    @inbounds dY_2_dt_n[i,j,k] = kt_Y_2[i,j,k]*dt*fac + dY_2_dt_n[i,j,k]
    @inbounds dY_3_dt_n[i,j,k] = kt_Y_3[i,j,k]*dt*fac + dY_3_dt_n[i,j,k]
    @inbounds dY_4_dt_n[i,j,k] = kt_Y_4[i,j,k]*dt*fac + dY_4_dt_n[i,j,k]
    # dΓ_1_dt_n = kt_Γ_1*dt*fac + dΓ_1_dt
    # dΓ_2_dt_n = kt_Γ_2*dt*fac + dΓ_2_dt
    # dΓ_3_dt_n = kt_Γ_3*dt*fac + dΓ_3_dt
    # dΣ_dt_n = kt_Σ*dt*fac + dΣ_dt

    return
end

@views function triv_updater!(ϕ_1_n,ϕ_2_n,ϕ_3_n,ϕ_4_n,
    W_1_2_n,W_1_3_n,W_1_4_n,
    W_2_2_n,W_2_3_n,W_2_4_n,
    W_3_2_n,W_3_3_n,W_3_4_n,
    Y_2_n,Y_3_n,Y_4_n,
    kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
    kt_W_1_2,kt_W_1_3,kt_W_1_4,
    kt_W_2_2,kt_W_2_3,kt_W_2_4,
    kt_W_3_2,kt_W_3_3,kt_W_3_4,
    kt_Y_2,kt_Y_3,kt_Y_4,
    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    dt,fac,fac_2)

    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
    (blockIdx().y - 1) * blockDim().y + threadIdx().y,
    (blockIdx().z - 1) * blockDim().z + threadIdx().z

    @inbounds ϕ_1_n[i,j,k] =ϕ_1_n[i,j,k]+dt*fac*((dϕ_1_dt[i,j,k])+(kt_ϕ_1[i,j,k])*(dt/fac_2))
    @inbounds ϕ_2_n[i,j,k] =ϕ_2_n[i,j,k]+dt*fac*((dϕ_2_dt[i,j,k])+(kt_ϕ_2[i,j,k])*(dt/fac_2))
    @inbounds ϕ_3_n[i,j,k] =ϕ_3_n[i,j,k]+dt*fac*((dϕ_3_dt[i,j,k])+(kt_ϕ_3[i,j,k])*(dt/fac_2))
    @inbounds ϕ_4_n[i,j,k] =ϕ_4_n[i,j,k]+dt*fac*((dϕ_4_dt[i,j,k])+(kt_ϕ_4[i,j,k])*(dt/fac_2))

    @inbounds W_1_2_n[i,j,k] =W_1_2_n[i,j,k]+dt*fac*((dW_1_2_dt[i,j,k])+(kt_W_1_2[i,j,k])*(dt/fac_2))
    @inbounds W_1_3_n[i,j,k] =W_1_3_n[i,j,k]+dt*fac*((dW_1_3_dt[i,j,k])+(kt_W_1_3[i,j,k])*(dt/fac_2))
    @inbounds W_1_4_n[i,j,k] =W_1_4_n[i,j,k]+dt*fac*((dW_1_4_dt[i,j,k])+(kt_W_1_4[i,j,k])*(dt/fac_2))

    @inbounds W_2_2_n[i,j,k] =W_2_2_n[i,j,k]+dt*fac*((dW_2_2_dt[i,j,k])+(kt_W_2_2[i,j,k])*(dt/fac_2))
    @inbounds W_2_3_n[i,j,k] =W_2_3_n[i,j,k]+dt*fac*((dW_2_3_dt[i,j,k])+(kt_W_2_3[i,j,k])*(dt/fac_2))
    @inbounds W_2_4_n[i,j,k] =W_2_4_n[i,j,k]+dt*fac*((dW_2_4_dt[i,j,k])+(kt_W_2_4[i,j,k])*(dt/fac_2))

    @inbounds W_3_2_n[i,j,k] =W_3_2_n[i,j,k]+dt*fac*((dW_3_2_dt[i,j,k])+(kt_W_3_2[i,j,k])*(dt/fac_2))
    @inbounds W_3_3_n[i,j,k] =W_3_3_n[i,j,k]+dt*fac*((dW_3_3_dt[i,j,k])+(kt_W_3_3[i,j,k])*(dt/fac_2))
    @inbounds W_3_4_n[i,j,k] =W_3_4_n[i,j,k]+dt*fac*((dW_3_4_dt[i,j,k])+(kt_W_3_4[i,j,k])*(dt/fac_2))

    @inbounds Y_2_n[i,j,k] =Y_2_n[i,j,k]+dt*fac*((dY_2_dt[i,j,k])+(kt_Y_2[i,j,k])*(dt/fac_2))

    @inbounds Y_3_n[i,j,k] =Y_3_n[i,j,k]+dt*fac*((dY_3_dt[i,j,k])+(kt_Y_3[i,j,k])*(dt/fac_2))

    @inbounds Y_4_n[i,j,k] =Y_4_n[i,j,k]+dt*fac*((dY_4_dt[i,j,k])+(kt_Y_4[i,j,k])*(dt/fac_2))

    return
end

@views function name_change!(ϕ_1_n,ϕ_2_n,ϕ_3_n,ϕ_4_n,
    W_1_2_n,W_1_3_n,W_1_4_n,
    W_2_2_n,W_2_3_n,W_2_4_n,
    W_3_2_n,W_3_3_n,W_3_4_n,
    Y_2_n,Y_3_n,Y_4_n,
    Γ_1_n,Γ_2_n,Γ_3_n,Σ_n,
    ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    Γ_1,Γ_2,Γ_3,Σ)

    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
    (blockIdx().y - 1) * blockDim().y + threadIdx().y,
    (blockIdx().z - 1) * blockDim().z + threadIdx().z
    
    @inbounds ϕ_1[i,j,k]=ϕ_1_n[i,j,k]
    @inbounds ϕ_2[i,j,k]=ϕ_2_n[i,j,k]
    @inbounds ϕ_3[i,j,k]=ϕ_3_n[i,j,k]
    @inbounds ϕ_4[i,j,k]=ϕ_4_n[i,j,k]
    @inbounds W_1_2[i,j,k]=W_1_2_n[i,j,k]
    @inbounds W_1_3[i,j,k]=W_1_3_n[i,j,k]
    @inbounds W_1_4[i,j,k]=W_1_4_n[i,j,k]
    @inbounds W_2_2[i,j,k]=W_2_2_n[i,j,k]
    @inbounds W_2_3[i,j,k]=W_2_3_n[i,j,k]
    @inbounds W_2_4[i,j,k]=W_2_4_n[i,j,k]
    @inbounds W_3_2[i,j,k]=W_3_2_n[i,j,k]
    @inbounds W_3_3[i,j,k]=W_3_3_n[i,j,k]
    @inbounds W_3_4[i,j,k]=W_3_4_n[i,j,k]
    @inbounds Y_2[i,j,k]=Y_2_n[i,j,k]
    @inbounds Y_3[i,j,k]=Y_3_n[i,j,k]
    @inbounds Y_4[i,j,k]=Y_4_n[i,j,k]
    @inbounds Γ_1[i,j,k]= Γ_1_n[i,j,k]
    @inbounds Γ_2[i,j,k]= Γ_2_n[i,j,k]
    @inbounds Γ_3[i,j,k]= Γ_3_n[i,j,k]
    @inbounds Σ[i,j,k]=Σ_n[i,j,k]

    return
end

@views function name_change_dot!(dϕ_1_dt_n,dϕ_2_dt_n,dϕ_3_dt_n,dϕ_4_dt_n,
    dW_1_2_dt_n,dW_1_3_dt_n,dW_1_4_dt_n,
    dW_2_2_dt_n,dW_2_3_dt_n,dW_2_4_dt_n,
    dW_3_2_dt_n,dW_3_3_dt_n,dW_3_4_dt_n,
    dY_2_dt_n,dY_3_dt_n,dY_4_dt_n,
    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt)

    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
    (blockIdx().y - 1) * blockDim().y + threadIdx().y,
    (blockIdx().z - 1) * blockDim().z + threadIdx().z
    
    @inbounds dϕ_1_dt[i,j,k]=dϕ_1_dt_n[i,j,k]
    @inbounds dϕ_2_dt[i,j,k]=dϕ_2_dt_n[i,j,k]
    @inbounds dϕ_3_dt[i,j,k]=dϕ_3_dt_n[i,j,k]
    @inbounds dϕ_4_dt[i,j,k]=dϕ_4_dt_n[i,j,k]
    @inbounds dW_1_2_dt[i,j,k]=dW_1_2_dt_n[i,j,k]
    @inbounds dW_1_3_dt[i,j,k]=dW_1_3_dt_n[i,j,k]
    @inbounds dW_1_4_dt[i,j,k]=dW_1_4_dt_n[i,j,k]
    @inbounds dW_2_2_dt[i,j,k]=dW_2_2_dt_n[i,j,k]
    @inbounds dW_2_3_dt[i,j,k]=dW_2_3_dt_n[i,j,k]
    @inbounds dW_2_4_dt[i,j,k]=dW_2_4_dt_n[i,j,k]
    @inbounds dW_3_2_dt[i,j,k]=dW_3_2_dt_n[i,j,k]
    @inbounds dW_3_3_dt[i,j,k]=dW_3_3_dt_n[i,j,k]
    @inbounds dW_3_4_dt[i,j,k]=dW_3_4_dt_n[i,j,k]
    @inbounds dY_2_dt[i,j,k]=dY_2_dt_n[i,j,k]
    @inbounds dY_3_dt[i,j,k]=dY_3_dt_n[i,j,k]
    @inbounds dY_4_dt[i,j,k]=dY_4_dt_n[i,j,k]
    
    return
end

function J_ext(i,dt,mH,t_sat;thermal_init=false)

    if ((i*dt*mH >= t_sat) && (thermal_init==true))
        J = mH^2
    else
        J = 0.0
    end
    return J
end

function run(;thermal_init=false)

    # Array initializations
    begin
        ϕ_1 = CUDA.zeros(prec,(Nx,Ny,Nz))
        ϕ_2 = CUDA.zeros(prec,(Nx,Ny,Nz))
        ϕ_3 = CUDA.zeros(prec,(Nx,Ny,Nz))
        ϕ_4 = CUDA.zeros(prec,(Nx,Ny,Nz))

        # W_1_1 = CUDA.zeros(prec,(Nx,Ny,Nz))
        W_1_2 = CUDA.zeros(prec,(Nx,Ny,Nz))
        W_1_3 = CUDA.zeros(prec,(Nx,Ny,Nz))
        W_1_4 = CUDA.zeros(prec,(Nx,Ny,Nz))
        # W_2_1 = CUDA.zeros(prec,(Nx,Ny,Nz))
        W_2_2 = CUDA.zeros(prec,(Nx,Ny,Nz))
        W_2_3 = CUDA.zeros(prec,(Nx,Ny,Nz))
        W_2_4 = CUDA.zeros(prec,(Nx,Ny,Nz))
        # W_3_1 = CUDA.zeros(prec,(Nx,Ny,Nz))
        W_3_2 = CUDA.zeros(prec,(Nx,Ny,Nz))
        W_3_3 = CUDA.zeros(prec,(Nx,Ny,Nz))
        W_3_4 = CUDA.zeros(prec,(Nx,Ny,Nz))
        # Y_1 = CUDA.zeros(prec,(Nx,Ny,Nz))
        Y_2 = CUDA.zeros(prec,(Nx,Ny,Nz))
        Y_3 = CUDA.zeros(prec,(Nx,Ny,Nz))
        Y_4 = CUDA.zeros(prec,(Nx,Ny,Nz))
        Γ_1 = CUDA.zeros(prec,(Nx,Ny,Nz))
        Γ_2 = CUDA.zeros(prec,(Nx,Ny,Nz))
        Γ_3 = CUDA.zeros(prec,(Nx,Ny,Nz))
        Σ = CUDA.zeros(prec,(Nx,Ny,Nz))

        # #Flux arrays
        k_Γ_1 = CUDA.zeros(prec,(Nx,Ny,Nz))
        k_Γ_2 = CUDA.zeros(prec,(Nx,Ny,Nz))
        k_Γ_3 = CUDA.zeros(prec,(Nx,Ny,Nz))
        k_Σ = CUDA.zeros(prec,(Nx,Ny,Nz))

        #Flux arrays
        kt_ϕ_1 = CUDA.zeros(prec,(Nx,Ny,Nz))
        kt_ϕ_2 = CUDA.zeros(prec,(Nx,Ny,Nz))
        kt_ϕ_3 = CUDA.zeros(prec,(Nx,Ny,Nz))
        kt_ϕ_4 = CUDA.zeros(prec,(Nx,Ny,Nz))

        # kt_W_1_1 = CUDA.zeros(prec,(Nx,Ny,Nz))
        kt_W_1_2 = CUDA.zeros(prec,(Nx,Ny,Nz))
        kt_W_1_3 = CUDA.zeros(prec,(Nx,Ny,Nz))
        kt_W_1_4 = CUDA.zeros(prec,(Nx,Ny,Nz))
        # kt_W_2_1 = CUDA.zeros(prec,(Nx,Ny,Nz))
        kt_W_2_2 = CUDA.zeros(prec,(Nx,Ny,Nz))
        kt_W_2_3 = CUDA.zeros(prec,(Nx,Ny,Nz))
        kt_W_2_4 = CUDA.zeros(prec,(Nx,Ny,Nz))
        # kt_W_3_1 = CUDA.zeros(prec,(Nx,Ny,Nz))
        kt_W_3_2 = CUDA.zeros(prec,(Nx,Ny,Nz))
        kt_W_3_3 = CUDA.zeros(prec,(Nx,Ny,Nz))
        kt_W_3_4 = CUDA.zeros(prec,(Nx,Ny,Nz))
        # kt_Y_1 = CUDA.zeros(prec,(Nx,Ny,Nz))
        kt_Y_2 = CUDA.zeros(prec,(Nx,Ny,Nz))
        kt_Y_3 = CUDA.zeros(prec,(Nx,Ny,Nz))
        kt_Y_4 = CUDA.zeros(prec,(Nx,Ny,Nz))

        #Updated field arrays
        ϕ_1_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        ϕ_2_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        ϕ_3_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        ϕ_4_n = CUDA.zeros(prec,(Nx,Ny,Nz))

        # W_1_1_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        W_1_2_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        W_1_3_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        W_1_4_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        # W_2_1_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        W_2_2_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        W_2_3_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        W_2_4_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        # W_3_1_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        W_3_2_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        W_3_3_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        W_3_4_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        # Y_1_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        Y_2_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        Y_3_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        Y_4_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        Γ_1_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        Γ_2_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        Γ_3_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        Σ_n = CUDA.zeros(prec,(Nx,Ny,Nz))

        dϕ_1_dt = CUDA.zeros(prec,(Nx,Ny,Nz))
        dϕ_2_dt = CUDA.zeros(prec,(Nx,Ny,Nz))
        dϕ_3_dt = CUDA.zeros(prec,(Nx,Ny,Nz))
        dϕ_4_dt = CUDA.zeros(prec,(Nx,Ny,Nz))
        #     dW_1_1_dt = CUDA.zeros(prec,(Nx,Ny,Nz))
        dW_1_2_dt = CUDA.zeros(prec,(Nx,Ny,Nz))
        dW_1_3_dt = CUDA.zeros(prec,(Nx,Ny,Nz))
        dW_1_4_dt = CUDA.zeros(prec,(Nx,Ny,Nz))
        #     dW_2_1_dt = CUDA.zeros(prec,(Nx,Ny,Nz))
        dW_2_2_dt = CUDA.zeros(prec,(Nx,Ny,Nz))
        dW_2_3_dt = CUDA.zeros(prec,(Nx,Ny,Nz))
        dW_2_4_dt = CUDA.zeros(prec,(Nx,Ny,Nz))
        #     dW_3_1_dt = CUDA.zeros(prec,(Nx,Ny,Nz))
        dW_3_2_dt = CUDA.zeros(prec,(Nx,Ny,Nz))
        dW_3_3_dt = CUDA.zeros(prec,(Nx,Ny,Nz))
        dW_3_4_dt = CUDA.zeros(prec,(Nx,Ny,Nz))
        #     dY_1_dt = CUDA.zeros(prec,(Nx,Ny,Nz))
        dY_2_dt = CUDA.zeros(prec,(Nx,Ny,Nz))
        dY_3_dt = CUDA.zeros(prec,(Nx,Ny,Nz))
        dY_4_dt = CUDA.zeros(prec,(Nx,Ny,Nz))

        dϕ_1_dt_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        dϕ_2_dt_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        dϕ_3_dt_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        dϕ_4_dt_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        #     dW_1_1_dt_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        dW_1_2_dt_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        dW_1_3_dt_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        dW_1_4_dt_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        #     dW_2_1_dt_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        dW_2_2_dt_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        dW_2_3_dt_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        dW_2_4_dt_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        #     dW_3_1_dt_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        dW_3_2_dt_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        dW_3_3_dt_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        dW_3_4_dt_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        #     dY_1_dt_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        dY_2_dt_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        dY_3_dt_n = CUDA.zeros(prec,(Nx,Ny,Nz))
        dY_4_dt_n = CUDA.zeros(prec,(Nx,Ny,Nz))


        #Temp field arrays

            # ϕ_1_t = CUDA.zeros(prec,(Nx,Ny,Nz))
            # ϕ_2_t = CUDA.zeros(prec,(Nx,Ny,Nz))
            # ϕ_3_t = CUDA.zeros(prec,(Nx,Ny,Nz))
            # ϕ_4_t = CUDA.zeros(prec,(Nx,Ny,Nz))

            # W_1_2_t = CUDA.zeros(prec,(Nx,Ny,Nz))
            # W_1_3_t = CUDA.zeros(prec,(Nx,Ny,Nz))
            # W_1_4_t = CUDA.zeros(prec,(Nx,Ny,Nz))

            # W_2_2_t = CUDA.zeros(prec,(Nx,Ny,Nz))
            # W_2_3_t = CUDA.zeros(prec,(Nx,Ny,Nz))
            # W_2_4_t = CUDA.zeros(prec,(Nx,Ny,Nz))

            # W_3_2_t = CUDA.zeros(prec,(Nx,Ny,Nz))
            # W_3_3_t = CUDA.zeros(prec,(Nx,Ny,Nz))
            # W_3_4_t = CUDA.zeros(prec,(Nx,Ny,Nz))

            # Y_2_t = CUDA.zeros(prec,(Nx,Ny,Nz))
            # Y_3_t = CUDA.zeros(prec,(Nx,Ny,Nz))
            # Y_4_t = CUDA.zeros(prec,(Nx,Ny,Nz))

            # Γ_1_t = CUDA.zeros(prec,(Nx,Ny,Nz))
            # Γ_2_t = CUDA.zeros(prec,(Nx,Ny,Nz))
            # Γ_3_t = CUDA.zeros(prec,(Nx,Ny,Nz))
            # Σ_t = CUDA.zeros(prec,(Nx,Ny,Nz))

            # dϕ_1_dt_t = CUDA.zeros(prec,(Nx,Ny,Nz))
            # dϕ_2_dt_t = CUDA.zeros(prec,(Nx,Ny,Nz))
            # dϕ_3_dt_t = CUDA.zeros(prec,(Nx,Ny,Nz))
            # dϕ_4_dt_t = CUDA.zeros(prec,(Nx,Ny,Nz))

            # dW_1_2_dt_t = CUDA.zeros(prec,(Nx,Ny,Nz))
            # dW_1_3_dt_t = CUDA.zeros(prec,(Nx,Ny,Nz))
            # dW_1_4_dt_t = CUDA.zeros(prec,(Nx,Ny,Nz))

            # dW_2_2_dt_t = CUDA.zeros(prec,(Nx,Ny,Nz))
            # dW_2_3_dt_t = CUDA.zeros(prec,(Nx,Ny,Nz))
            # dW_2_4_dt_t = CUDA.zeros(prec,(Nx,Ny,Nz))

            # dW_3_2_dt_t = CUDA.zeros(prec,(Nx,Ny,Nz))
            # dW_3_3_dt_t = CUDA.zeros(prec,(Nx,Ny,Nz))
            # dW_3_4_dt_t = CUDA.zeros(prec,(Nx,Ny,Nz))

            # dY_2_dt_t = CUDA.zeros(prec,(Nx,Ny,Nz))
            # dY_3_dt_t = CUDA.zeros(prec,(Nx,Ny,Nz))
            # dY_4_dt_t = CUDA.zeros(prec,(Nx,Ny,Nz))
        ###############
    end

    ##Energy arrays##
    begin
        KE_Phi = CUDA.zeros(prec,(Nx,Ny,Nz))
        GE_Phi = CUDA.zeros(prec,(Nx,Ny,Nz))
        PE_Phi = CUDA.zeros(prec,(Nx,Ny,Nz))
        ElectricE_W = CUDA.zeros(prec,(Nx,Ny,Nz))
        MagneticE_W = CUDA.zeros(prec,(Nx,Ny,Nz))
        ElectricE_Y = CUDA.zeros(prec,(Nx,Ny,Nz))
        MagneticE_Y = CUDA.zeros(prec,(Nx,Ny,Nz))
        total_energies = zeros((nsnaps+1,9))

        B_x = CUDA.zeros(prec,(Nx,Ny,Nz))
        B_y = CUDA.zeros(prec,(Nx,Ny,Nz))
        B_z = CUDA.zeros(prec,(Nx,Ny,Nz))

        B_x_2 = CUDA.zeros(prec,(Nx,Ny,Nz))
        B_y_2 = CUDA.zeros(prec,(Nx,Ny,Nz))
        B_z_2 = CUDA.zeros(prec,(Nx,Ny,Nz))

        spec_cut = [Nx÷4,Ny÷4,Nz÷4]
        N_bins = Kc_bin_nums(spec_cut[1],spec_cut[2],spec_cut[3])
        B_fft = zeros((nsnaps+1,N_bins,2))

        min_phi_arr = zeros((nsnaps+1))

    end
    CUDA.memory_status()

    ##########Configuring thread block grid###########

    thrds = (64,1,1)
    blks = (Nx÷thrds[1],Ny÷thrds[2],Nz÷thrds[3])
    println(string("#threads:",thrds," #blocks:",blks))
    
    rk4_kernel = @cuda launch=false rk4_step!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
        W_1_2,W_1_3,W_1_4,
        W_2_2,W_2_3,W_2_4,
        W_3_2,W_3_3,W_3_4,
        Y_2,Y_3,Y_4,
        Γ_1,Γ_2,Γ_3,Σ,
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
        gw,gy,gp2,vev,lambda,dx,0.0,γ)
    ##########

    # config = launch_configuration(rk4_kernel.fun)
    # thrds = (config.threads,1,1)
    # blks = (Nx÷thrds[1],Ny÷thrds[2],Nz÷thrds[3])
    
    ##########END Configuring thread block grid###########
    
    if thermal_init == true
    begin #Thermal initializing condition
        #Momentum space arrays
        # ϕ_1_k = zeros(prec,(Nx,Ny,Nz))
        # ϕ_2_k = zeros(prec,(Nx,Ny,Nz))
        # ϕ_3_k = zeros(prec,(Nx,Ny,Nz))
        # ϕ_4_k = zeros(prec,(Nx,Ny,Nz))
        seed_value = 123456

        iter = 5
        #Gauge fields#

        #W_1#
        begin
            Random.seed!(seed_value*(iter))
            r_1 = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1
            Random.seed!(seed_value*(iter))
            r_2 = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1
            Random.seed!(seed_value*(iter))
            r_3 = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1
            Random.seed!(seed_value*(iter))
            r_1_i = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1
            Random.seed!(seed_value*(iter))
            r_2_i = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1
            Random.seed!(seed_value*(iter))
            r_3_i = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1

            Random.seed!(seed_value*(iter))
            x_1 = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1
            Random.seed!(seed_value*(iter))
            x_2 = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1
            Random.seed!(seed_value*(iter))
            x_3 = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1
            Random.seed!(seed_value*(iter))
            x_4 = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1

            copyto!(ϕ_1,x_1)
            copyto!(ϕ_2,x_2)
            copyto!(ϕ_3,x_3)
            copyto!(ϕ_4,x_4)
            copyto!(W_1_2,r_1)
            copyto!(W_1_3,r_2)
            copyto!(W_1_4,r_3)
            copyto!(W_1_2_n,r_1_i)
            copyto!(W_1_3_n,r_2_i)
            copyto!(W_1_4_n,r_3_i)
            # println(Array(W_1_2)[10,10,10])
            @cuda threads=thrds blocks=blks thermal_initializer_gauge!(gw,gy,gp2,vev,dx,T,
                Nx,Ny,Nz,
                W_1_2,W_1_3,W_1_4,
                W_1_2_n,W_1_3_n,W_1_4_n,
                ϕ_1,ϕ_2,ϕ_3,ϕ_4)
            # println(Array(W_1_2)[10,10,10])
            W_1_2_k = real.(bfft(Array(W_1_2).+1im*Array(W_1_2_n)))#*sqrt(2.0)/(Nx*Ny*Nz*dx^3)
            copyto!(W_1_2,W_1_2_k)
            W_1_3_k = real.(bfft(Array(W_1_3).+1im*Array(W_1_3_n)))#*sqrt(2.0)/(Nx*Ny*Nz*dx^3)
            copyto!(W_1_3,W_1_3_k)
            W_1_4_k = real.(bfft(Array(W_1_4).+1im*Array(W_1_4_n)))#*sqrt(2.0)/(Nx*Ny*Nz*dx^3)
            copyto!(W_1_4,W_1_4_k)
            # println(Array(W_1_2)[10,10,10])
            # exit()
        end
        
        #W_2#
        begin
            Random.seed!(seed_value*(iter))
            r_1 = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1
            Random.seed!(seed_value*(iter))
            r_2 = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1
            Random.seed!(seed_value*(iter))
            r_3 = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1
            Random.seed!(seed_value*(iter))
            r_1_i = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1
            Random.seed!(seed_value*(iter))
            r_2_i = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1
            Random.seed!(seed_value*(iter))
            r_3_i = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1

            Random.seed!(seed_value*(iter))
            x_1 = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1
            Random.seed!(seed_value*(iter))
            x_2 = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1
            Random.seed!(seed_value*(iter))
            x_3 = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1
            Random.seed!(seed_value*(iter))
            x_4 = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1

            copyto!(ϕ_1,x_1)
            copyto!(ϕ_2,x_2)
            copyto!(ϕ_3,x_3)
            copyto!(ϕ_4,x_4)
            copyto!(W_2_2,r_1)
            copyto!(W_2_3,r_2)
            copyto!(W_2_4,r_3)
            copyto!(W_2_2_n,r_1_i)
            copyto!(W_2_3_n,r_2_i)
            copyto!(W_2_4_n,r_3_i)
            
            @cuda threads=thrds blocks=blks thermal_initializer_gauge!(gw,gy,gp2,vev,dx,T,
                Nx,Ny,Nz,
                W_2_2,W_2_3,W_2_4,
                W_2_2_n,W_2_3_n,W_2_4_n,
                ϕ_1,ϕ_2,ϕ_3,ϕ_4)
            
            W_2_2_k = real.(bfft(Array(W_2_2).+1im*Array(W_2_2_n)))#*sqrt(2.0)/(Nx*Ny*Nz*dx^3)
            copyto!(W_2_2,W_2_2_k)
            W_2_3_k = real.(bfft(Array(W_2_3).+1im*Array(W_2_3_n)))#*sqrt(2.0)/(Nx*Ny*Nz*dx^3)
            copyto!(W_2_3,W_2_3_k)
            W_2_4_k = real.(bfft(Array(W_2_4).+1im*Array(W_2_4_n)))#*sqrt(2.0)/(Nx*Ny*Nz*dx^3)
            copyto!(W_2_4,W_2_4_k)
        end

        #W_3#
        begin
            Random.seed!(seed_value*(iter))
            r_1 = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1
            Random.seed!(seed_value*(iter))
            r_2 = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1
            Random.seed!(seed_value*(iter))
            r_3 = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1
            Random.seed!(seed_value*(iter))
            r_1_i = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1
            Random.seed!(seed_value*(iter))
            r_2_i = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1
            Random.seed!(seed_value*(iter))
            r_3_i = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1

            Random.seed!(seed_value*(iter))
            x_1 = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1
            Random.seed!(seed_value*(iter))
            x_2 = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1
            Random.seed!(seed_value*(iter))
            x_3 = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1
            Random.seed!(seed_value*(iter))
            x_4 = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1

            copyto!(ϕ_1,x_1)
            copyto!(ϕ_2,x_2)
            copyto!(ϕ_3,x_3)
            copyto!(ϕ_4,x_4)
            copyto!(W_3_2,r_1)
            copyto!(W_3_3,r_2)
            copyto!(W_3_4,r_3)
            copyto!(W_3_2_n,r_1_i)
            copyto!(W_3_3_n,r_2_i)
            copyto!(W_3_4_n,r_3_i)
            
            @cuda threads=thrds blocks=blks thermal_initializer_gauge!(gw,gy,gp2,vev,dx,T,
                Nx,Ny,Nz,
                W_3_2,W_3_3,W_3_4,
                W_3_2_n,W_3_3_n,W_3_4_n,
                ϕ_1,ϕ_2,ϕ_3,ϕ_4)
            
            W_3_2_k = real.(bfft(Array(W_3_2).+1im*Array(W_3_2_n)))#*sqrt(2.0)/(Nx*Ny*Nz*dx^3)
            copyto!(W_3_2,W_3_2_k)
            W_3_3_k = real.(bfft(Array(W_3_3).+1im*Array(W_3_3_n)))#*sqrt(2.0)/(Nx*Ny*Nz*dx^3)
            copyto!(W_3_3,W_3_3_k)
            W_3_4_k = real.(bfft(Array(W_3_4).+1im*Array(W_3_4_n)))#*sqrt(2.0)/(Nx*Ny*Nz*dx^3)
            copyto!(W_3_4,W_3_4_k)
            
            # exit()
        end

        #Y#
        begin
            Random.seed!(seed_value*(iter))
            r_1 = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1
            Random.seed!(seed_value*(iter))
            r_2 = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1
            Random.seed!(seed_value*(iter))
            r_3 = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1
            Random.seed!(seed_value*(iter))
            r_1_i = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1
            Random.seed!(seed_value*(iter))
            r_2_i = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1
            Random.seed!(seed_value*(iter))
            r_3_i = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1

            Random.seed!(seed_value*(iter))
            x_1 = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1
            Random.seed!(seed_value*(iter))
            x_2 = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1
            Random.seed!(seed_value*(iter))
            x_3 = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1
            Random.seed!(seed_value*(iter))
            x_4 = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            iter=iter+1

            copyto!(ϕ_1,x_1)
            copyto!(ϕ_2,x_2)
            copyto!(ϕ_3,x_3)
            copyto!(ϕ_4,x_4)
            copyto!(Y_2,r_1)
            copyto!(Y_3,r_2)
            copyto!(Y_4,r_3)
            copyto!(Y_2_n,r_1_i)
            copyto!(Y_3_n,r_2_i)
            copyto!(Y_4_n,r_3_i)
            
            @cuda threads=thrds blocks=blks thermal_initializer_gauge!(gw,gy,gp2,vev,dx,T,
                Nx,Ny,Nz,
                Y_2,Y_3,Y_4,
                Y_2_n,Y_3_n,Y_4_n,
                ϕ_1,ϕ_2,ϕ_3,ϕ_4)

            Y_2_k = real.(bfft(Array(Y_2).+1im*Array(Y_2_n)))#*sqrt(2.0)/(Nx*Ny*Nz*dx^3)
            copyto!(Y_2,Y_2_k)
            Y_3_k = real.(bfft(Array(Y_3).+1im*Array(Y_3_n)))#*sqrt(2.0)/(Nx*Ny*Nz*dx^3)
            copyto!(Y_3,Y_3_k)
            Y_4_k = real.(bfft(Array(Y_4).+1im*Array(Y_4_n)))#*sqrt(2.0)/(Nx*Ny*Nz*dx^3)
            copyto!(Y_4,Y_4_k)
        end

        #ϕ#
        begin
            Random.seed!(seed_value)
            ϕ_1_k = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nx))
            Random.seed!(seed_value*2)
            ϕ_2_k = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            Random.seed!(seed_value*3)
            ϕ_3_k = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))
            Random.seed!(seed_value*4)
            ϕ_4_k = rand(Normal(0.0,sqrt(2.0)),(Nx,Ny,Nz))

            copyto!(ϕ_1,ϕ_1_k)
            copyto!(ϕ_2,ϕ_2_k)
            copyto!(ϕ_3,ϕ_3_k)
            copyto!(ϕ_4,ϕ_4_k)
            
            # @parallel (1:Nx,1:Ny,1:Nz) thermal_initializer!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,gw,gy,gp2,vev,dx,T,Nx,Ny,Nz,Nx_g,Ny_g,Nz_g,coords[1],coords[2],coords[3])
            @cuda threads=thrds blocks=blks thermal_initializer_ϕ!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
            gw,gy,gp2,vev,dx,T,meff_sq,
            Nx,Ny,Nz)

            ϕ_1_k = real.(bfft(Array(ϕ_1)))#*sqrt(2.0)/(Nx*Ny*Nz*dx^3)
            copyto!(ϕ_1,ϕ_1_k)
            ϕ_2_k = real.(bfft(Array(ϕ_2)))#*sqrt(2.0)/(Nx*Ny*Nz*dx^3)
            copyto!(ϕ_2,ϕ_2_k)
            ϕ_3_k = real.(bfft(Array(ϕ_3)))#*sqrt(2.0)/(Nx*Ny*Nz*dx^3)
            copyto!(ϕ_3,ϕ_3_k)
            ϕ_4_k = real.(bfft(Array(ϕ_4)))#*sqrt(2.0)/(Nx*Ny*Nz*dx^3)
            copyto!(ϕ_4,ϕ_4_k)
        end

    end
    else
        #Initializing random bubbles
        begin
            # Random.seed!(seed_value)
            # bubble_locs = rand(1:Nx,(no_bubbles,3))

            Xb_sample = range(bub_diam,Nx,step=bub_diam)
            Yb_sample = range(bub_diam,Ny,step=bub_diam)
            Zb_sample = range(bub_diam,Nz,step=bub_diam)

            Random.seed!(seed_value)
            xb_loc_idxs= rand(1:size(Xb_sample,1),no_bubbles)
            Random.seed!(seed_value*2)
            yb_loc_idxs= rand(1:size(Yb_sample,1),no_bubbles)
            Random.seed!(seed_value*3)
            zb_loc_idxs= rand(1:size(Zb_sample,1),no_bubbles)

            xb_locs = [Xb_sample[i] for i in xb_loc_idxs]
            yb_locs = [Yb_sample[i] for i in yb_loc_idxs]
            zb_locs = [Zb_sample[i] for i in zb_loc_idxs]

            #2 bubbles in y=0 plane#
            if (no_bubbles == 2)
                xb_locs = [Nx÷2-2*bub_diam,Nx÷2+2*bub_diam]
                yb_locs = [Ny÷2,Ny÷2]
                zb_locs = [Nz÷2,Nz÷2]
            end
            bubble_locs = int_prec.(hcat(xb_locs,yb_locs,zb_locs))

            println(string("bubble location matrix",bubble_locs))
            
            Random.seed!(seed_value)
            Hoft_arr = prec.(rand(Uniform(0,1),(no_bubbles,3)))

            bubs = []
            for bub_idx in range(1,no_bubbles)
                phi=gen_phi(Hoft_arr[bub_idx,:])
                ib,jb,kb = bubble_locs[bub_idx,:]
                # println(string(ib," ",jb," ",kb," ", phi))
                push!(bubs,[ib,jb,kb,real(phi[1]),imag(phi[1]),real(phi[2]),imag(phi[2])])
            end

            rkx=prec(pi/((Nx-1)*dx))
            rky=prec(pi/((Ny-1)*dx))
            rkz=prec(pi/((Nz-1)*dx))

            @time for b in range(1,size(bubs,1),step=1)
                ib,jb,kb,p1,p2,p3,p4 = bubs[b]
                # @cuda threads=thrds blocks=blks initializer!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,ib,jb,kb,p1,p2,p3,p4,rkx,rky,rkz,dx,mH)
                @cuda threads=thrds blocks=blks initializer!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,ib,jb,kb,p1,p2,p3,p4,rkx,rky,rkz,dx,mH)
                # synchronize()
            end
        end
    end
    # thermal_initializer!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,gw,gy,gp2,vev,dx,T,Nx,Ny,Nz)

    #compute energies and magnetic fields at initial time step
    @cuda threads=thrds blocks=blks compute_aux!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
        W_1_2,W_1_3,W_1_4,
        W_2_2,W_2_3,W_2_4,
        W_3_2,W_3_3,W_3_4,
        Y_2,Y_3,Y_4,
        dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
        dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
        dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
        dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
        dY_2_dt,dY_3_dt,dY_4_dt,
        KE_Phi,GE_Phi,PE_Phi,
        ElectricE_W,MagneticE_W,ElectricE_Y,MagneticE_Y,
        B_x,B_y,B_z,
        B_x_2,B_y_2,B_z_2,
        gw,gy,gp2,vev,lambda,θ_w,dx)
    
    # synchronize()
 
    min_phi_arr[1]=minimum(sqrt.(Array(ϕ_1).^2 .+ Array(ϕ_2).^2 .+ Array(ϕ_3).^2 .+ Array(ϕ_4).^2))

    # Compute fft and convolve spectrum
    @time begin
        B_x_fft = Array(fft(B_x))
        B_y_fft = Array(fft(B_y))
        B_z_fft = Array(fft(B_z))
        
        B_fft[1,:,:] = convolve_1d((real(conj.(B_x_fft).*B_x_fft.+
        conj.(B_y_fft).*B_y_fft.+
        conj.(B_z_fft).*B_z_fft)),Nx,Ny,Nz,spec_cut[1],spec_cut[2],spec_cut[3])
    end

    CUDA.memory_status()

    Vol = dx^3

    ##Sum energies##
    begin
        total_energies[1,1] = sum(Array(PE_Phi))*Vol
        total_energies[1,2] = sum(Array(KE_Phi))*Vol
        total_energies[1,3] = sum(Array(GE_Phi))*Vol
        total_energies[1,4] = sum(Array(ElectricE_W))*Vol
        total_energies[1,5] = sum(Array(MagneticE_W))*Vol
        total_energies[1,6] = sum(Array(ElectricE_Y))*Vol
        total_energies[1,7] = sum(Array(MagneticE_Y))*Vol
        total_energies[1,8] = sum(0.5*(Array(B_x).^2 .+Array(B_y).^2 .+Array(B_z).^2))*Vol
        total_energies[1,9] = sum(0.5*(Array(B_x_2).^2 .+Array(B_y_2).^2 .+Array(B_z_2).^2))*Vol
    end

    #Print energies##
    begin
        println(string("--------Energies--t:",0,"--process:----------"))
        println("Potentianl energy Higgs: ",total_energies[1,1])
        println("Kinetic energy Higgs: ",total_energies[1,2])
        println("Gradient energy Higgs:",total_energies[1,3])
        println("Magnetic energy W: ",total_energies[1,4])
        println("Electric energy W: ",total_energies[1,5])
        println("Magnetic energy Y: ",total_energies[1,6])
        println("Electric energy Y: ",total_energies[1,7])
        println("Total energy: ", sum(total_energies[1,1:end-2]))
        println("---------------------------------------")
    end

    ##PLOT##
    begin
        x=range(1,Nx,step=1)
        y=range(1,Ny,step=1)
        z=range(1,Nz,step=1)
        # println(size(x),size(y),size(z))

        gr()
        ENV["GKSwstype"]="nul"
        anim = Animation();
        # plot_1=contourf(z,x,(Array(ϕ_1)[:,Ny÷2,:]).^2+(Array(ϕ_2)[:,Ny÷2,:]).^2+(Array(ϕ_3)[:,Ny÷2,:]).^2+(Array(ϕ_4)[:,Ny÷2,:]).^2)

        # clim_1 = (minimum(Array(PE_Phi)[:,Ny÷2,:]),maximum(Array(PE_Phi)[:,Ny÷2,:]))
        # clim_2 = (minimum(Array(KE_Phi)[:,Ny÷2,:]),maximum(Array(KE_Phi)[:,Ny÷2,:]))
        # clim_3 = (minimum(Array(GE_Phi)[:,Ny÷2,:]),maximum(Array(GE_Phi)[:,Ny÷2,:]))

        clim_1 = (minimum(Array(PE_Phi)[:,Ny÷2,:]),maximum(Array(PE_Phi)[:,Ny÷2,:]))
        clim_2 = (minimum(Array(KE_Phi)[:,Ny÷2,:])/2.0,maximum(Array(KE_Phi)[:,Ny÷2,:])*2.0)
        clim_3 = (minimum(Array(GE_Phi)[:,Ny÷2,:])/2.0,maximum(Array(GE_Phi)[:,Ny÷2,:])*2.0)

        plot_1=heatmap(z,x,(Array(PE_Phi)[:,Ny÷2,:]),title="PE",clim=clim_1)
        # plot_2=heatmap(z,x,(Array(KE_Phi)[:,Ny÷2,:]),title="KE",clim=clim_2)
        plot_3=heatmap(z,x,(Array(GE_Phi)[:,Ny÷2,:]),title="GE",clim=clim_3)
        plot_4=contourf(z,x,(Array(ElectricE_W)[:,Ny÷2,:]+Array(MagneticE_W)[:,Ny÷2,:]),title="W E")
        plot_2=contourf(z,x,(Array(ElectricE_Y)[:,Ny÷2,:]+Array(MagneticE_Y)[:,Ny÷2,:]),title="Y E")

        # plot_4 = scatter([0],[total_energies[1,1] total_energies[1,2] total_energies[1,3] total_energies[1,8] sum(total_energies[1,1:end-2])],
        # label=["PE" "KE" "GE" "B" "Total"],xlims=(0,nte.*dt*mH))
        plot(plot_1,plot_2,plot_3,plot_4,layout=4,plot_title=string("it:",0),dpi=600)
        # plot(plot_1,title=string("it:",0),dpi=600)
        png(string("testini1",".png"))
        frame(anim)
    end
    # exit()

    ###END Initializing###

    #Counter for snaps
    snp_idx = 1
    @time for it in range(1,nte,step=1)
    CUDA.memory_status()

    # Jex = 0.0
    # Jex = mH^2

    Jex = J_ext(it,dt,mH,t_sat)

    println("--------",it,"--------------------")
    @time begin
        ##RK-1##
        # println("RK-1")
        @cuda threads=thrds blocks=blks rk4_step!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
            W_1_2,W_1_3,W_1_4,
            W_2_2,W_2_3,W_2_4,
            W_3_2,W_3_3,W_3_4,
            Y_2,Y_3,Y_4,
            Γ_1,Γ_2,Γ_3,Σ,
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
            gw,gy,gp2,vev,lambda,dx,Jex,γ)

        synchronize()

        # rk4_kernel(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
        #     W_1_2,W_1_3,W_1_4,
        #     W_2_2,W_2_3,W_2_4,
        #     W_3_2,W_3_3,W_3_4,
        #     Y_2,Y_3,Y_4,
        #     Γ_1,Γ_2,Γ_3,Σ,
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
        #     gw,gy,gp2,vev,lambda,dx,Jex,γ; threads=thrds, blocks=blks)

        # # synchronize()
        # exit()
        #Update new field arrays with first term k_1
        begin
            ϕ_1_n = dϕ_1_dt .*(dt/6.) + ϕ_1
            ϕ_2_n = dϕ_2_dt .*(dt/6.) + ϕ_2
            ϕ_3_n = dϕ_3_dt .*(dt/6.) + ϕ_3
            ϕ_4_n = dϕ_4_dt .*(dt/6.) + ϕ_4
            W_1_2_n = dW_1_2_dt .*(dt/6.) + W_1_2
            W_1_3_n = dW_1_3_dt .*(dt/6.) + W_1_3
            W_1_4_n = dW_1_4_dt .*(dt/6.) + W_1_4
            W_2_2_n = dW_2_2_dt .*(dt/6.) + W_2_2
            W_2_3_n = dW_2_3_dt .*(dt/6.) + W_2_3
            W_2_4_n = dW_2_4_dt .*(dt/6.) + W_2_4
            W_3_2_n = dW_3_2_dt .*(dt/6.) + W_3_2
            W_3_3_n = dW_3_3_dt .*(dt/6.) + W_3_3
            W_3_4_n = dW_3_4_dt .*(dt/6.) + W_3_4
            Y_2_n = dY_2_dt .*(dt/6.) + Y_2
            Y_3_n = dY_3_dt .*(dt/6.) + Y_3
            Y_4_n = dY_4_dt .*(dt/6.) + Y_4
            Γ_1_n = k_Γ_1 .*(dt/6.) + Γ_1
            Γ_2_n = k_Γ_2 .*(dt/6.) + Γ_2
            Γ_3_n = k_Γ_3 .*(dt/6.) + Γ_3
            Σ_n = k_Σ .*(dt/6.) + Σ
            # CUDA.memory_status()
            # # synchronize()
            dϕ_1_dt_n = kt_ϕ_1 .*(dt/6.) + dϕ_1_dt
            dϕ_2_dt_n = kt_ϕ_2 .*(dt/6.) + dϕ_2_dt
            dϕ_3_dt_n = kt_ϕ_3 .*(dt/6.) + dϕ_3_dt
            dϕ_4_dt_n = kt_ϕ_4 .*(dt/6.) + dϕ_4_dt
            dW_1_2_dt_n = kt_W_1_2 .*(dt/6.) + dW_1_2_dt
            dW_1_3_dt_n = kt_W_1_3 .*(dt/6.) + dW_1_3_dt
            dW_1_4_dt_n = kt_W_1_4 .*(dt/6.) + dW_1_4_dt
            dW_2_2_dt_n = kt_W_2_2 .*(dt/6.) + dW_2_2_dt
            dW_2_3_dt_n = kt_W_2_3 .*(dt/6.) + dW_2_3_dt
            dW_2_4_dt_n = kt_W_2_4 .*(dt/6.) + dW_2_4_dt
            dW_3_2_dt_n = kt_W_3_2 .*(dt/6.) + dW_3_2_dt
            dW_3_3_dt_n = kt_W_3_3 .*(dt/6.) + dW_3_3_dt
            dW_3_4_dt_n = kt_W_3_4 .*(dt/6.) + dW_3_4_dt
            dY_2_dt_n = kt_Y_2 .*(dt/6.) + dY_2_dt
            dY_3_dt_n = kt_Y_3 .*(dt/6.) + dY_3_dt
            dY_4_dt_n = kt_Y_4 .*(dt/6.) + dY_4_dt
        end

        # println("updates")
        # @cuda threads=thrds blocks=blks updater_1!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
        #     W_1_2,W_1_3,W_1_4,
        #     W_2_2,W_2_3,W_2_4,
        #     W_3_2,W_3_3,W_3_4,
        #     Y_2,Y_3,Y_4,
        #     Γ_1,Γ_2,Γ_3,Σ,
        #     ϕ_1_n,ϕ_2_n,ϕ_3_n,ϕ_4_n,
        #     W_1_2_n,W_1_3_n,W_1_4_n,
        #     W_2_2_n,W_2_3_n,W_2_4_n,
        #     W_3_2_n,W_3_3_n,W_3_4_n,
        #     Y_2_n,Y_3_n,Y_4_n,
        #     Γ_1_n,Γ_2_n,Γ_3_n,Σ_n,
        #     dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
        #     dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
        #     dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
        #     dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
        #     dY_2_dt,dY_3_dt,dY_4_dt,
        #     k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,dt,rk4_coeffs[1])
        # ####
        # synchronize()
        # @cuda threads=thrds blocks=blks updater_t_1!(dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
        #     dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
        #     dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
        #     dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
        #     dY_2_dt,dY_3_dt,dY_4_dt,
        #     dϕ_1_dt_n,dϕ_2_dt_n,dϕ_3_dt_n,dϕ_4_dt_n,
        #     dW_1_2_dt_n,dW_1_3_dt_n,dW_1_4_dt_n,
        #     dW_2_2_dt_n,dW_2_3_dt_n,dW_2_4_dt_n,
        #     dW_3_2_dt_n,dW_3_3_dt_n,dW_3_4_dt_n,
        #     dY_2_dt_n,dY_3_dt_n,dY_4_dt_n,
        #     kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
        #     kt_W_1_2,kt_W_1_3,kt_W_1_4,
        #     kt_W_2_2,kt_W_2_3,kt_W_2_4,
        #     kt_W_3_2,kt_W_3_3,kt_W_3_4,
        #     kt_Y_2,kt_Y_3,kt_Y_4,dt,rk4_coeffs[1])
        # synchronize()
        
        ##RK-2##
        
        #Update triv arrays directly with k-1 fluxes#
        begin
            ϕ_1_n = (dϕ_1_dt+kt_ϕ_1 .*(dt/2.)).*(dt/3.) + ϕ_1_n
            ϕ_2_n = (dϕ_2_dt+kt_ϕ_2 .*(dt/2.)).*(dt/3.) + ϕ_2_n
            ϕ_3_n = (dϕ_3_dt+kt_ϕ_3 .*(dt/2.)).*(dt/3.) + ϕ_3_n
            ϕ_4_n = (dϕ_4_dt+kt_ϕ_4 .*(dt/2.)).*(dt/3.) + ϕ_4_n
            W_1_2_n = (dW_1_2_dt+kt_W_1_2 .*(dt/2.)).*(dt/3.) + W_1_2_n
            W_1_3_n = (dW_1_3_dt+kt_W_1_3 .*(dt/2.)).*(dt/3.) + W_1_3_n
            W_1_4_n = (dW_1_4_dt+kt_W_1_4 .*(dt/2.)).*(dt/3.) + W_1_4_n
            W_2_2_n = (dW_2_2_dt+kt_W_2_2 .*(dt/2.)).*(dt/3.) + W_2_2_n
            W_2_3_n = (dW_2_3_dt+kt_W_2_3 .*(dt/2.)).*(dt/3.) + W_2_3_n
            W_2_4_n = (dW_2_4_dt+kt_W_2_4 .*(dt/2.)).*(dt/3.) + W_2_4_n
            W_3_2_n = (dW_3_2_dt+kt_W_3_2 .*(dt/2.)).*(dt/3.) + W_3_2_n
            W_3_3_n = (dW_3_3_dt+kt_W_3_3 .*(dt/2.)).*(dt/3.) + W_3_3_n
            W_3_4_n = (dW_3_4_dt+kt_W_3_4 .*(dt/2.)).*(dt/3.) + W_3_4_n
            Y_2_n = (dY_2_dt+kt_Y_2 .*(dt/2.)).*(dt/3.) + Y_2_n
            Y_3_n = (dY_3_dt+kt_Y_3 .*(dt/2.)).*(dt/3.) + Y_3_n
            Y_4_n = (dY_4_dt+kt_Y_4 .*(dt/2.)).*(dt/3.) + Y_4_n
        end
        # println("trivial updates")
        # @cuda threads=thrds blocks=blks triv_updater!(ϕ_1_n,ϕ_2_n,ϕ_3_n,ϕ_4_n,
        #     W_1_2_n,W_1_3_n,W_1_4_n,
        #     W_2_2_n,W_2_3_n,W_2_4_n,
        #     W_3_2_n,W_3_3_n,W_3_4_n,
        #     Y_2_n,Y_3_n,Y_4_n,
        #     kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
        #     kt_W_1_2,kt_W_1_3,kt_W_1_4,
        #     kt_W_2_2,kt_W_2_3,kt_W_2_4,
        #     kt_W_3_2,kt_W_3_3,kt_W_3_4,
        #     kt_Y_2,kt_Y_3,kt_Y_4,
        #     dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
        #     dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
        #     dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
        #     dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
        #     dY_2_dt,dY_3_dt,dY_4_dt,
        #     dt,rk4_coeffs[2],2.0)
        # synchronize()
        #Compute k_2 arrays
        # println("RK-2")
        
        @cuda threads=thrds blocks=blks rk4_step!(ϕ_1+dϕ_1_dt .*(dt/2.),ϕ_2+dϕ_2_dt .*(dt/2.),ϕ_3+dϕ_3_dt .*(dt/2.),ϕ_4+dϕ_4_dt .*(dt/2.),
            W_1_2+dW_1_2_dt .*(dt/2.),W_1_3+dW_1_3_dt .*(dt/2.),W_1_4+dW_1_4_dt .*(dt/2.),
            W_2_2+dW_2_2_dt .*(dt/2.),W_2_3+dW_2_3_dt .*(dt/2.),W_2_4+dW_2_4_dt .*(dt/2.),
            W_3_2+dW_3_2_dt .*(dt/2.),W_3_3+dW_3_3_dt .*(dt/2.),W_3_4+dW_3_4_dt .*(dt/2.),
            Y_2+dY_2_dt .*(dt/2.),Y_3+dY_3_dt .*(dt/2.),Y_4+dY_4_dt .*(dt/2.),
            Γ_1+k_Γ_1 .*(dt/2.),Γ_2+k_Γ_2 .*(dt/2.),Γ_3+k_Γ_3 .*(dt/2.),Σ+k_Σ .*(dt/2.),
            k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,
            dϕ_1_dt+kt_ϕ_1 .*(dt/2.),dϕ_2_dt+kt_ϕ_2 .*(dt/2.),dϕ_3_dt+kt_ϕ_3 .*(dt/2.),dϕ_4_dt+kt_ϕ_4 .*(dt/2.),
            dW_1_2_dt+kt_W_1_2 .*(dt/2.),dW_1_3_dt+kt_W_1_3 .*(dt/2.),dW_1_4_dt+kt_W_1_4 .*(dt/2.),
            dW_2_2_dt+kt_W_2_2 .*(dt/2.),dW_2_3_dt+kt_W_2_3 .*(dt/2.),dW_2_4_dt+kt_W_2_4 .*(dt/2.),
            dW_3_2_dt+kt_W_3_2 .*(dt/2.),dW_3_3_dt+kt_W_3_3 .*(dt/2.),dW_3_4_dt+kt_W_3_4 .*(dt/2.),
            dY_2_dt+kt_Y_2 .*(dt/2.),dY_3_dt+kt_Y_3 .*(dt/2.),dY_4_dt+kt_Y_4 .*(dt/2.),
            kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
            kt_W_1_2,kt_W_1_3,kt_W_1_4,
            kt_W_2_2,kt_W_2_3,kt_W_2_4,
            kt_W_3_2,kt_W_3_3,kt_W_3_4,
            kt_Y_2,kt_Y_3,kt_Y_4,
            gw,gy,gp2,vev,lambda,dx,Jex,γ)
        synchronize()

        # rk4_kernel(ϕ_1+dϕ_1_dt .*(dt/2.),ϕ_2+dϕ_2_dt .*(dt/2.),ϕ_3+dϕ_3_dt .*(dt/2.),ϕ_4+dϕ_4_dt .*(dt/2.),
        #     W_1_2+dW_1_2_dt .*(dt/2.),W_1_3+dW_1_3_dt .*(dt/2.),W_1_4+dW_1_4_dt .*(dt/2.),
        #     W_2_2+dW_2_2_dt .*(dt/2.),W_2_3+dW_2_3_dt .*(dt/2.),W_2_4+dW_2_4_dt .*(dt/2.),
        #     W_3_2+dW_3_2_dt .*(dt/2.),W_3_3+dW_3_3_dt .*(dt/2.),W_3_4+dW_3_4_dt .*(dt/2.),
        #     Y_2+dY_2_dt .*(dt/2.),Y_3+dY_3_dt .*(dt/2.),Y_4+dY_4_dt .*(dt/2.),
        #     Γ_1+k_Γ_1 .*(dt/2.),Γ_2+k_Γ_2 .*(dt/2.),Γ_3+k_Γ_3 .*(dt/2.),Σ+k_Σ .*(dt/2.),
        #     k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,
        #     dϕ_1_dt+kt_ϕ_1 .*(dt/2.),dϕ_2_dt+kt_ϕ_2 .*(dt/2.),dϕ_3_dt+kt_ϕ_3 .*(dt/2.),dϕ_4_dt+kt_ϕ_4 .*(dt/2.),
        #     dW_1_2_dt+kt_W_1_2 .*(dt/2.),dW_1_3_dt+kt_W_1_3 .*(dt/2.),dW_1_4_dt+kt_W_1_4 .*(dt/2.),
        #     dW_2_2_dt+kt_W_2_2 .*(dt/2.),dW_2_3_dt+kt_W_2_3 .*(dt/2.),dW_2_4_dt+kt_W_2_4 .*(dt/2.),
        #     dW_3_2_dt+kt_W_3_2 .*(dt/2.),dW_3_3_dt+kt_W_3_3 .*(dt/2.),dW_3_4_dt+kt_W_3_4 .*(dt/2.),
        #     dY_2_dt+kt_Y_2 .*(dt/2.),dY_3_dt+kt_Y_3 .*(dt/2.),dY_4_dt+kt_Y_4 .*(dt/2.),
        #     kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
        #     kt_W_1_2,kt_W_1_3,kt_W_1_4,
        #     kt_W_2_2,kt_W_2_3,kt_W_2_4,
        #     kt_W_3_2,kt_W_3_3,kt_W_3_4,
        #     kt_Y_2,kt_Y_3,kt_Y_4,
        #     gw,gy,gp2,vev,lambda,dx,Jex,γ;threads=thrds,blocks=blks)
        # # synchronize()

        ##
        # @cuda threads=thrds blocks=blks temp_updater_2!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
        #     W_1_2,W_1_3,W_1_4,
        #     W_2_2,W_2_3,W_2_4,
        #     W_3_2,W_3_3,W_3_4,
        #     Y_2,Y_3,Y_4,
        #     Γ_1,Γ_2,Γ_3,Σ,
        #     ϕ_1_t,ϕ_2_t,ϕ_3_t,ϕ_4_t,
        #     W_1_2_t,W_1_3_t,W_1_4_t,
        #     W_2_2_t,W_2_3_t,W_2_4_t,
        #     W_3_2_t,W_3_3_t,W_3_4_t,
        #     Y_2_t,Y_3_t,Y_4_t,
        #     Γ_1_t,Γ_2_t,Γ_3_t,Σ_t,
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
        #     k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,dt)
        # ########
        # @cuda threads=thrds blocks=blks temp_updater_t_2!(dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
        #     dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
        #     dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
        #     dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
        #     dY_2_dt,dY_3_dt,dY_4_dt,
        #     dϕ_1_dt_t,dϕ_2_dt_t,dϕ_3_dt_t,dϕ_4_dt_t,
        #     dW_1_2_dt_t,dW_1_3_dt_t,dW_1_4_dt_t,
        #     dW_2_2_dt_t,dW_2_3_dt_t,dW_2_4_dt_t,
        #     dW_3_2_dt_t,dW_3_3_dt_t,dW_3_4_dt_t,
        #     dY_2_dt_t,dY_3_dt_t,dY_4_dt_t,
        #     kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
        #     kt_W_1_2,kt_W_1_3,kt_W_1_4,
        #     kt_W_2_2,kt_W_2_3,kt_W_2_4,
        #     kt_W_3_2,kt_W_3_3,kt_W_3_4,
        #     kt_Y_2,kt_Y_3,kt_Y_4,dt)
        # ################
        # @cuda threads=thrds blocks=blks rk4_step!(ϕ_1_t,ϕ_2_t,ϕ_3_t,ϕ_4_t,
        #     W_1_2_t,W_1_3_t,W_1_4_t,
        #     W_2_2_t,W_2_3_t,W_2_4_t,
        #     W_3_2_t,W_3_3_t,W_3_4_t,
        #     Y_2_t,Y_3_t,Y_4_t,
        #     Γ_1_t,Γ_2_t,Γ_3_t,Σ_t,
        #     k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,
        #     dϕ_1_dt_t,dϕ_2_dt_t,dϕ_3_dt_t,dϕ_4_dt_t,
        #     dW_1_2_dt_t,dW_1_3_dt_t,dW_1_4_dt_t,
        #     dW_2_2_dt_t,dW_2_3_dt_t,dW_2_4_dt_t,
        #     dW_3_2_dt_t,dW_3_3_dt_t,dW_3_4_dt_t,
        #     dY_2_dt_t,dY_3_dt_t,dY_4_dt_t,
        #     kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
        #     kt_W_1_2,kt_W_1_3,kt_W_1_4,
        #     kt_W_2_2,kt_W_2_3,kt_W_2_4,
        #     kt_W_3_2,kt_W_3_3,kt_W_3_4,
        #     kt_Y_2,kt_Y_3,kt_Y_4,
        #     gw,gy,gp2,vev,lambda,dx,Jex,γ)
        #############
        # synchronize()

        #Update new field arrays with second term k_2

        begin
            Γ_1_n = k_Γ_1 .*(dt/3.) + Γ_1_n
            Γ_2_n = k_Γ_2 .*(dt/3.) + Γ_2_n
            Γ_3_n = k_Γ_3 .*(dt/3.) + Γ_3_n
            Σ_n = k_Σ .*(dt/3.) + Σ_n
            dϕ_1_dt_n = kt_ϕ_1 .*(dt/3.) + dϕ_1_dt_n
            dϕ_2_dt_n = kt_ϕ_2 .*(dt/3.) + dϕ_2_dt_n
            dϕ_3_dt_n = kt_ϕ_3 .*(dt/3.) + dϕ_3_dt_n
            dϕ_4_dt_n = kt_ϕ_4 .*(dt/3.) + dϕ_4_dt_n
            dW_1_2_dt_n = kt_W_1_2 .*(dt/3.) + dW_1_2_dt_n
            dW_1_3_dt_n = kt_W_1_3 .*(dt/3.) + dW_1_3_dt_n
            dW_1_4_dt_n = kt_W_1_4 .*(dt/3.) + dW_1_4_dt_n
            dW_2_2_dt_n = kt_W_2_2 .*(dt/3.) + dW_2_2_dt_n
            dW_2_3_dt_n = kt_W_2_3 .*(dt/3.) + dW_2_3_dt_n
            dW_2_4_dt_n = kt_W_2_4 .*(dt/3.) + dW_2_4_dt_n
            dW_3_2_dt_n = kt_W_3_2 .*(dt/3.) + dW_3_2_dt_n
            dW_3_3_dt_n = kt_W_3_3 .*(dt/3.) + dW_3_3_dt_n
            dW_3_4_dt_n = kt_W_3_4 .*(dt/3.) + dW_3_4_dt_n
            dY_2_dt_n = kt_Y_2 .*(dt/3.) + dY_2_dt_n
            dY_3_dt_n = kt_Y_3 .*(dt/3.) + dY_3_dt_n
            dY_4_dt_n = kt_Y_4 .*(dt/3.) + dY_4_dt_n
        end
        # println("updates")
        # @cuda threads=thrds blocks=blks updater_t!(dϕ_1_dt_n,dϕ_2_dt_n,dϕ_3_dt_n,dϕ_4_dt_n,
        #     dW_1_2_dt_n,dW_1_3_dt_n,dW_1_4_dt_n,
        #     dW_2_2_dt_n,dW_2_3_dt_n,dW_2_4_dt_n,
        #     dW_3_2_dt_n,dW_3_3_dt_n,dW_3_4_dt_n,
        #     dY_2_dt_n,dY_3_dt_n,dY_4_dt_n,
        #     kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
        #     kt_W_1_2,kt_W_1_3,kt_W_1_4,
        #     kt_W_2_2,kt_W_2_3,kt_W_2_4,
        #     kt_W_3_2,kt_W_3_3,kt_W_3_4,
        #     kt_Y_2,kt_Y_3,kt_Y_4,
        #     Γ_1_n,Γ_2_n,Γ_3_n,Σ_n,
        #     k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,dt,rk4_coeffs[2])
        # synchronize()
        #######

        ##RK-3##

        #Update triv arrays directly with k-2 fluxes#
        begin
            ϕ_1_n = (dϕ_1_dt+kt_ϕ_1 .*(dt/2.)).*(dt/3.) + ϕ_1_n
            ϕ_2_n = (dϕ_2_dt+kt_ϕ_2 .*(dt/2.)).*(dt/3.) + ϕ_2_n
            ϕ_3_n = (dϕ_3_dt+kt_ϕ_3 .*(dt/2.)).*(dt/3.) + ϕ_3_n
            ϕ_4_n = (dϕ_4_dt+kt_ϕ_4 .*(dt/2.)).*(dt/3.) + ϕ_4_n
            W_1_2_n = (dW_1_2_dt+kt_W_1_2 .*(dt/2.)).*(dt/3.) + W_1_2_n
            W_1_3_n = (dW_1_3_dt+kt_W_1_3 .*(dt/2.)).*(dt/3.) + W_1_3_n
            W_1_4_n = (dW_1_4_dt+kt_W_1_4 .*(dt/2.)).*(dt/3.) + W_1_4_n
            W_2_2_n = (dW_2_2_dt+kt_W_2_2 .*(dt/2.)).*(dt/3.) + W_2_2_n
            W_2_3_n = (dW_2_3_dt+kt_W_2_3 .*(dt/2.)).*(dt/3.) + W_2_3_n
            W_2_4_n = (dW_2_4_dt+kt_W_2_4 .*(dt/2.)).*(dt/3.) + W_2_4_n
            W_3_2_n = (dW_3_2_dt+kt_W_3_2 .*(dt/2.)).*(dt/3.) + W_3_2_n
            W_3_3_n = (dW_3_3_dt+kt_W_3_3 .*(dt/2.)).*(dt/3.) + W_3_3_n
            W_3_4_n = (dW_3_4_dt+kt_W_3_4 .*(dt/2.)).*(dt/3.) + W_3_4_n
            Y_2_n = (dY_2_dt+kt_Y_2 .*(dt/2.)).*(dt/3.) + Y_2_n
            Y_3_n = (dY_3_dt+kt_Y_3 .*(dt/2.)).*(dt/3.) + Y_3_n
            Y_4_n = (dY_4_dt+kt_Y_4 .*(dt/2.)).*(dt/3.) + Y_4_n
        end
        #Compute k_3 arrays
        # @cuda threads=thrds blocks=blks rk4_step!(ϕ_1+dϕ_1_dt .*(dt/2.),ϕ_2+dϕ_2_dt .*(dt/2.),ϕ_3+dϕ_3_dt .*(dt/2.),ϕ_4+dϕ_4_dt .*(dt/2.),
        #     W_1_2+dW_1_2_dt .*(dt/2.),W_1_3+dW_1_3_dt .*(dt/2.),W_1_4+dW_1_4_dt .*(dt/2.),
        #     W_2_2+dW_2_2_dt .*(dt/2.),W_2_3+dW_2_3_dt .*(dt/2.),W_2_4+dW_2_4_dt .*(dt/2.),
        #     W_3_2+dW_3_2_dt .*(dt/2.),W_3_3+dW_3_3_dt .*(dt/2.),W_3_4+dW_3_4_dt .*(dt/2.),
        #     Y_2+dY_2_dt .*(dt/2.),Y_3+dY_3_dt .*(dt/2.),Y_4+dY_4_dt .*(dt/2.),
        #     Γ_1+k_Γ_1 .*(dt/2.),Γ_2+k_Γ_2 .*(dt/2.),Γ_3+k_Γ_3 .*(dt/2.),Σ+k_Σ .*(dt/2.),
        #     k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,
        #     dϕ_1_dt+kt_ϕ_1 .*(dt/2.),dϕ_2_dt+kt_ϕ_2 .*(dt/2.),dϕ_3_dt+kt_ϕ_3 .*(dt/2.),dϕ_4_dt+kt_ϕ_4 .*(dt/2.),
        #     dW_1_2_dt+kt_W_1_2 .*(dt/2.),dW_1_3_dt+kt_W_1_3 .*(dt/2.),dW_1_4_dt+kt_W_1_4 .*(dt/2.),
        #     dW_2_2_dt+kt_W_2_2 .*(dt/2.),dW_2_3_dt+kt_W_2_3 .*(dt/2.),dW_2_4_dt+kt_W_2_4 .*(dt/2.),
        #     dW_3_2_dt+kt_W_3_2 .*(dt/2.),dW_3_3_dt+kt_W_3_3 .*(dt/2.),dW_3_4_dt+kt_W_3_4 .*(dt/2.),
        #     dY_2_dt+kt_Y_2 .*(dt/2.),dY_3_dt+kt_Y_3 .*(dt/2.),dY_4_dt+kt_Y_4 .*(dt/2.),
        #     kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
        #     kt_W_1_2,kt_W_1_3,kt_W_1_4,
        #     kt_W_2_2,kt_W_2_3,kt_W_2_4,
        #     kt_W_3_2,kt_W_3_3,kt_W_3_4,
        #     kt_Y_2,kt_Y_3,kt_Y_4,
        #     gw,gy,gp2,vev,lambda,dx)

        # synchronize()
        
        ##Passing f_{n-1} + 0.5*dt*k_2, 
        ## where k_2 = dfdt_{n-1}+dfdt_k2*0.5*dt
        # println("triv-updates")
        
        # @cuda threads=thrds blocks=blks triv_updater!(ϕ_1_n,ϕ_2_n,ϕ_3_n,ϕ_4_n,
        #     W_1_2_n,W_1_3_n,W_1_4_n,
        #     W_2_2_n,W_2_3_n,W_2_4_n,
        #     W_3_2_n,W_3_3_n,W_3_4_n,
        #     Y_2_n,Y_3_n,Y_4_n,
        #     kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
        #     kt_W_1_2,kt_W_1_3,kt_W_1_4,
        #     kt_W_2_2,kt_W_2_3,kt_W_2_4,
        #     kt_W_3_2,kt_W_3_3,kt_W_3_4,
        #     kt_Y_2,kt_Y_3,kt_Y_4,
        #     dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
        #     dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
        #     dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
        #     dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
        #     dY_2_dt,dY_3_dt,dY_4_dt,
        #     dt,rk4_coeffs[3],2.0)
        # synchronize()        
        ####
        # println("RK-3")

        @cuda threads=thrds blocks=blks rk4_step!(ϕ_1+(dϕ_1_dt+kt_ϕ_1 .*(dt/2.)) .*(dt/2.),
            ϕ_2+(dϕ_2_dt+kt_ϕ_2 .*(dt/2.)) .*(dt/2.),
            ϕ_3+(dϕ_3_dt+kt_ϕ_3 .*(dt/2.)) .*(dt/2.),
            ϕ_4+(dϕ_4_dt+kt_ϕ_4 .*(dt/2.)) .*(dt/2.),
            W_1_2+(dW_1_2_dt+kt_W_1_2 .*(dt/2.)) .*(dt/2.),
            W_1_3+(dW_1_3_dt+kt_W_1_3 .*(dt/2.)) .*(dt/2.),
            W_1_4+(dW_1_4_dt+kt_W_1_4 .*(dt/2.)) .*(dt/2.),
            W_2_2+(dW_2_2_dt+kt_W_2_2 .*(dt/2.)) .*(dt/2.),
            W_2_3+(dW_2_3_dt+kt_W_2_3 .*(dt/2.)) .*(dt/2.),
            W_2_4+(dW_2_4_dt+kt_W_2_4 .*(dt/2.)) .*(dt/2.),
            W_3_2+(dW_3_2_dt+kt_W_3_2 .*(dt/2.)) .*(dt/2.),
            W_3_3+(dW_3_3_dt+kt_W_3_3 .*(dt/2.)) .*(dt/2.),
            W_3_4+(dW_3_4_dt+kt_W_3_4 .*(dt/2.)) .*(dt/2.),
            Y_2+(dY_2_dt+kt_Y_2 .*(dt/2.)) .*(dt/2.),
            Y_3+(dY_3_dt+kt_Y_3 .*(dt/2.)) .*(dt/2.),
            Y_4+(dY_4_dt+kt_Y_4 .*(dt/2.)) .*(dt/2.),
            Γ_1+k_Γ_1 .*(dt/2.),Γ_2+k_Γ_2 .*(dt/2.),Γ_3+k_Γ_3 .*(dt/2.),Σ+k_Σ .*(dt/2.),
            k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,
            dϕ_1_dt+kt_ϕ_1 .*(dt/2.),dϕ_2_dt+kt_ϕ_2 .*(dt/2.),dϕ_3_dt+kt_ϕ_3 .*(dt/2.),dϕ_4_dt+kt_ϕ_4 .*(dt/2.),
            dW_1_2_dt+kt_W_1_2 .*(dt/2.),dW_1_3_dt+kt_W_1_3 .*(dt/2.),dW_1_4_dt+kt_W_1_4 .*(dt/2.),
            dW_2_2_dt+kt_W_2_2 .*(dt/2.),dW_2_3_dt+kt_W_2_3 .*(dt/2.),dW_2_4_dt+kt_W_2_4 .*(dt/2.),
            dW_3_2_dt+kt_W_3_2 .*(dt/2.),dW_3_3_dt+kt_W_3_3 .*(dt/2.),dW_3_4_dt+kt_W_3_4 .*(dt/2.),
            dY_2_dt+kt_Y_2 .*(dt/2.),dY_3_dt+kt_Y_3 .*(dt/2.),dY_4_dt+kt_Y_4 .*(dt/2.),
            kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
            kt_W_1_2,kt_W_1_3,kt_W_1_4,
            kt_W_2_2,kt_W_2_3,kt_W_2_4,
            kt_W_3_2,kt_W_3_3,kt_W_3_4,
            kt_Y_2,kt_Y_3,kt_Y_4,
            gw,gy,gp2,vev,lambda,dx,Jex,γ)

        synchronize()
        # rk4_kernel(ϕ_1+(dϕ_1_dt+kt_ϕ_1 .*(dt/2.)) .*(dt/2.),
        #     ϕ_2+(dϕ_2_dt+kt_ϕ_2 .*(dt/2.)) .*(dt/2.),
        #     ϕ_3+(dϕ_3_dt+kt_ϕ_3 .*(dt/2.)) .*(dt/2.),
        #     ϕ_4+(dϕ_4_dt+kt_ϕ_4 .*(dt/2.)) .*(dt/2.),
        #     W_1_2+(dW_1_2_dt+kt_W_1_2 .*(dt/2.)) .*(dt/2.),
        #     W_1_3+(dW_1_3_dt+kt_W_1_3 .*(dt/2.)) .*(dt/2.),
        #     W_1_4+(dW_1_4_dt+kt_W_1_4 .*(dt/2.)) .*(dt/2.),
        #     W_2_2+(dW_2_2_dt+kt_W_2_2 .*(dt/2.)) .*(dt/2.),
        #     W_2_3+(dW_2_3_dt+kt_W_2_3 .*(dt/2.)) .*(dt/2.),
        #     W_2_4+(dW_2_4_dt+kt_W_2_4 .*(dt/2.)) .*(dt/2.),
        #     W_3_2+(dW_3_2_dt+kt_W_3_2 .*(dt/2.)) .*(dt/2.),
        #     W_3_3+(dW_3_3_dt+kt_W_3_3 .*(dt/2.)) .*(dt/2.),
        #     W_3_4+(dW_3_4_dt+kt_W_3_4 .*(dt/2.)) .*(dt/2.),
        #     Y_2+(dY_2_dt+kt_Y_2 .*(dt/2.)) .*(dt/2.),
        #     Y_3+(dY_3_dt+kt_Y_3 .*(dt/2.)) .*(dt/2.),
        #     Y_4+(dY_4_dt+kt_Y_4 .*(dt/2.)) .*(dt/2.),
        #     Γ_1+k_Γ_1 .*(dt/2.),Γ_2+k_Γ_2 .*(dt/2.),Γ_3+k_Γ_3 .*(dt/2.),Σ+k_Σ .*(dt/2.),
        #     k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,
        #     dϕ_1_dt+kt_ϕ_1 .*(dt/2.),dϕ_2_dt+kt_ϕ_2 .*(dt/2.),dϕ_3_dt+kt_ϕ_3 .*(dt/2.),dϕ_4_dt+kt_ϕ_4 .*(dt/2.),
        #     dW_1_2_dt+kt_W_1_2 .*(dt/2.),dW_1_3_dt+kt_W_1_3 .*(dt/2.),dW_1_4_dt+kt_W_1_4 .*(dt/2.),
        #     dW_2_2_dt+kt_W_2_2 .*(dt/2.),dW_2_3_dt+kt_W_2_3 .*(dt/2.),dW_2_4_dt+kt_W_2_4 .*(dt/2.),
        #     dW_3_2_dt+kt_W_3_2 .*(dt/2.),dW_3_3_dt+kt_W_3_3 .*(dt/2.),dW_3_4_dt+kt_W_3_4 .*(dt/2.),
        #     dY_2_dt+kt_Y_2 .*(dt/2.),dY_3_dt+kt_Y_3 .*(dt/2.),dY_4_dt+kt_Y_4 .*(dt/2.),
        #     kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
        #     kt_W_1_2,kt_W_1_3,kt_W_1_4,
        #     kt_W_2_2,kt_W_2_3,kt_W_2_4,
        #     kt_W_3_2,kt_W_3_3,kt_W_3_4,
        #     kt_Y_2,kt_Y_3,kt_Y_4,
        #     gw,gy,gp2,vev,lambda,dx,Jex,γ;threads=thrds,blocks=blks)

        # # synchronize()
        # @cuda threads=thrds blocks=blks temp_updater_3!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
        #     W_1_2,W_1_3,W_1_4,
        #     W_2_2,W_2_3,W_2_4,
        #     W_3_2,W_3_3,W_3_4,
        #     Y_2,Y_3,Y_4,
        #     Γ_1,Γ_2,Γ_3,Σ,
        #     ϕ_1_t,ϕ_2_t,ϕ_3_t,ϕ_4_t,
        #     W_1_2_t,W_1_3_t,W_1_4_t,
        #     W_2_2_t,W_2_3_t,W_2_4_t,
        #     W_3_2_t,W_3_3_t,W_3_4_t,
        #     Y_2_t,Y_3_t,Y_4_t,
        #     Γ_1_t,Γ_2_t,Γ_3_t,Σ_t,
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
        #     k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,dt)
        # #######
        # @cuda threads=thrds blocks=blks temp_updater_t_3!(dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
        #     dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
        #     dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
        #     dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
        #     dY_2_dt,dY_3_dt,dY_4_dt,
        #     dϕ_1_dt_t,dϕ_2_dt_t,dϕ_3_dt_t,dϕ_4_dt_t,
        #     dW_1_2_dt_t,dW_1_3_dt_t,dW_1_4_dt_t,
        #     dW_2_2_dt_t,dW_2_3_dt_t,dW_2_4_dt_t,
        #     dW_3_2_dt_t,dW_3_3_dt_t,dW_3_4_dt_t,
        #     dY_2_dt_t,dY_3_dt_t,dY_4_dt_t,
        #     kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
        #     kt_W_1_2,kt_W_1_3,kt_W_1_4,
        #     kt_W_2_2,kt_W_2_3,kt_W_2_4,
        #     kt_W_3_2,kt_W_3_3,kt_W_3_4,
        #     kt_Y_2,kt_Y_3,kt_Y_4,
        #     k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,dt)
        # #######
        # @cuda threads=thrds blocks=blks rk4_step!(ϕ_1_t,ϕ_2_t,ϕ_3_t,ϕ_4_t,
        #     W_1_2_t,W_1_3_t,W_1_4_t,
        #     W_2_2_t,W_2_3_t,W_2_4_t,
        #     W_3_2_t,W_3_3_t,W_3_4_t,
        #     Y_2_t,Y_3_t,Y_4_t,
        #     Γ_1_t,Γ_2_t,Γ_3_t,Σ_t,
        #     k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,
        #     dϕ_1_dt_t,dϕ_2_dt_t,dϕ_3_dt_t,dϕ_4_dt_t,
        #     dW_1_2_dt_t,dW_1_3_dt_t,dW_1_4_dt_t,
        #     dW_2_2_dt_t,dW_2_3_dt_t,dW_2_4_dt_t,
        #     dW_3_2_dt_t,dW_3_3_dt_t,dW_3_4_dt_t,
        #     dY_2_dt_t,dY_3_dt_t,dY_4_dt_t,
        #     kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
        #     kt_W_1_2,kt_W_1_3,kt_W_1_4,
        #     kt_W_2_2,kt_W_2_3,kt_W_2_4,
        #     kt_W_3_2,kt_W_3_3,kt_W_3_4,
        #     kt_Y_2,kt_Y_3,kt_Y_4,
        #     gw,gy,gp2,vev,lambda,dx,Jex,γ)

        # #Update new field arrays with third term k_3
        begin
            Γ_1_n = k_Γ_1 .*(dt/3.) + Γ_1_n
            Γ_2_n = k_Γ_2 .*(dt/3.) + Γ_2_n
            Γ_3_n = k_Γ_3 .*(dt/3.) + Γ_3_n
            Σ_n = k_Σ .*(dt/3.) + Σ_n

            dϕ_1_dt_n = kt_ϕ_1 .*(dt/3.) + dϕ_1_dt_n
            dϕ_2_dt_n = kt_ϕ_2 .*(dt/3.) + dϕ_2_dt_n
            dϕ_3_dt_n = kt_ϕ_3 .*(dt/3.) + dϕ_3_dt_n
            dϕ_4_dt_n = kt_ϕ_4 .*(dt/3.) + dϕ_4_dt_n
            dW_1_2_dt_n = kt_W_1_2 .*(dt/3.) + dW_1_2_dt_n
            dW_1_3_dt_n = kt_W_1_3 .*(dt/3.) + dW_1_3_dt_n
            dW_1_4_dt_n = kt_W_1_4 .*(dt/3.) + dW_1_4_dt_n
            dW_2_2_dt_n = kt_W_2_2 .*(dt/3.) + dW_2_2_dt_n
            dW_2_3_dt_n = kt_W_2_3 .*(dt/3.) + dW_2_3_dt_n
            dW_2_4_dt_n = kt_W_2_4 .*(dt/3.) + dW_2_4_dt_n
            dW_3_2_dt_n = kt_W_3_2 .*(dt/3.) + dW_3_2_dt_n
            dW_3_3_dt_n = kt_W_3_3 .*(dt/3.) + dW_3_3_dt_n
            dW_3_4_dt_n = kt_W_3_4 .*(dt/3.) + dW_3_4_dt_n
            dY_2_dt_n = kt_Y_2 .*(dt/3.) + dY_2_dt_n
            dY_3_dt_n = kt_Y_3 .*(dt/3.) + dY_3_dt_n
            dY_4_dt_n = kt_Y_4 .*(dt/3.) + dY_4_dt_n
        end
        # println("updates")

        # @cuda threads=thrds blocks=blks updater_t!(dϕ_1_dt_n,dϕ_2_dt_n,dϕ_3_dt_n,dϕ_4_dt_n,
        #     dW_1_2_dt_n,dW_1_3_dt_n,dW_1_4_dt_n,
        #     dW_2_2_dt_n,dW_2_3_dt_n,dW_2_4_dt_n,
        #     dW_3_2_dt_n,dW_3_3_dt_n,dW_3_4_dt_n,
        #     dY_2_dt_n,dY_3_dt_n,dY_4_dt_n,
        #     kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
        #     kt_W_1_2,kt_W_1_3,kt_W_1_4,
        #     kt_W_2_2,kt_W_2_3,kt_W_2_4,
        #     kt_W_3_2,kt_W_3_3,kt_W_3_4,
        #     kt_Y_2,kt_Y_3,kt_Y_4,
        #     Γ_1_n,Γ_2_n,Γ_3_n,Σ_n,
        #     k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,dt,rk4_coeffs[3])
        # synchronize()
        
        #######
        
        ##RK-4##

        #Update triv arrays directly with k-3 fluxes#
        begin
            ϕ_1_n = (dϕ_1_dt+kt_ϕ_1 .*dt).*(dt/3.) + ϕ_1_n
            ϕ_2_n = (dϕ_2_dt+kt_ϕ_2 .*dt).*(dt/3.) + ϕ_2_n
            ϕ_3_n = (dϕ_3_dt+kt_ϕ_3 .*dt).*(dt/3.) + ϕ_3_n
            ϕ_4_n = (dϕ_4_dt+kt_ϕ_4 .*dt).*(dt/3.) + ϕ_4_n
            W_1_2_n = (dW_1_2_dt+kt_W_1_2 .*dt).*(dt/3.) + W_1_2_n
            W_1_3_n = (dW_1_3_dt+kt_W_1_3 .*dt).*(dt/3.) + W_1_3_n
            W_1_4_n = (dW_1_4_dt+kt_W_1_4 .*dt).*(dt/3.) + W_1_4_n
            W_2_2_n = (dW_2_2_dt+kt_W_2_2 .*dt).*(dt/3.) + W_2_2_n
            W_2_3_n = (dW_2_3_dt+kt_W_2_3 .*dt).*(dt/3.) + W_2_3_n
            W_2_4_n = (dW_2_4_dt+kt_W_2_4 .*dt).*(dt/3.) + W_2_4_n
            W_3_2_n = (dW_3_2_dt+kt_W_3_2 .*dt).*(dt/3.) + W_3_2_n
            W_3_3_n = (dW_3_3_dt+kt_W_3_3 .*dt).*(dt/3.) + W_3_3_n
            W_3_4_n = (dW_3_4_dt+kt_W_3_4 .*dt).*(dt/3.) + W_3_4_n
            Y_2_n = (dY_2_dt+kt_Y_2 .*dt).*(dt/3.) + Y_2_n
            Y_3_n = (dY_3_dt+kt_Y_3 .*dt).*(dt/3.) + Y_3_n
            Y_4_n = (dY_4_dt+kt_Y_4 .*dt).*(dt/3.) + Y_4_n
        end
        # println("triv-updates")
        
        # @cuda threads=thrds blocks=blks triv_updater!(ϕ_1_n,ϕ_2_n,ϕ_3_n,ϕ_4_n,
        #     W_1_2_n,W_1_3_n,W_1_4_n,
        #     W_2_2_n,W_2_3_n,W_2_4_n,
        #     W_3_2_n,W_3_3_n,W_3_4_n,
        #     Y_2_n,Y_3_n,Y_4_n,
        #     kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
        #     kt_W_1_2,kt_W_1_3,kt_W_1_4,
        #     kt_W_2_2,kt_W_2_3,kt_W_2_4,
        #     kt_W_3_2,kt_W_3_3,kt_W_3_4,
        #     kt_Y_2,kt_Y_3,kt_Y_4,
        #     dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
        #     dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
        #     dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
        #     dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
        #     dY_2_dt,dY_3_dt,dY_4_dt,
        #     dt,rk4_coeffs[4],1.0)
        # synchronize()
        
        #Compute k_4 arrays
        # @cuda threads=thrds blocks=blks rk4_step!(ϕ_1+dϕ_1_dt .*dt,ϕ_2+dϕ_2_dt .*dt,ϕ_3+dϕ_3_dt .*dt,ϕ_4+dϕ_4_dt .*dt,
        #     W_1_2+dW_1_2_dt .*dt,W_1_3+dW_1_3_dt .*dt,W_1_4+dW_1_4_dt .*dt,
        #     W_2_2+dW_2_2_dt .*dt,W_2_3+dW_2_3_dt .*dt,W_2_4+dW_2_4_dt .*dt,
        #     W_3_2+dW_3_2_dt .*dt,W_3_3+dW_3_3_dt .*dt,W_3_4+dW_3_4_dt .*dt,
        #     Y_2+dY_2_dt .*dt,Y_3+dY_3_dt .*dt,Y_4+dY_4_dt .*dt,
        #     Γ_1+k_Γ_1 .*dt,Γ_2+k_Γ_2 .*dt,Γ_3+k_Γ_3 .*dt,Σ+k_Σ .*dt,
        #     k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,
        #     dϕ_1_dt+kt_ϕ_1 .*dt,dϕ_2_dt+kt_ϕ_2 .*dt,dϕ_3_dt+kt_ϕ_3 .*dt,dϕ_4_dt+kt_ϕ_4 .*dt,
        #     dW_1_2_dt+kt_W_1_2 .*dt,dW_1_3_dt+kt_W_1_3 .*dt,dW_1_4_dt+kt_W_1_4 .*dt,
        #     dW_2_2_dt+kt_W_2_2 .*dt,dW_2_3_dt+kt_W_2_3 .*dt,dW_2_4_dt+kt_W_2_4 .*dt,
        #     dW_3_2_dt+kt_W_3_2 .*dt,dW_3_3_dt+kt_W_3_3 .*dt,dW_3_4_dt+kt_W_3_4 .*dt,
        #     dY_2_dt+kt_Y_2 .*dt,dY_3_dt+kt_Y_3 .*dt,dY_4_dt+kt_Y_4 .*dt,
        #     kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
        #     kt_W_1_2,kt_W_1_3,kt_W_1_4,
        #     kt_W_2_2,kt_W_2_3,kt_W_2_4,
        #     kt_W_3_2,kt_W_3_3,kt_W_3_4,
        #     kt_Y_2,kt_Y_3,kt_Y_4,
        #     gw,gy,gp2,vev,lambda,dx)

        # # synchronize()
        
        # println("RK-4")
        # @cuda threads=thrds blocks=blks temp_updater_4!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
        #     W_1_2,W_1_3,W_1_4,
        #     W_2_2,W_2_3,W_2_4,
        #     W_3_2,W_3_3,W_3_4,
        #     Y_2,Y_3,Y_4,
        #     Γ_1,Γ_2,Γ_3,Σ,
        #     ϕ_1_t,ϕ_2_t,ϕ_3_t,ϕ_4_t,
        #     W_1_2_t,W_1_3_t,W_1_4_t,
        #     W_2_2_t,W_2_3_t,W_2_4_t,
        #     W_3_2_t,W_3_3_t,W_3_4_t,
        #     Y_2_t,Y_3_t,Y_4_t,
        #     Γ_1_t,Γ_2_t,Γ_3_t,Σ_t,
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
        #     k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,dt)
        # ######
        # @cuda threads=thrds blocks=blks temp_updater_t_4!(dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
        #     dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
        #     dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
        #     dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
        #     dY_2_dt,dY_3_dt,dY_4_dt,
        #     dϕ_1_dt_t,dϕ_2_dt_t,dϕ_3_dt_t,dϕ_4_dt_t,
        #     dW_1_2_dt_t,dW_1_3_dt_t,dW_1_4_dt_t,
        #     dW_2_2_dt_t,dW_2_3_dt_t,dW_2_4_dt_t,
        #     dW_3_2_dt_t,dW_3_3_dt_t,dW_3_4_dt_t,
        #     dY_2_dt_t,dY_3_dt_t,dY_4_dt_t,
        #     kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
        #     kt_W_1_2,kt_W_1_3,kt_W_1_4,
        #     kt_W_2_2,kt_W_2_3,kt_W_2_4,
        #     kt_W_3_2,kt_W_3_3,kt_W_3_4,
        #     kt_Y_2,kt_Y_3,kt_Y_4,
        #     k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,dt)
        # ######
        # @cuda threads=thrds blocks=blks rk4_step!(ϕ_1_t,ϕ_2_t,ϕ_3_t,ϕ_4_t,
        #     W_1_2_t,W_1_3_t,W_1_4_t,
        #     W_2_2_t,W_2_3_t,W_2_4_t,
        #     W_3_2_t,W_3_3_t,W_3_4_t,
        #     Y_2_t,Y_3_t,Y_4_t,
        #     Γ_1_t,Γ_2_t,Γ_3_t,Σ_t,
        #     k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,
        #     dϕ_1_dt_t,dϕ_2_dt_t,dϕ_3_dt_t,dϕ_4_dt_t,
        #     dW_1_2_dt_t,dW_1_3_dt_t,dW_1_4_dt_t,
        #     dW_2_2_dt_t,dW_2_3_dt_t,dW_2_4_dt_t,
        #     dW_3_2_dt_t,dW_3_3_dt_t,dW_3_4_dt_t,
        #     dY_2_dt_t,dY_3_dt_t,dY_4_dt_t,
        #     kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
        #     kt_W_1_2,kt_W_1_3,kt_W_1_4,
        #     kt_W_2_2,kt_W_2_3,kt_W_2_4,
        #     kt_W_3_2,kt_W_3_3,kt_W_3_4,
        #     kt_Y_2,kt_Y_3,kt_Y_4,
        #     gw,gy,gp2,vev,lambda,dx,Jex,γ)
        # #####
        @cuda threads=thrds blocks=blks rk4_step!(ϕ_1+(dϕ_1_dt+kt_ϕ_1 .*(dt/2.)) .*(dt),
            ϕ_2+(dϕ_2_dt+kt_ϕ_2 .*(dt/2.)) .*(dt),
            ϕ_3+(dϕ_3_dt+kt_ϕ_3 .*(dt/2.)) .*(dt),
            ϕ_4+(dϕ_4_dt+kt_ϕ_4 .*(dt/2.)) .*(dt),
            W_1_2+(dW_1_2_dt+kt_W_1_2 .*(dt/2.)) .*(dt),
            W_1_3+(dW_1_3_dt+kt_W_1_3 .*(dt/2.)) .*(dt),
            W_1_4+(dW_1_4_dt+kt_W_1_4 .*(dt/2.)) .*(dt),
            W_2_2+(dW_2_2_dt+kt_W_2_2 .*(dt/2.)) .*(dt),
            W_2_3+(dW_2_3_dt+kt_W_2_3 .*(dt/2.)) .*(dt),
            W_2_4+(dW_2_4_dt+kt_W_2_4 .*(dt/2.)) .*(dt),
            W_3_2+(dW_3_2_dt+kt_W_3_2 .*(dt/2.)) .*(dt),
            W_3_3+(dW_3_3_dt+kt_W_3_3 .*(dt/2.)) .*(dt),
            W_3_4+(dW_3_4_dt+kt_W_3_4 .*(dt/2.)) .*(dt),
            Y_2+(dY_2_dt+kt_Y_2 .*(dt/2.)) .*(dt),
            Y_3+(dY_3_dt+kt_Y_3 .*(dt/2.)) .*(dt),
            Y_4+(dY_4_dt+kt_Y_4 .*(dt/2.)) .*(dt),
            Γ_1+k_Γ_1 .*dt,Γ_2+k_Γ_2 .*dt,Γ_3+k_Γ_3 .*dt,Σ+k_Σ .*dt,
            k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,
            dϕ_1_dt+kt_ϕ_1 .*dt,dϕ_2_dt+kt_ϕ_2 .*dt,dϕ_3_dt+kt_ϕ_3 .*dt,dϕ_4_dt+kt_ϕ_4 .*dt,
            dW_1_2_dt+kt_W_1_2 .*dt,dW_1_3_dt+kt_W_1_3 .*dt,dW_1_4_dt+kt_W_1_4 .*dt,
            dW_2_2_dt+kt_W_2_2 .*dt,dW_2_3_dt+kt_W_2_3 .*dt,dW_2_4_dt+kt_W_2_4 .*dt,
            dW_3_2_dt+kt_W_3_2 .*dt,dW_3_3_dt+kt_W_3_3 .*dt,dW_3_4_dt+kt_W_3_4 .*dt,
            dY_2_dt+kt_Y_2 .*dt,dY_3_dt+kt_Y_3 .*dt,dY_4_dt+kt_Y_4 .*dt,
            kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
            kt_W_1_2,kt_W_1_3,kt_W_1_4,
            kt_W_2_2,kt_W_2_3,kt_W_2_4,
            kt_W_3_2,kt_W_3_3,kt_W_3_4,
            kt_Y_2,kt_Y_3,kt_Y_4,
            gw,gy,gp2,vev,lambda,dx,Jex,γ)

        synchronize()
        # rk4_kernel(ϕ_1+(dϕ_1_dt+kt_ϕ_1 .*(dt/2.)) .*(dt),
        #     ϕ_2+(dϕ_2_dt+kt_ϕ_2 .*(dt/2.)) .*(dt),
        #     ϕ_3+(dϕ_3_dt+kt_ϕ_3 .*(dt/2.)) .*(dt),
        #     ϕ_4+(dϕ_4_dt+kt_ϕ_4 .*(dt/2.)) .*(dt),
        #     W_1_2+(dW_1_2_dt+kt_W_1_2 .*(dt/2.)) .*(dt),
        #     W_1_3+(dW_1_3_dt+kt_W_1_3 .*(dt/2.)) .*(dt),
        #     W_1_4+(dW_1_4_dt+kt_W_1_4 .*(dt/2.)) .*(dt),
        #     W_2_2+(dW_2_2_dt+kt_W_2_2 .*(dt/2.)) .*(dt),
        #     W_2_3+(dW_2_3_dt+kt_W_2_3 .*(dt/2.)) .*(dt),
        #     W_2_4+(dW_2_4_dt+kt_W_2_4 .*(dt/2.)) .*(dt),
        #     W_3_2+(dW_3_2_dt+kt_W_3_2 .*(dt/2.)) .*(dt),
        #     W_3_3+(dW_3_3_dt+kt_W_3_3 .*(dt/2.)) .*(dt),
        #     W_3_4+(dW_3_4_dt+kt_W_3_4 .*(dt/2.)) .*(dt),
        #     Y_2+(dY_2_dt+kt_Y_2 .*(dt/2.)) .*(dt),
        #     Y_3+(dY_3_dt+kt_Y_3 .*(dt/2.)) .*(dt),
        #     Y_4+(dY_4_dt+kt_Y_4 .*(dt/2.)) .*(dt),
        #     Γ_1+k_Γ_1 .*dt,Γ_2+k_Γ_2 .*dt,Γ_3+k_Γ_3 .*dt,Σ+k_Σ .*dt,
        #     k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,
        #     dϕ_1_dt+kt_ϕ_1 .*dt,dϕ_2_dt+kt_ϕ_2 .*dt,dϕ_3_dt+kt_ϕ_3 .*dt,dϕ_4_dt+kt_ϕ_4 .*dt,
        #     dW_1_2_dt+kt_W_1_2 .*dt,dW_1_3_dt+kt_W_1_3 .*dt,dW_1_4_dt+kt_W_1_4 .*dt,
        #     dW_2_2_dt+kt_W_2_2 .*dt,dW_2_3_dt+kt_W_2_3 .*dt,dW_2_4_dt+kt_W_2_4 .*dt,
        #     dW_3_2_dt+kt_W_3_2 .*dt,dW_3_3_dt+kt_W_3_3 .*dt,dW_3_4_dt+kt_W_3_4 .*dt,
        #     dY_2_dt+kt_Y_2 .*dt,dY_3_dt+kt_Y_3 .*dt,dY_4_dt+kt_Y_4 .*dt,
        #     kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
        #     kt_W_1_2,kt_W_1_3,kt_W_1_4,
        #     kt_W_2_2,kt_W_2_3,kt_W_2_4,
        #     kt_W_3_2,kt_W_3_3,kt_W_3_4,
        #     kt_Y_2,kt_Y_3,kt_Y_4,
        #     gw,gy,gp2,vev,lambda,dx,Jex,γ;threads=thrds,blocks=blks)

        # # synchronize()
        #Update new field arrays with first term k_4
        begin
            Γ_1_n = k_Γ_1 .*(dt/6.) + Γ_1_n
            Γ_2_n = k_Γ_2 .*(dt/6.) + Γ_2_n
            Γ_3_n = k_Γ_3 .*(dt/6.) + Γ_3_n
            Σ_n = k_Σ .*(dt/6.) + Σ_n

            dϕ_1_dt_n = kt_ϕ_1 .*(dt/6.) + dϕ_1_dt_n
            dϕ_2_dt_n = kt_ϕ_2 .*(dt/6.) + dϕ_2_dt_n
            dϕ_3_dt_n = kt_ϕ_3 .*(dt/6.) + dϕ_3_dt_n
            dϕ_4_dt_n = kt_ϕ_4 .*(dt/6.) + dϕ_4_dt_n
            dW_1_2_dt_n = kt_W_1_2 .*(dt/6.) + dW_1_2_dt_n
            dW_1_3_dt_n = kt_W_1_3 .*(dt/6.) + dW_1_3_dt_n
            dW_1_4_dt_n = kt_W_1_4 .*(dt/6.) + dW_1_4_dt_n
            dW_2_2_dt_n = kt_W_2_2 .*(dt/6.) + dW_2_2_dt_n
            dW_2_3_dt_n = kt_W_2_3 .*(dt/6.) + dW_2_3_dt_n
            dW_2_4_dt_n = kt_W_2_4 .*(dt/6.) + dW_2_4_dt_n
            dW_3_2_dt_n = kt_W_3_2 .*(dt/6.) + dW_3_2_dt_n
            dW_3_3_dt_n = kt_W_3_3 .*(dt/6.) + dW_3_3_dt_n
            dW_3_4_dt_n = kt_W_3_4 .*(dt/6.) + dW_3_4_dt_n
            dY_2_dt_n = kt_Y_2 .*(dt/6.) + dY_2_dt_n
            dY_3_dt_n = kt_Y_3 .*(dt/6.) + dY_3_dt_n
            dY_4_dt_n = kt_Y_4 .*(dt/6.) + dY_4_dt_n
        end
        # println("updates")
        # @cuda threads=thrds blocks=blks updater_t!(dϕ_1_dt_n,dϕ_2_dt_n,dϕ_3_dt_n,dϕ_4_dt_n,
        #     dW_1_2_dt_n,dW_1_3_dt_n,dW_1_4_dt_n,
        #     dW_2_2_dt_n,dW_2_3_dt_n,dW_2_4_dt_n,
        #     dW_3_2_dt_n,dW_3_3_dt_n,dW_3_4_dt_n,
        #     dY_2_dt_n,dY_3_dt_n,dY_4_dt_n,
        #     kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
        #     kt_W_1_2,kt_W_1_3,kt_W_1_4,
        #     kt_W_2_2,kt_W_2_3,kt_W_2_4,
        #     kt_W_3_2,kt_W_3_3,kt_W_3_4,
        #     kt_Y_2,kt_Y_3,kt_Y_4,
        #     Γ_1_n,Γ_2_n,Γ_3_n,Σ_n,
        #     k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,dt,rk4_coeffs[4])
        # synchronize()
        
        #######
        
        #Name change
        begin
            ϕ_1=ϕ_1_n
            ϕ_2=ϕ_2_n
            ϕ_3=ϕ_3_n
            ϕ_4=ϕ_4_n
            # W_1_1=W_1_1_n
            W_1_2=W_1_2_n
            W_1_3=W_1_3_n
            W_1_4=W_1_4_n
            # W_2_1=W_2_1_n
            W_2_2=W_2_2_n
            W_2_3=W_2_3_n
            W_2_4=W_2_4_n
            # W_3_1=W_3_1_n
            W_3_2=W_3_2_n
            W_3_3=W_3_3_n
            W_3_4=W_3_4_n
            Y_2=Y_2_n
            Y_3=Y_3_n
            Y_4=Y_4_n
            Γ_1 = Γ_1_n
            Γ_2 = Γ_2_n
            Γ_3 = Γ_3_n
            Σ = Σ_n
            
            dϕ_1_dt=dϕ_1_dt_n
            dϕ_2_dt=dϕ_2_dt_n
            dϕ_3_dt=dϕ_3_dt_n
            dϕ_4_dt=dϕ_4_dt_n
            # dW_1_1_dt=dW_1_1_dt_n
            dW_1_2_dt=dW_1_2_dt_n
            dW_1_3_dt=dW_1_3_dt_n
            dW_1_4_dt=dW_1_4_dt_n
            # dW_2_1_dt=dW_2_1_dt_n
            dW_2_2_dt=dW_2_2_dt_n
            dW_2_3_dt=dW_2_3_dt_n
            dW_2_4_dt=dW_2_4_dt_n
            # dW_3_1_dt=dW_3_1_dt_n
            dW_3_2_dt=dW_3_2_dt_n
            dW_3_3_dt=dW_3_3_dt_n
            dW_3_4_dt=dW_3_4_dt_n
            # dY_1_dt=dY_1_dt_n
            dY_2_dt=dY_2_dt_n
            dY_3_dt=dY_3_dt_n
            dY_4_dt=dY_4_dt_n
        end
        # println("name-changes")
        
        # @cuda threads=thrds blocks=blks name_change!(ϕ_1_n,ϕ_2_n,ϕ_3_n,ϕ_4_n,
        #     W_1_2_n,W_1_3_n,W_1_4_n,
        #     W_2_2_n,W_2_3_n,W_2_4_n,
        #     W_3_2_n,W_3_3_n,W_3_4_n,
        #     Y_2_n,Y_3_n,Y_4_n,
        #     Γ_1_n,Γ_2_n,Γ_3_n,Σ_n,
        #     ϕ_1,ϕ_2,ϕ_3,ϕ_4,
        #     W_1_2,W_1_3,W_1_4,
        #     W_2_2,W_2_3,W_2_4,
        #     W_3_2,W_3_3,W_3_4,
        #     Y_2,Y_3,Y_4,
        #     Γ_1,Γ_2,Γ_3,Σ)
        # synchronize()
        
        #######

        # @cuda threads=thrds blocks=blks name_change_dot!(dϕ_1_dt_n,dϕ_2_dt_n,dϕ_3_dt_n,dϕ_4_dt_n,
        #     dW_1_2_dt_n,dW_1_3_dt_n,dW_1_4_dt_n,
        #     dW_2_2_dt_n,dW_2_3_dt_n,dW_2_4_dt_n,
        #     dW_3_2_dt_n,dW_3_3_dt_n,dW_3_4_dt_n,
        #     dY_2_dt_n,dY_3_dt_n,dY_4_dt_n,
        #     dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
        #     dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
        #     dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
        #     dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
        #     dY_2_dt,dY_3_dt,dY_4_dt)
        #########
        
        synchronize()
    
    end
    
        if mod(it,dsnaps)==0

            #Compute energies and magnetic fields
            @cuda threads=thrds blocks=blks compute_aux!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
                W_1_2,W_1_3,W_1_4,
                W_2_2,W_2_3,W_2_4,
                W_3_2,W_3_3,W_3_4,
                Y_2,Y_3,Y_4,
                dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
                dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
                dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
                dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
                dY_2_dt,dY_3_dt,dY_4_dt,
                KE_Phi,GE_Phi,PE_Phi,
                ElectricE_W,MagneticE_W,ElectricE_Y,MagneticE_Y,
                B_x,B_y,B_z,
                B_x_2,B_y_2,B_z_2,
                gw,gy,gp2,vev,lambda,θ_w,dx)
            
            # synchronize()

            # @time begin
            # B_x_fft = Array(fft(B_x))
            # B_y_fft = Array(fft(B_y))
            # B_z_fft = Array(fft(B_z))
            
            # snp_idx = snp_idx+1
            # B_fft[snp_idx,:,:] = convolve_1d((real(conj.(B_x_fft).*B_x_fft.+
            # conj.(B_y_fft).*B_y_fft.+
            # conj.(B_z_fft).*B_z_fft)),Nx,Ny,Nz,spec_cut[1],spec_cut[2],spec_cut[3])
            # end
            
            snp_idx = snp_idx+1

        ##Sum energies##
            begin
                total_energies[snp_idx,1] = sum(Array(PE_Phi))*Vol
                total_energies[snp_idx,2] = sum(Array(KE_Phi))*Vol
                total_energies[snp_idx,3] = sum(Array(GE_Phi))*Vol
                total_energies[snp_idx,4] = sum(Array(ElectricE_W))*Vol
                total_energies[snp_idx,5] = sum(Array(MagneticE_W))*Vol
                total_energies[snp_idx,6] = sum(Array(ElectricE_Y))*Vol
                total_energies[snp_idx,7] = sum(Array(MagneticE_Y))*Vol
                total_energies[snp_idx,8] = sum(0.5*(Array(B_x).^2 .+Array(B_y).^2 .+Array(B_z).^2))*Vol
                total_energies[snp_idx,9] = sum(0.5*(Array(B_x_2).^2 .+Array(B_y_2).^2 .+Array(B_z_2).^2))*Vol

                min_phi_arr[snp_idx]=minimum(sqrt.(Array(ϕ_1).^2 .+ Array(ϕ_2).^2 .+ Array(ϕ_3).^2 .+ Array(ϕ_4).^2))
            end

        ##Print energies##
            begin
                println(snp_idx," E: ",sum(total_energies[snp_idx,:]))

                println(string("--------Energies--t:",it,"--process:----------"))
                println("Potentianl energy Higgs: ",total_energies[snp_idx,1])
                println("Kinetic energy Higgs: ",total_energies[snp_idx,2])
                println("Gradient energy Higgs:",total_energies[snp_idx,3])
                println("Magnetic energy W: ",total_energies[snp_idx,4])
                println("Electric energy W: ",total_energies[snp_idx,5])
                println("Magnetic energy Y: ",total_energies[snp_idx,6])
                println("Electric energy Y: ",total_energies[snp_idx,7])
                println("Total energy: ", sum(total_energies[snp_idx,1:end-2]))
                println("---------------------------------------")
            end

            # println("test:",Array(ϕ_1_n)[3,5,3])
            # exit()
            plot_1=heatmap(z,x,(Array(PE_Phi)[:,Ny÷2,:]),title="PE",clim=clim_1)
            plot_2=heatmap(z,x,(Array(KE_Phi)[:,Ny÷2,:]),title="KE",clim=clim_2)
            plot_3=heatmap(z,x,(Array(GE_Phi)[:,Ny÷2,:]),title="GE",clim=clim_3)

            # plot_3=plot(B_fft[snp_idx,2:end,1],(((B_fft[snp_idx,2:end,1]).^2)./((2*pi)^3*Nx^2)).*B_fft[snp_idx,2:end,2],xscale=:log10,yscale=:log10,minorgrid=true)
            # plot_4=contourf(z,x,(Array(ElectricE_W)[:,Ny÷2,:]+Array(MagneticE_W)[:,Ny÷2,:]+Array(ElectricE_Y)[:,Ny÷2,:]+Array(MagneticE_Y)[:,Ny÷2,:]),title="WY E")
            tot_energy_snp = [sum(total_energies[a,1:end-2]) for a in range(1,snp_idx,step=1)]
            plot_4 = plot(range(0,it,step=dsnaps).*dt*mH,[total_energies[1:snp_idx,1] total_energies[1:snp_idx,2] total_energies[1:snp_idx,3] total_energies[1:snp_idx,8] tot_energy_snp],
            label=["PE" "KE" "GE" "B" "Total"],xlims=(0,nte.*dt*mH))     
            plot(plot_1,plot_2,plot_3,plot_4,layout=4,plot_title=string("it:",it),dpi=600)
            frame(anim)
        end
    end
    gif(anim, "EW3d_test.mp4", fps = FPS)
    println("test:",Array(ϕ_1)[3,5,3])
    CUDA.memory_status()
    # println(size(ϕ_2))
    # println(size(findall(Array(ϕ_1_n).!=0)))
    # println(Array(out)[5,1,1])
    
    h5open(string("data.h5"), "w") do file
        write(file, "energies", total_energies)
        write(file, "min_phi", min_phi_arr)
    end

    @time begin
    B_x_fft = Array(fft(B_x))
    B_y_fft = Array(fft(B_y))
    B_z_fft = Array(fft(B_z))
    
    B_fft[end,:,:] = convolve_1d((real(conj.(B_x_fft).*B_x_fft.+
    conj.(B_y_fft).*B_y_fft.+
    conj.(B_z_fft).*B_z_fft)),Nx,Ny,Nz,spec_cut[1],spec_cut[2],spec_cut[3])
    end

    tot_energy_snp = [sum(total_energies[a,1:end-2]) for a in range(1,nsnaps+1,step=1)]

    gr()
    ENV["GKSwstype"]="nul"
    # plot(range(0,nte,step=dsnaps).*dt*mH,[total_energies[:,1] total_energies[:,2] total_energies[:,3] total_energies[:,4] total_energies[:,5] total_energies[:,6] total_energies[:,7]].+1.0,
    # label=["PE" "KE" "GE" "EEW" "MEW" "EEY" "MEY"],yscale=:log10,dpi=600)
    plot(range(0,nte,step=dsnaps).*dt*mH,[total_energies[:,1] total_energies[:,2] total_energies[:,3] total_energies[:,8] tot_energy_snp].+1.0,
            label=["PE" "KE" "GE" "B" "Total"],xlims=(0,nte.*dt*mH),yscale=:log10)
    png("energies.png")

    gr()
    ENV["GKSwstype"]="nul"
    # plot(range(0,nte,step=dsnaps).*dt*mH,[total_energies[:,1] total_energies[:,2] total_energies[:,3] total_energies[:,4] total_energies[:,5] total_energies[:,6] total_energies[:,7]].+1.0,
    # label=["PE" "KE" "GE" "EEW" "MEW" "EEY" "MEY"],yscale=:log10,dpi=600)
    plot(range(0,nte,step=dsnaps).*dt*mH,[total_energies[:,1] total_energies[:,2] total_energies[:,3] total_energies[:,8] tot_energy_snp],
            label=["PE" "KE" "GE" "B" "Total"],xlims=(0,nte.*dt*mH))
    png("energies-linear.png")
    
    gr()
    ENV["GKSwstype"]="nul"
    plot(range(0,nte,step=dsnaps).*dt*mH,min_phi_arr,dpi=600)
    png("min-phi.png")


    gr()
    ENV["GKSwstype"]="nul"
    y1 = (((B_fft[1,2:end,1]).^2)./((2*pi)^3*Nx^2)).*B_fft[1,2:end,2]
    y2 = (((B_fft[end,2:end,1]).^2)./((2*pi)^3*Nx^2)).*B_fft[end,2:end,2]
    plot(B_fft[end,2:end,1],[y1 y2],label=[0 nte],xscale=:log10,yscale=:log10,minorgrid=true)
    png("spectra.png")
    
    gr()
    ENV["GKSwstype"]="nul"
    p1=plot(range(0,nte,step=dsnaps).*dt*mH,[total_energies[:,2] total_energies[:,3]],
    label=["KE" "GE"],dpi=600)
    p2=plot(range(0,nte,step=dsnaps).*dt*mH,[total_energies[:,4] total_energies[:,5]],
    label=["Grad" "Momentum"],dpi=600)
    p3=plot(range(0,nte,step=dsnaps).*dt*mH,[total_energies[:,6] total_energies[:,7]],
    label=["Grad" "Momentum"],dpi=600)
    plot(p1,p2,p3,layout=grid(3, 1, heights=[0.33 ,0.33, 0.33]))
    png("grad-mom-energies.png")

    gr()
    ENV["GKSwstype"]="nul"
    plot(range(0,nte,step=dsnaps),total_energies[:,1].+total_energies[:,2].+total_energies[:,3],
    label="Higgs")
    plot!(range(0,nte,step=dsnaps),total_energies[:,4].+total_energies[:,5],
    label="SU(2)")
    plot!(range(0,nte,step=dsnaps),total_energies[:,6].+total_energies[:,7],
    label="U(1)")
    plot!(range(0,nte,step=dsnaps),tot_energy_snp,
    label="Total")

    # plot(p1,p2,p3,layout=grid(3, 1, heights=[0.33 ,0.33, 0.33]))
    png("total-energies-components.png")
    
end
run(thermal_init=true)
CUDA.memory_status()
