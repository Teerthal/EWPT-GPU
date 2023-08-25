####---Single GPU electroweak evoltion code using CUDA kernels
####RK4 evolution code for solving the SU(2)xU(1) equations of motion in the temporal gauge
####using the second order Crank-Nicolson method
####Spatial derivatives are 6th order
####Periodic Bounadry conditions are implemented

using CUDA#, CuArrays
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

# include("evolve_euler.jl")
# using .ev_spatial

# include("spatial_fluxes.jl")
# using .r_expressions

# diff_method = "abc"
# if diff_method == "abc"
#     diff = diff_abc
# end

function euler_t!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
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
    dϕ_1_dt_n,dϕ_2_dt_n,dϕ_3_dt_n,dϕ_4_dt_n,
    dW_1_2_dt_n,dW_1_3_dt_n,dW_1_4_dt_n,
    dW_2_2_dt_n,dW_2_3_dt_n,dW_2_4_dt_n,
    dW_3_2_dt_n,dW_3_3_dt_n,dW_3_4_dt_n,
    dY_2_dt_n,dY_3_dt_n,dY_4_dt_n,
    gw,gy,vev,lambda,dx,dt)
    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
              (blockIdx().y - 1) * blockDim().y + threadIdx().y,
              (blockIdx().z - 1) * blockDim().z + threadIdx().z
    
    dϕ_1_dx=dfdx(ϕ_1,i,j,k,0.,1)
    dϕ_2_dx=dfdx(ϕ_2,i,j,k,0.,1)
    dϕ_3_dx=dfdx(ϕ_3,i,j,k,0.,1)
    dϕ_4_dx=dfdx(ϕ_4,i,j,k,0.,1)
    # @cuprintln(dϕ_4_dx)
    dϕ_1_dy=dfdy(ϕ_1,i,j,k,0.,1)
    dϕ_2_dy=dfdy(ϕ_2,i,j,k,0.,1)
    dϕ_3_dy=dfdy(ϕ_3,i,j,k,0.,1)
    dϕ_4_dy=dfdy(ϕ_4,i,j,k,0.,1)

    dϕ_1_dz=dfdz(ϕ_1,i,j,k,0.,1)
    dϕ_2_dz=dfdz(ϕ_2,i,j,k,0.,1)
    dϕ_3_dz=dfdz(ϕ_3,i,j,k,0.,1)
    dϕ_4_dz=dfdz(ϕ_4,i,j,k,0.,1)

    dW_1_1_dx = dfdx(W_1_1,i,j,k,0.,1)
    dW_1_2_dx = dfdx(W_1_2,i,j,k,0.,1)
    dW_1_3_dx = dfdx(W_1_3,i,j,k,0.,1)
    dW_1_4_dx = dfdx(W_1_4,i,j,k,0.,1)

    dW_1_1_dy = dfdy(W_1_1,i,j,k,0.,1)
    dW_1_2_dy = dfdy(W_1_2,i,j,k,0.,1)
    dW_1_3_dy = dfdy(W_1_3,i,j,k,0.,1)
    dW_1_4_dy = dfdy(W_1_4,i,j,k,0.,1)

    dW_1_1_dz = dfdz(W_1_1,i,j,k,0.,1)
    dW_1_2_dz = dfdz(W_1_2,i,j,k,0.,1)
    dW_1_3_dz = dfdz(W_1_3,i,j,k,0.,1)
    dW_1_4_dz = dfdz(W_1_4,i,j,k,0.,1)

    dW_2_1_dx = dfdx(W_2_1,i,j,k,0.,1)
    dW_2_2_dx = dfdx(W_2_2,i,j,k,0.,1)
    dW_2_3_dx = dfdx(W_2_3,i,j,k,0.,1)
    dW_2_4_dx = dfdx(W_2_4,i,j,k,0.,1)

    dW_2_1_dy = dfdy(W_2_1,i,j,k,0.,1)
    dW_2_2_dy = dfdy(W_2_2,i,j,k,0.,1)
    dW_2_3_dy = dfdy(W_2_3,i,j,k,0.,1)
    dW_2_4_dy = dfdy(W_2_4,i,j,k,0.,1)

    dW_2_1_dz = dfdz(W_2_1,i,j,k,0.,1)
    dW_2_2_dz = dfdz(W_2_2,i,j,k,0.,1)
    dW_2_3_dz = dfdz(W_2_3,i,j,k,0.,1)
    dW_2_4_dz = dfdz(W_2_4,i,j,k,0.,1)

    dW_3_1_dx = dfdx(W_3_1,i,j,k,0.,1)
    dW_3_2_dx = dfdx(W_3_2,i,j,k,0.,1)
    dW_3_3_dx = dfdx(W_3_3,i,j,k,0.,1)
    dW_3_4_dx = dfdx(W_3_4,i,j,k,0.,1)

    dW_3_1_dy = dfdy(W_3_1,i,j,k,0.,1)
    dW_3_2_dy = dfdy(W_3_2,i,j,k,0.,1)
    dW_3_3_dy = dfdy(W_3_3,i,j,k,0.,1)
    dW_3_4_dy = dfdy(W_3_4,i,j,k,0.,1)

    dW_3_1_dz = dfdz(W_3_1,i,j,k,0.,1)
    dW_3_2_dz = dfdz(W_3_2,i,j,k,0.,1)
    dW_3_3_dz = dfdz(W_3_3,i,j,k,0.,1)
    dW_3_4_dz = dfdz(W_3_4,i,j,k,0.,1)

    dY_1_dx = dfdx(Y_1,i,j,k,0.,1)
    dY_2_dx = dfdx(Y_2,i,j,k,0.,1)
    dY_3_dx = dfdx(Y_3,i,j,k,0.,1)
    dY_4_dx = dfdx(Y_4,i,j,k,0.,1)

    dY_1_dy = dfdy(Y_1,i,j,k,0.,1)
    dY_2_dy = dfdy(Y_2,i,j,k,0.,1)
    dY_3_dy = dfdy(Y_3,i,j,k,0.,1)
    dY_4_dy = dfdy(Y_4,i,j,k,0.,1)

    dY_1_dz = dfdz(Y_1,i,j,k,0.,1)
    dY_2_dz = dfdz(Y_2,i,j,k,0.,1)
    dY_3_dz = dfdz(Y_3,i,j,k,0.,1)
    dY_4_dz = dfdz(Y_4,i,j,k,0.,1)

    dΓ_1_dx = dfdx(Γ_1,i,j,k,0.,1)
    dΓ_1_dy = dfdy(Γ_1,i,j,k,0.,1)
    dΓ_1_dz = dfdz(Γ_1,i,j,k,0.,1)

    dΓ_2_dx = dfdx(Γ_2,i,j,k,0.,1)
    dΓ_2_dy = dfdy(Γ_2,i,j,k,0.,1)
    dΓ_2_dz = dfdz(Γ_2,i,j,k,0.,1)

    dΓ_3_dx = dfdx(Γ_3,i,j,k,0.,1)
    dΓ_3_dy = dfdy(Γ_3,i,j,k,0.,1)
    dΓ_3_dz = dfdz(Γ_3,i,j,k,0.,1)
    
    dΣ_dx = dfdx(Σ,i,j,k,0.,1)
    dΣ_dy = dfdy(Σ,i,j,k,0.,1)
    dΣ_dz = dfdz(Σ,i,j,k,0.,1)

    d2ϕ_1_dx2=d2fdx2(ϕ_1,i,j,k,0.,1)
    d2ϕ_2_dx2=d2fdx2(ϕ_2,i,j,k,0.,1)
    d2ϕ_3_dx2=d2fdx2(ϕ_2,i,j,k,0.,1)
    d2ϕ_4_dx2=d2fdx2(ϕ_2,i,j,k,0.,1)

    d2ϕ_1_dy2=d2fdy2(ϕ_1,i,j,k,0.,1)
    d2ϕ_2_dy2=d2fdy2(ϕ_2,i,j,k,0.,1)
    d2ϕ_3_dy2=d2fdy2(ϕ_2,i,j,k,0.,1)
    d2ϕ_4_dy2=d2fdy2(ϕ_2,i,j,k,0.,1)

    d2ϕ_1_dz2=d2fdz2(ϕ_1,i,j,k,0.,1)
    d2ϕ_2_dz2=d2fdz2(ϕ_2,i,j,k,0.,1)
    d2ϕ_3_dz2=d2fdz2(ϕ_2,i,j,k,0.,1)
    d2ϕ_4_dz2=d2fdz2(ϕ_2,i,j,k,0.,1)

    d2W_1_1_dx2 = d2fdx2(W_1_1,i,j,k,0.,1)
    d2W_1_2_dx2 = d2fdx2(W_1_2,i,j,k,0.,1)
    d2W_1_3_dx2 = d2fdx2(W_1_3,i,j,k,0.,1)
    d2W_1_4_dx2 = d2fdx2(W_1_4,i,j,k,0.,1)

    d2W_1_1_dy2 = d2fdy2(W_1_1,i,j,k,0.,1)
    d2W_1_2_dy2 = d2fdy2(W_1_2,i,j,k,0.,1)
    d2W_1_3_dy2 = d2fdy2(W_1_3,i,j,k,0.,1)
    d2W_1_4_dy2 = d2fdy2(W_1_4,i,j,k,0.,1)

    d2W_1_1_dz2 = d2fdz2(W_1_1,i,j,k,0.,1)
    d2W_1_2_dz2 = d2fdz2(W_1_2,i,j,k,0.,1)
    d2W_1_3_dz2 = d2fdz2(W_1_3,i,j,k,0.,1)
    d2W_1_4_dz2 = d2fdz2(W_1_4,i,j,k,0.,1)

    d2W_2_1_dx2 = d2fdx2(W_2_1,i,j,k,0.,1)
    d2W_2_2_dx2 = d2fdx2(W_2_2,i,j,k,0.,1)
    d2W_2_3_dx2 = d2fdx2(W_2_3,i,j,k,0.,1)
    d2W_2_4_dx2 = d2fdx2(W_2_4,i,j,k,0.,1)

    d2W_2_1_dy2 = d2fdy2(W_2_1,i,j,k,0.,1)
    d2W_2_2_dy2 = d2fdy2(W_2_2,i,j,k,0.,1)
    d2W_2_3_dy2 = d2fdy2(W_2_3,i,j,k,0.,1)
    d2W_2_4_dy2 = d2fdy2(W_2_4,i,j,k,0.,1)

    d2W_2_1_dz2 = d2fdz2(W_2_1,i,j,k,0.,1)
    d2W_2_2_dz2 = d2fdz2(W_2_2,i,j,k,0.,1)
    d2W_2_3_dz2 = d2fdz2(W_2_3,i,j,k,0.,1)
    d2W_2_4_dz2 = d2fdz2(W_2_4,i,j,k,0.,1)

    d2W_3_1_dx2 = d2fdx2(W_3_1,i,j,k,0.,1)
    d2W_3_2_dx2 = d2fdx2(W_3_2,i,j,k,0.,1)
    d2W_3_3_dx2 = d2fdx2(W_3_3,i,j,k,0.,1)
    d2W_3_4_dx2 = d2fdx2(W_3_4,i,j,k,0.,1)

    d2W_3_1_dy2 = d2fdy2(W_3_1,i,j,k,0.,1)
    d2W_3_2_dy2 = d2fdy2(W_3_2,i,j,k,0.,1)
    d2W_3_3_dy2 = d2fdy2(W_3_3,i,j,k,0.,1)
    d2W_3_4_dy2 = d2fdy2(W_3_4,i,j,k,0.,1)

    d2W_3_1_dz2 = d2fdz2(W_3_1,i,j,k,0.,1)
    d2W_3_2_dz2 = d2fdz2(W_3_2,i,j,k,0.,1)
    d2W_3_3_dz2 = d2fdz2(W_3_3,i,j,k,0.,1)
    d2W_3_4_dz2 = d2fdz2(W_3_4,i,j,k,0.,1)

    d2Y_1_dx2 = d2fdx2(Y_1,i,j,k,0.,1)
    d2Y_2_dx2 = d2fdx2(Y_2,i,j,k,0.,1)
    d2Y_3_dx2 = d2fdx2(Y_3,i,j,k,0.,1)
    d2Y_4_dx2 = d2fdx2(Y_4,i,j,k,0.,1)

    d2Y_1_dy2 = d2fdy2(Y_1,i,j,k,0.,1)
    d2Y_2_dy2 = d2fdy2(Y_2,i,j,k,0.,1)
    d2Y_3_dy2 = d2fdy2(Y_3,i,j,k,0.,1)
    d2Y_4_dy2 = d2fdy2(Y_4,i,j,k,0.,1)

    d2Y_1_dz2 = d2fdz2(Y_1,i,j,k,0.,1)
    d2Y_2_dz2 = d2fdz2(Y_2,i,j,k,0.,1)
    d2Y_3_dz2 = d2fdz2(Y_3,i,j,k,0.,1)
    d2Y_4_dz2 = d2fdz2(Y_4,i,j,k,0.,1)

    ##Covariant Derivatives##

    # D_1ϕ_1(dϕ_1_dt,ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
    # W_1_1[i,j,k],W_2_1[i,j,k],W_3_1[i,j,k],Y_1[i,j,k],gw,gy)
    # D_1ϕ_2(dϕ_2_dt,ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
    # W_1_1[i,j,k],W_2_1[i,j,k],W_3_1[i,j,k],Y_1[i,j,k],gw,gy)
    # D_1ϕ_3(dϕ_3_dt,ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
    # W_1_1[i,j,k],W_2_1[i,j,k],W_3_1[i,j,k],Y_1[i,j,k],gw,gy)
    # D_1ϕ_4(dϕ_4_dt,ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
    # W_1_1[i,j,k],W_2_1[i,j,k],W_3_1[i,j,k],Y_1[i,j,k],gw,gy)
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
    # @cuprintln(dϕ_4_dx)
    # Field Strengths #
    # W_1_11()
    # W_1_12(dW_1_2_dt)
    # W_1_13(dW_1_3_dt)
    # W_1_14(dW_1_4_dt)
    # W_1_22(dW_2_2_dt)
    W_1_23 =W_1_xy(dW_1_2_dx,dW_1_1_dy,W_2_2,W_3_3,W_2_3,W_3_2,gw,i,j,k)
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

    #Fluxes
    # @cuprintln(dϕ_4_dx)
    #0618 Tried every which way to be able to call from module 
    #but some weird restrictions on number of arguments or how they are
    #packaged inside module functions is being a royal pain in the ass.
    #Resorting to just writing then out here
    #NOTE to self probably just write module functions for things
    #that only take few arguments and are simple:
    #ex: covariant derivs, derivs, field strengths,energies, electromagnetics

    # R_ϕ_1(d2ϕ_1_dx2,d2ϕ_1_dy2,d2ϕ_1_dz2,
    # ϕ_1,ϕ_2,ϕ_3,ϕ_4,Γ_1,Γ_2,Γ_3,Σ,gw,gy,vev,lambda,
    # W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4,
    # dϕ_4_dx,dϕ_4_dy,dϕ_4_dz,dϕ_3_dx,dϕ_3_dy,dϕ_3_dz,dϕ_2_dx,dϕ_2_dy,dϕ_2_dz,
    # Dx_ϕ_2,Dy_ϕ_2,Dz_ϕ_2,Dx_ϕ_3,Dy_ϕ_3,Dz_ϕ_3,Dx_ϕ_4,Dy_ϕ_4,Dz_ϕ_4,i,j,k)

    # donkey((d2ϕ_1_dx2,d2ϕ_1_dy2,d2ϕ_1_dz2),
    # (ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],Γ_1[i,j,k],Γ_2[i,j,k],Γ_3[i,j,k],Σ[i,j,k]),
    # gw,gy,vev,lambda,
    # (W_1_2[i,j,k],W_1_3[i,j,k],W_1_4[i,j,k],W_2_2[i,j,k],W_2_3[i,j,k],W_2_4[i,j,k],W_3_2[i,j,k],W_3_3[i,j,k],W_3_4[i,j,k],Y_2[i,j,k],Y_3[i,j,k],Y_4[i,j,k]),
    # (dϕ_4_dx,dϕ_4_dy,dϕ_4_dz,dϕ_3_dx,dϕ_3_dy,dϕ_3_dz,dϕ_2_dx,dϕ_2_dy,dϕ_2_dz),
    # (Dx_ϕ_2,Dy_ϕ_2,Dz_ϕ_2,Dx_ϕ_3,Dy_ϕ_3,Dz_ϕ_3,Dx_ϕ_4,Dy_ϕ_4,Dz_ϕ_4))

    # R_ϕ_1((d2ϕ_1_dx2,d2ϕ_1_dy2,d2ϕ_1_dz2),
    # (ϕ_1,ϕ_2,ϕ_3,ϕ_4,Γ_1,Γ_2,Γ_3,Σ),
    # gw,gy,vev,lambda,
    # (W_1_2,W_1_3,W_1_4,W_2_2,W_2_3,W_2_4,W_3_2,W_3_3,W_3_4,Y_2,Y_3,Y_4),
    # (dϕ_4_dx,dϕ_4_dy,dϕ_4_dz,dϕ_3_dx,dϕ_3_dy,dϕ_3_dz,dϕ_2_dx,dϕ_2_dy,dϕ_2_dz),
    # (Dx_ϕ_2,Dy_ϕ_2,Dz_ϕ_2,Dx_ϕ_3,Dy_ϕ_3,Dz_ϕ_3,Dx_ϕ_4,Dy_ϕ_4,Dz_ϕ_4),
    # (i,j,k))

#     R_ϕ_1=
    @inbounds dϕ_1_dt_n[i,j,k] = (dϕ_1_dt[i,j,k]+dt*(d2ϕ_1_dx2+d2ϕ_1_dy2+d2ϕ_1_dz2-
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

#     R_ϕ_2 = 
    @inbounds dϕ_2_dt_n[i,j,k] = (dϕ_2_dt[i,j,k]+dt*(d2ϕ_2_dx2+d2ϕ_2_dy2+d2ϕ_2_dz2+
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

#     R_ϕ_3 = 
    @inbounds dϕ_3_dt_n[i,j,k] = (dϕ_3_dt[i,j,k]+dt*(d2ϕ_3_dx2+d2ϕ_3_dy2+d2ϕ_3_dz2-
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

#     R_ϕ_4 = 
    @inbounds dϕ_4_dt_n[i,j,k] = (dϕ_4_dt[i,j,k]+dt*(d2ϕ_4_dx2+d2ϕ_4_dy2+d2ϕ_4_dz2+
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

#     R_W_1_2=
    @inbounds dW_1_2_dt_n[i,j,k] = (dW_1_2_dt[i,j,k]+dt*(d2W_1_2_dx2+d2W_1_2_dy2+d2W_1_2_dz2+
    gw*(-(dW_2_2_dx*W_3_2[i,j,k]-dW_3_2_dx*W_2_2[i,j,k])-
        (dW_2_2_dy*W_3_3[i,j,k]-dW_3_2_dy*W_2_3[i,j,k])-
        (dW_2_2_dz*W_3_4[i,j,k]-dW_3_2_dz*W_2_4[i,j,k])-
        (W_2_3[i,j,k]*W_3_23-W_3_3[i,j,k]*W_2_23)-
        (W_2_4[i,j,k]*W_3_24-W_3_4[i,j,k]*W_2_24))+
    gw*(ϕ_1[i,j,k]*Dx_ϕ_4-ϕ_2[i,j,k]*Dx_ϕ_3+ϕ_3[i,j,k]*Dx_ϕ_2-ϕ_4[i,j,k]*Dx_ϕ_1)-
    dΓ_1_dx-gw*(W_2_2[i,j,k]*Γ_3[i,j,k]-W_3_2[i,j,k]*Γ_2[i,j,k])-
    gw*(W_2_1[i,j,k]*dW_3_2_dt[i,j,k]-W_3_1[i,j,k]*dW_2_2_dt[i,j,k])))

#     R_W_1_3=
    @inbounds dW_1_3_dt_n[i,j,k] = (dW_1_3_dt[i,j,k]+dt*(d2W_1_3_dx2+d2W_1_3_dy2+d2W_1_3_dz2+
    gw*(-(dW_2_3_dx*W_3_2[i,j,k]-dW_3_3_dx*W_2_2[i,j,k])-
        (dW_2_3_dy*W_3_3[i,j,k]-dW_3_3_dy*W_2_3[i,j,k])-
        (dW_2_3_dz*W_3_4[i,j,k]-dW_3_3_dz*W_2_4[i,j,k])-
        (W_2_2[i,j,k]*(-W_3_23)-W_3_2[i,j,k]*(-W_2_23))-
        (W_2_4[i,j,k]*W_3_34-W_3_4[i,j,k]*W_2_34))+
    gw*(ϕ_1[i,j,k]*Dy_ϕ_4-ϕ_2[i,j,k]*Dy_ϕ_3+ϕ_3[i,j,k]*Dy_ϕ_2-ϕ_4[i,j,k]*Dy_ϕ_1)-
    dΓ_1_dy-gw*(W_2_3[i,j,k]*Γ_3[i,j,k]-W_3_3[i,j,k]*Γ_2[i,j,k])-
    gw*(W_2_1[i,j,k]*dW_3_3_dt[i,j,k]-W_3_1[i,j,k]*dW_2_3_dt[i,j,k])))

#     R_W_1_4=
    @inbounds dW_1_4_dt_n[i,j,k] = (dW_1_4_dt[i,j,k]+dt*(d2W_1_4_dx2+d2W_1_4_dy2+d2W_1_4_dz2+
    gw*(-(dW_2_4_dx*W_3_2[i,j,k]-dW_3_4_dx*W_2_2[i,j,k])-
        (dW_2_4_dy*W_3_3[i,j,k]-dW_3_4_dy*W_2_3[i,j,k])-
        (dW_2_4_dz*W_3_4[i,j,k]-dW_3_4_dz*W_2_4[i,j,k])-
        (W_2_2[i,j,k]*(-W_3_24)-W_3_2[i,j,k]*(-W_2_24))-
        (W_2_3[i,j,k]*(-W_3_34)-W_3_3[i,j,k]*(-W_2_34)))+
    gw*(ϕ_1[i,j,k]*Dz_ϕ_4-ϕ_2[i,j,k]*Dz_ϕ_3+ϕ_3[i,j,k]*Dz_ϕ_2-ϕ_4[i,j,k]*Dz_ϕ_1)-
    dΓ_1_dz-gw*(W_2_4[i,j,k]*Γ_3[i,j,k]-W_3_4[i,j,k]*Γ_2[i,j,k])-
    gw*(W_2_1[i,j,k]*dW_3_4_dt[i,j,k]-W_3_1[i,j,k]*dW_2_4_dt[i,j,k])))

#     R_W_2_2=
    @inbounds dW_2_2_dt_n[i,j,k] = (dW_2_2_dt[i,j,k]+dt*(d2W_2_2_dx2+d2W_2_2_dy2+d2W_2_2_dz2+
    gw*(-(dW_3_2_dx*W_1_2[i,j,k]-dW_1_2_dx*W_3_2[i,j,k])-
        (dW_3_2_dy*W_1_3[i,j,k]-dW_1_2_dy*W_3_3[i,j,k])-
        (dW_3_2_dz*W_1_4[i,j,k]-dW_1_2_dz*W_3_4[i,j,k])-
        (W_3_3[i,j,k]*W_1_23-W_1_3[i,j,k]*W_3_23)-
        (W_3_4[i,j,k]*W_1_24-W_1_4[i,j,k]*W_3_24))+
    gw*(-ϕ_1[i,j,k]*Dx_ϕ_3-ϕ_2[i,j,k]*Dx_ϕ_4+ϕ_3[i,j,k]*Dx_ϕ_1+ϕ_4[i,j,k]*Dx_ϕ_2)-
    dΓ_2_dx-gw*(W_3_2[i,j,k]*Γ_1[i,j,k]-W_1_2[i,j,k]*Γ_3[i,j,k])-
    gw*(W_3_1[i,j,k]*dW_1_2_dt[i,j,k]-W_1_1[i,j,k]*dW_3_2_dt[i,j,k])))

#     R_W_2_3=
    @inbounds dW_2_3_dt_n[i,j,k] = (dW_2_3_dt[i,j,k]+dt*(d2W_2_3_dx2+d2W_2_3_dy2+d2W_2_3_dz2+
    gw*(-(dW_3_3_dx*W_1_2[i,j,k]-dW_1_3_dx*W_3_2[i,j,k])-
        (dW_3_3_dy*W_1_3[i,j,k]-dW_1_3_dy*W_3_3[i,j,k])-
        (dW_3_3_dz*W_1_4[i,j,k]-dW_1_3_dz*W_3_4[i,j,k])-
        (W_3_2[i,j,k]*(-W_1_23)-W_1_2[i,j,k]*(-W_3_23))-
        (W_3_4[i,j,k]*(W_1_34)-W_1_4[i,j,k]*W_3_34))+
    gw*(-ϕ_1[i,j,k]*Dy_ϕ_3-ϕ_2[i,j,k]*Dy_ϕ_4+ϕ_3[i,j,k]*Dy_ϕ_1+ϕ_4[i,j,k]*Dy_ϕ_2)-
    dΓ_2_dy-gw*(W_3_3[i,j,k]*Γ_1[i,j,k]-W_1_3[i,j,k]*Γ_3[i,j,k])-
    gw*(W_3_1[i,j,k]*dW_1_3_dt[i,j,k]-W_1_1[i,j,k]*dW_3_3_dt[i,j,k])))

#     R_W_2_4=
    @inbounds dW_2_4_dt_n[i,j,k] = (dW_2_4_dt[i,j,k]+dt*(d2W_2_4_dx2+d2W_2_4_dy2+d2W_2_4_dz2+
    gw*(-(dW_3_4_dx*W_1_2[i,j,k]-dW_1_4_dx*W_3_2[i,j,k])-
        (dW_3_4_dy*W_1_3[i,j,k]-dW_1_4_dy*W_3_3[i,j,k])-
        (dW_3_4_dz*W_1_4[i,j,k]-dW_1_4_dz*W_3_4[i,j,k])-
        (W_3_2[i,j,k]*(-W_1_24)-W_1_2[i,j,k]*(-W_3_24))-
        (W_3_3[i,j,k]*(-W_1_34)-W_1_3[i,j,k]*(-W_3_34)))+
    gw*(-ϕ_1[i,j,k]*Dz_ϕ_3-ϕ_2[i,j,k]*Dz_ϕ_4+ϕ_3[i,j,k]*Dz_ϕ_1+ϕ_4[i,j,k]*Dz_ϕ_2)-
    dΓ_2_dz-gw*(W_3_4[i,j,k]*Γ_1[i,j,k]-W_1_4[i,j,k]*Γ_3[i,j,k])-
    gw*(W_3_1[i,j,k]*dW_1_4_dt[i,j,k]-W_1_1[i,j,k]*dW_3_4_dt[i,j,k])))

#     R_W_3_2=
    @inbounds dW_3_2_dt_n[i,j,k] = (dW_3_2_dt[i,j,k]+dt*(d2W_3_2_dx2+d2W_3_2_dy2+d2W_3_2_dz2+
    gw*(-(dW_1_2_dx*W_2_2[i,j,k]-dW_2_2_dx*W_1_2[i,j,k])-
        (dW_1_2_dy*W_2_3[i,j,k]-dW_2_2_dy*W_1_3[i,j,k])-
        (dW_1_2_dz*W_2_4[i,j,k]-dW_2_2_dz*W_1_4[i,j,k])-
        (W_1_3[i,j,k]*W_2_23-W_2_3[i,j,k]*W_1_23)-
        (W_1_4[i,j,k]*W_2_24-W_2_4[i,j,k]*W_1_24))+
    gw*(ϕ_1[i,j,k]*Dx_ϕ_2-ϕ_2[i,j,k]*Dx_ϕ_1-ϕ_3[i,j,k]*Dx_ϕ_4+ϕ_4[i,j,k]*Dx_ϕ_3)-
    dΓ_3_dx-gw*(W_1_2[i,j,k]*Γ_2[i,j,k]-W_2_2[i,j,k]*Γ_1[i,j,k])-
    gw*(W_1_1[i,j,k]*dW_2_2_dt[i,j,k]-W_2_1[i,j,k]*dW_1_2_dt[i,j,k])))

#     R_W_3_3=
    @inbounds dW_3_3_dt_n[i,j,k] = (dW_3_3_dt[i,j,k]+dt*(d2W_3_3_dx2+d2W_3_3_dy2+d2W_3_3_dz2+
    gw*(-(dW_1_3_dx*W_2_2[i,j,k]-dW_2_3_dx*W_1_2[i,j,k])-
        (dW_1_3_dy*W_2_3[i,j,k]-dW_2_3_dy*W_1_3[i,j,k])-
        (dW_1_3_dz*W_2_4[i,j,k]-dW_2_3_dz*W_1_4[i,j,k])-
        (W_1_2[i,j,k]*(-W_2_23)-W_2_2[i,j,k]*(-W_1_23))-
        (W_1_4[i,j,k]*W_2_34-W_2_4[i,j,k]*(W_1_34)))+
    gw*(ϕ_1[i,j,k]*Dy_ϕ_2-ϕ_2[i,j,k]*Dy_ϕ_1-ϕ_3[i,j,k]*Dy_ϕ_4+ϕ_4[i,j,k]*Dy_ϕ_3)-
    dΓ_3_dy-gw*(W_1_3[i,j,k]*Γ_2[i,j,k]-W_2_3[i,j,k]*Γ_1[i,j,k])-
    gw*(W_1_1[i,j,k]*dW_2_3_dt[i,j,k]-W_2_1[i,j,k]*dW_1_3_dt[i,j,k])))

#     R_W_3_4=
    @inbounds dW_3_4_dt_n[i,j,k] = (dW_3_4_dt[i,j,k]+dt*(d2W_3_4_dx2+d2W_3_4_dy2+d2W_3_4_dz2+
    gw*(-(dW_1_4_dx*W_2_2[i,j,k]-dW_2_4_dx*W_1_2[i,j,k])-
        (dW_1_4_dy*W_2_3[i,j,k]-dW_2_4_dy*W_1_3[i,j,k])-
        (dW_1_4_dz*W_2_4[i,j,k]-dW_2_4_dz*W_1_4[i,j,k])-
        (W_1_2[i,j,k]*(-W_2_24)-W_2_2[i,j,k]*(-W_1_24))-
        (W_1_3[i,j,k]*(-W_2_34)-W_2_3[i,j,k]*(-W_1_34)))+
    gw*(ϕ_1[i,j,k]*Dz_ϕ_2-ϕ_2[i,j,k]*Dz_ϕ_1-ϕ_3[i,j,k]*Dz_ϕ_4+ϕ_4[i,j,k]*Dz_ϕ_3)-
    dΓ_3_dz-gw*(W_1_4[i,j,k]*Γ_2[i,j,k]-W_2_4[i,j,k]*Γ_1[i,j,k])-
    gw*(W_1_1[i,j,k]*dW_2_4_dt[i,j,k]-W_2_1[i,j,k]*dW_1_4_dt[i,j,k])))

#     R_Y_2=
    @inbounds dY_2_dt_n[i,j,k] = (dY_2_dt[i,j,k]+dt*(d2Y_2_dx2+d2Y_2_dy2+d2Y_2_dz2+
    gy*(ϕ_1[i,j,k]*Dx_ϕ_2-ϕ_2[i,j,k]*Dx_ϕ_1+ϕ_3[i,j,k]*Dx_ϕ_4-ϕ_4[i,j,k]*Dx_ϕ_3)-dΣ_dx))

#     R_Y_3=
    @inbounds dY_3_dt_n[i,j,k] = (dY_3_dt[i,j,k]+dt*(d2Y_3_dx2+d2Y_3_dy2+d2Y_3_dz2+
    gy*(ϕ_1[i,j,k]*Dy_ϕ_2-ϕ_2[i,j,k]*Dy_ϕ_1+ϕ_3[i,j,k]*Dy_ϕ_4-ϕ_4[i,j,k]*Dy_ϕ_3)-dΣ_dy))

#     R_Y_4=
    @inbounds dY_4_dt_n[i,j,k] = (dY_4_dt[i,j,k]+dt*(d2Y_4_dx2+d2Y_4_dy2+d2Y_4_dz2+
    gy*(ϕ_1[i,j,k]*Dz_ϕ_2-ϕ_2[i,j,k]*Dz_ϕ_1+ϕ_3[i,j,k]*Dz_ϕ_4-ϕ_4[i,j,k]*Dz_ϕ_3)-dΣ_dz))

    # fluxes are trivial so no need to evolve this
    # @inbounds dΓ_1_dt_n[i,j,k] = dΓ_1_dt[i,j,k]#.+dt.*@inn(Γ_1)
    # @inbounds dΓ_2_dt_n[i,j,k] = dΓ_2_dt[i,j,k]#.+dt.*@inn(Γ_2)
    # @inbounds dΓ_3_dt_n[i,j,k] = dΓ_3_dt[i,j,k]#.+dt.*@inn(Γ_3)
    # @inbounds dΣ_dt_n[i,j,k] = dΣ_dt[i,j,k]#.+dt.*@inn(Σ)
    
    # a = ϕ_1[i,j,k]-ϕ_1[i+1,j,k]
    # b = ϕ_2[i,j,k]-ϕ_2[i+1,j,k]
#     x,y,z=cartesian_symm(i,j,k,0,0,0,Nx,Ny,Nz)
#     @inbounds Σ[i,j,k] = dϕ_1_dx + dϕ_2_dx + Dx_ϕ_1

    return
end

function leapforward!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_1,W_1_2,W_1_3,W_1_4,
    W_2_1,W_2_2,W_2_3,W_2_4,
    W_3_1,W_3_2,W_3_3,W_3_4,
    Y_1,Y_2,Y_3,Y_4,
    Γ_1,Γ_2,Γ_3,Σ,
    ϕ_1_n,ϕ_2_n,ϕ_3_n,ϕ_4_n,
    W_1_1_n,W_1_2_n,W_1_3_n,W_1_4_n,
    W_2_1_n,W_2_2_n,W_2_3_n,W_2_4_n,
    W_3_1_n,W_3_2_n,W_3_3_n,W_3_4_n,
    Y_1_n,Y_2_n,Y_3_n,Y_4_n,
    Γ_1_n,Γ_2_n,Γ_3_n,Σ_n,
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
    dΓ_1_dt_n,dΓ_2_dt_n,dΓ_3_dt_n,dΣ_dt_n,
    gw,gy,gp2,vev,lambda,dx,dt)
    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
              (blockIdx().y - 1) * blockDim().y + threadIdx().y,
              (blockIdx().z - 1) * blockDim().z + threadIdx().z
    
    dϕ_1_dx=dfdx(ϕ_1_n,i,j,k,0.,1)
    dϕ_2_dx=dfdx(ϕ_2_n,i,j,k,0.,1)
    dϕ_3_dx=dfdx(ϕ_3_n,i,j,k,0.,1)
    dϕ_4_dx=dfdx(ϕ_4_n,i,j,k,0.,1)
    # @cuprintln(dϕ_4_dx)
    dϕ_1_dy=dfdy(ϕ_1_n,i,j,k,0.,1)
    dϕ_2_dy=dfdy(ϕ_2_n,i,j,k,0.,1)
    dϕ_3_dy=dfdy(ϕ_3_n,i,j,k,0.,1)
    dϕ_4_dy=dfdy(ϕ_4_n,i,j,k,0.,1)

    dϕ_1_dz=dfdz(ϕ_1_n,i,j,k,0.,1)
    dϕ_2_dz=dfdz(ϕ_2_n,i,j,k,0.,1)
    dϕ_3_dz=dfdz(ϕ_3_n,i,j,k,0.,1)
    dϕ_4_dz=dfdz(ϕ_4_n,i,j,k,0.,1)

    dW_1_1_dx = dfdx(W_1_1_n,i,j,k,0.,1)
    dW_1_2_dx = dfdx(W_1_2_n,i,j,k,0.,1)
    dW_1_3_dx = dfdx(W_1_3_n,i,j,k,0.,1)
    dW_1_4_dx = dfdx(W_1_4_n,i,j,k,0.,1)

    dW_1_1_dy = dfdy(W_1_1_n,i,j,k,0.,1)
    dW_1_2_dy = dfdy(W_1_2_n,i,j,k,0.,1)
    dW_1_3_dy = dfdy(W_1_3_n,i,j,k,0.,1)
    dW_1_4_dy = dfdy(W_1_4_n,i,j,k,0.,1)

    dW_1_1_dz = dfdz(W_1_1_n,i,j,k,0.,1)
    dW_1_2_dz = dfdz(W_1_2_n,i,j,k,0.,1)
    dW_1_3_dz = dfdz(W_1_3_n,i,j,k,0.,1)
    dW_1_4_dz = dfdz(W_1_4_n,i,j,k,0.,1)

    dW_2_1_dx = dfdx(W_2_1_n,i,j,k,0.,1)
    dW_2_2_dx = dfdx(W_2_2_n,i,j,k,0.,1)
    dW_2_3_dx = dfdx(W_2_3_n,i,j,k,0.,1)
    dW_2_4_dx = dfdx(W_2_4_n,i,j,k,0.,1)

    dW_2_1_dy = dfdy(W_2_1_n,i,j,k,0.,1)
    dW_2_2_dy = dfdy(W_2_2_n,i,j,k,0.,1)
    dW_2_3_dy = dfdy(W_2_3_n,i,j,k,0.,1)
    dW_2_4_dy = dfdy(W_2_4_n,i,j,k,0.,1)

    dW_2_1_dz = dfdz(W_2_1_n,i,j,k,0.,1)
    dW_2_2_dz = dfdz(W_2_2_n,i,j,k,0.,1)
    dW_2_3_dz = dfdz(W_2_3_n,i,j,k,0.,1)
    dW_2_4_dz = dfdz(W_2_4_n,i,j,k,0.,1)

    dW_3_1_dx = dfdx(W_3_1_n,i,j,k,0.,1)
    dW_3_2_dx = dfdx(W_3_2_n,i,j,k,0.,1)
    dW_3_3_dx = dfdx(W_3_3_n,i,j,k,0.,1)
    dW_3_4_dx = dfdx(W_3_4_n,i,j,k,0.,1)

    dW_3_1_dy = dfdy(W_3_1_n,i,j,k,0.,1)
    dW_3_2_dy = dfdy(W_3_2_n,i,j,k,0.,1)
    dW_3_3_dy = dfdy(W_3_3_n,i,j,k,0.,1)
    dW_3_4_dy = dfdy(W_3_4_n,i,j,k,0.,1)

    dW_3_1_dz = dfdz(W_3_1_n,i,j,k,0.,1)
    dW_3_2_dz = dfdz(W_3_2_n,i,j,k,0.,1)
    dW_3_3_dz = dfdz(W_3_3_n,i,j,k,0.,1)
    dW_3_4_dz = dfdz(W_3_4_n,i,j,k,0.,1)

    dY_1_dx = dfdx(Y_1_n,i,j,k,0.,1)
    dY_2_dx = dfdx(Y_2_n,i,j,k,0.,1)
    dY_3_dx = dfdx(Y_3_n,i,j,k,0.,1)
    dY_4_dx = dfdx(Y_4_n,i,j,k,0.,1)

    dY_1_dy = dfdy(Y_1_n,i,j,k,0.,1)
    dY_2_dy = dfdy(Y_2_n,i,j,k,0.,1)
    dY_3_dy = dfdy(Y_3_n,i,j,k,0.,1)
    dY_4_dy = dfdy(Y_4_n,i,j,k,0.,1)

    dY_1_dz = dfdz(Y_1_n,i,j,k,0.,1)
    dY_2_dz = dfdz(Y_2_n,i,j,k,0.,1)
    dY_3_dz = dfdz(Y_3_n,i,j,k,0.,1)
    dY_4_dz = dfdz(Y_4_n,i,j,k,0.,1)

    dΓ_1_dx = dfdx(Γ_1_n,i,j,k,0.,1)
    dΓ_1_dy = dfdy(Γ_1_n,i,j,k,0.,1)
    dΓ_1_dz = dfdz(Γ_1_n,i,j,k,0.,1)

    dΓ_2_dx = dfdx(Γ_2_n,i,j,k,0.,1)
    dΓ_2_dy = dfdy(Γ_2_n,i,j,k,0.,1)
    dΓ_2_dz = dfdz(Γ_2_n,i,j,k,0.,1)

    dΓ_3_dx = dfdx(Γ_3_n,i,j,k,0.,1)
    dΓ_3_dy = dfdy(Γ_3_n,i,j,k,0.,1)
    dΓ_3_dz = dfdz(Γ_3_n,i,j,k,0.,1)
    
    dΣ_dx = dfdx(Σ_n,i,j,k,0.,1)
    dΣ_dy = dfdy(Σ_n,i,j,k,0.,1)
    dΣ_dz = dfdz(Σ_n,i,j,k,0.,1)

    d2ϕ_1_dx2=d2fdx2(ϕ_1_n,i,j,k,0.,1)
    d2ϕ_2_dx2=d2fdx2(ϕ_2_n,i,j,k,0.,1)
    d2ϕ_3_dx2=d2fdx2(ϕ_2_n,i,j,k,0.,1)
    d2ϕ_4_dx2=d2fdx2(ϕ_2_n,i,j,k,0.,1)

    d2ϕ_1_dy2=d2fdy2(ϕ_1_n,i,j,k,0.,1)
    d2ϕ_2_dy2=d2fdy2(ϕ_2_n,i,j,k,0.,1)
    d2ϕ_3_dy2=d2fdy2(ϕ_2_n,i,j,k,0.,1)
    d2ϕ_4_dy2=d2fdy2(ϕ_2_n,i,j,k,0.,1)

    d2ϕ_1_dz2=d2fdz2(ϕ_1_n,i,j,k,0.,1)
    d2ϕ_2_dz2=d2fdz2(ϕ_2_n,i,j,k,0.,1)
    d2ϕ_3_dz2=d2fdz2(ϕ_2_n,i,j,k,0.,1)
    d2ϕ_4_dz2=d2fdz2(ϕ_2_n,i,j,k,0.,1)

    d2W_1_1_dx2 = d2fdx2(W_1_1_n,i,j,k,0.,1)
    d2W_1_2_dx2 = d2fdx2(W_1_2_n,i,j,k,0.,1)
    d2W_1_3_dx2 = d2fdx2(W_1_3_n,i,j,k,0.,1)
    d2W_1_4_dx2 = d2fdx2(W_1_4_n,i,j,k,0.,1)

    d2W_1_1_dy2 = d2fdy2(W_1_1_n,i,j,k,0.,1)
    d2W_1_2_dy2 = d2fdy2(W_1_2_n,i,j,k,0.,1)
    d2W_1_3_dy2 = d2fdy2(W_1_3_n,i,j,k,0.,1)
    d2W_1_4_dy2 = d2fdy2(W_1_4_n,i,j,k,0.,1)

    d2W_1_1_dz2 = d2fdz2(W_1_1_n,i,j,k,0.,1)
    d2W_1_2_dz2 = d2fdz2(W_1_2_n,i,j,k,0.,1)
    d2W_1_3_dz2 = d2fdz2(W_1_3_n,i,j,k,0.,1)
    d2W_1_4_dz2 = d2fdz2(W_1_4_n,i,j,k,0.,1)

    d2W_2_1_dx2 = d2fdx2(W_2_1_n,i,j,k,0.,1)
    d2W_2_2_dx2 = d2fdx2(W_2_2_n,i,j,k,0.,1)
    d2W_2_3_dx2 = d2fdx2(W_2_3_n,i,j,k,0.,1)
    d2W_2_4_dx2 = d2fdx2(W_2_4_n,i,j,k,0.,1)

    d2W_2_1_dy2 = d2fdy2(W_2_1_n,i,j,k,0.,1)
    d2W_2_2_dy2 = d2fdy2(W_2_2_n,i,j,k,0.,1)
    d2W_2_3_dy2 = d2fdy2(W_2_3_n,i,j,k,0.,1)
    d2W_2_4_dy2 = d2fdy2(W_2_4_n,i,j,k,0.,1)

    d2W_2_1_dz2 = d2fdz2(W_2_1_n,i,j,k,0.,1)
    d2W_2_2_dz2 = d2fdz2(W_2_2_n,i,j,k,0.,1)
    d2W_2_3_dz2 = d2fdz2(W_2_3_n,i,j,k,0.,1)
    d2W_2_4_dz2 = d2fdz2(W_2_4_n,i,j,k,0.,1)

    d2W_3_1_dx2 = d2fdx2(W_3_1_n,i,j,k,0.,1)
    d2W_3_2_dx2 = d2fdx2(W_3_2_n,i,j,k,0.,1)
    d2W_3_3_dx2 = d2fdx2(W_3_3_n,i,j,k,0.,1)
    d2W_3_4_dx2 = d2fdx2(W_3_4_n,i,j,k,0.,1)

    d2W_3_1_dy2 = d2fdy2(W_3_1_n,i,j,k,0.,1)
    d2W_3_2_dy2 = d2fdy2(W_3_2_n,i,j,k,0.,1)
    d2W_3_3_dy2 = d2fdy2(W_3_3_n,i,j,k,0.,1)
    d2W_3_4_dy2 = d2fdy2(W_3_4_n,i,j,k,0.,1)

    d2W_3_1_dz2 = d2fdz2(W_3_1_n,i,j,k,0.,1)
    d2W_3_2_dz2 = d2fdz2(W_3_2_n,i,j,k,0.,1)
    d2W_3_3_dz2 = d2fdz2(W_3_3_n,i,j,k,0.,1)
    d2W_3_4_dz2 = d2fdz2(W_3_4_n,i,j,k,0.,1)

    d2Y_1_dx2 = d2fdx2(Y_1_n,i,j,k,0.,1)
    d2Y_2_dx2 = d2fdx2(Y_2_n,i,j,k,0.,1)
    d2Y_3_dx2 = d2fdx2(Y_3_n,i,j,k,0.,1)
    d2Y_4_dx2 = d2fdx2(Y_4_n,i,j,k,0.,1)

    d2Y_1_dy2 = d2fdy2(Y_1_n,i,j,k,0.,1)
    d2Y_2_dy2 = d2fdy2(Y_2_n,i,j,k,0.,1)
    d2Y_3_dy2 = d2fdy2(Y_3_n,i,j,k,0.,1)
    d2Y_4_dy2 = d2fdy2(Y_4_n,i,j,k,0.,1)

    d2Y_1_dz2 = d2fdz2(Y_1_n,i,j,k,0.,1)
    d2Y_2_dz2 = d2fdz2(Y_2_n,i,j,k,0.,1)
    d2Y_3_dz2 = d2fdz2(Y_3_n,i,j,k,0.,1)
    d2Y_4_dz2 = d2fdz2(Y_4_n,i,j,k,0.,1)

    ##Covariant Derivatives##

    Dt_ϕ_1 =D_1ϕ_1(dϕ_1_dt_n[i,j,k],ϕ_2_n[i,j,k],ϕ_3_n[i,j,k],ϕ_4_n[i,j,k],
    W_1_1_n[i,j,k],W_2_1_n[i,j,k],W_3_1_n[i,j,k],Y_1_n[i,j,k],gw,gy)
    Dt_ϕ_2 =D_1ϕ_2(dϕ_2_dt_n[i,j,k],ϕ_1_n[i,j,k],ϕ_3_n[i,j,k],ϕ_4_n[i,j,k],
    W_1_1_n[i,j,k],W_2_1_n[i,j,k],W_3_1_n[i,j,k],Y_1_n[i,j,k],gw,gy)
    Dt_ϕ_3 =D_1ϕ_3(dϕ_3_dt_n[i,j,k],ϕ_1_n[i,j,k],ϕ_2_n[i,j,k],ϕ_4_n[i,j,k],
    W_1_1_n[i,j,k],W_2_1_n[i,j,k],W_3_1_n[i,j,k],Y_1_n[i,j,k],gw,gy)
    Dt_ϕ_4 =D_1ϕ_4(dϕ_4_dt_n[i,j,k],ϕ_1_n[i,j,k],ϕ_2_n[i,j,k],ϕ_3_n[i,j,k],
    W_1_1_n[i,j,k],W_2_1_n[i,j,k],W_3_1_n[i,j,k],Y_1_n[i,j,k],gw,gy)
    Dx_ϕ_1 =D_2ϕ_1(dϕ_1_dx,ϕ_2_n[i,j,k],ϕ_3_n[i,j,k],ϕ_4_n[i,j,k],
    W_1_2_n[i,j,k],W_2_2_n[i,j,k],W_3_2_n[i,j,k],Y_2_n[i,j,k],gw,gy)
    Dx_ϕ_2 =D_2ϕ_2(dϕ_2_dx,ϕ_1_n[i,j,k],ϕ_3_n[i,j,k],ϕ_4_n[i,j,k],
    W_1_2_n[i,j,k],W_2_2_n[i,j,k],W_3_2_n[i,j,k],Y_2_n[i,j,k],gw,gy)
    Dx_ϕ_3 =D_2ϕ_3(dϕ_3_dx,ϕ_1_n[i,j,k],ϕ_2_n[i,j,k],ϕ_4_n[i,j,k],
    W_1_2_n[i,j,k],W_2_2_n[i,j,k],W_3_2_n[i,j,k],Y_2_n[i,j,k],gw,gy)
    Dx_ϕ_4 =D_2ϕ_4(dϕ_4_dx,ϕ_1_n[i,j,k],ϕ_2_n[i,j,k],ϕ_3_n[i,j,k],
    W_1_2_n[i,j,k],W_2_2_n[i,j,k],W_3_2_n[i,j,k],Y_2_n[i,j,k],gw,gy)
    Dy_ϕ_1 =D_3ϕ_1(dϕ_1_dy,ϕ_2_n[i,j,k],ϕ_3_n[i,j,k],ϕ_4_n[i,j,k],
    W_1_3_n[i,j,k],W_2_3_n[i,j,k],W_3_3_n[i,j,k],Y_3_n[i,j,k],gw,gy)
    Dy_ϕ_2 =D_3ϕ_2(dϕ_2_dy,ϕ_1_n[i,j,k],ϕ_3_n[i,j,k],ϕ_4_n[i,j,k],
    W_1_3_n[i,j,k],W_2_3_n[i,j,k],W_3_3_n[i,j,k],Y_3_n[i,j,k],gw,gy)
    Dy_ϕ_3 =D_3ϕ_3(dϕ_3_dy,ϕ_1_n[i,j,k],ϕ_2_n[i,j,k],ϕ_4_n[i,j,k],
    W_1_3_n[i,j,k],W_2_3_n[i,j,k],W_3_3_n[i,j,k],Y_3_n[i,j,k],gw,gy)
    Dy_ϕ_4 =D_3ϕ_4(dϕ_4_dy,ϕ_1_n[i,j,k],ϕ_2_n[i,j,k],ϕ_3_n[i,j,k],
    W_1_3_n[i,j,k],W_2_3_n[i,j,k],W_3_3_n[i,j,k],Y_3_n[i,j,k],gw,gy)
    Dz_ϕ_1 =D_4ϕ_1(dϕ_1_dz,ϕ_2_n[i,j,k],ϕ_3_n[i,j,k],ϕ_4_n[i,j,k],
    W_1_4_n[i,j,k],W_2_4_n[i,j,k],W_3_4_n[i,j,k],Y_4_n[i,j,k],gw,gy)
    Dz_ϕ_2 =D_4ϕ_2(dϕ_2_dz,ϕ_1_n[i,j,k],ϕ_3_n[i,j,k],ϕ_4_n[i,j,k],
    W_1_4_n[i,j,k],W_2_4_n[i,j,k],W_3_4_n[i,j,k],Y_4_n[i,j,k],gw,gy)
    Dz_ϕ_3 =D_4ϕ_3(dϕ_3_dz,ϕ_1_n[i,j,k],ϕ_2_n[i,j,k],ϕ_4_n[i,j,k],
    W_1_4_n[i,j,k],W_2_4_n[i,j,k],W_3_4_n[i,j,k],Y_4_n[i,j,k],gw,gy)
    Dz_ϕ_4 =D_4ϕ_4(dϕ_4_dz,ϕ_1_n[i,j,k],ϕ_2_n[i,j,k],ϕ_3_n[i,j,k],
    W_1_4_n[i,j,k],W_2_4_n[i,j,k],W_3_4_n[i,j,k],Y_4_n[i,j,k],gw,gy)
    # @cuprintln(dϕ_4_dx)
    # Field Strengths #
    # W_1_11()
    # W_1_12(dW_1_2_dt)
    # W_1_13(dW_1_3_dt)
    # W_1_14(dW_1_4_dt)
    # W_1_22(dW_2_2_dt)
    W_1_23 =W_1_xy(dW_1_2_dx,dW_1_1_dy,W_2_2_n,W_3_3_n,W_2_3_n,W_3_2_n,gw,i,j,k)
    W_1_24=W_1_xz(dW_1_4_dx,dW_1_2_dz,W_2_2_n,W_3_4_n,W_2_4_n,W_3_2_n,gw,i,j,k)
    # W_1_33(dW_3_3_dt)
    W_1_34=W_1_yz(dW_1_4_dy,dW_1_3_dz,W_2_3_n,W_3_4_n,W_2_4_n,W_3_3_n,gw,i,j,k)
    # W_1_44()
    # W_2_11()
    # W_2_12(dW_2_2_dt)
    # W_2_13(dW_2_3_dt)
    # W_2_14(dW_2_4_dt)
    # W_2_22()
    W_2_23=W_2_xy(dW_2_3_dx,dW_2_2_dy,W_3_2_n,W_1_3_n,W_3_3_n,W_1_2_n,gw,i,j,k)
    W_2_24=W_2_xz(dW_2_4_dx,dW_2_2_dz,W_3_2_n,W_1_4_n,W_3_4_n,W_1_2_n,gw,i,j,k)
    # W_2_33()
    W_2_34=W_2_yz(dW_2_4_dy,dW_2_3_dz,W_3_3_n,W_1_4_n,W_3_4_n,W_1_3_n,gw,i,j,k)
    # W_2_44()
    # W_3_11()
    # W_3_12(dW_3_2_dt)
    # W_3_13(dW_3_3_dt)
    # W_3_14(dW_3_4_dt)
    # W_3_22()
    W_3_23=W_3_xy(dW_3_3_dx,dW_3_2_dy,W_1_2_n,W_2_3_n,W_1_3_n,W_2_2_n,gw,i,j,k)
    W_3_24=W_3_xz(dW_3_4_dx,dW_3_2_dz,W_1_2_n,W_2_4_n,W_1_4_n,W_2_2_n,gw,i,j,k)
    # W_3_33(dW_3_3_dt)
    W_3_34=W_3_yz(dW_3_4_dy,dW_3_3_dz,W_1_3_n,W_2_4_n,W_1_4_n,W_2_3_n,gw,i,j,k)
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


#     R_ϕ_1=
    dϕ_1_dt_temp = (dϕ_1_dt[i,j,k]+dt*(d2ϕ_1_dx2+d2ϕ_1_dy2+d2ϕ_1_dz2-
    0.5*gw*(((-W_1_2_n[i,j,k]*dϕ_4_dx)-(W_1_3_n[i,j,k]*dϕ_4_dy)-(W_1_4_n[i,j,k]*dϕ_4_dz))-
    ((-W_2_2_n[i,j,k]*dϕ_3_dx)-(W_2_3_n[i,j,k]*dϕ_3_dy)-(W_2_4_n[i,j,k]*dϕ_3_dz))+
    ((-W_3_2_n[i,j,k]*dϕ_2_dx)-(W_3_3_n[i,j,k]*dϕ_2_dy)-(W_3_4_n[i,j,k]*dϕ_2_dz)))-
    0.5*gy*(-Y_2_n[i,j,k]*dϕ_2_dx-Y_3_n[i,j,k]*dϕ_2_dy-Y_4_n[i,j,k]*dϕ_2_dz)-
    0.5*gw*((-W_1_2_n[i,j,k]*Dx_ϕ_4-W_1_3_n[i,j,k]*Dy_ϕ_4-W_1_4_n[i,j,k]*Dz_ϕ_4)-
            (-W_2_2_n[i,j,k]*Dx_ϕ_3-W_2_3_n[i,j,k]*Dy_ϕ_3-W_2_4_n[i,j,k]*Dz_ϕ_3)+
            (-W_3_2_n[i,j,k]*Dx_ϕ_2-W_3_3_n[i,j,k]*Dy_ϕ_2-W_3_4_n[i,j,k]*Dz_ϕ_2))-
    0.5*gy*(-Y_2_n[i,j,k]*Dx_ϕ_2-Y_3_n[i,j,k]*Dy_ϕ_2-Y_4_n[i,j,k]*Dz_ϕ_2)-
    2.0*lambda*(ϕ_1_n[i,j,k]^2+ϕ_2_n[i,j,k]^2+ϕ_3_n[i,j,k]^2+ϕ_4_n[i,j,k]^2-vev^2)*ϕ_1_n[i,j,k]+
    0.5*((gw*Γ_3_n[i,j,k]+gy*Σ_n[i,j,k])*ϕ_2_n[i,j,k]-gw*Γ_2_n[i,j,k]*ϕ_3_n[i,j,k]+gw*Γ_1_n[i,j,k]*ϕ_4_n[i,j,k])))

#     R_ϕ_2 = 
    dϕ_2_dt_temp = (dϕ_2_dt[i,j,k]+dt*(d2ϕ_2_dx2+d2ϕ_2_dy2+d2ϕ_2_dz2+
    0.5*gw*((-W_1_2_n[i,j,k]*dϕ_3_dx-W_1_3_n[i,j,k]*dϕ_3_dy-W_1_4_n[i,j,k]*dϕ_3_dz)+
            (-W_2_2_n[i,j,k]*dϕ_4_dx-W_2_3_n[i,j,k]*dϕ_4_dy-W_2_4_n[i,j,k]*dϕ_4_dz)+
            (-W_3_2_n[i,j,k]*dϕ_1_dx-W_3_3_n[i,j,k]*dϕ_1_dy-W_3_4_n[i,j,k]*dϕ_1_dz))+
    0.5*gy*(-Y_2_n[i,j,k]*dϕ_1_dx-Y_3_n[i,j,k]*dϕ_1_dy-Y_4_n[i,j,k]*dϕ_1_dz)+
    0.5*gw*((-W_1_2_n[i,j,k]*Dx_ϕ_3-W_1_3_n[i,j,k]*Dy_ϕ_3-W_1_4_n[i,j,k]*Dz_ϕ_3)+
            (-W_2_2_n[i,j,k]*Dx_ϕ_4-W_2_3_n[i,j,k]*Dy_ϕ_4-W_2_4_n[i,j,k]*Dz_ϕ_4)+
            (-W_3_2_n[i,j,k]*Dx_ϕ_1-W_3_3_n[i,j,k]*Dy_ϕ_1-W_3_4_n[i,j,k]*Dz_ϕ_1))+
    0.5*gy*(-Y_2_n[i,j,k]*Dx_ϕ_1-Y_3_n[i,j,k]*Dy_ϕ_1-Y_4_n[i,j,k]*Dz_ϕ_1)-
    2.0*lambda*(ϕ_1_n[i,j,k]^2+ϕ_2_n[i,j,k]^2+ϕ_3_n[i,j,k]^2+ϕ_4_n[i,j,k]^2-vev^2)*ϕ_2_n[i,j,k]-
    0.5*((gw*Γ_3_n[i,j,k]+gy*Σ_n[i,j,k])*ϕ_1_n[i,j,k]+gw*Γ_1_n[i,j,k]*ϕ_3_n[i,j,k]+gw*Γ_2_n[i,j,k]*ϕ_4_n[i,j,k])))

#     R_ϕ_3 = 
    dϕ_3_dt_temp = (dϕ_3_dt[i,j,k]+dt*(d2ϕ_3_dx2+d2ϕ_3_dy2+d2ϕ_3_dz2-
    0.5*gw*((-W_1_2_n[i,j,k]*dϕ_2_dx-W_1_3_n[i,j,k]*dϕ_2_dy-W_1_4_n[i,j,k]*dϕ_2_dz)+
            (-W_2_2_n[i,j,k]*dϕ_1_dx-W_2_3_n[i,j,k]*dϕ_1_dy-W_2_4_n[i,j,k]*dϕ_1_dz)-
            (-W_3_2_n[i,j,k]*dϕ_4_dx-W_3_3_n[i,j,k]*dϕ_4_dy-W_3_4_n[i,j,k]*dϕ_4_dz))-
    0.5*gy*(-Y_2_n[i,j,k]*dϕ_4_dx-Y_3_n[i,j,k]*dϕ_4_dy-Y_4_n[i,j,k]*dϕ_4_dz)-
    0.5*gw*((-W_1_2_n[i,j,k]*Dx_ϕ_2-W_1_3_n[i,j,k]*Dy_ϕ_2-W_1_4_n[i,j,k]*Dz_ϕ_2)+
            (-W_2_2_n[i,j,k]*Dx_ϕ_1-W_2_3_n[i,j,k]*Dy_ϕ_1-W_2_4_n[i,j,k]*Dz_ϕ_1)-
            (-W_3_2_n[i,j,k]*Dx_ϕ_4-W_3_3_n[i,j,k]*Dy_ϕ_4-W_3_4_n[i,j,k]*Dz_ϕ_4))-
    0.5*gy*(-Y_2_n[i,j,k]*Dx_ϕ_4-Y_3_n[i,j,k]*Dy_ϕ_4-Y_4_n[i,j,k]*Dz_ϕ_4)-
    2.0*lambda*(ϕ_1_n[i,j,k]^2+ϕ_2_n[i,j,k]^2+ϕ_3_n[i,j,k]^2+ϕ_4_n[i,j,k]^2-vev^2)*ϕ_3_n[i,j,k]+
    0.5*((-gw*Γ_3_n[i,j,k]+gy*Σ_n[i,j,k])*ϕ_4_n[i,j,k]+gw*Γ_2_n[i,j,k]*ϕ_1_n[i,j,k]+gw*Γ_1_n[i,j,k]*ϕ_2_n[i,j,k])))

#     R_ϕ_4 = 
    dϕ_4_dt_temp = (dϕ_4_dt[i,j,k]+dt*(d2ϕ_4_dx2+d2ϕ_4_dy2+d2ϕ_4_dz2+
    0.5*gw*((-W_1_2_n[i,j,k]*dϕ_1_dx-W_1_3_n[i,j,k]*dϕ_1_dy-W_1_4_n[i,j,k]*dϕ_1_dz)-
            (-W_2_2_n[i,j,k]*dϕ_2_dx-W_2_3_n[i,j,k]*dϕ_2_dy-W_2_4_n[i,j,k]*dϕ_2_dz)-
            (-W_3_2_n[i,j,k]*dϕ_3_dx-W_3_3_n[i,j,k]*dϕ_3_dy-W_3_4_n[i,j,k]*dϕ_3_dz))+
    0.5*gy*(-Y_2_n[i,j,k]*dϕ_3_dx-Y_3_n[i,j,k]*dϕ_3_dy-Y_4_n[i,j,k]*dϕ_3_dz)+
    0.5*gw*((-W_1_2_n[i,j,k]*Dx_ϕ_1-W_1_3_n[i,j,k]*Dy_ϕ_1-W_1_4_n[i,j,k]*Dz_ϕ_1)-
            (-W_2_2_n[i,j,k]*Dx_ϕ_2-W_2_3_n[i,j,k]*Dy_ϕ_2-W_2_4_n[i,j,k]*Dz_ϕ_2)-
            (-W_3_2_n[i,j,k]*Dx_ϕ_3-W_3_3_n[i,j,k]*Dy_ϕ_3-W_3_4_n[i,j,k]*Dz_ϕ_3))+
    0.5*gy*(-Y_2_n[i,j,k]*Dx_ϕ_3-Y_3_n[i,j,k]*Dy_ϕ_3-Y_4_n[i,j,k]*Dz_ϕ_3)-
    2.0*lambda*(ϕ_1_n[i,j,k]^2+ϕ_2_n[i,j,k]^2+ϕ_3_n[i,j,k]^2+ϕ_4_n[i,j,k]^2-vev^2)*ϕ_4_n[i,j,k]-
    0.5*((-gw*Γ_3_n[i,j,k]+gy*Σ_n[i,j,k])*ϕ_3_n[i,j,k]+gw*Γ_1_n[i,j,k]*ϕ_1_n[i,j,k]-gw*Γ_2_n[i,j,k]*ϕ_2_n[i,j,k])))

#     R_W_1_2=
    dW_1_2_dt_temp = (dW_1_2_dt[i,j,k]+dt*(d2W_1_2_dx2+d2W_1_2_dy2+d2W_1_2_dz2+
    gw*(-(dW_2_2_dx*W_3_2_n[i,j,k]-dW_3_2_dx*W_2_2_n[i,j,k])-
        (dW_2_2_dy*W_3_3_n[i,j,k]-dW_3_2_dy*W_2_3_n[i,j,k])-
        (dW_2_2_dz*W_3_4_n[i,j,k]-dW_3_2_dz*W_2_4_n[i,j,k])-
        (W_2_3_n[i,j,k]*W_3_23-W_3_3_n[i,j,k]*W_2_23)-
        (W_2_4_n[i,j,k]*W_3_24-W_3_4_n[i,j,k]*W_2_24))+
    gw*(ϕ_1_n[i,j,k]*Dx_ϕ_4-ϕ_2_n[i,j,k]*Dx_ϕ_3+ϕ_3_n[i,j,k]*Dx_ϕ_2-ϕ_4_n[i,j,k]*Dx_ϕ_1)-
    dΓ_1_dx-gw*(W_2_2_n[i,j,k]*Γ_3_n[i,j,k]-W_3_2_n[i,j,k]*Γ_2_n[i,j,k])-
    gw*(W_2_1_n[i,j,k]*dW_3_2_dt_n[i,j,k]-W_3_1_n[i,j,k]*dW_2_2_dt_n[i,j,k])))

#     R_W_1_3=
    dW_1_3_dt_temp = (dW_1_3_dt[i,j,k]+dt*(d2W_1_3_dx2+d2W_1_3_dy2+d2W_1_3_dz2+
    gw*(-(dW_2_3_dx*W_3_2_n[i,j,k]-dW_3_3_dx*W_2_2_n[i,j,k])-
        (dW_2_3_dy*W_3_3_n[i,j,k]-dW_3_3_dy*W_2_3_n[i,j,k])-
        (dW_2_3_dz*W_3_4_n[i,j,k]-dW_3_3_dz*W_2_4_n[i,j,k])-
        (W_2_2_n[i,j,k]*(-W_3_23)-W_3_2_n[i,j,k]*(-W_2_23))-
        (W_2_4_n[i,j,k]*W_3_34-W_3_4_n[i,j,k]*W_2_34))+
    gw*(ϕ_1_n[i,j,k]*Dy_ϕ_4-ϕ_2_n[i,j,k]*Dy_ϕ_3+ϕ_3_n[i,j,k]*Dy_ϕ_2-ϕ_4_n[i,j,k]*Dy_ϕ_1)-
    dΓ_1_dy-gw*(W_2_3_n[i,j,k]*Γ_3_n[i,j,k]-W_3_3_n[i,j,k]*Γ_2_n[i,j,k])-
    gw*(W_2_1_n[i,j,k]*dW_3_3_dt_n[i,j,k]-W_3_1_n[i,j,k]*dW_2_3_dt_n[i,j,k])))

#     R_W_1_4=
    dW_1_4_dt_temp = (dW_1_4_dt[i,j,k]+dt*(d2W_1_4_dx2+d2W_1_4_dy2+d2W_1_4_dz2+
    gw*(-(dW_2_4_dx*W_3_2_n[i,j,k]-dW_3_4_dx*W_2_2_n[i,j,k])-
        (dW_2_4_dy*W_3_3_n[i,j,k]-dW_3_4_dy*W_2_3_n[i,j,k])-
        (dW_2_4_dz*W_3_4_n[i,j,k]-dW_3_4_dz*W_2_4_n[i,j,k])-
        (W_2_2_n[i,j,k]*(-W_3_24)-W_3_2_n[i,j,k]*(-W_2_24))-
        (W_2_3_n[i,j,k]*(-W_3_34)-W_3_3_n[i,j,k]*(-W_2_34)))+
    gw*(ϕ_1_n[i,j,k]*Dz_ϕ_4-ϕ_2_n[i,j,k]*Dz_ϕ_3+ϕ_3_n[i,j,k]*Dz_ϕ_2-ϕ_4_n[i,j,k]*Dz_ϕ_1)-
    dΓ_1_dz-gw*(W_2_4_n[i,j,k]*Γ_3_n[i,j,k]-W_3_4_n[i,j,k]*Γ_2_n[i,j,k])-
    gw*(W_2_1_n[i,j,k]*dW_3_4_dt_n[i,j,k]-W_3_1_n[i,j,k]*dW_2_4_dt_n[i,j,k])))

#     R_W_2_2=
    dW_2_2_dt_temp = (dW_2_2_dt[i,j,k]+dt*(d2W_2_2_dx2+d2W_2_2_dy2+d2W_2_2_dz2+
    gw*(-(dW_3_2_dx*W_1_2_n[i,j,k]-dW_1_2_dx*W_3_2_n[i,j,k])-
        (dW_3_2_dy*W_1_3_n[i,j,k]-dW_1_2_dy*W_3_3_n[i,j,k])-
        (dW_3_2_dz*W_1_4_n[i,j,k]-dW_1_2_dz*W_3_4_n[i,j,k])-
        (W_3_3_n[i,j,k]*W_1_23-W_1_3_n[i,j,k]*W_3_23)-
        (W_3_4_n[i,j,k]*W_1_24-W_1_4_n[i,j,k]*W_3_24))+
    gw*(-ϕ_1_n[i,j,k]*Dx_ϕ_3-ϕ_2_n[i,j,k]*Dx_ϕ_4+ϕ_3_n[i,j,k]*Dx_ϕ_1+ϕ_4_n[i,j,k]*Dx_ϕ_2)-
    dΓ_2_dx-gw*(W_3_2_n[i,j,k]*Γ_1_n[i,j,k]-W_1_2_n[i,j,k]*Γ_3_n[i,j,k])-
    gw*(W_3_1_n[i,j,k]*dW_1_2_dt_n[i,j,k]-W_1_1_n[i,j,k]*dW_3_2_dt_n[i,j,k])))

#     R_W_2_3=
    dW_2_3_dt_temp = (dW_2_3_dt[i,j,k]+dt*(d2W_2_3_dx2+d2W_2_3_dy2+d2W_2_3_dz2+
    gw*(-(dW_3_3_dx*W_1_2_n[i,j,k]-dW_1_3_dx*W_3_2_n[i,j,k])-
        (dW_3_3_dy*W_1_3_n[i,j,k]-dW_1_3_dy*W_3_3_n[i,j,k])-
        (dW_3_3_dz*W_1_4_n[i,j,k]-dW_1_3_dz*W_3_4_n[i,j,k])-
        (W_3_2_n[i,j,k]*(-W_1_23)-W_1_2_n[i,j,k]*(-W_3_23))-
        (W_3_4_n[i,j,k]*(W_1_34)-W_1_4_n[i,j,k]*W_3_34))+
    gw*(-ϕ_1_n[i,j,k]*Dy_ϕ_3-ϕ_2_n[i,j,k]*Dy_ϕ_4+ϕ_3_n[i,j,k]*Dy_ϕ_1+ϕ_4_n[i,j,k]*Dy_ϕ_2)-
    dΓ_2_dy-gw*(W_3_3_n[i,j,k]*Γ_1_n[i,j,k]-W_1_3_n[i,j,k]*Γ_3_n[i,j,k])-
    gw*(W_3_1_n[i,j,k]*dW_1_3_dt_n[i,j,k]-W_1_1_n[i,j,k]*dW_3_3_dt_n[i,j,k])))

#     R_W_2_4=
    dW_2_4_dt_temp = (dW_2_4_dt[i,j,k]+dt*(d2W_2_4_dx2+d2W_2_4_dy2+d2W_2_4_dz2+
    gw*(-(dW_3_4_dx*W_1_2_n[i,j,k]-dW_1_4_dx*W_3_2_n[i,j,k])-
        (dW_3_4_dy*W_1_3_n[i,j,k]-dW_1_4_dy*W_3_3_n[i,j,k])-
        (dW_3_4_dz*W_1_4_n[i,j,k]-dW_1_4_dz*W_3_4_n[i,j,k])-
        (W_3_2_n[i,j,k]*(-W_1_24)-W_1_2_n[i,j,k]*(-W_3_24))-
        (W_3_3_n[i,j,k]*(-W_1_34)-W_1_3_n[i,j,k]*(-W_3_34)))+
    gw*(-ϕ_1_n[i,j,k]*Dz_ϕ_3-ϕ_2_n[i,j,k]*Dz_ϕ_4+ϕ_3_n[i,j,k]*Dz_ϕ_1+ϕ_4_n[i,j,k]*Dz_ϕ_2)-
    dΓ_2_dz-gw*(W_3_4_n[i,j,k]*Γ_1_n[i,j,k]-W_1_4_n[i,j,k]*Γ_3_n[i,j,k])-
    gw*(W_3_1_n[i,j,k]*dW_1_4_dt_n[i,j,k]-W_1_1_n[i,j,k]*dW_3_4_dt_n[i,j,k])))

#     R_W_3_2=
    dW_3_2_dt_temp = (dW_3_2_dt[i,j,k]+dt*(d2W_3_2_dx2+d2W_3_2_dy2+d2W_3_2_dz2+
    gw*(-(dW_1_2_dx*W_2_2_n[i,j,k]-dW_2_2_dx*W_1_2_n[i,j,k])-
        (dW_1_2_dy*W_2_3_n[i,j,k]-dW_2_2_dy*W_1_3_n[i,j,k])-
        (dW_1_2_dz*W_2_4_n[i,j,k]-dW_2_2_dz*W_1_4_n[i,j,k])-
        (W_1_3_n[i,j,k]*W_2_23-W_2_3_n[i,j,k]*W_1_23)-
        (W_1_4_n[i,j,k]*W_2_24-W_2_4_n[i,j,k]*W_1_24))+
    gw*(ϕ_1_n[i,j,k]*Dx_ϕ_2-ϕ_2_n[i,j,k]*Dx_ϕ_1-ϕ_3_n[i,j,k]*Dx_ϕ_4+ϕ_4_n[i,j,k]*Dx_ϕ_3)-
    dΓ_3_dx-gw*(W_1_2_n[i,j,k]*Γ_2_n[i,j,k]-W_2_2_n[i,j,k]*Γ_1_n[i,j,k])-
    gw*(W_1_1_n[i,j,k]*dW_2_2_dt_n[i,j,k]-W_2_1_n[i,j,k]*dW_1_2_dt_n[i,j,k])))

#     R_W_3_3=
    dW_3_3_dt_temp = (dW_3_3_dt[i,j,k]+dt*(d2W_3_3_dx2+d2W_3_3_dy2+d2W_3_3_dz2+
    gw*(-(dW_1_3_dx*W_2_2_n[i,j,k]-dW_2_3_dx*W_1_2_n[i,j,k])-
        (dW_1_3_dy*W_2_3_n[i,j,k]-dW_2_3_dy*W_1_3_n[i,j,k])-
        (dW_1_3_dz*W_2_4_n[i,j,k]-dW_2_3_dz*W_1_4_n[i,j,k])-
        (W_1_2_n[i,j,k]*(-W_2_23)-W_2_2_n[i,j,k]*(-W_1_23))-
        (W_1_4_n[i,j,k]*W_2_34-W_2_4_n[i,j,k]*(W_1_34)))+
    gw*(ϕ_1_n[i,j,k]*Dy_ϕ_2-ϕ_2_n[i,j,k]*Dy_ϕ_1-ϕ_3_n[i,j,k]*Dy_ϕ_4+ϕ_4_n[i,j,k]*Dy_ϕ_3)-
    dΓ_3_dy-gw*(W_1_3_n[i,j,k]*Γ_2_n[i,j,k]-W_2_3_n[i,j,k]*Γ_1_n[i,j,k])-
    gw*(W_1_1_n[i,j,k]*dW_2_3_dt_n[i,j,k]-W_2_1_n[i,j,k]*dW_1_3_dt_n[i,j,k])))

#     R_W_3_4=
    dW_3_4_dt_temp = (dW_3_4_dt[i,j,k]+dt*(d2W_3_4_dx2+d2W_3_4_dy2+d2W_3_4_dz2+
    gw*(-(dW_1_4_dx*W_2_2_n[i,j,k]-dW_2_4_dx*W_1_2_n[i,j,k])-
        (dW_1_4_dy*W_2_3_n[i,j,k]-dW_2_4_dy*W_1_3_n[i,j,k])-
        (dW_1_4_dz*W_2_4_n[i,j,k]-dW_2_4_dz*W_1_4_n[i,j,k])-
        (W_1_2_n[i,j,k]*(-W_2_24)-W_2_2_n[i,j,k]*(-W_1_24))-
        (W_1_3_n[i,j,k]*(-W_2_34)-W_2_3_n[i,j,k]*(-W_1_34)))+
    gw*(ϕ_1_n[i,j,k]*Dz_ϕ_2-ϕ_2_n[i,j,k]*Dz_ϕ_1-ϕ_3_n[i,j,k]*Dz_ϕ_4+ϕ_4_n[i,j,k]*Dz_ϕ_3)-
    dΓ_3_dz-gw*(W_1_4_n[i,j,k]*Γ_2_n[i,j,k]-W_2_4_n[i,j,k]*Γ_1_n[i,j,k])-
    gw*(W_1_1_n[i,j,k]*dW_2_4_dt_n[i,j,k]-W_2_1_n[i,j,k]*dW_1_4_dt_n[i,j,k])))

#     R_Y_2=
    dY_2_dt_temp = (dY_2_dt[i,j,k]+dt*(d2Y_2_dx2+d2Y_2_dy2+d2Y_2_dz2+
    gy*(ϕ_1_n[i,j,k]*Dx_ϕ_2-ϕ_2_n[i,j,k]*Dx_ϕ_1+ϕ_3_n[i,j,k]*Dx_ϕ_4-ϕ_4_n[i,j,k]*Dx_ϕ_3)-dΣ_dx))

#     R_Y_3=
    dY_3_dt_temp = (dY_3_dt[i,j,k]+dt*(d2Y_3_dx2+d2Y_3_dy2+d2Y_3_dz2+
    gy*(ϕ_1_n[i,j,k]*Dy_ϕ_2-ϕ_2_n[i,j,k]*Dy_ϕ_1+ϕ_3_n[i,j,k]*Dy_ϕ_4-ϕ_4_n[i,j,k]*Dy_ϕ_3)-dΣ_dy))

#     R_Y_4=
    dY_4_dt_temp = (dY_4_dt[i,j,k]+dt*(d2Y_4_dx2+d2Y_4_dy2+d2Y_4_dz2+
    gy*(ϕ_1_n[i,j,k]*Dz_ϕ_2-ϕ_2_n[i,j,k]*Dz_ϕ_1+ϕ_3_n[i,j,k]*Dz_ϕ_4-ϕ_4_n[i,j,k]*Dz_ϕ_3)-dΣ_dz))

    # s[1]=
    ϕ_1_temp =(ϕ_1[i,j,k]).+dt.*(dϕ_1_dt_n[i,j,k])
    # s[2]=
    ϕ_2_temp =(ϕ_2[i,j,k]).+dt.*(dϕ_2_dt_n[i,j,k])
    # s[3]=
    ϕ_3_temp =(ϕ_3[i,j,k]).+dt.*(dϕ_3_dt_n[i,j,k])
    # s[4]=
    ϕ_4_temp =(ϕ_4[i,j,k]).+dt.*(dϕ_4_dt_n[i,j,k])
    # c
    # c Eq. (2.11) of Baumgarte&Shapiro is $\partial_t A_i = -E_i -...$ so 
    # c we are taking fd(...)=+\partial_t A_i = -E_i (note the sign).
    # s[5]=0.
    # s[6]=
    W_1_2_temp =(W_1_2[i,j,k]).+dt.*((dW_1_2_dt_n[i,j,k]).+
    # c in the gauge $W^a_0=0=Y_0$, f(5...)=0=f(9...)=f(13...) and the line
    # c below vanishes.
        dW_1_1_dx .-gw.*((W_2_1_n[i,j,k]).*(W_3_2_n[i,j,k]).-(W_3_1_n[i,j,k]).*(W_2_2_n[i,j,k])))
    # s[7]=
    W_1_3_temp =(W_1_3[i,j,k]).+dt.*((dW_1_3_dt_n[i,j,k]).+
        dW_1_1_dy.-gw.*((W_2_1_n[i,j,k]).*(W_3_3_n[i,j,k]).-(W_3_1_n[i,j,k]).*(W_2_3_n[i,j,k])))
    # s[8]=
    W_1_4_temp =(W_1_4[i,j,k]).+dt.*((dW_1_4_dt_n[i,j,k]).+
        dW_1_1_dz.-gw.*((W_2_1_n[i,j,k]).*(W_3_4_n[i,j,k]).-(W_3_1_n[i,j,k]).*(W_2_4_n[i,j,k])))

    # s[9]=0.
    # s[10]=
    W_2_2_temp =(W_2_2[i,j,k]).+dt.*((dW_2_2_dt_n[i,j,k]).+
        dW_2_1_dx.-gw.*((W_3_1_n[i,j,k]).*(W_1_2_n[i,j,k]).-(W_1_1_n[i,j,k]).*(W_3_2_n[i,j,k])))
    # s[11]=
    W_2_3_temp =(W_2_3[i,j,k]).+dt.*((dW_2_3_dt_n[i,j,k]).+
        dW_2_1_dy.-gw.*((W_3_1_n[i,j,k]).*(W_1_3_n[i,j,k]).-(W_1_1_n[i,j,k]).*(W_3_3_n[i,j,k])))
    # s[12]=
    W_2_4_temp =(W_2_4[i,j,k]).+dt.*((dW_2_4_dt_n[i,j,k]).+
        dW_2_1_dz.-gw.*((W_3_1_n[i,j,k]).*(W_1_4_n[i,j,k]).-(W_1_1_n[i,j,k]).*(W_3_4_n[i,j,k])))

    # s[13]=0.
    # s[14]=
    W_3_2_temp =(W_3_2[i,j,k]).+dt.*((dW_3_2_dt_n[i,j,k]).+
        dW_3_1_dx.-gw.*((W_1_1_n[i,j,k]).*(W_2_2_n[i,j,k]).-(W_2_1_n[i,j,k]).*(W_1_2_n[i,j,k])))
    # s[15]=
    W_3_3_temp =(W_3_3[i,j,k]).+dt.*((dW_3_3_dt_n[i,j,k]).+
        dW_3_1_dy.-gw.*((W_1_1_n[i,j,k]).*(W_2_3_n[i,j,k]).-(W_2_1_n[i,j,k]).*(W_1_3_n[i,j,k])))
    # s[16]=
    W_3_4_temp =(W_3_4[i,j,k]).+dt.*((dW_3_4_dt_n[i,j,k]).+
        dW_3_1_dz.-gw.*((W_1_1_n[i,j,k]).*(W_2_4_n[i,j,k]).-(W_2_1_n[i,j,k]).*(W_1_4_n[i,j,k])))

    # s[17]=0.
    # s[18]=
    Y_2_temp =(Y_2[i,j,k]).+dt.*((dY_2_dt_n[i,j,k]).+dY_1_dx)
    # s[19]=
    Y_3_temp =(Y_3[i,j,k]).+dt.*((dY_3_dt_n[i,j,k]).+dY_1_dy)
    # s[20]=
    Y_4_temp =(Y_4[i,j,k]).+dt.*((dY_4_dt_n[i,j,k]).+dY_1_dz)

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
    Γ_1_temp =(Γ_1[i,j,k]).+
    dt.*((1.0.-gp2).*(d2W_1_2_dx2 .+d2W_1_3_dy2 .+d2W_1_4_dz2).+
        gp2 .*gw.*(
        -((W_2_2_n[i,j,k]).*(dW_3_2_dt_n[i,j,k]).-
        (W_3_2_n[i,j,k]).*(dW_2_2_dt_n[i,j,k])).-
        ((W_2_3_n[i,j,k]).*(dW_3_3_dt_n[i,j,k]).-
        (W_3_3_n[i,j,k]).*(dW_2_3_dt_n[i,j,k])).-
        ((W_2_4_n[i,j,k]).*(dW_3_4_dt_n[i,j,k]).-
        (W_3_4_n[i,j,k]).*(dW_2_4_dt_n[i,j,k]))).+
    # c charge from Higgs: 
        gp2 .*gw.*((ϕ_1_n[i,j,k]).*((dϕ_4_dt_n[i,j,k]) .-Dt_ϕ_4).-
        (ϕ_4_n[i,j,k]).*((dϕ_1_dt_n[i,j,k]) .-Dt_ϕ_1).+
        (ϕ_3_n[i,j,k]).*((dϕ_2_dt_n[i,j,k]) .-Dt_ϕ_2).-
        (ϕ_2_n[i,j,k]).*((dϕ_3_dt_n[i,j,k]) .-Dt_ϕ_3)))

    # s(Γ_2)=
    Γ_2_temp =(Γ_2[i,j,k]).+
    dt.*((1.0.-gp2).*(d2W_2_2_dx2 .+d2W_2_3_dy2 .+d2W_2_4_dz2).+
        gp2 .*gw.*(
        -((W_3_2_n[i,j,k]).*(dW_1_2_dt_n[i,j,k]).-
        (W_1_2_n[i,j,k]).*(dW_3_2_dt_n[i,j,k])).-
        ((W_3_3_n[i,j,k]).*(dW_1_3_dt_n[i,j,k]).-
        (W_1_3_n[i,j,k]).*(dW_3_3_dt_n[i,j,k])).-
        ((W_3_4_n[i,j,k]).*(dW_1_4_dt_n[i,j,k]).-
        (W_1_4_n[i,j,k]).*(dW_3_4_dt_n[i,j,k]))).+
    # c charge from Higgs: 
        gp2 .*gw.*((ϕ_3_n[i,j,k]).*((dϕ_1_dt_n[i,j,k]) .-Dt_ϕ_1).-
        (ϕ_1_n[i,j,k]).*((dϕ_3_dt_n[i,j,k]) .-Dt_ϕ_3).+
        (ϕ_4_n[i,j,k]).*((dϕ_2_dt_n[i,j,k]) .-Dt_ϕ_2).-
        (ϕ_2_n[i,j,k]).*((dϕ_4_dt_n[i,j,k]) .-Dt_ϕ_4)))

    # s(Γ_3)=
    Γ_3_temp =(Γ_3[i,j,k]).+
    dt.*((1.0.-gp2).*(d2W_3_2_dx2 .+d2W_3_3_dy2 .+d2W_3_4_dz2).+
        gp2 .*gw.*(
        -((W_1_2_n[i,j,k]).*(dW_2_2_dt_n[i,j,k]).-
        (W_2_2_n[i,j,k]).*(dW_1_2_dt_n[i,j,k])).-
        ((W_1_3_n[i,j,k]).*(dW_2_3_dt_n[i,j,k]).-
        (W_2_3_n[i,j,k]).*(dW_1_3_dt_n[i,j,k])).-
        ((W_1_4_n[i,j,k]).*(dW_2_4_dt_n[i,j,k]).-
        (W_2_4_n[i,j,k]).*(dW_1_4_dt_n[i,j,k]))).+
    # c current from Higgs: 
        gp2 .*gw.*((ϕ_1_n[i,j,k]).*((dϕ_2_dt_n[i,j,k]) .-Dt_ϕ_2).-
        (ϕ_2_n[i,j,k]).*((dϕ_1_dt_n[i,j,k]) .-Dt_ϕ_1).+
        (ϕ_4_n[i,j,k]).*((dϕ_3_dt_n[i,j,k]) .-Dt_ϕ_3).-
        (ϕ_3_n[i,j,k]).*((dϕ_4_dt_n[i,j,k]) .-Dt_ϕ_4)))

    # s(Σ)=
    Σ_temp =(Σ[i,j,k]).+
    dt.*((1.0.-gp2).*(d2Y_2_dx2 .+d2Y_3_dy2 .+d2Y_4_dz2).+
    # c current from Higgs: 
        gp2 .*gy.*((ϕ_1_n[i,j,k])*((dϕ_2_dt_n[i,j,k]) .-Dt_ϕ_2).-
        (ϕ_2_n[i,j,k]).*((dϕ_1_dt_n[i,j,k]) .-Dt_ϕ_1).+
        (ϕ_3_n[i,j,k]).*((dϕ_4_dt_n[i,j,k]) .-Dt_ϕ_4).-
        (ϕ_4_n[i,j,k]).*((dϕ_3_dt_n[i,j,k]) .-Dt_ϕ_3)))


    @inbounds ϕ_1_n[i,j,k] = ϕ_1_temp
    @inbounds ϕ_2_n[i,j,k] = ϕ_2_temp
    @inbounds ϕ_3_n[i,j,k] = ϕ_3_temp
    @inbounds ϕ_4_n[i,j,k] = ϕ_4_temp

    # @inbounds W_1_1_n[i,j,k] = W_1_1_temp
    @inbounds W_1_2_n[i,j,k] = W_1_2_temp
    @inbounds W_1_3_n[i,j,k] = W_1_3_temp
    @inbounds W_1_4_n[i,j,k] = W_1_4_temp
    # @inbounds W_2_1_n[i,j,k] = W_2_1_temp
    @inbounds W_2_2_n[i,j,k] = W_2_2_temp
    @inbounds W_2_3_n[i,j,k] = W_2_3_temp
    @inbounds W_2_4_n[i,j,k] = W_2_4_temp
    # @inbounds W_3_1_n[i,j,k] = W_3_1_temp
    @inbounds W_3_2_n[i,j,k] = W_3_2_temp
    @inbounds W_3_3_n[i,j,k] = W_3_3_temp
    @inbounds W_3_4_n[i,j,k] = W_3_4_temp
    # @inbounds Y_1_n[i,j,k] = Y_1_temp
    @inbounds Y_2_n[i,j,k] = Y_2_temp
    @inbounds Y_3_n[i,j,k] = Y_3_temp
    @inbounds Y_4_n[i,j,k] = Y_4_temp
    @inbounds Γ_1_n[i,j,k] = Γ_1_temp
    @inbounds Γ_2_n[i,j,k] = Γ_2_temp
    @inbounds Γ_3_n[i,j,k] = Γ_3_temp
    @inbounds Σ_n[i,j,k] = Σ_temp

    @inbounds dϕ_1_dt_n[i,j,k] = dϕ_1_dt_temp
    @inbounds dϕ_2_dt_n[i,j,k] = dϕ_2_dt_temp
    @inbounds dϕ_3_dt_n[i,j,k] = dϕ_3_dt_temp
    @inbounds dϕ_4_dt_n[i,j,k] = dϕ_4_dt_temp

    # @inbounds dW_1_1_dt_n[i,j,k] = dW_1_1_dt_temp
    @inbounds dW_1_2_dt_n[i,j,k] = dW_1_2_dt_temp
    @inbounds dW_1_3_dt_n[i,j,k] = dW_1_3_dt_temp
    @inbounds dW_1_4_dt_n[i,j,k] = dW_1_4_dt_temp
    # @inbounds dW_2_1_dt_n[i,j,k] = dW_2_1_dt_temp
    @inbounds dW_2_2_dt_n[i,j,k] = dW_2_2_dt_temp
    @inbounds dW_2_3_dt_n[i,j,k] = dW_2_3_dt_temp
    @inbounds dW_2_4_dt_n[i,j,k] = dW_2_4_dt_temp
    # @inbounds dW_3_1_dt_n[i,j,k] = dW_3_1_dt_temp
    @inbounds dW_3_2_dt_n[i,j,k] = dW_3_2_dt_temp
    @inbounds dW_3_3_dt_n[i,j,k] = dW_3_3_dt_temp
    @inbounds dW_3_4_dt_n[i,j,k] = dW_3_4_dt_temp
    # @inbounds dY_1_dt_n[i,j,k] = dY_1_dt_temp
    @inbounds dY_2_dt_n[i,j,k] = dY_2_dt_temp
    @inbounds dY_3_dt_n[i,j,k] = dY_3_dt_temp
    @inbounds dY_4_dt_n[i,j,k] = dY_4_dt_temp
    # @inbounds dΓ_1_dt_n[i,j,k] = dΓ_1_dt_temp
    # @inbounds dΓ_2_dt_n[i,j,k] = dΓ_2_dt_temp
    # @inbounds dΓ_3_dt_n[i,j,k] = dΓ_3_dt_temp
    # @inbounds dΣ_dt_n[i,j,k] = dΣ_dt_temp
    
    # a = ϕ_1[i,j,k]-ϕ_1[i+1,j,k]
    # b = ϕ_2[i,j,k]-ϕ_2[i+1,j,k]
#     x,y,z=cartesian_symm(i,j,k,0,0,0,Nx,Ny,Nz)
#     @inbounds Σ[i,j,k] = dϕ_1_dx + dϕ_2_dx + Dx_ϕ_1

    return
end

function evolve_sp!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
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
    ϕ_1_n,ϕ_2_n,ϕ_3_n,ϕ_4_n,
    W_1_2_n,W_1_3_n,W_1_4_n,
    W_2_2_n,W_2_3_n,W_2_4_n,
    W_3_2_n,W_3_3_n,W_3_4_n,
    Y_2_n,Y_3_n,Y_4_n,
    Γ_1_n,Γ_2_n,Γ_3_n,Σ_n,
    gw,gy,gp2,lambda,vev,dx,dt)
    
    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
              (blockIdx().y - 1) * blockDim().y + threadIdx().y,
              (blockIdx().z - 1) * blockDim().z + threadIdx().z

#     dϕ_1_dx=dfdx(ϕ_1,i,j,k,0.,1)
#     dϕ_2_dx=dfdx(ϕ_2,i,j,k,0.,1)
#     dϕ_3_dx=dfdx(ϕ_2,i,j,k,0.,1)
#     dϕ_4_dx=dfdx(ϕ_2,i,j,k,0.,1)
#     # @cuprintln(dϕ_4_dx)
#     dϕ_1_dy=dfdy(ϕ_1,i,j,k,0.,1)
#     dϕ_2_dy=dfdy(ϕ_2,i,j,k,0.,1)
#     dϕ_3_dy=dfdy(ϕ_2,i,j,k,0.,1)
#     dϕ_4_dy=dfdy(ϕ_2,i,j,k,0.,1)

#     dϕ_1_dz=dfdz(ϕ_1,i,j,k,0.,1)
#     dϕ_2_dz=dfdz(ϕ_2,i,j,k,0.,1)
#     dϕ_3_dz=dfdz(ϕ_2,i,j,k,0.,1)
#     dϕ_4_dz=dfdz(ϕ_2,i,j,k,0.,1)

    dW_1_1_dx = dfdx(W_1_1,i,j,k,0.,1)
#     dW_1_2_dx = dfdx(W_1_2,i,j,k,0.,1)
#     dW_1_3_dx = dfdx(W_1_3,i,j,k,0.,1)
#     dW_1_4_dx = dfdx(W_1_4,i,j,k,0.,1)

    dW_1_1_dy = dfdy(W_1_1,i,j,k,0.,1)
#     dW_1_2_dy = dfdy(W_1_2,i,j,k,0.,1)
#     dW_1_3_dy = dfdy(W_1_3,i,j,k,0.,1)
#     dW_1_4_dy = dfdy(W_1_4,i,j,k,0.,1)

    dW_1_1_dz = dfdz(W_1_1,i,j,k,0.,1)
#     dW_1_2_dz = dfdz(W_1_2,i,j,k,0.,1)
#     dW_1_3_dz = dfdz(W_1_3,i,j,k,0.,1)
#     dW_1_4_dz = dfdz(W_1_4,i,j,k,0.,1)

    dW_2_1_dx = dfdx(W_2_1,i,j,k,0.,1)
#     dW_2_2_dx = dfdx(W_2_2,i,j,k,0.,1)
#     dW_2_3_dx = dfdx(W_2_3,i,j,k,0.,1)
#     dW_2_4_dx = dfdx(W_2_4,i,j,k,0.,1)

    dW_2_1_dy = dfdy(W_2_1,i,j,k,0.,1)
#     dW_2_2_dy = dfdy(W_2_2,i,j,k,0.,1)
#     dW_2_3_dy = dfdy(W_2_3,i,j,k,0.,1)
#     dW_2_4_dy = dfdy(W_2_4,i,j,k,0.,1)

    dW_2_1_dz = dfdz(W_2_1,i,j,k,0.,1)
#     dW_2_2_dz = dfdz(W_2_2,i,j,k,0.,1)
#     dW_2_3_dz = dfdz(W_2_3,i,j,k,0.,1)
#     dW_2_4_dz = dfdz(W_2_4,i,j,k,0.,1)

    dW_3_1_dx = dfdx(W_3_1,i,j,k,0.,1)
#     dW_3_2_dx = dfdx(W_3_2,i,j,k,0.,1)
#     dW_3_3_dx = dfdx(W_3_3,i,j,k,0.,1)
#     dW_3_4_dx = dfdx(W_3_4,i,j,k,0.,1)

    dW_3_1_dy = dfdy(W_3_1,i,j,k,0.,1)
#     dW_3_2_dy = dfdy(W_3_2,i,j,k,0.,1)
#     dW_3_3_dy = dfdy(W_3_3,i,j,k,0.,1)
#     dW_3_4_dy = dfdy(W_3_4,i,j,k,0.,1)

    dW_3_1_dz = dfdz(W_3_1,i,j,k,0.,1)
#     dW_3_2_dz = dfdz(W_3_2,i,j,k,0.,1)
#     dW_3_3_dz = dfdz(W_3_3,i,j,k,0.,1)
#     dW_3_4_dz = dfdz(W_3_4,i,j,k,0.,1)

    dY_1_dx = dfdx(Y_1,i,j,k,0.,1)
#     dY_2_dx = dfdx(Y_2,i,j,k,0.,1)
#     dY_3_dx = dfdx(Y_3,i,j,k,0.,1)
#     dY_4_dx = dfdx(Y_4,i,j,k,0.,1)

    dY_1_dy = dfdy(Y_1,i,j,k,0.,1)
#     dY_2_dy = dfdy(Y_2,i,j,k,0.,1)
#     dY_3_dy = dfdy(Y_3,i,j,k,0.,1)
#     dY_4_dy = dfdy(Y_4,i,j,k,0.,1)

    dY_1_dz = dfdz(Y_1,i,j,k,0.,1)
#     dY_2_dz = dfdz(Y_2,i,j,k,0.,1)
#     dY_3_dz = dfdz(Y_3,i,j,k,0.,1)
#     dY_4_dz = dfdz(Y_4,i,j,k,0.,1)

#     d2W_1_1_dx2 = d2fdx2(W_1_1,i,j,k,0.,1)
    d2W_1_2_dx2 = d2fdx2(W_1_2,i,j,k,0.,1)
#     d2W_1_3_dx2 = d2fdx2(W_1_3,i,j,k,0.,1)
#     d2W_1_4_dx2 = d2fdx2(W_1_4,i,j,k,0.,1)

#     d2W_1_1_dy2 = d2fdy2(W_1_1,i,j,k,0.,1)
#     d2W_1_2_dy2 = d2fdy2(W_1_2,i,j,k,0.,1)
    d2W_1_3_dy2 = d2fdy2(W_1_3,i,j,k,0.,1)
#     d2W_1_4_dy2 = d2fdy2(W_1_4,i,j,k,0.,1)

#     d2W_1_1_dz2 = d2fdz2(W_1_1,i,j,k,0.,1)
#     d2W_1_2_dz2 = d2fdz2(W_1_2,i,j,k,0.,1)
#     d2W_1_3_dz2 = d2fdz2(W_1_3,i,j,k,0.,1)
    d2W_1_4_dz2 = d2fdz2(W_1_4,i,j,k,0.,1)

#     d2W_2_1_dx2 = d2fdx2(W_2_1,i,j,k,0.,1)
    d2W_2_2_dx2 = d2fdx2(W_2_2,i,j,k,0.,1)
#     d2W_2_3_dx2 = d2fdx2(W_2_3,i,j,k,0.,1)
#     d2W_2_4_dx2 = d2fdx2(W_2_4,i,j,k,0.,1)

#     d2W_2_1_dy2 = d2fdy2(W_2_1,i,j,k,0.,1)
#     d2W_2_2_dy2 = d2fdy2(W_2_2,i,j,k,0.,1)
    d2W_2_3_dy2 = d2fdy2(W_2_3,i,j,k,0.,1)
#     d2W_2_4_dy2 = d2fdy2(W_2_4,i,j,k,0.,1)

#     d2W_2_1_dz2 = d2fdz2(W_2_1,i,j,k,0.,1)
#     d2W_2_2_dz2 = d2fdz2(W_2_2,i,j,k,0.,1)
#     d2W_2_3_dz2 = d2fdz2(W_2_3,i,j,k,0.,1)
    d2W_2_4_dz2 = d2fdz2(W_2_4,i,j,k,0.,1)

#     d2W_3_1_dx2 = d2fdx2(W_3_1,i,j,k,0.,1)
    d2W_3_2_dx2 = d2fdx2(W_3_2,i,j,k,0.,1)
#     d2W_3_3_dx2 = d2fdx2(W_3_3,i,j,k,0.,1)
#     d2W_3_4_dx2 = d2fdx2(W_3_4,i,j,k,0.,1)

#     d2W_3_1_dy2 = d2fdy2(W_3_1,i,j,k,0.,1)
#     d2W_3_2_dy2 = d2fdy2(W_3_2,i,j,k,0.,1)
    d2W_3_3_dy2 = d2fdy2(W_3_3,i,j,k,0.,1)
#     d2W_3_4_dy2 = d2fdy2(W_3_4,i,j,k,0.,1)

#     d2W_3_1_dz2 = d2fdz2(W_3_1,i,j,k,0.,1)
#     d2W_3_2_dz2 = d2fdz2(W_3_2,i,j,k,0.,1)
#     d2W_3_3_dz2 = d2fdz2(W_3_3,i,j,k,0.,1)
    d2W_3_4_dz2 = d2fdz2(W_3_4,i,j,k,0.,1)

#     d2Y_1_dx2 = d2fdx2(Y_1,i,j,k,0.,1)
    d2Y_2_dx2 = d2fdx2(Y_2,i,j,k,0.,1)
#     d2Y_3_dx2 = d2fdx2(Y_3,i,j,k,0.,1)
#     d2Y_4_dx2 = d2fdx2(Y_4,i,j,k,0.,1)

#     d2Y_1_dy2 = d2fdy2(Y_1,i,j,k,0.,1)
#     d2Y_2_dy2 = d2fdy2(Y_2,i,j,k,0.,1)
    d2Y_3_dy2 = d2fdy2(Y_3,i,j,k,0.,1)
#     d2Y_4_dy2 = d2fdy2(Y_4,i,j,k,0.,1)

#     d2Y_1_dz2 = d2fdz2(Y_1,i,j,k,0.,1)
#     d2Y_2_dz2 = d2fdz2(Y_2,i,j,k,0.,1)
#     d2Y_3_dz2 = d2fdz2(Y_3,i,j,k,0.,1)
    d2Y_4_dz2 = d2fdz2(Y_4,i,j,k,0.,1)

    ##Covariant Derivatives##

    Dt_ϕ_1 =D_1ϕ_1(dϕ_1_dt[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
    W_1_1[i,j,k],W_2_1[i,j,k],W_3_1[i,j,k],Y_1[i,j,k],gw,gy)
    Dt_ϕ_2 =D_1ϕ_2(dϕ_2_dt[i,j,k],ϕ_1[i,j,k],ϕ_3[i,j,k],ϕ_4[i,j,k],
    W_1_1[i,j,k],W_2_1[i,j,k],W_3_1[i,j,k],Y_1[i,j,k],gw,gy)
    Dt_ϕ_3 =D_1ϕ_3(dϕ_3_dt[i,j,k],ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_4[i,j,k],
    W_1_1[i,j,k],W_2_1[i,j,k],W_3_1[i,j,k],Y_1[i,j,k],gw,gy)
    Dt_ϕ_4 =D_1ϕ_4(dϕ_4_dt[i,j,k],ϕ_1[i,j,k],ϕ_2[i,j,k],ϕ_3[i,j,k],
    W_1_1[i,j,k],W_2_1[i,j,k],W_3_1[i,j,k],Y_1[i,j,k],gw,gy)
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

    # s[1]=
    @inbounds ϕ_1_n[i,j,k] =(ϕ_1[i,j,k]).+dt.*(dϕ_1_dt[i,j,k])
    # s[2]=
    @inbounds ϕ_2_n[i,j,k] =(ϕ_2[i,j,k]).+dt.*(dϕ_2_dt[i,j,k])
    # s[3]=
    @inbounds ϕ_3_n[i,j,k] =(ϕ_3[i,j,k]).+dt.*(dϕ_3_dt[i,j,k])
    # s[4]=
    @inbounds ϕ_4_n[i,j,k] =(ϕ_4[i,j,k]).+dt.*(dϕ_4_dt[i,j,k])
    # c
    # c Eq. (2.11) of Baumgarte&Shapiro is $\partial_t A_i = -E_i -...$ so 
    # c we are taking fd(...)=+\partial_t A_i = -E_i (note the sign).
    # s[5]=0.
    # s[6]=
    @inbounds W_1_2_n[i,j,k] =(W_1_2[i,j,k]).+dt.*((dW_1_2_dt[i,j,k]).+
    # c in the gauge $W^a_0=0=Y_0$, f(5...)=0=f(9...)=f(13...) and the line
    # c below vanishes.
        dW_1_1_dx .-gw.*((W_2_1[i,j,k]).*(W_3_2[i,j,k]).-(W_3_1[i,j,k]).*(W_2_2[i,j,k])))
    # s[7]=
    @inbounds W_1_3_n[i,j,k] =(W_1_3[i,j,k]).+dt.*((dW_1_3_dt[i,j,k]).+
        dW_1_1_dy.-gw.*((W_2_1[i,j,k]).*(W_3_3[i,j,k]).-(W_3_1[i,j,k]).*(W_2_3[i,j,k])))
    # s[8]=
    @inbounds W_1_4_n[i,j,k] =(W_1_4[i,j,k]).+dt.*((dW_1_4_dt[i,j,k]).+
        dW_1_1_dz.-gw.*((W_2_1[i,j,k]).*(W_3_4[i,j,k]).-(W_3_1[i,j,k]).*(W_2_4[i,j,k])))

    # s[9]=0.
    # s[10]=
    @inbounds W_2_2_n[i,j,k] =(W_2_2[i,j,k]).+dt.*((dW_2_2_dt[i,j,k]).+
        dW_2_1_dx.-gw.*((W_3_1[i,j,k]).*(W_1_2[i,j,k]).-(W_1_1[i,j,k]).*(W_3_2[i,j,k])))
    # s[11]=
    @inbounds W_2_3_n[i,j,k] =(W_2_3[i,j,k]).+dt.*((dW_2_3_dt[i,j,k]).+
        dW_2_1_dy.-gw.*((W_3_1[i,j,k]).*(W_1_3[i,j,k]).-(W_1_1[i,j,k]).*(W_3_3[i,j,k])))
    # s[12]=
    @inbounds W_2_4_n[i,j,k] =(W_2_4[i,j,k]).+dt.*((dW_2_4_dt[i,j,k]).+
        dW_2_1_dz.-gw.*((W_3_1[i,j,k]).*(W_1_4[i,j,k]).-(W_1_1[i,j,k]).*(W_3_4[i,j,k])))

    # s[13]=0.
    # s[14]=
    @inbounds W_3_2_n[i,j,k] =(W_3_2[i,j,k]).+dt.*((dW_3_2_dt[i,j,k]).+
        dW_3_1_dx.-gw.*((W_1_1[i,j,k]).*(W_2_2[i,j,k]).-(W_2_1[i,j,k]).*(W_1_2[i,j,k])))
    # s[15]=
    @inbounds W_3_3_n[i,j,k] =(W_3_3[i,j,k]).+dt.*((dW_3_3_dt[i,j,k]).+
        dW_3_1_dy.-gw.*((W_1_1[i,j,k]).*(W_2_3[i,j,k]).-(W_2_1[i,j,k]).*(W_1_3[i,j,k])))
    # s[16]=
    @inbounds W_3_4_n[i,j,k] =(W_3_4[i,j,k]).+dt.*((dW_3_4_dt[i,j,k]).+
        dW_3_1_dz.-gw.*((W_1_1[i,j,k]).*(W_2_4[i,j,k]).-(W_2_1[i,j,k]).*(W_1_4[i,j,k])))

    # s[17]=0.
    # s[18]=
    @inbounds Y_2_n[i,j,k] =(Y_2[i,j,k]).+dt.*((dY_2_dt[i,j,k]).+dY_1_dx)
    # s[19]=
    @inbounds Y_3_n[i,j,k] =(Y_3[i,j,k]).+dt.*((dY_3_dt[i,j,k]).+dY_1_dy)
    # s[20]=
    @inbounds Y_4_n[i,j,k] =(Y_4[i,j,k]).+dt.*((dY_4_dt[i,j,k]).+dY_1_dz)

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
    @inbounds Γ_1_n[i,j,k] =(Γ_1[i,j,k]).+
    dt.*((1.0.-gp2).*(d2W_1_2_dx2 .+d2W_1_3_dy2 .+d2W_1_4_dz2).+
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
    @inbounds Γ_2_n[i,j,k] =(Γ_2[i,j,k]).+
    dt.*((1.0.-gp2).*(d2W_2_2_dx2 .+d2W_2_3_dy2 .+d2W_2_4_dz2).+
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
    @inbounds Γ_3_n[i,j,k] =(Γ_3[i,j,k]).+
    dt.*((1.0.-gp2).*(d2W_3_2_dx2 .+d2W_3_3_dy2 .+d2W_3_4_dz2).+
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
    @inbounds Σ_n[i,j,k] =(Σ[i,j,k]).+
    dt.*((1.0.-gp2).*(d2Y_2_dx2 .+d2Y_3_dy2 .+d2Y_4_dz2).+
    # c current from Higgs: 
        gp2 .*gy.*((ϕ_1[i,j,k])*((dϕ_2_dt[i,j,k]) .-Dt_ϕ_2).-
        (ϕ_2[i,j,k]).*((dϕ_1_dt[i,j,k]) .-Dt_ϕ_1).+
        (ϕ_3[i,j,k]).*((dϕ_4_dt[i,j,k]) .-Dt_ϕ_4).-
        (ϕ_4[i,j,k]).*((dϕ_3_dt[i,j,k]) .-Dt_ϕ_3)))

    return
end

function average_half_step(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_1,W_1_2,W_1_3,W_1_4,
    W_2_1,W_2_2,W_2_3,W_2_4,
    W_3_1,W_3_2,W_3_3,W_3_4,
    Y_1,Y_2,Y_3,Y_4,
    Γ_1,Γ_2,Γ_3,Σ,
    ϕ_1_n,ϕ_2_n,ϕ_3_n,ϕ_4_n,
    W_1_2_n,W_1_3_n,W_1_4_n,
    W_2_2_n,W_2_3_n,W_2_4_n,
    W_3_2_n,W_3_3_n,W_3_4_n,
    Y_2_n,Y_3_n,Y_4_n,
    Γ_1_n,Γ_2_n,Γ_3_n,Σ_n)
    ϕ_1_n = (ϕ_1_n + ϕ_1).*0.5
    ϕ_2_n = (ϕ_2_n + ϕ_2).*0.5
    ϕ_3_n = (ϕ_3_n + ϕ_3).*0.5
    ϕ_4_n = (ϕ_4_n + ϕ_4).*0.5
    W_1_2_n = (W_1_2_n+W_1_2).*0.5
    W_1_3_n = (W_1_3_n+W_1_4).*0.5
    W_1_4_n = (W_1_4_n+W_1_3).*0.5
    W_2_2_n = (W_2_2_n+W_2_2).*0.5
    W_2_3_n = (W_2_3_n+W_2_4).*0.5
    W_2_4_n = (W_2_4_n+W_2_3).*0.5
    W_3_2_n = (W_3_2_n+W_3_2).*0.5
    W_3_3_n = (W_3_3_n+W_3_4).*0.5
    W_3_4_n = (W_3_4_n+W_3_3).*0.5
    Y_2_n = (Y_2_n+Y_2).*0.5
    Y_3_n = (Y_3_n+Y_3).*0.5
    Y_4_n = (Y_4_n+Y_4).*0.5
    Γ_1_n = (Γ_1_n+Γ_1).*0.5
    Γ_2_n = (Γ_2_n+Γ_2).*0.5
    Γ_3_n = (Γ_3_n+Γ_3).*0.5
    Σ_n = (Σ_n+Σ).*0.5

    return
end

function average_half_step_t(dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    dΓ_1_dt,dΓ_2_dt,dΓ_3_dt,dΣ_dt,
    dϕ_1_dt_n,dϕ_2_dt_n,dϕ_3_dt_n,dϕ_4_dt_n,
    dW_1_2_dt_n,dW_1_3_dt_n,dW_1_4_dt_n,
    dW_2_2_dt_n,dW_2_3_dt_n,dW_2_4_dt_n,
    dW_3_2_dt_n,dW_3_3_dt_n,dW_3_4_dt_n,
    dY_2_dt_n,dY_3_dt_n,dY_4_dt_n,
    dΓ_1_dt_n,dΓ_2_dt_n,dΓ_3_dt_n,dΣ_dt_n)

    dϕ_1_dt_n = (dϕ_1_dt_n + dϕ_1_dt).*0.5
    dϕ_2_dt_n = (dϕ_2_dt_n + dϕ_2_dt).*0.5
    dϕ_3_dt_n = (dϕ_3_dt_n + dϕ_3_dt).*0.5
    dϕ_4_dt_n = (dϕ_4_dt_n + dϕ_4_dt).*0.5
    dW_1_2_dt_n = (dW_1_2_dt_n+dW_1_2_dt).*0.5
    dW_1_3_dt_n = (dW_1_3_dt_n+dW_1_4_dt).*0.5
    dW_1_4_dt_n = (dW_1_4_dt_n+dW_1_3_dt).*0.5
    dW_2_2_dt_n = (dW_2_2_dt_n+dW_2_2_dt).*0.5
    dW_2_3_dt_n = (dW_2_3_dt_n+dW_2_4_dt).*0.5
    dW_2_4_dt_n = (dW_2_4_dt_n+dW_2_3_dt).*0.5
    dW_3_2_dt_n = (dW_3_2_dt_n+dW_3_2_dt).*0.5
    dW_3_3_dt_n = (dW_3_3_dt_n+dW_3_4_dt).*0.5
    dW_3_4_dt_n = (dW_3_4_dt_n+dW_3_3_dt).*0.5
    dY_2_dt_n = (dY_2_dt_n+dY_2_dt).*0.5
    dY_3_dt_n = (dY_3_dt_n+dY_3_dt).*0.5
    dY_4_dt_n = (dY_4_dt_n+dY_4_dt).*0.5
    dΓ_1_dt_n = (dΓ_1_dt_n+dΓ_1_dt).*0.5
    dΓ_2_dt_n = (dΓ_2_dt_n+dΓ_2_dt).*0.5
    dΓ_3_dt_n = (dΓ_3_dt_n+dΓ_3_dt).*0.5
    dΣ_dt_n = (dΣ_dt_n+dΣ_dt).*0.5
    return
end

function run()

    Nx=Ny=Nz=32*8
    println(Nx,",",Ny,",",Nz)
    gw = 0.65
    gy = 0.34521
    gp2 = 0.75
    vev = 1.0
    lambda = 1.0/4.0
    dx=1.
    dt=1.
    # Array initializations
    ϕ_1 = CUDA.zeros(Float64,(Nx,Ny,Nz)).+0.1
    ϕ_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    ϕ_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    ϕ_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))

    W_1_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    W_1_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    W_1_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    W_1_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    W_2_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    W_2_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    W_2_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    W_2_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    W_3_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    W_3_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    W_3_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    W_3_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    Y_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    Y_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    Y_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    Y_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    Γ_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    Γ_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    Γ_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
    Σ = CUDA.zeros(Float64,(Nx,Ny,Nz))

    ϕ_1_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    ϕ_2_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    ϕ_3_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    ϕ_4_n = CUDA.zeros(Float64,(Nx,Ny,Nz))

    W_1_1_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    W_1_2_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    W_1_3_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    W_1_4_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    W_2_1_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    W_2_2_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    W_2_3_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    W_2_4_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    W_3_1_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    W_3_2_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    W_3_3_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    W_3_4_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    Y_1_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    Y_2_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    Y_3_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    Y_4_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    Γ_1_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    Γ_2_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    Γ_3_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    Σ_n = CUDA.zeros(Float64,(Nx,Ny,Nz))

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
    dΓ_1_dt = CUDA.zeros(Float64,(Nx,Ny,Nz))
    dΓ_2_dt = CUDA.zeros(Float64,(Nx,Ny,Nz))
    dΓ_3_dt = CUDA.zeros(Float64,(Nx,Ny,Nz))
    dΣ_dt = CUDA.zeros(Float64,(Nx,Ny,Nz))

    dϕ_1_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    dϕ_2_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    dϕ_3_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    dϕ_4_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     dW_1_1_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    dW_1_2_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    dW_1_3_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    dW_1_4_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     dW_2_1_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    dW_2_2_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    dW_2_3_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    dW_2_4_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     dW_3_1_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    dW_3_2_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    dW_3_3_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    dW_3_4_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
#     dY_1_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    dY_2_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    dY_3_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    dY_4_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    dΓ_1_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    dΓ_2_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    dΓ_3_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
    dΣ_dt_n = CUDA.zeros(Float64,(Nx,Ny,Nz))

    CUDA.memory_status()

    # println("test:",Array(ϕ_1_n)[3,5,3])
    ckernel = @cuda launch=false evolve_sp!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
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
    ϕ_1_n,ϕ_2_n,ϕ_3_n,ϕ_4_n,
    W_1_2_n,W_1_3_n,W_1_4_n,
    W_2_2_n,W_2_3_n,W_2_4_n,
    W_3_2_n,W_3_3_n,W_3_4_n,
    Y_2_n,Y_3_n,Y_4_n,
    Γ_1_n,Γ_2_n,Γ_3_n,Σ_n,
    gw,gy,gp2,lambda,vev,dx,dt)

    # ckernel2 = @cuda launch=false euler_t!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    # W_1_1,W_1_2,W_1_3,W_1_4,
    # W_2_1,W_2_2,W_2_3,W_2_4,
    # W_3_1,W_3_2,W_3_3,W_3_4,
    # Y_1,Y_2,Y_3,Y_4,
    # Γ_1,Γ_2,Γ_3,Σ,
    # dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    # dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    # dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    # dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    # dY_2_dt,dY_3_dt,dY_4_dt,
    # dϕ_1_dt_n,dϕ_2_dt_n,dϕ_3_dt_n,dϕ_4_dt_n,
    # dW_1_2_dt_n,dW_1_3_dt_n,dW_1_4_dt_n,
    # dW_2_2_dt_n,dW_2_3_dt_n,dW_2_4_dt_n,
    # dW_3_2_dt_n,dW_3_3_dt_n,dW_3_4_dt_n,
    # dY_2_dt_n,dY_3_dt_n,dY_4_dt_n,
    # dΓ_1_dt_n,dΓ_2_dt_n,dΓ_3_dt_n,dΣ_dt_n,
    # gw,gy,vev,lambda,dx,dt)

    config = launch_configuration(ckernel.fun)
    println(config.threads)
    println(config.blocks)
    thrds = min(Nx*Ny*Nz, config.threads)
    # blocks =  cld(Nx, threads)
    blks = config.blocks  #minimal suggested block size from launch configuration to achieve max occupancy
    println("configuring gpu topology")
    println(string("#threads:",thrds," #blocks:",blks))
    # threads = (32,32,32)
    # blocks  = (size(ϕ_1,1)÷threads[1],
    # size(ϕ_1,2)÷threads[2],
    # size(ϕ_1,3)÷threads[3])
    # thrds = 384 ÷ 4
    thrds = 384 ÷ 4
    blks = 1#08*10
    println(string("#threads:",thrds," #blocks:",blks))
    @time for it in range(1,2000,step=1)
        # threads = (128,128,128)
        # blocks  = (size(ϕ_1,1)÷threads[1],
        # size(ϕ_1,2)÷threads[2],
        # size(ϕ_1,3)÷threads[3])
        # @time @cuda blocks=blocks,threads=threads myKernel!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
        # W_1_1,W_1_2,W_1_3,W_1_4,
        # W_2_1,W_2_2,W_2_3,W_2_4,
        # W_3_1,W_3_2,W_3_3,W_3_4,
        # Y_1,Y_2,Y_3,Y_4,
        # Γ_1,Γ_2,Γ_3,Σ,Nx,Ny,Nz,gw,gy)

        @cuda threads=thrds blocks=blks evolve_sp!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
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
        ϕ_1_n,ϕ_2_n,ϕ_3_n,ϕ_4_n,
        W_1_2_n,W_1_3_n,W_1_4_n,
        W_2_2_n,W_2_3_n,W_2_4_n,
        W_3_2_n,W_3_3_n,W_3_4_n,
        Y_2_n,Y_3_n,Y_4_n,
        Γ_1_n,Γ_2_n,Γ_3_n,Σ_n,
        gw,gy,gp2,lambda,vev,dx,dt)

        @cuda threads=thrds blocks=blks euler_t!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
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
        dϕ_1_dt_n,dϕ_2_dt_n,dϕ_3_dt_n,dϕ_4_dt_n,
        dW_1_2_dt_n,dW_1_3_dt_n,dW_1_4_dt_n,
        dW_2_2_dt_n,dW_2_3_dt_n,dW_2_4_dt_n,
        dW_3_2_dt_n,dW_3_3_dt_n,dW_3_4_dt_n,
        dY_2_dt_n,dY_3_dt_n,dY_4_dt_n,
        gw,gy,vev,lambda,dx,dt)

        # @time ckernel(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
        # W_1_1,W_1_2,W_1_3,W_1_4,
        # W_2_1,W_2_2,W_2_3,W_2_4,
        # W_3_1,W_3_2,W_3_3,W_3_4,
        # Y_1,Y_2,Y_3,Y_4,
        # Γ_1,Γ_2,Γ_3,Σ,
        # dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
        # dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
        # dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
        # dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
        # dY_2_dt,dY_3_dt,dY_4_dt,
        # ϕ_1_n,ϕ_2_n,ϕ_3_n,ϕ_4_n,
        # W_1_2_n,W_1_3_n,W_1_4_n,
        # W_2_2_n,W_2_3_n,W_2_4_n,
        # W_3_2_n,W_3_3_n,W_3_4_n,
        # Y_2_n,Y_3_n,Y_4_n,
        # Γ_1_n,Γ_2_n,Γ_3_n,Σ_n,
        # gw,gy,gp2,lambda,vev,dx,dt;threads=thrds,blocks=blks)
        # println(string("#threads:",thrds," #blocks:",blks))
        # @time ckernel2(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
        # W_1_1,W_1_2,W_1_3,W_1_4,
        # W_2_1,W_2_2,W_2_3,W_2_4,
        # W_3_1,W_3_2,W_3_3,W_3_4,
        # Y_1,Y_2,Y_3,Y_4,
        # Γ_1,Γ_2,Γ_3,Σ,
        # dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
        # dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
        # dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
        # dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
        # dY_2_dt,dY_3_dt,dY_4_dt,
        # dϕ_1_dt_n,dϕ_2_dt_n,dϕ_3_dt_n,dϕ_4_dt_n,
        # dW_1_2_dt_n,dW_1_3_dt_n,dW_1_4_dt_n,
        # dW_2_2_dt_n,dW_2_3_dt_n,dW_2_4_dt_n,
        # dW_3_2_dt_n,dW_3_3_dt_n,dW_3_4_dt_n,
        # dY_2_dt_n,dY_3_dt_n,dY_4_dt_n,
        # dΓ_1_dt_n,dΓ_2_dt_n,dΓ_3_dt_n,dΣ_dt_n,
        # gw,gy,vev,lambda,dx,dt;threads=thrds,blocks=blks)

        synchronize()
        
        #Average half step
        ϕ_1_n = (ϕ_1_n + ϕ_1).*0.5
        ϕ_2_n = (ϕ_2_n + ϕ_2).*0.5
        ϕ_3_n = (ϕ_3_n + ϕ_3).*0.5
        ϕ_4_n = (ϕ_4_n + ϕ_4).*0.5
        W_1_2_n = (W_1_2_n+W_1_2).*0.5
        W_1_3_n = (W_1_3_n+W_1_4).*0.5
        W_1_4_n = (W_1_4_n+W_1_3).*0.5
        W_2_2_n = (W_2_2_n+W_2_2).*0.5
        W_2_3_n = (W_2_3_n+W_2_4).*0.5
        W_2_4_n = (W_2_4_n+W_2_3).*0.5
        W_3_2_n = (W_3_2_n+W_3_2).*0.5
        W_3_3_n = (W_3_3_n+W_3_4).*0.5
        W_3_4_n = (W_3_4_n+W_3_3).*0.5
        Y_2_n = (Y_2_n+Y_2).*0.5
        Y_3_n = (Y_3_n+Y_3).*0.5
        Y_4_n = (Y_4_n+Y_4).*0.5
        Γ_1_n = (Γ_1_n+Γ_1).*0.5
        Γ_2_n = (Γ_2_n+Γ_2).*0.5
        Γ_3_n = (Γ_3_n+Γ_3).*0.5
        Σ_n = (Σ_n+Σ).*0.5

        dϕ_1_dt_n = (dϕ_1_dt_n + dϕ_1_dt).*0.5
        dϕ_2_dt_n = (dϕ_2_dt_n + dϕ_2_dt).*0.5
        dϕ_3_dt_n = (dϕ_3_dt_n + dϕ_3_dt).*0.5
        dϕ_4_dt_n = (dϕ_4_dt_n + dϕ_4_dt).*0.5
        dW_1_2_dt_n = (dW_1_2_dt_n+dW_1_2_dt).*0.5
        dW_1_3_dt_n = (dW_1_3_dt_n+dW_1_4_dt).*0.5
        dW_1_4_dt_n = (dW_1_4_dt_n+dW_1_3_dt).*0.5
        dW_2_2_dt_n = (dW_2_2_dt_n+dW_2_2_dt).*0.5
        dW_2_3_dt_n = (dW_2_3_dt_n+dW_2_4_dt).*0.5
        dW_2_4_dt_n = (dW_2_4_dt_n+dW_2_3_dt).*0.5
        dW_3_2_dt_n = (dW_3_2_dt_n+dW_3_2_dt).*0.5
        dW_3_3_dt_n = (dW_3_3_dt_n+dW_3_4_dt).*0.5
        dW_3_4_dt_n = (dW_3_4_dt_n+dW_3_3_dt).*0.5
        dY_2_dt_n = (dY_2_dt_n+dY_2_dt).*0.5
        dY_3_dt_n = (dY_3_dt_n+dY_3_dt).*0.5
        dY_4_dt_n = (dY_4_dt_n+dY_4_dt).*0.5
        dΓ_1_dt_n = (dΓ_1_dt_n+dΓ_1_dt).*0.5
        dΓ_2_dt_n = (dΓ_2_dt_n+dΓ_2_dt).*0.5
        dΓ_3_dt_n = (dΓ_3_dt_n+dΓ_3_dt).*0.5
        dΣ_dt_n = (dΣ_dt_n+dΣ_dt).*0.5

        ######End average half step#######
        
        @cuda threads=thrds blocks=blks leapforward!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
        W_1_1,W_1_2,W_1_3,W_1_4,
        W_2_1,W_2_2,W_2_3,W_2_4,
        W_3_1,W_3_2,W_3_3,W_3_4,
        Y_1,Y_2,Y_3,Y_4,
        Γ_1,Γ_2,Γ_3,Σ,
        ϕ_1_n,ϕ_2_n,ϕ_3_n,ϕ_4_n,
        W_1_1_n,W_1_2_n,W_1_3_n,W_1_4_n,
        W_2_1_n,W_2_2_n,W_2_3_n,W_2_4_n,
        W_3_1_n,W_3_2_n,W_3_3_n,W_3_4_n,
        Y_1_n,Y_2_n,Y_3_n,Y_4_n,
        Γ_1_n,Γ_2_n,Γ_3_n,Σ_n,
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
        dΓ_1_dt_n,dΓ_2_dt_n,dΓ_3_dt_n,dΣ_dt_n,
        gw,gy,gp2,vev,lambda,dx,dt)
        
        synchronize()

        #Average half step
        ϕ_1_n = (ϕ_1_n + ϕ_1).*0.5
        ϕ_2_n = (ϕ_2_n + ϕ_2).*0.5
        ϕ_3_n = (ϕ_3_n + ϕ_3).*0.5
        ϕ_4_n = (ϕ_4_n + ϕ_4).*0.5
        W_1_2_n = (W_1_2_n+W_1_2).*0.5
        W_1_3_n = (W_1_3_n+W_1_4).*0.5
        W_1_4_n = (W_1_4_n+W_1_3).*0.5
        W_2_2_n = (W_2_2_n+W_2_2).*0.5
        W_2_3_n = (W_2_3_n+W_2_4).*0.5
        W_2_4_n = (W_2_4_n+W_2_3).*0.5
        W_3_2_n = (W_3_2_n+W_3_2).*0.5
        W_3_3_n = (W_3_3_n+W_3_4).*0.5
        W_3_4_n = (W_3_4_n+W_3_3).*0.5
        Y_2_n = (Y_2_n+Y_2).*0.5
        Y_3_n = (Y_3_n+Y_3).*0.5
        Y_4_n = (Y_4_n+Y_4).*0.5
        Γ_1_n = (Γ_1_n+Γ_1).*0.5
        Γ_2_n = (Γ_2_n+Γ_2).*0.5
        Γ_3_n = (Γ_3_n+Γ_3).*0.5
        Σ_n = (Σ_n+Σ).*0.5

        dϕ_1_dt_n = (dϕ_1_dt_n + dϕ_1_dt).*0.5
        dϕ_2_dt_n = (dϕ_2_dt_n + dϕ_2_dt).*0.5
        dϕ_3_dt_n = (dϕ_3_dt_n + dϕ_3_dt).*0.5
        dϕ_4_dt_n = (dϕ_4_dt_n + dϕ_4_dt).*0.5
        dW_1_2_dt_n = (dW_1_2_dt_n+dW_1_2_dt).*0.5
        dW_1_3_dt_n = (dW_1_3_dt_n+dW_1_4_dt).*0.5
        dW_1_4_dt_n = (dW_1_4_dt_n+dW_1_3_dt).*0.5
        dW_2_2_dt_n = (dW_2_2_dt_n+dW_2_2_dt).*0.5
        dW_2_3_dt_n = (dW_2_3_dt_n+dW_2_4_dt).*0.5
        dW_2_4_dt_n = (dW_2_4_dt_n+dW_2_3_dt).*0.5
        dW_3_2_dt_n = (dW_3_2_dt_n+dW_3_2_dt).*0.5
        dW_3_3_dt_n = (dW_3_3_dt_n+dW_3_4_dt).*0.5
        dW_3_4_dt_n = (dW_3_4_dt_n+dW_3_3_dt).*0.5
        dY_2_dt_n = (dY_2_dt_n+dY_2_dt).*0.5
        dY_3_dt_n = (dY_3_dt_n+dY_3_dt).*0.5
        dY_4_dt_n = (dY_4_dt_n+dY_4_dt).*0.5
        dΓ_1_dt_n = (dΓ_1_dt_n+dΓ_1_dt).*0.5
        dΓ_2_dt_n = (dΓ_2_dt_n+dΓ_2_dt).*0.5
        dΓ_3_dt_n = (dΓ_3_dt_n+dΓ_3_dt).*0.5
        dΣ_dt_n = (dΣ_dt_n+dΣ_dt).*0.5

        ######End average half step#######

        @cuda threads=thrds blocks=blks leapforward!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
        W_1_1,W_1_2,W_1_3,W_1_4,
        W_2_1,W_2_2,W_2_3,W_2_4,
        W_3_1,W_3_2,W_3_3,W_3_4,
        Y_1,Y_2,Y_3,Y_4,
        Γ_1,Γ_2,Γ_3,Σ,
        ϕ_1_n,ϕ_2_n,ϕ_3_n,ϕ_4_n,
        W_1_1_n,W_1_2_n,W_1_3_n,W_1_4_n,
        W_2_1_n,W_2_2_n,W_2_3_n,W_2_4_n,
        W_3_1_n,W_3_2_n,W_3_3_n,W_3_4_n,
        Y_1_n,Y_2_n,Y_3_n,Y_4_n,
        Γ_1_n,Γ_2_n,Γ_3_n,Σ_n,
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
        dΓ_1_dt_n,dΓ_2_dt_n,dΓ_3_dt_n,dΣ_dt_n,
        gw,gy,gp2,vev,lambda,dx,dt)
        
        synchronize()

        #Name change
        ϕ_1=ϕ_1_n
        ϕ_2=ϕ_2_n
        ϕ_3=ϕ_3_n
        ϕ_4=ϕ_4_n
        W_1_1=W_1_1_n
        W_1_2=W_1_2_n
        W_1_3=W_1_3_n
        W_1_4=W_1_4_n
        W_2_1=W_2_1_n
        W_2_2=W_2_2_n
        W_2_3=W_2_3_n
        W_2_4=W_2_4_n
        W_3_1=W_3_1_n
        W_3_2=W_3_2_n
        W_3_3=W_3_3_n
        W_3_4=W_3_4_n
        Y_1=Y_1_n
        Y_2=Y_2_n
        Y_3=Y_3_n
        Y_4=Y_4_n
        ϕ_1_n=ϕ_1
        ϕ_2_n=ϕ_2
        ϕ_3_n=ϕ_3
        ϕ_4_n=ϕ_4
        W_1_1_n=W_1_1
        W_1_2_n=W_1_2
        W_1_3_n=W_1_3
        W_1_4_n=W_1_4
        W_2_1_n=W_2_1
        W_2_2_n=W_2_2
        W_2_3_n=W_2_3
        W_2_4_n=W_2_4
        W_3_1_n=W_3_1
        W_3_2_n=W_3_2
        W_3_3_n=W_3_3
        W_3_4_n=W_3_4
        Y_1_n=Y_1
        Y_2_n=Y_2
        Y_3_n=Y_3
        Y_4_n=Y_4
        
        # println("test:",Array(ϕ_1_n)[3,5,3])
        # exit()
    end
    println("test:",Array(ϕ_1)[3,5,3])
    # println(size(ϕ_2))
    # println(size(findall(Array(ϕ_1_n).!=0)))
    # println(Array(out)[5,1,1])
    return
    
end
run()
CUDA.memory_status()
