module ev_spatial

export evolve_sp!
function evolve_sp!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_1,W_1_2,W_1_3,W_1_4,
    W_2_1,W_2_2,W_2_3,W_2_4,
    W_3_1,W_3_2,W_3_3,W_3_4,
    Y_1,Y_2,Y_3,Y_4,
    Γ_1,Γ_2,Γ_3,Σ,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    ϕ_1_n,ϕ_2_n,ϕ_3_n,ϕ_4_n,
    W_1_1_n,W_1_2_n,W_1_3_n,W_1_4_n,
    W_2_1_n,W_2_2_n,W_2_3_n,W_2_4_n,
    W_3_1_n,W_3_2_n,W_3_3_n,W_3_4_n,
    Y_1_n,Y_2_n,Y_3_n,Y_4_n,
    Γ_1_n,Γ_2_n,Γ_3_n,Σ_n,
    Nx,Ny,Nz,
    gw,gy,vev,lambda,dx,dt)
    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
              (blockIdx().y - 1) * blockDim().y + threadIdx().y,
              (blockIdx().z - 1) * blockDim().z + threadIdx().z
    # @inbounds ϕ_2[i,j,k] = d_xa(ϕ_1)[i,j,k]
    # if (i>2 && i<(size(ϕ_2,1)-2) && j>2 && j<(size(ϕ_2,2)-2) && k>2 && k<(size(ϕ_2,3)-2))
    #     a = diffx_inn(ϕ_2,i,j,k)
    #     b = diffx_inn(ϕ_1,i,j,k)
    #     # a = ϕ_1[i,j,k]-ϕ_1[i+1,j,k]
    #     # b = ϕ_2[i,j,k]-ϕ_2[i+1,j,k]
    #     @inbounds out[i,j,k] = a + b
    #     # @inbounds ϕ_2[i,j,k] = ϕ_1[i,j,k]
    # elseif (i==2 || i==(size(ϕ_2,1)-1))
    #     a = diffx_2(ϕ_2,i,j,k)
    #     b = diffx_2(ϕ_1,i,j,k)
    #     # a = ϕ_1[i,j,k]-ϕ_1[i+1,j,k]
    #     # b = ϕ_2[i,j,k]-ϕ_2[i+1,j,k]
    #     @inbounds out[i,j,k] = a + b
    # elseif i==(size(ϕ_2,1))
    #     a = diffx_bound(ϕ_2,i,j,k)
    #     b = diffx_bound(ϕ_1,i,j,k)
    #     # a = ϕ_1[i,j,k]-ϕ_1[i+1,j,k]
    #     # b = ϕ_2[i,j,k]-ϕ_2[i+1,j,k]
    #     @inbounds out[i,j,k] = a + b
    # end
    # a_x,a_y,a_z = [0.,0.,0.]
    # b_x,b_y,b_z = [0.,0.,0.]
    # a0 = [0.,0.,0.]
    # b0 = [0.,0.,0.]
    
    dϕ_1_dx=dfdx(ϕ_1,i,j,k,0.,1)
    dϕ_2_dx=dfdx(ϕ_2,i,j,k,0.,1)
    dϕ_3_dx=dfdx(ϕ_2,i,j,k,0.,1)
    dϕ_4_dx=dfdx(ϕ_2,i,j,k,0.,1)
    # @cuprintln(dϕ_4_dx)
    dϕ_1_dy=dfdy(ϕ_1,i,j,k,0.,1)
    dϕ_2_dy=dfdy(ϕ_2,i,j,k,0.,1)
    dϕ_3_dy=dfdy(ϕ_2,i,j,k,0.,1)
    dϕ_4_dy=dfdy(ϕ_2,i,j,k,0.,1)

    dϕ_1_dz=dfdz(ϕ_1,i,j,k,0.,1)
    dϕ_2_dz=dfdz(ϕ_2,i,j,k,0.,1)
    dϕ_3_dz=dfdz(ϕ_2,i,j,k,0.,1)
    dϕ_4_dz=dfdz(ϕ_2,i,j,k,0.,1)

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
    @inbounds ϕ_1_n[i,j,k] = (ϕ_1[i,j,k]+dt*(d2ϕ_1_dx2+d2ϕ_1_dy2+d2ϕ_1_dz2-
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
    @inbounds ϕ_2_n[i,j,k] = (ϕ_2[i,j,k]+dt*(d2ϕ_2_dx2+d2ϕ_2_dy2+d2ϕ_2_dz2+
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
    @inbounds ϕ_3_n[i,j,k] = (ϕ_3[i,j,k]+dt*(d2ϕ_3_dx2+d2ϕ_3_dy2+d2ϕ_3_dz2-
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
    @inbounds ϕ_4_n[i,j,k] = (ϕ_4[i,j,k]+dt*(d2ϕ_4_dx2+d2ϕ_4_dy2+d2ϕ_4_dz2+
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
    @inbounds W_1_2_n[i,j,k] = (W_1_2[i,j,k]+dt*(d2W_1_2_dx2+d2W_1_2_dy2+d2W_1_2_dz2+
    gw*(-(dW_2_2_dx*W_3_2[i,j,k]-dW_3_2_dx*W_2_2[i,j,k])-
        (dW_2_2_dy*W_3_3[i,j,k]-dW_3_2_dy*W_2_3[i,j,k])-
        (dW_2_2_dz*W_3_4[i,j,k]-dW_3_2_dz*W_2_4[i,j,k])-
        (W_2_3[i,j,k]*W_3_23-W_3_3[i,j,k]*W_2_23)-
        (W_2_4[i,j,k]*W_3_24-W_3_4[i,j,k]*W_2_24))+
    gw*(ϕ_1[i,j,k]*Dx_ϕ_4-ϕ_2[i,j,k]*Dx_ϕ_3+ϕ_3[i,j,k]*Dx_ϕ_2-ϕ_4[i,j,k]*Dx_ϕ_1)-
    dΓ_1_dx-gw*(W_2_2[i,j,k]*Γ_3[i,j,k]-W_3_2[i,j,k]*Γ_2[i,j,k])-
    gw*(W_2_1[i,j,k]*dW_3_2_dt[i,j,k]-W_3_1[i,j,k]*dW_2_2_dt[i,j,k])))

#     R_W_1_3=
    @inbounds W_1_3_n[i,j,k] = (W_1_3[i,j,k]+dt*(d2W_1_3_dx2+d2W_1_3_dy2+d2W_1_3_dz2+
    gw*(-(dW_2_3_dx*W_3_2[i,j,k]-dW_3_3_dx*W_2_2[i,j,k])-
        (dW_2_3_dy*W_3_3[i,j,k]-dW_3_3_dy*W_2_3[i,j,k])-
        (dW_2_3_dz*W_3_4[i,j,k]-dW_3_3_dz*W_2_4[i,j,k])-
        (W_2_2[i,j,k]*(-W_3_23)-W_3_2[i,j,k]*(-W_2_23))-
        (W_2_4[i,j,k]*W_3_34-W_3_4[i,j,k]*W_2_34))+
    gw*(ϕ_1[i,j,k]*Dy_ϕ_4-ϕ_2[i,j,k]*Dy_ϕ_3+ϕ_3[i,j,k]*Dy_ϕ_2-ϕ_4[i,j,k]*Dy_ϕ_1)-
    dΓ_1_dy-gw*(W_2_3[i,j,k]*Γ_3[i,j,k]-W_3_3[i,j,k]*Γ_2[i,j,k])-
    gw*(W_2_1[i,j,k]*dW_3_3_dt[i,j,k]-W_3_1[i,j,k]*dW_2_3_dt[i,j,k])))

#     R_W_1_4=
    @inbounds W_1_4_n[i,j,k] = (W_1_4[i,j,k]+dt*(d2W_1_4_dx2+d2W_1_4_dy2+d2W_1_4_dz2+
    gw*(-(dW_2_4_dx*W_3_2[i,j,k]-dW_3_4_dx*W_2_2[i,j,k])-
        (dW_2_4_dy*W_3_3[i,j,k]-dW_3_4_dy*W_2_3[i,j,k])-
        (dW_2_4_dz*W_3_4[i,j,k]-dW_3_4_dz*W_2_4[i,j,k])-
        (W_2_2[i,j,k]*(-W_3_24)-W_3_2[i,j,k]*(-W_2_24))-
        (W_2_3[i,j,k]*(-W_3_34)-W_3_3[i,j,k]*(-W_2_34)))+
    gw*(ϕ_1[i,j,k]*Dz_ϕ_4-ϕ_2[i,j,k]*Dz_ϕ_3+ϕ_3[i,j,k]*Dz_ϕ_2-ϕ_4[i,j,k]*Dz_ϕ_1)-
    dΓ_1_dz-gw*(W_2_4[i,j,k]*Γ_3[i,j,k]-W_3_4[i,j,k]*Γ_2[i,j,k])-
    gw*(W_2_1[i,j,k]*dW_3_4_dt[i,j,k]-W_3_1[i,j,k]*dW_2_4_dt[i,j,k])))

#     R_W_2_2=
    @inbounds W_2_2_n[i,j,k] = (W_2_2[i,j,k]+dt*(d2W_2_2_dx2+d2W_2_2_dy2+d2W_2_2_dz2+
    gw*(-(dW_3_2_dx*W_1_2[i,j,k]-dW_1_2_dx*W_3_2[i,j,k])-
        (dW_3_2_dy*W_1_3[i,j,k]-dW_1_2_dy*W_3_3[i,j,k])-
        (dW_3_2_dz*W_1_4[i,j,k]-dW_1_2_dz*W_3_4[i,j,k])-
        (W_3_3[i,j,k]*W_1_23-W_1_3[i,j,k]*W_3_23)-
        (W_3_4[i,j,k]*W_1_24-W_1_4[i,j,k]*W_3_24))+
    gw*(-ϕ_1[i,j,k]*Dx_ϕ_3-ϕ_2[i,j,k]*Dx_ϕ_4+ϕ_3[i,j,k]*Dx_ϕ_1+ϕ_4[i,j,k]*Dx_ϕ_2)-
    dΓ_2_dx-gw*(W_3_2[i,j,k]*Γ_1[i,j,k]-W_1_2[i,j,k]*Γ_3[i,j,k])-
    gw*(W_3_1[i,j,k]*dW_1_2_dt[i,j,k]-W_1_1[i,j,k]*dW_3_2_dt[i,j,k])))

#     R_W_2_3=
    @inbounds W_2_3_n[i,j,k] = (W_2_3[i,j,k]+dt*(d2W_2_3_dx2+d2W_2_3_dy2+d2W_2_3_dz2+
    gw*(-(dW_3_3_dx*W_1_2[i,j,k]-dW_1_3_dx*W_3_2[i,j,k])-
        (dW_3_3_dy*W_1_3[i,j,k]-dW_1_3_dy*W_3_3[i,j,k])-
        (dW_3_3_dz*W_1_4[i,j,k]-dW_1_3_dz*W_3_4[i,j,k])-
        (W_3_2[i,j,k]*(-W_1_23)-W_1_2[i,j,k]*(-W_3_23))-
        (W_3_4[i,j,k]*(W_1_34)-W_1_4[i,j,k]*W_3_34))+
    gw*(-ϕ_1[i,j,k]*Dy_ϕ_3-ϕ_2[i,j,k]*Dy_ϕ_4+ϕ_3[i,j,k]*Dy_ϕ_1+ϕ_4[i,j,k]*Dy_ϕ_2)-
    dΓ_2_dy-gw*(W_3_3[i,j,k]*Γ_1[i,j,k]-W_1_3[i,j,k]*Γ_3[i,j,k])-
    gw*(W_3_1[i,j,k]*dW_1_3_dt[i,j,k]-W_1_1[i,j,k]*dW_3_3_dt[i,j,k])))

#     R_W_2_4=
    @inbounds W_2_4_n[i,j,k] = (W_2_4[i,j,k]+dt*(d2W_2_4_dx2+d2W_2_4_dy2+d2W_2_4_dz2+
    gw*(-(dW_3_4_dx*W_1_2[i,j,k]-dW_1_4_dx*W_3_2[i,j,k])-
        (dW_3_4_dy*W_1_3[i,j,k]-dW_1_4_dy*W_3_3[i,j,k])-
        (dW_3_4_dz*W_1_4[i,j,k]-dW_1_4_dz*W_3_4[i,j,k])-
        (W_3_2[i,j,k]*(-W_1_24)-W_1_2[i,j,k]*(-W_3_24))-
        (W_3_3[i,j,k]*(-W_1_34)-W_1_3[i,j,k]*(-W_3_34)))+
    gw*(-ϕ_1[i,j,k]*Dz_ϕ_3-ϕ_2[i,j,k]*Dz_ϕ_4+ϕ_3[i,j,k]*Dz_ϕ_1+ϕ_4[i,j,k]*Dz_ϕ_2)-
    dΓ_2_dz-gw*(W_3_4[i,j,k]*Γ_1[i,j,k]-W_1_4[i,j,k]*Γ_3[i,j,k])-
    gw*(W_3_1[i,j,k]*dW_1_4_dt[i,j,k]-W_1_1[i,j,k]*dW_3_4_dt[i,j,k])))

#     R_W_3_2=
    @inbounds W_3_2_n[i,j,k] = (W_3_2[i,j,k]+dt*(d2W_3_2_dx2+d2W_3_2_dy2+d2W_3_2_dz2+
    gw*(-(dW_1_2_dx*W_2_2[i,j,k]-dW_2_2_dx*W_1_2[i,j,k])-
        (dW_1_2_dy*W_2_3[i,j,k]-dW_2_2_dy*W_1_3[i,j,k])-
        (dW_1_2_dz*W_2_4[i,j,k]-dW_2_2_dz*W_1_4[i,j,k])-
        (W_1_3[i,j,k]*W_2_23-W_2_3[i,j,k]*W_1_23)-
        (W_1_4[i,j,k]*W_2_24-W_2_4[i,j,k]*W_1_24))+
    gw*(ϕ_1[i,j,k]*Dx_ϕ_2-ϕ_2[i,j,k]*Dx_ϕ_1-ϕ_3[i,j,k]*Dx_ϕ_4+ϕ_4[i,j,k]*Dx_ϕ_3)-
    dΓ_3_dx-gw*(W_1_2[i,j,k]*Γ_2[i,j,k]-W_2_2[i,j,k]*Γ_1[i,j,k])-
    gw*(W_1_1[i,j,k]*dW_2_2_dt[i,j,k]-W_2_1[i,j,k]*dW_1_2_dt[i,j,k])))

#     R_W_3_3=
    @inbounds W_3_3_n[i,j,k] = (W_3_3[i,j,k]+dt*(d2W_3_3_dx2+d2W_3_3_dy2+d2W_3_3_dz2+
    gw*(-(dW_1_3_dx*W_2_2[i,j,k]-dW_2_3_dx*W_1_2[i,j,k])-
        (dW_1_3_dy*W_2_3[i,j,k]-dW_2_3_dy*W_1_3[i,j,k])-
        (dW_1_3_dz*W_2_4[i,j,k]-dW_2_3_dz*W_1_4[i,j,k])-
        (W_1_2[i,j,k]*(-W_2_23)-W_2_2[i,j,k]*(-W_1_23))-
        (W_1_4[i,j,k]*W_2_34-W_2_4[i,j,k]*(W_1_34)))+
    gw*(ϕ_1[i,j,k]*Dy_ϕ_2-ϕ_2[i,j,k]*Dy_ϕ_1-ϕ_3[i,j,k]*Dy_ϕ_4+ϕ_4[i,j,k]*Dy_ϕ_3)-
    dΓ_3_dy-gw*(W_1_3[i,j,k]*Γ_2[i,j,k]-W_2_3[i,j,k]*Γ_1[i,j,k])-
    gw*(W_1_1[i,j,k]*dW_2_3_dt[i,j,k]-W_2_1[i,j,k]*dW_1_3_dt[i,j,k])))

#     R_W_3_4=
    @inbounds W_3_4_n[i,j,k] = (W_3_4[i,j,k]+dt*(d2W_3_4_dx2+d2W_3_4_dy2+d2W_3_4_dz2+
    gw*(-(dW_1_4_dx*W_2_2[i,j,k]-dW_2_4_dx*W_1_2[i,j,k])-
        (dW_1_4_dy*W_2_3[i,j,k]-dW_2_4_dy*W_1_3[i,j,k])-
        (dW_1_4_dz*W_2_4[i,j,k]-dW_2_4_dz*W_1_4[i,j,k])-
        (W_1_2[i,j,k]*(-W_2_24)-W_2_2[i,j,k]*(-W_1_24))-
        (W_1_3[i,j,k]*(-W_2_34)-W_2_3[i,j,k]*(-W_1_34)))+
    gw*(ϕ_1[i,j,k]*Dz_ϕ_2-ϕ_2[i,j,k]*Dz_ϕ_1-ϕ_3[i,j,k]*Dz_ϕ_4+ϕ_4[i,j,k]*Dz_ϕ_3)-
    dΓ_3_dz-gw*(W_1_4[i,j,k]*Γ_2[i,j,k]-W_2_4[i,j,k]*Γ_1[i,j,k])-
    gw*(W_1_1[i,j,k]*dW_2_4_dt[i,j,k]-W_2_1[i,j,k]*dW_1_4_dt[i,j,k])))

#     R_Y_2=
    @inbounds Y_2_n[i,j,k] = (Y_2[i,j,k]+dt*(d2Y_2_dx2+d2Y_2_dy2+d2Y_2_dz2+
    gy*(ϕ_1[i,j,k]*Dx_ϕ_2-ϕ_2[i,j,k]*Dx_ϕ_1+ϕ_3[i,j,k]*Dx_ϕ_4-ϕ_4[i,j,k]*Dx_ϕ_3)-dΣ_dx))

#     R_Y_3=
    @inbounds Y_3_n[i,j,k] = (Y_3[i,j,k]+dt*(d2Y_3_dx2+d2Y_3_dy2+d2Y_3_dz2+
    gy*(ϕ_1[i,j,k]*Dy_ϕ_2-ϕ_2[i,j,k]*Dy_ϕ_1+ϕ_3[i,j,k]*Dy_ϕ_4-ϕ_4[i,j,k]*Dy_ϕ_3)-dΣ_dy))

#     R_Y_4=
    @inbounds Y_4_n[i,j,k] = (Y_4[i,j,k]+dt*(d2Y_4_dx2+d2Y_4_dy2+d2Y_4_dz2+
    gy*(ϕ_1[i,j,k]*Dz_ϕ_2-ϕ_2[i,j,k]*Dz_ϕ_1+ϕ_3[i,j,k]*Dz_ϕ_4-ϕ_4[i,j,k]*Dz_ϕ_3)-dΣ_dz))


    # a = ϕ_1[i,j,k]-ϕ_1[i+1,j,k]
    # b = ϕ_2[i,j,k]-ϕ_2[i+1,j,k]
    # x,y,z=cartesian_symm(i,j,k,0,0,0,Nx,Ny,Nz)
    # @inbounds Σ[i,j,k] = dϕ_1_dx + dϕ_2_dx + Dx_ϕ_1
    return
end

end
