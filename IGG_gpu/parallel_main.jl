####---CPU/GPU evolution code tester for multiple devices
####---Used to test initializing and evolving over multiple deives
###July 23: ParallelStencil and ImplicitGlobalGrid based codes
###--------are completely functional with the correct
###--------custom instantiation,gathering and plotting routines
###--------Important Note: Need to employ MPI barriers manually
###--------when executing single process routines like 
###--------plotting and outputting data arrays. This has been tested and 
###--------funcitoning correctly.

const USE_GPU = false
using ImplicitGlobalGrid
import MPI
using ParallelStencil
using ParallelStencil.FiniteDifferences3D

include("parameters.jl")
using .parameters

include("random_ini_gen.jl")
using .randomizer

using Plots
pythonplot()

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3);
else
    @init_parallel_stencil(Threads, Float64, 3);
end

@parallel function diffusion3D_step!(T2, T, Ci, lam, dt, dx, dy, dz)
    @inn(T2) = @inn(T) + dt*(lam*@inn(Ci)*(@d2_xi(T)/dx^2 + @d2_yi(T)/dy^2 + @d2_zi(T)/dz^2));
    return
end

@parallel function cov_derivs!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_1,W_1_2,W_1_3,W_1_4,
    W_2_1,W_2_2,W_2_3,W_2_4,
    W_3_1,W_3_2,W_3_3,W_3_4,
    Y_1,Y_2,Y_3,Y_4,
    D_1ϕ_1,D_1ϕ_2,D_1ϕ_3,D_1ϕ_4,
    D_2ϕ_1,D_2ϕ_2,D_2ϕ_3,D_2ϕ_4,
    D_3ϕ_1,D_3ϕ_2,D_3ϕ_3,D_3ϕ_4,
    D_4ϕ_1,D_4ϕ_2,D_4ϕ_3,D_4ϕ_4)
    # ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_1,W_1_2,W_1_3,W_1_4,W_2_1,W_2_2,W_2_3,W_2_4,W_3_1,W_3_2,W_3_3,W_3_4,Y_1,Y_2,Y_3,Y_4=f
    # @inn(D_1ϕ_1) = 0.5*gw*(@inn(W_1_1)*@inn(ϕ_1)-@inn(W_2_1)*@inn(ϕ_3)+@inn(W_3_1)*@inn(ϕ_2))+0.5*gy*(@inn(Y_1)*@inn(ϕ_2))
    @inn(D_1ϕ_1) =  0.5*gw*(@inn(W_1_1)*@inn(ϕ_4)-@inn(W_2_1)*@inn(ϕ_3)+@inn(W_3_1)*@inn(ϕ_2))+0.5*gy*(@inn(Y_1)*@inn(ϕ_2))
    @inn(D_1ϕ_2) = -0.5*gw*(@inn(W_1_1)*@inn(ϕ_3)+@inn(W_2_1)*@inn(ϕ_4)+@inn(W_3_1)*@inn(ϕ_1))-0.5*gy*(@inn(Y_1)*@inn(ϕ_1))
    @inn(D_1ϕ_3) =  0.5*gw*(@inn(W_1_1)*@inn(ϕ_2)+@inn(W_2_1)*@inn(ϕ_1)-@inn(W_3_1)*@inn(ϕ_4))+0.5*gy*(@inn(Y_1)*@inn(ϕ_4))
    @inn(D_1ϕ_4) = -0.5*gw*(@inn(W_1_1)*@inn(ϕ_1)-@inn(W_2_1)*@inn(ϕ_2)-@inn(W_3_1)*@inn(ϕ_3))-0.5*gy*(@inn(Y_1)*@inn(ϕ_3))
    @inn(D_2ϕ_1) = @d_xi(ϕ_1) + 0.5*gw*(@inn(W_1_2)*@inn(ϕ_4)-@inn(W_2_2)*@inn(ϕ_3)+@inn(W_3_2)*@inn(ϕ_2))+0.5*gy*(@inn(Y_2)*@inn(ϕ_2))
    @inn(D_2ϕ_2) = @d_xi(ϕ_2) - 0.5*gw*(@inn(W_1_2)*@inn(ϕ_3)+@inn(W_2_2)*@inn(ϕ_4)+@inn(W_3_2)*@inn(ϕ_1))-0.5*gy*(@inn(Y_2)*@inn(ϕ_1))
    @inn(D_2ϕ_3) = @d_xi(ϕ_3) + 0.5*gw*(@inn(W_1_2)*@inn(ϕ_2)+@inn(W_2_2)*@inn(ϕ_1)-@inn(W_3_2)*@inn(ϕ_4))+0.5*gy*(@inn(Y_2)*@inn(ϕ_4))
    @inn(D_2ϕ_4) = @d_xi(ϕ_4) - 0.5*gw*(@inn(W_1_2)*@inn(ϕ_1)-@inn(W_2_2)*@inn(ϕ_2)-@inn(W_3_2)*@inn(ϕ_3))-0.5*gy*(@inn(Y_2)*@inn(ϕ_3))
    @inn(D_3ϕ_1) = @d_yi(ϕ_1) + 0.5*gw*(@inn(W_1_3)*@inn(ϕ_4)-@inn(W_2_3)*@inn(ϕ_3)+@inn(W_3_3)*@inn(ϕ_2))+0.5*gy*(@inn(Y_3)*@inn(ϕ_2))
    @inn(D_3ϕ_2) = @d_yi(ϕ_2) - 0.5*gw*(@inn(W_1_3)*@inn(ϕ_3)+@inn(W_2_3)*@inn(ϕ_4)+@inn(W_3_3)*@inn(ϕ_1))-0.5*gy*(@inn(Y_3)*@inn(ϕ_1))
    @inn(D_3ϕ_3) = @d_yi(ϕ_3) + 0.5*gw*(@inn(W_1_3)*@inn(ϕ_2)+@inn(W_2_3)*@inn(ϕ_1)-@inn(W_3_3)*@inn(ϕ_4))+0.5*gy*(@inn(Y_3)*@inn(ϕ_4))
    @inn(D_3ϕ_4) = @d_yi(ϕ_4) - 0.5*gw*(@inn(W_1_3)*@inn(ϕ_1)-@inn(W_2_3)*@inn(ϕ_2)-@inn(W_3_3)*@inn(ϕ_3))-0.5*gy*(@inn(Y_3)*@inn(ϕ_3))
    @inn(D_4ϕ_1) = @d_zi(ϕ_1) + 0.5*gw*(@inn(W_1_4)*@inn(ϕ_4)-@inn(W_2_4)*@inn(ϕ_3)+@inn(W_3_4)*@inn(ϕ_2))+0.5*gy*(@inn(Y_4)*@inn(ϕ_2))
    @inn(D_4ϕ_2) = @d_zi(ϕ_2) - 0.5*gw*(@inn(W_1_4)*@inn(ϕ_3)+@inn(W_2_4)*@inn(ϕ_4)+@inn(W_3_4)*@inn(ϕ_1))-0.5*gy*(@inn(Y_4)*@inn(ϕ_1))
    @inn(D_4ϕ_3) = @d_zi(ϕ_3) + 0.5*gw*(@inn(W_1_4)*@inn(ϕ_2)+@inn(W_2_4)*@inn(ϕ_1)-@inn(W_3_4)*@inn(ϕ_4))+0.5*gy*(@inn(Y_4)*@inn(ϕ_4))
    @inn(D_4ϕ_4) = @d_zi(ϕ_4) - 0.5*gw*(@inn(W_1_4)*@inn(ϕ_1)-@inn(W_2_4)*@inn(ϕ_2)-@inn(W_3_4)*@inn(ϕ_3))-0.5*gy*(@inn(Y_4)*@inn(ϕ_3))
    
    return
end

@parallel function field_strengths!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_1,W_1_2,W_1_3,W_1_4,
    W_2_1,W_2_2,W_2_3,W_2_4,
    W_3_1,W_3_2,W_3_3,W_3_4,
    Y_1,Y_2,Y_3,Y_4,
    D_1ϕ_1,D_1ϕ_2,D_1ϕ_3,D_1ϕ_4,
    D_2ϕ_1,D_2ϕ_2,D_2ϕ_3,D_2ϕ_4,
    D_3ϕ_1,D_3ϕ_2,D_3ϕ_3,D_3ϕ_4,
    D_4ϕ_1,D_4ϕ_2,D_4ϕ_3,D_4ϕ_4)

    r[1]=laplacian[1]
      -0.5*gw*((-g[6]*dfdx[4]-g[7]*dfdy[4]-g[8]*dfdz[4])
              -(-g[10]*dfdx[3]-g[11]*dfdy[3]-g[12]*dfdz[3])
              +(-g[14]*dfdx[2]-g[15]*dfdy[2]-g[16]*dfdz[2]))
      -0.5*gy*(-g[18]*dfdx[2]-g[19]*dfdy[2]-g[20]*dfdz[2])
      -0.5*gw*((-g[6]*cd[2,4]-g[7]*cd[3,4]-g[8]*cd[4,4])
          -(-g[10]*cd[2,3]-g[11]*cd[3,3]-g[12]*cd[4,3])
          +(-g[14]*cd[2,2]-g[15]*cd[3,2]-g[16]*cd[4,2]))
      -0.5*gy*(-g[18]*cd[2,2]-g[19]*cd[3,2]-g[20]*cd[4,2])
      -2.0*lambda*(g[1]^2+g[2]^2+g[3]^2+g[4]^2-vev^2)*g[1]
      +0.5*((gw*g[23]+gy*g[24])*g[2]-gw*g[22]*g[3]+gw*g[21]*g[4])

    r[2]=laplacian[2]
      +0.5*gw*((-g[6]*dfdx[3]-g[7]*dfdy[3]-g[8]*dfdz[3])
          +(-g[10]*dfdx[4]-g[11]*dfdy[4]-g[12]*dfdz[4])
          +(-g[14]*dfdx[1]-g[15]*dfdy[1]-g[16]*dfdz[1]))
      +0.5*gy*(-g[18]*dfdx[1]-g[19]*dfdy[1]-g[20]*dfdz[1])
      +0.5*gw*((-g[6]*cd[2,3]-g[7]*cd[3,3]-g[8]*cd[4,3])
          +(-g[10]*cd[2,4]-g[11]*cd[3,4]-g[12]*cd[4,4])
          +(-g[14]*cd[2,1]-g[15]*cd[3,1]-g[16]*cd[4,1]))
      +0.5*gy*(-g[18]*cd[2,1]-g[19]*cd[3,1]-g[20]*cd[4,1])
      -2.0*lambda*(g[1]^2+g[2]^2+g[3]^2+g[4]^2-vev^2)*g[2]
      -0.5*((gw*g[23]+gy*g[24])*g[1]+gw*g[21]*g[3]+gw*g[22]*g[4])

    r[3]=laplacian[3]
      -0.5*gw*((-g[6]*dfdx[2]-g[7]*dfdy[2]-g[8]*dfdz[2])
          +(-g[10]*dfdx[1]-g[11]*dfdy[1]-g[12]*dfdz[1])
          -(-g[14]*dfdx[4]-g[15]*dfdy[4]-g[16]*dfdz[4]))
      -0.5*gy*(-g[18]*dfdx[4]-g[19]*dfdy[4]-g[20]*dfdz[4])
      -0.5*gw*((-g[6]*cd[2,2]-g[7]*cd[3,2]-g[8]*cd[4,2])
          +(-g[10]*cd[2,1]-g[11]*cd[3,1]-g[12]*cd[4,1])
          -(-g[14]*cd[2,4]-g[15]*cd[3,4]-g[16]*cd[4,4]))
      -0.5*gy*(-g[18]*cd[2,4]-g[19]*cd[3,4]-g[20]*cd[4,4])
      -2.0*lambda*(g[1]^2+g[2]^2+g[3]^2+g[4]^2-vev^2)*g[3]
      +0.5*((-gw*g[23]+gy*g[24])*g[4]+gw*g[22]*g[1]+gw*g[21]*g[2])

    r[4]=laplacian[4]
      +0.5*gw*((-g[6]*dfdx[1]-g[7]*dfdy[1]-g[8]*dfdz[1])
          -(-g[10]*dfdx[2]-g[11]*dfdy[2]-g[12]*dfdz[2])
          -(-g[14]*dfdx[3]-g[15]*dfdy[3]-g[16]*dfdz[3]))
      +0.5*gy*(-g[18]*dfdx[3]-g[19]*dfdy[3]-g[20]*dfdz[3])
      +0.5*gw*((-g[6]*cd[2,1]-g[7]*cd[3,1]-g[8]*cd[4,1])
          -(-g[10]*cd[2,2]-g[11]*cd[3,2]-g[12]*cd[4,2])
          -(-g[14]*cd[2,3]-g[15]*cd[3,3]-g[16]*cd[4,3]))
      +0.5*gy*(-g[18]*cd[2,3]-g[19]*cd[3,3]-g[20]*cd[4,3])
      -2.0*lambda*(g[1]^2+g[2]^2+g[3]^2+g[4]^2-vev^2)*g[4]
      -0.5*((-gw*g[23]+gy*g[24])*g[3]+gw*g[21]*g[1]-gw*g[22]*g[2])
    return
end

# @parallel_indices (ix,iy,iz) function cov_derivs!(f)
#     ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_1,W_1_2,W_1_3,W_1_4,W_2_1,W_2_2,W_2_3,W_2_4,W_3_1,W_3_2,W_3_3,W_3_4,Y_1,Y_2,Y_3,Y_4=f
#     # @inn(D_1ϕ_1) = 0.5*gw*(@inn(W_1_1)*@inn(ϕ_1)-@inn(W_2_1)*@inn(ϕ_3)+@inn(W_3_1)*@inn(ϕ_2))+0.5*gy*(@inn(Y_1)*@inn(ϕ_2))
#     D_1ϕ_1 = 0.5*gw*((W_1_1[ix,iy,iz])*(ϕ_4[ix,iy,iz])-(W_2_1[ix,iy,iz])*(ϕ_3[ix,iy,iz])+(W_3_1[ix,iy,iz])*(ϕ_2[ix,iy,iz]))+0.5*gy*((Y_1[ix,iy,iz])*(ϕ_2[ix,iy,iz]))
#     D_1ϕ_2 = 0.5*gw*((W_1_1[ix,iy,iz])*(ϕ_3[ix,iy,iz])-(W_2_1[ix,iy,iz])*(ϕ_4[ix,iy,iz])+(W_3_1[ix,iy,iz])*(ϕ_1[ix,iy,iz]))+0.5*gy*((Y_1[ix,iy,iz])*(ϕ_1[ix,iy,iz]))
#     D_1ϕ_3 = 0.5*gw*((W_1_1[ix,iy,iz])*(ϕ_2[ix,iy,iz])-(W_2_1[ix,iy,iz])*(ϕ_1[ix,iy,iz])+(W_3_1[ix,iy,iz])*(ϕ_4[ix,iy,iz]))+0.5*gy*((Y_1[ix,iy,iz])*(ϕ_4[ix,iy,iz]))
#     D_1ϕ_4 = 0.5*gw*((W_1_1[ix,iy,iz])*(ϕ_1[ix,iy,iz])-(W_2_1[ix,iy,iz])*(ϕ_2[ix,iy,iz])+(W_3_1[ix,iy,iz])*(ϕ_3[ix,iy,iz]))+0.5*gy*((Y_1[ix,iy,iz])*(ϕ_3[ix,iy,iz]))
#     D_2ϕ_1 = 0.5*gw*((W_1_2[ix,iy,iz])*(ϕ_4[ix,iy,iz])-(W_2_2[ix,iy,iz])*(ϕ_3[ix,iy,iz])+(W_3_2[ix,iy,iz])*(ϕ_2[ix,iy,iz]))+0.5*gy*((Y_2[ix,iy,iz])*(ϕ_2[ix,iy,iz]))
#     D_2ϕ_2 = 0.5*gw*((W_1_2[ix,iy,iz])*(ϕ_3[ix,iy,iz])-(W_2_2[ix,iy,iz])*(ϕ_4[ix,iy,iz])+(W_3_2[ix,iy,iz])*(ϕ_1[ix,iy,iz]))+0.5*gy*((Y_2[ix,iy,iz])*(ϕ_1[ix,iy,iz]))
#     D_2ϕ_3 = 0.5*gw*((W_1_2[ix,iy,iz])*(ϕ_2[ix,iy,iz])-(W_2_2[ix,iy,iz])*(ϕ_1[ix,iy,iz])+(W_3_2[ix,iy,iz])*(ϕ_4[ix,iy,iz]))+0.5*gy*((Y_2[ix,iy,iz])*(ϕ_4[ix,iy,iz]))
#     D_2ϕ_4 = 0.5*gw*((W_1_2[ix,iy,iz])*(ϕ_1[ix,iy,iz])-(W_2_2[ix,iy,iz])*(ϕ_2[ix,iy,iz])+(W_3_2[ix,iy,iz])*(ϕ_3[ix,iy,iz]))+0.5*gy*((Y_2[ix,iy,iz])*(ϕ_3[ix,iy,iz]))
#     D_3ϕ_1 = 0.5*gw*((W_1_3[ix,iy,iz])*(ϕ_4[ix,iy,iz])-(W_2_3[ix,iy,iz])*(ϕ_3[ix,iy,iz])+(W_3_3[ix,iy,iz])*(ϕ_2[ix,iy,iz]))+0.5*gy*((Y_3[ix,iy,iz])*(ϕ_2[ix,iy,iz]))
#     D_3ϕ_2 = 0.5*gw*((W_1_3[ix,iy,iz])*(ϕ_3[ix,iy,iz])-(W_2_3[ix,iy,iz])*(ϕ_4[ix,iy,iz])+(W_3_3[ix,iy,iz])*(ϕ_1[ix,iy,iz]))+0.5*gy*((Y_3[ix,iy,iz])*(ϕ_1[ix,iy,iz]))
#     D_3ϕ_3 = 0.5*gw*((W_1_3[ix,iy,iz])*(ϕ_2[ix,iy,iz])-(W_2_3[ix,iy,iz])*(ϕ_1[ix,iy,iz])+(W_3_3[ix,iy,iz])*(ϕ_4[ix,iy,iz]))+0.5*gy*((Y_3[ix,iy,iz])*(ϕ_4[ix,iy,iz]))
#     D_3ϕ_4 = 0.5*gw*((W_1_3[ix,iy,iz])*(ϕ_1[ix,iy,iz])-(W_2_3[ix,iy,iz])*(ϕ_2[ix,iy,iz])+(W_3_3[ix,iy,iz])*(ϕ_3[ix,iy,iz]))+0.5*gy*((Y_3[ix,iy,iz])*(ϕ_3[ix,iy,iz]))
#     D_4ϕ_1 = 0.5*gw*((W_1_4[ix,iy,iz])*(ϕ_4[ix,iy,iz])-(W_2_4[ix,iy,iz])*(ϕ_3[ix,iy,iz])+(W_3_4[ix,iy,iz])*(ϕ_2[ix,iy,iz]))+0.5*gy*((Y_4[ix,iy,iz])*(ϕ_2[ix,iy,iz]))
#     D_4ϕ_2 = 0.5*gw*((W_1_4[ix,iy,iz])*(ϕ_3[ix,iy,iz])-(W_2_4[ix,iy,iz])*(ϕ_4[ix,iy,iz])+(W_3_4[ix,iy,iz])*(ϕ_1[ix,iy,iz]))+0.5*gy*((Y_4[ix,iy,iz])*(ϕ_1[ix,iy,iz]))
#     D_4ϕ_3 = 0.5*gw*((W_1_4[ix,iy,iz])*(ϕ_2[ix,iy,iz])-(W_2_4[ix,iy,iz])*(ϕ_1[ix,iy,iz])+(W_3_4[ix,iy,iz])*(ϕ_4[ix,iy,iz]))+0.5*gy*((Y_4[ix,iy,iz])*(ϕ_4[ix,iy,iz]))
#     D_4ϕ_4 = 0.5*gw*((W_1_4[ix,iy,iz])*(ϕ_1[ix,iy,iz])-(W_2_4[ix,iy,iz])*(ϕ_2[ix,iy,iz])+(W_3_4[ix,iy,iz])*(ϕ_3[ix,iy,iz]))+0.5*gy*((Y_4[ix,iy,iz])*(ϕ_3[ix,iy,iz]))

#     W_11 = @zeros(size((W_1_1)))
#     W_1_23 = @d_xi(W_1_3)/dx - @d_yi(W_1_2)/dx + gw*(W_2_2*W_3_3-W_2_3*W_3_2)
#     println(W_1_23);exit()
#     return
# end


function diffusion3D()

# Numerics
nx, ny, nz = latx,laty,latz;                              # Number of gridpoints dimensions x, y and z.

init_global_grid(nx, ny, nz);

# Array initializations
ϕ_1 = @zeros(nx,ny,nz)
ϕ_2 = @zeros(nx,ny,nz)
ϕ_3 = @zeros(nx,ny,nz)
ϕ_4 = @zeros(nx,ny,nz)
W_1_1 = @zeros(nx,ny,nz)
W_1_2 = @zeros(nx,ny,nz)
W_1_3 = @zeros(nx,ny,nz)
W_1_4 = @zeros(nx,ny,nz)
W_2_1 = @zeros(nx,ny,nz)
W_2_2 = @zeros(nx,ny,nz)
W_2_3 = @zeros(nx,ny,nz)
W_2_4 = @zeros(nx,ny,nz)
W_3_1 = @zeros(nx,ny,nz)
W_3_2 = @zeros(nx,ny,nz)
W_3_3 = @zeros(nx,ny,nz)
W_3_4 = @zeros(nx,ny,nz)
Y_1 = @zeros(nx,ny,nz)
Y_2 = @zeros(nx,ny,nz)
Y_3 = @zeros(nx,ny,nz)
Y_4 = @zeros(nx,ny,nz)
f=(ϕ_1,ϕ_2,ϕ_3,ϕ_4,W_1_1,W_1_2,W_1_3,W_1_4,W_2_1,W_2_2,W_2_3,W_2_4,W_3_1,W_3_2,W_3_3,W_3_4,Y_1,Y_2,Y_3,Y_4)
D_1ϕ_1= @zeros(nx,ny,nz)
D_1ϕ_2= @zeros(nx,ny,nz)
D_1ϕ_3= @zeros(nx,ny,nz)
D_1ϕ_4= @zeros(nx,ny,nz)
D_2ϕ_1= @zeros(nx,ny,nz)
D_2ϕ_2= @zeros(nx,ny,nz)
D_2ϕ_3= @zeros(nx,ny,nz)
D_2ϕ_4= @zeros(nx,ny,nz)
D_3ϕ_1= @zeros(nx,ny,nz)
D_3ϕ_2= @zeros(nx,ny,nz)
D_3ϕ_3= @zeros(nx,ny,nz)
D_3ϕ_4= @zeros(nx,ny,nz)
D_4ϕ_1= @zeros(nx,ny,nz)
D_4ϕ_2= @zeros(nx,ny,nz)
D_4ϕ_3= @zeros(nx,ny,nz)
D_4ϕ_4= @zeros(nx,ny,nz)
# f = @zeros(nx,ny,nz,nf)
# cv_arr = @zeros(nx,ny,nz,4,4)
phi_arr= zeros((nx,ny,nx,4))
phi_arr .= random_gen(phi_arr)
x=z=range(1,latx,step=1)
# f .=random_gen(f)
ϕ_1 .= phi_arr[:,:,:,1]
ϕ_2 .= phi_arr[:,:,:,2]
ϕ_3 .= phi_arr[:,:,:,3]
ϕ_4 .= phi_arr[:,:,:,4]

# plt=contourf(z,x,f[:,Integer(ceil(latx/2)),:,1])
# display(plt)
# exit()

@parallel cov_derivs!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
W_1_1,W_1_2,W_1_3,W_1_4,
W_2_1,W_2_2,W_2_3,W_2_4,
W_3_1,W_3_2,W_3_3,W_3_4,
Y_1,Y_2,Y_3,Y_4,
D_1ϕ_1,D_1ϕ_2,D_1ϕ_3,D_1ϕ_4,
D_2ϕ_1,D_2ϕ_2,D_2ϕ_3,D_2ϕ_4,
D_3ϕ_1,D_3ϕ_2,D_3ϕ_3,D_3ϕ_4,
D_4ϕ_1,D_4ϕ_2,D_4ϕ_3,D_4ϕ_4)
# @parallel (1:latx,1:laty,1:latz) cov_derivs!(f)
update_halo!(D_1ϕ_1)
update_halo!(D_4ϕ_4)
plt=contourf(z,x,D_1ϕ_1[:,5,:])
display(plt)
# exit()
# Time loop
for it = 1:nt
    @parallel diffusion3D_step!(T2, T, Ci, lam, dt, dx, dy, dz);
    update_halo!(T2);
    T, T2 = T2, T;
end

finalize_global_grid();
end

diffusion3D()
