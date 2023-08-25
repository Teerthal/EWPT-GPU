#Parallel stencil example from
#https://github.com/omlins/ParallelStencil.jl
# for testing macros

using CUDA
using Plots

@inbounds @views macro d_xa(A) esc(:( ($A[2:end  , :     ] .- $A[1:end-1, :     ]) )) end
@inbounds @views macro d_xi(A) esc(:( ($A[2:end  ,2:end-1] .- $A[1:end-1,2:end-1]) )) end
@inbounds @views macro d_ya(A) esc(:( ($A[ :     ,2:end  ] .- $A[ :     ,1:end-1]) )) end
@inbounds @views macro d_yi(A) esc(:( ($A[2:end-1,2:end  ] .- $A[2:end-1,1:end-1]) )) end
@inbounds @views macro  inn(A) esc(:( $A[2:end-1,2:end-1]                          )) end

@inbounds @views function diffusion2D_step!(T, Ci, qTx, qTy, dTdt, lam, dt, _dx, _dy)
    qTx     .= .-lam.*@d_xi(T).*_dx                              # Fourier's law of heat conduction: qT_x  = -λ ∂T/∂x
    qTy     .= .-lam.*@d_yi(T).*_dy                              # ...                               qT_y  = -λ ∂T/∂y
    dTdt    .= @inn(Ci).*(.-@d_xa(qTx).*_dx .- @d_ya(qTy).*_dy)  # Conservation of energy:           ∂T/∂t = 1/cp (-∂qT_x/∂x - ∂qT_y/∂y)
    @inn(T) .= @inn(T) .+ dt.*dTdt                               # Update of temperature             T_new = T_old + ∂t ∂T/∂t
end
     


function diffusion2D()
    # Physics
    lam      = 1.0                                          # Thermal conductivity
    c0       = 2.0                                          # Heat capacity
    lx, ly   = 10.0, 10.0                                   # Length of computational domain in dimension x and y

    # Numerics
    nx, ny   = 32*2, 32*2                                   # Number of gridpoints in dimensions x and y
    nt       = 100                                          # Number of time steps
    dx       = lx/(nx-1)                                    # Space step in x-dimension
    dy       = ly/(ny-1)                                    # Space step in y-dimension
    _dx, _dy = 1.0/dx, 1.0/dy

    # Array initializations
    T    = CUDA.zeros(Float64, nx, ny)                      # Temperature
    Ci   = CUDA.zeros(Float64, nx, ny)                      # 1/Heat capacity
    qTx  = CUDA.zeros(Float64, nx-1, ny-2)                  # Heat flux, x component
    qTy  = CUDA.zeros(Float64, nx-2, ny-1)                  # Heat flux, y component
    dTdt = CUDA.zeros(Float64, nx-2, ny-2)                  # Change of Temperature in time
    T .= 0.0
    Ci .= 0.0
    qTx .= 0.0
    qTy .= 0.0
    dTdt .= 0.0
    # Initial conditions
    Ci .= 1/c0                                              # 1/Heat capacity (could vary in space)
    T  .= CuArray([10.0*exp(-(((ix-1)*dx-lx/2)/2)^2-(((iy-1)*dy-ly/2)/2)^2) for ix=1:size(T,1), iy=1:size(T,2)]) # Initialization of Gaussian temperature anomaly

    # Time loop
    dt  = min(dx^2,dy^2)/lam/maximum(Ci)/4.1                # Time step for 2D Heat diffusion
    opts = (aspect_ratio=1, xlims=(1, nx), ylims=(1, ny), clims=(0.0, 10.0), c=:davos, xlabel="Lx", ylabel="Ly") # plotting options
    @gif for it = 1:nt
        diffusion2D_step!(T, Ci, qTx, qTy, dTdt, lam, dt, _dx, _dy) # Diffusion time step.
        heatmap(Array(T)'; opts...)
    end
end

@time diffusion2D()
