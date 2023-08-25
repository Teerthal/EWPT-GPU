#Test scripts for instantiating Higgs bubbles on lattices loaded onto GPUs

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

include("random_ini_gen.jl")
using .randomizer

include("macros.jl")
using .compute_macros
using Distributions


@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3);
else
    @init_parallel_stencil(Threads, Float64, 3);
end

# Multi device version
@parallel_indices (ix,iy,iz) function initializer!(ϕ_1_i,ϕ_2_i,ϕ_3_i,ϕ_4_i,ib,jb,kb,p1,p2,p3,p4,rkx,rky,rkz,dx,mH)
    rb = sqrt((1.0/(rkx^2))*sin(rkx*((ix-1)-(ib-0.5))*dx)^2+(1.0/(rky^2))*sin(rky*((iy-1)-(jb-0.5))*dx)^2+(1.0/(rkz^2))*sin(rkz*((iz-1)-(kb-0.5))*dx)^2)
    rmag = (1.0+(sqrt(2.0)-1.0)^2)*exp(-mH*rb/sqrt(2.0))/(1.0+((sqrt(2.0)-1.0)^2)*exp(-sqrt(2.0)*mH*rb))
    ϕ_1_i[ix,iy,iz]=ϕ_1_i[ix,iy,iz]+rmag*p1
    ϕ_2_i[ix,iy,iz]=ϕ_2_i[ix,iy,iz]+rmag*p2
    ϕ_3_i[ix,iy,iz]=ϕ_3_i[ix,iy,iz]+rmag*p3
    ϕ_4_i[ix,iy,iz]=ϕ_4_i[ix,iy,iz]+rmag*p4
    return
end

@parallel function renormalize!(ϕ_1,ϕ_2,ϕ_3,ϕ_4)
    @all(ϕ_1)=@all(ϕ_1)/(sqrt(@all(ϕ_1)*@all(ϕ_1)+@all(ϕ_2)*@all(ϕ_2)+@all(ϕ_3)*@all(ϕ_3)+@all(ϕ_4)*@all(ϕ_4)))
    @all(ϕ_2)=@all(ϕ_2)/(sqrt(@all(ϕ_1)*@all(ϕ_1)+@all(ϕ_2)*@all(ϕ_2)+@all(ϕ_3)*@all(ϕ_3)+@all(ϕ_4)*@all(ϕ_4)))
    @all(ϕ_3)=@all(ϕ_3)/(sqrt(@all(ϕ_1)*@all(ϕ_1)+@all(ϕ_2)*@all(ϕ_2)+@all(ϕ_3)*@all(ϕ_3)+@all(ϕ_4)*@all(ϕ_4)))
    @all(ϕ_4)=@all(ϕ_4)/(sqrt(@all(ϕ_1)*@all(ϕ_1)+@all(ϕ_2)*@all(ϕ_2)+@all(ϕ_3)*@all(ϕ_3)+@all(ϕ_4)*@all(ϕ_4)))
    return
end

@views function uniform_bubbles(dims)

    # Numerics
    nx, ny, nz = latx,laty,latz;                              # Number of gridpoints dimensions x, y and z.
    # dims = [2,1,1]

    # Array initializations
    ϕ_1_i = @zeros(nx*dims[1],ny*dims[2],nz*dims[3])
    ϕ_2_i = @zeros(nx*dims[1],ny*dims[2],nz*dims[3])
    ϕ_3_i = @zeros(nx*dims[1],ny*dims[2],nz*dims[3])
    ϕ_4_i = @zeros(nx*dims[1],ny*dims[2],nz*dims[3])

    bubs = []
    xb_locs = range(bub_diam,stop=nx*dims[1],step=bub_diam)
    yb_locs = range(bub_diam,stop=ny*dims[2],step=bub_diam)
    zb_locs = range(bub_diam,stop=nz*dims[3],step=bub_diam)

    Random.seed!(123456789)

    Hoft_arr = rand(Uniform(0,1),(size(xb_locs,1),size(yb_locs,1),size(zb_locs,1),3))

    println(Hoft_arr[1,1,1,:])
    for (kb_idx,kb) in enumerate(zb_locs)
        for (jb_idx,jb) in enumerate(yb_locs)
            for (ib_idx,ib) in enumerate(xb_locs)
                phi=gen_phi(Hoft_arr[ib_idx,jb_idx,kb_idx,:])
                push!(bubs,[ib,jb,kb,real(phi[1]),imag(phi[1]),real(phi[2]),imag(phi[2])])
            end
        end
    end

    println(string("# bubbles: ",size(bubs,1)))

    rkx=pi/(nx*dims[1]*dx)
    rky=pi/(ny*dims[2]*dx)
    rkz=pi/(nz*dims[3]*dx)

    @time for b in range(1,size(bubs,1),step=1)
        ib,jb,kb,p1,p2,p3,p4 = bubs[b]
        @parallel (1:nx*dims[1],1:ny*dims[2],1:nz*dims[3]) initializer!(ϕ_1_i,ϕ_2_i,ϕ_3_i,ϕ_4_i,ib,jb,kb,p1,p2,p3,p4,rkx,rky,rkz,dx,mH)
    end

    # println(size(ϕ_1_i));exit()

    ####Initialize and plot first frame####

    x=range(1,latx*dims[1],step=1)
    y=range(1,laty*dims[2],step=1)
    z=range(1,latz*dims[3],step=1)
    gr()
    ENV["GKSwstype"]="nul"
    contourf(z,x,(Array(ϕ_1_i)[:,ny*dims[2]÷2,:]))
    png(string("testini1",".png"))
    exit()
    # anim = Animation();
    # nx_v = (nx-2)*dims[1];
    # ny_v = (ny-2)*dims[2];
    # nz_v = (nz-2)*dims[3];
    # T_v  = zeros(nx_v, ny_v, nz_v);
    # T_nohalo = zeros(nx-2, ny-2, nz-2);

    # p1=contourf(z,x,(Array(ϕ_1)[:,ny÷2,:].*Array(ϕ_1)[:,ny÷2,:].+Array(ϕ_2)[:,ny÷2,:].*Array(ϕ_2)[:,ny÷2,:].+Array(ϕ_3)[:,ny÷2,:].*Array(ϕ_3)[:,ny÷2,:].+Array(ϕ_4)[:,ny÷2,:].*Array(ϕ_4)[:,ny÷2,:]))
    # p2=contourf(z,x,Array(ϕ_2)[:,ny÷2,:])
    # p3=contourf(z,x,Array(ϕ_3)[:,ny÷2,:])
    # p4=contourf(z,x,Array(ϕ_4)[:,ny÷2,:])

    # plot(p1,p2,p3,p4,layout=4)
    # png(string("testini1",".png"))

    ##############End-first-frame_plot###

    # vtk_grid("raw_0",x,y,z) do vtk
    #     vtk["V"] = Float32.(Array(E_V))
    # end

    # h5open(string("raw_0.h5"), "w") do file
    tic()
    h5open(string("ini_phi_",me,".h5"), "w") do file
        write(file, "Phi_1", Array(ϕ_1_i))
        write(file, "Phi_2", Array(ϕ_2_i))
        write(file, "Phi_3", Array(ϕ_3_i))
        write(file, "Phi_4", Array(ϕ_4_i))
    end
    println("Initial save time: ", toc())

    # if (me==0) gif(anim, "EW3d_test.gif", fps = 10) end

end

@views function random_bubbles(no_bubbles,dims,seed_value)

    # Numerics
    nx, ny, nz = latx,laty,latz;                              # Number of gridpoints dimensions x, y and z.

    # Array initializations
    ϕ_1_i = @zeros(nx*dims[1],ny*dims[2],nz*dims[3])
    ϕ_2_i = @zeros(nx*dims[1],ny*dims[2],nz*dims[3])
    ϕ_3_i = @zeros(nx*dims[1],ny*dims[2],nz*dims[3])
    ϕ_4_i = @zeros(nx*dims[1],ny*dims[2],nz*dims[3])
    println(size(ϕ_4_i))

    # xb_locs = range(bub_diam,stop=nx*dims[1],step=bub_diam)
    # yb_locs = range(bub_diam,stop=ny*dims[2],step=bub_diam)
    # zb_locs = range(bub_diam,stop=nz*dims[3],step=bub_diam)

    Random.seed!(seed_value)
    bubble_locs = rand(1:nx,(no_bubbles,3))
    #Locs are generate on cubic lattice.
    #Rescaling to arbitrary dims
    for q in range(1,3)
        bubble_locs[:,q]=bubble_locs[:,q].*dims[q]
    end
    println(string("bubble location matrix",size(bubble_locs)))
    
    Random.seed!(seed_value)
    Hoft_arr = rand(Uniform(0,1),(no_bubbles,3))

    bubs = []
    for bub_idx in range(1,no_bubbles)
        phi=gen_phi(Hoft_arr[bub_idx,:])
        ib,jb,kb = bubble_locs[bub_idx,:]
        push!(bubs,[ib,jb,kb,real(phi[1]),imag(phi[1]),real(phi[2]),imag(phi[2])])
    end

    rkx=pi/(nx*dims[1]*dx)
    rky=pi/(ny*dims[2]*dx)
    rkz=pi/(nz*dims[3]*dx)

    @time for b in range(1,size(bubs,1),step=1)
        ib,jb,kb,p1,p2,p3,p4 = bubs[b]
        @parallel (1:nx*dims[1],1:ny*dims[2],1:nz*dims[3]) initializer!(ϕ_1_i,ϕ_2_i,ϕ_3_i,ϕ_4_i,ib,jb,kb,p1,p2,p3,p4,rkx,rky,rkz,dx,mH)
    end

    # println(size(ϕ_1_i));exit()

    ####Initialize and plot first frame####

    x=range(1,latx*dims[1],step=1)
    y=range(1,laty*dims[2],step=1)
    z=range(1,latz*dims[3],step=1)
    println(size(x),size(y),size(z))
    gr()
    ENV["GKSwstype"]="nul"
    contourf(z,x,(Array(ϕ_1_i)[:,ny*dims[2]÷2,:]))
    png(string("testini1",".png"))
    exit()
    # anim = Animation();
    # nx_v = (nx-2)*dims[1];
    # ny_v = (ny-2)*dims[2];
    # nz_v = (nz-2)*dims[3];
    # T_v  = zeros(nx_v, ny_v, nz_v);
    # T_nohalo = zeros(nx-2, ny-2, nz-2);

    # p1=contourf(z,x,(Array(ϕ_1)[:,ny÷2,:].*Array(ϕ_1)[:,ny÷2,:].+Array(ϕ_2)[:,ny÷2,:].*Array(ϕ_2)[:,ny÷2,:].+Array(ϕ_3)[:,ny÷2,:].*Array(ϕ_3)[:,ny÷2,:].+Array(ϕ_4)[:,ny÷2,:].*Array(ϕ_4)[:,ny÷2,:]))
    # p2=contourf(z,x,Array(ϕ_2)[:,ny÷2,:])
    # p3=contourf(z,x,Array(ϕ_3)[:,ny÷2,:])
    # p4=contourf(z,x,Array(ϕ_4)[:,ny÷2,:])

    # plot(p1,p2,p3,p4,layout=4)
    # png(string("testini1",".png"))

    ##############End-first-frame_plot###

    # vtk_grid("raw_0",x,y,z) do vtk
    #     vtk["V"] = Float32.(Array(E_V))
    # end

    # h5open(string("raw_0.h5"), "w") do file
    tic()
    h5open(string("ini_phi_",me,".h5"), "w") do file
        write(file, "Phi_1", Array(ϕ_1_i))
        write(file, "Phi_2", Array(ϕ_2_i))
        write(file, "Phi_3", Array(ϕ_3_i))
        write(file, "Phi_4", Array(ϕ_4_i))
    end
    println("Initial save time: ", toc())

    # if (me==0) gif(anim, "EW3d_test.gif", fps = 10) end

end

dims = [1,1,1]
seed_value = 1234567890
no_bubbles = 10

@time random_bubbles(no_bubbles,dims,seed_value)

# @time uniform_bubbles(dims)
