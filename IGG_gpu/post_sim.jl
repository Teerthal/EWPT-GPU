#Post sim routines to construct Viz VTK files
#and magnetic field spectrum convolution

using WriteVTK
using HDF5
using Plots
using Statistics
using StatsBase
using FFTW

include("parameters.jl")
using .parameters


gr()
ENV["GKSwstype"]="nul"
anim = Animation();


function K_c_mag(x,y,z,Nx,Ny,Nz)
    x=x-1
    y=y-1
    z=z-1
    if x <= Nx÷2
        K_x = x
    else
        K_x = x - Nx
    end

    if y <= Ny÷2
        K_y = y
    else
        K_y = y - Ny
    end

    if z <= Nz÷2
        K_z = z
    else
        K_z = z - Nz
    end

    return sqrt(K_x^2+K_y^2+K_z^2)
    # return sqrt((x-1)^2+(y-1)^2+(z-1)^2)
end
function Kc_bin_nums(Nx,Ny,Nz)
    lis = []
    for i in range(1,Nx,step=1)
        for j in range(1,Ny,step=1)
            for k in range(1,Nz,step=1)
                push!(lis,K_c_mag(i,j,k,Nx,Ny,Nz))
            end
        end
    end
    return size(unique!(sort!(lis)),1)
end
function convolve_1d(B_fft,Nx,Ny,Nz,cutx,cuty,cutz)
    
    N_bins = Kc_bin_nums(cutx,cuty,cutz)
    spec_stack = zeros((N_bins,2))#;println(size(spec_stack));exit()
    sorted = zeros((Nx*Ny*Nz,2))
    idx = 1
    # println(N_bins)
    for i in range(1,Nx,step=1)
        for j in range(1,Ny,step=1)
            for k in range(1,Nz,step=1)
                sorted[idx,1]=K_c_mag(i,j,k,Nx,Ny,Nz)
                sorted[idx,2]=B_fft[i,j,k]
                idx=idx+1
            end
        end
    end
    sorted = sortslices(sorted,dims=1)
    unique_ks = unique!(sorted[:,1])[1:N_bins]
    # println(size(unique_ks));exit()
    k_raw = sorted[:,1]
    for (k_idx, k) in enumerate(unique_ks)
        # @time idxs = findall(k_raw.==k)
        spec_stack[k_idx,1] = k
        # println(size(sorted[idxs,2]))
        spec_stack[k_idx,2] = mean(sorted[findall(k_raw.==k),2])
        # mean(sorted[idxs,2])
        # println(k," ",sum(sorted[idxs,2]))
    end
    
    return spec_stack
end

#manually overriding nt
# nt=7000

dims = [2,1,1]
nx_g, ny_g, nz_g = (latx-2)*dims[1], (laty-2)*dims[2], (latz-2)*dims[3]
println(nx_g," ",ny_g," ",nz_g)
process_num = dims[1]*dims[2]*dims[3]
E_V_global = zeros(Float32,(nx_g,ny_g,nz_g))
E_global = zeros(Float32,(nx_g,ny_g,nz_g))
B_x_global = zeros(Float32,(nx_g,ny_g,nz_g))
B_y_global = zeros(Float32,(nx_g,ny_g,nz_g))
B_z_global = zeros(Float32,(nx_g,ny_g,nz_g))
#Stitch data if using multiple devices
function stitcher(i)
    for p in range(0,process_num-1,step=1)
        if dims[2]!=1
            p_k = floor(Int,p/(dims[1]*dims[2]))
            p_j = floor(Int,(p - dims[1]*dims[2]*p_k)/dims[2])
            p_i = p-(dims[1]*dims[2]*p_k)-(dims[2]*p_j)
        else
            p_k = 0
            p_j = 0
            p_i = p#-(dims[1]*dims[2]*p_k)-(dims[2]*p_j)
        end
        println(p,p_i,p_j,p_k)
        file = h5open(string("raw_",i,"_",p,".h5"),"r")
        E_V = read(file["E_V"])
        E = read(file["E"])
        B_x = read(file["B_x"])
        B_y = read(file["B_y"])
        B_z = read(file["B_z"])
        close(file)
        p_l = p_i*(latx-2)+1
        p_r = (p_i+1)*(latx-2)
        p_b = p_j*(laty-2)+1
        p_f = (p_j+1)*(laty-2)
        p_d = p_k*(latz-2)+1
        p_u = (p_k+1)*(latz-2)
        # println(size(E_global))
        # println(size(E_global[p_l:p_r,p_b:p_f,p_d:p_u]))
        # println(size(E))
        E_V_global[p_l:p_r,p_b:p_f,p_d:p_u] = E_V
        E_global[p_l:p_r,p_b:p_f,p_d:p_u] = E
        B_x_global[p_l:p_r,p_b:p_f,p_d:p_u] = B_x
        B_y_global[p_l:p_r,p_b:p_f,p_d:p_u] = B_y
        B_z_global[p_l:p_r,p_b:p_f,p_d:p_u] = B_z
    end
    return E_global,E_V_global,B_x_global,B_y_global,B_z_global
end

#spec_cut allows to convolve upto a certain wavenumber to save on compute time
# spec_cut = [nx_g÷2,ny_g÷2,nz_g÷2]
#larger cut for testing
spec_cut = [nx_g÷4,ny_g÷4,nz_g÷4]
println(spec_cut)
# if spec_cut!=latx
N_bins = Kc_bin_nums(spec_cut[1],spec_cut[2],spec_cut[3])
#number of time bins is nspas+1 to account for the 0th or the initial data time stamp
B_fft = zeros((nsnaps+1,N_bins,2))

#Calculating spectrum at every other step cause expenseive
spec_cal_times = range(0,nt,step=dsnaps)

x=range(1,nx_g,step=1)
y=range(1,ny_g,step=1)
z=range(1,nz_g,step=1)

# @time Threads.@threads for idx in range(1,size(spec_cal_times,1),step=1)
# # for idx in range(1,size(spec_cal_times),step=1)
#     #Read hdf5 files
#     i=spec_cal_times[idx]
#     # file = h5open(string("raw_",i,".h5"),"r") 
#     # B_x_fft = fft(read(file["B_x"]))
#     # B_y_fft = fft(read(file["B_y"]))
#     # B_z_fft = fft(read(file["B_z"]))
#     E_global,E_V_global,B_x_global,B_y_global,B_z_global = stitcher(i)
#     B_x_fft = fft(B_x_global)
#     B_y_fft = fft(B_y_global)
#     B_z_fft = fft(B_z_global)
    
#     # close(file)
#     println(size(B_x_fft))    
#     B_fft[idx,:,:] = convolve_1d((real(conj.(B_x_fft).*B_x_fft.+
#     conj.(B_y_fft).*B_y_fft.+
#     conj.(B_z_fft).*B_z_fft)),nx_g,ny_g,nz_g,spec_cut[1],spec_cut[2],spec_cut[3])
    
# end

# #plotting not thread-safe. need to do it linearly
# for idx in range(1,size(spec_cal_times,1),step=1)
#     i=spec_cal_times[idx]
#     plot(B_fft[idx,2:end,1],(((B_fft[idx,2:end,1]).^2)./((2*pi)^3*latx^2)).*B_fft[idx,2:end,2])
#     plot!(xscale=:log10,yscale=:log10,minorgrid=true,dpi=600)
#     png(string("E_M_",i))
# end

potentials = []
energies = []
# B_arr = zeros(Float32,(nx_g,ny_g,nz_g,3))
@time for (itr,i) in enumerate(range(0,nt,step=dsnaps))
    #Read hdf5 files

    # file = h5open(string("raw_",i,".h5"),"r") 
    # E_V = read(file["E_V"])
    # E = read(file["E"])
    # B_x = read(file["B_x"])
    # B_y = read(file["B_y"])
    # B_z = read(file["B_z"])

    # close(file)
    E,E_V,B_x,B_y,B_z = stitcher(i)
    
    vtk_grid(string("raw_",i),x,y,z) do vtk
        vtk["V"] = E_V
        vtk["E"] = E
        # vtk["B"] = sqrt.((B_x.^2).+(B_y).^2 .+(B_z))
    end

    push!(energies,[i,sum(E)])
    push!(potentials,[i,sum(E_V)])
    p1=contourf(z,x,E_V[:,ny_g÷2,:])
    p3=contourf(z,x,E[:,ny_g÷2,:])
    # exit()
    # if i==0
    #     p2=scatter([potentials[1][1]],[energies[1][2],potentials[1][2]],label=["E","V"],xlims=(0,nt),
    #     yscale=:log10,ylims=(0.01*energies[1][2],1.1*energies[1][2]))
    #     # p4=scatter([energies[1][1]],[energies[1][2]],xlims=(0,nt),
    #     # yscale=:log10,ylims=(0.1*energies[1][2],1.1*energies[1][2]))
    #     p4 = plot(B_fft[1,2:end,1],(((B_fft[1,2:end,1]).^2)./((2*pi)^3*latx^2)).*B_fft[1,2:end,2],
    #     xscale=:log10,yscale=:log10,minorgrid=true)
    # else
    #     p2=plot([potentials[i][1] for i in range(1,size(energies,1),step=1)],
    #     [[energies[i][2] for i in range(1,size(energies,1),step=1)],
    #      [potentials[i][2] for i in range(1,size(energies,1),step=1)]],label=["E","V"],
    #     xlims=(0,nt),yscale=:log10,ylims=(0.01*energies[1][2],1.1*energies[1][2]))
    #     # p4=plot([energies[i][1] for i in range(1,size(energies,1),step=1)],
    #     # [energies[i][2] for i in range(1,size(energies,1),step=1)],
    #     # xlims=(0,nt),yscale=:log10,ylims=(0.1*energies[1][2],1.1*energies[1][2]))
    #     p4 = plot(B_fft[itr,2:end,1],(((B_fft[itr,2:end,1]).^2)./((2*pi)^3*latx^2)).*B_fft[itr,2:end,2],
    #     xscale=:log10,yscale=:log10,minorgrid=true)

    # end
    # plot(p2)
    # png("inienergies.png");exit()
    # plot(p1,p2,p3,p4,layout=4,title=string("it:",i),dpi=600)
    # plot(p1,p2,p3,layout=3,title=string("it:",i),dpi=600)
    plot(p1,p3,layout=2,title=string("it:",i),dpi=600)
    if i==0 png("initial.png") end
    frame(anim)
    # exit()
    # rm(string("raw_",i,".h5"))
end

gif(anim, "EW3d_test.gif", fps = 5)