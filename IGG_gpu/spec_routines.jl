#Magnetic field spectrum convolution routine

module spec_convolver

using Statistics

export K_c_mag
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

export Kc_bin_nums
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

export convolve_1d

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

end