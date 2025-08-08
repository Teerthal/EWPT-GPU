using CUDA#, CuArrays
using Random
using StatsBase
using Distributions
using Plots
using CUDA.CUFFT
using Statistics
import FFTW.bfft
using MPI
using HDF5

CUDA.memory_status()

include("parameters.jl")
using .parameters

include("diff_scheme_multi.jl")
using .differentiations

# include("coordinates.jl")
# using .coords

include("cov_derivs_temporal.jl")
using .covariant_derivatives

include("field_strengths.jl")
using .f_strengths

include("convenients.jl")
using .convenience_functions

include("spec_routines.jl")
using .spec_convolver

##Start MPI-Grid##

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

println(collect(devices()))

# comm_l = MPI.Comm_split_type(comm, MPI.MPI_COMM_TYPE_SHARED, me)
# me_l   = MPI.Comm_rank(comm_l)
# GPU_ID = CUDA.device!(me_l)

comm_l = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, me)
me_l   = MPI.Comm_rank(comm_l)
GPU_ID = CUDA.device!(me_l)

# GPU_ID = CUDA.device!(me)

# sleep(0.5me)
println("Hello world, I am $(me) of $(MPI.Comm_size(comm_cart)) using $(GPU_ID)")
MPI.Barrier(comm_cart)
# exit()

# select_device()                                               # select one GPU per MPI local rank (if >1 GPU per node)

@views function update_halo!(A,neighbors_x,neighbors_y,neighbors_z,comm)

    ###31-05:updated with new MPI.sendrecv routines###
	#x direction
	#Left
    if neighbors_x[1] >= 0
		sendbuf = Array(A[5:7 ,:,:])
		recvbuf = zeros(size(sendbuf))
        MPI.Sendrecv!(sendbuf, recvbuf, comm_cart,dest=neighbors_x[1], sendtag=11, recvtag=10)
		# copyto!(A[1:3 ,:,:],recvbuf)
        A[1:3 ,:,:]=recvbuf
	end
	#Right
	if neighbors_x[2] >= 0
		sendbuf = Array(A[end-6:end-4 ,:,:])
		recvbuf = zeros(size(sendbuf))
    	MPI.Sendrecv!(sendbuf, recvbuf, comm_cart,dest=neighbors_x[2], sendtag=10, recvtag=11)
		# copyto!(A[end-2:end ,:,:],recvbuf)
        A[end-2:end ,:,:]=recvbuf
	end
	# #y direction
	# #Back
	
	if neighbors_y[1] >= 0
		sendbuf = Array(A[:,5:7 ,:])
		recvbuf = zeros(size(sendbuf))
        MPI.Sendrecv!(sendbuf, recvbuf, comm_cart,dest=neighbors_y[1], sendtag=21, recvtag=20)
		# copyto!(A[:,1:3 ,:],recvbuf)
        A[:,1:3 ,:]=recvbuf
	end
	#Forward
	if neighbors_y[2] >= 0
		sendbuf = Array(A[:,end-6:end-4 , :])
		recvbuf = zeros(size(sendbuf))
        MPI.Sendrecv!(sendbuf, recvbuf, comm_cart,dest=neighbors_y[2], sendtag=20, recvtag=21)
		# copyto!(A[:,end-2:end , :],recvbuf)
        A[:,end-2:end , :]=recvbuf
	end
	
	#z direction
	#Up
	if neighbors_z[1] >= 0
		sendbuf = Array(A[:,:,5:7])
		recvbuf = zeros(size(sendbuf))
        MPI.Sendrecv!(sendbuf, recvbuf, comm_cart,dest=neighbors_z[1], sendtag=31, recvtag=30)
		# copyto!(A[:,:,1:3],recvbuf)
        A[:,:,1:3]=recvbuf
	end
	#Down
	if neighbors_z[2] >= 0
		sendbuf = Array(A[:,:,end-6:end-4])
		recvbuf = zeros(size(sendbuf))
        MPI.Sendrecv!(sendbuf, recvbuf, comm_cart,dest=neighbors_z[2], sendtag=30, recvtag=31)
		# copyto!(A[:,:,end-2:end],recvbuf)
        A[:,:,end-2:end]=recvbuf
	end
	return
end

@views function boundary_x(A,dims,comm,coords)
    # left_rank = MPI.Cart_rank(comm,[0,0,0])
    # right_rank = MPI.Cart_rank(comm,[dims[1]-1,0,0])
    ###31-05:updated with new MPI.sendrecv routines###
    if dims[1]>1
        if (coords[1]==(dims[1]-1))
        # if me == right_rank
            sendbuf=Array(A[end-6:end-4,:,:])
            recvbuf=zeros(size(A[end-2:end,:,:]))
            send_rank = MPI.Cart_rank(comm,[0,coords[2],coords[3]])
            MPI.Sendrecv!(sendbuf, recvbuf, comm_cart,dest=send_rank, sendtag=send_rank+10, recvtag=me+10)
            copyto!(A[end-2:end,:,:],recvbuf)
        end
        if (coords[1]==0)
            recv_rank = MPI.Cart_rank(comm,[dims[1]-1,coords[2],coords[3]])
            sendbuf=Array(A[5:7,:,:])
            recvbuf=zeros(size(Array(A[end-6:end-4,:,:])))
            MPI.Sendrecv!(sendbuf, recvbuf, comm_cart,dest=recv_rank, sendtag=recv_rank+10, recvtag=me+10)
            copyto!(A[1:3,:,:],recvbuf)
        end
    else
        A[1:3,:,:]=A[end-6:end-4,:,:]
        A[end-2:end,:,:]=A[5:7,:,:]

    end

    return
end

@views function boundary_y(A,dims,comm,coords)
    # left_rank = MPI.Cart_rank(comm,[0,0,0])
    # right_rank = MPI.Cart_rank(comm,[dims[1]-1,0,0])
    ###31-05:updated with new MPI.sendrecv routines###
    if dims[2]>1
        if (coords[2]==(dims[2]-1))
        # if me == right_rank
            sendbuf=Array(A[:,end-6:end-4,:])
            recvbuf=zeros(size(A[:,end-2:end,:]))
            send_rank = MPI.Cart_rank(comm,[coords[1],0,coords[3]])
            MPI.Sendrecv!(sendbuf, recvbuf, comm_cart,dest=send_rank, sendtag=send_rank+100, recvtag=me+100)
            copyto!(A[:,end-2:end,:],recvbuf)
        end
        if (coords[2]==0)
            recv_rank = MPI.Cart_rank(comm,[coords[1],dims[2]-1,coords[3]])
            sendbuf=Array(A[:,5:7,:])
            recvbuf=zeros(size(Array(A[:,end-6:end-4,:])))
            MPI.Sendrecv!(sendbuf, recvbuf, comm_cart,dest=recv_rank, sendtag=recv_rank+100, recvtag=me+100)
            copyto!(A[:,1:3,:],recvbuf)
        end
    else
        # A[:,1:6,:]=A[:,end-5:end,:]
        A[:,1:3,:]=A[:,end-6:end-4,:]
        A[:,end-2:end,:]=A[:,5:7,:]

    end

    return
end

@views function boundary_z(A,dims,comm,coords)
    # left_rank = MPI.Cart_rank(comm,[0,0,0])
    # right_rank = MPI.Cart_rank(comm,[dims[1]-1,0,0])
    ###31-05:updated with new MPI.sendrecv routines###
    if dims[3]>1
        if (coords[3]==(dims[3]-1))
        # if me == right_rank
            sendbuf=Array(A[:,:,end-6:end-4])
            recvbuf=zeros(size(A[:,:,end-2:end]))
            send_rank = MPI.Cart_rank(comm,[coords[1],coords[2],0])
            MPI.Sendrecv!(sendbuf, recvbuf, comm_cart,dest=send_rank, sendtag=send_rank+500, recvtag=me+500)
            copyto!(A[:,:,end-2:end],recvbuf)
        end
        if (coords[3]==0)
            recv_rank = MPI.Cart_rank(comm,[coords[1],coords[2],dims[3]-1])
            sendbuf=Array(A[:,:,5:7])
            recvbuf=zeros(size(Array(A[:,:,end-6:end-4])))
            MPI.Sendrecv!(sendbuf, recvbuf, comm_cart,dest=recv_rank, sendtag=recv_rank+500, recvtag=me+500)
            copyto!(A[:,:,1:3],recvbuf)
        end
    else
        A[:,:,1:3]=A[:,:,end-6:end-4]
        A[:,:,end-2:end]=A[:,:,5:7]
    end

    return
end

@views function update_calls!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
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
    neighbors_x,neighbors_y,neighbors_z,comm_cart)

    #f arrays#
        MPI.Barrier(comm_cart)
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

    #_dt arrays
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

    #Boundary updates
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

    return
end

@views function update_calls_k!(k_ϕ_1,k_ϕ_2,k_ϕ_3,k_ϕ_4,
    k_W_1_2,k_W_1_3,k_W_1_4,
    k_W_2_2,k_W_2_3,k_W_2_4,
    k_W_3_2,k_W_3_3,k_W_3_4,
    k_Y_2,k_Y_3,k_Y_4,
    k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,
    kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
    kt_W_1_2,kt_W_1_3,kt_W_1_4,
    kt_W_2_2,kt_W_2_3,kt_W_2_4,
    kt_W_3_2,kt_W_3_3,kt_W_3_4,
    kt_Y_2,kt_Y_3,kt_Y_4,
    neighbors_x,neighbors_y,neighbors_z,comm_cart)

    #f arrays#
        MPI.Barrier(comm_cart)
        update_halo!(kt_ϕ_1,neighbors_x,neighbors_y,neighbors_z,comm_cart)
        MPI.Barrier(comm_cart)
        update_halo!(kt_ϕ_2,neighbors_x,neighbors_y,neighbors_z,comm_cart)
        MPI.Barrier(comm_cart)
        update_halo!(kt_ϕ_3,neighbors_x,neighbors_y,neighbors_z,comm_cart)
        MPI.Barrier(comm_cart)
        update_halo!(kt_ϕ_4,neighbors_x,neighbors_y,neighbors_z,comm_cart)
        MPI.Barrier(comm_cart)
        update_halo!(kt_W_1_2,neighbors_x,neighbors_y,neighbors_z,comm_cart)
        MPI.Barrier(comm_cart)
        update_halo!(kt_W_1_3,neighbors_x,neighbors_y,neighbors_z,comm_cart)
        MPI.Barrier(comm_cart)
        update_halo!(kt_W_1_4,neighbors_x,neighbors_y,neighbors_z,comm_cart)
        MPI.Barrier(comm_cart)
        update_halo!(kt_W_2_2,neighbors_x,neighbors_y,neighbors_z,comm_cart)
        MPI.Barrier(comm_cart)
        update_halo!(kt_W_2_3,neighbors_x,neighbors_y,neighbors_z,comm_cart)
        MPI.Barrier(comm_cart)
        update_halo!(kt_W_2_4,neighbors_x,neighbors_y,neighbors_z,comm_cart)
        MPI.Barrier(comm_cart)
        update_halo!(kt_W_3_2,neighbors_x,neighbors_y,neighbors_z,comm_cart)
        MPI.Barrier(comm_cart)
        update_halo!(kt_W_3_3,neighbors_x,neighbors_y,neighbors_z,comm_cart)
        MPI.Barrier(comm_cart)
        update_halo!(kt_W_3_4,neighbors_x,neighbors_y,neighbors_z,comm_cart)
        MPI.Barrier(comm_cart)
        update_halo!(kt_Y_2,neighbors_x,neighbors_y,neighbors_z,comm_cart)
        MPI.Barrier(comm_cart)
        update_halo!(kt_Y_3,neighbors_x,neighbors_y,neighbors_z,comm_cart)
        MPI.Barrier(comm_cart)
        update_halo!(kt_Y_4,neighbors_x,neighbors_y,neighbors_z,comm_cart)

        MPI.Barrier(comm_cart)
        update_halo!(k_Γ_1,neighbors_x,neighbors_y,neighbors_z,comm_cart)
        MPI.Barrier(comm_cart)
        update_halo!(k_Γ_2,neighbors_x,neighbors_y,neighbors_z,comm_cart)
        MPI.Barrier(comm_cart)
        update_halo!(k_Γ_3,neighbors_x,neighbors_y,neighbors_z,comm_cart)
        MPI.Barrier(comm_cart)
        update_halo!(k_Σ,neighbors_x,neighbors_y,neighbors_z,comm_cart)
        MPI.Barrier(comm_cart)
        update_halo!(k_ϕ_1,neighbors_x,neighbors_y,neighbors_z,comm_cart)
        MPI.Barrier(comm_cart)
        update_halo!(k_ϕ_2,neighbors_x,neighbors_y,neighbors_z,comm_cart)
        MPI.Barrier(comm_cart)
        update_halo!(k_ϕ_3,neighbors_x,neighbors_y,neighbors_z,comm_cart)
        MPI.Barrier(comm_cart)
        update_halo!(k_ϕ_4,neighbors_x,neighbors_y,neighbors_z,comm_cart)
        MPI.Barrier(comm_cart)
        update_halo!(k_W_1_2,neighbors_x,neighbors_y,neighbors_z,comm_cart)
        MPI.Barrier(comm_cart)
        update_halo!(k_W_1_3,neighbors_x,neighbors_y,neighbors_z,comm_cart)
        MPI.Barrier(comm_cart)
        update_halo!(k_W_1_4,neighbors_x,neighbors_y,neighbors_z,comm_cart)
        MPI.Barrier(comm_cart)
        update_halo!(k_W_2_2,neighbors_x,neighbors_y,neighbors_z,comm_cart)
        MPI.Barrier(comm_cart)
        update_halo!(k_W_2_3,neighbors_x,neighbors_y,neighbors_z,comm_cart)
        MPI.Barrier(comm_cart)
        update_halo!(k_W_2_4,neighbors_x,neighbors_y,neighbors_z,comm_cart)
        MPI.Barrier(comm_cart)
        update_halo!(k_W_3_2,neighbors_x,neighbors_y,neighbors_z,comm_cart)
        MPI.Barrier(comm_cart)
        update_halo!(k_W_3_3,neighbors_x,neighbors_y,neighbors_z,comm_cart)
        MPI.Barrier(comm_cart)
        update_halo!(k_W_3_4,neighbors_x,neighbors_y,neighbors_z,comm_cart)
        MPI.Barrier(comm_cart)
        update_halo!(k_Y_2,neighbors_x,neighbors_y,neighbors_z,comm_cart)
        MPI.Barrier(comm_cart)
        update_halo!(k_Y_3,neighbors_x,neighbors_y,neighbors_z,comm_cart)
        MPI.Barrier(comm_cart)
        update_halo!(k_Y_4,neighbors_x,neighbors_y,neighbors_z,comm_cart)
        

    #Boundary updates
    ##X boundaries##
        MPI.Barrier(comm_cart)
        boundary_x(kt_ϕ_1,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_x(kt_ϕ_2,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_x(kt_ϕ_3,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_x(kt_ϕ_4,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_x(kt_W_1_2,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_x(kt_W_1_3,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_x(kt_W_1_4,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_x(kt_W_2_2,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_x(kt_W_2_3,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_x(kt_W_2_4,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_x(kt_W_3_2,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_x(kt_W_3_3,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_x(kt_W_3_4,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_x(kt_Y_2,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_x(kt_Y_3,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_x(kt_Y_4,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)

        boundary_x(k_ϕ_1,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_x(k_ϕ_2,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_x(k_ϕ_3,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_x(k_ϕ_4,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_x(k_W_1_2,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_x(k_W_1_3,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_x(k_W_1_4,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_x(k_W_2_2,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_x(k_W_2_3,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_x(k_W_2_4,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_x(k_W_3_2,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_x(k_W_3_3,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_x(k_W_3_4,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_x(k_Y_2,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_x(k_Y_3,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_x(k_Y_4,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_x(k_Γ_1,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_x(k_Γ_2,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_x(k_Γ_3,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_x(k_Σ,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
    
    ##Y boundaries##
        MPI.Barrier(comm_cart)
        boundary_y(kt_ϕ_1,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_y(kt_ϕ_2,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_y(kt_ϕ_3,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_y(kt_ϕ_4,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_y(kt_W_1_2,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_y(kt_W_1_3,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_y(kt_W_1_4,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_y(kt_W_2_2,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_y(kt_W_2_3,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_y(kt_W_2_4,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_y(kt_W_3_2,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_y(kt_W_3_3,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_y(kt_W_3_4,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_y(kt_Y_2,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_y(kt_Y_3,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_y(kt_Y_4,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)

        boundary_y(k_ϕ_1,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_y(k_ϕ_2,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_y(k_ϕ_3,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_y(k_ϕ_4,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_y(k_W_1_2,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_y(k_W_1_3,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_y(k_W_1_4,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_y(k_W_2_2,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_y(k_W_2_3,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_y(k_W_2_4,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_y(k_W_3_2,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_y(k_W_3_3,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_y(k_W_3_4,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_y(k_Y_2,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_y(k_Y_3,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_y(k_Y_4,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_y(k_Γ_1,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_y(k_Γ_2,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_y(k_Γ_3,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_y(k_Σ,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)

    ##Z boundaries##
        MPI.Barrier(comm_cart)
        boundary_z(kt_ϕ_1,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_z(kt_ϕ_2,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_z(kt_ϕ_3,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_z(kt_ϕ_4,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_z(kt_W_1_2,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_z(kt_W_1_3,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_z(kt_W_1_4,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_z(kt_W_2_2,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_z(kt_W_2_3,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_z(kt_W_2_4,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_z(kt_W_3_2,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_z(kt_W_3_3,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_z(kt_W_3_4,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_z(kt_Y_2,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_z(kt_Y_3,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_z(kt_Y_4,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)

        boundary_z(k_ϕ_1,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_z(k_ϕ_2,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_z(k_ϕ_3,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_z(k_ϕ_4,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_z(k_W_1_2,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_z(k_W_1_3,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_z(k_W_1_4,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_z(k_W_2_2,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_z(k_W_2_3,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_z(k_W_2_4,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_z(k_W_3_2,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_z(k_W_3_3,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_z(k_W_3_4,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_z(k_Y_2,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_z(k_Y_3,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_z(k_Y_4,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_z(k_Γ_1,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_z(k_Γ_2,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_z(k_Γ_3,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)
        boundary_z(k_Σ,dims,comm_cart,coords)
        MPI.Barrier(comm_cart)

    return
end

# @views function update_calls_k!(k_ϕ_1,k_ϕ_2,k_ϕ_3,k_ϕ_4,
#     k_W_1_2,k_W_1_3,k_W_1_4,
#     k_W_2_2,k_W_2_3,k_W_2_4,
#     k_W_3_2,k_W_3_3,k_W_3_4,
#     k_Y_2,k_Y_3,k_Y_4,
#     k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,
#     kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
#     kt_W_1_2,kt_W_1_3,kt_W_1_4,
#     kt_W_2_2,kt_W_2_3,kt_W_2_4,
#     kt_W_3_2,kt_W_3_3,kt_W_3_4,
#     kt_Y_2,kt_Y_3,kt_Y_4,
#     neighbors_x,neighbors_y,neighbors_z,comm_cart)

#     #f arrays#
#         #MPI.Barrier(comm_cart)
#         update_halo!(kt_ϕ_1,neighbors_x,neighbors_y,neighbors_z,comm_cart)
#         #MPI.Barrier(comm_cart)
#         update_halo!(kt_ϕ_2,neighbors_x,neighbors_y,neighbors_z,comm_cart)
#         #MPI.Barrier(comm_cart)
#         update_halo!(kt_ϕ_3,neighbors_x,neighbors_y,neighbors_z,comm_cart)
#         #MPI.Barrier(comm_cart)
#         update_halo!(kt_ϕ_4,neighbors_x,neighbors_y,neighbors_z,comm_cart)
#         #MPI.Barrier(comm_cart)
#         update_halo!(kt_W_1_2,neighbors_x,neighbors_y,neighbors_z,comm_cart)
#         #MPI.Barrier(comm_cart)
#         update_halo!(kt_W_1_3,neighbors_x,neighbors_y,neighbors_z,comm_cart)
#         #MPI.Barrier(comm_cart)
#         update_halo!(kt_W_1_4,neighbors_x,neighbors_y,neighbors_z,comm_cart)
#         #MPI.Barrier(comm_cart)
#         update_halo!(kt_W_2_2,neighbors_x,neighbors_y,neighbors_z,comm_cart)
#         #MPI.Barrier(comm_cart)
#         update_halo!(kt_W_2_3,neighbors_x,neighbors_y,neighbors_z,comm_cart)
#         #MPI.Barrier(comm_cart)
#         update_halo!(kt_W_2_4,neighbors_x,neighbors_y,neighbors_z,comm_cart)
#         #MPI.Barrier(comm_cart)
#         update_halo!(kt_W_3_2,neighbors_x,neighbors_y,neighbors_z,comm_cart)
#         #MPI.Barrier(comm_cart)
#         update_halo!(kt_W_3_3,neighbors_x,neighbors_y,neighbors_z,comm_cart)
#         #MPI.Barrier(comm_cart)
#         update_halo!(kt_W_3_4,neighbors_x,neighbors_y,neighbors_z,comm_cart)
#         #MPI.Barrier(comm_cart)
#         update_halo!(kt_Y_2,neighbors_x,neighbors_y,neighbors_z,comm_cart)
#         #MPI.Barrier(comm_cart)
#         update_halo!(kt_Y_3,neighbors_x,neighbors_y,neighbors_z,comm_cart)
#         #MPI.Barrier(comm_cart)
#         update_halo!(kt_Y_4,neighbors_x,neighbors_y,neighbors_z,comm_cart)

#         #MPI.Barrier(comm_cart)
#         update_halo!(k_Γ_1,neighbors_x,neighbors_y,neighbors_z,comm_cart)
#         #MPI.Barrier(comm_cart)
#         update_halo!(k_Γ_2,neighbors_x,neighbors_y,neighbors_z,comm_cart)
#         #MPI.Barrier(comm_cart)
#         update_halo!(k_Γ_3,neighbors_x,neighbors_y,neighbors_z,comm_cart)
#         #MPI.Barrier(comm_cart)
#         update_halo!(k_Σ,neighbors_x,neighbors_y,neighbors_z,comm_cart)
#         #MPI.Barrier(comm_cart)
#         update_halo!(k_ϕ_1,neighbors_x,neighbors_y,neighbors_z,comm_cart)
#         #MPI.Barrier(comm_cart)
#         update_halo!(k_ϕ_2,neighbors_x,neighbors_y,neighbors_z,comm_cart)
#         #MPI.Barrier(comm_cart)
#         update_halo!(k_ϕ_3,neighbors_x,neighbors_y,neighbors_z,comm_cart)
#         #MPI.Barrier(comm_cart)
#         update_halo!(k_ϕ_4,neighbors_x,neighbors_y,neighbors_z,comm_cart)
#         #MPI.Barrier(comm_cart)
#         update_halo!(k_W_1_2,neighbors_x,neighbors_y,neighbors_z,comm_cart)
#         #MPI.Barrier(comm_cart)
#         update_halo!(k_W_1_3,neighbors_x,neighbors_y,neighbors_z,comm_cart)
#         #MPI.Barrier(comm_cart)
#         update_halo!(k_W_1_4,neighbors_x,neighbors_y,neighbors_z,comm_cart)
#         #MPI.Barrier(comm_cart)
#         update_halo!(k_W_2_2,neighbors_x,neighbors_y,neighbors_z,comm_cart)
#         #MPI.Barrier(comm_cart)
#         update_halo!(k_W_2_3,neighbors_x,neighbors_y,neighbors_z,comm_cart)
#         #MPI.Barrier(comm_cart)
#         update_halo!(k_W_2_4,neighbors_x,neighbors_y,neighbors_z,comm_cart)
#         #MPI.Barrier(comm_cart)
#         update_halo!(k_W_3_2,neighbors_x,neighbors_y,neighbors_z,comm_cart)
#         #MPI.Barrier(comm_cart)
#         update_halo!(k_W_3_3,neighbors_x,neighbors_y,neighbors_z,comm_cart)
#         #MPI.Barrier(comm_cart)
#         update_halo!(k_W_3_4,neighbors_x,neighbors_y,neighbors_z,comm_cart)
#         #MPI.Barrier(comm_cart)
#         update_halo!(k_Y_2,neighbors_x,neighbors_y,neighbors_z,comm_cart)
#         #MPI.Barrier(comm_cart)
#         update_halo!(k_Y_3,neighbors_x,neighbors_y,neighbors_z,comm_cart)
#         #MPI.Barrier(comm_cart)
#         update_halo!(k_Y_4,neighbors_x,neighbors_y,neighbors_z,comm_cart)
        

#     #Boundary updates
#     ##X boundaries##
#         #MPI.Barrier(comm_cart)
#         boundary_x(kt_ϕ_1,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_x(kt_ϕ_2,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_x(kt_ϕ_3,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_x(kt_ϕ_4,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_x(kt_W_1_2,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_x(kt_W_1_3,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_x(kt_W_1_4,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_x(kt_W_2_2,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_x(kt_W_2_3,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_x(kt_W_2_4,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_x(kt_W_3_2,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_x(kt_W_3_3,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_x(kt_W_3_4,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_x(kt_Y_2,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_x(kt_Y_3,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_x(kt_Y_4,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)

#         boundary_x(k_ϕ_1,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_x(k_ϕ_2,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_x(k_ϕ_3,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_x(k_ϕ_4,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_x(k_W_1_2,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_x(k_W_1_3,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_x(k_W_1_4,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_x(k_W_2_2,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_x(k_W_2_3,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_x(k_W_2_4,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_x(k_W_3_2,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_x(k_W_3_3,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_x(k_W_3_4,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_x(k_Y_2,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_x(k_Y_3,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_x(k_Y_4,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_x(k_Γ_1,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_x(k_Γ_2,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_x(k_Γ_3,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_x(k_Σ,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
    
#     ##Y boundaries##
#         #MPI.Barrier(comm_cart)
#         boundary_y(kt_ϕ_1,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_y(kt_ϕ_2,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_y(kt_ϕ_3,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_y(kt_ϕ_4,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_y(kt_W_1_2,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_y(kt_W_1_3,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_y(kt_W_1_4,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_y(kt_W_2_2,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_y(kt_W_2_3,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_y(kt_W_2_4,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_y(kt_W_3_2,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_y(kt_W_3_3,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_y(kt_W_3_4,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_y(kt_Y_2,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_y(kt_Y_3,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_y(kt_Y_4,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)

#         boundary_y(k_ϕ_1,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_y(k_ϕ_2,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_y(k_ϕ_3,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_y(k_ϕ_4,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_y(k_W_1_2,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_y(k_W_1_3,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_y(k_W_1_4,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_y(k_W_2_2,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_y(k_W_2_3,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_y(k_W_2_4,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_y(k_W_3_2,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_y(k_W_3_3,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_y(k_W_3_4,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_y(k_Y_2,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_y(k_Y_3,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_y(k_Y_4,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_y(k_Γ_1,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_y(k_Γ_2,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_y(k_Γ_3,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_y(k_Σ,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)

#     ##Z boundaries##
#         #MPI.Barrier(comm_cart)
#         boundary_z(kt_ϕ_1,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_z(kt_ϕ_2,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_z(kt_ϕ_3,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_z(kt_ϕ_4,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_z(kt_W_1_2,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_z(kt_W_1_3,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_z(kt_W_1_4,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_z(kt_W_2_2,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_z(kt_W_2_3,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_z(kt_W_2_4,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_z(kt_W_3_2,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_z(kt_W_3_3,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_z(kt_W_3_4,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_z(kt_Y_2,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_z(kt_Y_3,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_z(kt_Y_4,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)

#         boundary_z(k_ϕ_1,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_z(k_ϕ_2,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_z(k_ϕ_3,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_z(k_ϕ_4,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_z(k_W_1_2,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_z(k_W_1_3,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_z(k_W_1_4,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_z(k_W_2_2,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_z(k_W_2_3,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_z(k_W_2_4,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_z(k_W_3_2,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_z(k_W_3_3,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_z(k_W_3_4,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_z(k_Y_2,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_z(k_Y_3,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_z(k_Y_4,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_z(k_Γ_1,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_z(k_Γ_2,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_z(k_Γ_3,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)
#         boundary_z(k_Σ,dims,comm_cart,coords)
#         #MPI.Barrier(comm_cart)

#     return
# end

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

@views function gather_metrics(A,A_global,me,comm,nprocs,metrics_length)
	sendbuf=A
    # nprocs = MPI.Comm_size(comm)
	# println(sendbuf[1,1,1])
    if me!=0
        req=MPI.Isend(sendbuf,0,0,comm)
		MPI.Wait!(req)
    else
		# println(size(sendbuf))
		# println(size(A_global[1:Nx,1:Ny,1:Nz]))
		A_global[1:metrics_length]=sendbuf
        for p in range(1,nprocs-1,step=1)
            A_c = zeros(size(sendbuf))
            req = MPI.Irecv!(A_c,p,0,comm)
			MPI.Wait!(req)
            A_global[(1+me*metrics_length):(me+1)*metrics_length]=A_c
			# println(A_c[1,1,1])
        end
    end
    return
end

function x_g(i,Nx,grid_x,dx)
    # return (i-4)*dx+(grid_x+1)*(Nx-6)*dx
    return (i-4)*dx+(grid_x)*(Nx-6)*dx
end

function y_g(j,Ny,grid_y,dx)
    # return (j-4)*dx+(grid_y+1)*(Ny-6)*dx
    return (j-4)*dx+(grid_y)*(Ny-6)*dx
end

function z_g(k,Nz,grid_z,dx)
    # return (k-4)*dx+(grid_z+1)*(Nz-6)*dx
    return (k-4)*dx+(grid_z)*(Nz-6)*dx
end

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
    gw,gy,gp2,β_W,β_Y,vev,lambda,dx,Jex,γ)
    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
              (blockIdx().y - 1) * blockDim().y + threadIdx().y,
              (blockIdx().z - 1) * blockDim().z + threadIdx().z
    if (i>3 && i<(size(ϕ_1,1)-2) && j>3 && j<(size(ϕ_1,2)-2) && k>3 && k<(size(ϕ_1,3)-2))
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
        0.5*Jex*ϕ_1[i,j,k]-
        γ*ϕ_1[i,j,k]*(ϕ_1[i,j,k]*dϕ_1_dt[i,j,k])/(ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2)+
        β_W*(dW_1_2_dt[i,j,k]*W_1_34+dW_1_3_dt[i,j,k]*(-W_1_24)+dW_1_4_dt[i,j,k]*W_1_23+
             dW_2_2_dt[i,j,k]*W_2_34+dW_1_3_dt[i,j,k]*(-W_2_24)+dW_1_4_dt[i,j,k]*W_2_23+
             dW_3_2_dt[i,j,k]*W_3_34+dW_1_3_dt[i,j,k]*(-W_3_24)+dW_1_4_dt[i,j,k]*W_3_23)*ϕ_1[i,j,k]+
        β_Y*(dY_2_dt[i,j,k]*Y_3_4+dY_3_dt[i,j,k]*(-Y_2_4)+dY_4_dt[i,j,k]*Y_2_3)*ϕ_1[i,j,k])

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
        0.5*Jex*ϕ_2[i,j,k]-
        γ*ϕ_2[i,j,k]*(ϕ_2[i,j,k]*dϕ_2_dt[i,j,k])/(ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2)+
        β_W*(dW_1_2_dt[i,j,k]*W_1_34+dW_1_3_dt[i,j,k]*(-W_1_24)+dW_1_4_dt[i,j,k]*W_1_23+
             dW_2_2_dt[i,j,k]*W_2_34+dW_1_3_dt[i,j,k]*(-W_2_24)+dW_1_4_dt[i,j,k]*W_2_23+
             dW_3_2_dt[i,j,k]*W_3_34+dW_1_3_dt[i,j,k]*(-W_3_24)+dW_1_4_dt[i,j,k]*W_3_23)*ϕ_2[i,j,k]+
        β_Y*(dY_2_dt[i,j,k]*Y_3_4+dY_3_dt[i,j,k]*(-Y_2_4)+dY_4_dt[i,j,k]*Y_2_3)*ϕ_2[i,j,k])

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
        0.5*Jex*ϕ_3[i,j,k]-
        γ*ϕ_3[i,j,k]*(ϕ_3[i,j,k]*dϕ_3_dt[i,j,k])/(ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2)+
        β_W*(dW_1_2_dt[i,j,k]*W_1_34+dW_1_3_dt[i,j,k]*(-W_1_24)+dW_1_4_dt[i,j,k]*W_1_23+
             dW_2_2_dt[i,j,k]*W_2_34+dW_1_3_dt[i,j,k]*(-W_2_24)+dW_1_4_dt[i,j,k]*W_2_23+
             dW_3_2_dt[i,j,k]*W_3_34+dW_1_3_dt[i,j,k]*(-W_3_24)+dW_1_4_dt[i,j,k]*W_3_23)*ϕ_3[i,j,k]+
        β_Y*(dY_2_dt[i,j,k]*Y_3_4+dY_3_dt[i,j,k]*(-Y_2_4)+dY_4_dt[i,j,k]*Y_2_3)*ϕ_3[i,j,k])

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
        0.5*Jex*ϕ_4[i,j,k]-
        γ*ϕ_4[i,j,k]*(ϕ_4[i,j,k]*dϕ_4_dt[i,j,k])/(ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2)+
        β_W*(dW_1_2_dt[i,j,k]*W_1_34+dW_1_3_dt[i,j,k]*(-W_1_24)+dW_1_4_dt[i,j,k]*W_1_23+
             dW_2_2_dt[i,j,k]*W_2_34+dW_1_3_dt[i,j,k]*(-W_2_24)+dW_1_4_dt[i,j,k]*W_2_23+
             dW_3_2_dt[i,j,k]*W_3_34+dW_1_3_dt[i,j,k]*(-W_3_24)+dW_1_4_dt[i,j,k]*W_3_23)*ϕ_4[i,j,k]+
        β_Y*(dY_2_dt[i,j,k]*Y_3_4+dY_3_dt[i,j,k]*(-Y_2_4)+dY_4_dt[i,1j,k]*Y_2_3)*ϕ_4[i,j,k])

    @inbounds kt_W_1_2[i,j,k] = (d2W_1_2_dx2+d2W_1_2_dy2+d2W_1_2_dz2+
        gw*(-(dW_2_2_dx*W_3_2[i,j,k]-dW_3_2_dx*W_2_2[i,j,k])-
        (dW_2_2_dy*W_3_3[i,j,k]-dW_3_2_dy*W_2_3[i,j,k])-
        (dW_2_2_dz*W_3_4[i,j,k]-dW_3_2_dz*W_2_4[i,j,k])-
        (W_2_3[i,j,k]*W_3_23-W_3_3[i,j,k]*W_2_23)-
        (W_2_4[i,j,k]*W_3_24-W_3_4[i,j,k]*W_2_24))+
        gw*(ϕ_1[i,j,k]*Dx_ϕ_4-ϕ_2[i,j,k]*Dx_ϕ_3+ϕ_3[i,j,k]*Dx_ϕ_2-ϕ_4[i,j,k]*Dx_ϕ_1)-
        dΓ_1_dx-gw*(W_2_2[i,j,k]*Γ_3[i,j,k]-W_3_2[i,j,k]*Γ_2[i,j,k])-
        β_W*(dϕ_1_dt[i,j,k]*ϕ_1[i,j,k]+dϕ_2_dt[i,j,k]*ϕ_2[i,j,k]+
             dϕ_3_dt[i,j,k]*ϕ_3[i,j,k]+dϕ_4_dt[i,j,k]*ϕ_4[i,j,k])*2.0*W_1_34+
        2.0*β_W*((dϕ_1_dy*ϕ_1[i,j,k]+dϕ_2_dy*ϕ_2[i,j,k]+
                  dϕ_3_dy*ϕ_3[i,j,k]+dϕ_4_dy*ϕ_4[i,j,k])*dW_1_4_dt[i,j,k]-
                 (dϕ_1_dz*ϕ_1[i,j,k]+dϕ_2_dz*ϕ_2[i,j,k]+
                  dϕ_3_dz*ϕ_3[i,j,k]+dϕ_4_dz*ϕ_4[i,j,k])*dW_1_3_dt[i,j,k]))
        # -γ*dW_1_2_dt[i,j,k])

    @inbounds kt_W_1_3[i,j,k] = (d2W_1_3_dx2+d2W_1_3_dy2+d2W_1_3_dz2+
        gw*(-(dW_2_3_dx*W_3_2[i,j,k]-dW_3_3_dx*W_2_2[i,j,k])-
        (dW_2_3_dy*W_3_3[i,j,k]-dW_3_3_dy*W_2_3[i,j,k])-
        (dW_2_3_dz*W_3_4[i,j,k]-dW_3_3_dz*W_2_4[i,j,k])-
        (W_2_2[i,j,k]*(-W_3_23)-W_3_2[i,j,k]*(-W_2_23))-
        (W_2_4[i,j,k]*W_3_34-W_3_4[i,j,k]*W_2_34))+
        gw*(ϕ_1[i,j,k]*Dy_ϕ_4-ϕ_2[i,j,k]*Dy_ϕ_3+ϕ_3[i,j,k]*Dy_ϕ_2-ϕ_4[i,j,k]*Dy_ϕ_1)-
        dΓ_1_dy-gw*(W_2_3[i,j,k]*Γ_3[i,j,k]-W_3_3[i,j,k]*Γ_2[i,j,k])-
        β_W*(dϕ_1_dt[i,j,k]*ϕ_1[i,j,k]+dϕ_2_dt[i,j,k]*ϕ_2[i,j,k]+
             dϕ_3_dt[i,j,k]*ϕ_3[i,j,k]+dϕ_4_dt[i,j,k]*ϕ_4[i,j,k])*2.0*(-W_1_24)+
        2.0*β_W*((dϕ_1_dz*ϕ_1[i,j,k]+dϕ_2_dz*ϕ_2[i,j,k]+
                  dϕ_3_dz*ϕ_3[i,j,k]+dϕ_4_dz*ϕ_4[i,j,k])*dW_1_2_dt[i,j,k]-
                 (dϕ_1_dx*ϕ_1[i,j,k]+dϕ_2_dx*ϕ_2[i,j,k]+
                  dϕ_3_dx*ϕ_3[i,j,k]+dϕ_4_dx*ϕ_4[i,j,k])*dW_1_4_dt[i,j,k]))
        # -γ*dW_1_3_dt[i,j,k])

    @inbounds kt_W_1_4[i,j,k] = (d2W_1_4_dx2+d2W_1_4_dy2+d2W_1_4_dz2+
        gw*(-(dW_2_4_dx*W_3_2[i,j,k]-dW_3_4_dx*W_2_2[i,j,k])-
        (dW_2_4_dy*W_3_3[i,j,k]-dW_3_4_dy*W_2_3[i,j,k])-
        (dW_2_4_dz*W_3_4[i,j,k]-dW_3_4_dz*W_2_4[i,j,k])-
        (W_2_2[i,j,k]*(-W_3_24)-W_3_2[i,j,k]*(-W_2_24))-
        (W_2_3[i,j,k]*(-W_3_34)-W_3_3[i,j,k]*(-W_2_34)))+
        gw*(ϕ_1[i,j,k]*Dz_ϕ_4-ϕ_2[i,j,k]*Dz_ϕ_3+ϕ_3[i,j,k]*Dz_ϕ_2-ϕ_4[i,j,k]*Dz_ϕ_1)-
        dΓ_1_dz-gw*(W_2_4[i,j,k]*Γ_3[i,j,k]-W_3_4[i,j,k]*Γ_2[i,j,k])-
        β_W*(dϕ_1_dt[i,j,k]*ϕ_1[i,j,k]+dϕ_2_dt[i,j,k]*ϕ_2[i,j,k]+
             dϕ_3_dt[i,j,k]*ϕ_3[i,j,k]+dϕ_4_dt[i,j,k]*ϕ_4[i,j,k])*2.0*W_1_23+
        2.0*β_W*((dϕ_1_dx*ϕ_1[i,j,k]+dϕ_2_dx*ϕ_2[i,j,k]+
                  dϕ_3_dx*ϕ_3[i,j,k]+dϕ_4_dx*ϕ_4[i,j,k])*dW_1_3_dt[i,j,k]-
                 (dϕ_1_dy*ϕ_1[i,j,k]+dϕ_2_dy*ϕ_2[i,j,k]+
                  dϕ_3_dy*ϕ_3[i,j,k]+dϕ_4_dy*ϕ_4[i,j,k])*dW_1_2_dt[i,j,k]))
        # -γ*dW_1_4_dt[i,j,k])

    @inbounds kt_W_2_2[i,j,k] = (d2W_2_2_dx2+d2W_2_2_dy2+d2W_2_2_dz2+
        gw*(-(dW_3_2_dx*W_1_2[i,j,k]-dW_1_2_dx*W_3_2[i,j,k])-
        (dW_3_2_dy*W_1_3[i,j,k]-dW_1_2_dy*W_3_3[i,j,k])-
        (dW_3_2_dz*W_1_4[i,j,k]-dW_1_2_dz*W_3_4[i,j,k])-
        (W_3_3[i,j,k]*W_1_23-W_1_3[i,j,k]*W_3_23)-
        (W_3_4[i,j,k]*W_1_24-W_1_4[i,j,k]*W_3_24))+
        gw*(-ϕ_1[i,j,k]*Dx_ϕ_3-ϕ_2[i,j,k]*Dx_ϕ_4+ϕ_3[i,j,k]*Dx_ϕ_1+ϕ_4[i,j,k]*Dx_ϕ_2)-
        dΓ_2_dx-gw*(W_3_2[i,j,k]*Γ_1[i,j,k]-W_1_2[i,j,k]*Γ_3[i,j,k])-
        β_W*(dϕ_1_dt[i,j,k]*ϕ_1[i,j,k]+dϕ_2_dt[i,j,k]*ϕ_2[i,j,k]+
             dϕ_3_dt[i,j,k]*ϕ_3[i,j,k]+dϕ_4_dt[i,j,k]*ϕ_4[i,j,k])*2.0*W_2_34+
        2.0*β_W*((dϕ_1_dy*ϕ_1[i,j,k]+dϕ_2_dy*ϕ_2[i,j,k]+
                  dϕ_3_dy*ϕ_3[i,j,k]+dϕ_4_dy*ϕ_4[i,j,k])*dW_2_4_dt[i,j,k]-
                 (dϕ_1_dz*ϕ_1[i,j,k]+dϕ_2_dz*ϕ_2[i,j,k]+
                  dϕ_3_dz*ϕ_3[i,j,k]+dϕ_4_dz*ϕ_4[i,j,k])*dW_2_3_dt[i,j,k]))
        # -γ*dW_2_2_dt[i,j,k])

    @inbounds kt_W_2_3[i,j,k] = (d2W_2_3_dx2+d2W_2_3_dy2+d2W_2_3_dz2+
        gw*(-(dW_3_3_dx*W_1_2[i,j,k]-dW_1_3_dx*W_3_2[i,j,k])-
        (dW_3_3_dy*W_1_3[i,j,k]-dW_1_3_dy*W_3_3[i,j,k])-
        (dW_3_3_dz*W_1_4[i,j,k]-dW_1_3_dz*W_3_4[i,j,k])-
        (W_3_2[i,j,k]*(-W_1_23)-W_1_2[i,j,k]*(-W_3_23))-
        (W_3_4[i,j,k]*(W_1_34)-W_1_4[i,j,k]*W_3_34))+
        gw*(-ϕ_1[i,j,k]*Dy_ϕ_3-ϕ_2[i,j,k]*Dy_ϕ_4+ϕ_3[i,j,k]*Dy_ϕ_1+ϕ_4[i,j,k]*Dy_ϕ_2)-
        dΓ_2_dy-gw*(W_3_3[i,j,k]*Γ_1[i,j,k]-W_1_3[i,j,k]*Γ_3[i,j,k])-
        β_W*(dϕ_1_dt[i,j,k]*ϕ_1[i,j,k]+dϕ_2_dt[i,j,k]*ϕ_2[i,j,k]+
             dϕ_3_dt[i,j,k]*ϕ_3[i,j,k]+dϕ_4_dt[i,j,k]*ϕ_4[i,j,k])*2.0*(-W_2_24)+
        2.0*β_W*((dϕ_1_dz*ϕ_1[i,j,k]+dϕ_2_dz*ϕ_2[i,j,k]+
                  dϕ_3_dz*ϕ_3[i,j,k]+dϕ_4_dz*ϕ_4[i,j,k])*dW_2_2_dt[i,j,k]-
                 (dϕ_1_dx*ϕ_1[i,j,k]+dϕ_2_dx*ϕ_2[i,j,k]+
                  dϕ_3_dx*ϕ_3[i,j,k]+dϕ_4_dx*ϕ_4[i,j,k])*dW_2_4_dt[i,j,k]))
        # -γ*dW_2_3_dt[i,j,k])

    @inbounds kt_W_2_4[i,j,k] = (d2W_2_4_dx2+d2W_2_4_dy2+d2W_2_4_dz2+
        gw*(-(dW_3_4_dx*W_1_2[i,j,k]-dW_1_4_dx*W_3_2[i,j,k])-
        (dW_3_4_dy*W_1_3[i,j,k]-dW_1_4_dy*W_3_3[i,j,k])-
        (dW_3_4_dz*W_1_4[i,j,k]-dW_1_4_dz*W_3_4[i,j,k])-
        (W_3_2[i,j,k]*(-W_1_24)-W_1_2[i,j,k]*(-W_3_24))-
        (W_3_3[i,j,k]*(-W_1_34)-W_1_3[i,j,k]*(-W_3_34)))+
        gw*(-ϕ_1[i,j,k]*Dz_ϕ_3-ϕ_2[i,j,k]*Dz_ϕ_4+ϕ_3[i,j,k]*Dz_ϕ_1+ϕ_4[i,j,k]*Dz_ϕ_2)-
        dΓ_2_dz-gw*(W_3_4[i,j,k]*Γ_1[i,j,k]-W_1_4[i,j,k]*Γ_3[i,j,k])-
        β_W*(dϕ_1_dt[i,j,k]*ϕ_1[i,j,k]+dϕ_2_dt[i,j,k]*ϕ_2[i,j,k]+
             dϕ_3_dt[i,j,k]*ϕ_3[i,j,k]+dϕ_4_dt[i,j,k]*ϕ_4[i,j,k])*2.0*W_2_23+
        2.0*β_W*((dϕ_1_dx*ϕ_1[i,j,k]+dϕ_2_dx*ϕ_2[i,j,k]+
                  dϕ_3_dx*ϕ_3[i,j,k]+dϕ_4_dx*ϕ_4[i,j,k])*dW_2_3_dt[i,j,k]-
                 (dϕ_1_dy*ϕ_1[i,j,k]+dϕ_2_dy*ϕ_2[i,j,k]+
                  dϕ_3_dy*ϕ_3[i,j,k]+dϕ_4_dy*ϕ_4[i,j,k])*dW_2_2_dt[i,j,k]))
        # -γ*dW_2_4_dt[i,j,k])

    @inbounds kt_W_3_2[i,j,k] = (d2W_3_2_dx2+d2W_3_2_dy2+d2W_3_2_dz2+
        gw*(-(dW_1_2_dx*W_2_2[i,j,k]-dW_2_2_dx*W_1_2[i,j,k])-
        (dW_1_2_dy*W_2_3[i,j,k]-dW_2_2_dy*W_1_3[i,j,k])-
        (dW_1_2_dz*W_2_4[i,j,k]-dW_2_2_dz*W_1_4[i,j,k])-
        (W_1_3[i,j,k]*W_2_23-W_2_3[i,j,k]*W_1_23)-
        (W_1_4[i,j,k]*W_2_24-W_2_4[i,j,k]*W_1_24))+
        gw*(ϕ_1[i,j,k]*Dx_ϕ_2-ϕ_2[i,j,k]*Dx_ϕ_1-ϕ_3[i,j,k]*Dx_ϕ_4+ϕ_4[i,j,k]*Dx_ϕ_3)-
        dΓ_3_dx-gw*(W_1_2[i,j,k]*Γ_2[i,j,k]-W_2_2[i,j,k]*Γ_1[i,j,k])-
        β_W*(dϕ_1_dt[i,j,k]*ϕ_1[i,j,k]+dϕ_2_dt[i,j,k]*ϕ_2[i,j,k]+
             dϕ_3_dt[i,j,k]*ϕ_3[i,j,k]+dϕ_4_dt[i,j,k]*ϕ_4[i,j,k])*2.0*W_3_34+
        2.0*β_W*((dϕ_1_dy*ϕ_1[i,j,k]+dϕ_2_dy*ϕ_2[i,j,k]+
                  dϕ_3_dy*ϕ_3[i,j,k]+dϕ_4_dy*ϕ_4[i,j,k])*dW_3_4_dt[i,j,k]-
                 (dϕ_1_dz*ϕ_1[i,j,k]+dϕ_2_dz*ϕ_2[i,j,k]+
                  dϕ_3_dz*ϕ_3[i,j,k]+dϕ_4_dz*ϕ_4[i,j,k])*dW_3_3_dt[i,j,k]))
        # -γ*dW_3_2_dt[i,j,k])

    @inbounds kt_W_3_3[i,j,k] = (d2W_3_3_dx2+d2W_3_3_dy2+d2W_3_3_dz2+
        gw*(-(dW_1_3_dx*W_2_2[i,j,k]-dW_2_3_dx*W_1_2[i,j,k])-
            (dW_1_3_dy*W_2_3[i,j,k]-dW_2_3_dy*W_1_3[i,j,k])-
            (dW_1_3_dz*W_2_4[i,j,k]-dW_2_3_dz*W_1_4[i,j,k])-
            (W_1_2[i,j,k]*(-W_2_23)-W_2_2[i,j,k]*(-W_1_23))-
            (W_1_4[i,j,k]*W_2_34-W_2_4[i,j,k]*(W_1_34)))+
        gw*(ϕ_1[i,j,k]*Dy_ϕ_2-ϕ_2[i,j,k]*Dy_ϕ_1-ϕ_3[i,j,k]*Dy_ϕ_4+ϕ_4[i,j,k]*Dy_ϕ_3)-
        dΓ_3_dy-gw*(W_1_3[i,j,k]*Γ_2[i,j,k]-W_2_3[i,j,k]*Γ_1[i,j,k])-
        β_W*(dϕ_1_dt[i,j,k]*ϕ_1[i,j,k]+dϕ_2_dt[i,j,k]*ϕ_2[i,j,k]+
             dϕ_3_dt[i,j,k]*ϕ_3[i,j,k]+dϕ_4_dt[i,j,k]*ϕ_4[i,j,k])*2.0*(-W_3_24)+
        2.0*β_W*((dϕ_1_dz*ϕ_1[i,j,k]+dϕ_2_dz*ϕ_2[i,j,k]+
                  dϕ_3_dz*ϕ_3[i,j,k]+dϕ_4_dz*ϕ_4[i,j,k])*dW_3_2_dt[i,j,k]-
                 (dϕ_1_dx*ϕ_1[i,j,k]+dϕ_2_dx*ϕ_2[i,j,k]+
                  dϕ_3_dx*ϕ_3[i,j,k]+dϕ_4_dx*ϕ_4[i,j,k])*dW_3_4_dt[i,j,k]))
        # -γ*dW_3_3_dt[i,j,k])

    @inbounds kt_W_3_4[i,j,k] = (d2W_3_4_dx2+d2W_3_4_dy2+d2W_3_4_dz2+
        gw*(-(dW_1_4_dx*W_2_2[i,j,k]-dW_2_4_dx*W_1_2[i,j,k])-
            (dW_1_4_dy*W_2_3[i,j,k]-dW_2_4_dy*W_1_3[i,j,k])-
            (dW_1_4_dz*W_2_4[i,j,k]-dW_2_4_dz*W_1_4[i,j,k])-
            (W_1_2[i,j,k]*(-W_2_24)-W_2_2[i,j,k]*(-W_1_24))-
            (W_1_3[i,j,k]*(-W_2_34)-W_2_3[i,j,k]*(-W_1_34)))+
        gw*(ϕ_1[i,j,k]*Dz_ϕ_2-ϕ_2[i,j,k]*Dz_ϕ_1-ϕ_3[i,j,k]*Dz_ϕ_4+ϕ_4[i,j,k]*Dz_ϕ_3)-
        dΓ_3_dz-gw*(W_1_4[i,j,k]*Γ_2[i,j,k]-W_2_4[i,j,k]*Γ_1[i,j,k])-
        β_W*(dϕ_1_dt[i,j,k]*ϕ_1[i,j,k]+dϕ_2_dt[i,j,k]*ϕ_2[i,j,k]+
             dϕ_3_dt[i,j,k]*ϕ_3[i,j,k]+dϕ_4_dt[i,j,k]*ϕ_4[i,j,k])*2.0*W_3_23+
        2.0*β_W*((dϕ_1_dx*ϕ_1[i,j,k]+dϕ_2_dx*ϕ_2[i,j,k]+
                  dϕ_3_dx*ϕ_3[i,j,k]+dϕ_4_dx*ϕ_4[i,j,k])*dW_3_3_dt[i,j,k]-
                 (dϕ_1_dy*ϕ_1[i,j,k]+dϕ_2_dy*ϕ_2[i,j,k]+
                  dϕ_3_dy*ϕ_3[i,j,k]+dϕ_4_dy*ϕ_4[i,j,k])*dW_3_2_dt[i,j,k]))
        # -γ*dW_3_4_dt[i,j,k])

    @inbounds kt_Y_2[i,j,k] = ((d2Y_2_dx2+d2Y_2_dy2+d2Y_2_dz2+
        gy*(ϕ_1[i,j,k]*Dx_ϕ_2-ϕ_2[i,j,k]*Dx_ϕ_1+ϕ_3[i,j,k]*Dx_ϕ_4-ϕ_4[i,j,k]*Dx_ϕ_3)-dΣ_dx)-
        β_Y*(dϕ_1_dt[i,j,k]*ϕ_1[i,j,k]+dϕ_2_dt[i,j,k]*ϕ_2[i,j,k]+
             dϕ_3_dt[i,j,k]*ϕ_3[i,j,k]+dϕ_4_dt[i,j,k]*ϕ_4[i,j,k])*2.0*Y_3_4+
        2.0*β_Y*((dϕ_1_dy*ϕ_1[i,j,k]+dϕ_2_dy*ϕ_2[i,j,k]+
                  dϕ_3_dy*ϕ_3[i,j,k]+dϕ_4_dy*ϕ_4[i,j,k])*dY_4_dt[i,j,k]-
                 (dϕ_1_dz*ϕ_1[i,j,k]+dϕ_2_dz*ϕ_2[i,j,k]+
                  dϕ_3_dz*ϕ_3[i,j,k]+dϕ_4_dz*ϕ_4[i,j,k])*dY_3_dt[i,j,k]))
        # -γ*dY_2_dt[i,j,k])

    @inbounds kt_Y_3[i,j,k] = ((d2Y_3_dx2+d2Y_3_dy2+d2Y_3_dz2+
        gy*(ϕ_1[i,j,k]*Dy_ϕ_2-ϕ_2[i,j,k]*Dy_ϕ_1+ϕ_3[i,j,k]*Dy_ϕ_4-ϕ_4[i,j,k]*Dy_ϕ_3)-dΣ_dy)-
        β_Y*(dϕ_1_dt[i,j,k]*ϕ_1[i,j,k]+dϕ_2_dt[i,j,k]*ϕ_2[i,j,k]+
             dϕ_3_dt[i,j,k]*ϕ_3[i,j,k]+dϕ_4_dt[i,j,k]*ϕ_4[i,j,k])*2.0*(-Y_2_4)+
        2.0*β_Y*((dϕ_1_dz*ϕ_1[i,j,k]+dϕ_2_dz*ϕ_2[i,j,k]+
                  dϕ_3_dz*ϕ_3[i,j,k]+dϕ_4_dz*ϕ_4[i,j,k])*dY_2_dt[i,j,k]-
                 (dϕ_1_dx*ϕ_1[i,j,k]+dϕ_2_dx*ϕ_2[i,j,k]+
                  dϕ_3_dx*ϕ_3[i,j,k]+dϕ_4_dx*ϕ_4[i,j,k])*dY_4_dt[i,j,k]))
        # -γ*dY_3_dt[i,j,k])

    @inbounds kt_Y_4[i,j,k] = ((d2Y_4_dx2+d2Y_4_dy2+d2Y_4_dz2+
        gy*(ϕ_1[i,j,k]*Dz_ϕ_2-ϕ_2[i,j,k]*Dz_ϕ_1+ϕ_3[i,j,k]*Dz_ϕ_4-ϕ_4[i,j,k]*Dz_ϕ_3)-dΣ_dz)-
        β_Y*(dϕ_1_dt[i,j,k]*ϕ_1[i,j,k]+dϕ_2_dt[i,j,k]*ϕ_2[i,j,k]+
             dϕ_3_dt[i,j,k]*ϕ_3[i,j,k]+dϕ_4_dt[i,j,k]*ϕ_4[i,j,k])*2.0*Y_2_3+
        2.0*β_Y*((dϕ_1_dx*ϕ_1[i,j,k]+dϕ_2_dx*ϕ_2[i,j,k]+
                  dϕ_3_dx*ϕ_3[i,j,k]+dϕ_4_dx*ϕ_4[i,j,k])*dY_3_dt[i,j,k]-
                 (dϕ_1_dy*ϕ_1[i,j,k]+dϕ_2_dy*ϕ_2[i,j,k]+
                  dϕ_3_dy*ϕ_3[i,j,k]+dϕ_4_dy*ϕ_4[i,j,k])*dY_2_dt[i,j,k]))
        # -γ*dY_4_dt[i,j,k])
    #########

    ##s-fluxes##
      
    @inbounds k_ϕ_1[i,j,k] =dϕ_1_dt[i,j,k]
    @inbounds k_ϕ_2[i,j,k] =dϕ_2_dt[i,j,k]
    @inbounds k_ϕ_3[i,j,k] =dϕ_3_dt[i,j,k]
    @inbounds k_ϕ_4[i,j,k] =dϕ_4_dt[i,j,k]
    @inbounds k_W_1_2[i,j,k] =dW_1_2_dt[i,j,k]
    @inbounds k_W_1_3[i,j,k] =dW_1_3_dt[i,j,k]
    @inbounds k_W_1_4[i,j,k] =dW_1_4_dt[i,j,k]
    @inbounds k_W_2_2[i,j,k] =dW_2_2_dt[i,j,k]
    @inbounds k_W_2_3[i,j,k] =dW_2_3_dt[i,j,k]
    @inbounds k_W_2_4[i,j,k] =dW_2_4_dt[i,j,k]
    @inbounds k_W_3_2[i,j,k] =dW_3_2_dt[i,j,k]
    @inbounds k_W_3_3[i,j,k] =dW_3_3_dt[i,j,k]
    @inbounds k_W_3_4[i,j,k] =dW_3_4_dt[i,j,k]
    @inbounds k_Y_2[i,j,k] =dY_2_dt[i,j,k]
    @inbounds k_Y_3[i,j,k] =dY_3_dt[i,j,k]
    @inbounds k_Y_4[i,j,k] =dY_4_dt[i,j,k]
    
    # s(Γ_1)=
    @inbounds k_Γ_1[i,j,k] =((1.0.-gp2).*(dfdx(dW_1_2_dt,i,j,k,0.,dx) .+
    dfdy(dW_1_3_dt,i,j,k,0.,dx) .+ dfdz(dW_1_4_dt,i,j,k,0.,dx)).+
        gp2.*gw.*(
        -((W_2_2[i,j,k]).*(dW_3_2_dt[i,j,k]).-
        (W_3_2[i,j,k]).*(dW_2_2_dt[i,j,k])).-
        ((W_2_3[i,j,k]).*(dW_3_3_dt[i,j,k]).-
        (W_3_3[i,j,k]).*(dW_2_3_dt[i,j,k])).-
        ((W_2_4[i,j,k]).*(dW_3_4_dt[i,j,k]).-
        (W_3_4[i,j,k]).*(dW_2_4_dt[i,j,k]))).+
        # c charge from Higgs: 
        gp2.*gw.*((ϕ_1[i,j,k]).*((dϕ_4_dt[i,j,k])).-
        (ϕ_4[i,j,k]).*((dϕ_1_dt[i,j,k])).+
        (ϕ_3[i,j,k]).*((dϕ_2_dt[i,j,k])).-
        (ϕ_2[i,j,k]).*((dϕ_3_dt[i,j,k]))))

    # s(Γ_2)=
    @inbounds k_Γ_2[i,j,k] =((1.0.-gp2).*(dfdx(dW_2_2_dt,i,j,k,0.,dx) .+
    dfdy(dW_2_3_dt,i,j,k,0.,dx) .+ dfdz(dW_2_4_dt,i,j,k,0.,dx)).+
        gp2.*gw.*(
        -((W_3_2[i,j,k]).*(dW_1_2_dt[i,j,k]).-
        (W_1_2[i,j,k]).*(dW_3_2_dt[i,j,k])).-
        ((W_3_3[i,j,k]).*(dW_1_3_dt[i,j,k]).-
        (W_1_3[i,j,k]).*(dW_3_3_dt[i,j,k])).-
        ((W_3_4[i,j,k]).*(dW_1_4_dt[i,j,k]).-
        (W_1_4[i,j,k]).*(dW_3_4_dt[i,j,k]))).+
        # c charge from Higgs: 
        gp2.*gw.*((ϕ_3[i,j,k]).*((dϕ_1_dt[i,j,k])).-
        (ϕ_1[i,j,k]).*((dϕ_3_dt[i,j,k])).+
        (ϕ_4[i,j,k]).*((dϕ_2_dt[i,j,k])).-
        (ϕ_2[i,j,k]).*((dϕ_4_dt[i,j,k]))))

    # s(Γ_3)=
    @inbounds k_Γ_3[i,j,k] =((1.0.-gp2).*(dfdx(dW_3_2_dt,i,j,k,0.,dx) .+
    dfdy(dW_3_3_dt,i,j,k,0.,dx) .+ dfdz(dW_3_4_dt,i,j,k,0.,dx)).+
        gp2.*gw.*(
        -((W_1_2[i,j,k]).*(dW_2_2_dt[i,j,k]).-
        (W_2_2[i,j,k]).*(dW_1_2_dt[i,j,k])).-
        ((W_1_3[i,j,k]).*(dW_2_3_dt[i,j,k]).-
        (W_2_3[i,j,k]).*(dW_1_3_dt[i,j,k])).-
        ((W_1_4[i,j,k]).*(dW_2_4_dt[i,j,k]).-
        (W_2_4[i,j,k]).*(dW_1_4_dt[i,j,k]))).+
        # c current from Higgs: 
        gp2.*gw.*((ϕ_1[i,j,k]).*((dϕ_2_dt[i,j,k])).-
        (ϕ_2[i,j,k]).*((dϕ_1_dt[i,j,k])).+
        (ϕ_4[i,j,k]).*((dϕ_3_dt[i,j,k])).-
        (ϕ_3[i,j,k]).*((dϕ_4_dt[i,j,k]))))

    # s(Σ)=
    @inbounds k_Σ[i,j,k] =((1.0.-gp2).*(dfdx(dY_2_dt,i,j,k,0.,dx) .+
    dfdy(dY_3_dt,i,j,k,0.,dx) .+ dfdz(dY_4_dt,i,j,k,0.,dx)).+
        # c current from Higgs: 
        gp2.*gy.*((ϕ_1[i,j,k])*((dϕ_2_dt[i,j,k])).-
        (ϕ_2[i,j,k]).*((dϕ_1_dt[i,j,k])).+
        (ϕ_3[i,j,k]).*((dϕ_4_dt[i,j,k])).-
        (ϕ_4[i,j,k]).*((dϕ_3_dt[i,j,k]))))
    ###
    
    end
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

@views function compute_PE!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    PE_Phi,gw,gy,gp2,vev,lambda,θ_w,dx)
    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
              (blockIdx().y - 1) * blockDim().y + threadIdx().y,
              (blockIdx().z - 1) * blockDim().z + threadIdx().z
    if (i>3 && i<(size(ϕ_1,1)-2) && j>3 && j<(size(ϕ_1,2)-2) && k>3 && k<(size(ϕ_1,3)-2))
    # Energy expressions
    @inbounds PE_Phi[i,j,k] = (lambda*((ϕ_1[i,j,k])^2+
        (ϕ_2[i,j,k])^2+(ϕ_3[i,j,k])^2+(ϕ_4[i,j,k])^2-vev^2)^2)    
    end
    return
end

@views function compute_GE!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    GE_Phi,gw,gy,gp2,vev,lambda,θ_w,dx)
    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
              (blockIdx().y - 1) * blockDim().y + threadIdx().y,
              (blockIdx().z - 1) * blockDim().z + threadIdx().z
    if (i>3 && i<(size(ϕ_1,1)-2) && j>3 && j<(size(ϕ_1,2)-2) && k>3 && k<(size(ϕ_1,3)-2))
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

        W_1_23 =W_1_xy(dW_1_3_dx,dW_1_2_dy,W_2_2,W_3_3,W_2_3,W_3_2,gw,i,j,k)
        W_1_24=W_1_xz(dW_1_4_dx,dW_1_2_dz,W_2_2,W_3_4,W_2_4,W_3_2,gw,i,j,k)
        W_1_34=W_1_yz(dW_1_4_dy,dW_1_3_dz,W_2_3,W_3_4,W_2_4,W_3_3,gw,i,j,k)
        W_2_23=W_2_xy(dW_2_3_dx,dW_2_2_dy,W_3_2,W_1_3,W_3_3,W_1_2,gw,i,j,k)
        W_2_24=W_2_xz(dW_2_4_dx,dW_2_2_dz,W_3_2,W_1_4,W_3_4,W_1_2,gw,i,j,k)
        W_2_34=W_2_yz(dW_2_4_dy,dW_2_3_dz,W_3_3,W_1_4,W_3_4,W_1_3,gw,i,j,k)
        W_3_23=W_3_xy(dW_3_3_dx,dW_3_2_dy,W_1_2,W_2_3,W_1_3,W_2_2,gw,i,j,k)
        W_3_24=W_3_xz(dW_3_4_dx,dW_3_2_dz,W_1_2,W_2_4,W_1_4,W_2_2,gw,i,j,k)
        W_3_34=W_3_yz(dW_3_4_dy,dW_3_3_dz,W_1_3,W_2_4,W_1_4,W_2_3,gw,i,j,k)
        Y_2_3=Y_2_y(dY_3_dx,dY_2_dy)
        Y_2_4=Y_2_z(dY_4_dx,dY_2_dz)
        Y_3_4=Y_3_z(dY_4_dy,dY_3_dz)
    #END field strengths#

    ##Verified energy expressions: 2-25-24##

    @inbounds GE_Phi[i,j,k] = (Dx_ϕ_1^2+Dy_ϕ_1^2+Dz_ϕ_1^2+
                               Dx_ϕ_2^2+Dy_ϕ_2^2+Dz_ϕ_2^2+
                               Dx_ϕ_3^2+Dy_ϕ_3^2+Dz_ϕ_3^2+
                               Dx_ϕ_4^2+Dy_ϕ_4^2+Dz_ϕ_4^2)
    end
    return
end

@views function compute_KE!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    KE_Phi,gw,gy,gp2,vev,lambda,θ_w,dx)
    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
              (blockIdx().y - 1) * blockDim().y + threadIdx().y,
              (blockIdx().z - 1) * blockDim().z + threadIdx().z
    if (i>3 && i<(size(ϕ_1,1)-2) && j>3 && j<(size(ϕ_1,2)-2) && k>3 && k<(size(ϕ_1,3)-2))
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

        W_1_23 =W_1_xy(dW_1_3_dx,dW_1_2_dy,W_2_2,W_3_3,W_2_3,W_3_2,gw,i,j,k)
        W_1_24=W_1_xz(dW_1_4_dx,dW_1_2_dz,W_2_2,W_3_4,W_2_4,W_3_2,gw,i,j,k)
        W_1_34=W_1_yz(dW_1_4_dy,dW_1_3_dz,W_2_3,W_3_4,W_2_4,W_3_3,gw,i,j,k)
        W_2_23=W_2_xy(dW_2_3_dx,dW_2_2_dy,W_3_2,W_1_3,W_3_3,W_1_2,gw,i,j,k)
        W_2_24=W_2_xz(dW_2_4_dx,dW_2_2_dz,W_3_2,W_1_4,W_3_4,W_1_2,gw,i,j,k)
        W_2_34=W_2_yz(dW_2_4_dy,dW_2_3_dz,W_3_3,W_1_4,W_3_4,W_1_3,gw,i,j,k)
        W_3_23=W_3_xy(dW_3_3_dx,dW_3_2_dy,W_1_2,W_2_3,W_1_3,W_2_2,gw,i,j,k)
        W_3_24=W_3_xz(dW_3_4_dx,dW_3_2_dz,W_1_2,W_2_4,W_1_4,W_2_2,gw,i,j,k)
        W_3_34=W_3_yz(dW_3_4_dy,dW_3_3_dz,W_1_3,W_2_4,W_1_4,W_2_3,gw,i,j,k)
        Y_2_3=Y_2_y(dY_3_dx,dY_2_dy)
        Y_2_4=Y_2_z(dY_4_dx,dY_2_dz)
        Y_3_4=Y_3_z(dY_4_dy,dY_3_dz)
    #END field strengths#

    ##Verified energy expressions: 2-25-24##

    @inbounds KE_Phi[i,j,k] = (Dt_ϕ_1^2+Dt_ϕ_2^2+Dt_ϕ_3^2+Dt_ϕ_4^2)
    end
    return
end

@views function compute_EEW!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    ElectricE_W,gw,gy,gp2,vev,lambda,θ_w,dx)
    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
              (blockIdx().y - 1) * blockDim().y + threadIdx().y,
              (blockIdx().z - 1) * blockDim().z + threadIdx().z
    if (i>3 && i<(size(ϕ_1,1)-2) && j>3 && j<(size(ϕ_1,2)-2) && k>3 && k<(size(ϕ_1,3)-2))
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

        W_1_23 =W_1_xy(dW_1_3_dx,dW_1_2_dy,W_2_2,W_3_3,W_2_3,W_3_2,gw,i,j,k)
        W_1_24=W_1_xz(dW_1_4_dx,dW_1_2_dz,W_2_2,W_3_4,W_2_4,W_3_2,gw,i,j,k)
        W_1_34=W_1_yz(dW_1_4_dy,dW_1_3_dz,W_2_3,W_3_4,W_2_4,W_3_3,gw,i,j,k)
        W_2_23=W_2_xy(dW_2_3_dx,dW_2_2_dy,W_3_2,W_1_3,W_3_3,W_1_2,gw,i,j,k)
        W_2_24=W_2_xz(dW_2_4_dx,dW_2_2_dz,W_3_2,W_1_4,W_3_4,W_1_2,gw,i,j,k)
        W_2_34=W_2_yz(dW_2_4_dy,dW_2_3_dz,W_3_3,W_1_4,W_3_4,W_1_3,gw,i,j,k)
        W_3_23=W_3_xy(dW_3_3_dx,dW_3_2_dy,W_1_2,W_2_3,W_1_3,W_2_2,gw,i,j,k)
        W_3_24=W_3_xz(dW_3_4_dx,dW_3_2_dz,W_1_2,W_2_4,W_1_4,W_2_2,gw,i,j,k)
        W_3_34=W_3_yz(dW_3_4_dy,dW_3_3_dz,W_1_3,W_2_4,W_1_4,W_2_3,gw,i,j,k)
        Y_2_3=Y_2_y(dY_3_dx,dY_2_dy)
        Y_2_4=Y_2_z(dY_4_dx,dY_2_dz)
        Y_3_4=Y_3_z(dY_4_dy,dY_3_dz)
    #END field strengths#

    ##Verified energy expressions: 2-25-24##

    @inbounds ElectricE_W[i,j,k] =(0.5*
        ((dW_1_2_dt[i,j,k])^2+(dW_1_3_dt[i,j,k])^2+(dW_1_4_dt[i,j,k])^2+
        (dW_2_2_dt[i,j,k])^2+(dW_2_3_dt[i,j,k])^2+(dW_2_4_dt[i,j,k])^2+
        (dW_3_2_dt[i,j,k])^2+(dW_3_3_dt[i,j,k])^2+(dW_3_4_dt[i,j,k])^2))
    end
    return
end

@views function compute_MEW!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    MagneticE_W,gw,gy,gp2,vev,lambda,θ_w,dx)
    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
              (blockIdx().y - 1) * blockDim().y + threadIdx().y,
              (blockIdx().z - 1) * blockDim().z + threadIdx().z
    
    # 2-24-24: Checked and correct sp derivative calls below:
    # found and resolved incorrect calls to second order spatial
    # derivatives below
    if (i>3 && i<(size(ϕ_1,1)-2) && j>3 && j<(size(ϕ_1,2)-2) && k>3 && k<(size(ϕ_1,3)-2))
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

        W_1_23 =W_1_xy(dW_1_3_dx,dW_1_2_dy,W_2_2,W_3_3,W_2_3,W_3_2,gw,i,j,k)
        W_1_24=W_1_xz(dW_1_4_dx,dW_1_2_dz,W_2_2,W_3_4,W_2_4,W_3_2,gw,i,j,k)
        W_1_34=W_1_yz(dW_1_4_dy,dW_1_3_dz,W_2_3,W_3_4,W_2_4,W_3_3,gw,i,j,k)
        W_2_23=W_2_xy(dW_2_3_dx,dW_2_2_dy,W_3_2,W_1_3,W_3_3,W_1_2,gw,i,j,k)
        W_2_24=W_2_xz(dW_2_4_dx,dW_2_2_dz,W_3_2,W_1_4,W_3_4,W_1_2,gw,i,j,k)
        W_2_34=W_2_yz(dW_2_4_dy,dW_2_3_dz,W_3_3,W_1_4,W_3_4,W_1_3,gw,i,j,k)
        W_3_23=W_3_xy(dW_3_3_dx,dW_3_2_dy,W_1_2,W_2_3,W_1_3,W_2_2,gw,i,j,k)
        W_3_24=W_3_xz(dW_3_4_dx,dW_3_2_dz,W_1_2,W_2_4,W_1_4,W_2_2,gw,i,j,k)
        W_3_34=W_3_yz(dW_3_4_dy,dW_3_3_dz,W_1_3,W_2_4,W_1_4,W_2_3,gw,i,j,k)
        Y_2_3=Y_2_y(dY_3_dx,dY_2_dy)
        Y_2_4=Y_2_z(dY_4_dx,dY_2_dz)
        Y_3_4=Y_3_z(dY_4_dy,dY_3_dz)
    #END field strengths#

    ##Verified energy expressions: 2-25-24##

    @inbounds MagneticE_W[i,j,k] = (0.5*
        (W_1_23^2+W_1_24^2+W_1_34^2+
        W_2_23^2+W_2_24^2+W_2_34^2+
        W_3_23^2+W_3_24^2+W_3_34^2))
    end
    return
end

@views function compute_EEY!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    ElectricE_Y,gw,gy,gp2,vev,lambda,θ_w,dx)
    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
              (blockIdx().y - 1) * blockDim().y + threadIdx().y,
              (blockIdx().z - 1) * blockDim().z + threadIdx().z
    if (i>3 && i<(size(ϕ_1,1)-2) && j>3 && j<(size(ϕ_1,2)-2) && k>3 && k<(size(ϕ_1,3)-2))
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

        W_1_23 =W_1_xy(dW_1_3_dx,dW_1_2_dy,W_2_2,W_3_3,W_2_3,W_3_2,gw,i,j,k)
        W_1_24=W_1_xz(dW_1_4_dx,dW_1_2_dz,W_2_2,W_3_4,W_2_4,W_3_2,gw,i,j,k)
        W_1_34=W_1_yz(dW_1_4_dy,dW_1_3_dz,W_2_3,W_3_4,W_2_4,W_3_3,gw,i,j,k)
        W_2_23=W_2_xy(dW_2_3_dx,dW_2_2_dy,W_3_2,W_1_3,W_3_3,W_1_2,gw,i,j,k)
        W_2_24=W_2_xz(dW_2_4_dx,dW_2_2_dz,W_3_2,W_1_4,W_3_4,W_1_2,gw,i,j,k)
        W_2_34=W_2_yz(dW_2_4_dy,dW_2_3_dz,W_3_3,W_1_4,W_3_4,W_1_3,gw,i,j,k)
        W_3_23=W_3_xy(dW_3_3_dx,dW_3_2_dy,W_1_2,W_2_3,W_1_3,W_2_2,gw,i,j,k)
        W_3_24=W_3_xz(dW_3_4_dx,dW_3_2_dz,W_1_2,W_2_4,W_1_4,W_2_2,gw,i,j,k)
        W_3_34=W_3_yz(dW_3_4_dy,dW_3_3_dz,W_1_3,W_2_4,W_1_4,W_2_3,gw,i,j,k)
        Y_2_3=Y_2_y(dY_3_dx,dY_2_dy)
        Y_2_4=Y_2_z(dY_4_dx,dY_2_dz)
        Y_3_4=Y_3_z(dY_4_dy,dY_3_dz)
    #END field strengths#

    ##Verified energy expressions: 2-25-24##

    @inbounds ElectricE_Y[i,j,k] = (0.5*
        ((dY_2_dt[i,j,k])^2+(dY_3_dt[i,j,k])^2+(dY_4_dt[i,j,k])^2))
    end
    return
end

@views function compute_MEY!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    MagneticE_Y,gw,gy,gp2,vev,lambda,θ_w,dx)
    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
              (blockIdx().y - 1) * blockDim().y + threadIdx().y,
              (blockIdx().z - 1) * blockDim().z + threadIdx().z
    if (i>3 && i<(size(ϕ_1,1)-2) && j>3 && j<(size(ϕ_1,2)-2) && k>3 && k<(size(ϕ_1,3)-2))
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

        W_1_23 =W_1_xy(dW_1_3_dx,dW_1_2_dy,W_2_2,W_3_3,W_2_3,W_3_2,gw,i,j,k)
        W_1_24=W_1_xz(dW_1_4_dx,dW_1_2_dz,W_2_2,W_3_4,W_2_4,W_3_2,gw,i,j,k)
        W_1_34=W_1_yz(dW_1_4_dy,dW_1_3_dz,W_2_3,W_3_4,W_2_4,W_3_3,gw,i,j,k)
        W_2_23=W_2_xy(dW_2_3_dx,dW_2_2_dy,W_3_2,W_1_3,W_3_3,W_1_2,gw,i,j,k)
        W_2_24=W_2_xz(dW_2_4_dx,dW_2_2_dz,W_3_2,W_1_4,W_3_4,W_1_2,gw,i,j,k)
        W_2_34=W_2_yz(dW_2_4_dy,dW_2_3_dz,W_3_3,W_1_4,W_3_4,W_1_3,gw,i,j,k)
        W_3_23=W_3_xy(dW_3_3_dx,dW_3_2_dy,W_1_2,W_2_3,W_1_3,W_2_2,gw,i,j,k)
        W_3_24=W_3_xz(dW_3_4_dx,dW_3_2_dz,W_1_2,W_2_4,W_1_4,W_2_2,gw,i,j,k)
        W_3_34=W_3_yz(dW_3_4_dy,dW_3_3_dz,W_1_3,W_2_4,W_1_4,W_2_3,gw,i,j,k)
        Y_2_3=Y_2_y(dY_3_dx,dY_2_dy)
        Y_2_4=Y_2_z(dY_4_dx,dY_2_dz)
        Y_3_4=Y_3_z(dY_4_dy,dY_3_dz)
    #END field strengths#

    ##Verified energy expressions: 2-25-24##

    @inbounds MagneticE_Y[i,j,k] = (0.5*(Y_2_3^2+Y_2_4^2+Y_3_4^2))
    end
    return
end

@views function compute_Bx!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    B_x,gw,gy,gp2,vev,lambda,θ_w,dx)
    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
              (blockIdx().y - 1) * blockDim().y + threadIdx().y,
              (blockIdx().z - 1) * blockDim().z + threadIdx().z
    if (i>3 && i<(size(ϕ_1,1)-2) && j>3 && j<(size(ϕ_1,2)-2) && k>3 && k<(size(ϕ_1,3)-2))
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

        W_1_23 =W_1_xy(dW_1_3_dx,dW_1_2_dy,W_2_2,W_3_3,W_2_3,W_3_2,gw,i,j,k)
        W_1_24=W_1_xz(dW_1_4_dx,dW_1_2_dz,W_2_2,W_3_4,W_2_4,W_3_2,gw,i,j,k)
        W_1_34=W_1_yz(dW_1_4_dy,dW_1_3_dz,W_2_3,W_3_4,W_2_4,W_3_3,gw,i,j,k)
        W_2_23=W_2_xy(dW_2_3_dx,dW_2_2_dy,W_3_2,W_1_3,W_3_3,W_1_2,gw,i,j,k)
        W_2_24=W_2_xz(dW_2_4_dx,dW_2_2_dz,W_3_2,W_1_4,W_3_4,W_1_2,gw,i,j,k)
        W_2_34=W_2_yz(dW_2_4_dy,dW_2_3_dz,W_3_3,W_1_4,W_3_4,W_1_3,gw,i,j,k)
        W_3_23=W_3_xy(dW_3_3_dx,dW_3_2_dy,W_1_2,W_2_3,W_1_3,W_2_2,gw,i,j,k)
        W_3_24=W_3_xz(dW_3_4_dx,dW_3_2_dz,W_1_2,W_2_4,W_1_4,W_2_2,gw,i,j,k)
        W_3_34=W_3_yz(dW_3_4_dy,dW_3_3_dz,W_1_3,W_2_4,W_1_4,W_2_3,gw,i,j,k)
        Y_2_3=Y_2_y(dY_3_dx,dY_2_dy)
        Y_2_4=Y_2_z(dY_4_dx,dY_2_dz)
        Y_3_4=Y_3_z(dY_4_dy,dY_3_dz)
    #END field strengths#

    ##Verified energy expressions: 2-25-24##

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
    @inbounds B_x[i,j,k] = -(sin(θ_w)*(n_1*W_1_34+n_2*W_2_34+n_3*W_3_34)+cos(θ_w)*Y_3_4
        +(4. *sin(θ_w)/(gw*vev^2))*(Dy_ϕ_1*Dz_ϕ_2-Dy_ϕ_2*Dz_ϕ_1
        +Dy_ϕ_3*Dz_ϕ_4-Dy_ϕ_4*Dz_ϕ_3))
    end
    return
end

@views function compute_By!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    B_y,gw,gy,gp2,vev,lambda,θ_w,dx)
    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
              (blockIdx().y - 1) * blockDim().y + threadIdx().y,
              (blockIdx().z - 1) * blockDim().z + threadIdx().z
    if (i>3 && i<(size(ϕ_1,1)-2) && j>3 && j<(size(ϕ_1,2)-2) && k>3 && k<(size(ϕ_1,3)-2))
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

        W_1_23 =W_1_xy(dW_1_3_dx,dW_1_2_dy,W_2_2,W_3_3,W_2_3,W_3_2,gw,i,j,k)
        W_1_24=W_1_xz(dW_1_4_dx,dW_1_2_dz,W_2_2,W_3_4,W_2_4,W_3_2,gw,i,j,k)
        W_1_34=W_1_yz(dW_1_4_dy,dW_1_3_dz,W_2_3,W_3_4,W_2_4,W_3_3,gw,i,j,k)
        W_2_23=W_2_xy(dW_2_3_dx,dW_2_2_dy,W_3_2,W_1_3,W_3_3,W_1_2,gw,i,j,k)
        W_2_24=W_2_xz(dW_2_4_dx,dW_2_2_dz,W_3_2,W_1_4,W_3_4,W_1_2,gw,i,j,k)
        W_2_34=W_2_yz(dW_2_4_dy,dW_2_3_dz,W_3_3,W_1_4,W_3_4,W_1_3,gw,i,j,k)
        W_3_23=W_3_xy(dW_3_3_dx,dW_3_2_dy,W_1_2,W_2_3,W_1_3,W_2_2,gw,i,j,k)
        W_3_24=W_3_xz(dW_3_4_dx,dW_3_2_dz,W_1_2,W_2_4,W_1_4,W_2_2,gw,i,j,k)
        W_3_34=W_3_yz(dW_3_4_dy,dW_3_3_dz,W_1_3,W_2_4,W_1_4,W_2_3,gw,i,j,k)
        Y_2_3=Y_2_y(dY_3_dx,dY_2_dy)
        Y_2_4=Y_2_z(dY_4_dx,dY_2_dz)
        Y_3_4=Y_3_z(dY_4_dy,dY_3_dz)
    #END field strengths#

    ##Verified energy expressions: 2-25-24##

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

    @inbounds B_y[i,j,k] = -(sin(θ_w)*(n_1*(-W_1_24)+n_2*(-W_2_24)+n_3*(-W_3_24))+cos(θ_w)*(-Y_2_4)
        +(4. *sin(θ_w)/(gw*vev^2))*(Dz_ϕ_1*Dx_ϕ_2-Dz_ϕ_2*Dx_ϕ_1
        +Dz_ϕ_3*Dx_ϕ_4-Dz_ϕ_4*Dx_ϕ_3))
    end
    return
end

@views function compute_Bz!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    B_z,gw,gy,gp2,vev,lambda,θ_w,dx)
    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
              (blockIdx().y - 1) * blockDim().y + threadIdx().y,
              (blockIdx().z - 1) * blockDim().z + threadIdx().z
    if (i>3 && i<(size(ϕ_1,1)-2) && j>3 && j<(size(ϕ_1,2)-2) && k>3 && k<(size(ϕ_1,3)-2))
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

        W_1_23 =W_1_xy(dW_1_3_dx,dW_1_2_dy,W_2_2,W_3_3,W_2_3,W_3_2,gw,i,j,k)
        W_1_24=W_1_xz(dW_1_4_dx,dW_1_2_dz,W_2_2,W_3_4,W_2_4,W_3_2,gw,i,j,k)
        W_1_34=W_1_yz(dW_1_4_dy,dW_1_3_dz,W_2_3,W_3_4,W_2_4,W_3_3,gw,i,j,k)
        W_2_23=W_2_xy(dW_2_3_dx,dW_2_2_dy,W_3_2,W_1_3,W_3_3,W_1_2,gw,i,j,k)
        W_2_24=W_2_xz(dW_2_4_dx,dW_2_2_dz,W_3_2,W_1_4,W_3_4,W_1_2,gw,i,j,k)
        W_2_34=W_2_yz(dW_2_4_dy,dW_2_3_dz,W_3_3,W_1_4,W_3_4,W_1_3,gw,i,j,k)
        W_3_23=W_3_xy(dW_3_3_dx,dW_3_2_dy,W_1_2,W_2_3,W_1_3,W_2_2,gw,i,j,k)
        W_3_24=W_3_xz(dW_3_4_dx,dW_3_2_dz,W_1_2,W_2_4,W_1_4,W_2_2,gw,i,j,k)
        W_3_34=W_3_yz(dW_3_4_dy,dW_3_3_dz,W_1_3,W_2_4,W_1_4,W_2_3,gw,i,j,k)
        Y_2_3=Y_2_y(dY_3_dx,dY_2_dy)
        Y_2_4=Y_2_z(dY_4_dx,dY_2_dz)
        Y_3_4=Y_3_z(dY_4_dy,dY_3_dz)
    #END field strengths#

    ##Verified energy expressions: 2-25-24##

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

    @inbounds B_z[i,j,k] = -(sin(θ_w)*(n_1*(W_1_23)+n_2*W_2_23+n_3*W_3_23)+cos(θ_w)*Y_2_3
        +(4. *sin(θ_w)/(gw*vev^2))*(Dx_ϕ_1*Dy_ϕ_2-Dx_ϕ_2*Dy_ϕ_1
        +Dx_ϕ_3*Dy_ϕ_4-Dx_ϕ_4*Dy_ϕ_3))
    end
    return
end

@views function compute_Ax!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    A_x,gw,gy,gp2,vev,lambda,θ_w,dx)
    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
              (blockIdx().y - 1) * blockDim().y + threadIdx().y,
              (blockIdx().z - 1) * blockDim().z + threadIdx().z
    
    if (i>3 && i<(size(ϕ_1,1)-2) && j>3 && j<(size(ϕ_1,2)-2) && k>3 && k<(size(ϕ_1,3)-2))

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

        W_1_23 =W_1_xy(dW_1_3_dx,dW_1_2_dy,W_2_2,W_3_3,W_2_3,W_3_2,gw,i,j,k)
        W_1_24=W_1_xz(dW_1_4_dx,dW_1_2_dz,W_2_2,W_3_4,W_2_4,W_3_2,gw,i,j,k)
        W_1_34=W_1_yz(dW_1_4_dy,dW_1_3_dz,W_2_3,W_3_4,W_2_4,W_3_3,gw,i,j,k)
        W_2_23=W_2_xy(dW_2_3_dx,dW_2_2_dy,W_3_2,W_1_3,W_3_3,W_1_2,gw,i,j,k)
        W_2_24=W_2_xz(dW_2_4_dx,dW_2_2_dz,W_3_2,W_1_4,W_3_4,W_1_2,gw,i,j,k)
        W_2_34=W_2_yz(dW_2_4_dy,dW_2_3_dz,W_3_3,W_1_4,W_3_4,W_1_3,gw,i,j,k)
        W_3_23=W_3_xy(dW_3_3_dx,dW_3_2_dy,W_1_2,W_2_3,W_1_3,W_2_2,gw,i,j,k)
        W_3_24=W_3_xz(dW_3_4_dx,dW_3_2_dz,W_1_2,W_2_4,W_1_4,W_2_2,gw,i,j,k)
        W_3_34=W_3_yz(dW_3_4_dy,dW_3_3_dz,W_1_3,W_2_4,W_1_4,W_2_3,gw,i,j,k)
        Y_2_3=Y_2_y(dY_3_dx,dY_2_dy)
        Y_2_4=Y_2_z(dY_4_dx,dY_2_dz)
        Y_3_4=Y_3_z(dY_4_dy,dY_3_dz)
    #END field strengths#

    ##Verified energy expressions: 2-25-24##

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
    @inbounds A_x[i,j,k] = -(sin(θ_w)*(n_1*W_1_34+n_2*W_2_34+n_3*W_3_34)+cos(θ_w)*Y_3_4)
    end
    return
end

@views function compute_Ay!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    A_y,gw,gy,gp2,vev,lambda,θ_w,dx)
    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
              (blockIdx().y - 1) * blockDim().y + threadIdx().y,
              (blockIdx().z - 1) * blockDim().z + threadIdx().z
    if (i>3 && i<(size(ϕ_1,1)-2) && j>3 && j<(size(ϕ_1,2)-2) && k>3 && k<(size(ϕ_1,3)-2))

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

        W_1_23 =W_1_xy(dW_1_3_dx,dW_1_2_dy,W_2_2,W_3_3,W_2_3,W_3_2,gw,i,j,k)
        W_1_24=W_1_xz(dW_1_4_dx,dW_1_2_dz,W_2_2,W_3_4,W_2_4,W_3_2,gw,i,j,k)
        W_1_34=W_1_yz(dW_1_4_dy,dW_1_3_dz,W_2_3,W_3_4,W_2_4,W_3_3,gw,i,j,k)
        W_2_23=W_2_xy(dW_2_3_dx,dW_2_2_dy,W_3_2,W_1_3,W_3_3,W_1_2,gw,i,j,k)
        W_2_24=W_2_xz(dW_2_4_dx,dW_2_2_dz,W_3_2,W_1_4,W_3_4,W_1_2,gw,i,j,k)
        W_2_34=W_2_yz(dW_2_4_dy,dW_2_3_dz,W_3_3,W_1_4,W_3_4,W_1_3,gw,i,j,k)
        W_3_23=W_3_xy(dW_3_3_dx,dW_3_2_dy,W_1_2,W_2_3,W_1_3,W_2_2,gw,i,j,k)
        W_3_24=W_3_xz(dW_3_4_dx,dW_3_2_dz,W_1_2,W_2_4,W_1_4,W_2_2,gw,i,j,k)
        W_3_34=W_3_yz(dW_3_4_dy,dW_3_3_dz,W_1_3,W_2_4,W_1_4,W_2_3,gw,i,j,k)
        Y_2_3=Y_2_y(dY_3_dx,dY_2_dy)
        Y_2_4=Y_2_z(dY_4_dx,dY_2_dz)
        Y_3_4=Y_3_z(dY_4_dy,dY_3_dz)
    #END field strengths#

    ##Verified energy expressions: 2-25-24##

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

    @inbounds A_y[i,j,k] = -(sin(θ_w)*(n_1*(-W_1_24)+n_2*(-W_2_24)+n_3*(-W_3_24))+cos(θ_w)*(-Y_2_4))
    end
    return
end

@views function compute_Az!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    A_z,gw,gy,gp2,vev,lambda,θ_w,dx)
    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
              (blockIdx().y - 1) * blockDim().y + threadIdx().y,
              (blockIdx().z - 1) * blockDim().z + threadIdx().z
    if (i>3 && i<(size(ϕ_1,1)-2) && j>3 && j<(size(ϕ_1,2)-2) && k>3 && k<(size(ϕ_1,3)-2))

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

        W_1_23 =W_1_xy(dW_1_3_dx,dW_1_2_dy,W_2_2,W_3_3,W_2_3,W_3_2,gw,i,j,k)
        W_1_24=W_1_xz(dW_1_4_dx,dW_1_2_dz,W_2_2,W_3_4,W_2_4,W_3_2,gw,i,j,k)
        W_1_34=W_1_yz(dW_1_4_dy,dW_1_3_dz,W_2_3,W_3_4,W_2_4,W_3_3,gw,i,j,k)
        W_2_23=W_2_xy(dW_2_3_dx,dW_2_2_dy,W_3_2,W_1_3,W_3_3,W_1_2,gw,i,j,k)
        W_2_24=W_2_xz(dW_2_4_dx,dW_2_2_dz,W_3_2,W_1_4,W_3_4,W_1_2,gw,i,j,k)
        W_2_34=W_2_yz(dW_2_4_dy,dW_2_3_dz,W_3_3,W_1_4,W_3_4,W_1_3,gw,i,j,k)
        W_3_23=W_3_xy(dW_3_3_dx,dW_3_2_dy,W_1_2,W_2_3,W_1_3,W_2_2,gw,i,j,k)
        W_3_24=W_3_xz(dW_3_4_dx,dW_3_2_dz,W_1_2,W_2_4,W_1_4,W_2_2,gw,i,j,k)
        W_3_34=W_3_yz(dW_3_4_dy,dW_3_3_dz,W_1_3,W_2_4,W_1_4,W_2_3,gw,i,j,k)
        Y_2_3=Y_2_y(dY_3_dx,dY_2_dy)
        Y_2_4=Y_2_z(dY_4_dx,dY_2_dz)
        Y_3_4=Y_3_z(dY_4_dy,dY_3_dz)
    #END field strengths#

    ##Verified energy expressions: 2-25-24##

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

    @inbounds A_z[i,j,k] = -(sin(θ_w)*(n_1*(W_1_23)+n_2*W_2_23+n_3*W_3_23)+cos(θ_w)*Y_2_3)
    end
    return
end

@views function compute_B_2!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    B_x_2,B_y_2,B_z_2,gw,gy,gp2,vev,lambda,θ_w,dx)
    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
              (blockIdx().y - 1) * blockDim().y + threadIdx().y,
              (blockIdx().z - 1) * blockDim().z + threadIdx().z
    if (i>3 && i<(size(ϕ_1,1)-2) && j>3 && j<(size(ϕ_1,2)-2) && k>3 && k<(size(ϕ_1,3)-2))
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

        W_1_23 =W_1_xy(dW_1_3_dx,dW_1_2_dy,W_2_2,W_3_3,W_2_3,W_3_2,gw,i,j,k)
        W_1_24=W_1_xz(dW_1_4_dx,dW_1_2_dz,W_2_2,W_3_4,W_2_4,W_3_2,gw,i,j,k)
        W_1_34=W_1_yz(dW_1_4_dy,dW_1_3_dz,W_2_3,W_3_4,W_2_4,W_3_3,gw,i,j,k)
        W_2_23=W_2_xy(dW_2_3_dx,dW_2_2_dy,W_3_2,W_1_3,W_3_3,W_1_2,gw,i,j,k)
        W_2_24=W_2_xz(dW_2_4_dx,dW_2_2_dz,W_3_2,W_1_4,W_3_4,W_1_2,gw,i,j,k)
        W_2_34=W_2_yz(dW_2_4_dy,dW_2_3_dz,W_3_3,W_1_4,W_3_4,W_1_3,gw,i,j,k)
        W_3_23=W_3_xy(dW_3_3_dx,dW_3_2_dy,W_1_2,W_2_3,W_1_3,W_2_2,gw,i,j,k)
        W_3_24=W_3_xz(dW_3_4_dx,dW_3_2_dz,W_1_2,W_2_4,W_1_4,W_2_2,gw,i,j,k)
        W_3_34=W_3_yz(dW_3_4_dy,dW_3_3_dz,W_1_3,W_2_4,W_1_4,W_2_3,gw,i,j,k)
        Y_2_3=Y_2_y(dY_3_dx,dY_2_dy)
        Y_2_4=Y_2_z(dY_4_dx,dY_2_dz)
        Y_3_4=Y_3_z(dY_4_dy,dY_3_dz)
    #END field strengths#

    ##Verified energy expressions: 2-25-24##

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
    ##Bx_2#
    if ϕ_mag == 0.0
        @inbounds B_x_2[i,j,k] = 0.0
        @inbounds B_y_2[i,j,k] = 0.0
        @inbounds B_z_2[i,j,k] = 0.0
    else
        @inbounds B_x_2[i,j,k] = -(sin(θ_w)*(n_1*W_1_34+n_2*W_2_34+n_3*W_3_34)+cos(θ_w)*Y_3_4
        +(4. *sin(θ_w)/(gw*ϕ_mag^2))*(Dy_ϕ_1*Dz_ϕ_2-Dy_ϕ_2*Dz_ϕ_1
        +Dy_ϕ_3*Dz_ϕ_4-Dy_ϕ_4*Dz_ϕ_3))

        @inbounds B_y_2[i,j,k] = -(sin(θ_w)*(n_1*(-W_1_24)+n_2*(-W_2_24)+n_3*(-W_3_24))+cos(θ_w)*(-Y_2_4)
        +(4. *sin(θ_w)/(gw*ϕ_mag^2))*(Dz_ϕ_1*Dx_ϕ_2-Dz_ϕ_2*Dx_ϕ_1
        +Dz_ϕ_3*Dx_ϕ_4-Dz_ϕ_4*Dx_ϕ_3))

        @inbounds B_z_2[i,j,k] = -(sin(θ_w)*(n_1*(W_1_23)+n_2*W_2_23+n_3*W_3_23)+cos(θ_w)*Y_2_3
        +(4. *sin(θ_w)/(gw*ϕ_mag^2))*(Dx_ϕ_1*Dy_ϕ_2-Dx_ϕ_2*Dy_ϕ_1
        +Dx_ϕ_3*Dy_ϕ_4-Dx_ϕ_4*Dy_ϕ_3))
    end
    end
    return
end

@views function compute_B_2_x!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    B_x_2,gw,gy,gp2,vev,lambda,θ_w,dx)
    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
              (blockIdx().y - 1) * blockDim().y + threadIdx().y,
              (blockIdx().z - 1) * blockDim().z + threadIdx().z
    if (i>3 && i<(size(ϕ_1,1)-2) && j>3 && j<(size(ϕ_1,2)-2) && k>3 && k<(size(ϕ_1,3)-2))
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

        W_1_23 =W_1_xy(dW_1_3_dx,dW_1_2_dy,W_2_2,W_3_3,W_2_3,W_3_2,gw,i,j,k)
        W_1_24=W_1_xz(dW_1_4_dx,dW_1_2_dz,W_2_2,W_3_4,W_2_4,W_3_2,gw,i,j,k)
        W_1_34=W_1_yz(dW_1_4_dy,dW_1_3_dz,W_2_3,W_3_4,W_2_4,W_3_3,gw,i,j,k)
        W_2_23=W_2_xy(dW_2_3_dx,dW_2_2_dy,W_3_2,W_1_3,W_3_3,W_1_2,gw,i,j,k)
        W_2_24=W_2_xz(dW_2_4_dx,dW_2_2_dz,W_3_2,W_1_4,W_3_4,W_1_2,gw,i,j,k)
        W_2_34=W_2_yz(dW_2_4_dy,dW_2_3_dz,W_3_3,W_1_4,W_3_4,W_1_3,gw,i,j,k)
        W_3_23=W_3_xy(dW_3_3_dx,dW_3_2_dy,W_1_2,W_2_3,W_1_3,W_2_2,gw,i,j,k)
        W_3_24=W_3_xz(dW_3_4_dx,dW_3_2_dz,W_1_2,W_2_4,W_1_4,W_2_2,gw,i,j,k)
        W_3_34=W_3_yz(dW_3_4_dy,dW_3_3_dz,W_1_3,W_2_4,W_1_4,W_2_3,gw,i,j,k)
        Y_2_3=Y_2_y(dY_3_dx,dY_2_dy)
        Y_2_4=Y_2_z(dY_4_dx,dY_2_dz)
        Y_3_4=Y_3_z(dY_4_dy,dY_3_dz)
    #END field strengths#

    ##Verified energy expressions: 2-25-24##

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
    ##Bx_2#
    if ϕ_mag == 0.0
        @inbounds B_x_2[i,j,k] = 0.0
    else
        @inbounds B_x_2[i,j,k] = -(sin(θ_w)*(n_1*W_1_34+n_2*W_2_34+n_3*W_3_34)+cos(θ_w)*Y_3_4
        +(4. *sin(θ_w)/(gw*ϕ_mag^2))*(Dy_ϕ_1*Dz_ϕ_2-Dy_ϕ_2*Dz_ϕ_1
        +Dy_ϕ_3*Dz_ϕ_4-Dy_ϕ_4*Dz_ϕ_3))
    end
    end
    return
end

@views function compute_B_2_y!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    B_y_2,gw,gy,gp2,vev,lambda,θ_w,dx)
    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
              (blockIdx().y - 1) * blockDim().y + threadIdx().y,
              (blockIdx().z - 1) * blockDim().z + threadIdx().z
    if (i>3 && i<(size(ϕ_1,1)-2) && j>3 && j<(size(ϕ_1,2)-2) && k>3 && k<(size(ϕ_1,3)-2))
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

        W_1_23 =W_1_xy(dW_1_3_dx,dW_1_2_dy,W_2_2,W_3_3,W_2_3,W_3_2,gw,i,j,k)
        W_1_24=W_1_xz(dW_1_4_dx,dW_1_2_dz,W_2_2,W_3_4,W_2_4,W_3_2,gw,i,j,k)
        W_1_34=W_1_yz(dW_1_4_dy,dW_1_3_dz,W_2_3,W_3_4,W_2_4,W_3_3,gw,i,j,k)
        W_2_23=W_2_xy(dW_2_3_dx,dW_2_2_dy,W_3_2,W_1_3,W_3_3,W_1_2,gw,i,j,k)
        W_2_24=W_2_xz(dW_2_4_dx,dW_2_2_dz,W_3_2,W_1_4,W_3_4,W_1_2,gw,i,j,k)
        W_2_34=W_2_yz(dW_2_4_dy,dW_2_3_dz,W_3_3,W_1_4,W_3_4,W_1_3,gw,i,j,k)
        W_3_23=W_3_xy(dW_3_3_dx,dW_3_2_dy,W_1_2,W_2_3,W_1_3,W_2_2,gw,i,j,k)
        W_3_24=W_3_xz(dW_3_4_dx,dW_3_2_dz,W_1_2,W_2_4,W_1_4,W_2_2,gw,i,j,k)
        W_3_34=W_3_yz(dW_3_4_dy,dW_3_3_dz,W_1_3,W_2_4,W_1_4,W_2_3,gw,i,j,k)
        Y_2_3=Y_2_y(dY_3_dx,dY_2_dy)
        Y_2_4=Y_2_z(dY_4_dx,dY_2_dz)
        Y_3_4=Y_3_z(dY_4_dy,dY_3_dz)
    #END field strengths#

    ##Verified energy expressions: 2-25-24##

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
    ##Bx_2#
    if ϕ_mag == 0.0
        @inbounds B_y_2[i,j,k] = 0.0
    else
        @inbounds B_y_2[i,j,k] = -(sin(θ_w)*(n_1*(-W_1_24)+n_2*(-W_2_24)+n_3*(-W_3_24))+cos(θ_w)*(-Y_2_4)
        +(4. *sin(θ_w)/(gw*ϕ_mag^2))*(Dz_ϕ_1*Dx_ϕ_2-Dz_ϕ_2*Dx_ϕ_1
        +Dz_ϕ_3*Dx_ϕ_4-Dz_ϕ_4*Dx_ϕ_3))
    end
    end
    return
end

@views function compute_B_2_z!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    B_z_2,gw,gy,gp2,vev,lambda,θ_w,dx)
    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
              (blockIdx().y - 1) * blockDim().y + threadIdx().y,
              (blockIdx().z - 1) * blockDim().z + threadIdx().z
    if (i>3 && i<(size(ϕ_1,1)-2) && j>3 && j<(size(ϕ_1,2)-2) && k>3 && k<(size(ϕ_1,3)-2))
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

        W_1_23 =W_1_xy(dW_1_3_dx,dW_1_2_dy,W_2_2,W_3_3,W_2_3,W_3_2,gw,i,j,k)
        W_1_24=W_1_xz(dW_1_4_dx,dW_1_2_dz,W_2_2,W_3_4,W_2_4,W_3_2,gw,i,j,k)
        W_1_34=W_1_yz(dW_1_4_dy,dW_1_3_dz,W_2_3,W_3_4,W_2_4,W_3_3,gw,i,j,k)
        W_2_23=W_2_xy(dW_2_3_dx,dW_2_2_dy,W_3_2,W_1_3,W_3_3,W_1_2,gw,i,j,k)
        W_2_24=W_2_xz(dW_2_4_dx,dW_2_2_dz,W_3_2,W_1_4,W_3_4,W_1_2,gw,i,j,k)
        W_2_34=W_2_yz(dW_2_4_dy,dW_2_3_dz,W_3_3,W_1_4,W_3_4,W_1_3,gw,i,j,k)
        W_3_23=W_3_xy(dW_3_3_dx,dW_3_2_dy,W_1_2,W_2_3,W_1_3,W_2_2,gw,i,j,k)
        W_3_24=W_3_xz(dW_3_4_dx,dW_3_2_dz,W_1_2,W_2_4,W_1_4,W_2_2,gw,i,j,k)
        W_3_34=W_3_yz(dW_3_4_dy,dW_3_3_dz,W_1_3,W_2_4,W_1_4,W_2_3,gw,i,j,k)
        Y_2_3=Y_2_y(dY_3_dx,dY_2_dy)
        Y_2_4=Y_2_z(dY_4_dx,dY_2_dz)
        Y_3_4=Y_3_z(dY_4_dy,dY_3_dz)
    #END field strengths#

    ##Verified energy expressions: 2-25-24##

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
    ##Bx_2#
    if ϕ_mag == 0.0
        @inbounds B_z_2[i,j,k] = 0.0
    else
        @inbounds B_z_2[i,j,k] = -(sin(θ_w)*(n_1*(W_1_23)+n_2*W_2_23+n_3*W_3_23)+cos(θ_w)*Y_2_3
        +(4. *sin(θ_w)/(gw*ϕ_mag^2))*(Dx_ϕ_1*Dy_ϕ_2-Dx_ϕ_2*Dy_ϕ_1
        +Dx_ϕ_3*Dy_ϕ_4-Dx_ϕ_4*Dy_ϕ_3))
    end
    end
    return
end

@views function compute_B_3_x!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    n_1,n_2,n_3,
    modϕ_1,modϕ_2,modϕ_3,modϕ_4,
    B_x_2,gw,gy,gp2,vev,lambda,θ_w,dx)
    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
              (blockIdx().y - 1) * blockDim().y + threadIdx().y,
              (blockIdx().z - 1) * blockDim().z + threadIdx().z
    if (i>3 && i<(size(ϕ_1,1)-2) && j>3 && j<(size(ϕ_1,2)-2) && k>3 && k<(size(ϕ_1,3)-2))
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

        W_1_23 =W_1_xy(dW_1_3_dx,dW_1_2_dy,W_2_2,W_3_3,W_2_3,W_3_2,gw,i,j,k)
        W_1_24=W_1_xz(dW_1_4_dx,dW_1_2_dz,W_2_2,W_3_4,W_2_4,W_3_2,gw,i,j,k)
        W_1_34=W_1_yz(dW_1_4_dy,dW_1_3_dz,W_2_3,W_3_4,W_2_4,W_3_3,gw,i,j,k)
        W_2_23=W_2_xy(dW_2_3_dx,dW_2_2_dy,W_3_2,W_1_3,W_3_3,W_1_2,gw,i,j,k)
        W_2_24=W_2_xz(dW_2_4_dx,dW_2_2_dz,W_3_2,W_1_4,W_3_4,W_1_2,gw,i,j,k)
        W_2_34=W_2_yz(dW_2_4_dy,dW_2_3_dz,W_3_3,W_1_4,W_3_4,W_1_3,gw,i,j,k)
        W_3_23=W_3_xy(dW_3_3_dx,dW_3_2_dy,W_1_2,W_2_3,W_1_3,W_2_2,gw,i,j,k)
        W_3_24=W_3_xz(dW_3_4_dx,dW_3_2_dz,W_1_2,W_2_4,W_1_4,W_2_2,gw,i,j,k)
        W_3_34=W_3_yz(dW_3_4_dy,dW_3_3_dz,W_1_3,W_2_4,W_1_4,W_2_3,gw,i,j,k)
        Y_2_3=Y_2_y(dY_3_dx,dY_2_dy)
        Y_2_4=Y_2_z(dY_4_dx,dY_2_dz)
        Y_3_4=Y_3_z(dY_4_dy,dY_3_dz)
    #END field strengths#

    ##Verified energy expressions: 2-25-24##

    # ϕ_mag = (ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2)

    # # Higgs n definitions

    # if ϕ_mag == 0.0
    #     n_1= 0.0
    #     n_2= 0.0
    #     n_3= 0.0
    # else
    #     n_1=-2. *(ϕ_1[i,j,k]*ϕ_3[i,j,k]+ϕ_2[i,j,k]*ϕ_4[i,j,k])/ϕ_mag
    #     n_2=+2. *(ϕ_2[i,j,k]*ϕ_3[i,j,k]-ϕ_1[i,j,k]*ϕ_4[i,j,k])/ϕ_mag
    #     n_3=(-ϕ_1[i,j,k]^2-ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2)/ϕ_mag
    # end
    
    ##Bx_2#
    # if ϕ_mag == 0.0
    #     @inbounds B_x_2[i,j,k] = 0.0
    # else
    #     @inbounds B_x_2[i,j,k] = -(sin(θ_w)*(n_1*W_1_34+n_2*W_2_34+n_3*W_3_34)+cos(θ_w)*Y_3_4
    #     +(4. *sin(θ_w)/(gw*ϕ_mag^2))*(Dy_ϕ_1*Dz_ϕ_2-Dy_ϕ_2*Dz_ϕ_1
    #     +Dy_ϕ_3*Dz_ϕ_4-Dy_ϕ_4*Dz_ϕ_3))
    # end

    #spatial derivatives of n
        dn_1_dx=dfdx(n_1,i,j,k,0.,dx)
        dn_2_dx=dfdx(n_2,i,j,k,0.,dx)
        dn_3_dx=dfdx(n_3,i,j,k,0.,dx)

        dn_1_dy=dfdy(n_1,i,j,k,0.,dx)
        dn_2_dy=dfdy(n_2,i,j,k,0.,dx)
        dn_3_dy=dfdy(n_3,i,j,k,0.,dx)

        dn_1_dz=dfdz(n_1,i,j,k,0.,dx)
        dn_2_dz=dfdz(n_2,i,j,k,0.,dx)
        dn_3_dz=dfdz(n_3,i,j,k,0.,dx)

    #spatial derivatives of modphi
        dmodϕ_1_dx=dfdx(modϕ_1,i,j,k,0.,dx)
        dmodϕ_2_dx=dfdx(modϕ_2,i,j,k,0.,dx)
        dmodϕ_3_dx=dfdx(modϕ_3,i,j,k,0.,dx)
        dmodϕ_4_dx=dfdx(modϕ_4,i,j,k,0.,dx)
        
        dmodϕ_1_dy=dfdy(modϕ_1,i,j,k,0.,dx)
        dmodϕ_2_dy=dfdy(modϕ_2,i,j,k,0.,dx)
        dmodϕ_3_dy=dfdy(modϕ_3,i,j,k,0.,dx)
        dmodϕ_4_dy=dfdy(modϕ_4,i,j,k,0.,dx)

        dmodϕ_1_dz=dfdz(modϕ_1,i,j,k,0.,dx)
        dmodϕ_2_dz=dfdz(modϕ_2,i,j,k,0.,dx)
        dmodϕ_3_dz=dfdz(modϕ_3,i,j,k,0.,dx)
        dmodϕ_4_dz=dfdz(modϕ_4,i,j,k,0.,dx)

    @inbounds B_x_2[i,j,k] = -(sin(θ_w)*((dn_1_dy*W_1_4[i,j,k]+dn_2_dy*W_2_4[i,j,k]+dn_3_dy*W_2_4[i,j,k])-
                                            (dn_1_dz*W_1_3[i,j,k]+dn_2_dz*W_2_3[i,j,k]+dn_3_dz*W_2_3[i,j,k])+
                                            (n_1[i,j,k]*dW_1_4_dy+n_2[i,j,k]*dW_2_4_dy+n_3[i,j,k]*dW_2_4_dy)-
                                            (n_1[i,j,k]*dW_1_3_dz+n_2[i,j,k]*dW_2_3_dz+n_3[i,j,k]*dW_2_3_dz))+
                                cos(θ_w)*(dY_4_dy-dY_3_dz)
                                +(4. *sin(θ_w)/(gw))*(dmodϕ_1_dy*dmodϕ_2_dz-dmodϕ_2_dy*dmodϕ_1_dz
                                                            + dmodϕ_3_dy*dmodϕ_4_dz-dmodϕ_4_dy*dmodϕ_3_dz))

    # @inbounds B_x_2[i,j,k] = -(sin(θ_w)*((dn_1_dy*W_1_4[i,j,k]+dn_2_dy*W_2_4[i,j,k]+dn_3_dy*W_3_4[i,j,k])-
    #                                     (dn_1_dz*W_1_3[i,j,k]+dn_2_dz*W_2_3[i,j,k]+dn_3_dz*W_3_3[i,j,k])+
    #                                     (n_1[i,j,k]*dW_1_4_dy+n_2[i,j,k]*dW_2_4_dy+n_3[i,j,k]*dW_3_4_dy)-
    #                                     (n_1[i,j,k]*dW_1_3_dz+n_2[i,j,k]*dW_2_3_dz+n_3[i,j,k]*dW_3_3_dz))+
    #                         cos(θ_w)*(dY_4_dy-dY_3_dz)-
    #                         (2.0*sin(θ_w)/(gw))*(n_1[i,j,k]*dn_2_dy*dn_3_dz-n_1[i,j,k]*dn_3_dy*dn_2_dz+
    #                                              n_2[i,j,k]*dn_3_dy*dn_1_dz-n_2[i,j,k]*dn_1_dy*dn_3_dz+
    #                                              n_3[i,j,k]*dn_1_dy*dn_2_dz-n_3[i,j,k]*dn_2_dy*dn_1_dz))
    end
    return
end

@views function compute_B_3_y!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    n_1,n_2,n_3,
    modϕ_1,modϕ_2,modϕ_3,modϕ_4,
    B_y_2,gw,gy,gp2,vev,lambda,θ_w,dx)
    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
              (blockIdx().y - 1) * blockDim().y + threadIdx().y,
              (blockIdx().z - 1) * blockDim().z + threadIdx().z
    if (i>3 && i<(size(ϕ_1,1)-2) && j>3 && j<(size(ϕ_1,2)-2) && k>3 && k<(size(ϕ_1,3)-2))
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

        W_1_23 =W_1_xy(dW_1_3_dx,dW_1_2_dy,W_2_2,W_3_3,W_2_3,W_3_2,gw,i,j,k)
        W_1_24=W_1_xz(dW_1_4_dx,dW_1_2_dz,W_2_2,W_3_4,W_2_4,W_3_2,gw,i,j,k)
        W_1_34=W_1_yz(dW_1_4_dy,dW_1_3_dz,W_2_3,W_3_4,W_2_4,W_3_3,gw,i,j,k)
        W_2_23=W_2_xy(dW_2_3_dx,dW_2_2_dy,W_3_2,W_1_3,W_3_3,W_1_2,gw,i,j,k)
        W_2_24=W_2_xz(dW_2_4_dx,dW_2_2_dz,W_3_2,W_1_4,W_3_4,W_1_2,gw,i,j,k)
        W_2_34=W_2_yz(dW_2_4_dy,dW_2_3_dz,W_3_3,W_1_4,W_3_4,W_1_3,gw,i,j,k)
        W_3_23=W_3_xy(dW_3_3_dx,dW_3_2_dy,W_1_2,W_2_3,W_1_3,W_2_2,gw,i,j,k)
        W_3_24=W_3_xz(dW_3_4_dx,dW_3_2_dz,W_1_2,W_2_4,W_1_4,W_2_2,gw,i,j,k)
        W_3_34=W_3_yz(dW_3_4_dy,dW_3_3_dz,W_1_3,W_2_4,W_1_4,W_2_3,gw,i,j,k)
        Y_2_3=Y_2_y(dY_3_dx,dY_2_dy)
        Y_2_4=Y_2_z(dY_4_dx,dY_2_dz)
        Y_3_4=Y_3_z(dY_4_dy,dY_3_dz)
    #END field strengths#

    ##Verified energy expressions: 2-25-24##

    # ϕ_mag = (ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2)

    # # Higgs n definitions

    # if ϕ_mag == 0.0
    #     n_1= 0.0
    #     n_2= 0.0
    #     n_3= 0.0
    # else
    #     n_1=-2. *(ϕ_1[i,j,k]*ϕ_3[i,j,k]+ϕ_2[i,j,k]*ϕ_4[i,j,k])/ϕ_mag
    #     n_2=+2. *(ϕ_2[i,j,k]*ϕ_3[i,j,k]-ϕ_1[i,j,k]*ϕ_4[i,j,k])/ϕ_mag
    #     n_3=(-ϕ_1[i,j,k]^2-ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2)/ϕ_mag
    # end
    
    ##Bx_2#
    # if ϕ_mag == 0.0
    #     @inbounds B_x_2[i,j,k] = 0.0
    # else
    #     @inbounds B_x_2[i,j,k] = -(sin(θ_w)*(n_1*W_1_34+n_2*W_2_34+n_3*W_3_34)+cos(θ_w)*Y_3_4
    #     +(4. *sin(θ_w)/(gw*ϕ_mag^2))*(Dy_ϕ_1*Dz_ϕ_2-Dy_ϕ_2*Dz_ϕ_1
    #     +Dy_ϕ_3*Dz_ϕ_4-Dy_ϕ_4*Dz_ϕ_3))
    # end

    #spatial derivatives of n
        dn_1_dx=dfdx(n_1,i,j,k,0.,dx)
        dn_2_dx=dfdx(n_2,i,j,k,0.,dx)
        dn_3_dx=dfdx(n_3,i,j,k,0.,dx)

        dn_1_dy=dfdy(n_1,i,j,k,0.,dx)
        dn_2_dy=dfdy(n_2,i,j,k,0.,dx)
        dn_3_dy=dfdy(n_3,i,j,k,0.,dx)

        dn_1_dz=dfdz(n_1,i,j,k,0.,dx)
        dn_2_dz=dfdz(n_2,i,j,k,0.,dx)
        dn_3_dz=dfdz(n_3,i,j,k,0.,dx)

    #spatial derivatives of modphi
        dmodϕ_1_dx=dfdx(modϕ_1,i,j,k,0.,dx)
        dmodϕ_2_dx=dfdx(modϕ_2,i,j,k,0.,dx)
        dmodϕ_3_dx=dfdx(modϕ_3,i,j,k,0.,dx)
        dmodϕ_4_dx=dfdx(modϕ_4,i,j,k,0.,dx)
        
        dmodϕ_1_dy=dfdy(modϕ_1,i,j,k,0.,dx)
        dmodϕ_2_dy=dfdy(modϕ_2,i,j,k,0.,dx)
        dmodϕ_3_dy=dfdy(modϕ_3,i,j,k,0.,dx)
        dmodϕ_4_dy=dfdy(modϕ_4,i,j,k,0.,dx)

        dmodϕ_1_dz=dfdz(modϕ_1,i,j,k,0.,dx)
        dmodϕ_2_dz=dfdz(modϕ_2,i,j,k,0.,dx)
        dmodϕ_3_dz=dfdz(modϕ_3,i,j,k,0.,dx)
        dmodϕ_4_dz=dfdz(modϕ_4,i,j,k,0.,dx)

    @inbounds B_y_2[i,j,k] = -(sin(θ_w)*((dn_1_dz*W_1_2[i,j,k]+dn_2_dz*W_2_2[i,j,k]+dn_3_dz*W_3_2[i,j,k])-
                                            (dn_1_dx*W_1_4[i,j,k]+dn_2_dx*W_2_4[i,j,k]+dn_3_dx*W_3_4[i,j,k])+
                                            (n_1[i,j,k]*dW_1_2_dz+n_2[i,j,k]*dW_2_2_dz+n_3[i,j,k]*dW_3_2_dz)-
                                            (n_1[i,j,k]*dW_1_4_dx+n_2[i,j,k]*dW_2_4_dx+n_3[i,j,k]*dW_3_4_dx))+
                                cos(θ_w)*(dY_2_dz-dY_4_dx)+
                                (4. *sin(θ_w)/(gw))*(dmodϕ_1_dz*dmodϕ_2_dx-dmodϕ_2_dz*dmodϕ_1_dx
                                                     +dmodϕ_3_dz*dmodϕ_4_dx-dmodϕ_4_dz*dmodϕ_3_dx))

    # @inbounds B_y_2[i,j,k] = -(sin(θ_w)*((dn_1_dz*W_1_2[i,j,k]+dn_2_dz*W_2_2[i,j,k]+dn_3_dz*W_3_2[i,j,k])-
    #                                     (dn_1_dx*W_1_4[i,j,k]+dn_2_dx*W_2_4[i,j,k]+dn_3_dx*W_3_4[i,j,k])+
    #                                     (n_1[i,j,k]*dW_1_2_dz+n_2[i,j,k]*dW_2_2_dz+n_3[i,j,k]*dW_3_2_dz)-
    #                                     (n_1[i,j,k]*dW_1_4_dx+n_2[i,j,k]*dW_2_4_dx+n_3[i,j,k]*dW_3_4_dx))+
    #                         cos(θ_w)*(dY_2_dz-dY_4_dx)-
    #                         (2.0*sin(θ_w)/(gw))*(n_1[i,j,k]*dn_2_dz*dn_3_dx-n_1[i,j,k]*dn_3_dz*dn_2_dx+
    #                                              n_2[i,j,k]*dn_3_dz*dn_1_dx-n_2[i,j,k]*dn_1_dz*dn_3_dx+
    #                                              n_3[i,j,k]*dn_1_dz*dn_2_dx-n_3[i,j,k]*dn_2_dz*dn_1_dx))
    end
    return
end

@views function compute_B_3_z!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_1_3,W_1_4,
    W_2_2,W_2_3,W_2_4,
    W_3_2,W_3_3,W_3_4,
    Y_2,Y_3,Y_4,
    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
    dY_2_dt,dY_3_dt,dY_4_dt,
    n_1,n_2,n_3,
    modϕ_1,modϕ_2,modϕ_3,modϕ_4,
    B_z_2,gw,gy,gp2,vev,lambda,θ_w,dx)
    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
              (blockIdx().y - 1) * blockDim().y + threadIdx().y,
              (blockIdx().z - 1) * blockDim().z + threadIdx().z
    if (i>3 && i<(size(ϕ_1,1)-2) && j>3 && j<(size(ϕ_1,2)-2) && k>3 && k<(size(ϕ_1,3)-2))
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

        W_1_23 =W_1_xy(dW_1_3_dx,dW_1_2_dy,W_2_2,W_3_3,W_2_3,W_3_2,gw,i,j,k)
        W_1_24=W_1_xz(dW_1_4_dx,dW_1_2_dz,W_2_2,W_3_4,W_2_4,W_3_2,gw,i,j,k)
        W_1_34=W_1_yz(dW_1_4_dy,dW_1_3_dz,W_2_3,W_3_4,W_2_4,W_3_3,gw,i,j,k)
        W_2_23=W_2_xy(dW_2_3_dx,dW_2_2_dy,W_3_2,W_1_3,W_3_3,W_1_2,gw,i,j,k)
        W_2_24=W_2_xz(dW_2_4_dx,dW_2_2_dz,W_3_2,W_1_4,W_3_4,W_1_2,gw,i,j,k)
        W_2_34=W_2_yz(dW_2_4_dy,dW_2_3_dz,W_3_3,W_1_4,W_3_4,W_1_3,gw,i,j,k)
        W_3_23=W_3_xy(dW_3_3_dx,dW_3_2_dy,W_1_2,W_2_3,W_1_3,W_2_2,gw,i,j,k)
        W_3_24=W_3_xz(dW_3_4_dx,dW_3_2_dz,W_1_2,W_2_4,W_1_4,W_2_2,gw,i,j,k)
        W_3_34=W_3_yz(dW_3_4_dy,dW_3_3_dz,W_1_3,W_2_4,W_1_4,W_2_3,gw,i,j,k)
        Y_2_3=Y_2_y(dY_3_dx,dY_2_dy)
        Y_2_4=Y_2_z(dY_4_dx,dY_2_dz)
        Y_3_4=Y_3_z(dY_4_dy,dY_3_dz)
    #END field strengths#

    ##Verified energy expressions: 2-25-24##

    # ϕ_mag = (ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2)

    # # Higgs n definitions

    # if ϕ_mag == 0.0
    #     n_1= 0.0
    #     n_2= 0.0
    #     n_3= 0.0
    # else
    #     n_1=-2. *(ϕ_1[i,j,k]*ϕ_3[i,j,k]+ϕ_2[i,j,k]*ϕ_4[i,j,k])/ϕ_mag
    #     n_2=+2. *(ϕ_2[i,j,k]*ϕ_3[i,j,k]-ϕ_1[i,j,k]*ϕ_4[i,j,k])/ϕ_mag
    #     n_3=(-ϕ_1[i,j,k]^2-ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2)/ϕ_mag
    # end
    
    ##Bx_2#
    # if ϕ_mag == 0.0
    #     @inbounds B_x_2[i,j,k] = 0.0
    # else
    #     @inbounds B_x_2[i,j,k] = -(sin(θ_w)*(n_1*W_1_34+n_2*W_2_34+n_3*W_3_34)+cos(θ_w)*Y_3_4
    #     +(4. *sin(θ_w)/(gw*ϕ_mag^2))*(Dy_ϕ_1*Dz_ϕ_2-Dy_ϕ_2*Dz_ϕ_1
    #     +Dy_ϕ_3*Dz_ϕ_4-Dy_ϕ_4*Dz_ϕ_3))
    # end

    #spatial derivatives of n
        dn_1_dx=dfdx(n_1,i,j,k,0.,dx)
        dn_2_dx=dfdx(n_2,i,j,k,0.,dx)
        dn_3_dx=dfdx(n_3,i,j,k,0.,dx)

        dn_1_dy=dfdy(n_1,i,j,k,0.,dx)
        dn_2_dy=dfdy(n_2,i,j,k,0.,dx)
        dn_3_dy=dfdy(n_3,i,j,k,0.,dx)

        dn_1_dz=dfdz(n_1,i,j,k,0.,dx)
        dn_2_dz=dfdz(n_2,i,j,k,0.,dx)
        dn_3_dz=dfdz(n_3,i,j,k,0.,dx)

    #spatial derivatives of modphi
        dmodϕ_1_dx=dfdx(modϕ_1,i,j,k,0.,dx)
        dmodϕ_2_dx=dfdx(modϕ_2,i,j,k,0.,dx)
        dmodϕ_3_dx=dfdx(modϕ_3,i,j,k,0.,dx)
        dmodϕ_4_dx=dfdx(modϕ_4,i,j,k,0.,dx)
        
        dmodϕ_1_dy=dfdy(modϕ_1,i,j,k,0.,dx)
        dmodϕ_2_dy=dfdy(modϕ_2,i,j,k,0.,dx)
        dmodϕ_3_dy=dfdy(modϕ_3,i,j,k,0.,dx)
        dmodϕ_4_dy=dfdy(modϕ_4,i,j,k,0.,dx)

        dmodϕ_1_dz=dfdz(modϕ_1,i,j,k,0.,dx)
        dmodϕ_2_dz=dfdz(modϕ_2,i,j,k,0.,dx)
        dmodϕ_3_dz=dfdz(modϕ_3,i,j,k,0.,dx)
        dmodϕ_4_dz=dfdz(modϕ_4,i,j,k,0.,dx)

    @inbounds B_z_2[i,j,k] = -(sin(θ_w)*((dn_1_dx*W_1_3[i,j,k]+dn_2_dx*W_2_3[i,j,k]+dn_3_dx*W_3_3[i,j,k])-
                                            (dn_1_dy*W_1_2[i,j,k]+dn_2_dy*W_2_2[i,j,k]+dn_3_dy*W_3_2[i,j,k])+
                                            (n_1[i,j,k]*dW_1_3_dx+n_2[i,j,k]*dW_2_3_dx+n_3[i,j,k]*dW_3_3_dx)-
                                            (n_1[i,j,k]*dW_1_2_dy+n_2[i,j,k]*dW_2_2_dy+n_3[i,j,k]*dW_3_2_dy))+
                                cos(θ_w)*(dY_3_dx-dY_2_dy)+
                                (4. *sin(θ_w)/(gw))*(dmodϕ_1_dx*dmodϕ_2_dy-dmodϕ_2_dx*dmodϕ_1_dy
                                                     +dmodϕ_3_dx*dmodϕ_4_dy-dmodϕ_4_dx*dmodϕ_3_dy))
    # @inbounds B_z_2[i,j,k] = -(sin(θ_w)*((dn_1_dx*W_1_3[i,j,k]+dn_2_dx*W_2_3[i,j,k]+dn_3_dx*W_3_3[i,j,k])-
    #                                         (dn_1_dy*W_1_2[i,j,k]+dn_2_dy*W_2_2[i,j,k]+dn_3_dy*W_3_2[i,j,k])+
    #                                         (n_1[i,j,k]*dW_1_3_dx+n_2[i,j,k]*dW_2_3_dx+n_3[i,j,k]*dW_3_3_dx)-
    #                                         (n_1[i,j,k]*dW_1_2_dy+n_2[i,j,k]*dW_2_2_dy+n_3[i,j,k]*dW_3_2_dy))+
    #                             cos(θ_w)*(dY_3_dx-dY_2_dy)-
    #                             (2.0*sin(θ_w)/(gw))*(n_1[i,j,k]*dn_2_dx*dn_3_dy-n_1[i,j,k]*dn_3_dx*dn_2_dy+
    #                                                  n_2[i,j,k]*dn_3_dx*dn_1_dy-n_2[i,j,k]*dn_1_dx*dn_3_dy+
    #                                                  n_3[i,j,k]*dn_1_dx*dn_2_dy-n_3[i,j,k]*dn_2_dx*dn_1_dy))
    end
    return
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

function global_idxs(i,j,k,grid_x,grid_y,grid_z,Nx,Ny,Nz,Nx_g,Ny_g,Nz_g)
    i_g=(i-3)+(grid_x)*(Nx-6)
    j_g=(j-3)+(grid_y)*(Ny-6)
    k_g=(k-3)+(grid_z)*(Nz-6)

    if i_g<1
        i_g = Nx_g - i_g
    elseif i_g>Nx_g
        i_g = i_g - Nx_g
    end

    if j_g<1
        j_g = Nx_g - j_g
    elseif j_g>Nx_g
        j_g = j_g - Nx_g
    end

    if k_g<1
        k_g = Nx_g - k_g
    elseif k_g>Nx_g
        k_g = k_g - Nx_g
    end
    return i_g,j_g,k_g
end

function thermal_initializer_ϕ!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
    gw,gy,gp2,vev,dx,T,meff_sq,
    Nx,Ny,Nz,Nx_g,Ny_g,Nz_g,grid_x,grid_y,grid_z)
    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
    (blockIdx().y - 1) * blockDim().y + threadIdx().y,
    (blockIdx().z - 1) * blockDim().z + threadIdx().z

    i_g,j_g,k_g = global_idxs(i,j,k,grid_x,grid_y,grid_z,Nx,Ny,Nz,Nx_g,Ny_g,Nz_g)
    
    kx = 2.0*sin(0.5*i_g*2.0*pi/Nx_g)/dx
    ky = 2.0*sin(0.5*j_g*2.0*pi/Ny_g)/dx
    kz = 2.0*sin(0.5*k_g*2.0*pi/Nz_g)/dx

    mod_k = sqrt(2.0*((1.0-cos(i_g*2.0*pi/Nx_g))/(dx^2)+
                 (1.0-cos(j_g*2.0*pi/Ny_g))/(dx^2)+
                 (1.0-cos(k_g*2.0*pi/Nz_g))/(dx^2)))

    # ω_p = sqrt(k_mag(i,j,k,Nx_g,Ny_g,Nz_g,dx)^2+32.0)
    ω_p = sqrt(mod_k^2+meff_sq)
    n_p = 1.0/(exp(ω_p/T)-1.0)
    vol = Nx_g*Ny_g*Nz_g*dx^3
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
    Nx,Ny,Nz,Nx_g,Ny_g,Nz_g,grid_x,grid_y,grid_z,
    r_1,r_2,r_3,
    r_1_i,r_2_i,r_3_i,
    x_1,x_2,x_3,x_4)

    # i_g = (ix-2)+(grid_x)*(size(r_1,1)-2)
    # j_g = (iy-2)+(grid_y)*(size(r_1,2)-2)
    # k_g = (iz-2)+(grid_z)*(size(r_1,3)-2)

    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
    (blockIdx().y - 1) * blockDim().y + threadIdx().y,
    (blockIdx().z - 1) * blockDim().z + threadIdx().z

    i_g,j_g,k_g = global_idxs(i,j,k,grid_x,grid_y,grid_z,Nx,Ny,Nz,Nx_g,Ny_g,Nz_g)
    
    kx = 2.0*sin(0.5*i_g*2.0*pi/Nx_g)/dx
    ky = 2.0*sin(0.5*j_g*2.0*pi/Ny_g)/dx
    kz = 2.0*sin(0.5*k_g*2.0*pi/Nz_g)/dx

    mod_k = sqrt(2.0*((1.0-cos(i_g*2.0*pi/Nx_g))/(dx^2)+
                 (1.0-cos(j_g*2.0*pi/Ny_g))/(dx^2)+
                 (1.0-cos(k_g*2.0*pi/Nz_g))/(dx^2)))

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
    vol = Nx_g*Ny_g*Nz_g*dx^3

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

function updater_1!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
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
    k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,fac)

    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
    (blockIdx().y - 1) * blockDim().y + threadIdx().y,
    (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i>3 && i<size(size(ϕ_1,1)-2) && j>3 && j<size(size(ϕ_1,2)-2) && k>3 && k<size(size(ϕ_1,3)-2))

    @inbounds ϕ_1_n[i,j,k] = dϕ_1_dt[i,j,k] *fac + ϕ_1[i,j,k]
    @inbounds ϕ_2_n[i,j,k] = dϕ_2_dt[i,j,k] *fac + ϕ_2[i,j,k]
    @inbounds ϕ_3_n[i,j,k] = dϕ_3_dt[i,j,k] *fac + ϕ_3[i,j,k]
    @inbounds ϕ_4_n[i,j,k] = dϕ_4_dt[i,j,k] *fac + ϕ_4[i,j,k]
    @inbounds W_1_2_n[i,j,k] = dW_1_2_dt[i,j,k] *fac + W_1_2[i,j,k]
    @inbounds W_1_3_n[i,j,k] = dW_1_3_dt[i,j,k] *fac + W_1_3[i,j,k]
    @inbounds W_1_4_n[i,j,k] = dW_1_4_dt[i,j,k] *fac + W_1_4[i,j,k]
    @inbounds W_2_2_n[i,j,k] = dW_2_2_dt[i,j,k] *fac + W_2_2[i,j,k]
    @inbounds W_2_3_n[i,j,k] = dW_2_3_dt[i,j,k] *fac + W_2_3[i,j,k]
    @inbounds W_2_4_n[i,j,k] = dW_2_4_dt[i,j,k] *fac + W_2_4[i,j,k]
    @inbounds W_3_2_n[i,j,k] = dW_3_2_dt[i,j,k] *fac + W_3_2[i,j,k]
    @inbounds W_3_3_n[i,j,k] = dW_3_3_dt[i,j,k] *fac + W_3_3[i,j,k]
    @inbounds W_3_4_n[i,j,k] = dW_3_4_dt[i,j,k] *fac + W_3_4[i,j,k]
    @inbounds Y_2_n[i,j,k] = dY_2_dt[i,j,k] *fac + Y_2[i,j,k]
    @inbounds Y_3_n[i,j,k] = dY_3_dt[i,j,k] *fac + Y_3[i,j,k]
    @inbounds Y_4_n[i,j,k] = dY_4_dt[i,j,k] *fac + Y_4[i,j,k]
    @inbounds Γ_1_n[i,j,k] = k_Γ_1[i,j,k] *fac + Γ_1[i,j,k]
    @inbounds Γ_2_n[i,j,k] = k_Γ_2[i,j,k] *fac + Γ_2[i,j,k]
    @inbounds Γ_3_n[i,j,k] = k_Γ_3[i,j,k] *fac + Γ_3[i,j,k]
    @inbounds Σ_n[i,j,k] = k_Σ[i,j,k] *fac + Σ[i,j,k]

    end
    return
end

function updater_t_1!(dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
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
    kt_Y_2,kt_Y_3,kt_Y_4,fac)

    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
    (blockIdx().y - 1) * blockDim().y + threadIdx().y,
    (blockIdx().z - 1) * blockDim().z + threadIdx().z
    if (i>3 && i<size(size(ϕ_1,1)-2) && j>3 && j<size(size(ϕ_1,2)-2) && k>3 && k<size(size(ϕ_1,3)-2))

    @inbounds dϕ_1_dt_n[i,j,k] = kt_ϕ_1[i,j,k] *fac + dϕ_1_dt[i,j,k]
    @inbounds dϕ_2_dt_n[i,j,k] = kt_ϕ_2[i,j,k] *fac + dϕ_2_dt[i,j,k]
    @inbounds dϕ_3_dt_n[i,j,k] = kt_ϕ_3[i,j,k] *fac + dϕ_3_dt[i,j,k]
    @inbounds dϕ_4_dt_n[i,j,k] = kt_ϕ_4[i,j,k] *fac + dϕ_4_dt[i,j,k]
    @inbounds dW_1_2_dt_n[i,j,k] = kt_W_1_2[i,j,k] *fac + dW_1_2_dt[i,j,k]
    @inbounds dW_1_3_dt_n[i,j,k] = kt_W_1_3[i,j,k] *fac + dW_1_3_dt[i,j,k]
    @inbounds dW_1_4_dt_n[i,j,k] = kt_W_1_4[i,j,k] *fac + dW_1_4_dt[i,j,k]
    @inbounds dW_2_2_dt_n[i,j,k] = kt_W_2_2[i,j,k] *fac + dW_2_2_dt[i,j,k]
    @inbounds dW_2_3_dt_n[i,j,k] = kt_W_2_3[i,j,k] *fac + dW_2_3_dt[i,j,k]
    @inbounds dW_2_4_dt_n[i,j,k] = kt_W_2_4[i,j,k] *fac + dW_2_4_dt[i,j,k]
    @inbounds dW_3_2_dt_n[i,j,k] = kt_W_3_2[i,j,k] *fac + dW_3_2_dt[i,j,k]
    @inbounds dW_3_3_dt_n[i,j,k] = kt_W_3_3[i,j,k] *fac + dW_3_3_dt[i,j,k]
    @inbounds dW_3_4_dt_n[i,j,k] = kt_W_3_4[i,j,k] *fac + dW_3_4_dt[i,j,k]
    @inbounds dY_2_dt_n[i,j,k] = kt_Y_2[i,j,k] *fac + dY_2_dt[i,j,k]
    @inbounds dY_3_dt_n[i,j,k] = kt_Y_3[i,j,k] *fac + dY_3_dt[i,j,k]
    @inbounds dY_4_dt_n[i,j,k] = kt_Y_4[i,j,k] *fac + dY_4_dt[i,j,k]
    # dΓ_1_dt_n = kt_Γ_1 *fac + dΓ_1_dt
    # dΓ_2_dt_n = kt_Γ_2 *fac + dΓ_2_dt
    # dΓ_3_dt_n = kt_Γ_3 *fac + dΓ_3_dt
    # dΣ_dt_n = kt_Σ *fac + dΣ_dt
    end
    return
end

function updater!(ϕ_1_n,ϕ_2_n,ϕ_3_n,ϕ_4_n,
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
    k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,fac)

    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
    (blockIdx().y - 1) * blockDim().y + threadIdx().y,
    (blockIdx().z - 1) * blockDim().z + threadIdx().z
    if (i>3 && i<size(size(ϕ_1,1)-2) && j>3 && j<size(size(ϕ_1,2)-2) && k>3 && k<size(size(ϕ_1,3)-2))

    @inbounds ϕ_1_n[i,j,k] = dϕ_1_dt[i,j,k] *fac + ϕ_1_n[i,j,k]
    @inbounds ϕ_2_n[i,j,k] = dϕ_2_dt[i,j,k] *fac + ϕ_2_n[i,j,k]
    @inbounds ϕ_3_n[i,j,k] = dϕ_3_dt[i,j,k] *fac + ϕ_3_n[i,j,k]
    @inbounds ϕ_4_n[i,j,k] = dϕ_4_dt[i,j,k] *fac + ϕ_4_n[i,j,k]
    @inbounds W_1_2_n[i,j,k] = dW_1_2_dt[i,j,k] *fac + W_1_2_n[i,j,k]
    @inbounds W_1_3_n[i,j,k] = dW_1_3_dt[i,j,k] *fac + W_1_3_n[i,j,k]
    @inbounds W_1_4_n[i,j,k] = dW_1_4_dt[i,j,k] *fac + W_1_4_n[i,j,k]
    @inbounds W_2_2_n[i,j,k] = dW_2_2_dt[i,j,k] *fac + W_2_2_n[i,j,k]
    @inbounds W_2_3_n[i,j,k] = dW_2_3_dt[i,j,k] *fac + W_2_3_n[i,j,k]
    @inbounds W_2_4_n[i,j,k] = dW_2_4_dt[i,j,k] *fac + W_2_4_n[i,j,k]
    @inbounds W_3_2_n[i,j,k] = dW_3_2_dt[i,j,k] *fac + W_3_2_n[i,j,k]
    @inbounds W_3_3_n[i,j,k] = dW_3_3_dt[i,j,k] *fac + W_3_3_n[i,j,k]
    @inbounds W_3_4_n[i,j,k] = dW_3_4_dt[i,j,k] *fac + W_3_4_n[i,j,k]
    @inbounds Y_2_n[i,j,k] = dY_2_dt[i,j,k] *fac + Y_2_n[i,j,k]
    @inbounds Y_3_n[i,j,k] = dY_3_dt[i,j,k] *fac + Y_3_n[i,j,k]
    @inbounds Y_4_n[i,j,k] = dY_4_dt[i,j,k] *fac + Y_4_n[i,j,k]
    @inbounds Γ_1_n[i,j,k] = k_Γ_1[i,j,k] *fac + Γ_1_n[i,j,k]
    @inbounds Γ_2_n[i,j,k] = k_Γ_2[i,j,k] *fac + Γ_2_n[i,j,k]
    @inbounds Γ_3_n[i,j,k] = k_Γ_3[i,j,k] *fac + Γ_3_n[i,j,k]
    @inbounds Σ_n[i,j,k] = k_Σ[i,j,k] *fac + Σ_n[i,j,k]
    end
    return
end

function updater_t!(dϕ_1_dt_n,dϕ_2_dt_n,dϕ_3_dt_n,dϕ_4_dt_n,
    dW_1_2_dt_n,dW_1_3_dt_n,dW_1_4_dt_n,
    dW_2_2_dt_n,dW_2_3_dt_n,dW_2_4_dt_n,
    dW_3_2_dt_n,dW_3_3_dt_n,dW_3_4_dt_n,
    dY_2_dt_n,dY_3_dt_n,dY_4_dt_n,
    kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
    kt_W_1_2,kt_W_1_3,kt_W_1_4,
    kt_W_2_2,kt_W_2_3,kt_W_2_4,
    kt_W_3_2,kt_W_3_3,kt_W_3_4,
    kt_Y_2,kt_Y_3,kt_Y_4,fac)

    i, j, k = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
    (blockIdx().y - 1) * blockDim().y + threadIdx().y,
    (blockIdx().z - 1) * blockDim().z + threadIdx().z
    if (i>3 && i<size(size(ϕ_1,1)-2) && j>3 && j<size(size(ϕ_1,2)-2) && k>3 && k<size(size(ϕ_1,3)-2))

    @inbounds dϕ_1_dt_n[i,j,k] = kt_ϕ_1[i,j,k] *fac + dϕ_1_dt_n[i,j,k]
    @inbounds dϕ_2_dt_n[i,j,k] = kt_ϕ_2[i,j,k] *fac + dϕ_2_dt_n[i,j,k]
    @inbounds dϕ_3_dt_n[i,j,k] = kt_ϕ_3[i,j,k] *fac + dϕ_3_dt_n[i,j,k]
    @inbounds dϕ_4_dt_n[i,j,k] = kt_ϕ_4[i,j,k] *fac + dϕ_4_dt_n[i,j,k]
    @inbounds dW_1_2_dt_n[i,j,k] = kt_W_1_2[i,j,k] *fac + dW_1_2_dt_n[i,j,k]
    @inbounds dW_1_3_dt_n[i,j,k] = kt_W_1_3[i,j,k] *fac + dW_1_3_dt_n[i,j,k]
    @inbounds dW_1_4_dt_n[i,j,k] = kt_W_1_4[i,j,k] *fac + dW_1_4_dt_n[i,j,k]
    @inbounds dW_2_2_dt_n[i,j,k] = kt_W_2_2[i,j,k] *fac + dW_2_2_dt_n[i,j,k]
    @inbounds dW_2_3_dt_n[i,j,k] = kt_W_2_3[i,j,k] *fac + dW_2_3_dt_n[i,j,k]
    @inbounds dW_2_4_dt_n[i,j,k] = kt_W_2_4[i,j,k] *fac + dW_2_4_dt_n[i,j,k]
    @inbounds dW_3_2_dt_n[i,j,k] = kt_W_3_2[i,j,k] *fac + dW_3_2_dt_n[i,j,k]
    @inbounds dW_3_3_dt_n[i,j,k] = kt_W_3_3[i,j,k] *fac + dW_3_3_dt_n[i,j,k]
    @inbounds dW_3_4_dt_n[i,j,k] = kt_W_3_4[i,j,k] *fac + dW_3_4_dt_n[i,j,k]
    @inbounds dY_2_dt_n[i,j,k] = kt_Y_2[i,j,k] *fac + dY_2_dt_n[i,j,k]
    @inbounds dY_3_dt_n[i,j,k] = kt_Y_3[i,j,k] *fac + dY_3_dt_n[i,j,k]
    @inbounds dY_4_dt_n[i,j,k] = kt_Y_4[i,j,k] *fac + dY_4_dt_n[i,j,k]
    # dΓ_1_dt_n = kt_Γ_1 *fac + dΓ_1_dt
    # dΓ_2_dt_n = kt_Γ_2 *fac + dΓ_2_dt
    # dΓ_3_dt_n = kt_Γ_3 *fac + dΓ_3_dt
    # dΣ_dt_n = kt_Σ *fac + dΣ_dt
    end
    return
end

function J_ext(i,dt,mH,t_sat;thermal_init=false)

    if ((i*dt*mH <= t_sat) && (thermal_init==true))
        J = mH^2
    else
        J = 0.0
    end
    return J
end

function damp(i,dt,mH,t_sat;thermal_init=false)

    if ((i*dt*mH < t_sat) && (thermal_init==true))
        d = 0.0
    else
        d = γ_inp
    end
    return d
end

function run(;thermal_init=false)

    ##Global-grid-size##

    Nx_g =(Nx-6)*dims[1]
    Ny_g =(Ny-6)*dims[2]
    Nz_g =(Nz-6)*dims[3]

    if me == 0
        println("--------Grid-Size----------")
        println(Nx_g," , ",Ny_g, " , ",Nz_g)
        println("---------------------------")
    end

    # Array initializations
    begin
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

        # #Flux arrays
        k_ϕ_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        k_ϕ_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        k_ϕ_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        k_ϕ_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))

        # W_1_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        k_W_1_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        k_W_1_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        k_W_1_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        # W_2_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        k_W_2_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        k_W_2_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        k_W_2_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        # W_3_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        k_W_3_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        k_W_3_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        k_W_3_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        # Y_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        k_Y_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        k_Y_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        k_Y_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))

        k_Γ_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        k_Γ_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        k_Γ_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        k_Σ = CUDA.zeros(Float64,(Nx,Ny,Nz))

        #Flux arrays
        kt_ϕ_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        kt_ϕ_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        kt_ϕ_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        kt_ϕ_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))

        # kt_W_1_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        kt_W_1_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        kt_W_1_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        kt_W_1_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        # kt_W_2_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        kt_W_2_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        kt_W_2_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        kt_W_2_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        # kt_W_3_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        kt_W_3_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        kt_W_3_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        kt_W_3_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        # kt_Y_1 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        kt_Y_2 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        kt_Y_3 = CUDA.zeros(Float64,(Nx,Ny,Nz))
        kt_Y_4 = CUDA.zeros(Float64,(Nx,Ny,Nz))

        #Updated field arrays
        ϕ_1_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
        ϕ_2_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
        ϕ_3_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
        ϕ_4_n = CUDA.zeros(Float64,(Nx,Ny,Nz))

        # W_1_1_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
        W_1_2_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
        W_1_3_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
        W_1_4_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
        # W_2_1_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
        W_2_2_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
        W_2_3_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
        W_2_4_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
        # W_3_1_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
        W_3_2_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
        W_3_3_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
        W_3_4_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
        # Y_1_n = CUDA.zeros(Float64,(Nx,Ny,Nz))
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

    end

    ##Energy arrays##
    begin

        E_t = CUDA.zeros(Float64,(Nx,Ny,Nz))

        spec_cut = [Nx÷4,Ny÷4,Nz÷4]
        N_bins = Kc_bin_nums(spec_cut[1],spec_cut[2],spec_cut[3])
        B_fft = zeros((nsnaps+1,N_bins,2))

        total_energies = zeros((nsnaps+1,10))

    end
    CUDA.memory_status()

    ##########Configuring thread block grid###########

    thrds = (32,1,1)
    blks = (Nx÷thrds[1],Ny÷thrds[2],Nz÷thrds[3])

    if me == 0
        println(string("#threads:",thrds," #blocks:",blks))
    end
    ##########END Configuring thread block grid###########
    seed_value = base_seed
    
    if thermal_init == true
    
    #Thermal initializing condition
        begin 

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
                    Nx,Ny,Nz,Nx_g,Ny_g,Nz_g,coords[1],coords[2],coords[3],
                    W_1_2,W_1_3,W_1_4,
                    W_1_2_n,W_1_3_n,W_1_4_n,
                    ϕ_1,ϕ_2,ϕ_3,ϕ_4)
                synchronize()                    
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
                    Nx,Ny,Nz,Nx_g,Ny_g,Nz_g,coords[1],coords[2],coords[3],
                    W_2_2,W_2_3,W_2_4,
                    W_2_2_n,W_2_3_n,W_2_4_n,
                    ϕ_1,ϕ_2,ϕ_3,ϕ_4)
                synchronize()
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
                    Nx,Ny,Nz,Nx_g,Ny_g,Nz_g,coords[1],coords[2],coords[3],
                    W_3_2,W_3_3,W_3_4,
                    W_3_2_n,W_3_3_n,W_3_4_n,
                    ϕ_1,ϕ_2,ϕ_3,ϕ_4)
                synchronize()
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
                    Nx,Ny,Nz,Nx_g,Ny_g,Nz_g,coords[1],coords[2],coords[3],
                    Y_2,Y_3,Y_4,
                    Y_2_n,Y_3_n,Y_4_n,
                    ϕ_1,ϕ_2,ϕ_3,ϕ_4)
                synchronize()
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
                # @cuda threads=thrds blocks=blks thermal_initializer_ϕ!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
                    # gw,gy,gp2,vev,dx,T,meff_sq,
                    # Nx,Ny,Nz)
                @cuda threads=thrds blocks=blks thermal_initializer_ϕ!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
                    gw,gy,gp2,vev,dx,T,meff_sq,
                    Nx,Ny,Nz,Nx_g,Ny_g,Nz_g,coords[1],coords[2],coords[3])
                synchronize()
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
    #Initializing random bubbles
    else
        begin

            Xb_sample = range(4,Nx_g,step=bub_diam)
            Yb_sample = range(4,Ny_g,step=bub_diam)
            Zb_sample = range(4,Nz_g,step=bub_diam)

            coord_samples = []
            for s_i in Xb_sample
                for s_j in Yb_sample
                    for s_k in Zb_sample
                        push!(coord_samples,[s_i,s_j,s_k])
                    end
                end
            end
            
            ##Randomly choose no_bubbles elements of the sample sub-lattice meshgrid
            ##Will throw error if no_bubbles>possible choices
            Random.seed!(seed_value)
            choice_locs = sample(coord_samples,no_bubbles,replace=false)
            xb_locs = [a[1] for a in choice_locs]
            yb_locs = [a[2] for a in choice_locs]
            zb_locs = [a[3] for a in choice_locs]

            #2 bubbles in y=0 plane#
            if (no_bubbles == 2)
                xb_locs = [Nx_g÷2-2*bub_diam,Nx_g÷2+2*bub_diam]
                yb_locs = [Ny_g÷2,Ny_g÷2]
                zb_locs = [Nz_g÷2,Nz_g÷2]
            end

            bubble_locs = hcat(xb_locs,yb_locs,zb_locs)

            println(string("bubble location matrix",bubble_locs))
            
            Random.seed!(seed_value)
            Hoft_arr = rand(Uniform(0,1),(no_bubbles,3))

            bubs = []
            for bub_idx in range(1,no_bubbles)
                phi=gen_phi(Hoft_arr[bub_idx,:])
                ib,jb,kb = bubble_locs[bub_idx,:]
                # println(string(ib," ",jb," ",kb," ", phi))
                push!(bubs,[ib,jb,kb,real(phi[1]),imag(phi[1]),real(phi[2]),imag(phi[2])])
            end

            rkx=pi/((Nx_g-1)*dx)
            rky=pi/((Ny_g-1)*dx)
            rkz=pi/((Nz_g-1)*dx)

            @time for b in range(1,size(bubs,1),step=1)
                ib,jb,kb,p1,p2,p3,p4 = bubs[b]
                @cuda threads=thrds blocks=blks initializer!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,ib,jb,kb,p1,p2,p3,p4,rkx,rky,rkz,dx,mH,Nx,Ny,Nz,coords[1],coords[2],coords[3])
                synchronize()
            end
        end
    end

    #compute energies and magnetic fields at initial time step

    Vol = dx^3

    #local-total-declarations
        pe_local = 0.0
        ke_local = 0.0
        ge_local = 0.0
        eew_local = 0.0
        mew_local = 0.0
        eey_local = 0.0
        mey_local = 0.0
        bxe_local = 0.0
        bye_local = 0.0
        bze_local = 0.0
        bx2e_local = 0.0
        by2e_local = 0.0
        bz2e_local = 0.0
    ##

    #Simulatneous compute and gather with dummy gpu arrays
        @cuda threads=thrds blocks=blks compute_PE!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
            E_t,gw,gy,gp2,vev,lambda,θ_w,dx)
        synchronize()
        # gather(E_t,PE_Phi_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
        pe_local = sum(Array(E_t))*Vol
        # MPI.Barrier(comm_cart)
        @cuda threads=thrds blocks=blks compute_KE!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
            W_1_2,W_1_3,W_1_4,
            W_2_2,W_2_3,W_2_4,
            W_3_2,W_3_3,W_3_4,
            Y_2,Y_3,Y_4,
            dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
            dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
            dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
            dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
            dY_2_dt,dY_3_dt,dY_4_dt,
            E_t,gw,gy,gp2,vev,lambda,θ_w,dx)
        synchronize()
        # gather(E_t,KE_Phi_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
        ke_local = sum(Array(E_t))*Vol
        # MPI.Barrier(comm_cart)
        @cuda threads=thrds blocks=blks compute_GE!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
            W_1_2,W_1_3,W_1_4,
            W_2_2,W_2_3,W_2_4,
            W_3_2,W_3_3,W_3_4,
            Y_2,Y_3,Y_4,
            dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
            dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
            dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
            dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
            dY_2_dt,dY_3_dt,dY_4_dt,
            E_t,gw,gy,gp2,vev,lambda,θ_w,dx)
        synchronize()
        # gather(E_t,GE_Phi_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
        ge_local = sum(Array(E_t))*Vol
        # MPI.Barrier(comm_cart)
        @cuda threads=thrds blocks=blks compute_MEW!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
            W_1_2,W_1_3,W_1_4,
            W_2_2,W_2_3,W_2_4,
            W_3_2,W_3_3,W_3_4,
            Y_2,Y_3,Y_4,
            dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
            dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
            dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
            dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
            dY_2_dt,dY_3_dt,dY_4_dt,
            E_t,gw,gy,gp2,vev,lambda,θ_w,dx)
        synchronize()
        # gather(E_t,MagneticE_W_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
        mew_local = sum(Array(E_t))*Vol
        # MPI.Barrier(comm_cart)
        @cuda threads=thrds blocks=blks compute_EEW!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
            W_1_2,W_1_3,W_1_4,
            W_2_2,W_2_3,W_2_4,
            W_3_2,W_3_3,W_3_4,
            Y_2,Y_3,Y_4,
            dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
            dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
            dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
            dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
            dY_2_dt,dY_3_dt,dY_4_dt,
            E_t,gw,gy,gp2,vev,lambda,θ_w,dx)
        synchronize()
        # gather(E_t,ElectricE_W_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
        eew_local = sum(Array(E_t))*Vol
        # MPI.Barrier(comm_cart)
        @cuda threads=thrds blocks=blks compute_MEY!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
            W_1_2,W_1_3,W_1_4,
            W_2_2,W_2_3,W_2_4,
            W_3_2,W_3_3,W_3_4,
            Y_2,Y_3,Y_4,
            dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
            dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
            dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
            dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
            dY_2_dt,dY_3_dt,dY_4_dt,
            E_t,gw,gy,gp2,vev,lambda,θ_w,dx)
        synchronize()
        # gather(E_t,MagneticE_Y_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
        mey_local = sum(Array(E_t))*Vol
        # MPI.Barrier(comm_cart)
        @cuda threads=thrds blocks=blks compute_EEY!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
            W_1_2,W_1_3,W_1_4,
            W_2_2,W_2_3,W_2_4,
            W_3_2,W_3_3,W_3_4,
            Y_2,Y_3,Y_4,
            dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
            dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
            dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
            dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
            dY_2_dt,dY_3_dt,dY_4_dt,
            E_t,gw,gy,gp2,vev,lambda,θ_w,dx)
        synchronize()
        # gather(E_t,ElectricE_Y_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
        eey_local = sum(Array(E_t))*Vol
        # MPI.Barrier(comm_cart)
        @cuda threads=thrds blocks=blks compute_Bx!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
            W_1_2,W_1_3,W_1_4,
            W_2_2,W_2_3,W_2_4,
            W_3_2,W_3_3,W_3_4,
            Y_2,Y_3,Y_4,
            dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
            dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
            dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
            dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
            dY_2_dt,dY_3_dt,dY_4_dt,
            E_t,gw,gy,gp2,vev,lambda,θ_w,dx)
        synchronize()
        # gather(E_t,B_x_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
        bxe_local = sum(Array(E_t).^2)
        B_x_local_arr = Array(E_t)
        # B_x_fft = Array(fft(E_t))
        # MPI.Barrier(comm_cart)
        @cuda threads=thrds blocks=blks compute_By!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
            W_1_2,W_1_3,W_1_4,
            W_2_2,W_2_3,W_2_4,
            W_3_2,W_3_3,W_3_4,
            Y_2,Y_3,Y_4,
            dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
            dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
            dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
            dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
            dY_2_dt,dY_3_dt,dY_4_dt,
            E_t,gw,gy,gp2,vev,lambda,θ_w,dx)
        synchronize()
        # gather(E_t,B_y_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
        bye_local = sum(Array(E_t).^2)
        B_y_local_arr = Array(E_t)
        # B_y_fft = Array(fft(E_t))
        # MPI.Barrier(comm_cart)
        @cuda threads=thrds blocks=blks compute_Bz!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
            W_1_2,W_1_3,W_1_4,
            W_2_2,W_2_3,W_2_4,
            W_3_2,W_3_3,W_3_4,
            Y_2,Y_3,Y_4,
            dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
            dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
            dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
            dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
            dY_2_dt,dY_3_dt,dY_4_dt,
            E_t,gw,gy,gp2,vev,lambda,θ_w,dx)
        synchronize()
        # gather(E_t,B_z_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
        bze_local = sum(Array(E_t).^2)
        B_z_local_arr = Array(E_t)
        # B_z_fft = Array(fft(E_t))
        # MPI.Barrier(comm_cart)
        @cuda threads=thrds blocks=blks compute_B_2_x!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
            W_1_2,W_1_3,W_1_4,
            W_2_2,W_2_3,W_2_4,
            W_3_2,W_3_3,W_3_4,
            Y_2,Y_3,Y_4,
            dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
            dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
            dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
            dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
            dY_2_dt,dY_3_dt,dY_4_dt,
            E_t,gw,gy,gp2,vev,lambda,θ_w,dx)
        synchronize()
        # gather(E_t,B_x_2_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
        bx2e_local = sum(Array(E_t).^2)
        B_x2_local_arr = Array(E_t)
        # MPI.Barrier(comm_cart)
        @cuda threads=thrds blocks=blks compute_B_2_y!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
            W_1_2,W_1_3,W_1_4,
            W_2_2,W_2_3,W_2_4,
            W_3_2,W_3_3,W_3_4,
            Y_2,Y_3,Y_4,
            dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
            dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
            dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
            dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
            dY_2_dt,dY_3_dt,dY_4_dt,
            E_t,gw,gy,gp2,vev,lambda,θ_w,dx)
        synchronize()
        # gather(E_t,B_y_2_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
        by2e_local = sum(Array(E_t).^2)
        B_y2_local_arr = Array(E_t)
        # MPI.Barrier(comm_cart)
        @cuda threads=thrds blocks=blks compute_B_2_z!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
            W_1_2,W_1_3,W_1_4,
            W_2_2,W_2_3,W_2_4,
            W_3_2,W_3_3,W_3_4,
            Y_2,Y_3,Y_4,
            dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
            dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
            dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
            dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
            dY_2_dt,dY_3_dt,dY_4_dt,
            E_t,gw,gy,gp2,vev,lambda,θ_w,dx)
        synchronize()
        # gather(E_t,B_z_2_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
        bz2e_local = sum(Array(E_t).^2)
        B_z2_local_arr = Array(E_t)
        # MPI.Barrier(comm_cart)

        @cuda threads=thrds blocks=blks compute_Ax!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
            W_1_2,W_1_3,W_1_4,
            W_2_2,W_2_3,W_2_4,
            W_3_2,W_3_3,W_3_4,
            Y_2,Y_3,Y_4,
            dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
            dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
            dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
            dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
            dY_2_dt,dY_3_dt,dY_4_dt,
            E_t,gw,gy,gp2,vev,lambda,θ_w,dx)
        synchronize()
        Ax_local_arr = Array(E_t)
        # MPI.Barrier(comm_cart)
        @cuda threads=thrds blocks=blks compute_Ay!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
            W_1_2,W_1_3,W_1_4,
            W_2_2,W_2_3,W_2_4,
            W_3_2,W_3_3,W_3_4,
            Y_2,Y_3,Y_4,
            dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
            dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
            dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
            dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
            dY_2_dt,dY_3_dt,dY_4_dt,
            E_t,gw,gy,gp2,vev,lambda,θ_w,dx)
        synchronize()
        Ay_local_arr = Array(E_t)
        # MPI.Barrier(comm_cart)
        @cuda threads=thrds blocks=blks compute_Az!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
            W_1_2,W_1_3,W_1_4,
            W_2_2,W_2_3,W_2_4,
            W_3_2,W_3_3,W_3_4,
            Y_2,Y_3,Y_4,
            dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
            dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
            dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
            dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
            dY_2_dt,dY_3_dt,dY_4_dt,
            E_t,gw,gy,gp2,vev,lambda,θ_w,dx)
        synchronize()
        Az_local_arr = Array(E_t)
        # MPI.Barrier(comm_cart)

        phi_local_arr=sqrt.(Array(ϕ_1).^2+Array(ϕ_2).^2+Array(ϕ_3).^2+Array(ϕ_4).^2)
    ############
    MPI.Barrier(comm_cart)

    CUDA.memory_status()

    ##Sum local energies##
    begin
        total_energies[1,1] = pe_local
        total_energies[1,2] = ke_local
        total_energies[1,3] = ge_local
        total_energies[1,4] = eew_local
        total_energies[1,5] = mew_local
        total_energies[1,6] = eey_local
        total_energies[1,7] = mey_local
        total_energies[1,8] = (0.5*(bxe_local .+bye_local .+bze_local))*Vol
        total_energies[1,9] = (0.5*(bx2e_local .+by2e_local .+bz2e_local))*Vol
        total_energies[1,10]=minimum(sqrt.(Array(ϕ_1).^2 .+ Array(ϕ_2).^2 .+ Array(ϕ_3).^2 .+ Array(ϕ_4).^2))
    end

    ##Unresolved issue with gathering and printing summed energies##
    ##Seems to work with local energies##
    ## Gather summed energies in a global array ##
        metrics_length = size(total_energies,2)
        total_energies_global = zeros((size(total_energies,2)*nprocs))
        energy_stack = zeros((nsnaps+1,metrics_length))
        gather_metrics(total_energies[1,:],total_energies_global,me,comm_cart,nprocs,metrics_length)
        MPI.Barrier(comm_cart)
    ######
    
    h5open(string("local_energies_proc_",me,"_",0,"_data.h5"), "w") do file
        write(file, "energies local", total_energies)
    end 

    h5open(string("proc_",me,"_",0,"_data.h5"), "w") do file
        write(file, "B_x", B_x_local_arr)
        write(file, "B_y", B_y_local_arr)
        write(file, "B_z", B_z_local_arr)
        # write(file, "B_x2", B_x2_local_arr)
        # write(file, "B_y2", B_y2_local_arr)
        # write(file, "B_z2", B_z2_local_arr)
        write(file, "Ax", Ax_local_arr)
        write(file, "Ay", Ay_local_arr)
        write(file, "Az", Az_local_arr)
        write(file,"phi", phi_local_arr)
    end 
    MPI.Barrier(comm_cart)

    #Print energies##
    if me == 0

        begin
            proc_iter = range(1,nprocs,step=1)

            total_pe_global = sum([total_energies_global[a] for a in metrics_length.*(proc_iter.-1).+1])
            total_ke_global = sum([total_energies_global[a] for a in metrics_length.*(proc_iter.-1).+2])
            total_ge_global = sum([total_energies_global[a] for a in metrics_length.*(proc_iter.-1).+3])
            total_mew_global = sum([total_energies_global[a] for a in metrics_length.*(proc_iter.-1).+4])
            total_eew_global = sum([total_energies_global[a] for a in metrics_length.*(proc_iter.-1).+5])
            total_mey_global = sum([total_energies_global[a] for a in metrics_length.*(proc_iter.-1).+6])
            total_eey_global = sum([total_energies_global[a] for a in metrics_length.*(proc_iter.-1).+7])
            total_em_mag_global = sum([total_energies_global[a] for a in metrics_length.*(proc_iter.-1).+8])
            total_em_mag_2_global = sum([total_energies_global[a] for a in metrics_length.*(proc_iter.-1).+9])
            min_phi_global = minimum([total_energies_global[a] for a in metrics_length.*(proc_iter.-1).+10])
            total_e_global = (total_pe_global+total_ke_global+total_ge_global+total_mew_global+
                              total_eew_global+total_mey_global+total_eey_global)

            energy_stack[1,1] = total_pe_global
            energy_stack[1,2] = total_ke_global
            energy_stack[1,3] = total_ge_global
            energy_stack[1,4] = total_mew_global
            energy_stack[1,5] = total_eew_global
            energy_stack[1,6] = total_mey_global
            energy_stack[1,7] = total_eey_global
            energy_stack[1,8] = total_em_mag_global
            energy_stack[1,9] = total_em_mag_2_global
            energy_stack[1,10] = min_phi_global

            
            println(string("--------Energies--t:",0,"--process:----------"))
            println("Potentianl energy Higgs: ",total_pe_global)
            println("Kinetic energy Higgs: ",total_ke_global)
            println("Gradient energy Higgs:",total_ge_global,)
            println("Magnetic energy W: ",total_mew_global)
            println("Electric energy W: ",total_eew_global)
            println("Magnetic energy Y: ",total_mey_global)
            println("Electric energy Y: ",total_eey_global)
            println("EM-Magnetic energy: ",total_em_mag_global)
            println("EM-Magnetic energy(2): ",total_em_mag_2_global)
            println("Total energy: ", total_e_global)
            println("Min Phi",min_phi_global)
            println("---------------------------------------")
        end
        
    end

    MPI.Barrier(comm_cart)
    println("starting time iterations")
    ###END Initializing###
    
    #Counter for snaps
    snp_idx = 1
    for it in range(1,nte,step=1)
    
        # Jex = 0.0
        Jex = J_ext(it,dt,mH,t_sat,thermal_init=thermal_init)
        γ = damp(it,dt,mH,t_sat,thermal_init=thermal_init)
        
        ##RK-4##
        begin
            ##RK-1##
            
            @cuda always_inline=true threads=thrds blocks=blks rk4_step!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
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
                gw,gy,gp2,β_W,β_Y,vev,lambda,dx,Jex,γ)
            ############
            synchronize()
            
            ##--Halo-Update-k-arrays##
            update_calls_k!(k_ϕ_1,k_ϕ_2,k_ϕ_3,k_ϕ_4,
                k_W_1_2,k_W_1_3,k_W_1_4,
                k_W_2_2,k_W_2_3,k_W_2_4,
                k_W_3_2,k_W_3_3,k_W_3_4,
                k_Y_2,k_Y_3,k_Y_4,
                k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,
                kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
                kt_W_1_2,kt_W_1_3,kt_W_1_4,
                kt_W_2_2,kt_W_2_3,kt_W_2_4,
                kt_W_3_2,kt_W_3_3,kt_W_3_4,
                kt_Y_2,kt_Y_3,kt_Y_4,
                neighbors_x,neighbors_y,neighbors_z,comm_cart)
            ########
            
            #Update new field arrays with first term k_1
            begin
                ϕ_1_n = k_ϕ_1.*(dt/6.) + ϕ_1
                ϕ_2_n = k_ϕ_2.*(dt/6.) + ϕ_2
                ϕ_3_n = k_ϕ_3.*(dt/6.) + ϕ_3
                ϕ_4_n = k_ϕ_4.*(dt/6.) + ϕ_4
                W_1_2_n = k_W_1_2.*(dt/6.) + W_1_2
                W_1_3_n = k_W_1_3.*(dt/6.) + W_1_3
                W_1_4_n = k_W_1_4.*(dt/6.) + W_1_4
                W_2_2_n = k_W_2_2.*(dt/6.) + W_2_2
                W_2_3_n = k_W_2_3.*(dt/6.) + W_2_3
                W_2_4_n = k_W_2_4.*(dt/6.) + W_2_4
                W_3_2_n = k_W_3_2.*(dt/6.) + W_3_2
                W_3_3_n = k_W_3_3.*(dt/6.) + W_3_3
                W_3_4_n = k_W_3_4.*(dt/6.) + W_3_4
                Y_2_n = k_Y_2.*(dt/6.) + Y_2
                Y_3_n = k_Y_3.*(dt/6.) + Y_3
                Y_4_n = k_Y_4.*(dt/6.) + Y_4
                Γ_1_n = k_Γ_1.*(dt/6.) + Γ_1
                Γ_2_n = k_Γ_2.*(dt/6.) + Γ_2
                Γ_3_n = k_Γ_3.*(dt/6.) + Γ_3
                Σ_n = k_Σ.*(dt/6.) + Σ
                # CUDA.memory_status()
                # synchronize()
                dϕ_1_dt_n = kt_ϕ_1.*(dt/6.) + dϕ_1_dt
                dϕ_2_dt_n = kt_ϕ_2.*(dt/6.) + dϕ_2_dt
                dϕ_3_dt_n = kt_ϕ_3.*(dt/6.) + dϕ_3_dt
                dϕ_4_dt_n = kt_ϕ_4.*(dt/6.) + dϕ_4_dt
                dW_1_2_dt_n = kt_W_1_2.*(dt/6.) + dW_1_2_dt
                dW_1_3_dt_n = kt_W_1_3.*(dt/6.) + dW_1_3_dt
                dW_1_4_dt_n = kt_W_1_4.*(dt/6.) + dW_1_4_dt
                dW_2_2_dt_n = kt_W_2_2.*(dt/6.) + dW_2_2_dt
                dW_2_3_dt_n = kt_W_2_3.*(dt/6.) + dW_2_3_dt
                dW_2_4_dt_n = kt_W_2_4.*(dt/6.) + dW_2_4_dt
                dW_3_2_dt_n = kt_W_3_2.*(dt/6.) + dW_3_2_dt
                dW_3_3_dt_n = kt_W_3_3.*(dt/6.) + dW_3_3_dt
                dW_3_4_dt_n = kt_W_3_4.*(dt/6.) + dW_3_4_dt
                dY_2_dt_n = kt_Y_2.*(dt/6.) + dY_2_dt
                dY_3_dt_n = kt_Y_3.*(dt/6.) + dY_3_dt
                dY_4_dt_n = kt_Y_4.*(dt/6.) + dY_4_dt
            end
            synchronize()

            ##RK-2##
            
            #Compute k_2 arrays
            @cuda always_inline=true threads=thrds blocks=blks rk4_step!(ϕ_1+k_ϕ_1.*(dt/2.),ϕ_2+k_ϕ_2.*(dt/2.),ϕ_3+k_ϕ_3.*(dt/2.),ϕ_4+k_ϕ_4.*(dt/2.),
                W_1_2+k_W_1_2.*(dt/2.),W_1_3+k_W_1_3.*(dt/2.),W_1_4+k_W_1_4.*(dt/2.),
                W_2_2+k_W_2_2.*(dt/2.),W_2_3+k_W_2_3.*(dt/2.),W_2_4+k_W_2_4.*(dt/2.),
                W_3_2+k_W_3_2.*(dt/2.),W_3_3+k_W_3_3.*(dt/2.),W_3_4+k_W_3_4.*(dt/2.),
                Y_2+k_Y_2.*(dt/2.),Y_3+k_Y_3.*(dt/2.),Y_4+k_Y_4.*(dt/2.),
                Γ_1+k_Γ_1.*(dt/2.),Γ_2+k_Γ_2.*(dt/2.),Γ_3+k_Γ_3.*(dt/2.),Σ+k_Σ.*(dt/2.),
                k_ϕ_1,k_ϕ_2,k_ϕ_3,k_ϕ_4,
                k_W_1_2,k_W_1_3,k_W_1_4,
                k_W_2_2,k_W_2_3,k_W_2_4,
                k_W_3_2,k_W_3_3,k_W_3_4,
                k_Y_2,k_Y_3,k_Y_4,
                k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,
                dϕ_1_dt+kt_ϕ_1.*(dt/2.),dϕ_2_dt+kt_ϕ_2.*(dt/2.),dϕ_3_dt+kt_ϕ_3.*(dt/2.),dϕ_4_dt+kt_ϕ_4.*(dt/2.),
                dW_1_2_dt+kt_W_1_2.*(dt/2.),dW_1_3_dt+kt_W_1_3.*(dt/2.),dW_1_4_dt+kt_W_1_4.*(dt/2.),
                dW_2_2_dt+kt_W_2_2.*(dt/2.),dW_2_3_dt+kt_W_2_3.*(dt/2.),dW_2_4_dt+kt_W_2_4.*(dt/2.),
                dW_3_2_dt+kt_W_3_2.*(dt/2.),dW_3_3_dt+kt_W_3_3.*(dt/2.),dW_3_4_dt+kt_W_3_4.*(dt/2.),
                dY_2_dt+kt_Y_2.*(dt/2.),dY_3_dt+kt_Y_3.*(dt/2.),dY_4_dt+kt_Y_4.*(dt/2.),
                kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
                kt_W_1_2,kt_W_1_3,kt_W_1_4,
                kt_W_2_2,kt_W_2_3,kt_W_2_4,
                kt_W_3_2,kt_W_3_3,kt_W_3_4,
                kt_Y_2,kt_Y_3,kt_Y_4,
                gw,gy,gp2,β_W,β_Y,vev,lambda,dx,Jex,γ)
            ######
            synchronize()

            ##--Halo-Update-k-arrays##
            update_calls_k!(k_ϕ_1,k_ϕ_2,k_ϕ_3,k_ϕ_4,
                k_W_1_2,k_W_1_3,k_W_1_4,
                k_W_2_2,k_W_2_3,k_W_2_4,
                k_W_3_2,k_W_3_3,k_W_3_4,
                k_Y_2,k_Y_3,k_Y_4,
                k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,
                kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
                kt_W_1_2,kt_W_1_3,kt_W_1_4,
                kt_W_2_2,kt_W_2_3,kt_W_2_4,
                kt_W_3_2,kt_W_3_3,kt_W_3_4,
                kt_Y_2,kt_Y_3,kt_Y_4,
                neighbors_x,neighbors_y,neighbors_z,comm_cart)
            ########

            #Update new field arrays with second term k_2
            begin
                ϕ_1_n = k_ϕ_1.*(dt/3.) + ϕ_1_n
                ϕ_2_n = k_ϕ_2.*(dt/3.) + ϕ_2_n
                ϕ_3_n = k_ϕ_3.*(dt/3.) + ϕ_3_n
                ϕ_4_n = k_ϕ_4.*(dt/3.) + ϕ_4_n
                W_1_2_n = k_W_1_2.*(dt/3.) + W_1_2_n
                W_1_3_n = k_W_1_3.*(dt/3.) + W_1_3_n
                W_1_4_n = k_W_1_4.*(dt/3.) + W_1_4_n
                W_2_2_n = k_W_2_2.*(dt/3.) + W_2_2_n
                W_2_3_n = k_W_2_3.*(dt/3.) + W_2_3_n
                W_2_4_n = k_W_2_4.*(dt/3.) + W_2_4_n
                W_3_2_n = k_W_3_2.*(dt/3.) + W_3_2_n
                W_3_3_n = k_W_3_3.*(dt/3.) + W_3_3_n
                W_3_4_n = k_W_3_4.*(dt/3.) + W_3_4_n
                Y_2_n = k_Y_2.*(dt/3.) + Y_2_n
                Y_3_n = k_Y_3.*(dt/3.) + Y_3_n
                Y_4_n = k_Y_4.*(dt/3.) + Y_4_n
                Γ_1_n = k_Γ_1.*(dt/3.) + Γ_1_n
                Γ_2_n = k_Γ_2.*(dt/3.) + Γ_2_n
                Γ_3_n = k_Γ_3.*(dt/3.) + Γ_3_n
                Σ_n = k_Σ.*(dt/3.) + Σ_n
                # CUDA.memory_status()
                # synchronize()
                dϕ_1_dt_n = kt_ϕ_1.*(dt/3.) + dϕ_1_dt_n
                dϕ_2_dt_n = kt_ϕ_2.*(dt/3.) + dϕ_2_dt_n
                dϕ_3_dt_n = kt_ϕ_3.*(dt/3.) + dϕ_3_dt_n
                dϕ_4_dt_n = kt_ϕ_4.*(dt/3.) + dϕ_4_dt_n
                dW_1_2_dt_n = kt_W_1_2.*(dt/3.) + dW_1_2_dt_n
                dW_1_3_dt_n = kt_W_1_3.*(dt/3.) + dW_1_3_dt_n
                dW_1_4_dt_n = kt_W_1_4.*(dt/3.) + dW_1_4_dt_n
                dW_2_2_dt_n = kt_W_2_2.*(dt/3.) + dW_2_2_dt_n
                dW_2_3_dt_n = kt_W_2_3.*(dt/3.) + dW_2_3_dt_n
                dW_2_4_dt_n = kt_W_2_4.*(dt/3.) + dW_2_4_dt_n
                dW_3_2_dt_n = kt_W_3_2.*(dt/3.) + dW_3_2_dt_n
                dW_3_3_dt_n = kt_W_3_3.*(dt/3.) + dW_3_3_dt_n
                dW_3_4_dt_n = kt_W_3_4.*(dt/3.) + dW_3_4_dt_n
                dY_2_dt_n = kt_Y_2.*(dt/3.) + dY_2_dt_n
                dY_3_dt_n = kt_Y_3.*(dt/3.) + dY_3_dt_n
                dY_4_dt_n = kt_Y_4.*(dt/3.) + dY_4_dt_n
            end
            synchronize()

            ##RK-3##

            #Compute k_3 arrays

            @cuda always_inline=true threads=thrds blocks=blks rk4_step!(ϕ_1+k_ϕ_1.*(dt/2.),ϕ_2+k_ϕ_2.*(dt/2.),ϕ_3+k_ϕ_3.*(dt/2.),ϕ_4+k_ϕ_4.*(dt/2.),
                W_1_2+k_W_1_2.*(dt/2.),W_1_3+k_W_1_3.*(dt/2.),W_1_4+k_W_1_4.*(dt/2.),
                W_2_2+k_W_2_2.*(dt/2.),W_2_3+k_W_2_3.*(dt/2.),W_2_4+k_W_2_4.*(dt/2.),
                W_3_2+k_W_3_2.*(dt/2.),W_3_3+k_W_3_3.*(dt/2.),W_3_4+k_W_3_4.*(dt/2.),
                Y_2+k_Y_2.*(dt/2.),Y_3+k_Y_3.*(dt/2.),Y_4+k_Y_4.*(dt/2.),
                Γ_1+k_Γ_1.*(dt/2.),Γ_2+k_Γ_2.*(dt/2.),Γ_3+k_Γ_3.*(dt/2.),Σ+k_Σ.*(dt/2.),
                k_ϕ_1,k_ϕ_2,k_ϕ_3,k_ϕ_4,
                k_W_1_2,k_W_1_3,k_W_1_4,
                k_W_2_2,k_W_2_3,k_W_2_4,
                k_W_3_2,k_W_3_3,k_W_3_4,
                k_Y_2,k_Y_3,k_Y_4,
                k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,
                dϕ_1_dt+kt_ϕ_1.*(dt/2.),dϕ_2_dt+kt_ϕ_2.*(dt/2.),dϕ_3_dt+kt_ϕ_3.*(dt/2.),dϕ_4_dt+kt_ϕ_4.*(dt/2.),
                dW_1_2_dt+kt_W_1_2.*(dt/2.),dW_1_3_dt+kt_W_1_3.*(dt/2.),dW_1_4_dt+kt_W_1_4.*(dt/2.),
                dW_2_2_dt+kt_W_2_2.*(dt/2.),dW_2_3_dt+kt_W_2_3.*(dt/2.),dW_2_4_dt+kt_W_2_4.*(dt/2.),
                dW_3_2_dt+kt_W_3_2.*(dt/2.),dW_3_3_dt+kt_W_3_3.*(dt/2.),dW_3_4_dt+kt_W_3_4.*(dt/2.),
                dY_2_dt+kt_Y_2.*(dt/2.),dY_3_dt+kt_Y_3.*(dt/2.),dY_4_dt+kt_Y_4.*(dt/2.),
                kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
                kt_W_1_2,kt_W_1_3,kt_W_1_4,
                kt_W_2_2,kt_W_2_3,kt_W_2_4,
                kt_W_3_2,kt_W_3_3,kt_W_3_4,
                kt_Y_2,kt_Y_3,kt_Y_4,
                gw,gy,gp2,β_W,β_Y,vev,lambda,dx,Jex,γ)
            ####
            synchronize()
            
            ##--Halo-Update-k-arrays##
            update_calls_k!(k_ϕ_1,k_ϕ_2,k_ϕ_3,k_ϕ_4,
                k_W_1_2,k_W_1_3,k_W_1_4,
                k_W_2_2,k_W_2_3,k_W_2_4,
                k_W_3_2,k_W_3_3,k_W_3_4,
                k_Y_2,k_Y_3,k_Y_4,
                k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,
                kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
                kt_W_1_2,kt_W_1_3,kt_W_1_4,
                kt_W_2_2,kt_W_2_3,kt_W_2_4,
                kt_W_3_2,kt_W_3_3,kt_W_3_4,
                kt_Y_2,kt_Y_3,kt_Y_4,
                neighbors_x,neighbors_y,neighbors_z,comm_cart)
            ########
            
            #Update new field arrays with third term k_3
            begin
                ϕ_1_n = k_ϕ_1.*(dt/3.) + ϕ_1_n
                ϕ_2_n = k_ϕ_2.*(dt/3.) + ϕ_2_n
                ϕ_3_n = k_ϕ_3.*(dt/3.) + ϕ_3_n
                ϕ_4_n = k_ϕ_4.*(dt/3.) + ϕ_4_n
                W_1_2_n = k_W_1_2.*(dt/3.) + W_1_2_n
                W_1_3_n = k_W_1_3.*(dt/3.) + W_1_3_n
                W_1_4_n = k_W_1_4.*(dt/3.) + W_1_4_n
                W_2_2_n = k_W_2_2.*(dt/3.) + W_2_2_n
                W_2_3_n = k_W_2_3.*(dt/3.) + W_2_3_n
                W_2_4_n = k_W_2_4.*(dt/3.) + W_2_4_n
                W_3_2_n = k_W_3_2.*(dt/3.) + W_3_2_n
                W_3_3_n = k_W_3_3.*(dt/3.) + W_3_3_n
                W_3_4_n = k_W_3_4.*(dt/3.) + W_3_4_n
                Y_2_n = k_Y_2.*(dt/3.) + Y_2_n
                Y_3_n = k_Y_3.*(dt/3.) + Y_3_n
                Y_4_n = k_Y_4.*(dt/3.) + Y_4_n
                Γ_1_n = k_Γ_1.*(dt/3.) + Γ_1_n
                Γ_2_n = k_Γ_2.*(dt/3.) + Γ_2_n
                Γ_3_n = k_Γ_3.*(dt/3.) + Γ_3_n
                Σ_n = k_Σ.*(dt/3.) + Σ_n
                # CUDA.memory_status()
                # synchronize()
                dϕ_1_dt_n = kt_ϕ_1.*(dt/3.) + dϕ_1_dt_n
                dϕ_2_dt_n = kt_ϕ_2.*(dt/3.) + dϕ_2_dt_n
                dϕ_3_dt_n = kt_ϕ_3.*(dt/3.) + dϕ_3_dt_n
                dϕ_4_dt_n = kt_ϕ_4.*(dt/3.) + dϕ_4_dt_n
                dW_1_2_dt_n = kt_W_1_2.*(dt/3.) + dW_1_2_dt_n
                dW_1_3_dt_n = kt_W_1_3.*(dt/3.) + dW_1_3_dt_n
                dW_1_4_dt_n = kt_W_1_4.*(dt/3.) + dW_1_4_dt_n
                dW_2_2_dt_n = kt_W_2_2.*(dt/3.) + dW_2_2_dt_n
                dW_2_3_dt_n = kt_W_2_3.*(dt/3.) + dW_2_3_dt_n
                dW_2_4_dt_n = kt_W_2_4.*(dt/3.) + dW_2_4_dt_n
                dW_3_2_dt_n = kt_W_3_2.*(dt/3.) + dW_3_2_dt_n
                dW_3_3_dt_n = kt_W_3_3.*(dt/3.) + dW_3_3_dt_n
                dW_3_4_dt_n = kt_W_3_4.*(dt/3.) + dW_3_4_dt_n
                dY_2_dt_n = kt_Y_2.*(dt/3.) + dY_2_dt_n
                dY_3_dt_n = kt_Y_3.*(dt/3.) + dY_3_dt_n
                dY_4_dt_n = kt_Y_4.*(dt/3.) + dY_4_dt_n
            end
            MPI.Barrier(comm_cart)
            synchronize()

            ##RK-4##

            #Compute k_4 arrays

            @cuda always_inline=true threads=thrds blocks=blks rk4_step!(ϕ_1+k_ϕ_1.*dt,ϕ_2+k_ϕ_2.*dt,ϕ_3+k_ϕ_3.*dt,ϕ_4+k_ϕ_4.*dt,
                W_1_2+k_W_1_2.*dt,W_1_3+k_W_1_3.*dt,W_1_4+k_W_1_4.*dt,
                W_2_2+k_W_2_2.*dt,W_2_3+k_W_2_3.*dt,W_2_4+k_W_2_4.*dt,
                W_3_2+k_W_3_2.*dt,W_3_3+k_W_3_3.*dt,W_3_4+k_W_3_4.*dt,
                Y_2+k_Y_2.*dt,Y_3+k_Y_3.*dt,Y_4+k_Y_4.*dt,
                Γ_1+k_Γ_1.*dt,Γ_2+k_Γ_2.*dt,Γ_3+k_Γ_3.*dt,Σ+k_Σ.*dt,
                k_ϕ_1,k_ϕ_2,k_ϕ_3,k_ϕ_4,
                k_W_1_2,k_W_1_3,k_W_1_4,
                k_W_2_2,k_W_2_3,k_W_2_4,
                k_W_3_2,k_W_3_3,k_W_3_4,
                k_Y_2,k_Y_3,k_Y_4,
                k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,
                dϕ_1_dt+kt_ϕ_1.*dt,dϕ_2_dt+kt_ϕ_2.*dt,dϕ_3_dt+kt_ϕ_3.*dt,dϕ_4_dt+kt_ϕ_4.*dt,
                dW_1_2_dt+kt_W_1_2.*dt,dW_1_3_dt+kt_W_1_3.*dt,dW_1_4_dt+kt_W_1_4.*dt,
                dW_2_2_dt+kt_W_2_2.*dt,dW_2_3_dt+kt_W_2_3.*dt,dW_2_4_dt+kt_W_2_4.*dt,
                dW_3_2_dt+kt_W_3_2.*dt,dW_3_3_dt+kt_W_3_3.*dt,dW_3_4_dt+kt_W_3_4.*dt,
                dY_2_dt+kt_Y_2.*dt,dY_3_dt+kt_Y_3.*dt,dY_4_dt+kt_Y_4.*dt,
                kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
                kt_W_1_2,kt_W_1_3,kt_W_1_4,
                kt_W_2_2,kt_W_2_3,kt_W_2_4,
                kt_W_3_2,kt_W_3_3,kt_W_3_4,
                kt_Y_2,kt_Y_3,kt_Y_4,
                gw,gy,gp2,β_W,β_Y,vev,lambda,dx,Jex,γ)

            ########
            synchronize()      
            
            ##--Halo-Update-k-arrays##
            update_calls_k!(k_ϕ_1,k_ϕ_2,k_ϕ_3,k_ϕ_4,
                k_W_1_2,k_W_1_3,k_W_1_4,
                k_W_2_2,k_W_2_3,k_W_2_4,
                k_W_3_2,k_W_3_3,k_W_3_4,
                k_Y_2,k_Y_3,k_Y_4,
                k_Γ_1,k_Γ_2,k_Γ_3,k_Σ,
                kt_ϕ_1,kt_ϕ_2,kt_ϕ_3,kt_ϕ_4,
                kt_W_1_2,kt_W_1_3,kt_W_1_4,
                kt_W_2_2,kt_W_2_3,kt_W_2_4,
                kt_W_3_2,kt_W_3_3,kt_W_3_4,
                kt_Y_2,kt_Y_3,kt_Y_4,
                neighbors_x,neighbors_y,neighbors_z,comm_cart)
            ########
            
            #Update new field arrays with last term k_4 and name swap

            begin
                ϕ_1 = k_ϕ_1.*(dt/6.) + ϕ_1_n
                ϕ_2 = k_ϕ_2.*(dt/6.) + ϕ_2_n
                ϕ_3 = k_ϕ_3.*(dt/6.) + ϕ_3_n
                ϕ_4 = k_ϕ_4.*(dt/6.) + ϕ_4_n
                W_1_2 = k_W_1_2.*(dt/6.) + W_1_2_n
                W_1_3 = k_W_1_3.*(dt/6.) + W_1_3_n
                W_1_4 = k_W_1_4.*(dt/6.) + W_1_4_n
                W_2_2 = k_W_2_2.*(dt/6.) + W_2_2_n
                W_2_3 = k_W_2_3.*(dt/6.) + W_2_3_n
                W_2_4 = k_W_2_4.*(dt/6.) + W_2_4_n
                W_3_2 = k_W_3_2.*(dt/6.) + W_3_2_n
                W_3_3 = k_W_3_3.*(dt/6.) + W_3_3_n
                W_3_4 = k_W_3_4.*(dt/6.) + W_3_4_n
                Y_2 = k_Y_2.*(dt/6.) + Y_2_n
                Y_3 = k_Y_3.*(dt/6.) + Y_3_n
                Y_4 = k_Y_4.*(dt/6.) + Y_4_n
                Γ_1 = k_Γ_1.*(dt/6.) + Γ_1_n
                Γ_2 = k_Γ_2.*(dt/6.) + Γ_2_n
                Γ_3 = k_Γ_3.*(dt/6.) + Γ_3_n
                Σ = k_Σ.*(dt/6.) + Σ_n
                # CUDA.memory_status()
                # synchronize()
                dϕ_1_dt = kt_ϕ_1.*(dt/6.) + dϕ_1_dt_n
                dϕ_2_dt = kt_ϕ_2.*(dt/6.) + dϕ_2_dt_n
                dϕ_3_dt = kt_ϕ_3.*(dt/6.) + dϕ_3_dt_n
                dϕ_4_dt = kt_ϕ_4.*(dt/6.) + dϕ_4_dt_n
                dW_1_2_dt = kt_W_1_2.*(dt/6.) + dW_1_2_dt_n
                dW_1_3_dt = kt_W_1_3.*(dt/6.) + dW_1_3_dt_n
                dW_1_4_dt = kt_W_1_4.*(dt/6.) + dW_1_4_dt_n
                dW_2_2_dt = kt_W_2_2.*(dt/6.) + dW_2_2_dt_n
                dW_2_3_dt = kt_W_2_3.*(dt/6.) + dW_2_3_dt_n
                dW_2_4_dt = kt_W_2_4.*(dt/6.) + dW_2_4_dt_n
                dW_3_2_dt = kt_W_3_2.*(dt/6.) + dW_3_2_dt_n
                dW_3_3_dt = kt_W_3_3.*(dt/6.) + dW_3_3_dt_n
                dW_3_4_dt = kt_W_3_4.*(dt/6.) + dW_3_4_dt_n
                dY_2_dt = kt_Y_2.*(dt/6.) + dY_2_dt_n
                dY_3_dt = kt_Y_3.*(dt/6.) + dY_3_dt_n
                dY_4_dt = kt_Y_4.*(dt/6.) + dY_4_dt_n
            end
            synchronize()

            MPI.Barrier(comm_cart)

            update_calls!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
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
                neighbors_x,neighbors_y,neighbors_z,comm_cart)
            MPI.Barrier(comm_cart)
            # synchronize()

        end
        ########

        ###snapshots###
        if mod(it,dsnaps)==0
                
            snp_idx = snp_idx+1

            #local-total-declarations
                pe_local = 0.0
                ke_local = 0.0
                ge_local = 0.0
                eew_local = 0.0
                mew_local = 0.0
                eey_local = 0.0
                mey_local = 0.0
                bxe_local = 0.0
                bye_local = 0.0
                bze_local = 0.0
                bx2e_local = 0.0
                by2e_local = 0.0
                bz2e_local = 0.0
            #####

            #Simulatneous compute and gather with dummy gpu arrays
                @cuda always_inline=true threads=thrds blocks=blks compute_PE!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
                    E_t,gw,gy,gp2,vev,lambda,θ_w,dx)
                synchronize()
                # gather(E_t,PE_Phi_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
                pe_local = sum(Array(E_t))*Vol
                # MPI.Barrier(comm_cart)
                @cuda always_inline=true threads=thrds blocks=blks compute_KE!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
                    W_1_2,W_1_3,W_1_4,
                    W_2_2,W_2_3,W_2_4,
                    W_3_2,W_3_3,W_3_4,
                    Y_2,Y_3,Y_4,
                    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
                    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
                    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
                    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
                    dY_2_dt,dY_3_dt,dY_4_dt,
                    E_t,gw,gy,gp2,vev,lambda,θ_w,dx)
                synchronize()
                # gather(E_t,KE_Phi_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
                ke_local = sum(Array(E_t))*Vol
                # MPI.Barrier(comm_cart)
                @cuda always_inline=true threads=thrds blocks=blks compute_GE!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
                    W_1_2,W_1_3,W_1_4,
                    W_2_2,W_2_3,W_2_4,
                    W_3_2,W_3_3,W_3_4,
                    Y_2,Y_3,Y_4,
                    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
                    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
                    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
                    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
                    dY_2_dt,dY_3_dt,dY_4_dt,
                    E_t,gw,gy,gp2,vev,lambda,θ_w,dx)
                synchronize()
                # gather(E_t,GE_Phi_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
                ge_local = sum(Array(E_t))*Vol
                # MPI.Barrier(comm_cart)
                @cuda always_inline=true threads=thrds blocks=blks compute_MEW!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
                    W_1_2,W_1_3,W_1_4,
                    W_2_2,W_2_3,W_2_4,
                    W_3_2,W_3_3,W_3_4,
                    Y_2,Y_3,Y_4,
                    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
                    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
                    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
                    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
                    dY_2_dt,dY_3_dt,dY_4_dt,
                    E_t,gw,gy,gp2,vev,lambda,θ_w,dx)
                synchronize()
                # gather(E_t,MagneticE_W_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
                mew_local = sum(Array(E_t))*Vol
                # MPI.Barrier(comm_cart)
                @cuda always_inline=true threads=thrds blocks=blks compute_EEW!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
                    W_1_2,W_1_3,W_1_4,
                    W_2_2,W_2_3,W_2_4,
                    W_3_2,W_3_3,W_3_4,
                    Y_2,Y_3,Y_4,
                    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
                    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
                    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
                    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
                    dY_2_dt,dY_3_dt,dY_4_dt,
                    E_t,gw,gy,gp2,vev,lambda,θ_w,dx)
                synchronize()
                # gather(E_t,ElectricE_W_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
                eew_local = sum(Array(E_t))*Vol
                # MPI.Barrier(comm_cart)
                @cuda always_inline=true threads=thrds blocks=blks compute_MEY!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
                    W_1_2,W_1_3,W_1_4,
                    W_2_2,W_2_3,W_2_4,
                    W_3_2,W_3_3,W_3_4,
                    Y_2,Y_3,Y_4,
                    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
                    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
                    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
                    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
                    dY_2_dt,dY_3_dt,dY_4_dt,
                    E_t,gw,gy,gp2,vev,lambda,θ_w,dx)
                synchronize()
                # gather(E_t,MagneticE_Y_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
                mey_local = sum(Array(E_t))*Vol
                # MPI.Barrier(comm_cart)
                @cuda always_inline=true threads=thrds blocks=blks compute_EEY!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
                    W_1_2,W_1_3,W_1_4,
                    W_2_2,W_2_3,W_2_4,
                    W_3_2,W_3_3,W_3_4,
                    Y_2,Y_3,Y_4,
                    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
                    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
                    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
                    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
                    dY_2_dt,dY_3_dt,dY_4_dt,
                    E_t,gw,gy,gp2,vev,lambda,θ_w,dx)
                synchronize()
                # gather(E_t,ElectricE_Y_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
                eey_local = sum(Array(E_t))*Vol
                # MPI.Barrier(comm_cart)
                @cuda always_inline=true threads=thrds blocks=blks compute_Bx!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
                    W_1_2,W_1_3,W_1_4,
                    W_2_2,W_2_3,W_2_4,
                    W_3_2,W_3_3,W_3_4,
                    Y_2,Y_3,Y_4,
                    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
                    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
                    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
                    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
                    dY_2_dt,dY_3_dt,dY_4_dt,
                    E_t,gw,gy,gp2,vev,lambda,θ_w,dx)
                synchronize()
                # gather(E_t,B_x_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
                bxe_local = sum(Array(E_t).^2)
                if mod(it,dsnaps_fft)==0
                    B_x_local_arr = Array(E_t)
                end
                # B_x_fft = Array(fft(E_t))
                # MPI.Barrier(comm_cart)
                @cuda always_inline=true threads=thrds blocks=blks compute_By!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
                    W_1_2,W_1_3,W_1_4,
                    W_2_2,W_2_3,W_2_4,
                    W_3_2,W_3_3,W_3_4,
                    Y_2,Y_3,Y_4,
                    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
                    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
                    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
                    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
                    dY_2_dt,dY_3_dt,dY_4_dt,
                    E_t,gw,gy,gp2,vev,lambda,θ_w,dx)
                synchronize()
                # gather(E_t,B_y_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
                bye_local = sum(Array(E_t).^2)
                if mod(it,dsnaps_fft)==0
                    B_y_local_arr = Array(E_t)
                end
                # B_y_fft = Array(fft(E_t))
                # MPI.Barrier(comm_cart)
                @cuda always_inline=true threads=thrds blocks=blks compute_Bz!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
                    W_1_2,W_1_3,W_1_4,
                    W_2_2,W_2_3,W_2_4,
                    W_3_2,W_3_3,W_3_4,
                    Y_2,Y_3,Y_4,
                    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
                    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
                    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
                    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
                    dY_2_dt,dY_3_dt,dY_4_dt,
                    E_t,gw,gy,gp2,vev,lambda,θ_w,dx)
                synchronize()
                # gather(E_t,B_z_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
                bze_local = sum(Array(E_t).^2)
                if mod(it,dsnaps_fft)==0
                    B_z_local_arr = Array(E_t)
                end
                # B_z_fft = Array(fft(E_t))
                # MPI.Barrier(comm_cart)
                @cuda always_inline=true threads=thrds blocks=blks compute_B_2_x!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
                    W_1_2,W_1_3,W_1_4,
                    W_2_2,W_2_3,W_2_4,
                    W_3_2,W_3_3,W_3_4,
                    Y_2,Y_3,Y_4,
                    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
                    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
                    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
                    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
                    dY_2_dt,dY_3_dt,dY_4_dt,
                    E_t,gw,gy,gp2,vev,lambda,θ_w,dx)
                synchronize()
                # gather(E_t,B_x_2_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
                bx2e_local = sum(Array(E_t).^2)
                if mod(it,dsnaps_fft)==0
                    B_x2_local_arr = Array(E_t)
                end
                # MPI.Barrier(comm_cart)
                @cuda always_inline=true threads=thrds blocks=blks compute_B_2_y!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
                    W_1_2,W_1_3,W_1_4,
                    W_2_2,W_2_3,W_2_4,
                    W_3_2,W_3_3,W_3_4,
                    Y_2,Y_3,Y_4,
                    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
                    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
                    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
                    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
                    dY_2_dt,dY_3_dt,dY_4_dt,
                    E_t,gw,gy,gp2,vev,lambda,θ_w,dx)
                synchronize()
                # gather(E_t,B_y_2_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
                by2e_local = sum(Array(E_t).^2)
                if mod(it,dsnaps_fft)==0
                    B_y2_local_arr = Array(E_t)
                end
                # MPI.Barrier(comm_cart)
                @cuda always_inline=true threads=thrds blocks=blks compute_B_2_z!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
                    W_1_2,W_1_3,W_1_4,
                    W_2_2,W_2_3,W_2_4,
                    W_3_2,W_3_3,W_3_4,
                    Y_2,Y_3,Y_4,
                    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
                    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
                    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
                    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
                    dY_2_dt,dY_3_dt,dY_4_dt,
                    E_t,gw,gy,gp2,vev,lambda,θ_w,dx)
                synchronize()
                # gather(E_t,B_z_2_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
                bz2e_local = sum(Array(E_t).^2)
                if mod(it,dsnaps_fft)==0
                    B_z2_local_arr = Array(E_t)
                end
                # MPI.Barrier(comm_cart)

                @cuda threads=thrds blocks=blks compute_Ax!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
                    W_1_2,W_1_3,W_1_4,
                    W_2_2,W_2_3,W_2_4,
                    W_3_2,W_3_3,W_3_4,
                    Y_2,Y_3,Y_4,
                    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
                    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
                    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
                    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
                    dY_2_dt,dY_3_dt,dY_4_dt,
                    E_t,gw,gy,gp2,vev,lambda,θ_w,dx)
                synchronize()
                if mod(it,dsnaps_fft)==0
                    Ax_local_arr = Array(E_t)
                end
                # MPI.Barrier(comm_cart)
                @cuda threads=thrds blocks=blks compute_Ay!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
                    W_1_2,W_1_3,W_1_4,
                    W_2_2,W_2_3,W_2_4,
                    W_3_2,W_3_3,W_3_4,
                    Y_2,Y_3,Y_4,
                    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
                    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
                    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
                    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
                    dY_2_dt,dY_3_dt,dY_4_dt,
                    E_t,gw,gy,gp2,vev,lambda,θ_w,dx)
                synchronize()
                if mod(it,dsnaps_fft)==0
                    Ay_local_arr = Array(E_t)
                end
                # MPI.Barrier(comm_cart)
                @cuda threads=thrds blocks=blks compute_Az!(ϕ_1,ϕ_2,ϕ_3,ϕ_4,
                    W_1_2,W_1_3,W_1_4,
                    W_2_2,W_2_3,W_2_4,
                    W_3_2,W_3_3,W_3_4,
                    Y_2,Y_3,Y_4,
                    dϕ_1_dt,dϕ_2_dt,dϕ_3_dt,dϕ_4_dt,
                    dW_1_2_dt,dW_1_3_dt,dW_1_4_dt,
                    dW_2_2_dt,dW_2_3_dt,dW_2_4_dt,
                    dW_3_2_dt,dW_3_3_dt,dW_3_4_dt,
                    dY_2_dt,dY_3_dt,dY_4_dt,
                    E_t,gw,gy,gp2,vev,lambda,θ_w,dx)
                synchronize()
                if mod(it,dsnaps_fft)==0
                    Az_local_arr = Array(E_t)
                end
                # MPI.Barrier(comm_cart)
        
                if mod(it,dsnaps_fft)==0
                    phi_local_arr=sqrt.(Array(ϕ_1).^2+Array(ϕ_2).^2+Array(ϕ_3).^2+Array(ϕ_4).^2)
                end
            ############
            MPI.Barrier(comm_cart)


            CUDA.memory_status()

            ##Sum local energies##
            begin
                total_energies[snp_idx,1] = pe_local
                total_energies[snp_idx,2] = ke_local
                total_energies[snp_idx,3] = ge_local
                total_energies[snp_idx,4] = eew_local
                total_energies[snp_idx,5] = mew_local
                total_energies[snp_idx,6] = eey_local
                total_energies[snp_idx,7] = mey_local
                total_energies[snp_idx,8] = (0.5*(bxe_local .+bye_local .+bze_local))*Vol
                total_energies[snp_idx,9] = (0.5*(bx2e_local .+by2e_local .+bz2e_local))*Vol
                total_energies[snp_idx,10]=minimum(sqrt.(Array(ϕ_1).^2 .+ Array(ϕ_2).^2 .+ Array(ϕ_3).^2 .+ Array(ϕ_4).^2))
            end

            # Gather into global metrics #

            gather_metrics(total_energies[snp_idx,:],total_energies_global,me,comm_cart,nprocs,metrics_length)
            MPI.Barrier(comm_cart)
            
            h5open(string("local_energies_proc_",me,"_",it,"_data.h5"), "w") do file
                write(file, "energies local", total_energies)
            end 
        
            h5open(string("proc_",me,"_",it,"_data.h5"), "w") do file
                write(file, "B_x", B_x_local_arr)
                write(file, "B_y", B_y_local_arr)
                write(file, "B_z", B_z_local_arr)
                # write(file, "B_x2", B_x2_local_arr)
                # write(file, "B_y2", B_y2_local_arr)
                # write(file, "B_z2", B_z2_local_arr)
                write(file, "Ax", Ax_local_arr)
                write(file, "Ay", Ay_local_arr)
                write(file, "Az", Az_local_arr)
                write(file,"phi", phi_local_arr)
            end

            MPI.Barrier(comm_cart)

            if me==0
                
                ##Print energies##

                begin
                    proc_iter = range(1,nprocs,step=1)
        
                    total_pe_global = sum([total_energies_global[a] for a in metrics_length.*(proc_iter.-1).+1])
                    total_ke_global = sum([total_energies_global[a] for a in metrics_length.*(proc_iter.-1).+2])
                    total_ge_global = sum([total_energies_global[a] for a in metrics_length.*(proc_iter.-1).+3])
                    total_mew_global = sum([total_energies_global[a] for a in metrics_length.*(proc_iter.-1).+4])
                    total_eew_global = sum([total_energies_global[a] for a in metrics_length.*(proc_iter.-1).+5])
                    total_mey_global = sum([total_energies_global[a] for a in metrics_length.*(proc_iter.-1).+6])
                    total_eey_global = sum([total_energies_global[a] for a in metrics_length.*(proc_iter.-1).+7])
                    total_em_mag_global = sum([total_energies_global[a] for a in metrics_length.*(proc_iter.-1).+8])
                    total_em_mag_2_global = sum([total_energies_global[a] for a in metrics_length.*(proc_iter.-1).+9])
                    min_phi_global = minimum([total_energies_global[a] for a in metrics_length.*(proc_iter.-1).+10])

                    total_e_global = (total_pe_global+total_ke_global+total_ge_global+total_mew_global+
                                        total_eew_global+total_mey_global+total_eey_global)
        
                    energy_stack[snp_idx,1] = total_pe_global
                    energy_stack[snp_idx,2] = total_ke_global
                    energy_stack[snp_idx,3] = total_ge_global
                    energy_stack[snp_idx,4] = total_mew_global
                    energy_stack[snp_idx,5] = total_eew_global
                    energy_stack[snp_idx,6] = total_mey_global
                    energy_stack[snp_idx,7] = total_eey_global
                    energy_stack[snp_idx,8] = total_em_mag_global
                    energy_stack[snp_idx,9] = total_em_mag_2_global
                    energy_stack[snp_idx,10] = min_phi_global
        
                    println(string("--------Energies--t:",it,"--process:----------"))
                    println("Potentianl energy Higgs: ",total_pe_global)
                    println("Kinetic energy Higgs: ",total_ke_global)
                    println("Gradient energy Higgs:",total_ge_global)
                    println("Magnetic energy W: ",total_mew_global)
                    println("Electric energy W: ",total_eew_global)
                    println("Magnetic energy Y: ",total_mey_global)
                    println("Electric energy Y: ",total_eey_global)
                    println("EM-Magnetic energy: ",total_em_mag_global)
                    println("EM-Magnetic energy(2): ",total_em_mag_2_global)
                    println("Total energy: ", total_e_global)
                    println("Min Phi",min_phi_global)
                    println("---------------------------------------")
                end

                ##Dump data##

                h5open(string(it,"_energies_data.h5"), "w") do file
                    write(file, "energies", energy_stack)
                    write(file, "energies_split", total_energies_global)
                end 

            end

            MPI.Barrier(comm_cart)


        end
        ###############

    end

    if me == 0
        # gif(anim, "EW3d_test.mp4", fps = FPS)

        tot_energy_snp = [sum(energy_stack[a,1:end-3]) for a in range(1,nsnaps+1,step=1)]

        gr()
        ENV["GKSwstype"]="nul"
        plot(range(0,nte,step=dsnaps).*dt*mH,[energy_stack[:,1] energy_stack[:,2] energy_stack[:,3] energy_stack[:,8] tot_energy_snp].+1.0,
                label=["PE" "KE" "GE" "B" "Total"],xlims=(0,nte.*dt*mH),yscale=:log10,dpi=600)
        png("energies.png")

        gr()
        ENV["GKSwstype"]="nul"
        plot(range(0,nte,step=dsnaps).*dt*mH,[energy_stack[:,1] energy_stack[:,2] energy_stack[:,3] energy_stack[:,8] tot_energy_snp],
                label=["PE" "KE" "GE" "B" "Total"],xlims=(0,nte.*dt*mH),dpi=600)
        png("energies-linear.png")


        # gr()
        # ENV["GKSwstype"]="nul"
        # y1 = (((B_fft[1,2:end,1]).^2)./((2*pi)^3*Nx^2)).*B_fft[1,2:end,2]
        # y2 = (((B_fft[end,2:end,1]).^2)./((2*pi)^3*Nx^2)).*B_fft[end,2:end,2]
        # plot(B_fft[end,2:end,1],[y1 y2],label=[0 nte],xscale=:log10,yscale=:log10,minorgrid=true)
        # png("spectra.png")

        gr()
        ENV["GKSwstype"]="nul"
        plot(range(0,nte,step=dsnaps).*dt*mH,energy_stack[:,10],dpi=600)
        png("min-phi.png")

    end

    MPI.Barrier(comm_cart)

    # ##FFT and convolve##
    # begin
    #     # B_x_fft = Array(fft(B_x))
    #     # B_y_fft = Array(fft(B_y))
    #     # B_z_fft = Array(fft(B_z))

    #     gather_fft(B_x_fft,B_x_fft_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
    #     gather_fft(B_y_fft,B_y_fft_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
    #     gather_fft(B_z_fft,B_z_fft_g,Nx-6,Ny-6,Nz-6,me,comm_cart,nprocs)
    #     MPI.Barrier(comm_cart)

    #     # if me==0
    #     #     B_fft[end,:,:] = convolve_1d((real(conj.(B_x_fft_g).*B_x_fft_g.+
    #     #     conj.(B_y_fft_g).*B_y_fft_g.+
    #     #     conj.(B_z_fft_g).*B_z_fft_g)),Nx_g,Ny_g,Nz_g,spec_cut[1],spec_cut[2],spec_cut[3])
    #     # end

    #     # MPI.Barrier(comm_cart)
    #     end

    # ##END FFT##
    
    # if me==0
    #     h5open(string("data.h5"), "w") do file
    #         write(file, "energies", energy_stack)
    #         write(file, "B_x", B_x_local_arr)
    #         write(file, "B_y", B_y_local_arr)
    #         write(file, "B_z", B_z_local_arr)
    #     end  
    # end
 
    MPI.Barrier(comm_cart)
    MPI.Finalize()
end
run(thermal_init=true)
CUDA.memory_status()
