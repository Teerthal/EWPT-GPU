#July 23
#Tested custom MPI/CUDA routines to confirm multi-GPU functionality
#Tested custom update_halo, gather and boundary communication kernels
#built on MPI initiated multi-GPU topology

using CUDA#, CuArrays
using MPI

# # Initialize MPI
# MPI.Init()

# # Get the number of available CUDA devices and rank of the current process
# # num_devices = CUDA.device_count()
# rank = MPI.Comm_rank(MPI.COMM_WORLD)
# println(rank)
# # Create CUDA-aware MPI communicator
# # cuda_comm = MPI.CUDA_COMM_WORLD

# # Set the active CUDA device based on the process rank
# println(CUDA.device!(rank))

# MPI.Finalize()

using MPI
println(MPI.has_cuda())
@views function update_halo!(A,neighbors_x,neighbors_y,neighbors_z,comm)
	#x direction
	#Left
    # println(neighbors_x,neighbors_y,neighbors_z)
	if neighbors_x[1] != MPI.MPI_PROC_NULL
        # buf_size = 3*(size(A,2)*size(A,3))
        # sendbuf = reshape(Array(A[4:6 ,:,:]),buf_size)
		sendbuf = Array(A[4:6 ,:,:])
        # println(size(sendbuf))
        # recvbuf = zeros(buf_size)
		recvbuf = zeros(size(sendbuf))
        # println(size(recvbuf))
		# r=MPI.Isend(sendbuf, neighbors_x[1], 0, comm)
		# MPI.Wait!(r)
		# r=MPI.Irecv!(recvbuf, neighbors_x[1], 1, comm)
		# MPI.Wait!(r)
		MPI.Sendrecv!(sendbuf,neighbors_x[1],0,
					  recvbuf,neighbors_x[1],1, comm)
        # copyto!(A[1:3 ,:,:],reshape(recvbuf,size(A[1:3 ,:,:])))
		copyto!(A[1:3 ,:,:],recvbuf)

        # sendbuf = zeros(size(A[4:6 ,:,:]))
        # recvbuf = zeros(size(A[1:3 ,:,:]))
        # copyto!(sendbuf,A[4:6 ,:,:])
		# MPI.Isend(sendbuf, neighbors_x[1], 0, comm)
		# MPI.Irecv!(recvbuf, neighbors_x[1], 1, comm)
        # copyto!(A[1:3 ,:,:],recvbuf)

	end
	#Right
	if neighbors_x[2] != MPI.MPI_PROC_NULL
        # buf_size = 3*(size(A,2)*size(A,3))
        # sendbuf = reshape(Array(A[end-5:end-3 ,:,:]),buf_size)
		sendbuf = Array(A[end-5:end-3 ,:,:])
        # recvbuf = zeros(buf_size)
		recvbuf = zeros(size(sendbuf))
		# r=MPI.Isend(sendbuf, neighbors_x[2], 1, comm)
		# MPI.Wait!(r)
		# r=MPI.Irecv!(recvbuf, neighbors_x[2], 0, comm)
		# MPI.Wait!(r)
		MPI.Sendrecv!(sendbuf,neighbors_x[2],1,
		recvbuf,neighbors_x[2],0, comm)
        # copyto!(A[end-2:end ,:,:],reshape(recvbuf,size(A[end-2:end ,:,:])))
		copyto!(A[end-2:end ,:,:],recvbuf)

        # sendbuf = zeros(size(A[end-5:end-3 ,:, :]))
        # recvbuf = zeros(size(A[end-2:end , :, :]))
        # copyto!(sendbuf,A[end-5:end-3 ,:, :])
		# MPI.Isend(sendbuf,  neighbors_x[2], 1, comm)
		# MPI.Irecv!(recvbuf, neighbors_x[2], 0, comm)
        # copyto!(A[end-2:end , :, :],recvbuf)
	end
	# #y direction
	# #Back
	
	if neighbors_y[1] != MPI.MPI_PROC_NULL
        # buf_size = 3*(size(A,1)*size(A,3))
        # sendbuf = reshape(Array(A[:,4:6 ,:]),buf_size)
		sendbuf = Array(A[:,4:6 ,:])
        # recvbuf = zeros(buf_size)
		recvbuf = zeros(size(sendbuf))
		# r=MPI.Isend(sendbuf, neighbors_y[1], 0, comm)
		# MPI.Wait!(r)
		# r=MPI.Irecv!(recvbuf, neighbors_y[1], 1, comm)
		# MPI.Wait!(r)
		MPI.Sendrecv!(sendbuf,neighbors_y[1],0,
		recvbuf,neighbors_y[1],1, comm)
        # copyto!(A[:,1:3 ,:],reshape(recvbuf,size(A[:,1:3 ,:])))
		copyto!(A[:,1:3 ,:],recvbuf)

        # sendbuf = zeros(size(A[:,4:6 ,:]))
        # recvbuf = zeros(size(A[:,1:3 ,:]))
        # copyto!(sendbuf,A[:,4:6 ,:])
		# MPI.Isend(sendbuf, neighbors_y[1], 0, comm)
		# MPI.Irecv!(recvbuf, neighbors_y[1], 1, comm)
        # copyto!(A[:,1:3,:],recvbuf)
	end
	#Forward
	if neighbors_y[2] != MPI.MPI_PROC_NULL
        # buf_size = 3*(size(A,1)*size(A,3))
        # sendbuf = reshape(Array(A[:,end-5:end-3 , :]),buf_size)
		sendbuf = Array(A[:,end-5:end-3 , :])
        # recvbuf = zeros(buf_size)
		recvbuf = zeros(size(sendbuf))
		# r=MPI.Isend(sendbuf, neighbors_y[2], 1, comm)
		# MPI.Wait!(r)
		# r=MPI.Irecv!(recvbuf, neighbors_y[2], 0, comm)
		# MPI.Wait!(r)
		MPI.Sendrecv!(sendbuf,neighbors_y[2],1,
		recvbuf,neighbors_y[2],0, comm)
        # copyto!(A[:,end-2:end , :],reshape(recvbuf,size(A[:,end-2:end , :])))
		copyto!(A[:,end-2:end , :],recvbuf)

        # sendbuf = zeros(size(A[:,end-5:end-3 , :]))
        # recvbuf = zeros(size(A[:,end-2:end , :]))
        # copyto!(sendbuf,A[:,end-5:end-3 , :])
		# MPI.Isend(sendbuf,  neighbors_y[2], 1, comm)
		# MPI.Irecv!(recvbuf, neighbors_y[2], 0, comm)
        # copyto!(A[:,end-2:end , :],recvbuf)
	end
	
	#z direction
	#Up
	if neighbors_z[1] != MPI.MPI_PROC_NULL
        # buf_size = 3*(size(A,1)*size(A,2))
        # sendbuf = reshape(Array(A[:,:,4:6]),buf_size)
		sendbuf = Array(A[:,:,4:6])
        # recvbuf = zeros(buf_size)
		recvbuf = zeros(size(sendbuf))
		# r=MPI.Isend(sendbuf, neighbors_z[1], 0, comm)
		# MPI.Wait!(r)
		# r=MPI.Irecv!(recvbuf, neighbors_z[1], 1, comm)
		# MPI.Wait!(r)
		MPI.Sendrecv!(sendbuf,neighbors_z[1],0,
		recvbuf,neighbors_z[1],1, comm)
        # copyto!(A[:,:,1:3],reshape(recvbuf,size(A[:,:,1:3])))
		copyto!(A[:,:,1:3],recvbuf)

        # sendbuf = zeros(size(A[:,:,4:6]))
        # recvbuf = zeros(size((A[:,:,1:3]))
        # copyto!(sendbuf,A[:,:,4:6])
		# MPI.Isend(sendbuf,  neighbors_z[1], 1, comm)
		# MPI.Irecv!(recvbuf, neighbors_z[1], 0, comm)
        # copyto!((A[:,:,1:3],recvbuf)
	end
	#Down
	if neighbors_z[2] != MPI.MPI_PROC_NULL
        # buf_size = 3*(size(A,1)*size(A,2))
        # sendbuf = reshape(Array(A[:,:,end-5:end-3]),buf_size)
		sendbuf = Array(A[:,:,end-5:end-3])
        # recvbuf = zeros(buf_size)
		recvbuf = zeros(size(sendbuf))
		# r=MPI.Isend(sendbuf, neighbors_z[2], 1, comm)
		# MPI.Wait!(r)
		# r=MPI.Irecv!(recvbuf, neighbors_z[2], 0, comm)
		# MPI.Wait!(r)
		MPI.Sendrecv!(sendbuf,neighbors_z[2],1,
		recvbuf,neighbors_z[2],0, comm)
        # copyto!(A[:,:,end-2:end],reshape(recvbuf,size(A[:,:,end-2:end])))
		copyto!(A[:,:,end-2:end],recvbuf)

        # sendbuf = zeros(size(A[:, :,end-5:end-3 ]))
        # recvbuf = zeros(size(A[:, :,end-2:end ]))
        # copyto!(sendbuf,A[:, :,end-5:end-3 ])
		# MPI.Isend(sendbuf,  neighbors_z[2], 1, comm)
		# MPI.Irecv!(recvbuf, neighbors_z[2], 0, comm)
        # copyto!(A[:, :,end-2:end],recvbuf)
	end
	return
end

@views function boundary_x(A,dims,comm,coords)
    # left_rank = MPI.Cart_rank(comm,[0,0,0])
    # right_rank = MPI.Cart_rank(comm,[dims[1]-1,0,0])
    if dims[1]>1
        if (coords[1]==(dims[1]-1))
        # if me == right_rank
            sendbuf=Array(A[end-5:end,:,:])
            send_rank = MPI.Cart_rank(comm,[0,coords[2],coords[3]])
            req=MPI.Isend(sendbuf,send_rank,0,comm)
            MPI.Wait!(req)
        end
        if (coords[1]==0)
            recv_rank = MPI.Cart_rank(comm,[dims[1]-1,coords[2],coords[3]])
            recvbuf=zeros(size(Array(A[end-5:end,:,:])))
            req=MPI.Irecv!(recvbuf,recv_rank,0,comm)
            MPI.Wait!(req)
            copyto!(A[1:6,:,:],recvbuf)
        end
    else
        A[1:6,:,:]=A[end-5:end,:,:]
    end

    return
end

@views function boundary_y(A,dims,comm,coords,nprocs,neighbors_x,neighbors_y,neighbors_z)
    # left_rank = MPI.Cart_rank(comm,[0,0,0])
    # right_rank = MPI.Cart_rank(comm,[dims[1]-1,0,0])
    if dims[2]>1
        if (coords[2]==(dims[2]-1))
        # if me == right_rank
            sendbuf=Array(A[:,end-5:end,:])
            send_rank = MPI.Cart_rank(comm,[coords[1],0,coords[3]])
            req=MPI.Isend(sendbuf,send_rank,0,comm)
            MPI.Wait!(req)
        end
        if (coords[2]==0)
            recv_rank = MPI.Cart_rank(comm,[coords[1],dims[2]-1,coords[3]])
            recvbuf=zeros(size(Array(A[:,end-5:end,:])))
            req=MPI.Irecv!(recvbuf,recv_rank,0,comm)
            MPI.Wait!(req)
            copyto!(A[:,1:6,:],recvbuf)
        end
    else
        A[:,1:6,:]=A[:,end-5:end,:]
    end

    return
end

@views function boundary_z(A,dims,comm,coords,nprocs,neighbors_x,neighbors_y,neighbors_z)
    # left_rank = MPI.Cart_rank(comm,[0,0,0])
    # right_rank = MPI.Cart_rank(comm,[dims[1]-1,0,0])
    if dims[3]>1
        if (coords[3]==(dims[3]-1))
        # if me == right_rank
            sendbuf=Array(A[:,:,end-5:end])
            send_rank = MPI.Cart_rank(comm,[coords[1],coords[2],0])
            req=MPI.Isend(sendbuf,send_rank,0,comm)
            MPI.Wait!(req)
        end
        if (coords[3]==0)
            recv_rank = MPI.Cart_rank(comm,[coords[1],coords[2],dims[3]-1])
            recvbuf=zeros(size(Array(A[:,:,end-5:end])))
            req=MPI.Isend(recvbuf,recv_rank,0,comm)
            MPI.Wait!(req)
            copyto!(A[:,:,1:6],recvbuf)
        end
    else
        A[:,:,1:6]=A[:,:,end-5:end]
    end

    return
end


@views function gather(A,A_global,Nx,Ny,Nz,me,comm)
	sendbuf=Array(A[4:end-3,4:end-3,4:end-3])
	println(sendbuf[1,1,1])
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
			println(A_c[1,1,1])
        end
    end
    return
end



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

#println(me)
#println(coords)
# if me == 0
	dims, periods, coords = MPI.Cart_get(comm_cart)
	println(dims)
	println(periods)
	println(coords)
	println(neighbors_x)
	println(neighbors_y)
	println(neighbors_z)
# end
# println(MPI.MPI_PROC_NULL)
println(string(me," ",coords))
# exit()
n =30*2#*4
u = CUDA.zeros(Float64,(n,n,n)).+me*2
u_g = zeros(((n-6)*dims[1],(n-6)*dims[2],(n-6)*dims[3]))

if me==0 println(u[1:6]) end
MPI.Barrier(comm_cart)
boundary_x(u,dims,comm_cart,coords)
MPI.Barrier(comm_cart)
if me==0 println(u[1:6]) end



MPI.Barrier(comm_cart)
gather(u,u_g,n-6,n-6,n-6,me,comm_cart)
MPI.Barrier(comm_cart)
println("finished gather")
# u = zeros((n,n,n))
println(typeof(u))

if me ==0
	println(u_g[1,1,1])
	println(u_g[end-6,1,1])
end
MPI.Barrier(comm_cart)

# if me==0
# println(size(u[:,:, 4:6]))
# println(size(u[:,:, 1:3]))
# println(size(u[:, :, end-5:end-3]))
# println(size(u[:, :, end-2:end]))
# end
# if me > 0
# 	MPI.Isend(u[:,:, 4:6], me-1, 0, comm_cart)
# 	MPI.Irecv!(u[:,:, 1:3], me-1, 1, comm_cart)
# end
# if me < nprocs - 1
# 	# Isend boundary data from the right device to the current device
# 	MPI.Isend(u[:, :, end-5:end-3], me+1, 1, comm_cart)
# 	MPI.Irecv!(u[:, :, end-2:end], me+1, 0, comm_cart)
# end

# if neighbors_x[1] != MPI.MPI_PROC_NULL
# MPI.Isend(u[:,:, 4:6], neighbors_x[1], 0, comm_cart)
# MPI.Irecv!(u[:,:, 1:3], neighbors_x[1], 1, comm_cart)
# end
# # Isend boundary data from the right device to the current device
# if neighbors_x[2] != MPI.MPI_PROC_NULL
# MPI.Isend(u[:, :, end-5:end-3], neighbors_x[2], 1, comm_cart)
# MPI.Irecv!(u[:, :, end-2:end], neighbors_x[2], 0, comm_cart)
# end
# for tries in 1:50

if me==0 println(u[end-6:end,1,1]) end

@time update_halo!(u,neighbors_x,neighbors_y,neighbors_z,comm_cart)
# end

MPI.Barrier(comm_cart)
if me==0 println(u[end-6:end,1,1]) end
println("finished")
MPI.Finalize()

