#Script to test MPI-CUDA multi gpu functionality
#All CUDA devices on multiple nodes have been tested 
#to display expected functionality for process topology
#and MPI commands

using MPI
using CUDA
MPI.Init()
println(MPI.has_cuda())
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
# select device
comm_l = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, rank)
rank_l = MPI.Comm_rank(comm_l)
gpu_id = CUDA.device!(rank_l)
# select device
size = MPI.Comm_size(comm)
dst  = mod(rank+1, size)
src  = mod(rank-1, size)
println("rank=$rank rank_loc=$rank_l (gpu_id=$gpu_id), size=$size, dst=$dst, src=$src")
N = 4
send_mesg = CuArray{Float64}(undef, N)
recv_mesg = CuArray{Float64}(undef, N)
fill!(send_mesg, Float64(rank))
CUDA.synchronize()
rank==0 && println("start sending...")
MPI.Sendrecv!(send_mesg, dst, 0, recv_mesg, src, 0, comm)
println("recv_mesg on proc $rank_l: $recv_mesg")
rank==0 && println("done.")