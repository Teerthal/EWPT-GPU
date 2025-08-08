using HDF5
using Plots
using Statistics
using StatsBase
# using FFTW
# using HDF5

directory = string("gamma-0.001-bubbles-1000-nte-50000-dx-0.05-_dt-25.0-latx-256-gp2-0.99-nsnaps-200-T-0.25")
directory = string("gamma-0.0-bubbles-20-nte-60000-dx-0.05-_dt-25.0-latx-128-gp2-0.99-nsnaps-100-T-0.25")

nprocs = 8
it = 0

# for me in 
# string("proc_",me,"_",it,"_data.h5")

# file = h5open(string(directory,"/",0,"_energies_data.h5"),"r")
# energies=read(file["energies"])
# energies_split=read(file["energies_split"])
# close(file)

# println(size(energies))
# println(size(energies_split))

# println(sum([energies_split[(me-1)*10+1] for me in range(1,64)]))

global_pe=0.
locals_pe = []
for me in range(0,nprocs-1)
    file = h5open(string(directory,"/","proc_",me,"_",it,"_data.h5"),"r")
    local_energies = read(file["energies local"])
    # println(me,":",size(local_energies))
    # println(local_energies[it+1,:])
    close(file)
    local_pe = local_energies[it+1,1]
    # println(me,":",local_pe)
    push!(locals_pe,local_pe)
    global global_pe = local_pe + global_pe
    # exit()
end
# println(locals_pe)
println(locals_pe)
println(minimum(locals_pe),",",maximum(locals_pe))
println(sum(locals_pe))
println(global_pe)

