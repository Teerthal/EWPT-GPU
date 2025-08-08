using WriteVTK
using HDF5
using Plots
using Statistics
using StatsBase
using FFTW

Nx=Ny=Nz=parse(Int,ARGS[6])#32*10
println(Nx,",",Ny,",",Nz)
gp2 = parse(prec,ARGS[7])
dx=parse(prec,ARGS[4])
dt=dx/(parse(prec,ARGS[5]))
nte = parse(int_prec,ARGS[3])
#Pheno damping term
γ=parse(prec,ARGS[1])
#Temperature-going with 1/4\eta for now
T=parse(prec,ARGS[9])

nsnaps=parse(int_prec,ARGS[8])

no_bubbles = parse(int_prec,ARGS[2])

master_dir = string("/home/tpatel28/topo_mag/EW_sim")
dir_name = string("gamma-",γ,"-bubbles-",no_bubbles,"-nte-",nte,"-dx-",dx,"-_dt-",_dt,"-latx-",latx,"-gp2-",gp2,"-nsnaps-",nsnaps,"-T-",T)

run_dir = string(master_dir,"/",dir_name)

function load_data(dir_path)

    file = h5open(string("data.h5"),"r")
        total_energies = read(file["energies"])
        B_fft = read(file["B_linear_fft"])
        fft_stack = read(file["3D_fft_stack"])
    close(file)

    return total_energies,B_fft,fft_stack
end