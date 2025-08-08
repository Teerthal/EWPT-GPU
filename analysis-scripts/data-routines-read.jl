module data_managers

using WriteVTK
using HDF5
using Plots
using Statistics
using StatsBase
using FFTW
using LaTeXStrings
using LsqFit
using Glob
using Distributed

export prec,int_prec
prec = Float64
int_prec = Int


include("spec_routines.jl")
using .spec_convolver

###Data-set-parameters###

run_type = "thermal-ensemble-prod"
# run_type = "single_runs"
# run_type = "single_runs/peak-scale-single-runs"
run_type = "thermal_ens_average"
# run_type = "single_runs/new-runs-dx-0.1"
# run_type = "thermal_ens_average-compare"
# run_type = "damp_compare"

export run_type

export para_arr,lambda,vev,skip,sweep,k_xlim_upper,rewrite,mH

#general-single-runs#
run_paras_1 = ["0.0", 2, 60000, 0.15, 10.0, 256, 0.99, 300, 0.25, 0.0, 0.0, 1]
para_arr = [run_paras_1]

##Single 0.15 runs##
run_paras_1 = ["0.001", 2, 120000, 0.15, 20.0, 256, 0.99, 300, 0.25, 0.0, 0.0, 2]
run_paras_1 = ["0.01", 2, 120000, 0.15, 20.0, 256, 0.99, 300, 0.25, 0.0, 0.0, 2]
run_paras_1 = ["0.001", 2, 120000, 0.1, 10.0, 256, 0.99, 300, 0.25, 0.0, 0.0, 2]

para_arr=[run_paras_1]

##Single 0.1 runs##
run_paras_1 = ["0.0", 2, 160000, 0.1, 10.0, 320, 0.99, 300, 0.25, 0.0, 0.0, 2]
run_paras_1 = ["0.0", 2, 160000, 0.1, 20.0, 320, 0.99, 300, 0.25, 0.0, 0.0, 2]
run_paras_1 = ["0.001", 2, 160000, 0.1, 20.0, 320, 0.99, 300, 0.25, 0.0, 0.0, 2]

para_arr=[run_paras_1]


##peak-scale-runs##

# run_paras_1 = ["0.0", 2, 50000, 0.15, 10.0, 192, 0.99, 500, 0.25, 0.0, 0.0, 1]
# run_paras_2 = ["0.0", 2, 50000, 0.15, 10.0, 256, 0.99, 500, 0.25, 0.0, 0.0, 1]
# run_paras_3 = ["0.0", 2, 50000, 0.15, 10.0, 320, 0.99, 500, 0.25, 0.0, 0.0, 1]

# para_arr = [run_paras_1,run_paras_2,run_paras_3]

#thermal-ensemble-runs#

# run_paras_1 = ["0.0", 2, 80000, 0.15, 10.0, 256, 0.99, 400, 0.25, 0.0, 0.0, 1]
run_paras_1 = ["0.0", 2, 80000, 0.15, 10.0, 320, 0.99, 400, 0.25, 0.0, 0.0, 1]
# run_paras_1 = ["0.001", 2, 100000, 0.15, 20.0, 320, 0.99, 500, 0.25, 0.0, 0.0, 1]
# run_paras_1 = ["0.0", 2, 160000, 0.1, 10.0, 320, 0.99, 400, 0.25, 0.0, 0.0, 1]

para_arr = [run_paras_1]

#cp term comparison#

# run_paras_1 = ["0.0", 2, 60000, 0.15, 10.0, 256, 0.99, 300, 0.25, 0.1, 0.0, 1]
# run_paras_2 = ["0.0", 2, 60000, 0.15, 10.0, 256, 0.99, 300, 0.25, 0.0, 0.1, 1]
# run_paras_3 = ["0.0", 2, 60000, 0.15, 10.0, 256, 0.99, 300, 0.25, 0.0, 0.0, 2]

# para_arr = [run_paras_1,run_paras_2,run_paras_3]


#damp comparison
run_paras_1=["0.001", 2, 160000, 0.1, 20.0, 320, 0.99, 400, 0.25, 0.0, 0.0, 3]
run_paras_2=["0.0001", 2, 160000, 0.1, 20.0, 320, 0.99, 400, 0.25, 0.0, 0.0, 3]
run_paras_3=["0.00001", 2, 160000, 0.1, 20.0, 320, 0.99, 400, 0.25, 0.0, 0.0, 3]
run_paras_4=["0.000001", 2, 160000, 0.1, 20.0, 320, 0.99, 400, 0.25, 0.0, 0.0, 3]
run_paras_5=["0.05", 2, 160000, 0.1, 20.0, 320, 0.99, 400, 0.25, 0.0, 0.0, 3]
run_paras_6=["0.005", 2, 160000, 0.1, 20.0, 320, 0.99, 400, 0.25, 0.0, 0.0, 3]
run_paras_7=["0.0005", 2, 160000, 0.1, 20.0, 320, 0.99, 400, 0.25, 0.0, 0.0, 3]
run_paras_8=["0.00005", 2, 160000, 0.1, 20.0, 320, 0.99, 400, 0.25, 0.0, 0.0, 3]
run_paras_9=["0.000005", 2, 160000, 0.1, 20.0, 320, 0.99, 400, 0.25, 0.0, 0.0, 3]
run_paras_10=["0.01", 2, 160000, 0.1, 20.0, 320, 0.99, 400, 0.25, 0.0, 0.0, 3]
run_paras_11=["0.0", 2, 160000, 0.1, 20.0, 320, 0.99, 400, 0.25, 0.0, 0.0, 2]

para_arr = [run_paras_1,run_paras_2,
            run_paras_3,run_paras_4,
            run_paras_6,
            run_paras_7,
            run_paras_9,run_paras_10,
            run_paras_11]

# @everywhere model=$(ARGS[1])
if run_type == "thermal-ensemble-prod"
    # run_no = parse(Int,model)
    run_no = parse(Int,ARGS[1])
    # run_no = 2
end

no_runs=25

# #256 run-dir#
# if run_type == "thermal_ens_average"
#     para_arr = [["0.0", 2, 80000, 0.15, 10.0, 256, 0.99, 400, 0.25, 0.0, 0.0, i] for i in range(1,no_runs,step=1)]
# end

# # 320 run-dir#
# if run_type == "thermal_ens_average"
#     para_arr = [["0.0", 2, 80000, 0.15, 10.0, 320, 0.99, 400, 0.25, 0.0, 0.0, i] for i in range(1,no_runs,step=1)]
# end

#320 run-dir#gamma:0.001
# if run_type == "thermal_ens_average"
#     para_arr = [["0.001", 2, 100000, 0.15, 20.0, 320, 0.99, 500, 0.25, 0.0, 0.0, i] for i in range(1,no_runs,step=1)]
# end

#320 run-dir#dx=0.1
# if run_type == "thermal_ens_average"
#     para_arr = [["0.0", 2, 160000, 0.1, 10.0, 320, 0.99, 400, 0.25, 0.0, 0.0, i] for i in range(1,no_runs,step=1)]
#     # para_arr = []0.0 2 160000 0.1 10.0 320 0.99 400 0.25 0.0 0.0
# end

##256 run-dir#dx=0.1#dt=dx/20#nte:320000#
if run_type == "thermal_ens_average"
   para_arr = [["0.0", 2, 320000, 0.1, 20.0, 256, 0.99, 400, 0.25, 0.0, 0.0, i] for i in range(1,no_runs,step=1)]
   # para_arr = []0.0 2 160000 0.1 10.0 320 0.99 400 0.25 0.0 0.0
end


#320 run-dir#dx=0.15#dt=dx/20#nte:100000#gamma:0.01#
if run_type == "thermal_ens_average"
   para_arr = [["0.001", 2, 100000, 0.15, 20.0, 320, 0.99, 500, 0.25, 0.0, 0.0, i] for i in range(1,no_runs,step=1)]
end

#  #256 run-dir#gamma:0.01
# if run_type == "thermal_ens_average"
#     para_arr = [["0.001", 2, 320000, 0.1, 20.0, 256, 0.99, 400, 0.25, 0.0, 0.0, i] for i in range(1,no_runs,step=1)]
# end

#  #256 run-dir#gamma:0.01#T:0.46
#  if run_type == "thermal_ens_average"
#     para_arr = [["0.001", 2, 320000, 0.1, 20.0, 256, 0.99, 400, 0.46, 0.0, 0.0, i] for i in range(1,no_runs,step=1)]
# end


if run_type == "thermal_ens_average-compare"
    para_arr_1 = [["0.0", 2, 80000, 0.15, 10.0, 256, 0.99, 400, 0.25, 0.0, 0.0, i] for i in range(1,no_runs,step=1)]
    para_arr_2 = [["0.0", 2, 80000, 0.15, 10.0, 320, 0.99, 400, 0.25, 0.0, 0.0, i] for i in range(1,no_runs,step=1)]
    para_arr = vcat(para_arr_1,para_arr_2)
end

lambda = prec(0.129)
vev = prec(1.0)
mH = prec(2.0*sqrt(lambda)*vev)

skip = 1
sweep = 4
k_xlim_upper = 6
rewrite=false

##Manually specify number of fft snaps for local or broken analysis
override_snap_no = 5

loc = "local"
loc = "cluster"

# master_dir = string("/run/media/teerthal/X9 Pro/ewsb-sim/cluster-data/peak-scale-runs")
export master_dir
master_dir = string("/scratch/tpatel28/EW_sim/",run_type)
if ((run_type=="thermal_ens_average") || (run_type=="thermal_ens_average-compare"))
    master_dir = string("/scratch/tpatel28/EW_sim/thermal-ensemble-prod")
end

export max_spec_t_idx
max_spec_t_idx = 25 #320#
# max_spec_t_idx = 20 #256#
# max_spec_t_idx = 0 #320.dx:0.1#

# max_spec_t_idx = 0 #general single runs#

max_spec_t_idx = 15 #320#gamma:0.001


export dir_name
function dir_name(arr)

    if ((run_type=="thermal-ensemble-prod") || run_type=="thermal_ens_average" || run_type=="thermal_ens_average-compare")
        name = string("gamma-",arr[1],"-bubbles-",
        floor(int_prec,arr[2]),
        "-nte-",floor(int_prec,arr[3]),
        "-dx-",arr[4],"-_dt-",arr[5],
        "-latx-",floor(int_prec,arr[6]),
        "-gp2-",arr[7],"-nsnaps-",floor(int_prec,arr[8]),
        "-T-",arr[9],
        "-BW-",arr[10],"-BY-",arr[11])
    else
        name=string("gamma-",arr[1],"-bubbles-",
        floor(int_prec,arr[2]),
        "-nte-",floor(int_prec,arr[3]),
        "-dx-",arr[4],"-_dt-",arr[5],
        "-latx-",floor(int_prec,arr[6]),
        "-gp2-",arr[7],"-nsnaps-",floor(int_prec,arr[8]),
        "-T-",arr[9],
        "-BW-",arr[10],"-BY-",arr[11],
        "-seed-",floor(int_prec,arr[12]))
    end

    return name
end

function run_dir(arr)
    if run_type == "single_runs"
        path = string(master_dir,"/",dir_name(arr))
    elseif ((run_type=="thermal-ensemble-prod"))
        path = string(master_dir,"/",dir_name(arr),"/run-",run_no)
    elseif ((run_type=="thermal_ens_average")||run_type=="thermal_ens_average-compare")
        path = string(master_dir,"/",dir_name(arr),"/run-",floor(int_prec,arr[12]))
    elseif run_type == "single_runs/peak-scale-single-runs"
        path = string(master_dir,"/",dir_name(arr))
    elseif run_type == "single_runs/new-runs-dx-0.1"
        path = string(master_dir,"/",dir_name(arr))
    end  
    return path
end

export run_dirs
run_dirs = [run_dir(i) for i in para_arr]

function file_names(dir_name)
    
    file_names_3d = glob("3d-data*",dir_name)
    file_names_fft = glob("3d-fft*",dir_name)
    
    return file_names_3d,file_names_fft
end

export paras
function paras(run_dir,arr)
    no_fft_snaps=51
    
    #Pheno damping term
    γ=parse(prec,arr[1])#parse(prec,ARGS[1])
    no_bubbles = floor(int_prec,arr[2])#parse(int_prec,ARGS[2])
    nte = floor(int_prec,arr[3])#parse(int_prec,ARGS[3])
    dx=arr[4]#parse(prec,ARGS[4])
    dt=dx/arr[5]#(parse(prec,ARGS[5]))

    N = arr[6]

    # println(Nx,",",Ny,",",Nz)
    gp2 = arr[7]#parse(prec,ARGS[7])

    nsnaps=floor(int_prec,arr[8])#parse(int_prec,ARGS[8])
    
    #Temperature-going with 1/4\eta for now
    T=arr[9]#parse(prec,ARGS[9])

    dsnaps = floor(int_prec,nte/nsnaps)
    dsnaps_fft=floor(int_prec,nte/no_fft_snaps)

    if loc=="cluster"
        file_names_3d,file_names_fft = file_names(run_dir)
        no_fft_snaps = size(file_names_fft,1)
    elseif loc=="local"
    ##Overiding for local##
        no_fft_snaps = override_snap_no
    end

    time_skip=1
    fft_time_steps = range(0,no_fft_snaps-1,step=time_skip)
    
    dsnaps_fft=floor(int_prec,nte/(no_fft_snaps-1))
    time_stamps = [x*dsnaps_fft for x in (fft_time_steps)]
    return γ,no_bubbles,nte,dx,dt,N,gp2,nsnaps,T,dsnaps,dsnaps_fft,no_fft_snaps,time_stamps
end

export load_data
function load_data(run_dir)

    file = h5open(string(run_dir,"/data.h5"),"r")
        total_energies = read(file["energies"])
        # B_fft = read(file["B_linear_fft"])
        # fft_stack = read(file["3D_fft_stack"])
        # phys_stack = read(file["3D_phys_stack"])
        # slices_arr = read(file["y_slices_snaps"])
        ##headers
            # Nx=write(file,"Nx",Nx)
            # gp2=write(file,"gp2",gp2)
            # dx=write(file,"dx",dx)
            # dt=write(file,"dt",dt)
            # nte=write(file,"nte",nte)
            # γ=write(file,"gamma",γ)
            # T=write(file,"T",T)
            # meff_sq=write(file,"meff_sq",meff_sq)
            # t_sat=write(file,"t_sat_jex",t_sat)
            # nsnaps=write(file,"nsnaps",nsnaps)
            # dsnaps=write(file,"dsnaps",dsnaps)
        ###

    close(file)

    return total_energies#,B_fft#,fft_stack,phys_stack
end

function load_fft_data(run_dir,idx)
    # fname = string(run_dir,"/3d-fft-data-",idx,".h5")
    
    file_names_3d,file_names_fft = file_names(run_dir)
    file = h5open(file_names_fft[idx],"r")
        
        B_x_fft=read(file["3D_fft_x"])
        B_y_fft=read(file["3D_fft_y"])
        B_z_fft=read(file["3D_fft_z"])
        B_x2_fft=read(file["3D_fft_x2"])
        B_y2_fft=read(file["3D_fft_y2"])
        B_z2_fft=read(file["3D_fft_z2"])

    close(file)

    return B_x_fft,B_y_fft,B_z_fft,B_x2_fft,B_y2_fft,B_z2_fft
end


function load_phys_3D_data(run_dir,idx)
    file_names_3d,file_names_fft = file_names(run_dir)

    file = h5open(file_names_3d[idx],"r")
        
        phys_stack=read(file["3D_phys_stack"])
        B_x = phys_stack[1,:,:,:]
        B_y = phys_stack[2,:,:,:]
        B_z = phys_stack[3,:,:,:]

        B_x2 = phys_stack[4,:,:,:]
        B_y2 = phys_stack[5,:,:,:]
        B_z2 = phys_stack[6,:,:,:]

        A_x = phys_stack[7,:,:,:]
        A_y = phys_stack[8,:,:,:]
        A_z = phys_stack[9,:,:,:]

    close(file)

    return B_x,B_y,B_z,B_x2,B_y2,B_z2,A_x,A_y,A_z
end

function load_A_fft_data(run_dir,idx)

    fname = string(run_dir,"/fft-A-",idx,".h5")

    if ((isfile(fname)==false) || (rewrite==true))

    B_x,B_y,B_z,B_x2,B_y2,B_z2,A_x,A_y,A_z = load_phys_3D_data(idx)
    A_x_fft = fft(A_x)
    A_y_fft = fft(A_y)
    A_z_fft = fft(A_z)
    
    h5open(fname, "w") do file

        write(file,string("A_x_fft"),A_x_fft)
        write(file,string("A_y_fft"),A_y_fft)
        write(file,string("A_z_fft"),A_z_fft)
    end

    else
        file=h5open(fname,"r")
        A_x_fft=read(file["A_x_fft"])
        A_y_fft=read(file["A_y_fft"])
        A_z_fft=read(file["A_z_fft"])
        close(file)
    end

    return A_x_fft,A_y_fft,A_z_fft
end


function load_H_fft_data(run_dir,idx)

    fname = string(run_dir,"/fft-H-",idx,".h5")

    if ((isfile(fname)==false) || (rewrite==true))

    B_x,B_y,B_z,B_x2,B_y2,B_z2,A_x,A_y,A_z = load_phys_3D_data(run_dir,idx)
    H_x_fft = fft(A_x)
    H_y_fft = fft(A_y)
    H_z_fft = fft(A_z)
    
    h5open(fname, "w") do file

        write(file,string("H_x_fft"),H_x_fft)
        write(file,string("H_y_fft"),H_y_fft)
        write(file,string("H_z_fft"),H_z_fft)
    end

    else
        file=h5open(fname,"r")
        H_x_fft=read(file["H_x_fft"])
        H_y_fft=read(file["H_y_fft"])
        H_z_fft=read(file["H_z_fft"])
        close(file)
    end

    return H_x_fft,H_y_fft,H_z_fft
end

function re_fft_compute(B_x_fft,B_y_fft,B_z_fft,run_idx;skip=1,B_type_idx=1)
    # B_k_t = real.(fft_stack[1+3(B_type_idx-1),:,:,:].*conj(fft_stack[1+3(B_type_idx-1),:,:,:]).+
    #     fft_stack[2+3(B_type_idx-1),:,:,:].*conj(fft_stack[2+3(B_type_idx-1),:,:,:]).+
    #     fft_stack[3+3(B_type_idx-1),:,:,:].*conj(fft_stack[3+3(B_type_idx-1),:,:,:]))

    γ,no_bubbles,nte,dx,dt,N,gp2,nsnaps,T,dsnaps,dsnaps_fft,no_fft_snaps,time_stamps = paras(run_dirs[run_idx],para_arr[run_idx])
    Nx=Ny=Nz=N
    B_k_t = real.(B_x_fft.*conj(B_x_fft).+
                    B_y_fft.*conj(B_y_fft).+
                    B_z_fft.*conj(B_z_fft))

    re_spec_cut = [Nx,Ny,Nz]
    @time begin
        B_fft_re = convolve_1d(real(B_k_t),Nx,Ny,Nz,re_spec_cut[1],re_spec_cut[2],re_spec_cut[3])
        B_fft_re_binned = convolve_1d_binned(real(B_k_t),Nx,Ny,Nz,re_spec_cut[1],re_spec_cut[2],re_spec_cut[3],1)
    end
    
    stacked_fft = zeros((Nx*Ny*Nz,2))
    iter=1
    for i in range(1,Nx,step=1)
        for j in range(1,Ny,step=1)
            for k in range(1,Nz,step=1)
                # push!(stacked_fft,[K_c_mag(i,j,k,Nx,Ny,Nz),B_k_t[i,j,k]])
                stacked_fft[iter,1]=K_c_mag(i,j,k,Nx,Ny,Nz)
                stacked_fft[iter,2]=B_k_t[i,j,k]
                iter=iter+1
            end
        end
    end

    k_points =reduce(vcat,stacked_fft[2:skip:end,1]).*2*pi/(dx*Nx) #[a[1] for a in stacked_fft[2:skip:end]]*2*pi/(dx*Nx)
    B_k_points = reduce(vcat,stacked_fft[2:skip:end,2]).*(k_points.^2/(dx*Nx)^3) #.*([a[2] for a in stacked_fft[2:skip:end]])
    
    return k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned
end

function re_fft_compute_H(H_x_fft,H_y_fft,H_z_fft,run_idx;skip=1,B_type_idx=1)
    γ,no_bubbles,nte,dx,dt,N,gp2,nsnaps,T,dsnaps,dsnaps_fft,no_fft_snaps,time_stamps = paras(run_dirs[run_idx],para_arr[run_idx])
    Nx=Ny=Nz=N

    H_k_t = real.(H_x_fft.*conj(H_x_fft).+
                    H_y_fft.*conj(H_y_fft).+
                    H_z_fft.*conj(H_z_fft))

    re_spec_cut = [Nx,Ny,Nz]
    @time begin
        H_fft_re = convolve_1d(real(H_k_t),Nx,Ny,Nz,re_spec_cut[1],re_spec_cut[2],re_spec_cut[3])
        H_fft_re_binned = convolve_1d_binned(real(H_k_t),Nx,Ny,Nz,re_spec_cut[1],re_spec_cut[2],re_spec_cut[3],1)
    end
    
    stacked_fft_H = zeros((Nx*Ny*Nz,2))
    iter=1
    for i in range(1,Nx,step=1)
        for j in range(1,Ny,step=1)
            for k in range(1,Nz,step=1)
                stacked_fft_H[iter,1]=K_c_mag(i,j,k,Nx,Ny,Nz)
                stacked_fft_H[iter,2]=H_k_t[i,j,k]
                iter=iter+1
            end
        end
    end

    k_points_H =reduce(vcat,stacked_fft_H[2:skip:end,1]).*2*pi/(dx*Nx) #[a[1] for a in stacked_fft_H[2:skip:end]]*2*pi/(dx*Nx)
    H_k_points = reduce(vcat,stacked_fft_H[2:skip:end,2]).*(k_points_H.^2/(dx*Nx)^3) #.*([a[2] for a in stacked_fft_H[2:skip:end]])
    
    return k_points_H,H_k_points,H_fft_re,stacked_fft_H,H_fft_re_binned
end

function re_fft_compute_hel(B_x_fft,B_y_fft,B_z_fft,run_idx;skip=1,B_type_idx=1)
    γ,no_bubbles,nte,dx,dt,N,gp2,nsnaps,T,dsnaps,dsnaps_fft,no_fft_snaps,time_stamps = paras(run_dirs[run_idx],para_arr[run_idx])
    Nx=Ny=Nz=N

    # H_k_t = real.(H_x_fft.*conj(H_x_fft).+
    #                 H_y_fft.*conj(H_y_fft).+
    #                 H_z_fft.*conj(H_z_fft))
    
    karr = K_vec_arr(Nx,Ny,Nz).*(2*pi/(dx*Nx))

    hel_k_t = (1/(pi^2*(N*dx)^3))*imag((karr[:,:,:,1].*conj(B_y_fft).*(B_z_fft)).+
                                        (karr[:,:,:,2].*conj(B_z_fft).*(B_x_fft)).+
                                        (karr[:,:,:,3].*conj(B_x_fft).*(B_y_fft)))

    re_spec_cut = [Nx,Ny,Nz]
    @time begin
        hel_fft_re = convolve_1d(real(hel_k_t),Nx,Ny,Nz,re_spec_cut[1],re_spec_cut[2],re_spec_cut[3])
        hel_fft_re_binned = convolve_1d_binned(real(hel_k_t),Nx,Ny,Nz,re_spec_cut[1],re_spec_cut[2],re_spec_cut[3],1)
    end
    
    stacked_fft_hel = zeros((Nx*Ny*Nz,2))
    iter=1
    for i in range(1,Nx,step=1)
        for j in range(1,Ny,step=1)
            for k in range(1,Nz,step=1)
                stacked_fft_hel[iter,1]=K_c_mag(i,j,k,Nx,Ny,Nz)
                stacked_fft_hel[iter,2]=hel_k_t[i,j,k]
                iter=iter+1
            end
        end
    end

    k_points_hel =reduce(vcat,stacked_fft_hel[2:skip:end,1]).*2*pi/(dx*Nx) #[a[1] for a in stacked_fft_H[2:skip:end]]*2*pi/(dx*Nx)
    hel_k_points = reduce(vcat,stacked_fft_hel[2:skip:end,2]).*(k_points_hel.^2/(dx*Nx)^3) #.*([a[2] for a in stacked_fft_H[2:skip:end]])

    return k_points_hel,hel_k_points,hel_fft_re,stacked_fft_hel,hel_fft_re_binned
end

# export load_processed_fft_data
# function load_processed_fft_data(run_dir,fft_snp_idx,run_idx)
#     k_points=B_k_points=B_fft_re=stacked_fft=B_fft_re_binned=
#     H_k_points=H_fft_re=stacked_fft_H=H_fft_re_binned=
#     hel_k_points=hel_fft_re=stacked_fft_hel=hel_fft_re_binned=[]

#     # println(isfile(file_names_fft[fft_snp_idx]))
#     if ((isfile(string(run_dir,"/data-re-",fft_snp_idx,".h5"))==false) || (rewrite==true))
#         println("file does not exist or rewriting")
#         # restacked_fft = []

#         B_x_fft,B_y_fft,B_z_fft,B_x2_fft,B_y2_fft,B_z2_fft = load_fft_data(run_dir,fft_snp_idx)
#         # A_x_fft,A_y_fft,A_z_fft=load_A_fft_data(fft_snp_idx)
#         H_x_fft,H_y_fft,H_z_fft = load_H_fft_data(run_dir,fft_snp_idx)
#         # exit()

#         h5open(string(run_dir,"/data-re-",fft_snp_idx,".h5"), "w") do file

#             k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned=re_fft_compute(B_x_fft,B_y_fft,B_z_fft,run_idx,skip=skip,B_type_idx=2)
#             # push!(restacked_fft,[k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned])
#             # k_points_H,H_k_points,H_fft_re,stacked_fft_H,H_fft_re_binned=re_fft_compute_H(A_x_fft,A_y_fft,A_z_fft,B_x_fft,B_y_fft,B_z_fft,skip=skip,B_type_idx=2)
#             k_points_H,H_k_points,H_fft_re,stacked_fft_H,H_fft_re_binned=re_fft_compute_H(H_x_fft,H_y_fft,H_z_fft,run_idx,skip=skip,B_type_idx=2)

#             write(file,string("restack_",fft_snp_idx,"_kpoints"),k_points)
#             write(file,string("restack_",fft_snp_idx,"_Bkpoints"),B_k_points)
#             write(file,string("restack_",fft_snp_idx,"_Bfftre"),B_fft_re)
#             write(file,string("restack_",fft_snp_idx,"_stackedfft"),stacked_fft)
#             write(file,string("restack_",fft_snp_idx,"_Bfftrebinned"),B_fft_re_binned)

#             write(file,string("restack_",fft_snp_idx,"_Hkpoints"),H_k_points)
#             write(file,string("restack_",fft_snp_idx,"_Hfftre"),H_fft_re)
#             write(file,string("restack_",fft_snp_idx,"_Hstackedfft"),stacked_fft_H)
#             write(file,string("restack_",fft_snp_idx,"_Hfftrebinned"),H_fft_re_binned)

#         end
    

#     else
#         println("file exists. loading existing data")

#         file=h5open(string(run_dir,"/data-re-",fft_snp_idx,".h5"),"r")

#         # restacked_fft = []
        
#             k_points=read(file[string("restack_",fft_snp_idx,"_kpoints")])
#             B_k_points=read(file[string("restack_",fft_snp_idx,"_Bkpoints")])
#             B_fft_re=read(file[string("restack_",fft_snp_idx,"_Bfftre")])
#             stacked_fft=read(file[string("restack_",fft_snp_idx,"_stackedfft")])
#             B_fft_re_binned=read(file[string("restack_",fft_snp_idx,"_Bfftrebinned")])
#             # push!(restacked_fft,[k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned])

#             H_k_points=read(file[string("restack_",fft_snp_idx,"_Hkpoints")])
#             H_fft_re=read(file[string("restack_",fft_snp_idx,"_Hfftre")])
#             stacked_fft_H=read(file[string("restack_",fft_snp_idx,"_Hstackedfft")])
#             H_fft_re_binned=read(file[string("restack_",fft_snp_idx,"_Hfftrebinned")])
#         close(file)
#     end

#     ##Helicity data load##

#     if ((isfile(string(run_dir,"/data-re-hel-",fft_snp_idx,".h5"))==false) || (rewrite==true))
#         println("hel file does not exist or rewriting")
#         # restacked_fft = []

#         B_x_fft,B_y_fft,B_z_fft,B_x2_fft,B_y2_fft,B_z2_fft = load_fft_data(run_dir,fft_snp_idx)

#         h5open(string(run_dir,"/data-re-hel-",fft_snp_idx,".h5"), "w") do file

#             k_points_hel,hel_k_points,hel_fft_re,stacked_fft_hel,hel_fft_re_binned = re_fft_compute_hel(B_x_fft,B_y_fft,B_z_fft,run_idx;skip=1,B_type_idx=2)
#             write(file,string("restack_",fft_snp_idx,"_helkpoints"),hel_k_points)
#             write(file,string("restack_",fft_snp_idx,"_helfftre"),hel_fft_re)
#             write(file,string("restack_",fft_snp_idx,"_helstackedfft"),stacked_fft_hel)
#             write(file,string("restack_",fft_snp_idx,"_helfftrebinned"),hel_fft_re_binned)

#         end
    
#     else
#         println("hel file exists. loading existing data")

#         file=h5open(string(run_dir,"/data-re-hel-",fft_snp_idx,".h5"),"r")

#             hel_k_points=read(file[string("restack_",fft_snp_idx,"_helkpoints")])
#             hel_fft_re=read(file[string("restack_",fft_snp_idx,"_helfftre")])
#             stacked_fft_hel=read(file[string("restack_",fft_snp_idx,"_helstackedfft")])
#             hel_fft_re_binned=read(file[string("restack_",fft_snp_idx,"_helfftrebinned")])
#         close(file)
#     end

#     return k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned,
#             H_k_points,H_fft_re,stacked_fft_H,H_fft_re_binned,
#             hel_k_points,hel_fft_re,stacked_fft_hel,hel_fft_re_binned
# end


# export load_less_processed_fft_data
# function load_less_processed_fft_data(run_dir,fft_snp_idx,run_idx)
#     B_fft_re=B_fft_re_binned=H_fft_re=H_fft_re_binned=hel_fft_re=hel_fft_re_binned=[]
#     # println(isfile(file_names_fft[fft_snp_idx]))
#     if ((isfile(string(run_dir,"/data-less-re-",fft_snp_idx,".h5"))==false) || (rewrite==true))
#         println("file does not exist or rewriting")

#         k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned,
#         H_k_points,H_fft_re,stacked_fft_H,H_fft_re_binned,
#         hel_k_points,hel_fft_re,stacked_fft_hel,hel_fft_re_binned=load_processed_fft_data(run_dir,fft_snp_idx,run_idx)

#         h5open(string(run_dir,"/data-less-re-",fft_snp_idx,".h5"), "w") do file

#             write(file,string("restack_",fft_snp_idx,"_Bfftre"),B_fft_re)
#             write(file,string("restack_",fft_snp_idx,"_Bfftrebinned"),B_fft_re_binned)

#             write(file,string("restack_",fft_snp_idx,"_Hfftre"),H_fft_re)
#             write(file,string("restack_",fft_snp_idx,"_Hfftrebinned"),H_fft_re_binned)

#             write(file,string("restack_",fft_snp_idx,"_helfftre"),hel_fft_re)
#             write(file,string("restack_",fft_snp_idx,"_helfftrebinned"),hel_fft_re_binned)

#         end
    

#     else
#         println("file exists. loading existing data")

#         file=h5open(string(run_dir,"/data-less-re-",fft_snp_idx,".h5"),"r")

#         # restacked_fft = []
        
#             B_fft_re=read(file[string("restack_",fft_snp_idx,"_Bfftre")])
#             B_fft_re_binned=read(file[string("restack_",fft_snp_idx,"_Bfftrebinned")])

#             H_fft_re=read(file[string("restack_",fft_snp_idx,"_Hfftre")])
#             H_fft_re_binned=read(file[string("restack_",fft_snp_idx,"_Hfftrebinned")])

#             hel_fft_re=read(file[string("restack_",fft_snp_idx,"_helfftre")])
#             hel_fft_re_binned=read(file[string("restack_",fft_snp_idx,"_helfftrebinned")])


#         close(file)
#     end
#     return B_fft_re,B_fft_re_binned,H_fft_re,H_fft_re_binned,hel_fft_re,hel_fft_re_binned
# end


export load_processed_fft_data
function load_processed_fft_data(run_dir,fft_snp_idx,run_idx)
    k_points=B_k_points=B_fft_re=stacked_fft=B_fft_re_binned=
    H_k_points=H_fft_re=stacked_fft_H=H_fft_re_binned=
    hel_k_points=hel_fft_re=stacked_fft_hel=hel_fft_re_binned=[]

    # println(isfile(file_names_fft[fft_snp_idx]))
    if ((isfile(string(run_dir,"/data-re-",fft_snp_idx,".h5"))==false) || (rewrite==true))
        println("file does not exist or rewriting")
        # restacked_fft = []

        B_x_fft,B_y_fft,B_z_fft,B_x2_fft,B_y2_fft,B_z2_fft = load_fft_data(run_dir,fft_snp_idx)

        h5open(string(run_dir,"/data-re-",fft_snp_idx,".h5"), "w") do file

            k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned=re_fft_compute(B_x_fft,B_y_fft,B_z_fft,run_idx,skip=skip,B_type_idx=2)

            write(file,string("restack_",fft_snp_idx,"_kpoints"),k_points)
            write(file,string("restack_",fft_snp_idx,"_Bkpoints"),B_k_points)
            write(file,string("restack_",fft_snp_idx,"_Bfftre"),B_fft_re)
            write(file,string("restack_",fft_snp_idx,"_stackedfft"),stacked_fft)
            write(file,string("restack_",fft_snp_idx,"_Bfftrebinned"),B_fft_re_binned)

        end
    

    else
        println("file exists. loading existing data")

        file=h5open(string(run_dir,"/data-re-",fft_snp_idx,".h5"),"r")

        # restacked_fft = []
        
            k_points=read(file[string("restack_",fft_snp_idx,"_kpoints")])
            B_k_points=read(file[string("restack_",fft_snp_idx,"_Bkpoints")])
            B_fft_re=read(file[string("restack_",fft_snp_idx,"_Bfftre")])
            stacked_fft=read(file[string("restack_",fft_snp_idx,"_stackedfft")])
            B_fft_re_binned=read(file[string("restack_",fft_snp_idx,"_Bfftrebinned")])
        close(file)
    end

    ###hosking-hel-fft-ddata-load###
    if ((isfile(string(run_dir,"/data-re-H-",fft_snp_idx,".h5"))==false) || (rewrite==true))
        println("hosminghel spec file does not exist or rewriting")
        # println(run_idx,",",fft_snp_idx)
        # exit()
        # restacked_fft = []

        H_x_fft,H_y_fft,H_z_fft = load_H_fft_data(run_dir,fft_snp_idx)

        h5open(string(run_dir,"/data-re-H-",fft_snp_idx,".h5"), "w") do file

            k_points_H,H_k_points,H_fft_re,stacked_fft_H,H_fft_re_binned=re_fft_compute_H(H_x_fft,H_y_fft,H_z_fft,run_idx,skip=skip,B_type_idx=2)

            write(file,string("restack_",fft_snp_idx,"_Hkpoints"),H_k_points)
            write(file,string("restack_",fft_snp_idx,"_Hfftre"),H_fft_re)
            write(file,string("restack_",fft_snp_idx,"_Hstackedfft"),stacked_fft_H)
            write(file,string("restack_",fft_snp_idx,"_Hfftrebinned"),H_fft_re_binned)

        end
    

    else
        println("hoking-hel spec file exists. loading existing data")
        # println(run_idx,",",fft_snp_idx)

        file=h5open(string(run_dir,"/data-re-H-",fft_snp_idx,".h5"),"r")
            H_k_points=read(file[string("restack_",fft_snp_idx,"_Hkpoints")])
            H_fft_re=read(file[string("restack_",fft_snp_idx,"_Hfftre")])
            stacked_fft_H=read(file[string("restack_",fft_snp_idx,"_Hstackedfft")])
            H_fft_re_binned=read(file[string("restack_",fft_snp_idx,"_Hfftrebinned")])
        close(file)
    end

    ##Helicity data load##

    if ((isfile(string(run_dir,"/data-re-hel-",fft_snp_idx,".h5"))==false) || (rewrite==true))
        println("hel file does not exist or rewriting")
        # restacked_fft = []

        B_x_fft,B_y_fft,B_z_fft,B_x2_fft,B_y2_fft,B_z2_fft = load_fft_data(run_dir,fft_snp_idx)

        h5open(string(run_dir,"/data-re-hel-",fft_snp_idx,".h5"), "w") do file

            k_points_hel,hel_k_points,hel_fft_re,stacked_fft_hel,hel_fft_re_binned = re_fft_compute_hel(B_x_fft,B_y_fft,B_z_fft,run_idx;skip=1,B_type_idx=2)
            write(file,string("restack_",fft_snp_idx,"_helkpoints"),hel_k_points)
            write(file,string("restack_",fft_snp_idx,"_helfftre"),hel_fft_re)
            write(file,string("restack_",fft_snp_idx,"_helstackedfft"),stacked_fft_hel)
            write(file,string("restack_",fft_snp_idx,"_helfftrebinned"),hel_fft_re_binned)

        end
    
    else
        println("hel file exists. loading existing data")

        file=h5open(string(run_dir,"/data-re-hel-",fft_snp_idx,".h5"),"r")

            hel_k_points=read(file[string("restack_",fft_snp_idx,"_helkpoints")])
            hel_fft_re=read(file[string("restack_",fft_snp_idx,"_helfftre")])
            stacked_fft_hel=read(file[string("restack_",fft_snp_idx,"_helstackedfft")])
            hel_fft_re_binned=read(file[string("restack_",fft_snp_idx,"_helfftrebinned")])
        close(file)
    end

    return k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned,
            H_k_points,H_fft_re,stacked_fft_H,H_fft_re_binned,
            hel_k_points,hel_fft_re,stacked_fft_hel,hel_fft_re_binned
end

export load_less_processed_fft_data
function load_less_processed_fft_data(run_dir,fft_snp_idx,run_idx)
    B_fft_re=B_fft_re_binned=H_fft_re=H_fft_re_binned=hel_fft_re=hel_fft_re_binned=[]
    # println(isfile(file_names_fft[fft_snp_idx]))
    if ((isfile(string(run_dir,"/data-less-re-",fft_snp_idx,".h5"))==false) || (rewrite==true))
        println("file does not exist or rewriting")

        k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned,
        H_k_points,H_fft_re,stacked_fft_H,H_fft_re_binned,
        hel_k_points,hel_fft_re,stacked_fft_hel,hel_fft_re_binned=load_processed_fft_data(run_dir,fft_snp_idx,run_idx)

        h5open(string(run_dir,"/data-less-re-",fft_snp_idx,".h5"), "w") do file

            write(file,string("restack_",fft_snp_idx,"_Bfftre"),B_fft_re)
            write(file,string("restack_",fft_snp_idx,"_Bfftrebinned"),B_fft_re_binned)

        end
    

    else
        println("file exists. loading existing data")

        file=h5open(string(run_dir,"/data-less-re-",fft_snp_idx,".h5"),"r")

        # restacked_fft = []
        
            B_fft_re=read(file[string("restack_",fft_snp_idx,"_Bfftre")])
            B_fft_re_binned=read(file[string("restack_",fft_snp_idx,"_Bfftrebinned")])

        close(file)
    end

    ###hosking-hel-spec-data###

    if ((isfile(string(run_dir,"/data-less-re-H-",fft_snp_idx,".h5"))==false) || (rewrite==true))
        println("file does not exist or rewriting")

        k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned,
        H_k_points,H_fft_re,stacked_fft_H,H_fft_re_binned,
        hel_k_points,hel_fft_re,stacked_fft_hel,hel_fft_re_binned=load_processed_fft_data(run_dir,fft_snp_idx,run_idx)

        h5open(string(run_dir,"/data-less-re-H-",fft_snp_idx,".h5"), "w") do file

            write(file,string("restack_",fft_snp_idx,"_Hfftre"),H_fft_re)
            write(file,string("restack_",fft_snp_idx,"_Hfftrebinned"),H_fft_re_binned)

        end
    
    else
        println("file exists. loading existing data")

        file=h5open(string(run_dir,"/data-less-re-H-",fft_snp_idx,".h5"),"r")

            H_fft_re=read(file[string("restack_",fft_snp_idx,"_Hfftre")])
            H_fft_re_binned=read(file[string("restack_",fft_snp_idx,"_Hfftrebinned")])

        close(file)
    end

    ###hel-conv-data###
    if ((isfile(string(run_dir,"/data-less-re-hel-",fft_snp_idx,".h5"))==false) || (rewrite==true))
        println("file does not exist or rewriting")

        k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned,
        H_k_points,H_fft_re,stacked_fft_H,H_fft_re_binned,
        hel_k_points,hel_fft_re,stacked_fft_hel,hel_fft_re_binned=load_processed_fft_data(run_dir,fft_snp_idx,run_idx)

        h5open(string(run_dir,"/data-less-re-hel-",fft_snp_idx,".h5"), "w") do file

            write(file,string("restack_",fft_snp_idx,"_helfftre"),hel_fft_re)
            write(file,string("restack_",fft_snp_idx,"_helfftrebinned"),hel_fft_re_binned)

        end
    

    else
        println("file exists. loading existing data")

        file=h5open(string(run_dir,"/data-less-re-hel-",fft_snp_idx,".h5"),"r")

            hel_fft_re=read(file[string("restack_",fft_snp_idx,"_helfftre")])
            hel_fft_re_binned=read(file[string("restack_",fft_snp_idx,"_helfftrebinned")])

        close(file)
    end

    return B_fft_re,B_fft_re_binned,H_fft_re,H_fft_re_binned,hel_fft_re,hel_fft_re_binned
end

end