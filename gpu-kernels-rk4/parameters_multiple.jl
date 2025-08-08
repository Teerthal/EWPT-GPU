module parameters
using Random

export Nx,Ny,Nz
export gw,Wangle,gy,gp2,vev,lambda,mH,θ_w,γ_inp,T
export dx,dt,nte,tol,nsnaps,dsnaps,base_seed,no_bubbles,bub_diam
export FPS
export meff_sq

export prec,int_prec
prec=Float64
int_prec=Int64

Nx=Ny=Nz=128#parse(Int,ARGS[6])#32*10
println(Nx,",",Ny,",",Nz)
gw = prec(0.65)
Wangle =prec(asin(sqrt(0.22)))
gy=prec(gw*tan(Wangle))
gp2 = 0.99#parse(prec,ARGS[7])
vev = prec(1.0)
lambda = prec(0.129)
dx=0.25#parse(prec,ARGS[4])
dt=dx/10#(parse(prec,ARGS[5]))
mH = prec(2.0*sqrt(lambda)*vev)
nte = 30000#parse(int_prec,ARGS[3])
θ_w = Wangle
tol = prec(1e-4)
#Pheno damping term
γ_inp=0.0#parse(prec,ARGS[1])
#Temperature-going with 1/4\eta for now
T=0.25#parse(prec,ARGS[9])

export β_Y,β_W
β_Y = 0.0#parse(prec,ARGS[10])
β_W = 0.0#parse(prec,ARGS[11])

meff_sq = prec(8119.43/(174.0^2))#*0

export t_sat
t_sat = 2.0

nsnaps=100#parse(int_prec,ARGS[8])
# nsnaps=50
dsnaps = floor(int_prec,nte/nsnaps)

export nsnaps_fft,dsnaps_fft
nsnaps_fft=50
dsnaps_fft=floor(int_prec,nte/nsnaps_fft)

export no_snap_types
no_snap_types = 7

base_seed = 123456*1#parse(int_prec,ARGS[12]) ##Need to change up the seed just in case any 2 or more runs launch at the same time.

no_bubbles = 2#parse(int_prec,ARGS[2])
# bub_diam = int_prec(10)

println("Parameters")
println("γ: ",γ_inp)
println(string("dx: ",dx))
println(string("dt: ",dt))
println(string("nt: ",nte))
println(string("gp2: ",gp2))

bub_diam = floor(int_prec,30*0.1÷dx)

FPS = 2

##Rk-4-coefficients##
export rk4_coeffs
rk4_coeffs = [prec(1.0/6.),prec(1.0/3.),prec(1.0/3.),prec(1.0/6.)]

export no_cuda_threads
no_cuda_threads=64
# prinltn(string("# cuda threads: ", no_cuda_threads))

export spec_cut

spec_cut = [Nx÷4,Ny÷4,Nz÷4]

export Vol
Vol = dx^3

export out_3D
out_3D=true

export B_start_out_t
B_start_out_t = 100.0

export thrds,blks
thrds = (no_cuda_threads,1,1)
blks = (Nx÷thrds[1],Ny÷thrds[2],Nz÷thrds[3])

println("----------------CUDA-Topology-------------------")
println(string("#threads:",thrds," #blocks:",blks))
println("------------------------------------------------")


end