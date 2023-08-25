module parameters
export nf,gw,gy,lambda,vev,Wangle
export latx,laty,latz,nt,dx,gp2,relaxparam
export nsnaps,outerlayer
export xm,ym,zm,xa,ya,za
export twist,ms,mv,szoff
export senlen
export ang_vel,vel,nte,dt
export interval
export mH
export bub_diam
export no_bubbles

nf=25
gw=0.65
lambda=0.129
vev=1.0
Wangle =asin(sqrt(0.22))
gy=gw*tan(Wangle)
mH = 2.0*sqrt(lambda)*vev
bub_diam = 30
no_bubbles = 2

latx=120*2#372
laty=latx
latz=latx
nt=120000
dx=0.1
dt=dx/200.0
gp2=0.95
relaxparam=1.0

println("Parameters")
println(string("dx: ",dx))
println(string("dt: ",dt))
println(string("nt: ",nt))
println(string("gp2: ",gp2))

#Output step parameters
export nsnaps,dsnaps
nsnaps=200
dsnaps = floor(Int,nt/nsnaps)

#Viz parameters
export FPS
FPS=10

outerlayer=2

##Following parameters for single dumbbell simulations.
##Not yet implemented
itw=6
izm=5

twist=(itw)*π/6.

zm=(izm+0.5)*dx
xm=floor(latx/2)+0.5*dx
ym=floor(laty/2)+0.5*dx

ms=sqrt(2*lambda)*vev
mv=gw*vev

xa=xm
ya=ym
za=-zm

szoff = 0.0
senlen= 0.0

##Boost parameters
ang_vel = 0.0
vel=0.0

interval = 10

end