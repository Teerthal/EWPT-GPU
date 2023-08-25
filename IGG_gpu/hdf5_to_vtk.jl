using WriteVTK
using HDF5

#Coverting HDF5 file format to VTK files for viz

include("parameters.jl")
using .parameters

x=y=z=range(1,latx,step=1)

Threads.@threads for i in range(0,nt,step=dsnaps)
    c = h5open(string("raw_",i,".h5"), "r") do file
        read(file, "E_V")
    end
    
    vtk_grid(string("raw_",i),x,y,z) do vtk
        vtk["V"] = Float32.(c)
    end
    
    rm(string("raw_",i,".h5"))
end