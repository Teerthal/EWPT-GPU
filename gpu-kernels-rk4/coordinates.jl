module coords

export cartesian_symm

function cartesian_symm(i,j,k,x,y,z,Nx,Ny,Nz)
    x = i-Nx÷2
    y = j-Nx÷2
    z = k-Nx÷2
    return x,y,z
end

#Module end#
end