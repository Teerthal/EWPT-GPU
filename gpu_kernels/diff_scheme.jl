#Absorbing boundary conditions: 6th order interior and 4th order numerical differentiation scheme on the boundaries
#Periodic boundary conditions: 6th order numerical differentiation scheme

module differentiations

export diff_abc
export dfdx,dfdy,dfdz
export d2fdx2,d2fdy2,d2fdz2

# dA_dx,dA_dy,dA_dz=0.,0.,0.

# @views function diff_abc(A,i,j,k,dA_dx,dA_dy,dA_dz,dx)
# @views function diff_abc(A,i,j,k,dA,dx)
#     if (i>3 && i<(size(A,1)-3) && j>3 && j<(size(A,3)-3) && k>3 && k<(size(A,3)-3))
#         @views dA[1] = (A[i+3,j,k]-9. *A[i+2,j,k]+45. *A[i+1,j,k]-
#         45. *A[i-1,j,k]+9. *A[i-2,j,k]-A[i-3,j,k])/(60*dx)
#         @views dA[2] = (A[i,j+3,k]-9. *A[i,j+2,k]+45. *A[i,j+1,k]-
#         45. *A[i,j-1,k]+9. *A[i,j-2,k]-A[i,j-3,k])/(60*dx)
#         @views dA[3] = (A[i,j,k+3]-9. *A[i,j,k+2]+45. *A[i,j,k+1]-
#         45. *A[i,j,k-1]+9. *A[i,j,k-2]-A[i,j,k-3])/(60*dx)

#     elseif i==3 || i==(size(A,1)-3)
#         @views dA[1] = (1.0*A[i-2,j,k]-8. *A[i-1,j,k]+
#         +8. *A[i+1,j,k]-1. *A[i+2,j,k])/(12.0*dx)
#     elseif i==2 || i==(size(A,1)-2)
#         @views dA[1] = (-A[i-1,j,k]+A[i+1,j,k])/(2.0*dx)
#     elseif i==1
#         @views dA[1] = (-A[i,j,k]+A[i+1,j,k])/(dx)
#     elseif i==(size(A,1))
#         @views dA[1] = (-A[i-1,j,k]+A[i,j,k])/(dx)

#     elseif j==3 || i==(size(A,2)-3)
#         @views dA[2] = (1.0*A[i-2,j,k]-8. *A[i-1,j,k]+
#         +8. *A[i+1,j,k]-1. *A[i+2,j,k])/(12.0*dx)
#     elseif j==2 || i==(size(A,2)-2)
#         @views dA[2] = (-A[i-1,j,k]+A[i+1,j,k])/(2.0*dx)
#     elseif j==1
#         @views dA[2] = (-A[i,j,k]+A[i+1,j,k])/(dx)
#     elseif j==(size(A,2))
#         @views dA[2] = (-A[i-1,j,k]+A[i,j,k])/(dx)

#     elseif k==3 || i==(size(A,3)-3)
#         @views dA[3] = (1.0*A[i-2,j,k]-8. *A[i-1,j,k]+
#         +8. *A[i+1,j,k]-1. *A[i+2,j,k])/(12.0*dx)
#     elseif k==2 || i==(size(A,3)-2)
#         @views dA[3] = (-A[i-1,j,k]+A[i+1,j,k])/(2.0*dx)
#     elseif k==1
#         @views dA[3] = (-A[i,j,k]+A[i+1,j,k])/(dx)
#     elseif k==(size(A,3))
#         @views dA[3] = (-A[i-1,j,k]+A[i,j,k])/(dx)
#     end
#     # dA = [dA_dx,dA_dy,dA_dz]
#     return dA
# end

########one sided derivatives at boundaries#########

# @views function dfdx(A,i,j,k,diff,dx)
#     if (i>3 && i<(size(A,1)-3))
#         @views diff = (A[i+3,j,k]-9. *A[i+2,j,k]+45. *A[i+1,j,k]-
#         45. *A[i-1,j,k]+9. *A[i-2,j,k]-A[i-3,j,k])/(60*dx)
#     elseif (i==3 || i==(size(A,1)-3))
#         @views diff = (1.0*A[i-2,j,k]-8. *A[i-1,j,k]+
#         +8. *A[i+1,j,k]-1. *A[i+2,j,k])/(12.0*dx)
#     elseif (i==2 || i==(size(A,1)-2))
#         @views diff = (-A[i-1,j,k]+A[i+1,j,k])/(2.0*dx)
#     elseif (i==1)
#         @views diff = (-A[i,j,k]+A[i+1,j,k])/(dx)
#     elseif (i==(size(A,1)))
#         @views diff = (-A[i-1,j,k]+A[i,j,k])/(dx)
#     end
#     return diff
# end

# @views function dfdy(A,i,j,k,diff,dx)
#     if (j>3 && j<(size(A,3)-3))
#         @views diff = (A[i,j+3,k]-9. *A[i,j+2,k]+45. *A[i,j+1,k]-
#         45. *A[i,j-1,k]+9. *A[i,j-2,k]-A[i,j-3,k])/(60*dx)
#     elseif (j==3 || j==(size(A,2)-3))
#         @views diff = (1.0*A[i,j-2,k]-8. *A[i,j-1,k]+
#         +8. *A[i,j+1,k]-1. *A[i,j+2,k])/(12.0*dx)
#     elseif (j==2 || j==(size(A,2)-2))
#         @views diff = (-A[i,j-1,k]+A[i,j+1,k])/(2.0*dx)
#     elseif (j==1)
#         @views diff = (-A[i,j,k]+A[i,j+1,k])/(dx)
#     elseif (j==(size(A,2)))
#         @views diff = (-A[i,j-1,k]+A[i,j,k])/(dx)
#     end
#     return diff
# end

# @views function dfdz(A,i,j,k,diff,dx)
#     if (k>3 && k<(size(A,3)-3))
#         @views diff = (A[i,j,k+3]-9. *A[i,j,k+2]+45. *A[i,j,k+1]-
#         45. *A[i,j,k-1]+9. *A[i,j,k-2]-A[i,j,k-3])/(60*dx)
#     elseif (k==3 || k==(size(A,2)-3))
#         @views diff = (1.0*A[i-2,j,k]-8. *A[i-1,j,k]+
#         +8. *A[i,j,k+1]-1. *A[i,j,k+2])/(12.0*dx)
#     elseif (k==2 || k==(size(A,2)-2))
#         @views diff = (-A[i,j,k-1]+A[i,j,k+1])/(2.0*dx)
#     elseif (k==1)
#         @views diff = (-A[i,j,k]+A[i,j,k+1])/(dx)
#     elseif (k==(size(A,2)))
#         @views diff = (-A[i,j,k-1]+A[i,j,k])/(dx)
#     end
#     return diff
# end

# @views function d2fdx2(A,i,j,k,diff,dx)
#     if (i>3 && i<(size(A,1)-3))
#         @views diff = (A[i+3,j,k]/90. -3. *A[i+2,j,k]/20. +3. *A[i+1,j,k]/2. -
#         49. *A[i,j,k]/18. +3. *A[i-1,j,k]/2. -3. *A[i-2,j,k]/20. +A[i-3,j,k]/90.)/(dx*dx)
#     elseif (i==3 || i==(size(A,1)-3))
#         @views diff = (-1.0*A[i-2,j,k]/12. +4. *A[i-1,j,k]/3. +
#         -5. *A[i,j,k]/2. +4. *A[i+1,j,k]/3. -1. *A[i+2,j,k]/12.)/(dx*dx)
#     elseif (i==2 || i==(size(A,1)-2))
#         @views diff = (A[i-1,j,k]-2. *A[i,j,k]+A[i+1,j,k])/(dx*dx)
#     elseif (i==1)
#         @views diff = (A[i,j,k]-2. *A[i+1,j,k]+A[i+2,j,k])/(dx*dx)
#     elseif (i==(size(A,1)))
#         @views diff = (A[i-2,j,k]-2. *A[i-1,j,k]+A[i,j,k])/(dx*dx)
#     end
#     return diff
# end

# @views function d2fdy2(A,i,j,k,diff,dx)
#     if (j>3 && j<(size(A,3)-3))
#         @views diff = (A[i,j+3,k]/90. -3. *A[i,j+2,k]/20. +3. *A[i,j+1,k]/2. -
#         49. *A[i,j,k]/18. +3. *A[i,j-1,k]/2. -3. *A[i,j-2,k]/20. +A[i,j-3,k]/90.)/(dx*dx)
#     elseif (j==3 || j==(size(A,2)-3))
#         @views diff = (-1.0*A[i,j-2,k]/12. +4. *A[i,j-1,k]/3. +
#         -5. *A[i,j,k]/2. +4. *A[i,j+1,k]/3. -1. *A[i,j+2,k]/12.)/(dx*dx)
#     elseif (j==2 || j==(size(A,2)-2))
#         @views diff = (A[i,j-1,k]-2. *A[i,j,k]+A[i,j+1,k])/(dx*dx)
#     elseif (j==1)
#         @views diff = (A[i,j,k]-2. *A[i,j+1,k]+A[i,j+2,k])/(dx*dx)
#     elseif (j==(size(A,2)))
#         @views diff = (A[i,j-2,k]-2. *A[i,j-1,k]+A[i,j,k])/(dx*dx)
#     end
#     return diff
# end

# @views function d2fdz2(A,i,j,k,diff,dx)
#     if (k>3 && k<(size(A,3)-3))
#         @views diff = (A[i,j,k+3]/90. -3. *A[i,j,k+2]/20. +3. *A[i,j,k+1]/2. -
#         49. *A[i,j,k]/18. +3. *A[i,j,k-1]/2. -3. *A[i,j,k-2]/20. +A[i,j,k-3]/90.)/(dx*dx)
#     elseif (k==3 || k==(size(A,2)-3))
#         @views diff = (-1.0*A[i,j,k-2]/12. +4. *A[i,j,k-1]/3. +
#         -5. *A[i,j,k]/2. +4. *A[i,j,k+1]/3. -1. *A[i,j,k+2]/12.)/(dx*dx)
#     elseif (k==2 || k==(size(A,2)-2))
#         @views diff = (A[i,j,k-1]-2. *A[i,j,k]+A[i,j,k+1])/(dx*dx)
#     elseif (k==1)
#         @views diff = (A[i,j,k]-2. *A[i,j,k+1]+A[i,j,k+2])/(dx*dx)
#     elseif (k==(size(A,2)))
#         @views diff = (A[i,j,k-2]-2. *A[i,j,k-1]+A[i,j,k])/(dx*dx)
#     end
#     return diff
# end

@views function dfdx(A,i,j,k,diff,dx)
    if (i>3 && i<(size(A,1)-2))
        @views diff = (A[i+3,j,k]-9. *A[i+2,j,k]+45. *A[i+1,j,k]-
        45. *A[i-1,j,k]+9. *A[i-2,j,k]-A[i-3,j,k])/(60*dx)
    elseif (i==(size(A,1)-2))
        @views diff = (A[1,j,k]-9. *A[i+2,j,k]+45. *A[i+1,j,k]-
        45. *A[i-1,j,k]+9. *A[i-2,j,k]-A[i-3,j,k])/(60*dx)
    elseif (i==(size(A,1)-1))
        @views diff = (A[2,j,k]-9. *A[1,j,k]+45. *A[i+1,j,k]-
        45. *A[i-1,j,k]+9. *A[i-2,j,k]-A[i-3,j,k])/(60*dx)
    elseif (i==(size(A,1)))
        @views diff = (A[3,j,k]-9. *A[2,j,k]+45. *A[1,j,k]-
        45. *A[i-1,j,k]+9. *A[i-2,j,k]-A[i-3,j,k])/(60*dx)
    elseif (i==1)
        @views diff = (A[i+3,j,k]-9. *A[i+2,j,k]+45. *A[i+1,j,k]-
        45. *A[end,j,k]+9. *A[end-1,j,k]-A[end-2,j,k])/(60*dx)
    elseif (i==2)
        @views diff = (A[i+3,j,k]-9. *A[i+2,j,k]+45. *A[i+1,j,k]-
        45. *A[i-1,j,k]+9. *A[end,j,k]-A[end-1,j,k])/(60*dx)
    elseif (i==3)
        @views diff = (A[i+3,j,k]-9. *A[i+2,j,k]+45. *A[i+1,j,k]-
        45. *A[i-1,j,k]+9. *A[i-2,j,k]-A[end,j,k])/(60*dx)
    end
    return diff
end

@views function dfdy(A,i,j,k,diff,dx)
    if (j>3 && j<(size(A,2)-2))
        @views diff = (A[i,j+3,k]-9. *A[i,j+2,k]+45. *A[i,j+1,k]-
        45. *A[i,j-1,k]+9. *A[i,j-2,k]-A[i,j-3,k])/(60*dx)
    elseif (j==(size(A,2)-2))
        @views diff = (A[i,1,k]-9. *A[i,j+2,k]+45. *A[i,j+1,k]-
        45. *A[i,j-1,k]+9. *A[i,j-2,k]-A[i,j-3,k])/(60*dx)
    elseif (j==(size(A,2)-1))
        @views diff = (A[i,2,k]-9. *A[i,1,k]+45. *A[i,j+1,k]-
        45. *A[i,j-1,k]+9. *A[i,j-2,k]-A[i,j-3,k])/(60*dx)
    elseif (j==(size(A,2)))
        @views diff = (A[i,3,k]-9. *A[i,2,k]+45. *A[i,1,k]-
        45. *A[i,j-1,k]+9. *A[i,j-2,k]-A[i,j-3,k])/(60*dx)
    elseif (j==1)
        @views diff = (A[i,j+3,k]-9. *A[i,j+2,k]+45. *A[i,j+1,k]-
        45. *A[i,end,k]+9. *A[i,end-1,k]-A[i,end-2,k])/(60*dx)
    elseif (j==2)
        @views diff = (A[i,j+3,k]-9. *A[i,j+2,k]+45. *A[i,j+1,k]-
        45. *A[i,j-1,k]+9. *A[i,end,k]-A[i,end-1,k])/(60*dx)
    elseif (j==3)
        @views diff = (A[i,j+3,k]-9. *A[i,j+2,k]+45. *A[i,j+1,k]-
        45. *A[i,j-1,k]+9. *A[i,j-2,k]-A[i,end,k])/(60*dx)
    end
    return diff
end

@views function dfdz(A,i,j,k,diff,dx)
    if (k>3 && k<(size(A,3)-2))
        @views diff = (A[i,j,k+3]-9. *A[i,j,k+2]+45. *A[i,j,k+1]-
        45. *A[i,j,k-1]+9. *A[i,j,k-2]-A[i,j,k-3])/(60*dx)
    elseif (k==(size(A,3)-2))
        @views diff = (A[i,j,1]-9. *A[i,j,k+2]+45. *A[i,j,k+1]-
        45. *A[i,j,k-1]+9. *A[i,j,k-2]-A[i,j,k-3])/(60*dx)
    elseif (k==(size(A,3)-1))
        @views diff = (A[i,j,2]-9. *A[i,j,1]+45. *A[i,j,k+1]-
        45. *A[i,j,k-1]+9. *A[i,j,k-2]-A[i,j,k-3])/(60*dx)
    elseif (k==(size(A,3)))
        @views diff = (A[i,j,3]-9. *A[i,j,2]+45. *A[i,j,1]-
        45. *A[i,j,k-1]+9. *A[i,j,k-2]-A[i,j,k-3])/(60*dx)
    elseif (k==1)
        @views diff = (A[i,j,k+3]-9. *A[i,j,k+2]+45. *A[i,j,k+1]-
        45. *A[i,j,end]+9. *A[i,j,end-1]-A[i,j,end-2])/(60*dx)
    elseif (k==2)
        @views diff = (A[i,j,k+3]-9. *A[i,j,k+2]+45. *A[i,j,k+1]-
        45. *A[i,j,k-1]+9. *A[i,j,end]-A[i,j,end-1])/(60*dx)
    elseif (k==3)
        @views diff = (A[i,j,k+3]-9. *A[i,j,k+2]+45. *A[i,j,k+1]-
        45. *A[i,j,k-1]+9. *A[i,j,k-2]-A[i,j,end])/(60*dx)
    end
    return diff
end

@views function d2fdx2(A,i,j,k,diff,dx)
    if (i>3 && i<(size(A,1)-2))
        @views diff = (A[i+3,j,k]/90. -3. *A[i+2,j,k]/20. +3. *A[i+1,j,k]/2. -
        49. *A[i,j,k]/18. +3. *A[i-1,j,k]/2. -3. *A[i-2,j,k]/20. +A[i-3,j,k]/90.)/(dx*dx)
    elseif (i==(size(A,1)-2))
        @views diff = (A[1,j,k]/90. -3. *A[i+2,j,k]/20. +3. *A[i+1,j,k]/2. -
        49. *A[i,j,k]/18. +3. *A[i-1,j,k]/2. -3. *A[i-2,j,k]/20. +A[i-3,j,k]/90.)/(dx*dx)
    elseif (i==(size(A,1)-1))
        @views diff = (A[2,j,k]/90. -3. *A[1,j,k]/20. +3. *A[i+1,j,k]/2. -
        49. *A[i,j,k]/18. +3. *A[i-1,j,k]/2. -3. *A[i-2,j,k]/20. +A[i-3,j,k]/90.)/(dx*dx)
    elseif (i==(size(A,1)))
        @views diff = (A[3,j,k]/90. -3. *A[2,j,k]/20. +3. *A[1,j,k]/2. -
        49. *A[i,j,k]/18. +3. *A[i-1,j,k]/2. -3. *A[i-2,j,k]/20. +A[i-3,j,k]/90.)/(dx*dx)
    elseif (i==1)
        @views diff = (A[i+3,j,k]/90. -3. *A[i+2,j,k]/20. +3. *A[i+1,j,k]/2. -
        49. *A[i,j,k]/18. +3. *A[end,j,k]/2. -3. *A[end-1,j,k]/20. +A[end-2,j,k]/90.)/(dx*dx)
    elseif (i==2)
        @views diff = (A[i+3,j,k]/90. -3. *A[i+2,j,k]/20. +3. *A[i+1,j,k]/2. -
        49. *A[i,j,k]/18. +3. *A[i-1,j,k]/2. -3. *A[end,j,k]/20. +A[end-1,j,k]/90.)/(dx*dx)
    elseif (i==3)
        @views diff = (A[i+3,j,k]/90. -3. *A[i+2,j,k]/20. +3. *A[i+1,j,k]/2. -
        49. *A[i,j,k]/18. +3. *A[i-1,j,k]/2. -3. *A[i-2,j,k]/20. +A[end,j,k]/90.)/(dx*dx)
    end
    return diff
end

@views function d2fdy2(A,i,j,k,diff,dx)
    if (j>3 && j<(size(A,2)-2))
        @views diff = (A[i,j+3,k]/90. -3. *A[i,j+2,k]/20. +3. *A[i,j+1,k]/2. -
        49. *A[i,j,k]/18. +3. *A[i,j-1,k]/2. -3. *A[i,j-2,k]/20. +A[i,j-3,k]/90.)/(dx*dx)
    elseif (j==(size(A,2)-2))
        @views diff = (A[i,1,k]/90. -3. *A[i,j+2,k]/20. +3. *A[i,j+1,k]/2. -
        49. *A[i,j,k]/18. +3. *A[i,j-1,k]/2. -3. *A[i,j-2,k]/20. +A[i,j-3,k]/90.)/(dx*dx)
    elseif (j==(size(A,2)-1))
        @views diff = (A[i,2,k]/90. -3. *A[i,1,k]/20. +3. *A[i,j+1,k]/2. -
        49. *A[i,j,k]/18. +3. *A[i,j-1,k]/2. -3. *A[i,j-2,k]/20. +A[i,j-3,k]/90.)/(dx*dx)
    elseif (j==(size(A,2)))
        @views diff = (A[i,3,k]/90. -3. *A[i,2,k]/20. +3. *A[i,1,k]/2. -
        49. *A[i,j,k]/18. +3. *A[i,j-1,k]/2. -3. *A[i,j-2,k]/20. +A[i,j-3,k]/90.)/(dx*dx)
    elseif (j==1)
        @views diff = (A[i,j+3,k]/90. -3. *A[i,j+2,k]/20. +3. *A[i,j+1,k]/2. -
        49. *A[i,j,k]/18. +3. *A[i,end,k]/2. -3. *A[i,end-1,k]/20. +A[i,end-2,k]/90.)/(dx*dx)
    elseif (j==2)
        @views diff = (A[i,j+3,k]/90. -3. *A[i,j+2,k]/20. +3. *A[i,j+1,k]/2. -
        49. *A[i,j,k]/18. +3. *A[i,j-1,k]/2. -3. *A[i,end,k]/20. +A[i,end-1,k]/90.)/(dx*dx)
    elseif (j==3)
        @views diff = (A[i,j+3,k]/90. -3. *A[i,j+2,k]/20. +3. *A[i,j+1,k]/2. -
        49. *A[i,j,k]/18. +3. *A[i,j-1,k]/2. -3. *A[i,j-2,k]/20. +A[i,end,k]/90.)/(dx*dx)
    end
    return diff
end

@views function d2fdz2(A,i,j,k,diff,dx)
    if (k>3 && k<(size(A,3)-2))
        @views diff = (A[i,j,k+3]/90. -3. *A[i,j,k+2]/20. +3. *A[i,j,k+1]/2. -
        49. *A[i,j,k]/18. +3. *A[i,j,k-1]/2. -3. *A[i,j,k-2]/20. +A[i,j,k-3]/90.)/(dx*dx)
    elseif (k==(size(A,3)-2))
        @views diff = (A[i,j,1]/90. -3. *A[i,j,k+2]/20. +3. *A[i,j,k+1]/2. -
        49. *A[i,j,k]/18. +3. *A[i,j,k-1]/2. -3. *A[i,j,k-2]/20. +A[i,j,k-3]/90.)/(dx*dx)
    elseif (k==(size(A,3)-1))
        @views diff = (A[i,j,2]/90. -3. *A[i,j,1]/20. +3. *A[i,j,k+1]/2. -
        49. *A[i,j,k]/18. +3. *A[i,j,k-1]/2. -3. *A[i,j,k-2]/20. +A[i,j,k-3]/90.)/(dx*dx)
    elseif (k==(size(A,3)))
        @views diff = (A[i,j,3]/90. -3. *A[i,j,2]/20. +3. *A[i,j,1]/2. -
        49. *A[i,j,k]/18. +3. *A[i,j,k-1]/2. -3. *A[i,j,k-2]/20. +A[i,j,k-3]/90.)/(dx*dx)
    elseif (k==1)
        @views diff = (A[i,j,k+3]/90. -3. *A[i,j,k+2]/20. +3. *A[i,j,k+1]/2. -
        49. *A[i,j,k]/18. +3. *A[i,j,end]/2. -3. *A[i,j,end-1]/20. +A[i,j,end-2]/90.)/(dx*dx)
    elseif (k==2)
        @views diff = (A[i,j,k+3]/90. -3. *A[i,j,k+2]/20. +3. *A[i,j,k+1]/2. -
        49. *A[i,j,k]/18. +3. *A[i,j,k-1]/2. -3. *A[i,j,end]/20. +A[i,j,end-1]/90.)/(dx*dx)
    elseif (k==3)
        @views diff = (A[i,j,k+3]/90. -3. *A[i,j,k+2]/20. +3. *A[i,j,k+1]/2. -
        49. *A[i,j,k]/18. +3. *A[i,j,k-1]/2. -3. *A[i,j,k-2]/20. +A[i,j,end]/90.)/(dx*dx)
    end
    return diff
end

# @views function d2fdx2(A,i,j,k,diff,dx)
#     if (i>3 && i<(size(A,1)-3))
#         @views diff = (A[i+3,j,k]/90. -3. *A[i+2,j,k]/20. +3. *A[i+1,j,k]/2. -
#         49. *A[i,j,k]/18. +3. *A[i-1,j,k]/2. -3. *A[i-2,j,k]/20. +A[i-3,j,k]/90.)/(dx*dx)
#     elseif (i==3 || i==(size(A,1)-3))
#         @views diff = (-1.0*A[i-2,j,k]/12. +4. *A[i-1,j,k]/3. +
#         -5. *A[i,j,k]/2. +4. *A[i+1,j,k]/3. -1. *A[i+2,j,k]/12.)/(dx*dx)
#     elseif (i==2 || i==(size(A,1)-2))
#         @views diff = (A[i-1,j,k]-2. *A[i,j,k]+A[i+1,j,k])/(dx*dx)
#     elseif (i==1)
#         @views diff = (A[i,j,k]-2. *A[i+1,j,k]+A[i+2,j,k])/(dx*dx)
#     elseif (i==(size(A,1)))
#         @views diff = (A[i-2,j,k]-2. *A[i-1,j,k]+A[i,j,k])/(dx*dx)
#     end
#     return diff
# end

# @views function d2fdy2(A,i,j,k,diff,dx)
#     if (j>3 && j<(size(A,3)-3))
#         @views diff = (A[i,j+3,k]/90. -3. *A[i,j+2,k]/20. +3. *A[i,j+1,k]/2. -
#         49. *A[i,j,k]/18. +3. *A[i,j-1,k]/2. -3. *A[i,j-2,k]/20. +A[i,j-3,k]/90.)/(dx*dx)
#     elseif (j==3 || j==(size(A,2)-3))
#         @views diff = (-1.0*A[i,j-2,k]/12. +4. *A[i,j-1,k]/3. +
#         -5. *A[i,j,k]/2. +4. *A[i,j+1,k]/3. -1. *A[i,j+2,k]/12.)/(dx*dx)
#     elseif (j==2 || j==(size(A,2)-2))
#         @views diff = (A[i,j-1,k]-2. *A[i,j,k]+A[i,j+1,k])/(dx*dx)
#     elseif (j==1)
#         @views diff = (A[i,j,k]-2. *A[i,j+1,k]+A[i,j+2,k])/(dx*dx)
#     elseif (j==(size(A,2)))
#         @views diff = (A[i,j-2,k]-2. *A[i,j-1,k]+A[i,j,k])/(dx*dx)
#     end
#     return diff
# end

# @views function d2fdz2(A,i,j,k,diff,dx)
#     if (k>3 && k<(size(A,3)-3))
#         @views diff = (A[i,j,k+3]/90. -3. *A[i,j,k+2]/20. +3. *A[i,j,k+1]/2. -
#         49. *A[i,j,k]/18. +3. *A[i,j,k-1]/2. -3. *A[i,j,k-2]/20. +A[i,j,k-3]/90.)/(dx*dx)
#     elseif (k==3 || k==(size(A,2)-3))
#         @views diff = (-1.0*A[i,j,k-2]/12. +4. *A[i,j,k-1]/3. +
#         -5. *A[i,j,k]/2. +4. *A[i,j,k+1]/3. -1. *A[i,j,k+2]/12.)/(dx*dx)
#     elseif (k==2 || k==(size(A,2)-2))
#         @views diff = (A[i,j,k-1]-2. *A[i,j,k]+A[i,j,k+1])/(dx*dx)
#     elseif (k==1)
#         @views diff = (A[i,j,k]-2. *A[i,j,k+1]+A[i,j,k+2])/(dx*dx)
#     elseif (k==(size(A,2)))
#         @views diff = (A[i,j,k-2]-2. *A[i,j,k-1]+A[i,j,k])/(dx*dx)
#     end
#     return diff
# end

#Module end#
end