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
    return prec(diff)
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
    return prec(diff)
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
    return prec(diff)
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
    return prec(diff)
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
    return prec(diff)
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
    return prec(diff)
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

#############8th ORDER################################################


module differentiations_8th_order

export diff_abc
export dfdx,dfdy,dfdz
export d2fdx2,d2fdy2,d2fdz2

@views function dfdx(A,i,j,k,diff,dx)
    if (i>4 && i<(size(A,1)-3))
        @views diff = (3*A[i-4,j,k]-32*A[i-3,j,k]+168*A[i-2,j,k]-
        672*A[i-1,j,k]+672*A[i+1,j,k]-168*A[i+2,j,k]+
        32*A[i+3,j,k]-3*A[i+4,j,k])/(840*dx)
    elseif (i==(size(A,1)-3))
        @views diff = (3*A[i-4,j,k]-32*A[i-3,j,k]+168*A[i-2,j,k]-
        672*A[i-1,j,k]+672*A[i+1,j,k]-168*A[i+2,j,k]+
        32*A[i+3,j,k]-3*A[1,j,k])/(840*dx)
    elseif (i==(size(A,1)-2))
        @views diff = (3*A[i-4,j,k]-32*A[i-3,j,k]+168*A[i-2,j,k]-
        672*A[i-1,j,k]+672*A[i+1,j,k]-168*A[i+2,j,k]+
        32*A[1,j,k]-3*A[2,j,k])/(840*dx)
    elseif (i==(size(A,1)-1))
        @views diff = (3*A[i-4,j,k]-32*A[i-3,j,k]+168*A[i-2,j,k]-
        672*A[i-1,j,k]+672*A[i+1,j,k]-168*A[1,j,k]+
        32*A[2,j,k]-3*A[3,j,k])/(840*dx)
    elseif (i==(size(A,1)))
        @views diff = (3*A[i-4,j,k]-32*A[i-3,j,k]+168*A[i-2,j,k]-
        672*A[i-1,j,k]+672*A[1,j,k]-168*A[2,j,k]+
        32*A[3,j,k]-3*A[4,j,k])/(840*dx)
    elseif (i==1)
        @views diff = (3*A[end-3,j,k]-32*A[end-2,j,k]+168*A[end-1,j,k]-
        672*A[end,j,k]+672*A[i+1,j,k]-168*A[i+2,j,k]+
        32*A[i+3,j,k]-3*A[i+4,j,k])/(840*dx)
    elseif (i==2)
        @views diff = (3*A[end-2,j,k]-32*A[end-1,j,k]+168*A[end,j,k]-
        672*A[i-1,j,k]+672*A[i+1,j,k]-168*A[i+2,j,k]+
        32*A[i+3,j,k]-3*A[i+4,j,k])/(840*dx)
    elseif (i==3)
        @views diff = (3*A[end-1,j,k]-32*A[end,j,k]+168*A[i-2,j,k]-
        672*A[i-1,j,k]+672*A[i+1,j,k]-168*A[i+2,j,k]+
        32*A[i+3,j,k]-3*A[i+4,j,k])/(840*dx)
    elseif (i==4)
        @views diff = (3*A[end,j,k]-32*A[i-3,j,k]+168*A[i-2,j,k]-
        672*A[i-1,j,k]+672*A[i+1,j,k]-168*A[i+2,j,k]+
        32*A[i+3,j,k]-3*A[i+4,j,k])/(840*dx)
    end
    return prec(diff)
end

@views function dfdy(A,i,j,k,diff,dx)
    if (j>4 && j<(size(A,2)-3))
        @views diff = (3*A[i,j-4,k]-32*A[i,j-3,k]+168*A[i,j-2,k]-
        672*A[i,j-1,k]+672*A[i,j+1,k]-168*A[i,j+2,k]+
        32*A[i,j+3,k]-3*A[i,j+4,k])/(840*dx)
    elseif (j==(size(A,2)-3))
        @views diff = (3*A[i,j-4,k]-32*A[i,j-3,k]+168*A[i,j-2,k]-
        672*A[i,j-1,k]+672*A[i,j+1,k]-168*A[i,j+2,k]+
        32*A[i,j+3,k]-3*A[i,1,k])/(840*dx)
    elseif (j==(size(A,2)-2))
        @views diff = (3*A[i,j-4,k]-32*A[i,j-3,k]+168*A[i,j-2,k]-
        672*A[i,j-1,k]+672*A[i,j+1,k]-168*A[i,j+2,k]+
        32*A[i,1,k]-3*A[i,2,k])/(840*dx)
    elseif (j==(size(A,2)-1))
        @views diff = (3*A[i,j-4,k]-32*A[i,j-3,k]+168*A[i,j-2,k]-
        672*A[i,j-1,k]+672*A[i,j+1,k]-168*A[i,1,k]+
        32*A[i,2,k]-3*A[i,3,k])/(840*dx)
    elseif (j==(size(A,2)))
        @views diff = (3*A[i,j-4,k]-32*A[i,j-3,k]+168*A[i,j-2,k]-
        672*A[i,j-1,k]+672*A[i,1,k]-168*A[i,2,k]+
        32*A[i,3,k]-3*A[i,4,k])/(840*dx)
    elseif (j==1)
        @views diff = (3*A[i,end-3,k]-32*A[i,end-2,k]+168*A[i,end-1,k]-
        672*A[i,end,k]+672*A[i,j+1,k]-168*A[i,j+2,k]+
        32*A[i,j+3,k]-3*A[i,j+4,k])/(840*dx)
    elseif (j==2)
        @views diff = (3*A[i,end-2,k]-32*A[i,end-1,k]+168*A[i,end,k]-
        672*A[i,j-1,k]+672*A[i,j+1,k]-168*A[i,j+2,k]+
        32*A[i,j+3,k]-3*A[i,j+4,k])/(840*dx)
    elseif (j==3)
        @views diff = (3*A[i,end-1,k]-32*A[i,end,k]+168*A[i,j-2,k]-
        672*A[i,j-1,k]+672*A[i,j+1,k]-168*A[i,j+2,k]+
        32*A[i,j+3,k]-3*A[i,j+4,k])/(840*dx)
    elseif (j==4)
        @views diff = (3*A[i,end,k]-32*A[i,j-3,k]+168*A[i,j-2,k]-
        672*A[i,j-1,k]+672*A[i,j+1,k]-168*A[i,j+2,k]+
        32*A[i,j+3,k]-3*A[i,j+4,k])/(840*dx)
    end
    return prec(diff)
end

@views function dfdz(A,i,j,k,diff,dx)
    if (k>4 && k<(size(A,3)-3))
        @views diff = (3*A[i,j,k-4]-32*A[i,j,k-3]+168*A[i,j,k-2]-
        672*A[i,j,k-1]+672*A[i,j,k+1]-168*A[i,j,k+2]+
        32*A[i,j,k+3]-3*A[i,j,k+4])/(840*dx)
    elseif (k==(size(A,3)-3))
        @views diff = (3*A[i,j,k-4]-32*A[i,j,k-3]+168*A[i,j,k-2]-
        672*A[i,j,k-1]+672*A[i,j,k+1]-168*A[i,j,k+2]+
        32*A[i,j,k+3]-3*A[i,j,1])/(840*dx)
    elseif (k==(size(A,3)-2))
        @views diff = (3*A[i,j,k-4]-32*A[i,j,k-3]+168*A[i,j,k-2]-
        672*A[i,j,k-1]+672*A[i,j,k+1]-168*A[i,j,k+2]+
        32*A[i,j,1]-3*A[i,j,2])/(840*dx)
    elseif (k==(size(A,3)-1))
        @views diff = (3*A[i,j,k-4]-32*A[i,j,k-3]+168*A[i,j,k-2]-
        672*A[i,j,k-1]+672*A[i,j,k+1]-168*A[i,j,1]+
        32*A[i,j,2]-3*A[i,j,3])/(840*dx)
    elseif (k==(size(A,3)))
        @views diff = (3*A[i,j,k-4]-32*A[i,j,k-3]+168*A[i,j,k-2]-
        672*A[i,j,k-1]+672*A[i,j,1]-168*A[i,j,2]+
        32*A[i,j,3]-3*A[i,j,4])/(840*dx)
    elseif (k==1)
        @views diff = (3*A[i,j,end-3]-32*A[i,j,end-2]+168*A[i,j,end-1]-
        672*A[i,j,end]+672*A[i,j,k+1]-168*A[i,j,k+2]+
        32*A[i,j,k+3]-3*A[i,j,k+4])/(840*dx)
    elseif (k==2)
        @views diff = (3*A[i,j,end-2]-32*A[i,j,end-1]+168*A[i,j,end]-
        672*A[i,j,k-1]+672*A[i,j,k+1]-168*A[i,j,k+2]+
        32*A[i,j,k+3]-3*A[i,j,k+4])/(840*dx)
    elseif (k==3)
        @views diff = (3*A[i,j,end-1]-32*A[i,j,end]+168*A[i,j,k-2]-
        672*A[i,j,k-1]+672*A[i,j,k+1]-168*A[i,j,k+2]+
        32*A[i,j,k+3]-3*A[i,j,k+4])/(840*dx)
    elseif (k==4)
        @views diff = (3*A[i,j,end]-32*A[i,j,k-3]+168*A[i,j,k-2]-
        672*A[i,j,k-1]+672*A[i,j,k+1]-168*A[i,j,k+2]+
        32*A[i,j,k+3]-3*A[i,j,k+4])/(840*dx)
    end
    return prec(diff)
end

@views function d2fdx2(A,i,j,k,diff,dx)
    if (i>4 && i<(size(A,1)-3))
        @views diff = (-9*A[i-4,j,k]+128*A[i-3,j,k]-1008*A[i-2,j,k]+
        8064*A[i-1,j,k]-14350*A[i,j,k]+8064*A[i+1,j,k]-
        1008*A[i+2,j,k]+128*A[i+3,j,k]-9*A[i+4,j,k])/(5040*dx*dx)
    elseif (i==(size(A,1)-3))
        @views diff = (-9*A[i-4,j,k]+128*A[i-3,j,k]-1008*A[i-2,j,k]+
        8064*A[i-1,j,k]-14350*A[i,j,k]+8064*A[i+1,j,k]-
        1008*A[i+2,j,k]+128*A[i+3,j,k]-9*A[1,j,k])/(5040*dx*dx)
    elseif (i==(size(A,1)-2))
        @views diff = (-9*A[i-4,j,k]+128*A[i-3,j,k]-1008*A[i-2,j,k]+
        8064*A[i-1,j,k]-14350*A[i,j,k]+8064*A[i+1,j,k]-
        1008*A[i+2,j,k]+128*A[1,j,k]-9*A[2,j,k])/(5040*dx*dx)
    elseif (i==(size(A,1)-1))
        @views diff = (-9*A[i-4,j,k]+128*A[i-3,j,k]-1008*A[i-2,j,k]+
        8064*A[i-1,j,k]-14350*A[i,j,k]+8064*A[i+1,j,k]-
        1008*A[1,j,k]+128*A[2,j,k]-9*A[3,j,k])/(5040*dx*dx)
    elseif (i==(size(A,1)))
        @views diff = (-9*A[i-4,j,k]+128*A[i-3,j,k]-1008*A[i-2,j,k]+
        8064*A[i-1,j,k]-14350*A[i,j,k]+8064*A[1,j,k]-
        1008*A[2,j,k]+128*A[3,j,k]-9*A[4,j,k])/(5040*dx*dx)
    elseif (i==1)
        @views diff = (-9*A[end-3,j,k]+128*A[end-2,j,k]-1008*A[end-1,j,k]+
        8064*A[end,j,k]-14350*A[i,j,k]+8064*A[i+1,j,k]-
        1008*A[i+2,j,k]+128*A[i+3,j,k]-9*A[i+4,j,k])/(5040*dx*dx)
    elseif (i==2)
        @views diff = (-9*A[end-2,j,k]+128*A[end-1,j,k]-1008*A[end,j,k]+
        8064*A[i-1,j,k]-14350*A[i,j,k]+8064*A[i+1,j,k]-
        1008*A[i+2,j,k]+128*A[i+3,j,k]-9*A[i+4,j,k])/(5040*dx*dx)
    elseif (i==3)
        @views diff = (-9*A[end-1,j,k]+128*A[end,j,k]-1008*A[i-2,j,k]+
        8064*A[i-1,j,k]-14350*A[i,j,k]+8064*A[i+1,j,k]-
        1008*A[i+2,j,k]+128*A[i+3,j,k]-9*A[i+4,j,k])/(5040*dx*dx)
    elseif (i==4)
        @views diff = (-9*A[end,j,k]+128*A[i-3,j,k]-1008*A[i-2,j,k]+
        8064*A[i-1,j,k]-14350*A[i,j,k]+8064*A[i+1,j,k]-
        1008*A[i+2,j,k]+128*A[i+3,j,k]-9*A[i+4,j,k])/(5040*dx*dx)
    end
    return prec(diff)
end

@views function d2fdy2(A,i,j,k,diff,dx)
    if (j>4 && j<(size(A,2)-3))
        @views diff = (-9*A[i,j-4,k]+128*A[i,j-3,k]-1008*A[i,j-2,k]+
        8064*A[i,j-1,k]-14350*A[i,j,k]+8064*A[i,j+1,k]-
        1008*A[i,j+2,k]+128*A[i,j+3,k]-9*A[i,j+4,k])/(5040*dx*dx)
    elseif (j==(size(A,2)-3))
        @views diff = (-9*A[i,j-4,k]+128*A[i,j-3,k]-1008*A[i,j-2,k]+
        8064*A[i,j-1,k]-14350*A[i,j,k]+8064*A[i,j+1,k]-
        1008*A[i,j+2,k]+128*A[i,j+3,k]-9*A[i,1,k])/(5040*dx*dx)
    elseif (j==(size(A,2)-2))
        @views diff = (-9*A[i,j-4,k]+128*A[i,j-3,k]-1008*A[i,j-2,k]+
        8064*A[i,j-1,k]-14350*A[i,j,k]+8064*A[i,j+1,k]-
        1008*A[i,j+2,k]+128*A[i,1,k]-9*A[i,2,k])/(5040*dx*dx)
    elseif (j==(size(A,2)-1))
        @views diff = (-9*A[i,j-4,k]+128*A[i,j-3,k]-1008*A[i,j-2,k]+
        8064*A[i,j-1,k]-14350*A[i,j,k]+8064*A[i,j+1,k]-
        1008*A[i,1,k]+128*A[i,2,k]-9*A[i,3,k])/(5040*dx*dx)
    elseif (j==(size(A,2)))
        @views diff = (-9*A[i,j-4,k]+128*A[i,j-3,k]-1008*A[i,j-2,k]+
        8064*A[i,j-1,k]-14350*A[i,j,k]+8064*A[i,1,k]-
        1008*A[i,2,k]+128*A[i,3,k]-9*A[i,4,k])/(5040*dx*dx)
    elseif (j==1)
        @views diff = (-9*A[i,end-3,k]+128*A[i,end-2,k]-1008*A[i,end-1,k]+
        8064*A[i,end,k]-14350*A[i,j,k]+8064*A[i,j+1,k]-
        1008*A[i,j+2,k]+128*A[i,j+3,k]-9*A[i,j+4,k])/(5040*dx*dx)
    elseif (j==2)
        @views diff = (-9*A[i,end-2,k]+128*A[i,end-1,k]-1008*A[i,end,k]+
        8064*A[i,j-1,k]-14350*A[i,j,k]+8064*A[i,j+1,k]-
        1008*A[i,j+2,k]+128*A[i,j+3,k]-9*A[i,j+4,k])/(5040*dx*dx)
    elseif (j==3)
        @views diff = (-9*A[i,end-1,k]+128*A[i,end,k]-1008*A[i,j-2,k]+
        8064*A[i,j-1,k]-14350*A[i,j,k]+8064*A[i,j+1,k]-
        1008*A[i,j+2,k]+128*A[i,j+3,k]-9*A[i,j+4,k])/(5040*dx*dx)
    elseif (j==4)
        @views diff = (-9*A[i,end,k]+128*A[i,j-3,k]-1008*A[i,j-2,k]+
        8064*A[i,j-1,k]-14350*A[i,j,k]+8064*A[i,j+1,k]-
        1008*A[i,j+2,k]+128*A[i,j+3,k]-9*A[i,j+4,k])/(5040*dx*dx)
    end
    return prec(diff)
end

@views function d2fdz2(A,i,j,k,diff,dx)
    if (k>4 && k<(size(A,3)-3))
        @views diff = (-9*A[i,j,k-4]+128*A[i,j,k-3]-1008*A[i,j,k-2]+
        8064*A[i,j,k-1]-14350*A[i,j,k]+8064*A[i,j,k+1]-
        1008*A[i,j,k+2]+128*A[i,j,k+3]-9*A[i,j,k+4])/(5040*dx*dx)
    elseif (k==(size(A,3)-3))
        @views diff = (-9*A[i,j,k-4]+128*A[i,j,k-3]-1008*A[i,j,k-2]+
        8064*A[i,j,k-1]-14350*A[i,j,k]+8064*A[i,j,k+1]-
        1008*A[i,j,k+2]+128*A[i,j,k+3]-9*A[i,j,1])/(5040*dx*dx)
    elseif (k==(size(A,3)-2))
        @views diff = (-9*A[i,j,k-4]+128*A[i,j,k-3]-1008*A[i,j,k-2]+
        8064*A[i,j,k-1]-14350*A[i,j,k]+8064*A[i,j,k+1]-
        1008*A[i,j,k+2]+128*A[i,j,1]-9*A[i,j,2])/(5040*dx*dx)
    elseif (k==(size(A,3)-1))
        @views diff = (-9*A[i,j,k-4]+128*A[i,j,k-3]-1008*A[i,j,k-2]+
        8064*A[i,j,k-1]-14350*A[i,j,k]+8064*A[i,j,k+1]-
        1008*A[i,j,1]+128*A[i,j,2]-9*A[i,j,3])/(5040*dx*dx)
    elseif (k==(size(A,3)))
        @views diff = (-9*A[i,j,k-4]+128*A[i,j,k-3]-1008*A[i,j,k-2]+
        8064*A[i,j,k-1]-14350*A[i,j,k]+8064*A[i,j,1]-
        1008*A[i,j,2]+128*A[i,j,3]-9*A[i,j,4])/(5040*dx*dx)
    elseif (k==1)
        @views diff = (-9*A[i,j,end-3]+128*A[i,j,end-2]-1008*A[i,j,end-1]+
        8064*A[i,j,end]-14350*A[i,j,k]+8064*A[i,j,k+1]-
        1008*A[i,j,k+2]+128*A[i,j,k+3]-9*A[i,j,k+4])/(5040*dx*dx)
    elseif (k==2)
        @views diff = (-9*A[i,j,end-2]+128*A[i,j,end-1]-1008*A[i,j,end]+
        8064*A[i,j,k-1]-14350*A[i,j,k]+8064*A[i,j,k+1]-
        1008*A[i,j,k+2]+128*A[i,j,k+3]-9*A[i,j,k+4])/(5040*dx*dx)
    elseif (k==3)
        @views diff = (-9*A[i,j,end-1]+128*A[i,j,end]-1008*A[i,j,k-2]+
        8064*A[i,j,k-1]-14350*A[i,j,k]+8064*A[i,j,k+1]-
        1008*A[i,j,k+2]+128*A[i,j,k+3]-9*A[i,j,k+4])/(5040*dx*dx)
    elseif (k==4)
        @views diff = (-9*A[i,j,end]+128*A[i,j,k-3]-1008*A[i,j,k-2]+
        8064*A[i,j,k-1]-14350*A[i,j,k]+8064*A[i,j,k+1]-
        1008*A[i,j,k+2]+128*A[i,j,k+3]-9*A[i,j,k+4])/(5040*dx*dx)
    end
    return prec(diff)
end

#Module end#
end

#############10th ORDER################################################

module differentiations_10th_order

export diff_abc
export dfdx,dfdy,dfdz
export d2fdx2,d2fdy2,d2fdz2

@views function dfdx(A,i,j,k,diff,dx)
    if (i>5 && i<(size(A,1)-4))
        @views diff = (-2*A[i-5,j,k]+25*A[i-4,j,k]-150*A[i-3,j,k]+600*A[i-2,j,k]-
        2100*A[i-1,j,k]+2100*A[i+1,j,k]-600*A[i+2,j,k]+
        150*A[i+3,j,k]-25*A[i+4,j,k]+2*A[i+5,j,k])/(2520*dx)
    elseif (i==(size(A,1)-4))
        @views diff = (-2*A[i-5,j,k]+25*A[i-4,j,k]-150*A[i-3,j,k]+600*A[i-2,j,k]-
        2100*A[i-1,j,k]+2100*A[i+1,j,k]-600*A[i+2,j,k]+
        150*A[i+3,j,k]-25*A[i+4,j,k]+2*A[1,j,k])/(2520*dx)
    elseif (i==(size(A,1)-3))
        @views diff = (-2*A[i-5,j,k]+25*A[i-4,j,k]-150*A[i-3,j,k]+600*A[i-2,j,k]-
        2100*A[i-1,j,k]+2100*A[i+1,j,k]-600*A[i+2,j,k]+
        150*A[i+3,j,k]-25*A[1,j,k]+2*A[2,j,k])/(2520*dx)
    elseif (i==(size(A,1)-2))
        @views diff = (-2*A[i-5,j,k]+25*A[i-4,j,k]-150*A[i-3,j,k]+600*A[i-2,j,k]-
        2100*A[i-1,j,k]+2100*A[i+1,j,k]-600*A[i+2,j,k]+
        150*A[1,j,k]-25*A[2,j,k]+2*A[3,j,k])/(2520*dx)
    elseif (i==(size(A,1)-1))
        @views diff = (-2*A[i-5,j,k]+25*A[i-4,j,k]-150*A[i-3,j,k]+600*A[i-2,j,k]-
        2100*A[i-1,j,k]+2100*A[i+1,j,k]-600*A[1,j,k]+
        150*A[2,j,k]-25*A[3,j,k]+2*A[4,j,k])/(2520*dx)
    elseif (i==(size(A,1)))
        @views diff = (-2*A[i-5,j,k]+25*A[i-4,j,k]-150*A[i-3,j,k]+600*A[i-2,j,k]-
        2100*A[i-1,j,k]+2100*A[1,j,k]-600*A[2,j,k]+
        150*A[3,j,k]-25*A[4,j,k]+2*A[5,j,k])/(2520*dx)
    elseif (i==1)
        @views diff = (-2*A[end-4,j,k]+25*A[end-3,j,k]-150*A[end-2,j,k]+600*A[end-1,j,k]-
        2100*A[end,j,k]+2100*A[i+1,j,k]-600*A[i+2,j,k]+
        150*A[i+3,j,k]-25*A[i+4,j,k]+2*A[i+5,j,k])/(2520*dx)
    elseif (i==2)
        @views diff = (-2*A[end-3,j,k]+25*A[end-2,j,k]-150*A[end-1,j,k]+600*A[end,j,k]-
        2100*A[i-1,j,k]+2100*A[i+1,j,k]-600*A[i+2,j,k]+
        150*A[i+3,j,k]-25*A[i+4,j,k]+2*A[i+5,j,k])/(2520*dx)
    elseif (i==3)
        @views diff = (-2*A[end-2,j,k]+25*A[end-1,j,k]-150*A[end,j,k]+600*A[i-2,j,k]-
        2100*A[i-1,j,k]+2100*A[i+1,j,k]-600*A[i+2,j,k]+
        150*A[i+3,j,k]-25*A[i+4,j,k]+2*A[i+5,j,k])/(2520*dx)
    elseif (i==4)
        @views diff = (-2*A[end-1,j,k]+25*A[end,j,k]-150*A[i-3,j,k]+600*A[i-2,j,k]-
        2100*A[i-1,j,k]+2100*A[i+1,j,k]-600*A[i+2,j,k]+
        150*A[i+3,j,k]-25*A[i+4,j,k]+2*A[i+5,j,k])/(2520*dx)
    elseif (i==5)
        @views diff = (-2*A[end,j,k]+25*A[i-4,j,k]-150*A[i-3,j,k]+600*A[i-2,j,k]-
        2100*A[i-1,j,k]+2100*A[i+1,j,k]-600*A[i+2,j,k]+
        150*A[i+3,j,k]-25*A[i+4,j,k]+2*A[i+5,j,k])/(2520*dx)
    end
    return prec(diff)
end

@views function dfdy(A,i,j,k,diff,dx)
    if (j>5 && j<(size(A,2)-4))
        @views diff = (-2*A[i,j-5,k]+25*A[i,j-4,k]-150*A[i,j-3,k]+600*A[i,j-2,k]-
        2100*A[i,j-1,k]+2100*A[i,j+1,k]-600*A[i,j+2,k]+
        150*A[i,j+3,k]-25*A[i,j+4,k]+2*A[i,j+5,k])/(2520*dx)
    elseif (j==(size(A,2)-4))
        @views diff = (-2*A[i,j-5,k]+25*A[i,j-4,k]-150*A[i,j-3,k]+600*A[i,j-2,k]-
        2100*A[i,j-1,k]+2100*A[i,j+1,k]-600*A[i,j+2,k]+
        150*A[i,j+3,k]-25*A[i,j+4,k]+2*A[i,1,k])/(2520*dx)
    elseif (j==(size(A,2)-3))
        @views diff = (-2*A[i,j-5,k]+25*A[i,j-4,k]-150*A[i,j-3,k]+600*A[i,j-2,k]-
        2100*A[i,j-1,k]+2100*A[i,j+1,k]-600*A[i,j+2,k]+
        150*A[i,j+3,k]-25*A[i,1,k]+2*A[i,2,k])/(2520*dx)
    elseif (j==(size(A,2)-2))
        @views diff = (-2*A[i,j-5,k]+25*A[i,j-4,k]-150*A[i,j-3,k]+600*A[i,j-2,k]-
        2100*A[i,j-1,k]+2100*A[i,j+1,k]-600*A[i,j+2,k]+
        150*A[i,1,k]-25*A[i,2,k]+2*A[i,3,k])/(2520*dx)
    elseif (j==(size(A,2)-1))
        @views diff = (-2*A[i,j-5,k]+25*A[i,j-4,k]-150*A[i,j-3,k]+600*A[i,j-2,k]-
        2100*A[i,j-1,k]+2100*A[i,j+1,k]-600*A[i,1,k]+
        150*A[i,2,k]-25*A[i,3,k]+2*A[i,4,k])/(2520*dx)
    elseif (j==(size(A,2)))
        @views diff = (-2*A[i,j-5,k]+25*A[i,j-4,k]-150*A[i,j-3,k]+600*A[i,j-2,k]-
        2100*A[i,j-1,k]+2100*A[i,1,k]-600*A[i,2,k]+
        150*A[i,3,k]-25*A[i,4,k]+2*A[i,5,k])/(2520*dx)
    elseif (j==1)
        @views diff = (-2*A[i,end-4,k]+25*A[i,end-3,k]-150*A[i,end-2,k]+600*A[i,end-1,k]-
        2100*A[i,end,k]+2100*A[i,j+1,k]-600*A[i,j+2,k]+
        150*A[i,j+3,k]-25*A[i,j+4,k]+2*A[i,j+5,k])/(2520*dx)
    elseif (j==2)
        @views diff = (-2*A[i,end-3,k]+25*A[i,end-2,k]-150*A[i,end-1,k]+600*A[i,end,k]-
        2100*A[i,j-1,k]+2100*A[i,j+1,k]-600*A[i,j+2,k]+
        150*A[i,j+3,k]-25*A[i,j+4,k]+2*A[i,j+5,k])/(2520*dx)
    elseif (j==3)
        @views diff = (-2*A[i,end-2,k]+25*A[i,end-1,k]-150*A[i,end,k]+600*A[i,j-2,k]-
        2100*A[i,j-1,k]+2100*A[i,j+1,k]-600*A[i,j+2,k]+
        150*A[i,j+3,k]-25*A[i,j+4,k]+2*A[i,j+5,k])/(2520*dx)
    elseif (j==4)
        @views diff = (-2*A[i,end-1,k]+25*A[i,end,k]-150*A[i,j-3,k]+600*A[i,j-2,k]-
        2100*A[i,j-1,k]+2100*A[i,j+1,k]-600*A[i,j+2,k]+
        150*A[i,j+3,k]-25*A[i,j+4,k]+2*A[i,j+5,k])/(2520*dx)
    elseif (j==5)
        @views diff = (-2*A[i,end,k]+25*A[i,j-4,k]-150*A[i,j-3,k]+600*A[i,j-2,k]-
        2100*A[i,j-1,k]+2100*A[i,j+1,k]-600*A[i,j+2,k]+
        150*A[i,j+3,k]-25*A[i,j+4,k]+2*A[i,j+5,k])/(2520*dx)
    end
    return prec(diff)
end

@views function dfdz(A,i,j,k,diff,dx)
    if (k>5 && k<(size(A,3)-4))
        @views diff = (-2*A[i,j,k-5]+25*A[i,j,k-4]-150*A[i,j,k-3]+600*A[i,j,k-2]-
        2100*A[i,j,k-1]+2100*A[i,j,k+1]-600*A[i,j,k+2]+
        150*A[i,j,k+3]-25*A[i,j,k+4]+2*A[i,j,k+5])/(2520*dx)
    elseif (k==(size(A,3)-4))
        @views diff = (-2*A[i,j,k-5]+25*A[i,j,k-4]-150*A[i,j,k-3]+600*A[i,j,k-2]-
        2100*A[i,j,k-1]+2100*A[i,j,k+1]-600*A[i,j,k+2]+
        150*A[i,j,k+3]-25*A[i,j,k+4]+2*A[i,j,1])/(2520*dx)
    elseif (k==(size(A,3)-3))
        @views diff = (-2*A[i,j,k-5]+25*A[i,j,k-4]-150*A[i,j,k-3]+600*A[i,j,k-2]-
        2100*A[i,j,k-1]+2100*A[i,j,k+1]-600*A[i,j,k+2]+
        150*A[i,j,k+3]-25*A[i,j,1]+2*A[i,j,2])/(2520*dx)
    elseif (k==(size(A,3)-2))
        @views diff = (-2*A[i,j,k-5]+25*A[i,j,k-4]-150*A[i,j,k-3]+600*A[i,j,k-2]-
        2100*A[i,j,k-1]+2100*A[i,j,k+1]-600*A[i,j,k+2]+
        150*A[i,j,1]-25*A[i,j,2]+2*A[i,j,3])/(2520*dx)
    elseif (k==(size(A,3)-1))
        @views diff = (-2*A[i,j,k-5]+25*A[i,j,k-4]-150*A[i,j,k-3]+600*A[i,j,k-2]-
        2100*A[i,j,k-1]+2100*A[i,j,k+1]-600*A[i,j,1]+
        150*A[i,j,2]-25*A[i,j,3]+2*A[i,j,4])/(2520*dx)
    elseif (k==(size(A,3)))
        @views diff = (-2*A[i,j,k-5]+25*A[i,j,k-4]-150*A[i,j,k-3]+600*A[i,j,k-2]-
        2100*A[i,j,k-1]+2100*A[i,j,1]-600*A[i,j,2]+
        150*A[i,j,3]-25*A[i,j,4]+2*A[i,j,5])/(2520*dx)
    elseif (k==1)
        @views diff = (-2*A[i,j,end-4]+25*A[i,j,end-3]-150*A[i,j,end-2]+600*A[i,j,end-1]-
        2100*A[i,j,end]+2100*A[i,j,k+1]-600*A[i,j,k+2]+
        150*A[i,j,k+3]-25*A[i,j,k+4]+2*A[i,j,k+5])/(2520*dx)
    elseif (k==2)
        @views diff = (-2*A[i,j,end-3]+25*A[i,j,end-2]-150*A[i,j,end-1]+600*A[i,j,end]-
        2100*A[i,j,k-1]+2100*A[i,j,k+1]-600*A[i,j,k+2]+
        150*A[i,j,k+3]-25*A[i,j,k+4]+2*A[i,j,k+5])/(2520*dx)
    elseif (k==3)
        @views diff = (-2*A[i,j,end-2]+25*A[i,j,end-1]-150*A[i,j,end]+600*A[i,j,k-2]-
        2100*A[i,j,k-1]+2100*A[i,j,k+1]-600*A[i,j,k+2]+
        150*A[i,j,k+3]-25*A[i,j,k+4]+2*A[i,j,k+5])/(2520*dx)
    elseif (k==4)
        @views diff = (-2*A[i,j,end-1]+25*A[i,j,end]-150*A[i,j,k-3]+600*A[i,j,k-2]-
        2100*A[i,j,k-1]+2100*A[i,j,k+1]-600*A[i,j,k+2]+
        150*A[i,j,k+3]-25*A[i,j,k+4]+2*A[i,j,k+5])/(2520*dx)
    elseif (k==5)
        @views diff = (-2*A[i,j,end]+25*A[i,j,k-4]-150*A[i,j,k-3]+600*A[i,j,k-2]-
        2100*A[i,j,k-1]+2100*A[i,j,k+1]-600*A[i,j,k+2]+
        150*A[i,j,k+3]-25*A[i,j,k+4]+2*A[i,j,k+5])/(2520*dx)
    end
    return prec(diff)
end

@views function d2fdx2(A,i,j,k,diff,dx)
    if (i>5 && i<(size(A,1)-4))
        @views diff = (8*A[i-5,j,k]-125*A[i-4,j,k]+1000*A[i-3,j,k]-6000*A[i-2,j,k]+
        42000*A[i-1,j,k]-73766*A[i,j,k]+42000*A[i+1,j,k]-6000*A[i+2,j,k]+
        1000*A[i+3,j,k]-125*A[i+4,j,k]+8*A[i+5,j,k])/(25200*dx*dx)
    elseif (i==(size(A,1)-4))
        @views diff = (8*A[i-5,j,k]-125*A[i-4,j,k]+1000*A[i-3,j,k]-6000*A[i-2,j,k]+
        42000*A[i-1,j,k]-73766*A[i,j,k]+42000*A[i+1,j,k]-6000*A[i+2,j,k]+
        1000*A[i+3,j,k]-125*A[i+4,j,k]+8*A[1,j,k])/(25200*dx*dx)
    elseif (i==(size(A,1)-3))
        @views diff = (8*A[i-5,j,k]-125*A[i-4,j,k]+1000*A[i-3,j,k]-6000*A[i-2,j,k]+
        42000*A[i-1,j,k]-73766*A[i,j,k]+42000*A[i+1,j,k]-6000*A[i+2,j,k]+
        1000*A[i+3,j,k]-125*A[1,j,k]+8*A[2,j,k])/(25200*dx*dx)
    elseif (i==(size(A,1)-2))
        @views diff = (8*A[i-5,j,k]-125*A[i-4,j,k]+1000*A[i-3,j,k]-6000*A[i-2,j,k]+
        42000*A[i-1,j,k]-73766*A[i,j,k]+42000*A[i+1,j,k]-6000*A[i+2,j,k]+
        1000*A[1,j,k]-125*A[2,j,k]+8*A[3,j,k])/(25200*dx*dx)
    elseif (i==(size(A,1)-1))
        @views diff = (8*A[i-5,j,k]-125*A[i-4,j,k]+1000*A[i-3,j,k]-6000*A[i-2,j,k]+
        42000*A[i-1,j,k]-73766*A[i,j,k]+42000*A[i+1,j,k]-6000*A[1,j,k]+
        1000*A[2,j,k]-125*A[3,j,k]+8*A[4,j,k])/(25200*dx*dx)
    elseif (i==(size(A,1)))
        @views diff = (8*A[i-5,j,k]-125*A[i-4,j,k]+1000*A[i-3,j,k]-6000*A[i-2,j,k]+
        42000*A[i-1,j,k]-73766*A[i,j,k]+42000*A[1,j,k]-6000*A[2,j,k]+
        1000*A[3,j,k]-125*A[4,j,k]+8*A[5,j,k])/(25200*dx*dx)
    elseif (i==1)
        @views diff = (8*A[end-4,j,k]-125*A[end-3,j,k]+1000*A[end-2,j,k]-6000*A[end-1,j,k]+
        42000*A[end,j,k]-73766*A[i,j,k]+42000*A[i+1,j,k]-6000*A[i+2,j,k]+
        1000*A[i+3,j,k]-125*A[i+4,j,k]+8*A[i+5,j,k])/(25200*dx*dx)
    elseif (i==2)
        @views diff = (8*A[end-3,j,k]-125*A[end-2,j,k]+1000*A[end-1,j,k]-6000*A[end,j,k]+
        42000*A[i-1,j,k]-73766*A[i,j,k]+42000*A[i+1,j,k]-6000*A[i+2,j,k]+
        1000*A[i+3,j,k]-125*A[i+4,j,k]+8*A[i+5,j,k])/(25200*dx*dx)
    elseif (i==3)
        @views diff = (8*A[end-2,j,k]-125*A[end-1,j,k]+1000*A[end,j,k]-6000*A[i-2,j,k]+
        42000*A[i-1,j,k]-73766*A[i,j,k]+42000*A[i+1,j,k]-6000*A[i+2,j,k]+
        1000*A[i+3,j,k]-125*A[i+4,j,k]+8*A[i+5,j,k])/(25200*dx*dx)
    elseif (i==4)
        @views diff = (8*A[end-1,j,k]-125*A[end,j,k]+1000*A[i-3,j,k]-6000*A[i-2,j,k]+
        42000*A[i-1,j,k]-73766*A[i,j,k]+42000*A[i+1,j,k]-6000*A[i+2,j,k]+
        1000*A[i+3,j,k]-125*A[i+4,j,k]+8*A[i+5,j,k])/(25200*dx*dx)
    elseif (i==5)
        @views diff = (8*A[end,j,k]-125*A[i-4,j,k]+1000*A[i-3,j,k]-6000*A[i-2,j,k]+
        42000*A[i-1,j,k]-73766*A[i,j,k]+42000*A[i+1,j,k]-6000*A[i+2,j,k]+
        1000*A[i+3,j,k]-125*A[i+4,j,k]+8*A[i+5,j,k])/(25200*dx*dx)
    end
    return prec(diff)
end

@views function d2fdy2(A,i,j,k,diff,dx)
    if (j>5 && j<(size(A,2)-4))
        @views diff = (8*A[i,j-5,k]-125*A[i,j-4,k]+1000*A[i,j-3,k]-6000*A[i,j-2,k]+
        42000*A[i,j-1,k]-73766*A[i,j,k]+42000*A[i,j+1,k]-6000*A[i,j+2,k]+
        1000*A[i,j+3,k]-125*A[i,j+4,k]+8*A[i,j+5,k])/(25200*dx*dx)
    elseif (j==(size(A,2)-4))
        @views diff = (8*A[i,j-5,k]-125*A[i,j-4,k]+1000*A[i,j-3,k]-6000*A[i,j-2,k]+
        42000*A[i,j-1,k]-73766*A[i,j,k]+42000*A[i,j+1,k]-6000*A[i,j+2,k]+
        1000*A[i,j+3,k]-125*A[i,j+4,k]+8*A[i,1,k])/(25200*dx*dx)
    elseif (j==(size(A,2)-3))
        @views diff = (8*A[i,j-5,k]-125*A[i,j-4,k]+1000*A[i,j-3,k]-6000*A[i,j-2,k]+
        42000*A[i,j-1,k]-73766*A[i,j,k]+42000*A[i,j+1,k]-6000*A[i,j+2,k]+
        1000*A[i,j+3,k]-125*A[i,1,k]+8*A[i,2,k])/(25200*dx*dx)
    elseif (j==(size(A,2)-2))
        @views diff = (8*A[i,j-5,k]-125*A[i,j-4,k]+1000*A[i,j-3,k]-6000*A[i,j-2,k]+
        42000*A[i,j-1,k]-73766*A[i,j,k]+42000*A[i,j+1,k]-6000*A[i,j+2,k]+
        1000*A[i,1,k]-125*A[i,2,k]+8*A[i,3,k])/(25200*dx*dx)
    elseif (j==(size(A,2)-1))
        @views diff = (8*A[i,j-5,k]-125*A[i,j-4,k]+1000*A[i,j-3,k]-6000*A[i,j-2,k]+
        42000*A[i,j-1,k]-73766*A[i,j,k]+42000*A[i,j+1,k]-6000*A[i,1,k]+
        1000*A[i,2,k]-125*A[i,3,k]+8*A[i,4,k])/(25200*dx*dx)
    elseif (j==(size(A,2)))
        @views diff = (8*A[i,j-5,k]-125*A[i,j-4,k]+1000*A[i,j-3,k]-6000*A[i,j-2,k]+
        42000*A[i,j-1,k]-73766*A[i,j,k]+42000*A[i,1,k]-6000*A[i,2,k]+
        1000*A[i,3,k]-125*A[i,4,k]+8*A[i,5,k])/(25200*dx*dx)
    elseif (j==1)
        @views diff = (8*A[i,end-4,k]-125*A[i,end-3,k]+1000*A[i,end-2,k]-6000*A[i,end-1,k]+
        42000*A[i,end,k]-73766*A[i,j,k]+42000*A[i,j+1,k]-6000*A[i,j+2,k]+
        1000*A[i,j+3,k]-125*A[i,j+4,k]+8*A[i,j+5,k])/(25200*dx*dx)
    elseif (j==2)
        @views diff = (8*A[i,end-3,k]-125*A[i,end-2,k]+1000*A[i,end-1,k]-6000*A[i,end,k]+
        42000*A[i,j-1,k]-73766*A[i,j,k]+42000*A[i,j+1,k]-6000*A[i,j+2,k]+
        1000*A[i,j+3,k]-125*A[i,j+4,k]+8*A[i,j+5,k])/(25200*dx*dx)
    elseif (j==3)
        @views diff = (8*A[i,end-2,k]-125*A[i,end-1,k]+1000*A[i,end,k]-6000*A[i,j-2,k]+
        42000*A[i,j-1,k]-73766*A[i,j,k]+42000*A[i,j+1,k]-6000*A[i,j+2,k]+
        1000*A[i,j+3,k]-125*A[i,j+4,k]+8*A[i,j+5,k])/(25200*dx*dx)
    elseif (j==4)
        @views diff = (8*A[i,end-1,k]-125*A[i,end,k]+1000*A[i,j-3,k]-6000*A[i,j-2,k]+
        42000*A[i,j-1,k]-73766*A[i,j,k]+42000*A[i,j+1,k]-6000*A[i,j+2,k]+
        1000*A[i,j+3,k]-125*A[i,j+4,k]+8*A[i,j+5,k])/(25200*dx*dx)
    elseif (j==5)
        @views diff = (8*A[i,end,k]-125*A[i,j-4,k]+1000*A[i,j-3,k]-6000*A[i,j-2,k]+
        42000*A[i,j-1,k]-73766*A[i,j,k]+42000*A[i,j+1,k]-6000*A[i,j+2,k]+
        1000*A[i,j+3,k]-125*A[i,j+4,k]+8*A[i,j+5,k])/(25200*dx*dx)
    end
    return prec(diff)
end

@views function d2fdz2(A,i,j,k,diff,dx)
    if (i>5 && i<(size(A,3)-4))
        @views diff = (8*A[i,j,k-5]-125*A[i,j,k-4]+1000*A[i,j,k-3]-6000*A[i,j,k-2]+
        42000*A[i,j,k-1]-73766*A[i,j,k]+42000*A[i,j,k+1]-6000*A[i,j,k+2]+
        1000*A[i,j,k+3]-125*A[i,j,k+4]+8*A[i,j,k+5])/(25200*dx*dx)
    elseif (k==(size(A,3)-4))
        @views diff = (8*A[i,j,k-5]-125*A[i,j,k-4]+1000*A[i,j,k-3]-6000*A[i,j,k-2]+
        42000*A[i,j,k-1]-73766*A[i,j,k]+42000*A[i,j,k+1]-6000*A[i,j,k+2]+
        1000*A[i,j,k+3]-125*A[i,j,k+4]+8*A[i,j,1])/(25200*dx*dx)
    elseif (k==(size(A,3)-3))
        @views diff = (8*A[i,j,k-5]-125*A[i,j,k-4]+1000*A[i,j,k-3]-6000*A[i,j,k-2]+
        42000*A[i,j,k-1]-73766*A[i,j,k]+42000*A[i,j,k+1]-6000*A[i,j,k+2]+
        1000*A[i,j,k+3]-125*A[i,j,1]+8*A[i,j,2])/(25200*dx*dx)
    elseif (k==(size(A,3)-2))
        @views diff = (8*A[i,j,k-5]-125*A[i,j,k-4]+1000*A[i,j,k-3]-6000*A[i,j,k-2]+
        42000*A[i,j,k-1]-73766*A[i,j,k]+42000*A[i,j,k+1]-6000*A[i,j,k+2]+
        1000*A[i,j,1]-125*A[i,j,2]+8*A[i,j,3])/(25200*dx*dx)
    elseif (k==(size(A,3)-1))
        @views diff = (8*A[i,j,k-5]-125*A[i,j,k-4]+1000*A[i,j,k-3]-6000*A[i,j,k-2]+
        42000*A[i,j,k-1]-73766*A[i,j,k]+42000*A[i,j,k+1]-6000*A[i,j,1]+
        1000*A[i,j,2]-125*A[i,j,3]+8*A[i,j,4])/(25200*dx*dx)
    elseif (k==(size(A,3)))
        @views diff = (8*A[i,j,k-5]-125*A[i,j,k-4]+1000*A[i,j,k-3]-6000*A[i,j,k-2]+
        42000*A[i,j,k-1]-73766*A[i,j,k]+42000*A[i,j,1]-6000*A[i,j,2]+
        1000*A[i,j,3]-125*A[i,j,4]+8*A[i,j,5])/(25200*dx*dx)
    elseif (k==1)
        @views diff = (8*A[i,j,end-4]-125*A[i,j,end-3]+1000*A[i,j,end-2]-6000*A[i,j,end-1]+
        42000*A[i,j,end]-73766*A[i,j,k]+42000*A[i,j,k+1]-6000*A[i,j,k+2]+
        1000*A[i,j,k+3]-125*A[i,j,k+4]+8*A[i,j,k+5])/(25200*dx*dx)
    elseif (k==2)
        @views diff = (8*A[i,j,end-3]-125*A[i,j,end-2]+1000*A[i,j,end-1]-6000*A[i,j,end]+
        42000*A[i,j,k-1]-73766*A[i,j,k]+42000*A[i,j,k+1]-6000*A[i,j,k+2]+
        1000*A[i,j,k+3]-125*A[i,j,k+4]+8*A[i,j,k+5])/(25200*dx*dx)
    elseif (k==3)
        @views diff = (8*A[i,j,end-2]-125*A[i,j,end-1]+1000*A[i,j,end]-6000*A[i,j,k-2]+
        42000*A[i,j,k-1]-73766*A[i,j,k]+42000*A[i,j,k+1]-6000*A[i,j,k+2]+
        1000*A[i,j,k+3]-125*A[i,j,k+4]+8*A[i,j,k+5])/(25200*dx*dx)
    elseif (k==4)
        @views diff = (8*A[i,j,end-1]-125*A[i,j,end]+1000*A[i,j,k-3]-6000*A[i,j,k-2]+
        42000*A[i,j,k-1]-73766*A[i,j,k]+42000*A[i,j,k+1]-6000*A[i,j,k+2]+
        1000*A[i,j,k+3]-125*A[i,j,k+4]+8*A[i,j,k+5])/(25200*dx*dx)
    elseif (k==5)
        @views diff = (8*A[i,j,end]-125*A[i,j,k-4]+1000*A[i,j,k-3]-6000*A[i,j,k-2]+
        42000*A[i,j,k-1]-73766*A[i,j,k]+42000*A[i,j,k+1]-6000*A[i,j,k+2]+
        1000*A[i,j,k+3]-125*A[i,j,k+4]+8*A[i,j,k+5])/(25200*dx*dx)
    end
    return prec(diff)
end

#Module end#
end


#############12th ORDER################################################

module differentiations_12th_order

export diff_abc
export dfdx,dfdy,dfdz
export d2fdx2,d2fdy2,d2fdz2

@views function dfdx(A,i,j,k,diff,dx)
    if (i>6 && i<(size(A,1)-5))
        @views diff = (5*A[i-6,j,k]-72*A[i-5,j,k]+495*A[i-4,j,k]-
        2200*A[i-3,j,k]+7425*A[i-2,j,k]-23760*A[i-1,j,k]+
        23760*A[i+1,j,k]-7425*A[i+2,j,k]+2200*A[i+3,j,k]-
        495*A[i+4,j,k]+72*A[i+5,j,k]-5*A[i+6,j,k])/(27720*dx)
    elseif (i==(size(A,1)-5))
        @views diff = (5*A[i-6,j,k]-72*A[i-5,j,k]+495*A[i-4,j,k]-
        2200*A[i-3,j,k]+7425*A[i-2,j,k]-23760*A[i-1,j,k]+
        23760*A[i+1,j,k]-7425*A[i+2,j,k]+2200*A[i+3,j,k]-
        495*A[i+4,j,k]+72*A[i+5,j,k]-5*A[1,j,k])/(27720*dx)
    elseif (i==(size(A,1)-4))
        @views diff = (5*A[i-6,j,k]-72*A[i-5,j,k]+495*A[i-4,j,k]-
        2200*A[i-3,j,k]+7425*A[i-2,j,k]-23760*A[i-1,j,k]+
        23760*A[i+1,j,k]-7425*A[i+2,j,k]+2200*A[i+3,j,k]-
        495*A[i+4,j,k]+72*A[1,j,k]-5*A[2,j,k])/(27720*dx)
    elseif (i==(size(A,1)-3))
        @views diff = (5*A[i-6,j,k]-72*A[i-5,j,k]+495*A[i-4,j,k]-
        2200*A[i-3,j,k]+7425*A[i-2,j,k]-23760*A[i-1,j,k]+
        23760*A[i+1,j,k]-7425*A[i+2,j,k]+2200*A[i+3,j,k]-
        495*A[1,j,k]+72*A[2,j,k]-5*A[3,j,k])/(27720*dx)
    elseif (i==(size(A,1)-2))
        @views diff = (5*A[i-6,j,k]-72*A[i-5,j,k]+495*A[i-4,j,k]-
        2200*A[i-3,j,k]+7425*A[i-2,j,k]-23760*A[i-1,j,k]+
        23760*A[i+1,j,k]-7425*A[i+2,j,k]+2200*A[1,j,k]-
        495*A[2,j,k]+72*A[3,j,k]-5*A[4,j,k])/(27720*dx)
    elseif (i==(size(A,1)-1))
        @views diff = (5*A[i-6,j,k]-72*A[i-5,j,k]+495*A[i-4,j,k]-
        2200*A[i-3,j,k]+7425*A[i-2,j,k]-23760*A[i-1,j,k]+
        23760*A[i+1,j,k]-7425*A[1,j,k]+2200*A[2,j,k]-
        495*A[3,j,k]+72*A[4,j,k]-5*A[5,j,k])/(27720*dx)
    elseif (i==(size(A,1)))
        @views diff = (5*A[i-6,j,k]-72*A[i-5,j,k]+495*A[i-4,j,k]-
        2200*A[i-3,j,k]+7425*A[i-2,j,k]-23760*A[i-1,j,k]+
        23760*A[1,j,k]-7425*A[2,j,k]+2200*A[3,j,k]-
        495*A[4,j,k]+72*A[5,j,k]-5*A[6,j,k])/(27720*dx)
    elseif (i==1)
        @views diff = (5*A[end-5,j,k]-72*A[end-4,j,k]+495*A[end-3,j,k]-
        2200*A[end-2,j,k]+7425*A[end-1,j,k]-23760*A[end,j,k]+
        23760*A[i+1,j,k]-7425*A[i+2,j,k]+2200*A[i+3,j,k]-
        495*A[i+4,j,k]+72*A[i+5,j,k]-5*A[i+6,j,k])/(27720*dx)
    elseif (i==2)
        @views diff = (5*A[end-4,j,k]-72*A[end-3,j,k]+495*A[end-2,j,k]-
        2200*A[end-1,j,k]+7425*A[end,j,k]-23760*A[i-1,j,k]+
        23760*A[i+1,j,k]-7425*A[i+2,j,k]+2200*A[i+3,j,k]-
        495*A[i+4,j,k]+72*A[i+5,j,k]-5*A[i+6,j,k])/(27720*dx)
    elseif (i==3)
        @views diff = (5*A[end-3,j,k]-72*A[end-2,j,k]+495*A[end-1,j,k]-
        2200*A[end,j,k]+7425*A[i-2,j,k]-23760*A[i-1,j,k]+
        23760*A[i+1,j,k]-7425*A[i+2,j,k]+2200*A[i+3,j,k]-
        495*A[i+4,j,k]+72*A[i+5,j,k]-5*A[i+6,j,k])/(27720*dx)
    elseif (i==4)
        @views diff = (5*A[end-2,j,k]-72*A[end-1,j,k]+495*A[end,j,k]-
        2200*A[i-3,j,k]+7425*A[i-2,j,k]-23760*A[i-1,j,k]+
        23760*A[i+1,j,k]-7425*A[i+2,j,k]+2200*A[i+3,j,k]-
        495*A[i+4,j,k]+72*A[i+5,j,k]-5*A[i+6,j,k])/(27720*dx)
    elseif (i==5)
        @views diff = (5*A[end-1,j,k]-72*A[end,j,k]+495*A[i-4,j,k]-
        2200*A[i-3,j,k]+7425*A[i-2,j,k]-23760*A[i-1,j,k]+
        23760*A[i+1,j,k]-7425*A[i+2,j,k]+2200*A[i+3,j,k]-
        495*A[i+4,j,k]+72*A[i+5,j,k]-5*A[i+6,j,k])/(27720*dx)
    elseif (i==6)
        @views diff = (5*A[end,j,k]-72*A[i-5,j,k]+495*A[i-4,j,k]-
        2200*A[i-3,j,k]+7425*A[i-2,j,k]-23760*A[i-1,j,k]+
        23760*A[i+1,j,k]-7425*A[i+2,j,k]+2200*A[i+3,j,k]-
        495*A[i+4,j,k]+72*A[i+5,j,k]-5*A[i+6,j,k])/(27720*dx)
    end
    return prec(diff)
end

@views function dfdy(A,i,j,k,diff,dx)
    if (j>6 && j<(size(A,2)-5))
        @views diff = (5*A[i,j-6,k]-72*A[i,j-5,k]+495*A[i,j-4,k]-
        2200*A[i,j-3,k]+7425*A[i,j-2,k]-23760*A[i,j-1,k]+
        23760*A[i,j+1,k]-7425*A[i,j+2,k]+2200*A[i,j+3,k]-
        495*A[i,j+4,k]+72*A[i,j+5,k]-5*A[i,j+6,k])/(27720*dx)
    elseif (j==(size(A,2)-5))
        @views diff = (5*A[i,j-6,k]-72*A[i,j-5,k]+495*A[i,j-4,k]-
        2200*A[i,j-3,k]+7425*A[i,j-2,k]-23760*A[i,j-1,k]+
        23760*A[i,j+1,k]-7425*A[i,j+2,k]+2200*A[i,j+3,k]-
        495*A[i,j+4,k]+72*A[i,j+5,k]-5*A[i,1,k])/(27720*dx)
    elseif (j==(size(A,2)-4))
        @views diff = (5*A[i,j-6,k]-72*A[i,j-5,k]+495*A[i,j-4,k]-
        2200*A[i,j-3,k]+7425*A[i,j-2,k]-23760*A[i,j-1,k]+
        23760*A[i,j+1,k]-7425*A[i,j+2,k]+2200*A[i,j+3,k]-
        495*A[i,j+4,k]+72*A[i,1,k]-5*A[i,2,k])/(27720*dx)
    elseif (j==(size(A,2)-3))
        @views diff = (5*A[i,j-6,k]-72*A[i,j-5,k]+495*A[i,j-4,k]-
        2200*A[i,j-3,k]+7425*A[i,j-2,k]-23760*A[i,j-1,k]+
        23760*A[i,j+1,k]-7425*A[i,j+2,k]+2200*A[i,j+3,k]-
        495*A[i,1,k]+72*A[i,2,k]-5*A[i,3,k])/(27720*dx)
    elseif (j==(size(A,2)-2))
        @views diff = (5*A[i,j-6,k]-72*A[i,j-5,k]+495*A[i,j-4,k]-
        2200*A[i,j-3,k]+7425*A[i,j-2,k]-23760*A[i,j-1,k]+
        23760*A[i,j+1,k]-7425*A[i,j+2,k]+2200*A[i,1,k]-
        495*A[i,2,k]+72*A[i,3,k]-5*A[i,4,k])/(27720*dx)
    elseif (j==(size(A,2)-1))
        @views diff = (5*A[i,j-6,k]-72*A[i,j-5,k]+495*A[i,j-4,k]-
        2200*A[i,j-3,k]+7425*A[i,j-2,k]-23760*A[i,j-1,k]+
        23760*A[i,j+1,k]-7425*A[i,1,k]+2200*A[i,2,k]-
        495*A[i,3,k]+72*A[i,4,k]-5*A[i,5,k])/(27720*dx)
    elseif (j==(size(A,2)))
        @views diff = (5*A[i,j-6,k]-72*A[i,j-5,k]+495*A[i,j-4,k]-
        2200*A[i,j-3,k]+7425*A[i,j-2,k]-23760*A[i,j-1,k]+
        23760*A[i,1,k]-7425*A[i,2,k]+2200*A[i,3,k]-
        495*A[i,4,k]+72*A[i,5,k]-5*A[i,6,k])/(27720*dx)
    elseif (j==1)
        @views diff = (5*A[i,end-5,k]-72*A[i,end-4,k]+495*A[i,end-3,k]-
        2200*A[i,end-2,k]+7425*A[i,end-1,k]-23760*A[i,end,k]+
        23760*A[i,j+1,k]-7425*A[i,j+2,k]+2200*A[i,j+3,k]-
        495*A[i,j+4,k]+72*A[i,j+5,k]-5*A[i,j+6,k])/(27720*dx)
    elseif (j==2)
        @views diff = (5*A[i,end-4,k]-72*A[i,end-3,k]+495*A[i,end-2,k]-
        2200*A[i,end-1,k]+7425*A[i,end,k]-23760*A[i,j-1,k]+
        23760*A[i,j+1,k]-7425*A[i,j+2,k]+2200*A[i,j+3,k]-
        495*A[i,j+4,k]+72*A[i,j+5,k]-5*A[i,j+6,k])/(27720*dx)
    elseif (j==3)
        @views diff = (5*A[i,end-3,k]-72*A[i,end-2,k]+495*A[i,end-1,k]-
        2200*A[i,end,k]+7425*A[i,j-2,k]-23760*A[i,j-1,k]+
        23760*A[i,j+1,k]-7425*A[i,j+2,k]+2200*A[i,j+3,k]-
        495*A[i,j+4,k]+72*A[i,j+5,k]-5*A[i,j+6,k])/(27720*dx)
    elseif (j==4)
        @views diff = (5*A[i,end-2,k]-72*A[i,end-1,k]+495*A[i,end,k]-
        2200*A[i,j-3,k]+7425*A[i,j-2,k]-23760*A[i,j-1,k]+
        23760*A[i,j+1,k]-7425*A[i,j+2,k]+2200*A[i,j+3,k]-
        495*A[i,j+4,k]+72*A[i,j+5,k]-5*A[i,j+6,k])/(27720*dx)
    elseif (j==5)
        @views diff = (5*A[i,end-1,k]-72*A[i,end,k]+495*A[i,j-4,k]-
        2200*A[i,j-3,k]+7425*A[i,j-2,k]-23760*A[i,j-1,k]+
        23760*A[i,j+1,k]-7425*A[i,j+2,k]+2200*A[i,j+3,k]-
        495*A[i,j+4,k]+72*A[i,j+5,k]-5*A[i,j+6,k])/(27720*dx)
    elseif (j==6)
        @views diff = (5*A[i,end,k]-72*A[i,j-5,k]+495*A[i,j-4,k]-
        2200*A[i,j-3,k]+7425*A[i,j-2,k]-23760*A[i,j-1,k]+
        23760*A[i,j+1,k]-7425*A[i,j+2,k]+2200*A[i,j+3,k]-
        495*A[i,j+4,k]+72*A[i,j+5,k]-5*A[i,j+6,k])/(27720*dx)
    end
    return prec(diff)
end

@views function dfdz(A,i,j,k,diff,dx)
    if (k>6 && k<(size(A,3)-5))
        @views diff = (5*A[i,j,k-6]-72*A[i,j,k-5]+495*A[i,j,k-4]-
        2200*A[i,j,k-3]+7425*A[i,j,k-2]-23760*A[i,j,k-1]+
        23760*A[i,j,k+1]-7425*A[i,j,k+2]+2200*A[i,j,k+3]-
        495*A[i,j,k+4]+72*A[i,j,k+5]-5*A[i,j,k+6])/(27720*dx)
    elseif (k==(size(A,3)-5))
        @views diff = (5*A[i,j,k-6]-72*A[i,j,k-5]+495*A[i,j,k-4]-
        2200*A[i,j,k-3]+7425*A[i,j,k-2]-23760*A[i,j,k-1]+
        23760*A[i,j,k+1]-7425*A[i,j,k+2]+2200*A[i,j,k+3]-
        495*A[i,j,k+4]+72*A[i,j,k+5]-5*A[i,j,1])/(27720*dx)
    elseif (k==(size(A,3)-4))
        @views diff = (5*A[i,j,k-6]-72*A[i,j,k-5]+495*A[i,j,k-4]-
        2200*A[i,j,k-3]+7425*A[i,j,k-2]-23760*A[i,j,k-1]+
        23760*A[i,j,k+1]-7425*A[i,j,k+2]+2200*A[i,j,k+3]-
        495*A[i,j,k+4]+72*A[i,j,1]-5*A[i,j,2])/(27720*dx)
    elseif (k==(size(A,3)-3))
        @views diff = (5*A[i,j,k-6]-72*A[i,j,k-5]+495*A[i,j,k-4]-
        2200*A[i,j,k-3]+7425*A[i,j,k-2]-23760*A[i,j,k-1]+
        23760*A[i,j,k+1]-7425*A[i,j,k+2]+2200*A[i,j,k+3]-
        495*A[i,j,1]+72*A[i,j,2]-5*A[i,j,3])/(27720*dx)
    elseif (k==(size(A,3)-2))
        @views diff = (5*A[i,j,k-6]-72*A[i,j,k-5]+495*A[i,j,k-4]-
        2200*A[i,j,k-3]+7425*A[i,j,k-2]-23760*A[i,j,k-1]+
        23760*A[i,j,k+1]-7425*A[i,j,k+2]+2200*A[i,j,1]-
        495*A[i,j,2]+72*A[i,j,3]-5*A[i,j,4])/(27720*dx)
    elseif (k==(size(A,3)-1))
        @views diff = (5*A[i,j,k-6]-72*A[i,j,k-5]+495*A[i,j,k-4]-
        2200*A[i,j,k-3]+7425*A[i,j,k-2]-23760*A[i,j,k-1]+
        23760*A[i,j,k+1]-7425*A[i,j,1]+2200*A[i,j,2]-
        495*A[i,j,3]+72*A[i,j,4]-5*A[i,j,5])/(27720*dx)
    elseif (k==(size(A,3)))
        @views diff = (5*A[i,j,k-6]-72*A[i,j,k-5]+495*A[i,j,k-4]-
        2200*A[i,j,k-3]+7425*A[i,j,k-2]-23760*A[i,j,k-1]+
        23760*A[i,j,1]-7425*A[i,j,2]+2200*A[i,j,3]-
        495*A[i,j,4]+72*A[i,j,5]-5*A[i,j,6])/(27720*dx)
    elseif (k==1)
        @views diff = (5*A[i,j,end-5]-72*A[i,j,end-4]+495*A[i,j,end-3]-
        2200*A[i,j,end-2]+7425*A[i,j,end-1]-23760*A[i,j,end]+
        23760*A[i,j,k+1]-7425*A[i,j,k+2]+2200*A[i,j,k+3]-
        495*A[i,j,k+4]+72*A[i,j,k+5]-5*A[i,j,k+6])/(27720*dx)
    elseif (k==2)
        @views diff = (5*A[i,j,end-4]-72*A[i,j,end-3]+495*A[i,j,end-2]-
        2200*A[i,j,end-1]+7425*A[i,j,end]-23760*A[i,j,k-1]+
        23760*A[i,j,k+1]-7425*A[i,j,k+2]+2200*A[i,j,k+3]-
        495*A[i,j,k+4]+72*A[i,j,k+5]-5*A[i,j,k+6])/(27720*dx)
    elseif (k==3)
        @views diff = (5*A[i,j,end-3]-72*A[i,j,end-2]+495*A[i,j,end-1]-
        2200*A[i,j,end]+7425*A[i,j,k-2]-23760*A[i,j,k-1]+
        23760*A[i,j,k+1]-7425*A[i,j,k+2]+2200*A[i,j,k+3]-
        495*A[i,j,k+4]+72*A[i,j,k+5]-5*A[i,j,k+6])/(27720*dx)
    elseif (k==4)
        @views diff = (5*A[i,j,end-2]-72*A[i,j,end-1]+495*A[i,j,end]-
        2200*A[i,j,k-3]+7425*A[i,j,k-2]-23760*A[i,j,k-1]+
        23760*A[i,j,k+1]-7425*A[i,j,k+2]+2200*A[i,j,k+3]-
        495*A[i,j,k+4]+72*A[i,j,k+5]-5*A[i,j,k+6])/(27720*dx)
    elseif (k==5)
        @views diff = (5*A[i,j,end-1]-72*A[i,j,end]+495*A[i,j,k-4]-
        2200*A[i,j,k-3]+7425*A[i,j,k-2]-23760*A[i,j,k-1]+
        23760*A[i,j,k+1]-7425*A[i,j,k+2]+2200*A[i,j,k+3]-
        495*A[i,j,k+4]+72*A[i,j,k+5]-5*A[i,j,k+6])/(27720*dx)
    elseif (k==6)
        @views diff = (5*A[i,j,end]-72*A[i,j,k-5]+495*A[i,j,k-4]-
        2200*A[i,j,k-3]+7425*A[i,j,k-2]-23760*A[i,j,k-1]+
        23760*A[i,j,k+1]-7425*A[i,j,k+2]+2200*A[i,j,k+3]-
        495*A[i,j,k+4]+72*A[i,j,k+5]-5*A[i,j,k+6])/(27720*dx)
    end
    return prec(diff)
end

@views function d2fdx2(A,i,j,k,diff,dx)
    if (i>6 && i<(size(A,1)-5))
        @views diff = (-50*A[i-6,j,k]+864*A[i-5,j,k]-7425*A[i-4,j,k]+
        44000*A[i-3,j,k]-222750*A[i-2,j,k]+1425600*A[i-1,j,k]-2480478*A[i,j,k]+
        1425600*A[i+1,j,k]-222750*A[i+2,j,k]+44000*A[i+3,j,k]-
        7425*A[i+4,j,k]+864*A[i+5,j,k]-50*A[i+6,j,k])/(831600*dx*dx)
    elseif (i==(size(A,1)-5))
        @views diff = (-50*A[i-6,j,k]+864*A[i-5,j,k]-7425*A[i-4,j,k]+
        44000*A[i-3,j,k]-222750*A[i-2,j,k]+1425600*A[i-1,j,k]-2480478*A[i,j,k]+
        1425600*A[i+1,j,k]-222750*A[i+2,j,k]+44000*A[i+3,j,k]-
        7425*A[i+4,j,k]+864*A[i+5,j,k]-50*A[1,j,k])/(831600*dx*dx)
    elseif (i==(size(A,1)-4))
        @views diff = (-50*A[i-6,j,k]+864*A[i-5,j,k]-7425*A[i-4,j,k]+
        44000*A[i-3,j,k]-222750*A[i-2,j,k]+1425600*A[i-1,j,k]-2480478*A[i,j,k]+
        1425600*A[i+1,j,k]-222750*A[i+2,j,k]+44000*A[i+3,j,k]-
        7425*A[i+4,j,k]+864*A[1,j,k]-50*A[2,j,k])/(831600*dx*dx)
    elseif (i==(size(A,1)-3))
        @views diff = (-50*A[i-6,j,k]+864*A[i-5,j,k]-7425*A[i-4,j,k]+
        44000*A[i-3,j,k]-222750*A[i-2,j,k]+1425600*A[i-1,j,k]-2480478*A[i,j,k]+
        1425600*A[i+1,j,k]-222750*A[i+2,j,k]+44000*A[i+3,j,k]-
        7425*A[1,j,k]+864*A[2,j,k]-50*A[3,j,k])/(831600*dx*dx)
    elseif (i==(size(A,1)-2))
        @views diff = (-50*A[i-6,j,k]+864*A[i-5,j,k]-7425*A[i-4,j,k]+
        44000*A[i-3,j,k]-222750*A[i-2,j,k]+1425600*A[i-1,j,k]-2480478*A[i,j,k]+
        1425600*A[i+1,j,k]-222750*A[i+2,j,k]+44000*A[1,j,k]-
        7425*A[2,j,k]+864*A[3,j,k]-50*A[4,j,k])/(831600*dx*dx)
    elseif (i==(size(A,1)-1))
        @views diff = (-50*A[i-6,j,k]+864*A[i-5,j,k]-7425*A[i-4,j,k]+
        44000*A[i-3,j,k]-222750*A[i-2,j,k]+1425600*A[i-1,j,k]-2480478*A[i,j,k]+
        1425600*A[i+1,j,k]-222750*A[1,j,k]+44000*A[2,j,k]-
        7425*A[3,j,k]+864*A[4,j,k]-50*A[5,j,k])/(831600*dx*dx)
    elseif (i==(size(A,1)))
        @views diff = (-50*A[i-6,j,k]+864*A[i-5,j,k]-7425*A[i-4,j,k]+
        44000*A[i-3,j,k]-222750*A[i-2,j,k]+1425600*A[i-1,j,k]-2480478*A[i,j,k]+
        1425600*A[1,j,k]-222750*A[2,j,k]+44000*A[3,j,k]-
        7425*A[4,j,k]+864*A[5,j,k]-50*A[6,j,k])/(831600*dx*dx)
    elseif (i==1)
        @views diff = (-50*A[end-5,j,k]+864*A[end-4,j,k]-7425*A[end-3,j,k]+
        44000*A[end-2,j,k]-222750*A[end-1,j,k]+1425600*A[end,j,k]-2480478*A[i,j,k]+
        1425600*A[i+1,j,k]-222750*A[i+2,j,k]+44000*A[i+3,j,k]-
        7425*A[i+4,j,k]+864*A[i+5,j,k]-50*A[i+6,j,k])/(831600*dx*dx)
    elseif (i==2)
        @views diff = (-50*A[end-4,j,k]+864*A[end-3,j,k]-7425*A[end-2,j,k]+
        44000*A[end-1,j,k]-222750*A[end,j,k]+1425600*A[i-1,j,k]-2480478*A[i,j,k]+
        1425600*A[i+1,j,k]-222750*A[i+2,j,k]+44000*A[i+3,j,k]-
        7425*A[i+4,j,k]+864*A[i+5,j,k]-50*A[i+6,j,k])/(831600*dx*dx)
    elseif (i==3)
        @views diff = (-50*A[end-3,j,k]+864*A[end-2,j,k]-7425*A[end-1,j,k]+
        44000*A[end,j,k]-222750*A[i-2,j,k]+1425600*A[i-1,j,k]-2480478*A[i,j,k]+
        1425600*A[i+1,j,k]-222750*A[i+2,j,k]+44000*A[i+3,j,k]-
        7425*A[i+4,j,k]+864*A[i+5,j,k]-50*A[i+6,j,k])/(831600*dx*dx)
    elseif (i==4)
        @views diff = (-50*A[end-2,j,k]+864*A[end-1,j,k]-7425*A[end,j,k]+
        44000*A[i-3,j,k]-222750*A[i-2,j,k]+1425600*A[i-1,j,k]-2480478*A[i,j,k]+
        1425600*A[i+1,j,k]-222750*A[i+2,j,k]+44000*A[i+3,j,k]-
        7425*A[i+4,j,k]+864*A[i+5,j,k]-50*A[i+6,j,k])/(831600*dx*dx)
    elseif (i==5)
        @views diff = (-50*A[end-1,j,k]+864*A[end,j,k]-7425*A[i-4,j,k]+
        44000*A[i-3,j,k]-222750*A[i-2,j,k]+1425600*A[i-1,j,k]-2480478*A[i,j,k]+
        1425600*A[i+1,j,k]-222750*A[i+2,j,k]+44000*A[i+3,j,k]-
        7425*A[i+4,j,k]+864*A[i+5,j,k]-50*A[i+6,j,k])/(831600*dx*dx)
    elseif (i==6)
        @views diff = (-50*A[end,j,k]+864*A[i-5,j,k]-7425*A[i-4,j,k]+
        44000*A[i-3,j,k]-222750*A[i-2,j,k]+1425600*A[i-1,j,k]-2480478*A[i,j,k]+
        1425600*A[i+1,j,k]-222750*A[i+2,j,k]+44000*A[i+3,j,k]-
        7425*A[i+4,j,k]+864*A[i+5,j,k]-50*A[i+6,j,k])/(831600*dx*dx)
    end
    return prec(diff)
end

@views function d2fdy2(A,i,j,k,diff,dx)
    if (j>6 && j<(size(A,2)-5))
        @views diff = (-50*A[i,j-6,k]+864*A[i,j-5,k]-7425*A[i,j-4,k]+
        44000*A[i,j-3,k]-222750*A[i,j-2,k]+1425600*A[i,j-1,k]-2480478*A[i,j,k]+
        1425600*A[i,j+1,k]-222750*A[i,j+2,k]+44000*A[i,j+3,k]-
        7425*A[i,j+4,k]+864*A[i,j+5,k]-50*A[i,j+6,k])/(831600*dx*dx)
    elseif (j==(size(A,2)-5))
        @views diff = (-50*A[i,j-6,k]+864*A[i,j-5,k]-7425*A[i,j-4,k]+
        44000*A[i,j-3,k]-222750*A[i,j-2,k]+1425600*A[i,j-1,k]-2480478*A[i,j,k]+
        1425600*A[i,j+1,k]-222750*A[i,j+2,k]+44000*A[i,j+3,k]-
        7425*A[i,j+4,k]+864*A[i,j+5,k]-50*A[i,1,k])/(831600*dx*dx)
    elseif (j==(size(A,2)-4))
        @views diff = (-50*A[i,j-6,k]+864*A[i,j-5,k]-7425*A[i,j-4,k]+
        44000*A[i,j-3,k]-222750*A[i,j-2,k]+1425600*A[i,j-1,k]-2480478*A[i,j,k]+
        1425600*A[i,j+1,k]-222750*A[i,j+2,k]+44000*A[i,j+3,k]-
        7425*A[i,j+4,k]+864*A[i,1,k]-50*A[i,2,k])/(831600*dx*dx)
    elseif (j==(size(A,2)-3))
        @views diff = (-50*A[i,j-6,k]+864*A[i,j-5,k]-7425*A[i,j-4,k]+
        44000*A[i,j-3,k]-222750*A[i,j-2,k]+1425600*A[i,j-1,k]-2480478*A[i,j,k]+
        1425600*A[i,j+1,k]-222750*A[i,j+2,k]+44000*A[i,j+3,k]-
        7425*A[i,1,k]+864*A[i,2,k]-50*A[i,3,k])/(831600*dx*dx)
    elseif (j==(size(A,2)-2))
        @views diff = (-50*A[i,j-6,k]+864*A[i,j-5,k]-7425*A[i,j-4,k]+
        44000*A[i,j-3,k]-222750*A[i,j-2,k]+1425600*A[i,j-1,k]-2480478*A[i,j,k]+
        1425600*A[i,j+1,k]-222750*A[i,j+2,k]+44000*A[i,1,k]-
        7425*A[i,2,k]+864*A[i,3,k]-50*A[i,4,k])/(831600*dx*dx)
    elseif (j==(size(A,2)-1))
        @views diff = (-50*A[i,j-6,k]+864*A[i,j-5,k]-7425*A[i,j-4,k]+
        44000*A[i,j-3,k]-222750*A[i,j-2,k]+1425600*A[i,j-1,k]-2480478*A[i,j,k]+
        1425600*A[i,j+1,k]-222750*A[i,1,k]+44000*A[i,2,k]-
        7425*A[i,3,k]+864*A[i,4,k]-50*A[i,5,k])/(831600*dx*dx)
    elseif (j==(size(A,2)))
        @views diff = (-50*A[i,j-6,k]+864*A[i,j-5,k]-7425*A[i,j-4,k]+
        44000*A[i,j-3,k]-222750*A[i,j-2,k]+1425600*A[i,j-1,k]-2480478*A[i,j,k]+
        1425600*A[i,1,k]-222750*A[i,2,k]+44000*A[i,3,k]-
        7425*A[i,4,k]+864*A[i,5,k]-50*A[i,6,k])/(831600*dx*dx)
    elseif (j==1)
        @views diff = (-50*A[i,end-5,k]+864*A[i,end-4,k]-7425*A[i,end-3,k]+
        44000*A[i,end-2,k]-222750*A[i,end-1,k]+1425600*A[i,end,k]-2480478*A[i,j,k]+
        1425600*A[i,j+1,k]-222750*A[i,j+2,k]+44000*A[i,j+3,k]-
        7425*A[i,j+4,k]+864*A[i,j+5,k]-50*A[i,j+6,k])/(831600*dx*dx)
    elseif (j==2)
        @views diff = (-50*A[i,end-4,k]+864*A[i,end-3,k]-7425*A[i,end-2,k]+
        44000*A[i,end-1,k]-222750*A[i,end,k]+1425600*A[i,j-1,k]-2480478*A[i,j,k]+
        1425600*A[i,j+1,k]-222750*A[i,j+2,k]+44000*A[i,j+3,k]-
        7425*A[i,j+4,k]+864*A[i,j+5,k]-50*A[i,j+6,k])/(831600*dx*dx)
    elseif (j==3)
        @views diff = (-50*A[i,end-3,k]+864*A[i,end-2,k]-7425*A[i,end-1,k]+
        44000*A[i,end,k]-222750*A[i,j-2,k]+1425600*A[i,j-1,k]-2480478*A[i,j,k]+
        1425600*A[i,j+1,k]-222750*A[i,j+2,k]+44000*A[i,j+3,k]-
        7425*A[i,j+4,k]+864*A[i,j+5,k]-50*A[i,j+6,k])/(831600*dx*dx)
    elseif (j==4)
        @views diff = (-50*A[i,end-2,k]+864*A[i,end-1,k]-7425*A[i,end,k]+
        44000*A[i,j-3,k]-222750*A[i,j-2,k]+1425600*A[i,j-1,k]-2480478*A[i,j,k]+
        1425600*A[i,j+1,k]-222750*A[i,j+2,k]+44000*A[i,j+3,k]-
        7425*A[i,j+4,k]+864*A[i,j+5,k]-50*A[i,j+6,k])/(831600*dx*dx)
    elseif (j==5)
        @views diff = (-50*A[i,end-1,k]+864*A[i,end,k]-7425*A[i,j-4,k]+
        44000*A[i,j-3,k]-222750*A[i,j-2,k]+1425600*A[i,j-1,k]-2480478*A[i,j,k]+
        1425600*A[i,j+1,k]-222750*A[i,j+2,k]+44000*A[i,j+3,k]-
        7425*A[i,j+4,k]+864*A[i,j+5,k]-50*A[i,j+6,k])/(831600*dx*dx)
    elseif (j==6)
        @views diff = (-50*A[i,end,k]+864*A[i,j-5,k]-7425*A[i,j-4,k]+
        44000*A[i,j-3,k]-222750*A[i,j-2,k]+1425600*A[i,j-1,k]-2480478*A[i,j,k]+
        1425600*A[i,j+1,k]-222750*A[i,j+2,k]+44000*A[i,j+3,k]-
        7425*A[i,j+4,k]+864*A[i,j+5,k]-50*A[i,j+6,k])/(831600*dx*dx)
    end
    return prec(diff)
end

@views function d2fdz2(A,i,j,k,diff,dx)
    if (k>6 && k<(size(A,3)-5))
        @views diff = (-50*A[i,j,k-6]+864*A[i,j,k-5]-7425*A[i,j,k-4]+
        44000*A[i,j,k-3]-222750*A[i,j,k-2]+1425600*A[i,j,k-1]-2480478*A[i,j,k]+
        1425600*A[i,j,k+1]-222750*A[i,j,k+2]+44000*A[i,j,k+3]-
        7425*A[i,j,k+4]+864*A[i,j,k+5]-50*A[i,j,k+6])/(831600*dx*dx)
    elseif (k==(size(A,3)-5))
        @views diff = (-50*A[i,j,k-6]+864*A[i,j,k-5]-7425*A[i,j,k-4]+
        44000*A[i,j,k-3]-222750*A[i,j,k-2]+1425600*A[i,j,k-1]-2480478*A[i,j,k]+
        1425600*A[i,j,k+1]-222750*A[i,j,k+2]+44000*A[i,j,k+3]-
        7425*A[i,j,k+4]+864*A[i,j,k+5]-50*A[i,j,1])/(831600*dx*dx)
    elseif (k==(size(A,3)-4))
        @views diff = (-50*A[i,j,k-6]+864*A[i,j,k-5]-7425*A[i,j,k-4]+
        44000*A[i,j,k-3]-222750*A[i,j,k-2]+1425600*A[i,j,k-1]-2480478*A[i,j,k]+
        1425600*A[i,j,k+1]-222750*A[i,j,k+2]+44000*A[i,j,k+3]-
        7425*A[i,j,k+4]+864*A[i,j,1]-50*A[i,j,2])/(831600*dx*dx)
    elseif (k==(size(A,3)-3))
        @views diff = (-50*A[i,j,k-6]+864*A[i,j,k-5]-7425*A[i,j,k-4]+
        44000*A[i,j,k-3]-222750*A[i,j,k-2]+1425600*A[i,j,k-1]-2480478*A[i,j,k]+
        1425600*A[i,j,k+1]-222750*A[i,j,k+2]+44000*A[i,j,k+3]-
        7425*A[i,j,1]+864*A[i,j,2]-50*A[i,j,3])/(831600*dx*dx)
    elseif (k==(size(A,3)-2))
        @views diff = (-50*A[i,j,k-6]+864*A[i,j,k-5]-7425*A[i,j,k-4]+
        44000*A[i,j,k-3]-222750*A[i,j,k-2]+1425600*A[i,j,k-1]-2480478*A[i,j,k]+
        1425600*A[i,j,k+1]-222750*A[i,j,k+2]+44000*A[i,j,1]-
        7425*A[i,j,2]+864*A[i,j,3]-50*A[i,j,4])/(831600*dx*dx)
    elseif (k==(size(A,3)-1))
        @views diff = (-50*A[i,j,k-6]+864*A[i,j,k-5]-7425*A[i,j,k-4]+
        44000*A[i,j,k-3]-222750*A[i,j,k-2]+1425600*A[i,j,k-1]-2480478*A[i,j,k]+
        1425600*A[i,j,k+1]-222750*A[i,j,1]+44000*A[i,j,2]-
        7425*A[i,j,3]+864*A[i,j,4]-50*A[i,j,5])/(831600*dx*dx)
    elseif (k==(size(A,3)))
        @views diff = (-50*A[i,j,k-6]+864*A[i,j,k-5]-7425*A[i,j,k-4]+
        44000*A[i,j,k-3]-222750*A[i,j,k-2]+1425600*A[i,j,k-1]-2480478*A[i,j,k]+
        1425600*A[i,j,1]-222750*A[i,j,2]+44000*A[i,j,3]-
        7425*A[i,j,4]+864*A[i,j,5]-50*A[i,j,6])/(831600*dx*dx)
    elseif (k==1)
        @views diff = (-50*A[i,j,end-5]+864*A[i,j,end-4]-7425*A[i,j,end-3]+
        44000*A[i,j,end-2]-222750*A[i,j,end-1]+1425600*A[i,j,end]-2480478*A[i,j,k]+
        1425600*A[i,j,k+1]-222750*A[i,j,k+2]+44000*A[i,j,k+3]-
        7425*A[i,j,k+4]+864*A[i,j,k+5]-50*A[i,j,k+6])/(831600*dx*dx)
    elseif (k==2)
        @views diff = (-50*A[i,j,end-4]+864*A[i,j,end-3]-7425*A[i,j,end-2]+
        44000*A[i,j,end-1]-222750*A[i,j,end]+1425600*A[i,j,k-1]-2480478*A[i,j,k]+
        1425600*A[i,j,k+1]-222750*A[i,j,k+2]+44000*A[i,j,k+3]-
        7425*A[i,j,k+4]+864*A[i,j,k+5]-50*A[i,j,k+6])/(831600*dx*dx)
    elseif (k==3)
        @views diff = (-50*A[i,j,end-3]+864*A[i,j,end-2]-7425*A[i,j,end-1]+
        44000*A[i,j,end]-222750*A[i,j,k-2]+1425600*A[i,j,k-1]-2480478*A[i,j,k]+
        1425600*A[i,j,k+1]-222750*A[i,j,k+2]+44000*A[i,j,k+3]-
        7425*A[i,j,k+4]+864*A[i,j,k+5]-50*A[i,j,k+6])/(831600*dx*dx)
    elseif (k==4)
        @views diff = (-50*A[i,j,end-2]+864*A[i,j,end-1]-7425*A[i,j,end]+
        44000*A[i,j,k-3]-222750*A[i,j,k-2]+1425600*A[i,j,k-1]-2480478*A[i,j,k]+
        1425600*A[i,j,k+1]-222750*A[i,j,k+2]+44000*A[i,j,k+3]-
        7425*A[i,j,k+4]+864*A[i,j,k+5]-50*A[i,j,k+6])/(831600*dx*dx)
    elseif (k==5)
        @views diff = (-50*A[i,j,end-1]+864*A[i,j,end]-7425*A[i,j,k-4]+
        44000*A[i,j,k-3]-222750*A[i,j,k-2]+1425600*A[i,j,k-1]-2480478*A[i,j,k]+
        1425600*A[i,j,k+1]-222750*A[i,j,k+2]+44000*A[i,j,k+3]-
        7425*A[i,j,k+4]+864*A[i,j,k+5]-50*A[i,j,k+6])/(831600*dx*dx)
    elseif (k==6)
        @views diff = (-50*A[i,j,end]+864*A[i,j,k-5]-7425*A[i,j,k-4]+
        44000*A[i,j,k-3]-222750*A[i,j,k-2]+1425600*A[i,j,k-1]-2480478*A[i,j,k]+
        1425600*A[i,j,k+1]-222750*A[i,j,k+2]+44000*A[i,j,k+3]-
        7425*A[i,j,k+4]+864*A[i,j,k+5]-50*A[i,j,k+6])/(831600*dx*dx)
    end
    return prec(diff)
end

#Module end#
end


#############2nd ORDER################################################


module differentiations_2nd_order

export diff_abc
export dfdx,dfdy,dfdz
export d2fdx2,d2fdy2,d2fdz2

@views function dfdx(A,i,j,k,diff,dx)
    if (i>2 && i<(size(A,1)-1))
        @views diff = (-A[i-1,j,k]+A[i+1,j,k])/(2.0*dx)
    elseif (i==(size(A,1)))
        @views diff = (-A[i-1,j,k]+A[1,j,k])/(2.0*dx)
    elseif (i==1)
        @views diff = (-A[end,j,k]+A[i+1,j,k])/(2.0*dx)
    end
    return prec(diff)
end

@views function dfdy(A,i,j,k,diff,dx)
    if (j>2 && j<(size(A,2)-1))
        @views diff = (-A[i,j-1,k]+A[i,j+1,k])/(2.0*dx)
    elseif (j==(size(A,2)))
        @views diff = (-A[i,j-1,k]+A[i,1,k])/(2.0*dx)
    elseif (j==1)
        @views diff = (-A[i,end,k]+A[i,j+1,k])/(2.0*dx)
    end
    return prec(diff)
end

@views function dfdz(A,i,j,k,diff,dx)
    if (k>2 && k<(size(A,3)-1))
        @views diff = (-A[i,j,k-1]+A[i,j,k+1])/(2.0*dx)
    elseif (k==(size(A,3)))
        @views diff = (-A[i,j,k-1]+A[i,j,1])/(2.0*dx)
    elseif (k==1)
        @views diff = (-A[i,j,end]+A[i,j,k+1])/(2.0*dx)
    end
    return prec(diff)
end

@views function d2fdx2(A,i,j,k,diff,dx)
    if (i>2 && i<(size(A,1)-1))
        @views diff = (A[i-1,j,k]-2.0*A[i,j,k]+A[i+1,j,k])/(dx*dx)
    elseif (i==(size(A,1)))
        @views diff = (A[i-1,j,k]-2.0*A[i,j,k]+A[1,j,k])/(dx*dx)
    elseif (i==1)
        @views diff = (A[end,j,k]-2.0*A[i,j,k]+A[i+1,j,k])/(dx*dx)
    end
    return prec(diff)
end

@views function d2fdy2(A,i,j,k,diff,dx)
    if (j>2 && j<(size(A,2)-1))
        @views diff = (A[i,j-1,k]-2.0*A[i,j,k]+A[i,j+1,k])/(dx*dx)
    elseif (j==(size(A,2)))
        @views diff = (A[i,j-1,k]-2.0*A[i,j,k]+A[i,1,k])/(dx*dx)
    elseif (j==1)
        @views diff = (A[i,end,k]-2.0*A[i,j,k]+A[i,j+1,k])/(dx*dx)
    end
    return prec(diff)
end

@views function d2fdz2(A,i,j,k,diff,dx)
    if (k>2 && k<(size(A,3)-1))
        @views diff = (A[i,j,k-1]-2.0*A[i,j,k]+A[i,j,k+1])/(dx*dx)
    elseif (k==(size(A,3)))
        @views diff = (A[i,j,k-1]-2.0*A[i,j,k]+A[i,j,1])/(dx*dx)
    elseif (k==1)
        @views diff = (A[i,j,end]-2.0*A[i,j,k]+A[i,j,k+1])/(dx*dx)
    end
    return prec(diff)
end

#Module end#
end