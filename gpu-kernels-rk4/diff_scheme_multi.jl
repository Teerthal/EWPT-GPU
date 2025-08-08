module differentiations

export diff_abc
export dfdx,dfdy,dfdz
export d2fdx2,d2fdy2,d2fdz2

@views function dfdx(A,i,j,k,diff,dx)
    @views diff = (A[i+3,j,k]-9. *A[i+2,j,k]+45. *A[i+1,j,k]-
        45. *A[i-1,j,k]+9. *A[i-2,j,k]-A[i-3,j,k])/(60*dx)
    return diff
end

@views function dfdy(A,i,j,k,diff,dx)
    @views diff = (A[i,j+3,k]-9. *A[i,j+2,k]+45. *A[i,j+1,k]-
        45. *A[i,j-1,k]+9. *A[i,j-2,k]-A[i,j-3,k])/(60*dx)
    return diff
end

@views function dfdz(A,i,j,k,diff,dx)
    @views diff = (A[i,j,k+3]-9. *A[i,j,k+2]+45. *A[i,j,k+1]-
        45. *A[i,j,k-1]+9. *A[i,j,k-2]-A[i,j,k-3])/(60*dx)
    return diff
end

@views function d2fdx2(A,i,j,k,diff,dx)
    @views diff = (A[i+3,j,k]/90. -3. *A[i+2,j,k]/20. +3. *A[i+1,j,k]/2. -
        49. *A[i,j,k]/18. +3. *A[i-1,j,k]/2. -3. *A[i-2,j,k]/20. +A[i-3,j,k]/90.)/(dx*dx)
    return diff
end

@views function d2fdy2(A,i,j,k,diff,dx)
    @views diff = (A[i,j+3,k]/90. -3. *A[i,j+2,k]/20. +3. *A[i,j+1,k]/2. -
        49. *A[i,j,k]/18. +3. *A[i,j-1,k]/2. -3. *A[i,j-2,k]/20. +A[i,j-3,k]/90.)/(dx*dx)
    return diff
end

@views function d2fdz2(A,i,j,k,diff,dx)
    @views diff = (A[i,j,k+3]/90. -3. *A[i,j,k+2]/20. +3. *A[i,j,k+1]/2. -
        49. *A[i,j,k]/18. +3. *A[i,j,k-1]/2. -3. *A[i,j,k-2]/20. +A[i,j,k-3]/90.)/(dx*dx)
    return diff
end


#Module end#
end