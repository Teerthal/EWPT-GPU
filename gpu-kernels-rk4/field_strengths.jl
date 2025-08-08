module f_strengths

#W^1 strengths
export W_1_11
function W_1_11()
    return 0.
end

export W_1_12
function W_1_12(dW_1_2_dt)
    return dW_1_2_dt
end

export W_1_13
function W_1_13(dW_1_3_dt)
    return dW_1_3_dt
end

export W_1_14
function W_1_14(dW_1_4_dt)
    return dW_1_4_dt
end

# export W_1_21
# function W_1_21(dW_2_1_dt)
#     return
# end

export W_1_22
function W_1_22(dW_2_2_dt)
    return 0.
end

export W_1_xy
@views function W_1_xy(dW_1_3_dx,dW_1_2_dy,W_2_2,W_3_3,W_2_3,W_3_2,gw,i,j,k)
    return dW_1_3_dx - dW_1_2_dy + gw*(W_2_2[i,j,k]*W_3_3[i,j,k]-W_2_3[i,j,k]*W_3_2[i,j,k])
end

export W_1_xz
@views function W_1_xz(dW_1_4_dx,dW_1_2_dz,W_2_2,W_3_4,W_2_4,W_3_2,gw,i,j,k)
    return dW_1_4_dx - dW_1_2_dz + gw*(W_2_2[i,j,k]*W_3_4[i,j,k]-W_2_4[i,j,k]*W_3_2[i,j,k])
end

# export W_1_31
# function W_1_31(dW_3_1_dt)
#     return
# end

# export W_1_32
# function W_1_32(dW_3_2_dt)
#     return
# end

export W_1_yy
function W_1_yy(dW_3_3_dt)
    return 0.
end

export W_1_yz
@views function W_1_yz(dW_1_4_dy,dW_1_3_dz,W_2_3,W_3_4,W_2_4,W_3_3,gw,i,j,k)
    return dW_1_4_dy - dW_1_3_dz + gw*(W_2_3[i,j,k]*W_3_4[i,j,k]-W_2_4[i,j,k]*W_3_3[i,j,k])
end

# export W_1_41
# function W_1_41(dW_4_1_dt)
#     return
# end

# export W_1_42
# function W_1_42(dW_4_3_dt)
#     return
# end

# export W_1_43
# function W_1_43(dW_4_3_dt)
#     return
# end

export W_1_44
function W_1_44()
    return 0.
end

#W^2 strengths
export W_2_11
function W_2_11()
    return 0.
end

export W_2_12
function W_2_12(dW_2_2_dt)
    return dW_2_2_dt
end

export W_2_13
function W_2_13(dW_2_3_dt)
    return dW_2_3_dt
end

export W_2_14
function W_2_14(dW_2_4_dt)
    return dW_2_4_dt
end

# export W_2_21
# function W_2_21(dW_2_1_dt)
#     return
# end

export W_2_22
function W_2_22()
    return 0.
end

export W_2_xy
@views function W_2_xy(dW_2_3_dx,dW_2_2_dy,W_3_2,W_1_3,W_3_3,W_1_2,gw,i,j,k)
    return dW_2_3_dx - dW_2_2_dy + gw*(W_3_2[i,j,k]*W_1_3[i,j,k]-W_3_3[i,j,k]*W_1_2[i,j,k])
end

export W_2_xz
@views function W_2_xz(dW_2_4_dx,dW_2_2_dz,W_3_2,W_1_4,W_3_4,W_1_2,gw,i,j,k)
    return dW_2_4_dx - dW_2_2_dz + gw*(W_3_2[i,j,k]*W_1_4[i,j,k]-W_3_4[i,j,k]*W_1_2[i,j,k])
end

# export W_2_31
# function W_2_31(dW_3_1_dt)
#     return
# end

# export W_2_32
# function W_2_32(dW_3_2_dt)
#     return
# end

export W_2_33
function W_2_33()
    return 0.
end

export W_2_yz
@views function W_2_yz(dW_2_4_dy,dW_2_3_dz,W_3_3,W_1_4,W_3_4,W_1_3,gw,i,j,k)
    return dW_2_4_dy - dW_2_3_dz + gw*(W_3_3[i,j,k]*W_1_4[i,j,k]-W_3_4[i,j,k]*W_1_3[i,j,k])
end

# export W_2_41
# function W_2_41(dW_4_1_dt)
#     return
# end

# export W_2_42
# function W_2_42(dW_4_3_dt)
#     return
# end

# export W_2_43
# function W_2_43(dW_4_3_dt)
#     return
# end

export W_2_44
function W_2_44()
    return 0.
end

#W^3 strengths
export W_3_11
function W_3_11()
    return 0.
end

export W_3_12
function W_3_12(dW_3_2_dt)
    return dW_3_2_dt
end

export W_3_13
function W_3_13(dW_3_3_dt)
    return dW_3_3_dt
end

export W_3_14
function W_3_14(dW_3_4_dt)
    return dW_3_4_dt
end

# export W_3_21
# function W_3_21(dW_3_1_dt)
#     return
# end

export W_3_22
function W_3_22()
    return 0.
end

export W_3_xy
@views function W_3_xy(dW_3_3_dx,dW_3_2_dy,W_1_2,W_2_3,W_1_3,W_2_2,gw,i,j,k)
    return dW_3_3_dx - dW_3_2_dy + gw*(W_1_2[i,j,k]*W_2_3[i,j,k]-W_1_3[i,j,k]*W_2_2[i,j,k])
end

export W_3_xz
@views function W_3_xz(dW_3_4_dx,dW_3_2_dz,W_1_2,W_2_4,W_1_4,W_2_2,gw,i,j,k)
    return dW_3_4_dx - dW_3_2_dz + gw*(W_1_2[i,j,k]*W_2_4[i,j,k]-W_1_4[i,j,k]*W_2_2[i,j,k])
end

# export W_3_31
# function W_3_31(dW_3_1_dt)
#     return
# end

# export W_3_32
# function W_3_32(dW_3_2_dt)
#     return
# end

export W_3_33
function W_3_33(dW_3_3_dt)
    return 0.
end

export W_3_yz
@views function W_3_yz(dW_3_4_dy,dW_3_3_dz,W_1_3,W_2_4,W_1_4,W_2_3,gw,i,j,k)
    return dW_3_4_dy - dW_3_3_dz + gw*(W_1_3[i,j,k]*W_2_4[i,j,k]-W_1_4[i,j,k]*W_2_3[i,j,k])
end

# export W_3_41
# function W_3_41(dW_4_1_dt)
#     return
# end

# export W_3_42
# function W_3_42(dW_3_3_dt)
#     return
# end

# export W_3_43
# function W_3_43(dW_3_3_dt)
#     return
# end

export W_3_44
function W_3_44()
    return 0.
end


# Y_1 #

export Y_1_1
function Y_1_1()
    return 0.
end

export Y_1_2
function Y_1_2(dY_2_dt)
    return dY_2_dt
end

export Y_1_3
function Y_1_3(dY_3_dt)
    return dY_3_dt
end

export Y_1_4
function Y_1_4(dY_4_dt)
    return dY_4_dt
end

# Y_2 #

# export Y_2_1
# function Y_1_1()
#     return 0.
# end

export Y_2_2
function Y_2_2()
    return 0.
end

export Y_2_y
function Y_2_y(dY_3_dx,dY_2_dy)
    return dY_3_dx - dY_2_dy
end

export Y_2_z
function Y_2_z(dY_4_dx,dY_2_dz)
    return dY_4_dx - dY_2_dz
end

# Y_3 #

# export Y_3_1
# function Y_3_1()
#     return 0.
# end

# export Y_3_2
# function Y_3_2()
#     return 0.
# end

export Y_3_3
function Y_3_3()
    return 0.
end

export Y_3_z
function Y_3_z(dY_4_dy,dY_3_dz)
    return dY_4_dy - dY_3_dz
end

# Y_4 #

# export Y_4_1
# function Y_4_1()
#     return 0.
# end

# export Y_4_2
# function Y_4_2()
#     return 0.
# end

# export Y_4_3
# function Y_4_3()
#     return 0.
# end

export Y_4_4
function Y_4_4()
    return 0.
end


end