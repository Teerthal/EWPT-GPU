module covariant_derivatives

export D_1ϕ_1
@views function D_1ϕ_1(dϕ_1_dt,ϕ_2,ϕ_3,ϕ_4,
    W_1_1,W_2_1,W_3_1,Y_1,gw,gy)
    return (dϕ_1_dt+0.5*gw*((W_1_1)*(ϕ_4)-
    (W_2_1)*(ϕ_3)+
    (W_3_1)*(ϕ_2))+
    0.5*gy*((Y_1)*(ϕ_2)))
end

export D_1ϕ_2
@views function D_1ϕ_2(dϕ_2_dt,ϕ_1,ϕ_3,ϕ_4,
    W_1_1,W_2_1,W_3_1,Y_1,gw,gy)
    return (dϕ_2_dt-0.5*gw*((W_1_1)*(ϕ_3)+
    (W_2_1)*(ϕ_4)+
    (W_3_1)*(ϕ_1))-
    0.5*gy*((Y_1)*(ϕ_1)))
end

export D_1ϕ_3
@views function D_1ϕ_3(dϕ_3_dt,ϕ_1,ϕ_2,ϕ_4,
    W_1_1,W_2_1,W_3_1,Y_1,gw,gy)
    return (dϕ_3_dt+0.5*gw*((W_1_1)*(ϕ_2)+
    (W_2_1)*(ϕ_1)-
    (W_3_1)*(ϕ_4))+
    0.5*gy*((Y_1)*(ϕ_4)))
end

export D_1ϕ_4
@views function D_1ϕ_4(dϕ_4_dt,ϕ_1,ϕ_2,ϕ_3,
    W_1_1,W_2_1,W_3_1,Y_1,gw,gy)
    return (dϕ_4_dt-0.5*gw*((W_1_1)*(ϕ_1)-
    (W_2_1)*(ϕ_2)-
    (W_3_1)*(ϕ_3))-
    0.5*gy*((Y_1)*(ϕ_3)))
end

export D_2ϕ_1
@views function D_2ϕ_1(dϕ_1_dx,ϕ_2,ϕ_3,ϕ_4,
    W_1_2,W_2_2,W_3_2,Y_2,gw,gy)
    return (dϕ_1_dx+0.5*gw*((W_1_2)*(ϕ_4)-
    (W_2_2)*(ϕ_3)+
    (W_3_2)*(ϕ_2))+
    0.5*gy*((Y_2)*(ϕ_2)))
end

export D_2ϕ_2
@views function D_2ϕ_2(dϕ_2_dx,ϕ_1,ϕ_3,ϕ_4,
    W_1_2,W_2_2,W_3_2,Y_2,gw,gy)
    return (dϕ_2_dx-0.5*gw*((W_1_2)*(ϕ_3)+
    (W_2_2)*(ϕ_4)+
    (W_3_2)*(ϕ_1))-
    0.5*gy*((Y_2)*(ϕ_1)))
end

export D_2ϕ_3
@views function D_2ϕ_3(dϕ_3_dx,ϕ_1,ϕ_2,ϕ_4,
    W_1_2,W_2_2,W_3_2,Y_2,gw,gy)
    return (dϕ_3_dx+0.5*gw*((W_1_2)*(ϕ_2)+
    (W_2_2)*(ϕ_1)-
    (W_3_2)*(ϕ_4))+
    0.5*gy*((Y_2)*(ϕ_4)))
end

export D_2ϕ_4
@views function D_2ϕ_4(dϕ_4_dx,ϕ_1,ϕ_2,ϕ_3,
    W_1_2,W_2_2,W_3_2,Y_2,gw,gy)
    return (dϕ_4_dx-0.5*gw*((W_1_2)*(ϕ_1)-
    (W_2_2)*(ϕ_2)-
    (W_3_2)*(ϕ_3))-
    0.5*gy*((Y_2)*(ϕ_3)))
end

export D_3ϕ_1
@views function D_3ϕ_1(dϕ_1_dy,ϕ_2,ϕ_3,ϕ_4,
    W_1_3,W_2_3,W_3_3,Y_3,gw,gy)
    return (dϕ_1_dy+0.5*gw*((W_1_3)*(ϕ_4)-
    (W_2_3)*(ϕ_3)+
    (W_3_3)*(ϕ_2))+
    0.5*gy*((Y_3)*(ϕ_2)))
end

export D_3ϕ_2
@views function D_3ϕ_2(dϕ_2_dy,ϕ_1,ϕ_3,ϕ_4,
    W_1_3,W_2_3,W_3_3,Y_3,gw,gy)
    return (dϕ_2_dy-0.5*gw*((W_1_3)*(ϕ_3)+
    (W_2_3)*(ϕ_4)+
    (W_3_3)*(ϕ_1))-
    0.5*gy*((Y_3)*(ϕ_1)))
end

export D_3ϕ_3
@views function D_3ϕ_3(dϕ_3_dy,ϕ_1,ϕ_2,ϕ_4,
    W_1_3,W_2_3,W_3_3,Y_3,gw,gy)
    return (dϕ_3_dy+0.5*gw*((W_1_3)*(ϕ_2)+
    (W_2_3)*(ϕ_1)-
    (W_3_3)*(ϕ_4))+
    0.5*gy*((Y_3)*(ϕ_4)))
end

export D_3ϕ_4
@views function D_3ϕ_4(dϕ_4_dy,ϕ_1,ϕ_2,ϕ_3,
    W_1_3,W_2_3,W_3_3,Y_3,gw,gy)
    return (dϕ_4_dy-0.5*gw*((W_1_3)*(ϕ_1)-
    (W_2_3)*(ϕ_2)-
    (W_3_3)*(ϕ_3))-
    0.5*gy*((Y_3)*(ϕ_3)))
end

export D_4ϕ_1
@views function D_4ϕ_1(dϕ_1_dz,ϕ_2,ϕ_3,ϕ_4,
    W_1_4,W_2_4,W_3_4,Y_4,gw,gy)
    return (dϕ_1_dz+0.5*gw*((W_1_4)*(ϕ_4)-
    (W_2_4)*(ϕ_3)+
    (W_3_4)*(ϕ_2))+
    0.5*gy*((Y_4)*(ϕ_2)))
end

export D_4ϕ_2
@views function D_4ϕ_2(dϕ_2_dz,ϕ_1,ϕ_3,ϕ_4,
    W_1_4,W_2_4,W_3_4,Y_4,gw,gy)
    return (dϕ_2_dz-0.5*gw*((W_1_4)*(ϕ_3)+
    (W_2_4)*(ϕ_4)+
    (W_3_4)*(ϕ_1))-
    0.5*gy*((Y_4)*(ϕ_1)))
end

export D_4ϕ_3
@views function D_4ϕ_3(dϕ_3_dz,ϕ_1,ϕ_2,ϕ_4,
    W_1_4,W_2_4,W_3_4,Y_4,gw,gy)
    return (dϕ_3_dz+0.5*gw*((W_1_4)*(ϕ_2)+
    (W_2_4)*(ϕ_1)-
    (W_3_4)*(ϕ_4))+
    0.5*gy*((Y_4)*(ϕ_4)))
end

export D_4ϕ_4
@views function D_4ϕ_4(dϕ_4_dz,ϕ_1,ϕ_2,ϕ_3,
    W_1_4,W_2_4,W_3_4,Y_4,gw,gy)
    return (dϕ_4_dz-0.5*gw*((W_1_4)*(ϕ_1)-
    (W_2_4)*(ϕ_2)-
    (W_3_4)*(ϕ_3))-
    0.5*gy*((Y_4)*(ϕ_3)))
end

end