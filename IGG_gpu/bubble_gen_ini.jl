module bubbles

##Module to populate Higgs bubbles uniformly in lattice for testing purposes

using StatsBase
using Random
# using Plots

include("parameters.jl")
using .parameters

export bubbler
export gen_phi
function gen_phi(angles)
    alpha = 0.5*acos(sample([-1.0,1.0])*angles[1])
    beta = angles[2]*2.0*pi
    gamma = angles[3]*2.0*pi
    return [cos(alpha)*exp(1im*beta),sin(alpha)*exp(1im*gamma)]
end

function bubbler(phi_arr)
    bubs = []
    for kb in range(bub_diam,stop=latz,step=bub_diam)
        for jb in range(bub_diam,stop=laty,step=bub_diam)
            for ib in range(bub_diam,stop=latx,step=bub_diam)
                phi=gen_phi()
                # f[ib,jb,kb,1] = real(phi[1])
                # f[ib,jb,kb,2] = imag(phi[1])
                # f[ib,jb,kb,3] = real(phi[2])
                # f[ib,jb,kb,4] = imag(phi[2])
                # println(f[ib,jb,kb,1]^2+f[ib,jb,kb,2]^2+f[ib,jb,kb,3]^2+f[ib,jb,kb,4]^2)
                # if i==latx
                #     f[i,j,k,:]=f[1,j,k,:]
                # end
                # if j==laty
                #     f[i,j,k,:]=f[i,1,k,:]
                # end
                # if k==latz
                #     f[i,j,k,:]=f[i,j,1,:]
                # end
                push!(bubs,[ib,jb,kb,real(phi[1]),imag(phi[1]),real(phi[2]),imag(phi[2])])
            end
        end
    end
    # println(bubs)

    println("# bubbles: ",size(bubs,1))
    Len = (latx-1)*dx
    rk = pi/Len
    Threads.@threads for k in range(1,stop=latz,step=1)
        for j in range(1,stop=laty,step=1)
            for i in range(1,stop=latx,step=1)
                
                for b in range(1,size(bubs,1),step=1)
                    # println(bubs[b])
                    ib,jb,kb,p1,p2,p3,p4 = bubs[b]
                    
                    # rb=(1.0/rk)*sqrt(sin(rk*((i-latx/2)-(ib-0.5))*dx)^2+
                    # sin(rk*((j-laty/2)-(jb-0.5))*dx)^2+
                    # sin(rk*((k-latz/2)-(kb-0.5))*dx)^2)
                    # rb=(1.0/rk)*sqrt(sin(rk*((i-1)-(ib-0.5))*dx)^2+
                    # sin(rk*((j-1)-(jb-0.5))*dx)^2+
                    # sin(rk*((k-1)-(kb-0.5))*dx)^2)
                    # rmag=vev*(1.0+(sqrt(2.0)-1.0)^2)*exp(-mH*rb/sqrt(2.0))/(1.0+((sqrt(2.0)-1.0)^2)*exp(-sqrt(2.0)*mH*rb))
                    
                    phi_arr[i,j,k,1]=phi_arr[i,j,k,1]#+rmag*p1
                    phi_arr[i,j,k,2]=phi_arr[i,j,k,2]#+rmag*p2
                    phi_arr[i,j,k,3]=phi_arr[i,j,k,3]#+rmag*p3
                    phi_arr[i,j,k,4]=phi_arr[i,j,k,4]#+rmag*p4
                    # ϕ_1[i,j,k]=rmag*p1
                    # ϕ_2[i,j,k]=rmag*p2
                    # ϕ_3[i,j,k]=rmag*p3
                    # ϕ_4[i,j,k]=rmag*p4
                    
                    # println(ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2);exit()
                    # if (ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2)>1
                    #     println(ϕ_1[i,j,k]^2+ϕ_2[i,j,k]^2+ϕ_3[i,j,k]^2+ϕ_4[i,j,k]^2)
                    #     println(rb)
                    #     println(rmag)
                    #     println(f[i,j,k,1])
                    #     println(f[i,j,k,2])
                    #     println(f[i,j,k,3])
                    #     println(f[i,j,k,4])
                    #     exit()
                    # end
                end

            end
        end
    end

    # gr()
    # ENV["GKSwstype"]="nul"
    # x=z=range(1,latx,step=1)
    # println(maximum(ϕ_1))
    # println(maximum(vec(ϕ_1.*ϕ_1+ϕ_2.*ϕ_2+ϕ_3.*ϕ_3+ϕ_4.*ϕ_4)))
    # plt=contourf(z,x,ϕ_1[:,bub_diam,:].*ϕ_1[:,bub_diam,:]+ϕ_2[:,bub_diam,:].*ϕ_2[:,bub_diam,:]+ϕ_3[:,bub_diam,:].*ϕ_3[:,bub_diam,:]+ϕ_4[:,bub_diam,:].*ϕ_4[:,bub_diam,:])
    # png(string("testini1",".png"))
    # exit()
    return phi_arr
end

end