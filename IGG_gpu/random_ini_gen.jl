#Intiializing Module to generate bubbles randomly in lattice
#Aug 23: Current working version is directly implemented in the script
#and module needs to be updated.

module randomizer

using StatsBase
using Random

include("parameters.jl")
using .parameters

export random_gen

function gen_phi()
    alpha = 0.5*acos(sample([-1.0,1.0])*rand(Float64))
    beta = rand(Float64)*2.0*pi
    gamma = rand(Float64)*2.0*pi
    return [cos(alpha)*exp(1im*beta),sin(alpha)*exp(1im*gamma)]
end

function random_gen(f)

    for k in range(1,stop=latz,step=1)
        for j in range(1,stop=laty,step=1)
            for i in range(1,stop=latx,step=1)

                phi=gen_phi()
                f[i,j,k,1] = real(phi[1])
                f[i,j,k,2] = imag(phi[1])
                f[i,j,k,3] = real(phi[2])
                f[i,j,k,4] = imag(phi[2])

                if i==latx
                    f[i,j,k,:]=f[1,j,k,:]
                end
                if j==laty
                    f[i,j,k,:]=f[i,1,k,:]
                end
                if k==latz
                    f[i,j,k,:]=f[i,j,1,:]
                end

            end
        end
    end
    return f
end

end