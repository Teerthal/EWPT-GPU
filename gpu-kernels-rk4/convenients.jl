module convenience_functions

using StatsBase
using Random

export gen_phi

function gen_phi(inp)
    a,b,c = inp
    alpha = 0.5*acos(sample([-1.0,1.0])*a)
    beta = b*2.0*pi
    gamma = c*2.0*pi
    return [cos(alpha)*exp(1im*beta),sin(alpha)*exp(1im*gamma)]
end


end