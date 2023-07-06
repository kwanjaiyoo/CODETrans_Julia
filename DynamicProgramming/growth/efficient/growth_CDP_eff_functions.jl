# This file contains all functions used in 'growth_CDP_eff.jl'


function bmvipi(me, vime, PAR, DSS, SSK, SSS, TM)
    # me = 1 corresponds to value function interpolation
    # me = 0 corresponds to policy function interpolation
    # vime indicates the method used in value function interpolation with
    # vime = 0 referring to discrete maximization
    # vime = 1 referring to sophiscated optimization (fminbnd alike)
    
    α = PAR[1];
    β = PAR[2];
    δ = PAR[3];

end