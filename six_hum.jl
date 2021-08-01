using Random 
using Distributions

Random.seed!(1234)
description = "Hummert six-state model without conserved charge."

#F/RT 
ξ = (96.485332)/(298*8.314462) 

#index after which all states are open 
oi = 3
# indices of activated states 
ai = [1, 2, 4, 5]

function db_pars(pars::AbstractVector{}, db; fmt="x")::Vector{}
    """
    Uses OR -> OA (a2; c0, sc) and OA-OH (a3; e0, se) transitions to set detailed balance 
        (minimum spanning tree method described by Colquhoun)
    pars = parameters 
    DB_CONDITION = uses above method to set detailed balance, unless `x`
    """
    if db == "x"
        a10, b10, s1, a20, b20, s2, a30, b30, s3, a40, b40, s4, g1, g2, h1, h2 = exp.(pars) 
    elseif db == "h2"
        a10, b10, s1, a20, b20, s2, a30, b30, s3, a40, b40, s4, g1, g2, h1  = exp.(pars)
        h2 = (g2*b40*b30*h1*a10*a20)/(a40*a30*g1*b10*b20)
    elseif db == "a3"
        a10, b10, s1, a20, b20, s2, b30, a40, b40, s4, g1, g2, h1, h2  = exp.(pars)
        a30 = (a10*a20*g2*b40*b30*h1)/(a40*h2*b20*b10*h1)        
        s3 = 1/((1/s1) + (1/s2) - (1/s4))
    else
        println("   `db` not understood. Current options are 'h2' and 'a3'")
    end 
    
    if fmt == "log"
        return log.([a10, b10, s1, a20, b20, s2, a30, b30, s3, a40, b40, s4, g1, g2, h1, h2])
    else
        return [a10, b10, s1, a20, b20, s2, a30, b30, s3, a40, b40, s4, g1, g2, h1, h2]
    end 
end 

function get_vdep_pars(p::AbstractVector{}, v)
    a1, a2, a3, a4 = [p[i]*exp(-v/p[i+2]) for i = 1:3:10]
    b1, b2, b3, b4 = [p[i]*exp(v/p[i+1]) for i = 2:3:11]

    return [a1, b1, a2, b2, a3, b3, a4, b4]
end 

function tm(v, p::AbstractVector{}, db)
    a10, b10, s1, a20, b20, s2, a30, b30, s3, a40, b40, s4, g1, g2, h1, h2 = db_pars(p, db)

    a1 = a10*exp(-v/s1)
    b1 = b10*exp(v/s1)
    a3 = a30*exp(-v/s2)
    b3 = b30*exp(v/s2)

    a2 = a20*exp(-v/s3)
    b2 = b20*exp(v/s3)
    a4 = a40*exp(-v/s4)
    b4 = b40*exp(v/s4)    

    Q = [
        a1 -(b1+a2+a1) (b2-a1) -a1 -a1 -a1;
        0 a2 -(b2+g2) 0 0 h2;
        g1 -g1 -g1 -(h1+a3+g1) (b3-g1) -g1;
        0 0 0 a3 -(b3+a4) b4;
        0 0 g2 0 a4 -(h2+b4);
    ]
    return Q 
end 

function rhs!(du, u, Q, t)
    du .= Q[:,2:end] * u + Q[:,1] 
end 
function prepare_BoundsAndParams(pars::Array{Float64,1}, bounds, db)
    # remove entries from parameter and bound arrays given detailed balance 
    if db == true || db == "x" 
        return pars, bounds 
    else
        if DB_CONDITION == "h2"
            deleteat!(pars, 16)
            deleteat!(bounds, 16)
        elseif DB_CONDITION == "a3" 
            deleteat!(pars, [7, 9])
            deleteat!(bounds, [7, 9])
        end 
    end  
    return pars, bounds 
end 

bds = [(1e-10, 1e3), (1e-10, 1e3), (6, 1e3), #a1/b1 
        (1e-10, 1e3), (1e-10, 1e3), (6, 1e3), 
        (1e-10, 1e3), (1e-10, 1e3), (6, 1e3), #a3/b3 
        (1e-10, 1e3), (1e-10, 1e3), (6, 1e3), 
        (1e-5, 10), 
        (1e-8, 10), #g1
        (1e-4, 10),
        (1e-8, 1e-1), #h1 
]

ln_par_bds = [log.(x) for x in bds] 

p0 = [-1.366795614121635, -4.731276419445293, 4.146181643448856, 
        -3.0831331501017187, 2.3286672592590754, 2.9630705986068224, 
        -3.698762226164891, 1.5730159834565232, 4.15, 
        1.998606200149607, 0.8069046570541617, 2.96,
        -2.097864956067902, -6.917709307586828, -5.092722946312469, -4.884952245097995] 