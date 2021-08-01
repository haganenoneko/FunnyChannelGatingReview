using Random, Distributions

Random.seed!(1234)

description = "Reduced four-state model (C_A, O_R, O_A)"

# indices of states below are relative to the reduced, 3-state model. 
# open index 
oi = 2 
# activated states 
ai = [1,3] 
# indices to save when solving 
states_to_save = [2, 3] 
# parameter names 
parnames = ["a0", "sa", "b0", "sb", "c0", "sc", "d0", "sd", "g1", "g2", "h1", "h2"]

function db_pars(pars::AbstractVector{}, db; fmt="x")::Vector{}
    #pars are ln-transformed 
    if db == "x"
        a0, sa, b0, sb, c0, sc, d0, sd, g1, g2, h1, h2 = exp.(pars)
    else 
        if db in ["g1", "h1", "g2", "h2"]
            a0, sa, b0, sb, c0, sc, d0, sd = exp.(pars[1:8])
            
            if db == "g1"
                g2, h1, h2 = exp.(pars[9:end])
                g1 = (h1*b0*g2*c0)/(d0*h2*a0)
            elseif db == "h1"
                g1, g2, h2 = exp.(pars[9:end])
                h1 = (g1*d0*h2*a0)/(b0*g2*c0)
            elseif db == "g2"
                g1, h1, h2 = exp.(pars[9:end])
                g2 = (g1*d0*h2*a0)/(h1*b0*c0)
            elseif db == "h2"
                g1, g2, h1 = exp.(pars[9:end])
                h2 = (g2*c0*h1*b0)/(a0*g1*d0)
            end 
            
        else 
            if db == "c"
                a0, sa, b0, sb, d0, sd, g1, g2, h1, h2 = exp.(pars)
                c0 = (a0*d0*g1*h2)/(b0*g2*h1)
                sc = 1/( (1/sb) - (1/sd) + (1/sa) )

            elseif db == "b"
                a0, sa, c0, sc, d0, sd, g1, g2, h1, h2 = exp.(pars)
                b0 = (a0*g1*d0*h2)/(g2*c0*h1)
                sb = 1/( (1/sc) + (1/sd) - (1/sa) )
            elseif db == "d"
                a0, sa, b0, sb, c0, sc, g1, g2, h1, h2 = exp.(pars)
                d0 = (b0*g2*c0*h1)/(a0*g1*h2) 
                sd = 1/( (1/sb) + (1/sa) - (1/sc) )
            elseif db == "a"
                b0, sb, c0, sc, d0, sd, g1, g2, h1, h2 = exp.(pars)
                a0 = (b0*g2*c0*h1)/(g1*d0*h2)   
                sa = 1/( (1/sc) - (1/sb) + (1/sd) )
            else 
                error("   $db not understood as detailed balance condition.")
            end 
        end 
    end

    if fmt == "log"
        return log.([a0, sa, b0, sb, c0, sc, d0, sd, g1, g2, h1, h2]) 
    else
        return [a0, sa, b0, sb, c0, sc, d0, sd, g1, g2, h1, h2]
    end 
end 

function tm(v, p::AbstractVector{}, db)
    a0, sa, b0, sb, c0, sc, d0, sd, g1, g2, h1, h2 = db_pars(p, db)

    # hyperpolarization-activated
    a1 = a0*exp(-v/sa)
    b1 = b0*exp(v/sb)
    a2 = c0*exp(-v/sc)
    b2 = d0*exp(v/sd)

    Q = [a1  -(b1 + g1 + a1)  -a1  h1-a1; 
        g2  -g2  -(a2+h2+g2)  b2-g2;
        0  g1  a2  -(b2 + h1)]

    return Q 
end 

function get_vpars(v, p::AbstractVector{}, db)
    a0, sa, b0, sb, c0, sc, d0, sd, g1, g2, h1, h2 = db_pars(p, db)

    a1 = a0*exp(-v/sa)
    b1 = b0*exp(v/sb)
    a2 = c0*exp(-v/sc)
    b2 = d0*exp(v/sd)

    return [a1, b1, a2, b2, g1, g2, h1, h2, v]
end 

function rhs!(du, u, Q, t)
    du .= Q[:,2:end] * u + Q[:,1] 
end 


function prepare_BoundsAndParams(pars::Array{Float64,1}, bounds, db)
    # remove entries from parameter and bound arrays given detailed balance 
    if db == true || db == "x" 
        return pars, bounds 
    else
        ps = copy(pars)
        bds = copy(bounds)
        if db == "a" 
            deleteat!(ps, 1:2)
            deleteat!(bds, 1:2)
        elseif db == "b"
            deleteat!(ps, 3:4)
            deleteat!(bds, 3:4)
        elseif db == "c" 
            deleteat!(ps, 5:6)
            deleteat!(bds, 5:6)
        elseif db == "d" 
            deleteat!(ps, 7:8)
            deleteat!(bds, 7:8)
        else 
            println("$db not understood. Available are: 'a', 'b', 'c', 'd'.")
        end 
    end 
    return ps, bds 
end 

bds = [
        (1e-9, 1), #b0
        (7, 200), #sb
        
        (1e-9, 1e3), #a0
        (10, 4000), #sa                
        
        (1e-12, 1e-1), #d0
        (50, 200), #sd 
        
        (1e-12, 1e-1), #c0
        (10, 200), #sc,  

        (1e-9, 10.), #g1
        (1e-8, 10), #g2 

        (1e-9, 10), #h1
        (1e-9, 1e3), #h2
]

hnc2_bds = [
        (1e-7, 1), #a0
        (8, 300), #sa                
    
        (20, 1000), #b0
        (25, 500), #sb

        (1e-8, 1), #c0
        (10, 100), #sc,  
        
        (1e-8, 1), #d0
        (30, 100), #sd 

        (1e-6, 1.), #g1
        (1e-12, 1e-3), #g2 

        (1e-9, 1e-1), #h1
        (1e-9, 1e-1), #h2

]

ln_par_bds = [log.(x) for x in hnc2_bds] 

