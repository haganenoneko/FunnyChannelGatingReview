#use transition matrix defined in each model file, but general simulation method is here 
#specific for fitting figs. 4a and 4b from Ramentol et al., 2020

using DifferentialEquations, OrdinaryDiffEq
using Random, LinearAlgebra, Statistics
using GCMAES, BlackBoxOptim
using Zygote, ForwardDiff
using Optim, NLSolversBase, AdvancedHMC
using Glob, CSV, DataFrames, Dates
using Plots 

#convenience method to fit data
model_name = "four_3s_ramentol.jl"
include("C:\\Users\\delbe\\Downloads\\wut\\wut\\Post_grad\\UBC\\Research\\lab\\Github_repos\\hcn-gating-kinetics\\src\\models\\$model_name")
println("Using model $model_name")

#print description of model 
println(description)

### Conditions
DB_CONDITION = "c"     
# BB for BlackBoxOptim, GCMAES, or Optim (not implemented yet)       
OPT_OR_NOT = "GCMAES"       
FA_OR_WT = "mut" 
OPT_MAX_ITER = 800

# holding voltage for activation `va` and deactivation `vd`
if FA_OR_WT == "mut"
    vhold_a = -80 
    vhold_d = -80
else
    vhold_a = -10
    vhold_d = 40 
end 

SAVE_FIGS = false
if SAVE_FIGS == true 
    SAVE_NAME = "wt-d1"
end 

#open current time course data as dataframe 
function reader(f)    
    filename = string(basename(f))[1:end-4]
    df = CSV.read(f, DataFrame)

    #rename voltage(mV) in gv/fv df to Voltage 
    if Symbol("Voltage(mV)") in names(df)
        rename!(df, Symbol("Voltage(mV)") => :Voltage)
    end 
    
    #HOW TO INDEX DF WITH CONDITIONS:
    # df[df.Voltage .== -100, 2][1]
    # Note [1]-index at the end; this is needed b/c indexing a df returns an array, even if of size 1. 

    return df 
end

################################################# 
"""LOAD DATA FILES"""
#################################################
#data path 
data_path = ".\\data\\ramentol_digitized\\"

if FA_OR_WT == "mut"
    ss_path = string(data_path, "mut_ss.csv")
    ia_path = string(data_path, "mut_it_act.csv")
    id_path = string(data_path, "mut_it_de.csv")
    fa_path = string(data_path, "mut_ft_act.csv")
    fd_path = string(data_path, "mut_ft_de.csv")
else
    ss_path = string(data_path, "wt_ss.csv")
    ia_path = string(data_path, "wt_it_act.csv")
    id_path = string(data_path, "wt_it_de.csv")
    fa_path = string(data_path, "wt_ft_act.csv")
    fd_path = string(data_path, "wt_ft_de.csv")
end 

files = [glob(x)[1] for x in [ss_path, ia_path, id_path, fa_path, fd_path]] 
# println(files)
# asdf 

data_files = [reader(x) for x in files] 
data_files[2:end] = [x[:, 2:end] for x in data_files[2:end]] 
# println(data_files[1])

#Select detailed balance parameter 
if DB_CONDITION == true && "four" in model_name
    DB_CONDITION = 'c' 
elseif DB_CONDITION == true && model_name == "six_v4.jl"
    DB_CONDITION = true 
end 

println("Detailed balance using $DB_CONDITION")

# extract voltage, times from data 
function extract_VoltageAndTimes(data::Array{DataFrame, 1}; 
    convert_to_ms=false, convert_to_s=false)
    """
    Assume there are 2 datasets 
    data = Array of dataframes 
    convert_to_ms = whether to convert time in data from s to ms
    """
    # number of dataframes, i.e. datasets 
    Nd = length(data) 

    voltages = Array{Array{Float64,1}}(undef, Nd)
    times = Array{Array{Array{Float64,1}}}(undef, Nd)
    trace = Array{Array{Array{Float64,1}}}(undef, Nd)

    for i = 1:Nd
        colnames = names(data[i])
        # println(data[i][end-5:end,1:6])
        # times and voltage columns are present as: "-150_t, -150_i, etc.", so we remove the last two characters 
        # since the columns are paired, we take every second column for time or voltage 
        v = [parse(Float64, String(n)[1:end-2]) for n in colnames[1:2:end]]
        # println(typeof(v))
        voltages[i] = v 

        # number of traces is half the number of columns, since we have time columns for each trace 
        Ntraces = Int(length(colnames)/2)
        tᵢ = Array{Array{Float64,1}}(undef, Ntraces)
        yᵢ = Array{Array{Float64,1}}(undef, Ntraces)

        for j = 1:2:length(colnames)
            
            if convert_to_ms == true 
                data[i][:,j] .*= 1000 
            elseif convert_to_s == true 
                data[i][:,j] .*= 1e-3 
            end 

            tⱼ = collect( skipmissing( data[i][:,j] ) ) 
            # tⱼ = (tⱼ .- tⱼ[1]) .* 1000
            tⱼ = sort(abs.(tⱼ .- tⱼ[1]))
            # println(typeof(tⱼ))

            u = Int((j+1)/2)
            tᵢ[u] = tⱼ
            yᵢ[u] = collect(skipmissing(data[i][:,j+1]))
            if length(yᵢ[u]) < 1 
                println(data[i][:,j:j+1])
            end 
        end 

        times[i] = tᵢ
        trace[i] = yᵢ
    end 

    # indices of voltages in 1st dataset that are also in the second 
    shared_idx = findall(in(voltages[1]), voltages[2])
    
    return Dict("voltages" => voltages, "times" => times, "shared_idx" => shared_idx, "traces" => trace)
end 

# voltage, times, and shared indices for current-time 
extracted_i = extract_VoltageAndTimes(data_files[2:3])
extracted_f = extract_VoltageAndTimes(data_files[4:5])
# println(extracted_f["voltages"], "\n", extracted_f["shared_idx"])
# println(extracted_i["voltages"], "\n", extracted_i["shared_idx"])
# asdf

#################################################
"""SIMULATION METHODS""" 
#################################################
function moving_average(vs::Vector,n::Int) 
    return [sum(@view vs[i:(i+n-1)])/n for i in 1:(length(vs)-(n-1))]
end 

function get_ss(p::AbstractVector{}, vhold, db)::Vector{}
    Q = tm(vhold, p, db)
    return Q[:,2:end] \ (-1 .* Q[:,1])
end 

function apply_norm(arr, peak)
    for i in 1:length(arr)
        arr[i] = [y ./ peak for y in arr[i]]        
    end 
    return arr 
end 

function norm(y_act, y_de; peak_min=0.95)
    peak = maximum([maximum(y) for y in y_act])

    if peak > peak_min
        for i = 1:length(y_act) 
            y_act[i] ./= peak 
        end 
        for i = 1:length(y_de) 
            y_de[i] ./= peak 
        end 
    end 
    
    return y_act, y_de 
end 

function solve_odes(inits, Q, 
                    times::Array{Float64,1})
    # println(inits)
    prob = ODEProblem(rhs!, eltype(Q).(inits), (times[1], times[end]), Q)
    
    sol = solve(prob, TRBDF2(), 
            saveat = times,
            reltol=1e-6, abstol=1e-6, maxiters=1e6)

    return sol 
end 

function one_sim(pars::AbstractVector{T}, data, 
                u0::AbstractVector{}, Qhold, db; 
                pmax_min=0.7, states="open", normalize=false) where T 

    va, vd = data["voltages"]
    ta, td = data["times"]
    shared = data["shared_idx"]

    # y_act = Array{Array{Float64, 2}, 1}(undef, length(va))
    y_act = Array{Array{T, 1}, 1}(undef, length(va))
    y_de = Array{Array{T, 1}, 1}(undef, length(vd))

    j = 0 
    for i in 1:length(va)
        Qa = tm(va[i], pars, db)
        sol = solve_odes(u0, Qa, ta[i])
        
        if states == "open"
            y_act[i] = vec(sum(sol[oi:end,:], dims=1))
        else 
            y_act[i] = vec(sum(sol[ai,:], dims=1))
        end 

        if va[i] in vd 
            j += 1 
            h = shared[j] 

            # deactivation at `vhold_d`
            Qd = tm(vhold_d, pars, db)
            sol = solve_odes(sol[end], Qd, td[h])

            if states == "open"
                y_de[h] = vec(sum(sol[oi:end,:], dims=1))
            else                 
                y_de[h] = vec(sum(sol[ai,:], dims=1))
            end 
        end 
    end 
    
    if normalize == true 
        if states == "active"
            # subtract the minimum, as that seems to be have been done to mutant FV 
            min_f = minimum([minimum(y) for y in y_act])
            # but, require that minimum activated is <= 2% 
            if min_f <= 0.02
                for i = 1:length(y_act)
                    y_act[i] .-= min_f 
                end 
            end 
        end 
        # scale maximum value of open or activated states to 1 
        return norm(y_act, y_de; peak_min=pmax_min)
    else 
        return y_act, y_de 
    end 
end 

function one_sim_LossOnly(pars::AbstractVector{}, data, RawData, 
                        u0::AbstractVector{}, Qhold, db; 
                        pmax_min=0.7, states="open", normalize=false, method="sse") 

    va, vd = data["voltages"]
    ta, td = data["times"]
    shared = data["shared_idx"]
    traces = data["traces"]

    E = 0. 
    j = 0

    if normalize == false 
        for i in 1:length(va)
            Qa = tm(va[i], pars, db)
            sol = solve_odes(u0, Qa, ta[i])

            if states == "open"
                y_act = vec(sum(sol[oi:end,:], dims=1))
                x = traces[1][i] 
            else 
                y_act = vec(sum(sol[ai,:], dims=1))
                x = traces[1][i] 
            end 
            
            E += sum( (x .- y_act) .^ 2 )/length(ta[i])
            
            if va[i] in vd 
                j += 1 
                h = shared[j] 
                
                sol = solve_odes(sol[end], tm(vhold_d, pars, db), td[h])
                
                if states == "open"
                    y_de = vec(sum(sol[oi:end,:], dims=1))
                    x = traces[2][h] 
                else 
                    y_de = vec(sum(sol[ai,:], dims=1))
                    x = traces[2][h]
                end 
                
                E += sum( (x.- y_de) .^ 2 )/length(td[h])
            end 
        end 
        return E 
    else 
        y_act, y_de = one_sim(pars, data, u0, Qhold, db;
                        pmax_min=pmax_min, states=states, normalize=normalize)
        for (i, ysim) in enumerate(y_act)
            # if length(ysim) != length(traces[1][i])
            #     return 1e6
            # else 
            if method == "LLH"
                σ = std(ysim[end-5:end])^2
                E += (sum( (traces[1][i] .- ysim) .^2 ) / (2*(1.21e-4))) + (length(ysim)/2)*log(2*π*(1.21e-4))
            else 
                # E += sum( (traces[1][i] .- ysim) .^2 ) 
                E += sum( (traces[1][i] .- ysim) .^2 ) / length(ysim)
            end 
            # end 
        end 
        for (i, ysim) in enumerate(y_de)
            # if length(ysim) != length(traces[2][i])
            #     return 1e6 
            # else 
            if method == "LLH"
                σ = std(ysim[end-5:end])^2
                E += (sum( (traces[2][i] .- ysim) .^2 ) / (2*(1.21e-4))) + (length(ysim)/2)*log(2*π*(1.21e-4))
            else 
                # E += sum( (traces[2][i] .- ysim) .^2 ) 
                E += sum( (traces[2][i] .- ysim) .^2 ) / length(ysim)
            end 
            # end 
        end 
        return E
    end 
end 

function sim(pars::AbstractVector{}; 
            data_i = extracted_i,
            RawData_i = data_files[2:3], 
            data_f = extracted_f,
            RawData_f = data_files[4:5],
            db = DB_CONDITION,
            loss_only = false,
            error_method = "sse"
            )
    """
    data_i and data_f are Dicts with keys `voltages, times, and shared_idx`
        voltages = Array{Array{Float64,1}}
        times = Array{Array{Array{Float64,1}}}
        shared_idx = list of indices of voltages in the 2nd dataset that are also in the 1st dataset 
        `error_method` = 'sse' for sum of squared errors, 'LLH' for log likelihood 
    """
    #initial state vector (steady state at vhold) 
    Qhold = tm(vhold_a, pars, db)
    inits = get_ss(pars, vhold_a, db)

    if loss_only == true 
        if error_method == "sse"
            loss = one_sim_LossOnly(pars, data_i, RawData_i, inits, Qhold, db; 
                                    pmax_min=0.9, states="open", 
                                    normalize=true, method="sse")
            loss += one_sim_LossOnly(pars, data_f, RawData_f, inits, Qhold, db; 
                                    pmax_min=0.85, states="active", 
                                    normalize=true, method="sse")
            return loss 
        elseif error_method == "LLH"
            loss = one_sim_LossOnly(pars, data_i, RawData_i, inits, Qhold, db; 
                                    pmax_min=0.9, states="open",
                                    normalize=true, method="LLH")
            loss += one_sim_LossOnly(pars, data_f, RawData_f, inits, Qhold, db; 
                                    pmax_min=0.9, states="active", 
                                    normalize=true, method="LLH")
            return loss 
        else 
            println("`error_method` can only be 'sse' or 'LLH'.")
            asdf 
        end 
    else 
        sim_ia, sim_id = one_sim(pars, data_i, inits, Qhold, db; 
                                states="open", normalize=true)
        sim_fa, sim_fd = one_sim(pars, data_f, inits, Qhold, db; 
                                states="active", normalize=true)
        return sim_ia, sim_id, sim_fa, sim_fd 
    end 
end

#get a gv curve 
function sim_ss(pars::AbstractVector{T};
                data_ss = data_files[1],
                db = DB_CONDITION,
                loss_only=false, normalize=false) where T
    
    y = zeros(T, (size(data_ss)[1], 2))  
    for i in 1:size(data_ss)[1] 
        # println(data_ss)
        v = data_ss[i, 1]

        states = get_ss(pars, v, db)

        y[i,1] = sum(states[oi:end])
        y[i,2] = sum([states[a] for a in ai]) 
    end 

    if loss_only == true 
        E = 0.
        for i in 1:2
            if normalize
                y[:,i] .*= 1/maximum(y[:,i])
            end 
            E += sum((data_ss[:,i+1] .- y[:,i]) .^ 2)/length(y[:,i])
        end 
        return E 
    else 
        if normalize 
            for i in 1:2
                y[:,i] .*= 1/maximum(y[:,i])
            end 
        end 
        return [data_ss[:,1], y] 
    end 
end 

#################################################
# LOSS AND OBJECTIVE FUNCTIONS 
#################################################
function check_ranges()
    # check if popt is in ranges; if not, return closest bound 
    in_range = copy(popt) 

    for (i, b) in enumerate(ln_par_bds)
        if b[1] < popt[i] < b[2]
            continue
        else 
            println("$i th parameter is out of bounds, \n   ",
                    popt[i], " outside  $b")
            if popt[i] > b[2] 
                in_range[i] = b[2] 
            else 
                in_range[i] = b[1] 
            end 
        end 
    end 
    
    if in_range == popt 
        println("\n All parameters in-range.")
        println("\n Optimized cost = ", objective(popt))
        println("\n Exp params:    ")
        println(exp.(popt))
        return popt  
    else 
        println("\n Optimized cost = ", objective(in_range))
        println("\n Exp params:    ")
        println(exp.(in_range))
        return in_range 
    end 
end 

function objective(pars)
    # check for negative values in db_pars 
    if any(x -> x < 0, db_pars(pars, DB_CONDITION))
        # println("   negative")
        return 1e6
    else 
        # check whether parameters are in bounds 
        for (i, b) in enumerate(ln_par_bds)
            if b[1] <= pars[i] <= b[2]
                continue
            else 
                # println("   outside bound ", i)
                return 1e6
            end 
        end 

        # check that maximal steady state open probability is > 0.9
        if sum(get_ss(pars, 200, DB_CONDITION)[oi:end]) < 0.5
            # println("   maximal GV below threshold")
            return 1e6
        end 
    end 

    return sim(pars; loss_only=true) 
    # loss += 0.5*sim_ss(pars; loss_only=true) 
    # return loss 
end 

function mcmc_objective(pars)
    # check for negative values in db_pars 
    if any(x -> x < 0, db_pars(pars, DB_CONDITION))
        return 1e6
    else 
        # check whether parameters are in bounds 
        for (i, b) in enumerate(ln_par_bds)
            if b[1] <= pars[i] <= b[2]
                continue
            else 
                return 1e6
            end 
        end 

        # check that maximal steady state open probability is > 0.9
        if sum(get_ss(pars, 200, DB_CONDITION)[oi:end]) < 0.95
            return 1e6
        end 
    end 

    return sim(pars; loss_only=true, error_method="LLH") 
    # loss += 0.5*sim_ss(pars; loss_only=true) 
    # return loss 
end 

# gradient of objective function 
∇objective(pars) = ForwardDiff.gradient(objective, pars)
# println(∇objective(p0))
# asdf 

#################################################
# OPTIMIZATION 
#################################################
# remove parameters and bounds defined by detailed balance 
p0, ln_par_bds = prepare_BoundsAndParams(p0, ln_par_bds, DB_CONDITION)

# force parameters to closest bound if outside bounds 
for (i, b) in enumerate(ln_par_bds)
    if b[1] <= p0[i] <= b[2]
        continue
    elseif p0[i] < b[1]
        p0[i] = b[1] 
    else 
        p0[i] = b[2] 
    end 
end 

println("   Initial objective: ", objective(p0))
# asdf 

if OPT_OR_NOT == "GCMAES"
    popt, fmin, status = GCMAES.minimize((objective, ∇objective), p0, 0.4,
                                        [b[1] for b in ln_par_bds], 
                                        [b[2] for b in ln_par_bds], 
                                        maxiter = OPT_MAX_ITER
    )
    println("----------// RESULTS OF GCMA_ES //-----------",
            "\n     Status  ", status,
            "\n     Fmin = ", fmin,
            "\n     Bounds = ", ln_par_bds, 
            "\n     Full parameters... \n Real-space  ", 
            db_pars(popt, DB_CONDITION), 
            "\n Full parameters... \n Log-space  ", 
            log.(db_pars(popt, DB_CONDITION)))
elseif OPT_OR_NOT == "BB"
    optctrl = bbsetup(objective,
                    SearchRange = ln_par_bds,
                    Method = :adaptive_de_rand_1_bin_radiuslimited,
                    MaxStepsWithoutProgress = 2000,
                    MinDeltaFitnessTolerance = 1e-12, #regular = 1e-12 
                    PopulationSize=250,
                    TraceMode=:verbose,
                    TraceInterval=30.0,
                    MaxSteps= OPT_MAX_ITER,
                    MaxTime = 7200 #2h 
                    # CallbackFunction = callback, 
                    # CallbackInterval = 60.0
        )

    res = bboptimize(optctrl)
    popt = best_candidate(res)

    println("----------// RESULTS OF BBOPTIMIZE //-----------",
        "\n     Bounds... \n ", ln_par_bds,
        "\n     Full parameters... \n Real-space  ", 
        db_pars(popt, DB_CONDITION), 
        "Full parameters... \n Log-space  ", 
        db_pars(popt, DB_CONDITION; fmt="log"),
        )
elseif OPT_OR_NOT == "Optim"
    lower = [x[1] for x in ln_par_bds]
    upper = [x[2] for x in ln_par_bds]
    dfc = TwiceDifferentiableConstraints(lower, upper)


    # optimize with LBFGSB 
    # res = Optim.optimize(objective, p0, LBFGS(); autodiff = :forward)

    # optimize with box constraints 
    res = Optim.optimize(objective, dfc, p0, IPNewton())

    # popt = Optim.minimizer(res)
    
    println("----------// RESULTS OF Optim //-----------",
        "\n     Bounds... \n ", ln_par_bds,
        "\n     Full parameters... \n Real-space  ", 
        db_pars(popt, DB_CONDITION), 
        "\n     Full parameters... \n Log-space  ", 
        db_pars(popt, DB_CONDITION; fmt="log"),
        "\n     Function value... \n    ",
        Optim.minimum(res) 
        )
else
    popt = p0 
end 

#################################################
# SAMPLING
#################################################
function do_sampling()
    # parmaeter dimensionality
    D = length(popt)
    # initial parameter values 
    initial_θ = copy(popt)
    
    # Set the number of samples to draw and warmup iterations
    n_samples, n_adapts = 2_000, 1_000

    # Define a Hamiltonian system
    metric = DiagEuclideanMetric(D)
    hamiltonian = Hamiltonian(metric, mcmc_objective, ForwardDiff)

    # Define a leapfrog solver, with initial step size chosen heuristically
    initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
    integrator = Leapfrog(initial_ϵ)

    # Define an HMC sampler, with the following components
    #   - multinomial sampling scheme,
    #   - generalised No-U-Turn criteria, and
    #   - windowed adaption for step-size and diagonal mass matrix
    proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

    # Run the sampler to draw samples from the specified Gaussian, where
    #   - `samples` will store the samples
    #   - `stats` will store diagnostic statistics for each sample
    samples, stats = sample(hamiltonian, proposal, initial_θ, n_samples, adaptor, n_adapts; progress=true)

    output_dir = ".\\output\\MCMC\\"
    fname = Dates.format(Dates.now(), "ddmmyy_HHMMSS")
    writedlm(string(output_dir, "mcmc_samples_$fname,csv"), samples)
    writedlm(string(output_dir, "mcmc_stats_$fname.csv"), stats)
end 
# do_sampling()
#################################################
# OUTPUT 
#################################################
println("   Objective value: ", objective(popt))
println("   MCMC objective value: ", mcmc_objective(popt))

# apply detailed balance to optimized parameters (or p0, if no optimization was done)
# output in log format 
popt_full = db_pars(popt, DB_CONDITION; fmt="log")

#################################################
# PLOTTING 
#################################################
function plotter()
    """
    Uses `popt`, which is assumed to be log-transformed.
    Detail balance will be applied during simulation, so `popt` should not be full length if `DB_CONDITION` is not `'x'`.
    If optimization was not performed, `popt = p0`.
    `p0` should be defined in the corresponding model file. 
    """
    #simulate with optimal parameters 
    sim_ia, sim_id, sim_fa, sim_fd = sim(popt; loss_only = false)
    volts, y_ss = sim_ss(popt; loss_only = false)

    # steady state 
    plt_g = plot(
            xlabel="Voltage (mV)",
            ylabel="Normalized Conductance",
            legend=:outertopright
        );  
        
    plt_f = plot(
        xlabel="Voltage (mV)",
        ylabel="Normalized dFluorescence",
        legend=:outertopright
    ); 

    labels = ["GV", "FV"]
    plts = [plt_g, plt_f] 

    # plot steady state data 
    m = Matrix(data_files[1])
    for j in 2:size(m)[2]  
        plot!(plts[j-1], m[:, 1], m[:, j], 
                markershape = :circle, markercolor = :blue,  
                linecolor = :blue, linealpha = 0.3, 
                linewidth = 4, label = "Data/" * labels[j-1]
        )

        plot!(plts[j-1], m[:, 1], y_ss[:, j-1], 
                linestyle = :dash, linealpha = 0.5, 
                linecolor = :red, linewidth = 4, 
                markershape = :circle, markercolor = :red, 
                label = "Sim/" * labels[j-1]
        )

        display(plts[j-1]) 
    end 
    
    function plot_TCourse(title)
        if title == "current"
            ylab = "Normalized Current"
            ex = extracted_i
            y_a, y_d = sim_ia, sim_id 
            println([length(x) for x in y_d])
        else
            ylab = "Normalized Fluor"
            ex = extracted_f 
            y_a, y_d = sim_fa, sim_fd 
        end 

        plt_a = plot(
            xlabel="Time (ms)",
            ylabel=ylab,
            legend=:outertopright            
        )
        plt_d = plot(
            xlabel="Time (ms)",
            ylabel=ylab,
            legend=:outertopright            
        )
        
        va, vd = ex["voltages"]
        ta, td = ex["times"]
        traces = ex["traces"]
        # shared = ex["shared"]

        for (i, v) in enumerate(va)
            t = ta[i] 
            y = traces[1][i] 

            plot!(plt_a, t, y, 
                seriestype = :scatter, 
                markeralpha = 0.35,
                label = string(v)
            )

            # println(length(t), "    ", length(y_a[i]))
            if length(y_a[i]) > 1
                plot!(plt_a, t[1:length(y_a[i])], y_a[i],
                        linestyle = :dash,
                        linewidth = 4, 
                        label = "" 
                )
            end 
        end 

        for (i, v) in enumerate(vd) 
            # v = vd[i]
            t = td[i] 
            y = traces[2][i]             

            try 
                plot!(plt_d, t, y, 
                    seriestype = :scatter, 
                    markeralpha = 0.35,
                    label = string(v)
                )
            catch e 
                println(e)
            end 

            # println(length(t), "    ", length(y_d[i]))
            if length(y_d[i]) > 1
                # plot!(plt_d, t[1:length(y_d[i])], y_d[i],
                plot!(plt_d, t, y_d[i],
                        linestyle = :dash,
                        linewidth = 4, 
                        label = "" 
                )
            end 
        end

        display(plt_a) 
        display(plt_d)
    end 
                
    try 
        plot_TCourse("current")
    catch e 
        println(e)
    end 
    try 
        plot_TCourse("fluorescence")
    catch e 
        println(e)
    end 
end 

plotter()