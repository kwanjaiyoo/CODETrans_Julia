using LinearAlgebra, Statistics, Plots, NLsolve, Optim, Random, Parameters
using Optim: maximizer, maximum

include("/Users/kwyyoo/Dropbox/KJ/CODETrans/A1Markov/Julia/markovtheory.jl")
include("discreteBellman.jl")

# Model parameters
mp = @with_kw (α = .33,
               δ = .1,
               A = 1,
               β = .9,
               γ = 2);
@unpack α, β, A, δ, γ = mp()

# Algorithm parameters
simyes = 1;
stationary = 1;
tolv = 1e-7;

# Define the type of shock
shock = 3;
if shock == 1
    Π = [0.8 0.2; 0.2 0.8];
    P = ergodic(Π);
elseif shock == 2
    Π = [0.40 0.30 0.20 0.10;
    0.25 0.40 0.25 0.10;
    0.10 0.25 0.40 0.25;
    0.10 0.20 0.30 0.30];
    P = ergodic(Π);
elseif shock == 3
    ρ = 0.95;
    σ_ϵ = 0.00712;
    N = 7;
    m = 3;
    Π, θ, P, arho, asigma = markovapprox(ρ, σ_ϵ, N, m);
    θ = exp.(θ);
end

# Grid for the shock
if shock == 1
    θ_min = 0.9;
    θ_max = 1.1;
    θ = [θ_min; θ_max];
elseif shock == 2
    θ = [0.9775; 0.99; 1.01; 1.0225];
end
lθ = length(θ);

# Steady state
Eθ = P ⋅ θ;
k_ss = ( A * Eθ * α * β / (1 - β * (1 - δ)) )^( 1 / (1 - α) );
y_ss = Eθ * A * k_ss^α;
i_ss = δ * k_ss;
c_ss = y_ss - i_ss;

# Grid for capital
lk = 100;
k_min = 0.5 * k_ss;
k_max = 1.5 * k_ss;
k = range(k_min, k_max, length = lk);

gk = lk^2;
c = zeros(gk, lθ);
for t in 1:lθ
    for i in 1:lk
        for j in 1:lk
            c[ (i-1) * lk + j, t ] = A * θ[t] * k[i]^α + (1-δ) * k[i] - k[j];
            if c[ (i-1) * lk + j, t ] < 0
                c[ (i-1) * lk + j, t ] = 1e-7;                
            end            
        end        
    end
end

# Iterate on the value functions
maxiter = 1000;


@time Vstar, kindex = discreteBellman(Π, maxiter, tolv, lk, lθ, c; mp);

polk = k[kindex];
polc = A * (θ * ones(1, lk))' .* (k.^α * ones(1, lθ)) + (1 - δ) * k * ones(1, lθ) - polk;

# Find the stationary distribution
if lk < 10
    stationary = 0;
end

function stationarydist(stationary, kpol, kgrid, nk)
    if stationary == 1
        i = 1;
        while kpol[i, 1] >= kgrid[i]
            i = i + 1;
            if i >= nk
                break            
            end        
        end
        println("Lower support of stationary distribution of capital is ", k[i-1])
        
        j = 1;
        while kpol[j, lθ] >= kgrid[j]
            j = j + 1;
            if j >= nk
                break
            end
        end
        println("Upper support of stationary distribution of capital is ", k[j-1])
    end
end

stationarydist(stationary, polk, k, lk)


# Make plots of value and policy functions
# Value function
figure_valuefun = plot(k, Vstar[:, 1], title = "Value functions", xlabel = "capital stock", label = "θ_min")
plot!(k, Vstar[:, lθ], label = "θ_max")

fig_capitalpol = plot(k, polk[:, 1], title = "Capital policy functions", xlabel = "capital stock", label = "θ_min")
plot!(k, polk[:, lθ], label = "θ_max")

fig_conspol = plot(k, polc[:, 1], title = "Consumption policy functions", xlabel = "capital stock", label = "θ_min")
plot!(k, polc[:, lθ], label = "θ_max")

fig_policies = plot(fig_capitalpol, fig_conspol, layout = (2,1), legend = false)

# Simulation
T = 10000;
function simul(numperiod, markov, kpol, nk, indexk, theta, kgrid, alpha, delta, kss, css)
    shock = zeros(numperiod);
    capital = zeros(numperiod+1);
    output = zeros(numperiod);
    cons = zeros(numperiod);
    invest = zeros(numperiod);
    
    S = markovchain(markov, numperiod);
    indk = round(Int, nk/2);
    capital[1] = kpol[indk, 1];
    for t in 1:numperiod
        indk = indexk[indk, S[t]];
        shock[t] = theta[S[t]];
        capital[t+1] = kgrid[indk];
        output[t] = A * shock[t] * capital[t]^alpha;
        invest[t] = capital[t+1] - (1 - delta) * capital[t];
        cons[t] = output[t] - invest[t];
    end
    k_hat = (capital .- kss) / kss;
    c_hat = (cons .- css) / css;

    return output, capital, cons, invest, k_hat, c_hat
end

if simyes == 1
    Y, K, C, Inv, k_hat, c_hat = simul(T, Π, polk, lk, kindex, θ, k, α, δ, k_ss, c_ss);

    # Plot the simulation results
    time = 1:T;

    plot(time, K[2: T+1], title = "Simulation of all variables", xlabel = "time", label = "K")
    plot!(time, Y, label = "Y")
    plot!(time, Inv, label = "I")
    plot!(time, C, label = "C")
    
    fig_simulcapital = plot(time, k_hat[2: T+1], title = "Stochastic simulation of capital around its steady state", xlabel = "time", ylabel = "% deviation", label = "k̂");
    fig_simulconsumption = plot(time, c_hat, title = "Stochastic simulation of consumption around its steady state", xlabel = "time", ylabel = "% deviation", label = "ĉ");
    fig_simul = plot(fig_simulcapital, fig_simulconsumption, layout = (2, 1))
end