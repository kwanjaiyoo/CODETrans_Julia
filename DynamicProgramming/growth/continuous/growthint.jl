using LinearAlgebra, Statistics, Plots, NLsolve, Optim, Random, Interpolations, Parameters
using Optim: maximizer, maximum

include("/Users/kwyyoo/Dropbox/KJ/CODETrans/A1Markov/Julia/markovtheory.jl")
include("contiBellman.jl")

# Parameters
param = @with_kw (δ = .1, α = .33, A = 1, β = .9);
@unpack δ, α, A, β = param();

# Algorithm parameters
tol = 1e-5;
simyes = 1;

ρ = 0.95;
σ_ϵ = 0.00712;
n_θ = 7;
m = 3;
Π, θ, P, arho, asigma = markovapprox(ρ, σ_ϵ, n_θ, m);
θ = exp.(θ);

# Grid for capital
n_k = 1000;
k_min = 1;
#k_max = ( δ / θ(lt) )^( 1 / (α - 1) );
k_max = 3;
k = range(k_min, k_max, length=n_k);

@time Vstar, polk = contiBellman(Π, k, θ, n_k, n_θ, param)

polc = A * k .^ α * θ' + (1 - δ) * k * ones(1, n_θ) - polk;

## Make plots of value and policy functions
fig_value = plot(k, Vstar[:, 1], linecolor=:black, xlabel="capital stock", label="θ_min ");
plot!(k, Vstar[:, end], linecolor=:red, label="θ_max");
plot!(legend=:bottomright, title="Value function")

fig_polk = plot(k, polk[:, 1], title="Capital policy functions", xlabel="capital stock", label="θ_min");
plot!(k, polk[:, end], label="θ_max");
plot!(legend=:bottomright)

fig_polc = plot(k, polc[:, 1], title="Consumption policy functions", xlabel="capital stock", label="θ_min");
plot!(k, polc[:, end], label="θ_max");
plot!(legend=:bottomright)

fig_policies = plot(fig_polk, fig_polc, layout=(2, 1), legend=false)

# Simulation
function simul(numperiod, markov, nk, kgrid, θgrid)
    
    shock = zeros(numperiod);
    capital = zeros(numperiod+1);
    output = zeros(numperiod);
    invest = zeros(numperiod);
    consum = zeros(numperiod);

    S = markovchain(markov, numperiod);
    index_k = round(Int, nk / 2);
    capital[1] = kgrid[index_k];

    for time in 1:numperiod
        shock[time] = θgrid[S[time]];
        output[time] = A * shock[time] * capital[time]^α;
        kpr = LinearInterpolation(k, polk[:, S[time]]);
        capital[time + 1] = kpr(capital[time]);
        invest[time] = capital[time + 1] - (1 - δ) * capital[time]; 
        consum[time] = output[time] - invest[time]
    end

    return output, capital, invest, consum
end

if simyes == 1
    T = 100000;
    Y, K, In, C = simul(T, Π, n_k, k, θ);
    period = 1:T
    plot(title="Simulation of all variables", xlabel="period", legend=:bottom)
    plot!(period, Y, label="Y")
    plot!(period, C, label="C")
    plot!(period, K[2:T+1], label="K")
    plot!(period, In, label="I")
    
end