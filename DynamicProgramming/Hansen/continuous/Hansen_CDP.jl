## Hansen(1985): Continuous Dynamic Programming

using LinearAlgebra, Statistics, Random, Interpolations, Plots, NLopt, NLsolve, Parameters, Roots, Optim
using Optim: maximizer, maximum, minimizer, iterations, converged

include("/Users/kwyyoo/Dropbox/KJ/CODETrans/A1Markov/Julia/markovtheory.jl")
include("HansenCDPBellman.jl")
include("HansenCDPsimul.jl")
include("hp1.jl")

# Model Parameters
mp = @with_kw (α = .36, β = .99, a = 2, δ = .025, ρ = .95, σ_ϵ = .00712)

@unpack α, β, a, δ, ρ, σ_ϵ = mp()

# Algorithm Parameters
siyes = 1;
stationary = 1;
tolv = 1e-3;
loadin = 0;
loadsimuin = 1;

# Discretize continuous shocks into Markov Process
N = 7;
M = 3;
z̄ = 1;
Π, z, P, arho, asigma = markovapprox(ρ, σ_ϵ, N, M);
z = exp.(z);

# Grid for the shock
lt = length(z);
Pi = Π^10000;
Ez = 1;

# Steady state
xx = ( 1 - β*(1-δ) ) / ( β*α*Ez );
yy = ( (1/β + δ - 1) / α * ( 1 + (1 - α)/a ) - δ ) * a / ( (1-α)*Ez );
l_ss = xx/yy;
k_ss = xx^(1 / (α-1)) * l_ss;
y_ss = Ez * k_ss^α * l_ss^(1-α);
i_ss = δ * k_ss;
c_ss = y_ss - i_ss;

# Define capital grid
kmin = .894 * k_ss;
kmax = 1.115 * k_ss;
lk = 200;
k = range(kmin, kmax, length = lk);

@time Vstar, polk, poll, poly, polc, poli = HansenCDPBEllman(Π, k, z, lk, N, tolv, mp);

fig_value = plot(k, Vstar[:, 1], linecolor=:black, xlabel="capital stock", label="z_min");
plot!(k, Vstar[:, end], linecolor=:red, label="z_max");
plot!(legend=:bottomright, title="Value Functions")

fig_polk = plot(k, polk[:, 1], title="Capital Policy Functions", xlabel="capital stock", label="z_min");
plot!(k, polk[:, end], label="z_max");
plot!(legend=:bottomright)

fig_polc = plot(k, polc[:, 1], title="Consumption Policy Functions", xlabel="capital stock", label="z_min");
plot!(k, polc[:, end], label="z_max");
plot!(legend=:bottomright)

fig_polkpolc = plot(fig_polk, fig_polc, layout=(2, 1), legend=false)

fig_poll = plot(k, poll[:, 1], title = "Labor Policy Functions", xlabel = "capital stock", label = "z_min");
plot!(k, poll[:, end], label = "z_max");
plot!(legend=:bottomright)

# Simulation
T = 115;
n = 100;

@time stdv, stdv_stdv, corr, corr_stdv = HansenCDPsimul(hp1, T, n, Π, z, k, polk, poll, polc, mp);

println("HANSEN: std(x)/std(y) corr(x,y) for y,c,i,k,h,prod")
@show [[1.36 0.42 4.24 0.36 0.7 0.68]'./1.36 [1 0.89 0.99 0.06 0.98 0.98]']
println("std(x) std(x)/std(y)  stdv_stdv corr(x,y) corr_stdv for y,c,i,k,h,prod:")
@show [stdv' (stdv./stdv[1])' stdv_stdv' corr' corr_stdv']