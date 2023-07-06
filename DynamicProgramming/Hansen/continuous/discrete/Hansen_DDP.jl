using LinearAlgebra, Statistics, Interpolations, NLsolve, Parameters, Plots, Random, Roots
using Optim: maximize, maximizer

include("/Users/kwyyoo/Dropbox/KJ/CODETrans/A1Markov/Julia/markovtheory.jl");
include("HansenDDPBellman.jl")
include("hp1.jl")
include("HansenSimul.jl")

# MODEL PARAMETERS
mp = @with_kw (α = .26, β = .99, a = 2, δ = .025, ρ = .95, σ_ε = .00712);
@unpack α, β, a, δ, ρ, σ_ε = mp()

# ALGORITHM PARAMETERS
siyes = 1;
loadin = 0;
loadsimuin = 0;
stationary = 1;
tolv = 1e-12;

# DISCRETIZE CONTINUOUS SHOCKS INTO MARKOV PROCESS
N = 7;
m = 3;
zbar = 1;
Π, z, P, arho, asigma = markovapprox(ρ, σ_ε, N, m);
z = exp.(z);

lt = size(z, 1);
Pi = Π^10000;
Ez = 1;

# DEFINE STEADY STATE VARIABLES
xx = ( 1 - β * (1-δ) ) / ( β * α * Ez );
yy = ( ( 1 / β + δ - 1 ) / α * ( 1 + (1-α)/a ) - δ ) * a /((1-α) * Ez);
l_ss = xx / yy;  # for this set of parameters, the steady state labor approxiamately equals to 1/3;
k_ss = xx^(1/(α - 1))*l_ss;
y_ss = Ez * k_ss^α * l_ss^(1-α);
i_ss = δ * k_ss;
c_ss = y_ss-i_ss;

# DEFINE CAPITAL GRID
kmin = .894 * k_ss;
kmax = 1.115 * k_ss;
lk = 200;
gdk = (kmax - kmin) / (lk - 1);
k = range(kmin, kmax, length = lk);
gk = lk * lk;


# function lfoc(x, k, kprime, θ, parameter)
#     @unpack α, δ, a = parameter()

#     c = θ * k^α * x^(1-α) + (1-δ) * k - kprime;
#     F = c/(1-x) - (1-α)/a * θ * k^α * x^(-α);
#     return F
# end

@time Vstar, polk, poll, poly, poli, polc, kindex = HansenDDPBellman(Π, k, z, lk, lt, tolv, mp);

# FIGURES
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

fig_poli = plot(k, poli[:, 1], title = "Investment Policy Functions", xlabel = "capital stock", label = "z_min");
plot!(k, poli[:, end], label = "z_max");
plot!(legend=:bottomright)

fig_poll = plot(k, poll[:, 1], title = "Labor Policy Functions", xlabel = "capital stock", label = "z_min");
plot!(k, poll[:, end], label = "z_max");
plot!(legend=:bottomright)

fig_polipoll = plot(fig_poli, fig_poll, layout = (2, 1), legend = false)


# MONTE CARLO SIMULATION

T = 115;
N = 100;

stdv, stdv_stdv, corr, corr_stdv = HansenSimul(hp1, T, N, lk, Π, z, polk, poll, polc, kindex, mp)

print("HANSEN: std(x)/std(y) corr(x,y) for y,c,i,k,h,prod")
@show [ [1.36 0.42 4.24 0.36 0.7 0.68]'./1.36 [1 0.89 0.99 0.06 0.98 0.98]' ]
print("std(x) std(x)/std(y)  stdv_stdv corr(x,y) corr_stdv for y,c,i, k,h,prod:")
@show [stdv' (stdv./stdv[1])' stdv_stdv' corr' corr_stdv']