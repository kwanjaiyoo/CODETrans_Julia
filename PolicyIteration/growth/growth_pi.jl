using LinearAlgebra, Statistics, StaticArrays, Distributions, NLsolve, Optim, Plots, Parameters, Interpolations, Printf
include("policyiter.jl")
include("PIsimul.jl")
#PARAMETERS
mp = @with_kw (β = .99, σ = 1, δ = .025, θ = .36);

#GRID DIMENSION
ns = 2;
N = 301;
T = ns * N;

#SHOCK
zz = [0.99, 1.01];
ug = 8;
p = 1 - 1/ug;
Π = [p 1-p; 1-p p];

#GRID ON CAPITAL
kmin = 1;
kmax = 120;
k = range(kmin, kmax, length=N);

@time c, kpr, r, w = policyiter(zz, Π, k, N, ns, mp);

#PLOTS
plot(k, c[:, 1], title = "consumption", label = "low shock")
plot!(k, c[:, 2], label = "high shock")

plot(k, kpr[:, 1], title = "Capital", label = "low shock")
plot!(k, kpr[:, 2], label = "high shock")

#SIMULATION
T = 1000;
consumption, capital = PIsimul(T, kpr, k, zz, Π, mp);

plot(1:T, consumption, title = "consumption", xlabel = "period")
plot(1:T, capital[2:T+1], title = "capital", xlabel = "period")