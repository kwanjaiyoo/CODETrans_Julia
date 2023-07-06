using LinearAlgebra, Parameters, Plots, Random, Statistics, Distributions, Roots, DelimitedFiles, Printf

include("PEA_lfoc.jl")
include("nlls.jl")
include("hp1.jl")
include("SimulationPEA.jl")

#PARAMETER
mp = @with_kw (α = .36, β = .99, a = 2, δ = .025, ρ = .95, σ_ε = .00712, T = 10000, Ez = 1, λ = .75);
@unpack α, β, a, δ, ρ, σ_ε, Ez, T, λ = mp()

#ALGORITHM PARAMETERS
siyes = 1; #simulte model if equals to 1
stationary = 1;	#finds stationary distribution for capital

# DEFINE STEADY STATE VARIABLES
xx = ( 1 - β * (1-δ) ) / ( β * α * Ez );
yy = ( ( 1 / β + δ - 1 ) / α * ( 1 + (1-α)/a ) - δ ) * a /((1-α) * Ez);
l_ss = xx / yy;  # for this set of parameters, the steady state labor approxiamately equals to 1/3;
k_ss = xx^(1/(α - 1))*l_ss;
y_ss = Ez * k_ss^α * l_ss^(1-α);
i_ss = δ * k_ss;
c_ss = y_ss-i_ss;

steadystate = @with_kw (l_ss = l_ss, k_ss = k_ss, y_ss = y_ss, i_ss = i_ss, c_ss = c_ss);

# SIMULATION OF CONTINUOUS SHOCKS PROCESS
zt = ones(T+1);
εd = Normal(0, σ_ε);
ε = rand(εd, T+1);
for t = 2:T+1
    zt[t] = exp(ρ* log(zt[t-1]) + ε[t]);
end

@time bet = PEA_lfoc(nlls, zt, mp, steadystate);

#SIMULATION
T = 115;
N = 100;

@time ss, cc = SimulationPEA(hp1, T, N, bet, mp, steadystate)

stdv = mean(ss, dims = 1);
corr = mean(cc, dims = 1);

println("HANSEN: std(x)/std(y) corr(x,y) for y,i,c,k,h,prod")
[[1.36  4.24 0.42 0.36 0.7 0.68]'./1.36 [1 0.99 0.89 0.06 0.98 0.98]']
println("std(x) std(x)/std(y) corr(x,y) for y,i,c,k,h,prod:")
[stdv' (stdv./stdv[1])' corr']