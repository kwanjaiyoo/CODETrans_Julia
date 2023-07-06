using LinearAlgebra, Statistics, Random, Plots, Interpolations, Parameters, DelimitedFiles
using NLsolve, Optim, Roots

# include("acm.jl")
include("/Users/kwyyoo/Dropbox/KJ/CODETrans/A1Markov/Julia/Floden/Flodenmarkov.jl")
include("/Users/kwyyoo/Dropbox/KJ/CODETrans/A1Markov/Julia/markovtheory.jl")
include("policyiter_hansen.jl")
include("PIsimul_hansen.jl")
include("hp1.jl")

#PARAMETERS
loadin = 1;
mp = @with_kw (β = .99, σ = 1, δ = .025, θ = .36, a = 2, Ez = 1, λ = .9);
@unpack β, σ, δ, θ, a, Ez, λ = mp()
#SHOCK
zz, Π, Π_MF = addacooper(7, 0, .95, .00712);
zz = exp.(zz);
ns = length(zz);

N = 301;
T = ns*N;

#GRID ON CAPITAL
kmin = 1;
kmax = 120;
k = range(kmin, kmax, length=N);

#DEFINE STEADY STATE VARIABLES
xx = ( 1 - β * (1-δ) ) / ( β * θ * Ez );
yy = ( ( 1 / β + δ - 1 ) / θ * ( 1 + (1-θ)/a ) - δ ) * a /((1-θ) * Ez);
l_ss = xx / yy;  # for this set of parameters, the steady state labor approxiamately equals to 1/3;
k_ss = xx^(1/(θ - 1))*l_ss;
y_ss = Ez * k_ss^θ * l_ss^(1-θ);
i_ss = δ * k_ss;
c_ss = y_ss - i_ss;

ss = @with_kw (l_ss = l_ss, k_ss = k_ss, y_ss = y_ss, i_ss = i_ss, c_ss = c_ss);

if loadin == 0
    @time c, kpr, h, r, w = policyiter_hansen(zz, Π, k, N, ns, mp);
    writedlm("polc.csv", c, ',')
    writedlm("polk.csv", kpr, ',')
    writedlm("polh.csv", h, ',')
    writedlm("polr.csv", r, ',')
    writedlm("polw.csv", w, ',')
else
    c = readdlm("polc.csv", ',');
    kpr = readdlm("polk.csv", ',');
    h = readdlm("polh.csv", ',');
    r = readdlm("polr.csv", ',');
    w = readdlm("polw.csv", ',');
end

#PLOTS
plot(k, c, title = "consumption", xlabel = "capital", label = "")
plot(k, kpr, title = "capital", xlabel = "capital", label = "")
plot(k, h, title = "labor", xlabel = "capital", label = "")

#SIMULATION
T = 115;
N = 100;
std_mat, cc_mat = PIsimul_hansen(T, N, kpr, h, k, zz, Π, mp, ss);

stdd = mean(std_mat, dims = 1)
corr = mean(cc_mat, dims = 1)

println("HANSEN: std(x)/std(y) corr(x,y) for y,i,c,k,h,prod")
[[1.36  4.24 0.42 0.36 0.7 0.68]'./1.36 [1 0.99 0.89 0.06 0.98 0.98]']
println("std(x) std(x)/std(y) corr(x,y) for y,i,c,k,h,prod:")
[stdd' (stdd./stdd[1])' corr']