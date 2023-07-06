using LinearAlgebra, Statistics, Random, Distributions, Plots, Roots, NLopt, NLsolve, Parameters, Printf

include("ricatti1.jl")
include("hp1.jl")
include("lqsimul.jl")

#PARAMETERS
mp = @with_kw (α = .33, β = .96, ρ = .95, δ = .1, σ_ε = .007);
@unpack α, β, ρ, δ, σ_ε = mp()

#STEADY STATE
ss = @with_kw (zs = 0,
               ks = ( exp(zs) * α * β / (1 - β * (1 - δ)) ) ^ (1 / (1 - α)),
               is = δ * ks,
               cs = exp(zs) * ks^α - is);
@unpack zs, ks, is, cs = ss()

#QUADRATIC EXPANSION OF THE UTILITY FUNCTION
R = log(cs);
DJ = zeros(3);
DJ[1] = exp(zs) * ks^α / cs;
DJ[2] = exp(zs) * α * ks^(α - 1) / cs;
DJ[3] = -1 / cs;

Hzz = ( (exp(zs) * ks^α) * cs - (exp(zs) * ks^α)^2 ) / (cs^2);
Hkk = ( ( (exp(zs) * α * (α - 1) * ks^(α - 2)) * cs ) - (exp(zs)* α *ks^(α - 1))^2 ) / (cs^2);
Hxx = -1 / (cs^2);
Hzk = (((exp(zs) * α * ks^(α - 1)) * cs) - (exp(zs) * ks^α * exp(zs) * α * ks^(α - 1)) ) / (cs^2);
Hzx = (exp(zs) * ks^α) / (cs^2);
Hkx = (exp(zs) * α * ks^(α - 1)) / (cs^2);

DH = [Hzz Hzk Hzx;
      Hzk Hkk Hkx;
      Hzx Hkx Hxx];

S = [zs; ks];
C = [is];

B = [1 0 0 0;
     0 ρ 0 0;
     0 0 1-δ 1];

Σ = [0 0 0;
     0 σ_ε^2 0;
     0 0 0];

P, J, D = ricatti1(R, DJ, DH, S, C, B, Σ, β);

println("The optimal value function is [1 z s]P0[1; z; s]+d, where P and d are given by:")
@show P
@show D

println("The policy function is x=J[1; z; s] where J  is:")
@show J

#SIMULATION
T = 115;
N = 100;
ss, cc = lqsimul(T, N, P, J, D, ss, mp);

stdv = mean(ss, dims = 1);
corr = mean(cc, dims = 1);
@printf "std(x) std(x)/std(y) corr(x,y):"
[stdv' (stdv./stdv[1])' corr']