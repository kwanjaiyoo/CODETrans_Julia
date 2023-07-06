using LinearAlgebra, Random, Distributions, Statistics, Parameters, DelimitedFiles, Plots, Printf

include("PEA.jl")
include("nlls.jl")
include("hp1.jl")
#PARAMETERS
mp = @with_kw (β = .99,
               α = .36,
               δ = .025,
               T = 10000,
               sig = 1,
               ρ_z = .95,
               σ_ε = .01,
               λ = 0);
@unpack α, β, δ, T, sig, ρ_z, σ_ε, λ = mp();

#SHOCK
# z = zeros(T+1);
# dε = Normal(0, σ_ε);
# ε = rand(dε, T+1);
# for i in 1:T
#     z[i+1] = ρ_z * z[i] + ε[i];
# end
# z = exp.(z);

z = readdlm("shock.csv", ',');
z = vec(z);

#STEADY STATE
ss = @with_kw (zs = 1,
               ks = ((1 - β * (1-δ))/(β * α)) ^ (1/(α-1)),
               ys = zs * (ks^α),
               is = δ*ks,
               cs = ys-is);
@unpack zs, ks, ys, is, cs = ss();

#FIND PEA FUNCTION
@time bita = PEA(nlls, z, mp, ss);

#MAKE IMPULSE REPONSE
NS = 100 + 1;
eps = zeros(NS);
eps[2] = 1;
shock = zeros(NS);
shock[1] = 1;
c = zeros(NS);
c[1] = cs;
k = zeros(NS);
k[1] = ks;
y = zeros(NS);
y[1] = ys;
invest = zeros(NS);
invest[1] = is;
Pea = zeros(NS);
Pea[1] = 0;

for i in 2:NS
    shock[i] = exp(ρ_z * log(shock[i-1]) + eps[i]);
    Pea[i] = exp(bita[1] + bita[2] * log(shock[i]) + bita[3] * log(k[i-1]));
    c[i] = (β * Pea[i])^(-1/sig);
    y[i] = shock[i] * (k[i-1]^α);
    invest[i] = y[i] - c[i];
    k[i] = invest[i] + (1 - δ) * k[i-1];
end
#DEVIATIONS FROM STEADY STATE
c = log.(c ./ cs);
k = log.(k ./ ks);
y = log.(y ./ ys);
invest = log.(invest ./ is);
LOG_DEV = [c k y invest];

#PLOTS THE IMPULSE REPONSE FUNCTIONS
plot(1:NS-1, LOG_DEV[1:NS-1, :],
    title = "Response to a one percent deviation",
    label = ["consumption" "capital" "output" "investment"])


#CALCULATE THE STATISTICS
NR = 200 + 1;
sd = zeros(4);
rd = zeros(4);

c = zeros(NR);
c[1] = cs;
k = zeros(NR);
k[1] = ks;
y = zeros(NR);
y[1] = ys;
invest = zeros(NR);
invest[1] = is;
Pea = zeros(NR);
Pea[1] = 0;
dε = Normal(0, σ_ε);
ε = rand(dε, NR);
shock = zeros(NR);
shock[1] = 1;
for i in 2:NR
    shock[i] = exp(ρ_z * log(shock[i-1]) + ε[i]);
    Pea[i] = exp(bita[1] + bita[2] * log(shock[i]) + bita[3] * log(k[i-1]));
    c[i] = (β * Pea[i])^(-1/sig);
    y[i] = shock[i] * (k[i-1]^α);
    invest[i] = y[i] - c[i];
    k[i] = invest[i] + (1 - δ) * k[i-1];
end

c = log.(c ./ cs);
k = log.(k ./ ks);
y = log.(y ./ ys);
invest = log.(invest ./ is);
LOG_DEV = [c k y invest];

detrend = hp1([c k y invest], 1600);
c = detrend[:, 1];
k = detrend[:, 2];
y = detrend[:, 3];
invest = detrend[:, 4];

@printf "The average standard deviations for the variables are"
sd = [std(c) std(k) std(y) std(invest)]
@printf "The relative standard deviations are"
rd = sd ./ sd[3]