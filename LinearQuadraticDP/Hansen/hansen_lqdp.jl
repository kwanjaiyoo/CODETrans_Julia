using LinearAlgebra, Parameters, Plots, Random, Distributions, Statistics, Interpolations

include("ricatti1.jl")
include("lqsimul_hansen.jl")
include("hp1.jl")

#PARAMETERS
mp = @with_kw (β = .99, δ = .025, α = .36, ρ = .95, γ = 1, σ_ε = .00712);
@unpack β, δ, α, ρ, γ, σ_ε = mp();

#STEADY STATE
ss = @with_kw (zbar = 1,
               hbar = .3,
               kbar = hbar * ( (γ / β - (1 - δ)) / α )^(1 / (α - 1)),
               ibar = (γ - 1 + δ) * kbar,
               ybar = kbar^α * hbar^(1 - α),
               cbar = ybar - ibar,
               prodbar = ybar / hbar,
               Rbar = α * kbar^(α - 1) * hbar^(1 - α),
               wbar = (1 - α) * kbar^α * hbar^(-α),
               a = (1 - hbar) * wbar / cbar)
@unpack zbar, hbar, kbar, ibar, ybar, cbar, prodbar, Rbar, wbar, a = ss()
println("The steady state values of z, y, i, c,k and h are:")
@show [zbar ybar ibar cbar kbar hbar]

#OBTAIN A QUADRATIC APPROXIMATION OF THE RETURN FUNCTION
Ubar = log(cbar) + a * log(1 - hbar);

#CONSTRUCT THE QUADRATIC EXPANSION OF THE UTILITY FUNCTION
Uz = kbar^α * hbar^(1 - α) / cbar;
Uk = α * zbar * kbar^(α - 1) * hbar^(1 - α) / cbar;
Ui = -1 / cbar;
Uh = (1 - α) * zbar * kbar^α * hbar^(-α) / cbar - a / (1 - hbar);
DJ = [Uz;Uk;Ui;Uh];

c2  = cbar^2;
Ukk = ( (α - 1) * α * zbar * kbar^(α - 2) * hbar^(1 - α) * cbar -
        (α * zbar * kbar^(α - 1) * hbar^(1 - α)) * 
        (α * zbar * kbar^(α - 1) * hbar^(1 - α)) ) / c2;

Ukz = ( α * kbar^(α - 1) * hbar^(1 - α) * cbar -
        α * zbar * kbar^(α - 1) * hbar^(1-α) * kbar^α * hbar^(1-α) )/ c2;

Uki = α * zbar * kbar^(α - 1) * hbar^(1 - α) / c2;

Ukh = ((1 - α) * α * zbar * kbar^(α - 1) * hbar^(-α) * cbar -
       ( α * zbar * kbar^(α - 1) * hbar^(1 - α) ) *
       ( (1 - α) * zbar * kbar^α * hbar^(-α) ) ) / c2;

Uzz = - ( kbar^α * hbar^(1 - α) * kbar^α * hbar^(1 - α) ) / c2;

Uzi = kbar^α * hbar^(1 - α) / c2;

Uzh = ( (1 - α) * kbar^α * hbar^(-α) * cbar -
        kbar^α * hbar^(1 - α) * (1 - α) * zbar * kbar^α * hbar^(-α) ) / c2;

Uii = -1/c2;

Uih = (1 - α) * zbar * kbar^α * hbar^(-α) / c2;

Uhh = ( -α * (1 - α) * zbar * kbar^α * hbar^(-α - 1) * cbar -
        ( (1 - α) * zbar * kbar^α * hbar^(-α) ) * ( (1 - α) * zbar * kbar^α * hbar^(-α) ) ) / c2 - 
      a / ((1 - hbar)^2);

DH = [Uzz Ukz Uzi Uzh; Ukz Ukk Uki Ukh; Uzi Uki Uii Uih; Uzh Ukh Uih Uhh];

S = [zbar; kbar];
C = [ibar; hbar];

B = [ 1 0 0 0 0;
      1-ρ ρ	0 0 0;
      0 0 1-δ 1 0];
    
Σ = [0 	0 					0
       0 	σ_ε^2 	0
       0		0				0];

P, J, D = ricatti1(Ubar, DJ, DH, S, C, B, Σ, β);

println("The optimal value function is [1 z s]P0[1; z; s]+d, where P and d are given by:")
@show P
@show D
println("The policy function is x=J[1; z; s] where J  is:")
@show J

#SIMULATION
T = 115;
N = 100;
ss_mat, cc_mat = lqsimul_hansen(T, N, P, J, D, ss, mp);

stdv = mean(ss_mat, dims = 1);
corr = mean(cc_mat, dims = 1);
println("HANSEN: std(x)/std(y) corr(x,y) for y,i,c,k,h,prod")
@show [[1.36  4.24 0.42 0.36 0.7 0.68]'./1.36 [1 0.99 0.89 0.06 0.98 0.98]']
println("std(x) std(x)/std(y) corr(x,y) for y,i,c,k,h,prod:")
@show [stdv' (stdv./stdv[1])' corr']