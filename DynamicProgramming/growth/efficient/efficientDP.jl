using LinearAlgebra, Statistics, Plots, NLsolve, Optim, Random, Interpolations, Parameters
using Optim: maximizer, maximum

include("dpnew.jl");

α = .33;
β = .9;
σ = 2;
δ = .1;

# Steady state
kb = ( 1/(α*β) - (1-δ) )^( 1/α );
ib = δ * kb;
cb = kb^α - ib;

# Capital grid
NK = 200;
kmax = 1.5 * kb;
kmin = 0.5 * kb;

param = (PAR = [α, β, σ, δ], SSS = [0.9, 1.1], DSS = [kb, cb, ib],
                Kgrid = range(kmin, kmax, length = NK), TM = [0.8 0.2; 0.2 0.8]);
@unpack PAR, SSS, DSS, Kgrid, TM = param
results = dpnew(0; param);
gc = (SSS * ones(1, NK))' .* ((Kgrid.^α) * ones(1, length(SSS))) + (1 - δ) * Kgrid * ones(1, length(SSS)) - Kgrid[results.gk];


#Plots
#Make plots of value and policy functions
fig_value = plot(Kgrid, v[:, 1], linecolor=:black, xlabel="capital stock", label="θ_min");
plot!(Kgrid, v[:, end], linecolor=:red, label="θ_max");
plot!(legend=:bottomright, title="Value function")

fig_polk = plot(Kgrid, Kgrid[gk[:, 1]], title="Capital policy functions", xlabel="capital stock", label="θ_min");
plot!(Kgrid, Kgrid[gk[:, end]], label="θ_max");
plot!(Kgrid, Kgrid, linestyle = :dot )
plot!(legend=:bottomright)

fig_polc = plot(Kgrid, gc[:, 1], title="Consumption policy functions", xlabel="capital stock", label="θ_min");
plot!(Kgrid, gc[:, end], label="θ_max");
plot!(legend=:bottomright)

fig_policies = plot(fig_polk, fig_polc, layout=(2, 1), legend=false)



