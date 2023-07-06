function PEA(fun::Function, θ::Vector, parameter, steadystate, etol = 1e-3)
    
    @unpack α, β, δ, T, sig, ρ_z, σ_ε, λ = parameter()
    @unpack zs, ks, ys, is, cs = steadystate()
    
    c = zeros(T+1);
    k = zeros(T+1);
    inv = zeros(T+1);
    Pea = zeros(T+1);
    
    #INITIAL PARAMETERS FOR THE PEA FUNCTIONS
    bita = zeros(3);
    bita0 = ones(3);
    bita = (bita .- λ)/(1 .- λ);
    
    iter = 0;
    distance = Inf;
    while distance > etol
        iter += 1;
        bita = λ .* bita0 + (1-λ) .* bita;
        bita0 = copy(bita);
        
        k[1] = ks;
        c[1] = cs;
        inv[1] = is;
        
        for t in 2:T+1 
            Pea[t] = exp( bita[1] + bita[2] * log(θ[t]) + bita[3] * log(k[t-1]) );
            c[t] = (β * Pea[t])^(-1/sig);
            inv[t]= θ[t] * (k[t-1]^α) - c[t];
            k[t] = inv[t] + (1 - δ) * k[t-1];
        end
        E3 = (c[3:T+1].^(-sig)) .* (α * θ[3:T+1] .* (k[2:T].^(α - 1)) .+ (1 - δ));

        bita = fun(E3, log.(θ[2:T]), log.(k[1:T-1]), T, bita0);
        @show mean(k) mean(c) mean(inv);
        distance = maximum(abs.(bita - bita0));
        @show distance, iter;
    end
    
    return bita
end