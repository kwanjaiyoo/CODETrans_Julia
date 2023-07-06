function PEA_lfoc(fun::Function, θ::Vector, parameter, ss, etol = 1e-4)
    
    @unpack α, β, δ, T, a, λ = parameter()
    @unpack l_ss, k_ss, y_ss, i_ss, c_ss = ss()
    
    # LABOR WORKING HOURS
    lmin = 1e-5;
    lmax = 1-lmin;

    c = zeros(T+1);
    k = zeros(T+1);
    inv = zeros(T+1);
    l = zeros(T+1);
    y = zeros(T+1);
    Pea = zeros(T+1);
    
    #INITIAL PARAMETERS FOR THE PEA FUNCTIONS
    bita = zeros(3);
    bita[1] = (α * k_ss^(α-1) * l_ss^(1-α) + 1-δ) / c_ss;
    bita0 = ones(3);
    bita = (bita .- λ)/(1 .- λ);
    
    iter = 0;
    distance = Inf;
    while distance > etol
        iter += 1;
        bita = λ .* bita0 + (1-λ) .* bita;
        bita0 = copy(bita);
        
        k[1] = k_ss;
        c[1] = c_ss;
        inv[1] = i_ss;
        y[1] = y_ss;
        l[1] = l_ss;
        
        for t in 2:T+1 
            Pea[t] = exp( bita[1] + bita[2] * log(θ[t]) + bita[3] * log(k[t-1]) );
            c[t] = 1 / (β * Pea[t]);
            func(x) = a * c[t] - (1-α) * θ[t] * k[t-1]^α * x^(-α)*(1-x);
            l[t] = find_zero(func, (lmin,lmax));
            if l[t] < 0
                l[t] = 0;
            elseif l[t] > 1
                l[t] = 1;
            end
            y[t] = θ[t] * k[t-1]^α * l[t]^(1-α);
            k[t] = θ[t] * k[t-1]^α * l[t]^(1-α) + (1-δ) * k[t-1] - c[t];
            inv[t] = y[t] - c[t];
        end
        E3 = (c[3:T+1].^(-1)) .* (α * θ[3:T+1] .* (k[2:T].^(α-1) .* l[3:T+1] .^ (1-α)) .+ (1-δ));

        bita = fun(E3, log.(θ[2:T]), log.(k[1:T-1]), T, bita0);
        @show mean(k) mean(c) mean(inv) mean(l) mean(y);
        distance = maximum(abs.(bita - bita0));
        @show distance, iter;
    end
    
    return bita
end