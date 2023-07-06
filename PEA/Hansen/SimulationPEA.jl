function SimulationPEA(hp::Function, numperiod::Integer, numsimul::Integer, solPEA, parameter, ss)
    @unpack α, β, ρ, δ, σ_ε = parameter()
    @unpack k_ss = ss()

    ss = zeros(numsimul, 6);
    cc = zeros(numsimul, 6);
    shockd = Normal(0, σ_ε);
    lmin = 1e-5;
    lmax = 1 - lmin;
    for j in 1:numsimul
        kt = zeros(numperiod+1);
        kt[1] = k_ss;
        int = zeros(numperiod);
        ct = zeros(numperiod);
        yt = zeros(numperiod);
        lt = zeros(numperiod);
        prodt = zeros(numperiod);
        Pea = zeros(numperiod);
        zt = ones(numperiod+1);
        shockt = rand(shockd, numperiod+1);
        for t in 1:numperiod
            zt[t+1] = exp(ρ * log(zt[t]) + shockt[t]);
            Pea[t] = exp(solPEA[1] + solPEA[2] * log(zt[t]) + solPEA[3] * log(kt[t]));
            ct[t] = 1 / (β * Pea[t]);
            func(x) = a * ct[t] - (1 - α) * zt[t] * kt[t]^α * x^(-α) * (1-x);
            lt[t] = find_zero(func, (lmin, lmax));
            if lt[t] < 0
                lt[t] = 0;
            elseif lt[t] > 1
                lt[t] = 1;
            end
            yt[t] = zt[t] * kt[t]^α * lt[t]^(1-α);
            kt[t+1] = yt[t] + (1 - δ) * kt[t] - ct[t];
            int[t] = yt[t] - ct[t];
            prodt[t] = yt[t] / lt[t];
        end
        z = zt[1:numperiod];
        k = log.(kt[1:numperiod]);
        y = log.(yt);
        c = log.(ct);
        invest = log.(int);
        h = log.(lt);
        prod = log.(prodt);
        dhp = hp([y invest c k h prod], 1600)
        ss[j, :] = std(dhp, dims = 1) * 100;
        cc[j, :] = cor(dhp)[1, :]';
    end
    
    return ss, cc
end