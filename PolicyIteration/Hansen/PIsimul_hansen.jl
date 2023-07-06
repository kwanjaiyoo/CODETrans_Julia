function PIsimul_hansen(numperiod::Integer, numsimul::Integer, kprime::AbstractMatrix, labor::AbstractMatrix, kgrid, shock::AbstractVector, markov::AbstractMatrix, parameter, steadystate)
    @unpack β, δ, θ, σ = parameter()
    @unpack k_ss = steadystate()

    std_mat = zeros(numsimul, 6);
    cc_mat = zeros(numsimul, 6);
    for j in 1:numsimul
        S = markovchain(markov, numperiod);
        sz = zeros(numperiod + 1);
        kt = zeros(numperiod + 1);
        int = zeros(numperiod);
        ct = zeros(numperiod);
        yt = zeros(numperiod);
        lt = zeros(numperiod);
        prodt = zeros(numperiod);
        zt = zeros(numperiod);
        kt[1] = k_ss;
        interpk = [LinearInterpolation(kgrid, kprime[:, j]) for j in 1:length(shock)];
        interph = [LinearInterpolation(kgrid, labor[:, j]) for j in 1:length(shock)];
        for i in 1:numperiod
            sz = S[i];
            zt[i] = shock[sz];
            kt[i+1] = interpk[sz](kt[i]);
            lt[i] = interph[sz](kt[i]);
            yt[i] = zt[i] * lt[i]^(1-θ) * kt[i]^θ;
            ct[i] = yt[i] + (1-δ) * kt[i] - kt[i+1];
            int[i] = kt[i+1] - (1-δ) * kt[i];
            prodt[i] = yt[i] / lt[i];
        end
        kk = log.(kt[1:numperiod]);
        yy = log.(yt);
        cc = log.(ct);
        inn = log.(int);
        hh = log.(lt);
        prodd = log.(prodt);
        
        dhp = hp1([yy inn cc kk hh prodd], 1600);
        std_mat[j, :] = std(dhp, dims = 1) * 100;
        cc_mat[j, :] = cor(dhp)[1, :]';
    end
    
    
    return std_mat, cc_mat
end