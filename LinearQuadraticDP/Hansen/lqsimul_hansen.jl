function lqsimul_hansen(numperiod::Integer, numsimul::Integer, P::AbstractMatrix, J::AbstractMatrix, D::Real, steadystate, parameter)

    @unpack α, β, δ, ρ, σ_ε = parameter()
    @unpack kbar = steadystate()

    ss = zeros(numsimul, 6);
    cc = zeros(numsimul, 6);
    for j in 1:numsimul
        r = randn(numperiod + 1);
        z = ones(numperiod + 1);
        k = zeros(numperiod + 1);
        k[1] = kbar;
        invest = zeros(numperiod);
        c = zeros(numperiod);
        y = zeros(numperiod);
        h = zeros(numperiod);
        prod = zeros(numperiod);

        for i in 1:numperiod
            invest[i] = J[1, 1] * 1 + J[1, 2] * z[i] + J[1, 3] * k[i];
            h[i] = J[2, 1] * 1 + J[2, 2] * z[i] + J[2, 3] * k[i]; 
            y[i] = z[i] * k[i]^α * h[i]^(1 - α);
            c[i] = y[i] - invest[i];
            k[i+1] = (1 - δ) * k[i] + invest[i];
            z[i+1] = 1 - ρ + ρ * z[i] + σ_ε * r[i];
            prod[i] = y[i] / h[i];
        end
        z = z[1:T];
        k = log.(k[1:T]);
        y = log.(y);
        c = log.(c);
        invest = log.(invest);
        h = log.(h);
        prod = log.(prod);        
        dhp = hp1([y invest c k h prod], 1600);
        ss[j, :] = std(dhp, dims = 1) * 100;
        cc[j, :] = cor(dhp)[1, :]';
    end
    
    return ss, cc
end