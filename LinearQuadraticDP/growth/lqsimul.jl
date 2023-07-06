function lqsimul(numperiod::Integer, numsimul::Integer, P::AbstractMatrix, J::AbstractMatrix, D::Real, steadystate, parameter)

    @unpack α, β, δ, ρ, σ_ε = parameter()
    @unpack ks = steadystate()

    ss = zeros(numsimul, 4);
    cc = zeros(numsimul, 4);
    for j in 1:numsimul
        r = randn(numperiod + 1);
        z = ones(numperiod + 1);
        z[1] = 0;
        k = zeros(numperiod + 1);
        k[1] = ks;
        invest = zeros(numperiod);
        c = zeros(numperiod);
        y = zeros(numperiod);

        for i in 1:numperiod
            y[i] = exp(z[i]) * k[i]^α;
            invest[i] = J[1] * 1 + J[2] * z[i] + J[3] * k[i];
            c[i] = y[i] - invest[i];
            k[i+1] = (1 - δ) * k[i] + invest[i];
            z[i+1] = ρ * z[i] + σ_ε * r[i];
        end
        z = z[1:T];
        k = log.(k[1:T]);
        y = log.(y);
        c = log.(c);
        invest = log.(invest);
        
        dhp = hp1([y c invest k], 1600);
        ss[j, :] = std(dhp, dims = 1) * 100;
        cc[j, :] = cor(dhp)[1, :]';
    end
    
    return ss, cc
end