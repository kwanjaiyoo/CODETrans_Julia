function HansenCDPsimul(hp::Function, numperiod::Integer, numsimul::Integer, markov::AbstractMatrix, θgrid, kgrid, kpol, lpol, cpol, parameter)

    @unpack α, δ, a = parameter()
    
    tind = zeros(Integer, numsimul, numperiod+1);
    zt = zeros(numsimul, numperiod+1);
    ktime = zeros(numsimul, numperiod+1);
    ltime = zeros(numsimul, numperiod);
    output = zeros(numsimul, numperiod);
    invest = zeros(numsimul, numperiod);
    cons = zeros(numsimul, numperiod);
    ss_mat = zeros(numsimul, 6); 
    cc_mat = zeros(numsimul, 6);

    kpritp = [LinearInterpolation(kgrid, kpol[:, j]) for j in eachindex(θgrid)];
    litp = [LinearInterpolation(kgrid, lpol[:, j]) for j in eachindex(θgrid)];
    citp = [LinearInterpolation(kgrid, cpol[:, j]) for j in eachindex(θgrid)];
    for i in 1:numsimul
        tind[i, 1] = 4
        zt[i, 1] = θgrid[tind[i, 1]];
        kind = 34;
        ktime[i, 1] = kgrid[kind];
        for t in 1:numperiod
            ktime[i, t+1] = kpritp[tind[i, t]](ktime[i, t]);
            ltime[i, t] = litp[tind[i, t]](ktime[i, t]);
            output[i, t] = zt[i, t] * (ktime[i, t]^α * ltime[i, t]^(1-α));
            invest[i, t] = ktime[i, t+1] - (1-δ) * ktime[i, t];
            cons[i, t] = citp[tind[i, t]](ktime[i, t]);
            shock = rand();
            j = 1;
            while sum(markov[tind[i, t], 1:j]) < shock
                j += 1;
            end
            tind[i, t+1] = j;
            zt[i, t+1] = θgrid[j];
        end
        logy = log.(output[i, 1:T]);
        logc = log.(cons[i, 1:T]);
        loginv = log.(invest[i, 1:T]);
        logk = log.(ktime[i, 1:T]);
        logl = log.(ltime[i, 1:T]);
        logz = log.(zt[i, 1:T]);

        dhp, dtr = hp([logy logc loginv logk logl logz], 1600);
        ss_mat[i, :] = std(dhp, dims = 1) * 100;
        cc_mat[i, :] = cor(dhp)[1, :]';
    end
    stdv = mean(ss_mat, dims = 1);
    stdv_stdv = std(ss_mat, dims = 1);
    corr = mean(cc_mat, dims = 1);
    corr_stdv = std(cc_mat, dims = 1);

    return stdv, stdv_stdv, corr, corr_stdv
end