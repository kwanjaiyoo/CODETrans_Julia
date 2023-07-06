function HansenSimul(hp::Function, numperiod::Integer, numsimul::Integer, nk::Integer, markov::AbstractMatrix, θgrid, kpol, lpol, cpol, optk, parameter)

    @unpack α, δ, a = parameter()
    
    tind = zeros(Integer, numsimul, numperiod+1);
    kopt = zeros(numsimul, numperiod+1);
    zt = zeros(numsimul, numperiod+1);
    lopt = zeros(numsimul, numperiod);
    output = zeros(numsimul, numperiod);
    invest = zeros(numsimul, numperiod);
    cons = zeros(numsimul, numperiod);
    ss_mat = zeros(numsimul, 6); 
    cc_mat = zeros(numsimul, 6);
    for i in 1:numsimul
        tind[i, 1] = 4
        zt[i, 1] = θgrid[tind[i, 1]];
        indk = trunc(Int, nk/2);
        kopt[i, 1] = kpol[indk, 1];
        
        for t in 1:numperiod
            indk = optk[indk, tind[i, t]];
            kopt[i, t+1] = kpol[indk, tind[i, t]];
            lopt[i, t] = lpol[indk, tind[i, t]];
            output[i, t] = zt[i, t] * (kopt[i, t]^α * lopt[i, t]^(1-α));
            invest[i, t] = kopt[i, t+1] - (1-δ) * kopt[i, t];
            cons[i, t] = cpol[indk, tind[i, t]];
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
        logk = log.(kopt[i, 1:T]);
        logl = log.(lopt[i, 1:T]);
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