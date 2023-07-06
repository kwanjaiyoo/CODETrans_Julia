function PIsimul(numperiod::Integer, kprime::AbstractMatrix, kgrid, shock::AbstractVector, markov::AbstractMatrix, parameter)
    @unpack β, δ, θ, σ = parameter()

    zt = zeros(numperiod);
    zt[1] = shock[1];
    st = zeros(Int, numperiod);
    st[1] = 1;
    epsi = rand(numperiod);
    for i in 2:numperiod
        j = 1;
        while sum(markov[st[i-1], 1:j]) < epsi[i]
            j += 1;
        end
        st[i] = j;
        zt[i] = shock[j];
    end

    kt = zeros(numperiod+1);
    yt = zeros(numperiod);
    ct = zeros(numperiod);
    rt = zeros(numperiod);
    
    kt[1] = ( (1/β + δ - 1) / θ)^(1 / (θ - 1));
    for i in 1:numperiod
        sz = 1;
        if zt[i] > 1
            sz = 2;
        end
        
        yt[i] = zt[i] * kt[i]^θ;
        mm = argmin(abs.(kgrid .- kt[i]));
        if kt[i] < kgrid[mm]
            mm = mm - 1;
        end
        weight = (k[mm+1] - kt[i]) / (k[mm+1] - k[mm]);
        kt[i+1] = weight * kprime[mm, sz] + (1-weight) * kprime[mm+1, sz];
        ct[i] = yt[i] + (1 - δ) * kt[i] - kt[i+1];
        rt[i] = θ * zt[i] * kt[i]^(θ - 1) - δ;
    end

    return ct, kt
end