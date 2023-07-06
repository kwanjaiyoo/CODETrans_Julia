function policyiter_hansen(shock::AbstractVector, markov::AbstractMatrix, kgrid, nk::Integer, nθ::Integer, parameter)
    @unpack β, σ, δ, θ, λ = parameter()
    
    r = zeros(nk, nθ);
    w = zeros(nk, nθ);
    c = zeros(nk, nθ);
    kpr = zeros(nk, nθ);
    cpp = zeros(nθ);
    cn = zeros(nk, nθ);
    kprn = zeros(nk, nθ);
    h = zeros(nk, nθ);    
    hp = zeros(nθ);
    hn = zeros(nk, nθ);
    
    lmin = .00001;
    lmax = 1 - lmin;
    
    for m in 1:nθ
        for i in 1:nk
            h[i, m]=0.3;
            r[i, m] = θ * shock[m] * h[i, m]^(1-θ) * kgrid[i]^(θ-1) + 1 - δ;
            w[i, m] = (1-θ) * shock[m] * h[i, m]^(-θ) * kgrid[i]^θ;
            c[i, m] = max(.001, shock[m] * h[i, m]^(1-θ) * kgrid[i]^θ - δ * kgrid[i]);
            kpr[i, m] = max(kgrid[1], w[i, m] * h[i, m] + r[i, m] * kgrid[i] - c[i, m]);
            kpr[i, m] = min(kpr[i, m], kgrid[end]);
        end
    end
    
    iter = 0;
    err = ones(3);
    interma = 0;
    while maximum(err) > .0001 #&& iter < 1000
        iter += 1;
        for m in 1:nθ
            for i in 1:nk
                if interma == 1
                    interp = [LinearInterpolation(kgrid, c[:, j]) for j in 1:nθ];
                    cpp = [interp[m](kpr[i, m]) for m in 1:nθ].^(-σ);
                    interph = [LinearInterpolation(kgrid, h[:, j]) for j in 1:nθ];
                    hp = [interph[m](kpr[i, m]) for m in 1:nθ];
                else
                    mm = argmin(abs.(kpr[i, m] .- k));
                    if kpr[i, m] <= k[mm] && mm > 1
                        weight = (k[mm] - kpr[i, m]) / (k[mm] - k[mm-1]);
                        cpp = (weight * c[mm-1, :] + (1-weight) * c[mm,:]) .^ (-σ);
                        hp=((weight * h[mm-1, :] + (1-weight)* h[mm, :]));
                    else
                        weight = (k[mm+1] - kpr[i, m]) / (k[mm+1] - k[mm]);
                        cpp = (weight * c[mm, :] + (1-weight) * c[mm+1,:]) .^ (-σ);
                    end
                    # interp = [LinearInterpolation(kgrid, kpr[:, j]) for j in 1:nθ];
                    # cpp = [interp[m](c[i, m]) for m in 1:nθ].^(-σ);
                end
                cn[i, m] = max(.001, ( β * sum( cpp .* (θ * shock .* hp.^(1-θ) * kpr[i,m]^(θ - 1) .+ 1 .- δ) .* markov[:, m] ) )^(-1/σ) );
                lfoc(x) = a * cn[i, m] - (1 - θ) * shock[m] * kgrid[i]^θ * x^(-θ) * (1-x);
                hn[i, m] = find_zero(lfoc, (lmin, lmax));
                if h[i, m] < 0
                    h[i, m] = 0;
                elseif h[i, m] > 1
                    h[i, m] = 1;
                end

                r[i, m] = θ * shock[m] * hn[i, m]^(1 - θ) * kgrid[i]^(θ - 1) + 1 - δ;
                w[i, m] = (1 - θ) * shock[m] * hn[i, m]^(-θ) * kgrid[i]^θ;
                kprn[i, m] = max(kgrid[1], w[i,m] * hn[i, m] + r[i,m] * kgrid[i] - cn[i,m]);
                kprn[i, m] = min(kprn[i, m], kgrid[end]);
            end            
        end
        err[1] = maximum(abs.(c - cn));
        err[2] = maximum(abs.(kpr - kprn));
        err[3] = maximum(abs.(h - hn));
        
        c = λ * c + (1-λ) * cn;
        kpr = λ * kpr + (1-λ) * kprn;
        h = λ * h + (1-λ) * hn;
        @show maximum(err), iter
    end
    
    return c, kpr, h, r, w
end