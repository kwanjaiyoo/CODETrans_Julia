function policyiter(shock::AbstractVector, markov::AbstractMatrix, kgrid, nk::Integer, nθ::Integer, parameter)
    @unpack β, σ, δ, θ = parameter()
    
    r = zeros(nk, nθ);
    w = zeros(nk, nθ);
    c = zeros(nk, nθ);
    kpr = zeros(nk, nθ);
    cpp = zeros(nθ);
    cn = zeros(nk, nθ);
    kprn = zeros(nk, nθ);
    
    for m in 1:nθ
        for i in 1:nk
            r[i, m] = θ * shock[m] * kgrid[i]^(θ-1) + 1 - δ;
            w[i, m] = (1-θ) * shock[m] * kgrid[i]^θ;
            c[i, m] = max(.001, shock[m] * kgrid[i]^θ - δ * kgrid[i]);
            kpr[i, m] = max(kgrid[1], w[i, m] + r[i, m] * kgrid[i] - c[i, m]);
            kpr[i, m] = min(kpr[i, m], kgrid[end]);
        end
    end
    
    iter = 0;
    err = ones(2);
    interma = 0;
    while maximum(err) > .00001 && iter < 1000
        iter += 1;
        for m in 1:nθ
            for i in 1:nk
                if interma == 1
                    interp = [LinearInterpolation(kgrid, c[:, j]) for j in 1:nθ];
                    cpp = [interp[m](kpr[i, m]) for m in 1:nθ].^(-σ);
                else
                    mm = argmin(abs.(kpr[i, m] .- k));
                    if kpr[i, m] <= k[mm] && mm > 1
                        weight = (k[mm] - kpr[i, m]) / (k[mm] - k[mm-1]);
                        cpp = (weight * c[mm-1, :] + (1-weight) * c[mm,:]) .^ (-σ);
                    else
                        weight = (k[mm+1] - kpr[i, m]) / (k[mm+1] - k[mm]);
                        cpp = (weight * c[mm, :] + (1-weight) * c[mm+1,:]) .^ (-σ);
                    end
                    # interp = [LinearInterpolation(kgrid, kpr[:, j]) for j in 1:nθ];
                    # cpp = [interp[m](c[i, m]) for m in 1:nθ].^(-σ);
                end
                cn[i, m] = max(.001, ( β * sum( cpp .* (θ * shock * kpr[i,m]^(θ - 1) .+ 1 .- δ) .* markov[:, m] ) )^(-1/σ) );
                r[i, m] = θ * shock[m] * kgrid[i]^(θ - 1) + 1 - δ;
                w[i, m] = (1 - θ) * shock[m] * kgrid[i]^θ;
                kprn[i, m] = max(kgrid[1], w[i,m] + r[i,m] * kgrid[i] - cn[i,m]);
                kprn[i, m] = min(kprn[i, m], kgrid[end]);
            end            
        end
        err[1] = maximum(abs.(c - cn));
        err[2] = maximum(abs.(kpr - kprn));
        c = copy(cn);
        kpr = copy(kprn);
        
        @show maximum(err), iter
    end
    
    return c, kpr, r, w
end