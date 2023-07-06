function HansenDDPBellman(markov::AbstractMatrix, kgrid, θgrid, nk::Integer, nθ::Integer, tol::Real, parameter)
    @unpack α, δ, a = parameter()
    
    gk = nk^2;
    c = zeros(gk,nθ);
    l = zeros(gk,nθ);
    lmin = 1e-5;
    lmax = 1-lmin;

    for t in 1:nθ 
        for i in 1:nk
            for j in 1:nk
                lfoc(x) = (θgrid[t] * kgrid[i]^α * x^(1-α) + (1-δ) * kgrid[i] - kgrid[j])/(1-x) - (1-α)/a * θgrid[t] * kgrid[i]^α * x^(-α); 
                if kgrid[j] - (1-δ) * kgrid[i] > 0 && ( (kgrid[j] - (1-δ) * kgrid[i]) / (θgrid[t] * kgrid[i]^α) )^( 1 / (1-α) ) < lmax
                    L = find_zero(lfoc, ( ((kgrid[j] - (1-δ) * kgrid[i]) / (θgrid[t] * kgrid[i]^α)) ^ (1/(1-α)), lmax));             
                elseif kgrid[j] - (1-δ) * kgrid[i] < 0
                    L = find_zero(lfoc, (lmin,lmax));             
                end

                if L < 0
                    L = lmin;
                elseif L > 1
                    L = lmax;
                end

                l[(i-1) * nk + j, t] = L;
                c[(i-1) * nk + j, t] = θgrid[t] * kgrid[i]^α * L^(1-α) + (1-δ) * kgrid[i] - kgrid[j];
                
                if c[(i-1) * nk + j, t] < 0
                    c[(i-1) * nk + j, t] = 1e-12;                    
                end
            end
        end
    end
    
    U = log.(c) + a * log.(1 .- l);
    V0 = zeros(nk, nθ);
    V1 = zeros(nk, nθ);
    index = zeros(Integer, nk, nθ);
    distance = Inf;
    iter = 0;
    
    while distance > tol
        for t in 1:nθ
            for i in 1:nk
                Value = U[ (i-1) * nk + 1 : i * nk, t] + β * V0 * markov[t, :];
                V1[i, t] = maximum(Value);
                index[i, t] = argmax(Value);
            end                
        end
        distance = norm(V1 - V0);
        iter += 1;
        @show distance, iter
        V0 = copy(V1);
    end

    #POLICY FUNCTIONS
    poll = zeros(nk, nθ);
    for t in 1:nθ
        for i in 1:nk
            poll[i, t]=l[(i-1) * nk + index[i, t], t];            
        end        
    end
    polk = kgrid[index];
    poly = repeat(transpose(θgrid), nk, 1) .* repeat(kgrid.^α, 1, nθ) .* (poll.^(1-α));
    poli = polk - (1-δ) * repeat(kgrid, 1, nθ);
    polc = poly - poli;
    
    return V1, polk, poll, poly, poli, polc, index
end