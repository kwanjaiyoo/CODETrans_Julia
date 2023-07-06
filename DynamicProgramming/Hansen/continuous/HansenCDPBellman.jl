function HansenCDPBEllman(markov::AbstractMatrix, kgrid, θgrid, nk::Integer, nθ::Integer, tol::Real, parameters, numiter = 1000)
    @unpack α, β, δ, a = parameters()
    lmin = 1e-5;
    lmax = 1-lmin;

    V = zeros(nk, nθ);
    V_new = similar(V);
    polk = similar(V);
    poll = similar(V);
    
    u(c, l) = log(c) + a * log(1-l);
    f(k, l) = k^α * l^(1-α);
        
    distance = Inf;
    iter = 0;
    while distance > tol && iter <= numiter
        interp = [LinearInterpolation(kgrid, V[:, i]) for i in eachindex(θgrid)];
        for (j, θ_val) in enumerate(θgrid)
            for (i, k_val) in enumerate(kgrid)
                nextperiod(kpr) = sum( markov[j, jj] * interp[jj](kpr) for jj in eachindex(θgrid) );
                lfoc(x, kpr) = (θ_val * k_val^α * x^(1-α) + (1-δ) * k_val - kpr)/(1-x) - (1-α)/a * θ_val * k_val^α * x^(-α); # big problem: kpr = ?
                G(kpr) = x -> lfoc(x, kpr);
                l(kpr) = find_zero(G(kpr), (lmin, lmax));
                yd(kpr) = θ_val * f(k_val, l(kpr)) + (1 - δ) * k_val;
                value(kpr) = u(yd(kpr) - kpr, l(kpr)) + β * nextperiod(kpr);
                Tv = maximize(value, kgrid[1], kgrid[end]);
                V_new[i, j] = maximum(Tv);
                polk[i, j] = maximizer(Tv);
                poll[i, j] = l(polk[i, j]);
            end
        end
        
        distance = norm(V - V_new);
        iter = iter + 1;
        @show distance, iter;
        V = copy(V_new);
         
    end

    poly = repeat(transpose(θgrid), nk, 1) .* repeat(kgrid.^α, 1, nθ) .* (poll.^(1-α));
    poli = polk - (1-δ) * repeat(kgrid, 1, nθ);
    polc = poly - poli;
    
    return V_new, polk, poll, poly, polc, poli
end