function contiBellman(markov::AbstractMatrix, kgrid, θgrid, nk::Integer, nθ::Integer, parameters, tol = 1e-5, numiter = 1000)
    @unpack α, β, δ = parameters()
    
    V = zeros(nk, nθ);
    V_new = similar(V);
    polk = similar(V);
    
    f(x) = x^α;
    u(x) = log(x);
        
    distance = Inf;
    iter = 0;
    while distance > tol && iter <= numiter
        
        interp = [LinearInterpolation(kgrid, V[:, i]) for i in eachindex(θgrid)]
        for (j, θ_val) in enumerate(θgrid)
            for (i, k_val) in enumerate(kgrid)
                yd = θ_val * f(k_val) + (1 - δ) * k_val;
                nextperiod(kpr) = sum( markov[j, jj] * interp[jj](kpr) for jj in eachindex(θgrid) );
                value(kpr) = u(yd - kpr) + β * nextperiod(kpr);
                Tv = maximize(value, kgrid[1], min(kgrid[end], yd - 1e-5) );
                V_new[i, j] = maximum(Tv);
                polk[i, j] = maximizer(Tv);
            end
        end
        
        distance = norm(V - V_new);
        iter = iter + 1;
        @show distance, iter;
        V = copy(V_new); 
         
    end  
    
    return V_new, polk
end