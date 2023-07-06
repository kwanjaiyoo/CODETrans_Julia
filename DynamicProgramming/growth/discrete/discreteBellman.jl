function discreteBellman(markov:: AbstractMatrix, numiter:: Integer, tol:: Real, nk:: Integer, nθ:: Integer, consumption:: AbstractMatrix; mp)

    @unpack γ, β = mp()

    V0 = zeros(nk ,nθ);
    V1 = zeros(nk, nθ);
    index = zeros(Integer, nk, nθ);
    distance = Inf;
    iter = 0;
    
    # Utility function
    if γ == 1
        U = log.(consumption);
    else
        U = consumption.^(1-γ) / (1-γ);
    end
    
    while distance > tol && iter <= numiter
        for j in 1:nθ
            for i in 1:nk
                Value = U[ (i-1) * nk + 1 : (i-1) * nk + nk, j] + β * V0 * markov[j, :];
                V1[i, j] = maximum(Value);
                index[i, j] = argmax(Value);
            end
        end
        distance = norm(V1 - V0);
        iter = iter + 1;
        @show distance, iter;
        V0 = copy(V1); # V0 = V1 :: wrong -> In Julia, it means to create a new binding V0 which refer to whatever V1 is. When V1 is updated, V0 is updated automatically.
    end 
    return (Vstar = V1, kindex = index)   
end