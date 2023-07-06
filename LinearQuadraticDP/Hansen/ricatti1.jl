function ricatti1(U, Jacobian, Hessian, states, controls, B, Sigma, beta, tol = 1e-7)
    
    ns = length(states);
    nc = length(controls);
    
    WW = [states; controls];
    Q11 = U - WW' * Jacobian + .5 * WW' * Hessian * WW;
    Q12 = .5 * (Jacobian - Hessian * WW);
    Q22 = .5 * Hessian;
    
    QQ = [Q11 Q12';
    Q12 Q22];
    
    nq = ns + nc + 1;
    
    #PARTITION Q TO SEPARATE STATES AND CONTROLS
    Qff = QQ[1:ns+1, 1:ns+1];
    Qfx = QQ[ns+2:nq, 1:ns+1];
    Qxx = QQ[nq-nc+1:nq, nq-nc+1:nq];
    
    #INITIALIZE MATRICES
    P0 = -.1 * I;
    P1  = ones(ns + 1, ns + 1);
    
    #ITERATE ON BELLMAN'S EQUATION UNTIL CONVERGENCE
    while norm(P1 - P0) > tol
        P1 = copy(P0);
        M = B' * P0 * B;
        Mff = M[1:ns+1, 1:ns+1];
        Mfx = M[ns+2:nq, 1:ns+1];
        Mxx = M[nq-nc+1:nq, nq-nc+1:nq];
        
        P0 = Qff + beta * Mff - (Qfx + beta * Mfx)' * inv(Qxx + beta * Mxx) * (Qfx + beta * Mfx);
    end
    
    M = B' * P0 * B;
    Mff = M[1:ns+1, 1:ns+1];
    Mfx = M[ns+2:nq, 1:ns+1];
    Mxx = M[nq-nc+1:nq, nq-nc+1:nq];
    J = -inv(Qxx + beta * Mxx) * (Qfx + beta * Mfx);
    d = beta / (1 - beta) * tr(P0 * Sigma);
    
    return P1, J, d
end