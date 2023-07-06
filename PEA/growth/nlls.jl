function nlls(DEP, IND1, IND2, T, IBETA)

    GBETA = zeros(3);
    HALT = 0;
    
    MAX_IT = 100;
    smooth = .6;
    prec = .001;
    iter = 0;
    while HALT == 0 && iter < MAX_IT
        IBETA = smooth * IBETA + (1 - smooth) * GBETA;
        DER1 = exp.( IBETA[1] * ones(T-1) + IBETA[2] * IND1 + IBETA[3] * IND2 );
        F = copy(DER1);
        DER2 = F .* IND1;
        DER3 = F .* IND2;
        DER = [DER1 DER2 DER3];
        # 1.1 ESTIMATE COEFFICIENT OF LINEARIZED EQUATION
        #= 
        DEP - F + DER(Tx3)*IBETA(3x1) = DER(Tx3)*GBETA(3x1) + Error
        =#
        Y = DEP - F + DER * IBETA;
        GBETA = DER \ Y;
        diffnlls = sum(abs.(GBETA - IBETA));

        if diffnlls < prec
            HALT = 1;            
        end

        iter += 1;
    end
    return GBETA
end