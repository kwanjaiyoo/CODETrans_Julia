function dpnew(ir; param)
    # PAR: parameter vector
    # DSS: deterministic steady state
    # SSS: state space of shock
    # TM: transition matrix

    β = PAR[2]
    δ = PAR[4]

    cb = DSS[2]

    sss = SSS
    tm = TM
    ns = length(sss)

    kg = Kgrid
    nk = length(kg)

    # utility function
    fu(x; γ = PAR[3]) = x.^(1 - γ) / (1 - γ)
    # production function
    fprod(x; α = PAR[1]) = x.^α

    # initial value to be half of deterministic steady state
    vold = zeros(nk, ns)
    vnew = 0.5 * fu(cb) * ones(nk, ns) / (1 - β)

    @time begin

        iv = kron(ones(nk, 1), kg) - kron((1 - δ) * kg, ones(nk, 1))
        if ir == 1
            iv = max(iv, zeros(nk^2, 1))
        end

        iv = kron(ones(ns, 1), iv)

        c = kron( sss, kron(fprod(kg), ones(nk, 1)) ) - iv;
        Ic = findall(x -> x >= 0, c);
        u = -Inf * ones(nk^2 * ns, 1); 
        u[Ic] = fu( c[Ic] );
        u = reshape(u, (nk, nk, ns));

        while norm(vold .- vnew) > 1e-6
           @show norm(vold .- vnew)
           vold = copy(vnew);
           vful = vold * tm';
           vful = reshape( kron(ones(nk, 1), vful), (nk, nk, ns) );
           vnew = maximum(u + β * vful, dims = 1);
           gk = argmax(u + β * vful, dims = 1);
           vnew = reshape(vnew, (nk, ns));
           gk = reshape(gk, (nk, ns));            
        end
    end

    return (v = vnew, gk = gk)
end