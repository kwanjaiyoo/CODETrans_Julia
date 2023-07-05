#=

This file includes all Julia functions for Markov theory from graduate computational methods class taught by prof. Eva Carceles-Poveda from Stony Brook University.
    By executing this file, all functions below are generated and stored in Julia workspace.
    
    Several basic packages are assumed to be installed already into global environment.
    Packages are imported with "using" in the beginning.
    
    To install packages, please refer to the tutorial document.
    
    =#
    using LinearAlgebra, Distributions, Plots, Statistics
    
    #=  Function: y = lnshock(ρ, σ_ϵ, T)
    
    This function generates a T-dimensional lognormal shock with mean zero, persistence ρ and innovation standard deviation σ.
        
        Arguments
        - ρ::Real : persistence
        - σ::Real : standard deviation of ϵ
        - T::Integer : number of periods
        
        =#
        function lnshock(ρ::Real, σ_ϵ::Real, T = 200)
            
            r = randn(T+1);
            y = ones(T+1);
            
            for i in 2:T+1
                y[i] = exp( ρ * log( y[i-1]) + σ_ϵ * r[i] );     
            end
            return y
        end
        
        function lnshockM(μ::Real, ρ::Real, σ_ϵ::Real, T = 200)
            # This file generates a T-dimensional lognormal shock with mean zero, persistence ρ and innovation standard deviation σ_ϵ
            
            r = randn(T+1);
            y = ones(T+1) .* exp( μ / (1-ρ) );
            
            for i in 2:T+1
                y[i] = exp( μ + ρ * log( y[i-1]) + σ_ϵ * r[i] );     
            end
            return y
        end
        
        function iid(prob::AbstractVector, T = 100, me = 1)
            
            # This function generates a simulation from an iid random variable
            # prob: probability vector
            # T: the number of periods to be simulated
            # me: method ( two possibilities are included and it is the first by default )
            # y: shock realization return
            
            if typeof(T) != Int # if T is not an integer
                println("The T input must be an integer scalar")
                return
            end
            
            if abs( sum(prob) -  1 ) > 1e-10
                println("The probabilities do not sum to one. Normalize it")
                prob = prob ./ sum(prob);
            end
            
            #Creating the shock realization
            X = rand(T);
            m = length(prob);
            y = zeros(Int, T);
            if me == 1
                for i in 1:T
                    for j in 1:m
                        if X[i] < sum( prob[1:j] )
                            y[i] = j;
                            break
                        else
                            j = j + 1;                    
                        end   
                    end
                end
            elseif me == 2
                P = cumsum(prob);
                for i in m:-1:1
                    j = findall(x -> x < P[i], X)
                    y[j] .= i;
                end
            end
            return y
        end
        
        #= This function generates a simulation from a Markov chain.
            # Trans: transition matrix
            # T: number of periods
            # s0: initial state
            # me: method
            =#
            
            function markovchain(Trans::AbstractMatrix, T = 100, s0 = 1, me = 1)
                
                # Check whether the transition matrix is square
                s1, s2 = size(Trans);
                if s1 != s2
                    println("Error: Transition matrix must be square")
                    return
                end
                
                for k in 1:s1
                    if abs( sum(Trans[k, :]) - 1 ) > 1e-10
                        println("Row $k does not sum to one")
                        println("Normalizing row $k")
                        Trans[k, :] = Trans[k, :] ./ sum(Trans[k, :]);
                    end        
                end
                
                if s0 < 1 | s0 > s1
                    println("Initial state $s0 is out of range")
                    println("Initial state defaulting to 1")
                    s0 = 1;
                end
                
                # Create the shock realizations
                X = rand(T);
                y = zeros(Int, T+1);
                y[1] = s0;
                if me == 1
                    for t in 2:T
                        for j in 1:s1
                            if X[t - 1] < sum( Trans[ y[t - 1], 1:j ] )
                                y[t] = j;
                                break
                            else
                                j = j + 1;
                            end
                        end            
                    end
                elseif me == 2 # Not finished yet...
                    cum = zeros(s1, s2)
                    cum[:, 1] = Trans[:, 1];
                    cum[:, 2] = sum(Trans, dims = 2);
                end
                
                return y
            end
            
            function ergodic(Trans::Matrix, me = 1)    
                # Check whether the transition matrix is square
                s1, s2 = size(Trans);
                if s1 != s2
                    println("Error: Transition matrix must be square")
                    return
                end
                
                # Check whether the row-sum of transition matrix equals 1
                # If not, normalize to 1
                for i = 1:s1
                    if abs( sum(Trans[i,:]) - 1 ) > 1e-10
                        println("Row $i does not sum to 1")
                        println("Normalizing row $i")
                        Trans[i,:] = Trans[i,:] ./ sum(Trans[i,:])
                    end
                end
                
                va, vec = eigen(Trans');
                if count(i -> abs(i - 1) < 1e-10, va) == 1
                    if me == 1
                        #Solving the equation system
                        M = I - Trans';
                        one = ones(1,s1);
                        MM = [M[1:s1-1,:]; one];
                        V = [zeros(s1-1,1); 1];
                        P = MM \ V;
                    elseif me == 2
                        trans = Trans';
                        p0 = (1 / s1) * ones(s1, 1);
                        test = 1;
                        while test > 1e-8
                            p1 = trans * p0;
                            test = norm(p1 - p0);
                            p0 = p1;
                        end
                        P = p0;
                    elseif me == 3
                        # Use the limit of transition matrix
                        P = Trans^1e+8;
                        P = P[1, :]';
                    end
                else
                    println("Sorry, there is more than one distribution.")
                    println("There should exist a unique unit eigenvalue for the transition matrix.")
                    P = vec
                end
                
                return P
            end
            
            
            
            function markovapprox(ρ::Real, σ_ϵ::Real, N::Integer, m = 3)
                #= 
                This function approximates a first-order autoregressive process
                    Arguments
                    - ρ: persistence
                    - σ_ϵ: innovation standard deviation
                    - N: number of states in Markov chain
                    - m: width of discretized state spaces, Tauchen uses m = 3 with y_max = m * Var(y) and y_min = -m * Var(y)
                    
                    Returns
                    - Tran: transition matrix of the Markov chain
                    - s: discretized state space
                    - p: stationary distribution
                    - arho: approximated AR coefficient for the Markov chain
                    - asigma: approximated standard deviation for the Markov chain
                    =#
                    
                    # Discretize state space
                    σ_y = sqrt( σ_ϵ^2 / (1 - ρ^2) );
                    y_max = m * σ_y;
                    y_min = - y_max;
                    s = collect( range(y_min, y_max, length = N) );
                    w = s[2] - s[1];
                    
                    Tran = zeros(N, N);
                    for i in 1:N
                        for j in 2:N-1
                            Tran[i, j] = cdf( Normal(0, σ_ϵ), s[j] - ρ * s[i] + w/2) - cdf( Normal(0, σ_ϵ), s[j] - ρ * s[i] - w/2);
                        end 
                        Tran[i, 1] = cdf( Normal(0, σ_ϵ), s[1] - ρ * s[i] + w/2 );
                        Tran[i, N] = ccdf( Normal(0, σ_ϵ), s[N] - ρ * s[i] - w/2 );
                    end
                    
                    # Check if Transition matrix is well specified.
                    # Renormalize if needed
                    for k in 1:N
                        if abs( sum(Tran[k, :]) - 1 ) > 1e-10
                            println("Row $k does not sum to one")
                            println("Normalizing row $k")
                            Tran[k, :] = Tran[k, :] ./ sum(Tran[k, :]);
                        end        
                    end
                    
                    # Calculate the invariant distribution of Markov chain
                    P = ergodic(Tran);
                    
                    meanm = s ⋅ P;
                    varm = ( (s .- meanm).^2 ) ⋅ P;
                    midaut1 = (s .- meanm) * (s .- meanm)';
                    probmat = P * ones(1, N);
                    midaut2 = Tran .* probmat .* midaut1;
                    autcov1 = sum(midaut2);
                    
                    arho = autcov1 / varm;
                    asigma = sqrt(varm);
                    
                    # Compare moments of discretized Markov chain with those of original continuous process
                    println("Original Continuous Process: ρ = $ρ and σ_y = $σ_y")
                    println("Discretized Markov Chain: ρ = $arho and σ_y = $asigma")
                    
                    return Tran, s, P, arho, asigma
                end