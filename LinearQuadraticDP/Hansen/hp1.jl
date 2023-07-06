function hp1(y, w)
    
    if size(y,1)<size(y,2)
        y = transpose(y);
    end
    t = size(y,1);
    a = 6 * w + 1;
    b = -4 * w;
    c = w;
    # d = [c b a];
    # d = repeat(d, t, 1);
    aa = fill(a, t);
    bb = fill(b, t-1);
    cc = fill(c, t-2);
    mm = Tridiagonal(bb, aa, bb);
    mmm = diagm(2 => cc, -2 => cc);
    m = mm + mmm;
    
    m[1, 1] = 1+w;
    m[1, 2] = -2*w;
    m[2, 1] = -2*w;
    m[2, 2] = 5*w + 1;
    m[t-1, t-1] = 5*w + 1;
    m[t-1,t] = -2*w;
    m[t, t-1] = -2*w;
    m[t,t] = 1+w;

    ytr = inv(m) * y;
    yhp = y - ytr;
    
    return yhp
end