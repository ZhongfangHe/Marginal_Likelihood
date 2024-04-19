% Consider the model: yt = xt'*bt + N(0,s2t), bt = btm1 + N(0,diag(w)), b0 is given
% Compute the filtering moments p(bt|y1,y2,...,yt) = N(mt,Mt)

function [Km,KM] = Kalman_moments(y,x,s2,w,b0)
% Inputs:
%   y: a n-by-1 vector of target
%   x: a n-by-K matrix of regressors
%   s2: a n-by-1 vector of residual variance
%   w: a K-by-1 vector of TVP variance
%   b0: a K-by-1 vector of TVP starting value
% Outputs:
%   Km: a n-by-K matrix of filtering mean
%   KM: a n-by-1 cell of K-by-K matrices of filtering covariance matrix

[n,K] = size(x);
v = diag(w);
Km = zeros(n,K);
KM = cell(n,1);
for t = 1:n
    yt = y(t);
    xt = x(t,:)';
    s2t = s2(t);
    if t == 1
        Mtm1 = zeros(K,K);
        mtm1 = b0;
    else
        Mtm1 = KM{t-1}; 
        mtm1 = Km(t-1,:)';
    end
    
    tmp0 = (v + Mtm1)*xt; 
    tmp1 = s2t + xt'*tmp0;
    tmp2 = tmp0 * xt';
    slope_mat = eye(K) - tmp2/tmp1;
    Mt = slope_mat * (v+Mtm1);
    mt = slope_mat * mtm1 + Mt*xt*yt/s2t;

    Km(t,:) = mt';
    KM{t} = Mt;
end

