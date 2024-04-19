% Consider the transformation: hht = ht + zt, zt = -log((yt-xt'*b)^2/w)

function hh = htrans(h,y,x,b,w)
% Inputs:
%   h: a ndraws-by-n matrix of draws of ht
%   y: a n-by-1 vector of target
%   x: a n-by-K matrix of regressors
%   b: a ndraws-by-K matrix of draws of reg coef
%   w: a ndraws-by-1 vector of draws of df (student t distr)
% Outputs:
%   hh: a ndraws-by-n matrix of transformed ht

[ndraws,n] = size(h);
hh = zeros(ndraws,n);
for t = 1:n
    yt = y(t);
    xt = x(t,:)';
    ht = h(:,t);
    resid = yt - b*xt;
    resid2 = resid.^2;
    zt = -log(resid2./w);
    hht = ht + zt;
    hh(:,t) = hht;
end


