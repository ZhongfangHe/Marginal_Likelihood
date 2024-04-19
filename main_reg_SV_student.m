% Consider the linear regression model with SV and student-t distr:
% yt = xt'*b + exp(zt/2)*t(v), zt = (1-phi)*u + phi*ztm1 + etat, etat~N(0,s)
% compute its marginal likelihood by IS or GD

clear;
dbstop if warning;
dbstop if error;
rng(123); %rng(1234567);
addpath(genpath('C:\Users\Zhongfang\Documents\My Research\Bayesian_TVP\2021Feb\Functions'));
addpath(genpath('C:\Users\Zhongfang\Documents\My Research\Bayesian_TVP\2021Apr\Functions'));
addpath(genpath('C:\Users\Zhongfang\Documents\My Research\Bayesian_TVP\2021Jul\Functions'));
addpath(genpath('C:\Users\Zhongfang\Documents\My Research\Bayesian_TVP\2021Sep\Functions'));
addpath(genpath('C:\Users\Zhongfang\Documents\My Research\Bayesian_TVP\2024Apr_Marginal_Likelihood\Functions'));


%% Gather data
dgp = {'Simulation','EquityPremium'};
ind_dgp = 2;
disp(['DGP = ',dgp{ind_dgp}]);
if ind_dgp == 1 %simulation
    n = 300;
    K = 2;%10; %2;
    disp(['n = ',num2str(n), ', K = ', num2str(K)]);
    if K == 1
        x = ones(n,1);
    else
        x = [ones(n,1) rand(n,K-1)];
    end
    utrue = -3;
    phitrue = 0.9;
    s2true = 0.1; %SV parameters
    btrue = 0.1*ones(K,1); %reg parameters
    ztrue = zeros(n,1);
    y = zeros(n,1);
    for t = 1:n
        if t == 1
            ztm1 = utrue;
        else
            ztm1 = ztrue(t-1);
        end
        zt = (1-phitrue)*utrue + phitrue*ztm1 + sqrt(s2true)*randn;
        yt = x(t,:)*btrue + exp(0.5*zt)*randn;
        ztrue(t) = zt;
        y(t) = yt;
    end
else %equity premium    
    read_file = 'Equity_Qtrly_Github.xlsx';
    read_sheet = 'Data';
    data = readmatrix(read_file, 'Sheet', read_sheet, 'Range', 'B2:N297');
    [ng,nr] = size(data);
    equity = data(:,1);
    reg = data(:,2:nr);
    y = equity(2:ng);
    x = [ones(ng-1,1) equity(1:(ng-1)) reg(1:(ng-1),:)]; %full
    [n,K] = size(x);    
end


%% MCMC
tic;
nthin = 1;
ndraws = 5000*3*nthin; %5000;
burnin = 2000;
disp(['burnin = ',num2str(burnin),', ndraws = ',num2str(ndraws)]);
ntotal = burnin + ndraws;

b0_mean = zeros(K,1);
b0_var = 100*ones(K,1);

vp1 = 7; %4;
vp2 = 60; %30; 
v = 1/gamrnd(vp1,1/vp2);
f = 0.5*v;
d = 1./gamrnd(f,1/f,n,1);

muh0 = 0; invVmuh = 1/10; % mean: p(mu) ~ N(mu0, Vmu)
phiha = 8; phihb = 2; % AR(1): p(phi) ~ 0.5 * (1 + betarnd(a,b))
sigh2_s = 1; %var: p(sigh2) ~ G(0.5,2*sigh2_s)
priorSV = [muh0 invVmuh phiha phihb]'; %collect prior hyperparameters
muh = muh0 + sqrt(1/invVmuh) * randn;
phih = 0.5*(1+betarnd(phiha,phihb));
sigh2 = gamrnd(0.5,2*sigh2_s);
sigh = sqrt(sigh2);

b0_OLS = regress(y,x);
resid_OLS = y - x*b0_OLS;
hSV = log(var(resid_OLS))*ones(n,1); %initialize by log OLS residual variance.
hSVstar = (hSV-muh)/sigh;

pstar_v = 0.44; %univariate MH
AMH_v = 1/(pstar_v * (1-pstar_v));
logrw_v = 0; %set up autoMH for df

pstar_SV = 0.25; %multivariate MH
tmp_const = -norminv(0.5*pstar_SV);
KSV = 2;
AMH_SV = 1/(KSV * pstar_SV * (1-pstar_SV)) + (1-1/KSV)*0.5*sqrt(2*pi)*...
    exp(0.5*tmp_const*tmp_const)/tmp_const;
logrw_SV = 0;
paraSV_mean = zeros(KSV,1);
paraSV_cov = zeros(KSV,KSV); %set up autoMH for SV

draws.b = zeros(ndraws,K); %linear coef
draws.v = zeros(ndraws,1); %df para
draws.d = zeros(ndraws,n); %auxiliar variable
draws.SVpara = zeros(ndraws,4); % [mu phi sig2 sig]
draws.z = zeros(ndraws,n); %residual variance
draws.logrw_v = zeros(ndraws,1);
draws.count_v = 0;
draws.logrw_SV = zeros(ndraws,1);
draws.count_SV = 0;
for drawi = 1:ntotal
    % Draw mu, sig of the SV part
    count_SV = 0;

    muh_old = muh;
    sigh_old = sigh;
    paraSV_old = [muh_old sigh_old]';
    if drawi < 100
        A = eye(KSV);
    else  
        A = paraSV_cov + 1e-10 * eye(KSV) / drawi; %add a small constant
    end
    eps = mvnrnd(zeros(KSV,1),A)'; %correlated normal
    paraSV_new = paraSV_old + exp(logrw_SV) * eps; 
    muh_new = paraSV_new(1);
    sigh_new = paraSV_new(2);

    logprior_old_muh = -0.5*((muh_old-muh0)^2)*invVmuh;
    logprior_old_sigh = -0.5*(sigh_old^2)/sigh2_s;
    logprior_old = logprior_old_muh + logprior_old_sigh;

    logprior_new_muh = -0.5*((muh_new-muh0)^2)*invVmuh;
    logprior_new_sigh = -0.5*(sigh_new^2)/sigh2_s;
    logprior_new = logprior_new_muh + logprior_new_sigh;
    
    h_old = muh_old + sigh_old * hSVstar;
    yvar = exp(h_old).*d;
    ystd = sqrt(yvar);
    yy = y./ystd;
    xx = x./repmat(ystd,1,K);
    Binv = diag(1./b0_var) + xx'*xx;    
    Binvb = xx'*yy; 
    tmp1 = -0.5*n*muh_old-0.5*sigh_old*sum(hSVstar);
    Binv_half = chol(Binv)';
    logdet_Binv = 2*sum(log(diag(Binv_half)));
    tmp2 = -0.5*logdet_Binv;
    tmp3 = -0.5*(yy'*yy) + 0.5*Binvb'*(Binv\Binvb);
    loglike_old = tmp1+tmp2+tmp3;
    
    h_new = muh_new + sigh_new * hSVstar;
    yvar = exp(h_new).*d;
    ystd = sqrt(yvar);
    yy = y./ystd;
    xx = x./repmat(ystd,1,K);
    Binv = diag(1./b0_var) + xx'*xx;    
    Binvb = xx'*yy; 
    tmp1 = -0.5*n*muh_new-0.5*sigh_new*sum(hSVstar);
    Binv_half = chol(Binv)';
    logdet_Binv = 2*sum(log(diag(Binv_half)));
    tmp2 = -0.5*logdet_Binv;
    tmp3 = -0.5*(yy'*yy) + 0.5*Binvb'*(Binv\Binvb);
    loglike_new = tmp1+tmp2+tmp3;

    logprob = logprior_new + loglike_new - logprior_old - loglike_old;
    if log(rand) <= logprob
        paraSV = paraSV_new;
        muh = muh_new;
        sigh = sigh_new;
        if drawi > burnin
            count_SV = 1;
        end
    else
        paraSV = paraSV_old;
    end

    p = exp(min(0,logprob));
    dd = max(drawi/KSV, 20);
    logrwj = logrw_SV + AMH_SV * (p - pstar_SV)/dd;   
    logrw_SV = logrwj; %update proposal stdev

    paraSV_mean_old = paraSV_mean;
    paraSV_cov_old = paraSV_cov;
    paraSV_mean = (paraSV_mean_old * (drawi-1) + paraSV) / drawi;
    paraSV_cov = (drawi - 1) * (paraSV_cov_old + paraSV_mean_old * paraSV_mean_old') / drawi + ...
        paraSV * paraSV' / drawi - paraSV_mean * paraSV_mean'; %update the sample covariance    
    
    
    % Linear coef
    hSV = muh + sigh*hSVstar;
    yvar = exp(hSV).*d;
    ystd = sqrt(yvar); 
    yy = y./ystd;
    xx = x./repmat(ystd,1,K);
    Binv = diag(1./b0_var) + xx'*xx;    
    Binvb = xx'*yy; 
    tmp = mvnrnd(Binvb,Binv)';
    b = Binv\tmp; 

    % SV
    resid = (y-x*b)./sqrt(d);
    logz2 = log(resid.^2 + 1e-100);
    [hSV, muh, phih, sigh] = SV_update_hstar(logz2, hSV, ...
        muh, phih, sigh, sigh2_s, priorSV);
    hSVstar = (hSV - muh)/sigh; 
    

    % Degrees of freedeom
    count_v = 0;
    
    v_old = v;
    logv_new = log(v_old) + exp(logrw_v)*randn;
    v_new = exp(logv_new);

    logprior_old = -vp1*log(v_old) - vp2/v_old;
    logprior_new = -vp1*log(v_new) - vp2/v_new;

    resid = exp(-0.5*hSV).*(y-x*b);
    resid2 = resid.^2;
    tmp1 = n*(gammaln(0.5+0.5*v_old) - 0.5*log(pi*v_old) - gammaln(0.5*v_old));
    tmp2 = -0.5*(1+v_old)*sum(log(1+resid2/v_old));
    loglike_old = tmp1+tmp2;
    tmp1 = n*(gammaln(0.5+0.5*v_new) - 0.5*log(pi*v_new) - gammaln(0.5*v_new));
    tmp2 = -0.5*(1+v_new)*sum(log(1+resid2/v_new));
    loglike_new = tmp1+tmp2;

    logprob = logprior_new + loglike_new - logprior_old - loglike_old;
    if log(rand)<logprob
        v = v_new;
        if drawi > burnin
            count_v = 1;
        end    
    end 

    p = exp(min(0,logprob));
    logrwj = logrw_v + AMH_v * (p - pstar_v)/drawi;   
    logrw_v = logrwj; %update proposal stdev    
    
    % Auxiliary variable
    resid = exp(-0.5*hSV).*(y-x*b);
    resid2 = resid.^2;
    f = 0.5*v;
    d = 1./gamrnd((f+0.5)*ones(n,1), 1./(f +0.5*resid2));  
           
    if drawi > burnin
        i = drawi-burnin;
        draws.b(i,:) = b';
        draws.z(i,:) = hSV';
        draws.SVpara(i,:) = [muh phih sigh^2 sigh];
        draws.v(i) = v;
        draws.d(i,:) = d';
        draws.logrw_v(i) = logrw_v;
        draws.count_v = draws.count_v + count_v/ndraws;
        draws.logrw_SV(i) = logrw_SV;
        draws.count_SV = draws.count_SV + count_SV/ndraws;
    end   
end
disp('MCMC is completed!');
toc;
disp(' ');


% draws_full = draws;
% draws.b = Thin(draws_full.b,nthin);
% draws.z = Thin(draws_full.z,nthin);
% draws.SVpara = Thin(draws_full.SVpara,nthin);
% draws.v = Thin(draws_full.v,nthin);
% draws.d = Thin(draws_full.d,nthin);
% ndraws = size(draws.b,1);



%% 1. IS by using AR(1) for ht
tic;
u = draws.SVpara(:,1);
rho = draws.SVpara(:,2);
s = draws.SVpara(:,4);
para_est = [draws.b  u  log(1+rho)-log(1-rho)  log(s.^2)  log(draws.v)];
KK = size(para_est,2);
para_mean = mean(para_est)';
para_cov = cov(para_est);
para_covhalf = chol(para_cov)';
para_covinv = para_cov\eye(KK);
logdet_para_cov = 2*sum(log(diag(para_covhalf))); %calibrate Gaussian IS for fixed para

constvec = ones(ndraws,1);
R2vec = zeros(n,1);
ISpara = zeros(n,3+KK); %const, htm1, para_est, var(resid)
resid_mat = zeros(ndraws,n);
for t = 1:n
    ht = draws.z(:,t);
    if t > 1
        htm1 = draws.z(:,t-1);
        xx = [constvec htm1 para_est];
    else
        xx = [constvec para_est];
    end 
    yy = ht;
    Kxx = size(xx,2);
    Binv = eye(Kxx)/10000 + xx'*xx;
    Binvb = xx'*yy;
    coef = Binv\Binvb;
    yyfit = xx*coef;
    resid = yy-yyfit;
    if t > 1
        ISpara(t,:) = [coef' var(resid)];
    else
        ISpara(t,:) = [coef(1) 0 coef(2:Kxx)' var(resid)];
    end
    R2vec(t) = var(yyfit)/var(yy);
    resid_mat(:,t) = resid;
end %calibrate AR(1) IS for latent variable

nsim = ndraws*1;
logh = zeros(nsim,1);
logp = logh;
logy = logh;
logqh = logh;
logqp = logh;
logw = logh;
hj = zeros(n,1);
hj_mean = zeros(n,1);
for drawi = 1:nsim
    paraj = mvnrnd(para_mean, para_cov)';
    bj = paraj(1:K);
    uj = paraj(K+1);
    rhoj = (exp(paraj(K+2))-1)/(exp(paraj(K+2))+1);
    sj = sqrt(exp(paraj(K+3)));
    s2j = sj^2;
    rhoj2 = rhoj^2; 
    vj = exp(paraj(K+4));
    wj = vj; %simulate theta from IS   
    
    for t = 1:n
        if t == 1
            xxt = [1 0 paraj'];
        else
            xxt = [1 hj(t-1) paraj']; 
        end
        hj_mean(t) = xxt * ISpara(t,1:(KK+2))';
        hj(t) = hj_mean(t) + sqrt(ISpara(t,(KK+3)))*randn;
    end %simulate h from IS
    
    logpj_b = -0.5*K*log(2*pi) -0.5*sum(log(b0_var)) -0.5*sum(((bj-b0_mean).^2)./b0_var);
    logpj_SV = -0.5*log(2*pi/invVmuh) - 0.5*((uj-muh0)^2)*invVmuh ...
        +phiha*log(1+rhoj)+phihb*log(1-rhoj)-(phiha+phihb)*log(2)-betaln(phiha,phihb) ...
        -0.5*log(2*pi*sigh2_s) - 0.5*s2j/sigh2_s + 0.5*log(s2j);
    logpj_v = vp1*log(vp2)-gammaln(vp1)-vp1*log(vj)-vp2/vj;
    logpj = logpj_b + logpj_SV + logpj_v; %prior theta
     
    tmp = (hj(2:n)-(1-rhoj)*uj-rhoj*hj(1:n-1)).^2;
    loghj = -0.5*n*log(2*pi*s2j) + 0.5*log(1-rhoj2) - 0.5*(1-rhoj2)*((hj(1)-uj)^2)/s2j ...
        -0.5*sum(tmp)/s2j; %prior h    
    
    eps2 = (y - x*bj).^2;
    logyj = n*(gammaln(0.5+0.5*wj) - 0.5*log(pi*wj) - gammaln(0.5*wj)) - 0.5*sum(hj)...
        -0.5*(1+wj)*sum(log(1+eps2.*exp(-hj)/wj)); %likelihood
    
    resid2 = (hj - hj_mean).^2;
    ISd = ISpara(:,KK+3);
    logqhj = -0.5*n*log(2*pi) - 0.5*sum(log(ISd)) -0.5*sum(resid2./ISd); %IS h
    
    tmp = paraj - para_mean;
    logqpj = -0.5*KK*log(2*pi) - 0.5*logdet_para_cov - 0.5*tmp'*para_covinv*tmp; %IS theta
    
    logh(drawi) = loghj;
    logp(drawi) = logpj;
    logy(drawi) = logyj;
    logqh(drawi) = logqhj;
    logqp(drawi) = logqpj; 
    logw(drawi) = logpj+loghj+logyj-logqpj-logqhj;
end
wtmp = logw;
wlevel = max(wtmp); %mean(wtmp);
ew = exp(wtmp - wlevel);
tmp_mean = mean(ew);
tmp_std = std(ew);
disp('IS: AR(1) for ht');
disp(['LML = ',num2str(wlevel+log(tmp_mean)),' with std = ',num2str(tmp_std/tmp_mean)]);
toc;
disp(' ');



%% 2. IS by backward matching
tic;
u = draws.SVpara(:,1);
rho = draws.SVpara(:,2);
s = draws.SVpara(:,4);
para_est = [draws.b  u  log(1+rho)-log(1-rho)  log(s.^2) log(draws.v)];
KK = size(para_est,2);
para_mean = mean(para_est)';
para_cov = cov(para_est);
para_covhalf = chol(para_cov)';
para_covinv = para_cov\eye(KK);
logdet_para_cov = 2*sum(log(diag(para_covhalf))); %calibrate Gaussian IS for fixed para

R2vec2 = zeros(n,1);
coef_proxy = zeros(n,3);
constvec = ones(ndraws,1);
hhmat = htrans(draws.z,y,x,draws.b,draws.v);
for t = 1:n
    hht = hhmat(:,t);
    yy = log(1+exp(hht));
    xx = [constvec hht hht.^2];
    coef = regress(yy,xx);
    yyfit = xx*coef;
    if coef(3)<0
        xx = [constvec hht];
        coef_level = regress(yy,xx);
        coef = [coef_level; 0];
        yyfit = xx*coef_level;
    end %ensure the coef on hht2 is positive
    coef_proxy(t,:) = coef';
    R2vec2(t) = var(yyfit)/var(yy);
end %linearize log(1+exp(hht))

nsim = ndraws*1;
logh2 = zeros(nsim,1);
logp2 = logh2;
logy2 = logh2;
logqh2 = logh2;
logqp2 = logh2;
logw2 = logh2;
hj = zeros(n,1);
hj_mean = zeros(n,1);
ISpara2 = zeros(n,3); %const, htm1, var(resid)
for drawi = 1:nsim
    paraj = mvnrnd(para_mean, para_cov)';
    bj = paraj(1:K);
    uj = paraj(K+1);
    rhoj = (exp(paraj(K+2))-1)/(exp(paraj(K+2))+1);
    sj = sqrt(exp(paraj(K+3)));
    s2j = sj^2;
    s2j2 = s2j^2;
    rhoj2 = rhoj^2; 
    vj = exp(paraj(K+4));
    wj = vj; %simulate theta from IS 
    
    zz = -log(((y-x*bj).^2)./wj);
    cinv = 1/s2j + (1+wj)*coef_proxy(n,3);
    c = 1/cinv;
    bc = rhoj/s2j;
    ac = (1-rhoj)*uj/s2j + 0.5*wj - 0.5*(1+wj)*coef_proxy(n,2) - (1+wj)*zz(n)*coef_proxy(n,3);
    ISpara2(n,3) = c;
    ISpara2(n,2) = c*bc;
    ISpara2(n,1) = c*ac;
    t = n-1;
    while t >= 2
        ctp1 = ISpara2(t+1,3);
        cinv = -ctp1*rhoj2/s2j2 + (1+rhoj2)/s2j + (1+wj)*coef_proxy(t,3);
        c = 1/cinv;
        bc = rhoj/s2j;
        atp1 = ISpara2(t+1,1); 
        ac = atp1*rhoj/s2j + (1-rhoj)*(1-rhoj)*uj/s2j + 0.5*wj - 0.5*(1+wj)*coef_proxy(t,2)...
            -(1+wj)*zz(t)*coef_proxy(t,3);        
        ISpara2(t,3) = c;
        ISpara2(t,2) = c*bc;
        ISpara2(t,1) = c*ac;
        t = t-1;
    end
    ctp1 = ISpara2(2,3);
    cinv = -ctp1*rhoj2/s2j2 + 1/s2j + (1+wj)*coef_proxy(1,3);
    c = 1/cinv;
    atp1 = ISpara2(2,1); 
    ac = atp1*rhoj/s2j + (1-rhoj)*uj/s2j + 0.5*wj - 0.5*(1+wj)*coef_proxy(1,2)...
        -(1+wj)*coef_proxy(1,3)*zz(1);
    ISpara2(1,3) = c;
    ISpara2(1,1) = c*ac; %compute IS coef for h
    
    for t = 1:n
        if ISpara2(t,3)<=0
            error('ct<=0');
        end
        if t == 1
            xxt = [1 0];
        else
            xxt = [1 hj(t-1)]; 
        end
        hj_mean(t) = xxt * ISpara2(t,1:2)';
        hj(t) = hj_mean(t) + sqrt(ISpara2(t,3))*randn;
    end %simulate h from IS
    
    logpj_b = -0.5*K*log(2*pi) -0.5*sum(log(b0_var)) -0.5*sum(((bj-b0_mean).^2)./b0_var);
    logpj_SV = -0.5*log(2*pi/invVmuh) - 0.5*((uj-muh0)^2)*invVmuh ...
        +phiha*log(1+rhoj)+phihb*log(1-rhoj)-(phiha+phihb)*log(2)-betaln(phiha,phihb) ...
        -0.5*log(2*pi*sigh2_s) - 0.5*s2j/sigh2_s + 0.5*log(s2j);
    logpj_v = vp1*log(vp2)-gammaln(vp1)-vp1*log(vj)-vp2/vj; 
    logpj = logpj_b + logpj_SV + logpj_v; %prior theta   
     
%     loghj = -0.5*n*log(2*pi*s2j) + 0.5*log(1-rhoj2) - 0.5*(1-rhoj2)*uj*uj/s2j ...
%         -0.5*(n-1)*(1-rhoj)*(1-rhoj)*uj*uj/s2j; 
%     tmp1 = -0.5*(1-rhoj2)*zz(1)*zz(1)/s2j - (1-rhoj2)*uj*zz(1)/s2j;
%     tmp2 = -0.5*sum(zz(2:n).^2)/s2j - (1-rhoj)*uj*sum(zz(2:n))/s2j + rhoj*sum(zz(1:n-1).*zz(2:n))/s2j...
%         -0.5*rhoj2*sum(zz(1:n-1).^2)/s2j + rhoj*(1-rhoj)*uj*sum(zz(1:n-1))/s2j;
%     loghj = loghj + tmp1 + tmp2; %prior hh
    
%     yj = log(1+exp(-hhj));
%     yjfit = coef_proxy(:,1) + coef_proxy(:,2).*hhj + coef_proxy(:,3).*hhj.*hhj;
%     epsj = yj-yjfit;
%     logyj = n*(gammaln(0.5+0.5*wj) - 0.5*log(pi*wj) - gammaln(0.5*wj)) + 0.5*sum(zz) ...
%         -0.5*(1+wj)*sum((coef_proxy(:,1)+epsj)); %likelihood
    
%     ISa = ISpara2(:,1);
%     ISc = ISpara2(:,3);
%     logqhj = -0.5*n*log(2*pi) - 0.5*sum(log(ISc)) -0.5*sum(ISa.*ISa./ISc); %IS h
    
    tmp = (hj(2:n)-(1-rhoj)*uj - rhoj*hj(1:n-1)).^2;
    loghj = -0.5*n*log(2*pi*s2j) + 0.5*log(1-rhoj2) - 0.5*(1-rhoj2)*((hj(1)-uj)^2)/s2j ...
        -0.5*sum(tmp)/s2j; %prior h 
    
    logyj = n*(gammaln(0.5+0.5*wj) - 0.5*log(pi*wj) - gammaln(0.5*wj)) - 0.5*sum(hj) ...
        -0.5*(1+wj)*sum(log(1+exp(-hj-zz))); %likelihood
        
    resid2 = (hj - hj_mean).^2;
    ISc = ISpara2(:,3);
    logqhj = -0.5*n*log(2*pi) - 0.5*sum(log(ISc)) -0.5*sum(resid2./ISc); %IS h
    
    tmp = paraj - para_mean;
    logqpj = -0.5*KK*log(2*pi) - 0.5*logdet_para_cov - 0.5*tmp'*para_covinv*tmp; %IS theta
    
    logh2(drawi) = loghj;
    logp2(drawi) = logpj;
    logy2(drawi) = logyj;
    logqh2(drawi) = logqhj;
    logqp2(drawi) = logqpj;
    logw2(drawi) = logpj+loghj+logyj-logqpj-logqhj;
end
wtmp = logw2;
wlevel = max(wtmp); %mean(wtmp);
ew = exp(wtmp - wlevel);
tmp_mean = mean(ew);
tmp_std = std(ew);
disp('IS: backward matching');
disp(['LML = ',num2str(wlevel+log(tmp_mean)),' with std = ',num2str(tmp_std/tmp_mean)]);
toc;
disp(' ');


%% 3. GD by using AR(1) for ht
tic;
u = draws.SVpara(:,1);
rho = draws.SVpara(:,2);
s = draws.SVpara(:,4);
para_est = [draws.b  u  log(1+rho)-log(1-rho)  log(s.^2)  log(draws.v)];
KK = size(para_est,2);
para_mean = mean(para_est)';
para_cov = cov(para_est);
para_covhalf = chol(para_cov)';
para_covinv = para_cov\eye(KK);
logdet_para_cov = 2*sum(log(diag(para_covhalf))); %calibrate Gaussian IS for fixed para

constvec = ones(ndraws,1);
R2vec = zeros(n,1);
ISpara = zeros(n,3+KK); %const, htm1, para_est, var(resid)
resid_mat = zeros(ndraws,n);
for t = 1:n
    ht = draws.z(:,t);
    if t > 1
        htm1 = draws.z(:,t-1);
        xx = [constvec htm1 para_est];
    else
        xx = [constvec para_est];
    end 
    yy = ht;
    Kxx = size(xx,2);
    Binv = eye(Kxx)/10000 + xx'*xx;
    Binvb = xx'*yy;
    coef = Binv\Binvb;
    yyfit = xx*coef;
    resid = yy-yyfit;
    if t > 1
        ISpara(t,:) = [coef' var(resid)];
    else
        ISpara(t,:) = [coef(1) 0 coef(2:Kxx)' var(resid)];
    end
    R2vec(t) = var(yyfit)/var(yy);
    resid_mat(:,t) = resid;
end %calibrate AR(1) IS for latent variable

nsim = ndraws*1;
logh_gd = zeros(nsim,1);
logp_gd = logh_gd;
logy_gd = logh_gd;
logqh_gd = logh_gd;
logqp_gd = logh_gd;
logw_gd = logh_gd;
hj = zeros(n,1);
hj_mean = zeros(n,1);
for drawi = 1:nsim
    paraj = para_est(drawi,:)';
    bj = paraj(1:K);
    uj = paraj(K+1);
    rhoj = (exp(paraj(K+2))-1)/(exp(paraj(K+2))+1);
    sj = sqrt(exp(paraj(K+3)));
    s2j = sj^2;
    rhoj2 = rhoj^2; 
    vj = exp(paraj(K+4));
    wj = vj; %theta from posterior
    
    hj = draws.z(drawi,:)';
    for t = 1:n
        if t == 1
            xxt = [1 0 paraj'];
        else
            xxt = [1 hj(t-1) paraj']; 
        end
        hj_mean(t) = xxt * ISpara(t,1:(KK+2))';
    end %h from posterior
    
    logpj_b = -0.5*K*log(2*pi) -0.5*sum(log(b0_var)) -0.5*sum(((bj-b0_mean).^2)./b0_var);
    logpj_SV = -0.5*log(2*pi/invVmuh) - 0.5*((uj-muh0)^2)*invVmuh ...
        +phiha*log(1+rhoj)+phihb*log(1-rhoj)-(phiha+phihb)*log(2)-betaln(phiha,phihb) ...
        -0.5*log(2*pi*sigh2_s) - 0.5*s2j/sigh2_s + 0.5*log(s2j);
    logpj_v = vp1*log(vp2)-gammaln(vp1)-vp1*log(vj)-vp2/vj;
    logpj = logpj_b + logpj_SV + logpj_v; %prior theta
     
    tmp = (hj(2:n)-(1-rhoj)*uj-rhoj*hj(1:n-1)).^2;
    loghj = -0.5*n*log(2*pi*s2j) + 0.5*log(1-rhoj2) - 0.5*(1-rhoj2)*((hj(1)-uj)^2)/s2j ...
        -0.5*sum(tmp)/s2j; %prior h    
    
    eps2 = (y - x*bj).^2;
    logyj = n*(gammaln(0.5+0.5*wj) - 0.5*log(pi*wj) - gammaln(0.5*wj)) - 0.5*sum(hj)...
        -0.5*(1+wj)*sum(log(1+eps2.*exp(-hj)/wj)); %likelihood
    
    resid2 = (hj - hj_mean).^2;
    ISd = ISpara(:,KK+3);
    logqhj = -0.5*n*log(2*pi) - 0.5*sum(log(ISd)) -0.5*sum(resid2./ISd); %IS h
    
    tmp = paraj - para_mean;
    logqpj = -0.5*KK*log(2*pi) - 0.5*logdet_para_cov - 0.5*tmp'*para_covinv*tmp; %IS theta
    
    logh_gd(drawi) = loghj;
    logp_gd(drawi) = logpj;
    logy_gd(drawi) = logyj;
    logqh_gd(drawi) = logqhj;
    logqp_gd(drawi) = logqpj; 
    logw_gd(drawi) = logpj+loghj+logyj-logqpj-logqhj;
end
wtmp = -logw_gd;
wlevel = max(wtmp); %mean(wtmp);
ew = exp(wtmp - wlevel);
tmp_mean = mean(ew);
[~, hac_std, ~] = HAC_regression(ew,constvec); 
tmp_std = sqrt(ndraws)*hac_std; %tmp_std = std(ew);
disp('GD: AR(1) for ht');
disp(['LML = ',num2str(-wlevel-log(tmp_mean)),' with std = ',num2str(tmp_std/tmp_mean)]);
toc;
disp(' ');



%% 4. GD by backward matching
tic;
u = draws.SVpara(:,1);
rho = draws.SVpara(:,2);
s = draws.SVpara(:,4);
para_est = [draws.b  u  log(1+rho)-log(1-rho)  log(s.^2) log(draws.v)];
KK = size(para_est,2);
para_mean = mean(para_est)';
para_cov = cov(para_est);
para_covhalf = chol(para_cov)';
para_covinv = para_cov\eye(KK);
logdet_para_cov = 2*sum(log(diag(para_covhalf))); %calibrate Gaussian IS for fixed para

R2vec2 = zeros(n,1);
coef_proxy = zeros(n,3);
constvec = ones(ndraws,1);
hhmat = htrans(draws.z,y,x,draws.b,draws.v);
for t = 1:n
    hht = hhmat(:,t);
    yy = log(1+exp(hht)); %log(1+exp(-hht));
    xx = [constvec hht hht.^2];
    coef = regress(yy,xx);
    yyfit = xx*coef;
    if coef(3)<0
        xx = [constvec hht];
        coef_level = regress(yy,xx);
        coef = [coef_level; 0];
        yyfit = xx*coef_level;
    end %ensure the coef on hht2 is positive
    coef_proxy(t,:) = coef';
    R2vec2(t) = var(yyfit)/var(yy);
end %linearize log(1+exp(hht))

nsim = ndraws*1;
logh2_gd = zeros(nsim,1);
logp2_gd = logh2_gd;
logy2_gd = logh2_gd;
logqh2_gd = logh2_gd;
logqp2_gd = logh2_gd;
logw2_gd = logh2_gd;
hj_mean = zeros(n,1);
ISpara2 = zeros(n,3); %const, htm1, var(resid)
for drawi = 1:nsim
    paraj = para_est(drawi,:)';
    bj = paraj(1:K);
    uj = paraj(K+1);
    rhoj = (exp(paraj(K+2))-1)/(exp(paraj(K+2))+1);
    sj = sqrt(exp(paraj(K+3)));
    s2j = sj^2;
    s2j2 = s2j^2;
    rhoj2 = rhoj^2; 
    vj = exp(paraj(K+4));
    wj = vj; %theta from posterior 
    
    zz = -log(((y-x*bj).^2)./wj);
    cinv = 1/s2j + (1+wj)*coef_proxy(n,3);
    c = 1/cinv;
    bc = rhoj/s2j;
    ac = (1-rhoj)*uj/s2j + 0.5*wj - 0.5*(1+wj)*coef_proxy(n,2) - (1+wj)*zz(n)*coef_proxy(n,3);
    ISpara2(n,3) = c;
    ISpara2(n,2) = c*bc;
    ISpara2(n,1) = c*ac;
    t = n-1;
    while t >= 2
        ctp1 = ISpara2(t+1,3);
        cinv = -ctp1*rhoj2/s2j2 + (1+rhoj2)/s2j + (1+wj)*coef_proxy(t,3);
        c = 1/cinv;
        bc = rhoj/s2j;
        atp1 = ISpara2(t+1,1); 
        ac = atp1*rhoj/s2j + (1-rhoj)*(1-rhoj)*uj/s2j + 0.5*wj - 0.5*(1+wj)*coef_proxy(t,2)...
            -(1+wj)*zz(t)*coef_proxy(t,3);        
        ISpara2(t,3) = c;
        ISpara2(t,2) = c*bc;
        ISpara2(t,1) = c*ac;
        t = t-1;
    end
    ctp1 = ISpara2(2,3);
    cinv = -ctp1*rhoj2/s2j2 + 1/s2j + (1+wj)*coef_proxy(1,3);
    c = 1/cinv;
    atp1 = ISpara2(2,1); 
    ac = atp1*rhoj/s2j + (1-rhoj)*uj/s2j + 0.5*wj - 0.5*(1+wj)*coef_proxy(1,2)...
        -(1+wj)*coef_proxy(1,3)*zz(1);
    ISpara2(1,3) = c;
    ISpara2(1,1) = c*ac; %compute IS coef for h
    
    hj = draws.z(drawi,:)';
    for t = 1:n
        if ISpara2(t,3)<=0
            error('ct<=0');
        end
        if t == 1
            xxt = [1 0];
        else
            xxt = [1 hj(t-1)]; 
        end
        hj_mean(t) = xxt * ISpara2(t,1:2)';
    end %h from posterior
    
    logpj_b = -0.5*K*log(2*pi) -0.5*sum(log(b0_var)) -0.5*sum(((bj-b0_mean).^2)./b0_var);
    logpj_SV = -0.5*log(2*pi/invVmuh) - 0.5*((uj-muh0)^2)*invVmuh ...
        +phiha*log(1+rhoj)+phihb*log(1-rhoj)-(phiha+phihb)*log(2)-betaln(phiha,phihb) ...
        -0.5*log(2*pi*sigh2_s) - 0.5*s2j/sigh2_s + 0.5*log(s2j);
    logpj_v = vp1*log(vp2)-gammaln(vp1)-vp1*log(vj)-vp2/vj; 
    logpj = logpj_b + logpj_SV + logpj_v; %prior theta   
     
%     loghj = -0.5*n*log(2*pi*s2j) + 0.5*log(1-rhoj2) - 0.5*(1-rhoj2)*uj*uj/s2j ...
%         -0.5*(n-1)*(1-rhoj)*(1-rhoj)*uj*uj/s2j; 
%     tmp1 = -0.5*(1-rhoj2)*zz(1)*zz(1)/s2j - (1-rhoj2)*uj*zz(1)/s2j;
%     tmp2 = -0.5*sum(zz(2:n).^2)/s2j - (1-rhoj)*uj*sum(zz(2:n))/s2j + rhoj*sum(zz(1:n-1).*zz(2:n))/s2j...
%         -0.5*rhoj2*sum(zz(1:n-1).^2)/s2j + rhoj*(1-rhoj)*uj*sum(zz(1:n-1))/s2j;
%     loghj = loghj + tmp1 + tmp2; %prior hh
    
%     yj = log(1+exp(-hhj));
%     yjfit = coef_proxy(:,1) + coef_proxy(:,2).*hhj + coef_proxy(:,3).*hhj.*hhj;
%     epsj = yj-yjfit;
%     logyj = n*(gammaln(0.5+0.5*wj) - 0.5*log(pi*wj) - gammaln(0.5*wj)) + 0.5*sum(zz) ...
%         -0.5*(1+wj)*sum((coef_proxy(:,1)+epsj)); %likelihood
    
%     ISa = ISpara2(:,1);
%     ISc = ISpara2(:,3);
%     logqhj = -0.5*n*log(2*pi) - 0.5*sum(log(ISc)) -0.5*sum(ISa.*ISa./ISc); %IS h
    
    tmp = (hj(2:n)-(1-rhoj)*uj - rhoj*hj(1:n-1)).^2;
    loghj = -0.5*n*log(2*pi*s2j) + 0.5*log(1-rhoj2) - 0.5*(1-rhoj2)*((hj(1)-uj)^2)/s2j ...
        -0.5*sum(tmp)/s2j; %prior h 
    
    logyj = n*(gammaln(0.5+0.5*wj) - 0.5*log(pi*wj) - gammaln(0.5*wj)) - 0.5*sum(hj) ...
        -0.5*(1+wj)*sum(log(1+exp(-hj-zz))); %likelihood
        
    resid2 = (hj - hj_mean).^2;
    ISc = ISpara2(:,3);
    logqhj = -0.5*n*log(2*pi) - 0.5*sum(log(ISc)) -0.5*sum(resid2./ISc); %IS h
    
    tmp = paraj - para_mean;
    logqpj = -0.5*KK*log(2*pi) - 0.5*logdet_para_cov - 0.5*tmp'*para_covinv*tmp; %IS theta
    
    logh2_gd(drawi) = loghj;
    logp2_gd(drawi) = logpj;
    logy2_gd(drawi) = logyj;
    logqh2_gd(drawi) = logqhj;
    logqp2_gd(drawi) = logqpj;
    logw2_gd(drawi) = logpj+loghj+logyj-logqpj-logqhj;
end
wtmp = -logw2_gd;
wlevel = max(wtmp); %mean(wtmp);
ew = exp(wtmp - wlevel);
tmp_mean = mean(ew);
[~, hac_std, ~] = HAC_regression(ew,constvec); 
tmp_std = sqrt(ndraws)*hac_std; %tmp_std = std(ew);
disp('GD: backward matching');
disp(['LML = ',num2str(-wlevel-log(tmp_mean)),' with std = ',num2str(tmp_std/tmp_mean)]);
toc;
disp(' ');






