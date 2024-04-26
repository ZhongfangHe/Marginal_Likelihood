% Consider the linear regression model with SV:
% yt = xt'*b + N(0,exp(zt)), zt = (1-phi)*u + phi*ztm1 + etat, etat~N(0,s)
% compute its marginal likelihood by IS or GD

clear;
dbstop if warning;
dbstop if error;
rng(123456);


%% Gather data
dgp = {'Simulation','EquityPremium'};
ind_dgp = 2;
disp(['DGP = ',dgp{ind_dgp}]);
if ind_dgp == 1 %simulation
    n = 300;
    K = 2;%10; %2;
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
disp(['n = ',num2str(n), ', K = ', num2str(K)]);
    

%% MCMC
tic;
ndraws = 5000*2; %5000;
burnin = 2000;
disp(['burnin = ',num2str(burnin),', ndraws = ',num2str(ndraws)]);
ntotal = burnin + ndraws;

b0_mean = zeros(K,1);
b0_var = 100*ones(K,1);

muh0 = 0; invVmuh = 1/10; % mean: p(mu) ~ N(mu0, Vmu)
phiha = 8; phihb = 2; % AR(1): p(phi) ~ 0.5 * (1 + betarnd(a,b))
priorSV = [muh0 invVmuh phiha phihb]'; %collect prior hyperparameters
muh = muh0 + sqrt(1/invVmuh) * randn;
phih = 0.5*(1+betarnd(phiha,phihb));

sigh2_s = 1;
sigh2 = gamrnd(0.5,2*sigh2_s);
sigh = sqrt(sigh2);

b0_OLS = regress(y,x);
resid_OLS = y - x*b0_OLS;
hSV = log(var(resid_OLS))*ones(n,1); %initialize by log OLS residual variance.

draws.b = zeros(ndraws,K);
draws.SVpara = zeros(ndraws,4); % [mu phi sig2 sig]
draws.z = zeros(ndraws,n); %residual variance
for drawi = 1:ntotal
    yvar = exp(hSV);
    ystd = sqrt(yvar); 
    yy = y./ystd;
    xx = x./repmat(ystd,1,K);
    Binv = diag(1./b0_var) + xx'*xx;    
    Binvb = xx'*yy; 
    tmp = mvnrnd(Binvb,Binv)';
    b = Binv\tmp;
    
    resid = y - x*b;
    logz2 = log(resid.^2 + 1e-100);
    [hSV, muh, phih, sigh] = SV_update2(logz2, hSV, ...
        muh, phih, sigh, sigh2_s, priorSV);     
    if drawi > burnin
        i = drawi-burnin;
        draws.b(i,:) = b';
        draws.z(i,:) = hSV';
        draws.SVpara(i,:) = [muh phih sigh^2 sigh];
    end   
end
disp('MCMC is completed!');
toc;
disp(' ');


%% 1. IS by using AR(1) for ht
tic;
u = draws.SVpara(:,1);
rho = draws.SVpara(:,2);
s = draws.SVpara(:,4);
para_est = [draws.b  u  log(1+rho)-log(1-rho)  log(s.^2)];
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
    rhoj2 = rhoj^2; %simulate theta from IS   
    
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
    logpj = logpj_b + logpj_SV; %prior theta
     
    tmp = (hj(2:n)-(1-rhoj)*uj-rhoj*hj(1:n-1)).^2;
    loghj = -0.5*n*log(2*pi*s2j) + 0.5*log(1-rhoj2) - 0.5*(1-rhoj2)*((hj(1)-uj)^2)/s2j ...
        -0.5*sum(tmp)/s2j; %prior h    
    
    eps = y - x*bj;
    logyj = -0.5*n*log(2*pi) - 0.5*sum(hj+eps.*eps.*exp(-hj)); %likelihood
    
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
para_est = [draws.b  u  log(1+rho)-log(1-rho)  log(s.^2)];
KK = size(para_est,2);
para_mean = mean(para_est)';
para_cov = cov(para_est);
para_covhalf = chol(para_cov)';
para_covinv = para_cov\eye(KK);
logdet_para_cov = 2*sum(log(diag(para_covhalf))); %calibrate Gaussian IS for fixed para

R2vec2 = zeros(n,1);
coef_proxy = zeros(n,3);
constvec = ones(ndraws,1);
for t = 1:n
    ht = draws.z(:,t);
    yy = exp(-ht);
    xx = [constvec ht ht.^2];
    coef = regress(yy,xx);
    yyfit = xx*coef;
    if coef(3)<0
        xx = [constvec ht];
        coef_level = regress(yy,xx);
        coef = [coef_level; 0];
        yyfit = xx*coef_level;
    end %ensure the coef on ht2 is positive
    coef_proxy(t,:) = coef';
    R2vec2(t) = var(yyfit)/var(yy);
end %linearize exp(-ht)

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
    rhoj2 = rhoj^2; %simulate theta from IS 
    
    d = y - x*bj;
    cinv = 1/s2j + d(n)*d(n)*coef_proxy(n,3);
    c = 1/cinv;
    bc = rhoj/s2j;
    ac = (1-rhoj)*uj/s2j - 0.5 - 0.5*d(n)*d(n)*coef_proxy(n,2);
    ISpara2(n,3) = c;
    ISpara2(n,2) = c*bc;
    ISpara2(n,1) = c*ac;
    t = n-1;
    while t >= 2
        ctp1 = ISpara2(t+1,3);
        cinv = -ctp1*rhoj2/s2j2 + (1+rhoj2)/s2j + d(t)*d(t)*coef_proxy(t,3);
        c = 1/cinv;
        bc = rhoj/s2j;
        atp1 = ISpara2(t+1,1); 
        ac = atp1*rhoj/s2j + (1-rhoj)*(1-rhoj)*uj/s2j - 0.5 - 0.5*d(t)*d(t)*coef_proxy(t,2);
        ISpara2(t,3) = c;
        ISpara2(t,2) = c*bc;
        ISpara2(t,1) = c*ac;
        t = t-1;
    end
    ctp1 = ISpara2(2,3);
    cinv = -ctp1*rhoj2/s2j2 + 1/s2j + d(1)*d(1)*coef_proxy(1,3);
    c = 1/cinv;
    atp1 = ISpara2(2,1); 
    ac = atp1*rhoj/s2j + (1-rhoj)*uj/s2j - 0.5 - 0.5*d(1)*d(1)*coef_proxy(1,2);
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
    logpj = logpj_b + logpj_SV; %prior theta
     
    loghj = -0.5*n*log(2*pi*s2j) + 0.5*log(1-rhoj2) - 0.5*(1-rhoj2)*uj*uj/s2j ...
        -0.5*(n-1)*(1-rhoj)*(1-rhoj)*uj*uj/s2j; %prior h    
    
    yj = exp(-hj);
    yjfit = coef_proxy(:,1) + coef_proxy(:,2).*hj + coef_proxy(:,3).*hj.*hj;
    epsj = yj-yjfit;
    logyj = -0.5*n*log(2*pi) - 0.5*sum(d.*d.*(coef_proxy(:,1)+epsj)); %likelihood
    
    ISa = ISpara2(:,1);
    ISc = ISpara2(:,3);
    logqhj = -0.5*n*log(2*pi) - 0.5*sum(log(ISc)) -0.5*sum(ISa.*ISa./ISc); %IS h
    
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
para_est = [draws.b  u  log(1+rho)-log(1-rho)  log(s.^2)];
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

logh_gd = zeros(ndraws,1);
logp_gd = logh_gd;
logy_gd = logh_gd;
logqh_gd = logh_gd;
logqp_gd = logh_gd;
logw_gd = logh_gd;
hj_mean = zeros(n,1);
for drawi = 1:ndraws
    paraj = para_est(drawi,:)';
    bj = paraj(1:K);
    uj = paraj(K+1);
    rhoj = (exp(paraj(K+2))-1)/(exp(paraj(K+2))+1);
    sj = sqrt(exp(paraj(K+3)));
    s2j = sj^2;
    rhoj2 = rhoj^2; %theta from posterior   
    
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
    logpj = logpj_b + logpj_SV; %prior theta
     
    tmp = (hj(2:n)-(1-rhoj)*uj-rhoj*hj(1:n-1)).^2;
    loghj = -0.5*n*log(2*pi*s2j) + 0.5*log(1-rhoj2) - 0.5*(1-rhoj2)*((hj(1)-uj)^2)/s2j ...
        -0.5*sum(tmp)/s2j; %prior h    
    
    eps = y - x*bj;
    logyj = -0.5*n*log(2*pi) - 0.5*sum(hj+eps.*eps.*exp(-hj)); %likelihood
    
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
para_est = [draws.b  u  log(1+rho)-log(1-rho)  log(s.^2)];
KK = size(para_est,2);
para_mean = mean(para_est)';
para_cov = cov(para_est);
para_covhalf = chol(para_cov)';
para_covinv = para_cov\eye(KK);
logdet_para_cov = 2*sum(log(diag(para_covhalf))); %calibrate Gaussian IS for fixed para

coef_proxy = zeros(n,3);
constvec = ones(ndraws,1);
resid = zeros(ndraws,n);
R2vec2 = zeros(n,1);
for t = 1:n
    ht = draws.z(:,t);
    yy = exp(-ht);
    xx = [constvec ht ht.^2];
    coef = regress(yy,xx);
    yyfit = xx*coef;
    if coef(3)<0
        xx = [constvec ht];
        coef_level = regress(yy,xx);
        coef = [coef_level; 0];
        yyfit = xx*coef_level;
    end %ensure the coef on ht2 is positive
    coef_proxy(t,:) = coef';
    resid(:,t) = yy-yyfit;
    R2vec2(t) = var(yyfit)/var(yy);
end %linearize exp(-ht)

ndraws = ndraws*1;
logh2_gd = zeros(ndraws,1);
logp2_gd = logh2_gd;
logy2_gd = logh2_gd;
logqh2_gd = logh2_gd;
logqp2_gd = logh2_gd;
logw2_gd = logh2_gd;
ISpara2 = zeros(n,3); %const, htm1, var(resid)
hj_mean = zeros(n,1);
for drawi = 1:ndraws
    paraj = para_est(drawi,:)';
    bj = paraj(1:K);
    uj = paraj(K+1);
    rhoj = (exp(paraj(K+2))-1)/(exp(paraj(K+2))+1);
    sj = sqrt(exp(paraj(K+3)));
    s2j = sj^2;
    s2j2 = s2j^2;
    rhoj2 = rhoj^2; %theta from posterior
    
    d = y - x*bj;
    cinv = 1/s2j + d(n)*d(n)*coef_proxy(n,3);
    c = 1/cinv;
    bc = rhoj/s2j;
    ac = (1-rhoj)*uj/s2j - 0.5 - 0.5*d(n)*d(n)*coef_proxy(n,2);
    ISpara2(n,3) = c;
    ISpara2(n,2) = c*bc;
    ISpara2(n,1) = c*ac;
    t = n-1;
    while t >= 2
        ctp1 = ISpara2(t+1,3);
        cinv = -ctp1*rhoj2/s2j2 + (1+rhoj2)/s2j + d(t)*d(t)*coef_proxy(t,3);
        c = 1/cinv;
        bc = rhoj/s2j;
        atp1 = ISpara2(t+1,1); 
        ac = atp1*rhoj/s2j + (1-rhoj)*(1-rhoj)*uj/s2j - 0.5 - 0.5*d(t)*d(t)*coef_proxy(t,2);
        ISpara2(t,3) = c;
        ISpara2(t,2) = c*bc;
        ISpara2(t,1) = c*ac;
        t = t-1;
    end
    ctp1 = ISpara2(2,3);
    cinv = -ctp1*rhoj2/s2j2 + 1/s2j + d(1)*d(1)*coef_proxy(1,3);
    c = 1/cinv;
    atp1 = ISpara2(2,1); 
    ac = atp1*rhoj/s2j + (1-rhoj)*uj/s2j - 0.5 - 0.5*d(1)*d(1)*coef_proxy(1,2);
    ISpara2(1,3) = c;
    ISpara2(1,1) = c*ac; %compute IS coef for h
    
    hj = draws.z(drawi,:)';
    for t = 1:n
        if t == 1
            xxt = [1 0];
        else
            xxt = [1 hj(t-1)]; 
        end
        hj_mean(t) = xxt * ISpara2(t,1:2)';        
        if ISpara2(t,3)<=0
            error('ct<=0');
        end
    end %h from posterior
    
    logpj_b = -0.5*K*log(2*pi) -0.5*sum(log(b0_var)) -0.5*sum(((bj-b0_mean).^2)./b0_var);
    logpj_SV = -0.5*log(2*pi/invVmuh) - 0.5*((uj-muh0)^2)*invVmuh ...
        +phiha*log(1+rhoj)+phihb*log(1-rhoj)-(phiha+phihb)*log(2)-betaln(phiha,phihb) ...
        -0.5*log(2*pi*sigh2_s) - 0.5*s2j/sigh2_s + 0.5*log(s2j); 
    logpj = logpj_b + logpj_SV; %prior theta
     
    loghj = -0.5*n*log(2*pi*s2j) + 0.5*log(1-rhoj2) - 0.5*(1-rhoj2)*uj*uj/s2j ...
        -0.5*(n-1)*(1-rhoj)*(1-rhoj)*uj*uj/s2j; %prior h    
    
    epsj = resid(drawi,:)';
    logyj = -0.5*n*log(2*pi) - 0.5*sum(d.*d.*(coef_proxy(:,1)+epsj)); %likelihood
    
    ISa = ISpara2(:,1);
    ISc = ISpara2(:,3);
    logqhj = -0.5*n*log(2*pi) - 0.5*sum(log(ISc)) -0.5*sum(ISa.*ISa./ISc); %IS h

%     tmp = (hj(2:n)-(1-rhoj)*uj-rhoj*hj(1:n-1)).^2;
%     loghj = -0.5*n*log(2*pi*s2j) + 0.5*log(1-rhoj2) - 0.5*(1-rhoj2)*((hj(1)-uj)^2)/s2j ...
%         -0.5*sum(tmp)/s2j; %prior h  
%     
%     eps = y - x*bj;
%     logyj = -0.5*n*log(2*pi) - 0.5*sum(hj+eps.*eps.*exp(-hj)); %likelihood
%     
%     resid2 = (hj - hj_mean).^2;
%     ISd = ISpara2(:,3);
%     logqhj = -0.5*n*log(2*pi) - 0.5*sum(log(ISd)) -0.5*sum(resid2./ISd); %IS h
    
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




    



