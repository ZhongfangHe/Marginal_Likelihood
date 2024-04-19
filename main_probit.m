% Consider the probit model:
% yt = I{zt>0}, zt = xt'*b + etat, etat~N(0,1)
%
% Compute its marginal likelihood by IS and GD




clear;
dbstop if warning;
dbstop if error;
rng(123456);
addpath(genpath('C:\Users\Zhongfang\Documents\My Research\Bayesian_TVP\2021Feb\Functions'));
addpath(genpath('C:\Users\Zhongfang\Documents\My Research\Bayesian_TVP\2021Apr\Functions'));
addpath(genpath('C:\Users\Zhongfang\Documents\My Research\Bayesian_TVP\2021Jul\Functions'));
addpath(genpath('C:\Users\Zhongfang\Documents\My Research\Bayesian_TVP\2021Sep\Functions'));
addpath(genpath('C:\Users\Zhongfang\Documents\My Research\Bayesian_TVP\2024Apr_Marginal_Likelihood\Functions'));


%% Gather data for a probit model
dgp = {'Simulation','Recession'};
ind_dgp = 2;
disp(['DGP = ',dgp{ind_dgp}]);
if ind_dgp == 1 %simulation
    % btrue = [1 0.1 0.5]';
    btrue = 0.1*ones(2,1); %0.1*ones(50,1); %0.1*ones(80,1);
    K = length(btrue);
    n = 200;
    disp(['n = ',num2str(n),', K = ', num2str(K)]);
    if K == 1
        x = ones(n,1);
    else
        x = [ones(n,1) rand(n,K-1)-0.5]; %[ones(n,1)  cumsum(randn(n,K-1))];
    end 
    ztrue = x * btrue + randn(n,1);
    y = double(ztrue > 0);
else %recession data    
    read_file = 'Data_Recession.xlsx';
    read_sheet = 'Data'; 
    data_y = readmatrix(read_file, 'Sheet', read_sheet, 'Range', 'B2:B273'); %1954Q3 to 2021Q3
    data_x = readmatrix(read_file, 'Sheet', read_sheet, 'Range', 'C2:M273'); %1954Q3 to 2021Q3,11 non-constant regressors
    data = [data_y  data_x];
    [ntotal,ndata] = size(data);
    h = 4; %forecast horizon in quarters
    y = data(h+1:ntotal, 1); %recession indicator
    x = [ones(ntotal-h,1) data(1:ntotal-h, 2:ndata)]; %lagged regressors
    [n,K] = size(x);   
end
minNum = 1e-100;


%% MCMC
bvar0 = 100;
b_prior_cov = bvar0 * eye(K);
b_prior_cov_inv = b_prior_cov\eye(K);

xtimesx = x' * x;
Ainv = b_prior_cov_inv + xtimesx;
A = Ainv \ eye(K);
Ax = A*x'; 
pmat = eye(n) - x * A * x'; %maybe used repeatedly

burnin = 2000;
ndraws = 5000*2;
ntotal = burnin + ndraws;
disp('PXDA scale starts:');
draws_pxda2.b = zeros(ndraws,K);
draws_pxda2.z = zeros(ndraws,n);
draws_pxda2.s = zeros(ndraws,1);
tic;
z = zeros(n,1);
b = mvnrnd(zeros(K,1),b_prior_cov)';
for drawi = 1:ntotal
% draw z from p(z|y,b)
    for t = 1:n
        zt_mean = x(t,:) * b;
        if y(t) == 1
            z(t) = zt_mean + trandn(-zt_mean, Inf);
        else
            z(t) = zt_mean + trandn(-Inf, -zt_mean);
        end
    end

    % draw s from p(s|y,z) integrating out b (easier than conditioning on b which will need MH)
    rss = z' * pmat * z;
    s2 = 1/gamrnd(0.5*n, 2/rss);
    s = sqrt(s2);


    % draw b from p(b|y,z,s)
    Ainv = b_prior_cov_inv + xtimesx;
    A = xtimesx \ eye(K);
    zz = z/s;
    a = A * x' * zz;
    b = mvnrnd(a, A)';

    if drawi > burnin
        i = drawi - burnin;
        draws_pxda2.b(i,:) = b';
        draws_pxda2.z(i,:) = z';
        draws_pxda2.s(i) = s;
    end  

    if round(drawi/2000) == (drawi/2000)
        disp([num2str(drawi),' draws out of ', num2str(ntotal), ' have completed!']);
        toc;
    end    
end
disp('PXDA scale is completed!');
toc;
disp(' ');
draws = draws_pxda2;



%% 1. IS: direct Gaussian
tic;
bmean = mean(draws.b)';
bcov = cov(draws.b);
bcov_inv = bcov\eye(K);
b_covec = bcov_inv * bmean;
b_sw = bmean'*bcov_inv*bmean;
bcov_half = chol(bcov)';
logdet_bcov = 2*sum(log(diag(bcov_half)));
logw = zeros(ndraws,1);
eps = randn(ndraws,K);
for drawi = 1:ndraws
    bj = bmean + bcov_half * eps(drawi,:)';
    logpj = -0.5*K*log(2*pi*bvar0)-0.5*sum(bj.^2)/bvar0; %log prior
    zp = normcdf(x*bj);
    zp1 = zp;
    zp0 = 1-zp;
    idx = find(zp1<minNum);
    tmp = (zp1(idx)-minNum)/minNum; 
    zp1(idx) = minNum*exp(tmp - 0.5*(tmp.^2)); 
    idx = find(zp0<minNum);
    tmp = (zp0(idx)-minNum)/minNum; 
    zp0(idx) = minNum*exp(tmp - 0.5*(tmp.^2)); 
    logyj = sum(y.*log(zp1) + (1-y).*log(zp0)); %log likelihood
    logqj = -0.5*K*log(2*pi)-0.5*logdet_bcov-0.5*bj'*bcov_inv*bj-0.5*b_sw+bj'*b_covec; %log q
    wj = -logqj + logpj + logyj;
    logw(drawi) = wj;
end
wlevel = max(logw);
ew = exp(logw - wlevel);
tmp_mean = mean(ew);
tmp_std = std(ew);
disp('IS: direct Gaussian');
disp(['LML = ',num2str(wlevel+log(tmp_mean)),' with std = ',num2str(tmp_std/tmp_mean)]);
toc;
disp(' ');



%% 2. IS: backward matching
tic;
para_est = draws.b;
constvec = ones(ndraws,1);
coef_proxy = zeros(n,3);
R2vec = zeros(n,1);
for t = 1:n
    yt = y(t);
    xt = x(t,:)';
    zp = normcdf(para_est*xt);
    zp1 = zp;
    zp0 = 1-zp;
    idx = find(zp1<minNum);
    tmp = (zp1(idx)-minNum)/minNum; 
    zp1(idx) = minNum*exp(tmp - 0.5*(tmp.^2)); 
    idx = find(zp0<minNum);
    tmp = (zp0(idx)-minNum)/minNum; 
    zp0(idx) = minNum*exp(tmp - 0.5*(tmp.^2)); 
    yy = yt*log(zp1) + (1-yt)*log(zp0);
    tmp = para_est*xt;
    xx = [constvec tmp tmp.^2];
    Binv = eye(3)/10000 + xx'*xx;
    Binvb = xx'*yy;
    coef = Binv\Binvb;
    yyfit = xx*coef;
    if coef(3)>0
        xx = [constvec tmp];
        Binv = eye(2)/10000 + xx'*xx;
        Binvb = xx'*yy;
        coef_level = Binv\Binvb;
        coef = [coef_level; 0];
        yyfit = xx*coef_level;
    end
    coef_proxy(t,:) = coef';
    R2vec(t) = var(yyfit)/var(yy);
end %expand log likelihood

tmp1 = 0;
tmp2 = 0;
for t = 1:n
    xt = x(t,:)';
    tmp1 = tmp1 + coef_proxy(t,3)*(xt*xt');
    tmp2 = tmp2 + coef_proxy(t,2)*xt;
end
qcov_inv = b_prior_cov_inv - 2*tmp1;
q_covec = tmp2;
qcov = qcov_inv\eye(K);
qcov_half = chol(qcov)';
logdet_qcov = 2*sum(log(diag(qcov_half)));
qmean = qcov*q_covec;
q_sw = qmean'*qcov_inv*qmean; %calibrate IS based on expansion

logw2 = zeros(ndraws,1);
eps = randn(ndraws,K);
nsim = ndraws;
for drawi = 1:nsim
    bj = qmean + qcov_half*eps(drawi,:)';
    logpj = -0.5*K*log(2*pi*bvar0)-0.5*sum(bj.^2)/bvar0; %log prior
    zp = normcdf(x*bj);
    zp1 = zp;
    zp0 = 1-zp;
    idx = find(zp1<minNum);
    tmp = (zp1(idx)-minNum)/minNum; 
    zp1(idx) = minNum*exp(tmp - 0.5*(tmp.^2)); 
    idx = find(zp0<minNum);
    tmp = (zp0(idx)-minNum)/minNum; 
    zp0(idx) = minNum*exp(tmp - 0.5*(tmp.^2)); 
    logyj = sum(y.*log(zp1) + (1-y).*log(zp0)); %log likelihood
    logqj = -0.5*K*log(2*pi)-0.5*logdet_qcov-0.5*bj'*qcov_inv*bj-0.5*q_sw+bj'*q_covec;
    wj = -logqj + logpj + logyj;    
    logw2(drawi) = wj; 
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


%% 3. GD: direct Gaussian
tic;
bmean = mean(draws.b)';
bcov = cov(draws.b);
bcov_inv = bcov\eye(K);
b_covec = bcov_inv * bmean;
b_sw = bmean'*bcov_inv*bmean;
bcov_half = chol(bcov)';
logdet_bcov = 2*sum(log(diag(bcov_half)));
logw_gd = zeros(ndraws,1);
for drawi = 1:ndraws
    bj = draws.b(drawi,:)'; 
    logpj = -0.5*K*log(2*pi*bvar0)-0.5*sum(bj.^2)/bvar0; %log prior
    zp = normcdf(x*bj);
    zp1 = zp;
    zp0 = 1-zp;
    idx = find(zp1<minNum);
    tmp = (zp1(idx)-minNum)/minNum; 
    zp1(idx) = minNum*exp(tmp - 0.5*(tmp.^2)); 
    idx = find(zp0<minNum);
    tmp = (zp0(idx)-minNum)/minNum; 
    zp0(idx) = minNum*exp(tmp - 0.5*(tmp.^2)); 
    logyj = sum(y.*log(zp1) + (1-y).*log(zp0)); %log likelihood
    logqj = -0.5*K*log(2*pi)-0.5*logdet_bcov-0.5*bj'*bcov_inv*bj-0.5*b_sw+bj'*b_covec; %log q
    wj = -logqj + logpj + logyj;
    logw_gd(drawi) = wj;
end
wtmp = -logw_gd;
wlevel = max(wtmp); %mean(wtmp);
ew = exp(wtmp - wlevel);
tmp_mean = mean(ew);
[~, hac_std, ~] = HAC_regression(ew,constvec); 
tmp_std = sqrt(ndraws)*hac_std; %tmp_std = std(ew);
disp('GD: direct Gaussian');
disp(['LML = ',num2str(-wlevel-log(tmp_mean)),' with std = ',num2str(tmp_std/tmp_mean)]);
toc;
disp(' ');


%% 4. GD: backward matching
tic;
para_est = draws.b;
constvec = ones(ndraws,1);
coef_proxy = zeros(n,3);
R2vec = zeros(n,1);
for t = 1:n
    yt = y(t);
    xt = x(t,:)';
    zp = normcdf(para_est*xt);
    zp1 = zp;
    zp0 = 1-zp;
    idx = find(zp1<minNum);
    tmp = (zp1(idx)-minNum)/minNum; 
    zp1(idx) = minNum*exp(tmp - 0.5*(tmp.^2)); %Taylor expansion for x*bj~0
    idx = find(zp0<minNum);
    tmp = (zp0(idx)-minNum)/minNum; 
    zp0(idx) = minNum*exp(tmp - 0.5*(tmp.^2)); %Taylor expansion for x*bj~0
    yy = yt*log(zp1) + (1-yt)*log(zp0);
    tmp = para_est*xt;
    xx = [constvec tmp tmp.^2];
    Binv = eye(3)/10000 + xx'*xx;
    Binvb = xx'*yy;
    coef = Binv\Binvb;
    yyfit = xx*coef;
    if coef(3)>0
        xx = [constvec tmp];
        Binv = eye(2)/10000 + xx'*xx;
        Binvb = xx'*yy;
        coef_level = Binv\Binvb;
        coef = [coef_level; 0];
        yyfit = xx*coef_level;
    end
    coef_proxy(t,:) = coef';
    R2vec(t) = var(yyfit)/var(yy);
end %expand log likelihood

tmp1 = 0;
tmp2 = 0;
for t = 1:n
    xt = x(t,:)';
    tmp1 = tmp1 + coef_proxy(t,3)*(xt*xt');
    tmp2 = tmp2 + coef_proxy(t,2)*xt;
end
qcov_inv = b_prior_cov_inv - 2*tmp1;
q_covec = tmp2;
qcov = qcov_inv\eye(K);
qcov_half = chol(qcov)';
logdet_qcov = 2*sum(log(diag(qcov_half)));
qmean = qcov*q_covec;
q_sw = qmean'*qcov_inv*qmean; %calibrate IS based on expansion

logw2_gd = zeros(ndraws,1);
nsim = ndraws;
for drawi = 1:nsim
    bj = para_est(drawi,:)'; 
    logpj = -0.5*K*log(2*pi*bvar0)-0.5*sum(bj.^2)/bvar0; %log prior
    zp = normcdf(x*bj);
    zp1 = zp;
    zp0 = 1-zp;
    idx = find(zp1<minNum);
    tmp = (zp1(idx)-minNum)/minNum; 
    zp1(idx) = minNum*exp(tmp - 0.5*(tmp.^2));  
    idx = find(zp0<minNum);
    tmp = (zp0(idx)-minNum)/minNum; 
    zp0(idx) = minNum*exp(tmp - 0.5*(tmp.^2));  
    logyj = sum(y.*log(zp1) + (1-y).*log(zp0)); %log likelihood
    logqj = -0.5*K*log(2*pi)-0.5*logdet_qcov-0.5*bj'*qcov_inv*bj-0.5*q_sw+bj'*q_covec;
    wj = -logqj + logpj + logyj;    
    logw2_gd(drawi) = wj; 
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



%% GD: direct Gaussian (truncated)
% tic;
% truc_lvl = 0.95;
% bthresh = chi2inv(truc_lvl,K);
% bmean = mean(draws.b)';
% bcov = cov(draws.b);
% bcov_inv = bcov\eye(K);
% b_covec = bcov_inv * bmean;
% b_sw = bmean'*bcov_inv*bmean;
% bcov_half = chol(bcov)';
% logdet_bcov = 2*sum(log(diag(bcov_half)));
% w_MHMcov = [];
% minNum = 1e-100;
% for drawi = 1:ndraws
%     bj = draws.b(drawi,:)';
%     tmp = (bj-bmean)'*bcov_inv*(bj-bmean);
%     if tmp < bthresh
%         logpj = -0.5*K*log(2*pi*bvar0)-0.5*sum(bj.^2)/bvar0; %log prior
%         zp = normcdf(x*bj);
%         zp1 = zp;
%         zp1(zp1<minNum) = minNum;
%         zp0 = 1-zp;
%         zp0(zp0<minNum)= minNum;
%         logyj = sum(y.*log(zp1) + (1-y).*log(zp0)); %log likelihood
%         logqj = -0.5*K*log(2*pi)-0.5*logdet_bcov-0.5*bj'*bcov_inv*bj-0.5*b_sw+bj'*b_covec-log(truc_lvl);
%         wj = logqj - logpj - logyj;
%         w_MHMcov = [w_MHMcov; wj];
%     end
% end
% ew = exp(w_MHMcov);
% ewmean = mean(ew);
% nw = length(ew);
% ew_demean = ew - ewmean;
% nwlag = floor(4*(nw/100)^(2/9));
% ewvar = sum(ew_demean.^2);
% for j = 1:nwlag
%     tau_j = ew_demean(1:nw-j)' * ew_demean(j+1:nw);
%     ewvar = ewvar + (1 - j/(nwlag+1)) * (tau_j + tau_j');
% end
% ewvar = ewvar/nw; 
% py_mean = 1/ewmean;
% py_std = sqrt(ewvar/(ewmean^4));
% disp('Modified HM with posterior mean and cov');
% disp(['LML = ',num2str(log(py_mean)),' with std = ',num2str(py_std/abs(py_mean))]);
% toc;
% disp(' ');


%% Chib method
% tic;
% bb = median(draws.b)'; %selected local value
% logp = -0.5*K*log(2*pi*bvar0)-0.5*sum(bb.^2)/bvar0; %log prior
% zp = normcdf(x*bb);
% zp1 = zp;
% zp1(zp1<minNum) = minNum;
% zp0 = 1-zp;
% zp0(zp0<minNum)= minNum;
% logy = sum(y.*log(zp1) + (1-y).*log(zp0)); %log likelihood
% Binv = eye(K)/bvar0 + x'*x;
% B = Binv\eye(K);
% Bhalf = chol(B)';
% logdet_B = 2*sum(log(diag(Bhalf)));
% item1 = logp + logy - (-0.5*K*log(2*pi)-0.5*logdet_B-0.5*(bb'*Binv*bb));
% tmpvec = zeros(ndraws,1);
% for drawi = 1:ndraws
%     zj = draws.z(drawi,:)';
%     Binvb = x'*zj;
%     tmpvec(drawi) = -0.5*Binvb'*B*Binvb + bb'*Binvb;
% end
% wlevel = max(tmpvec);
% ew = exp(tmpvec-wlevel);
% ewmean = mean(ew);
% nw = length(ew);
% ew_demean = ew - ewmean;
% nwlag = floor(4*(nw/100)^(2/9));
% ewvar = sum(ew_demean.^2);
% for j = 1:nwlag
%     tau_j = ew_demean(1:nw-j)' * ew_demean(j+1:nw);
%     ewvar = ewvar + (1 - j/(nwlag+1)) * (tau_j + tau_j');
% end
% ewvar = ewvar/nw; 
% py_mean = 1/ewmean;
% py_std = sqrt(ewvar/(ewmean^4));
% disp('Chib method:');
% disp(['LML = ',num2str(item1-wlevel+log(py_mean)),' with std = ',num2str(py_std/abs(py_mean))]);
% toc;
% disp(' ');


