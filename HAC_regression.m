% Run regression y_t = x_t' * b + u_t
% the error term u_t could be serially correlated and heteroskedastic
% return the coefficient estimates, HAC standard errors and p-values

% The HAC estimator is Newey-West with the number of lags = floor(4*(T/100)^(2/9)) where T is the number of observations.

function [coef, hac_std, p_value] = HAC_regression(y,x)
% Inputs:
%    y: a T-by-1 vector of output data
%    x: a T-by-M matrix of input data, including constant
% Outputs:
%    coef: a (M+1)-by-1 vector of coefficients, with the first one being the constant
%    hac_std: a (M+1)-by-1 vector of the corresponding HAC standard errors
%    p_value: a (M+1)-by-1 vector of the corresponding p-values


%% check data
[T,M] = size(x);
nobs = length(y);
if nobs ~= T
    disp('y and x don''t have the same number of observations!');
    return;
end


%% calculate coefficient estimates and g_t
[coef,~,u] = regress(y,x);
g_mat = zeros(T,M);
for j = 1:M
    g_mat(:,j) = x(:,j).*u;
end


%% calculate the long-run variance of g_t by the NW method
nof_nw_lags = floor(4*(T/100)^(2/9));
k = nof_nw_lags + 1;
s = g_mat' * g_mat;
for j = 1:nof_nw_lags
    tau_j = g_mat(1:T-j,:)' * g_mat(j+1:T,:);
    s = s + (1 - j/k) * (tau_j + tau_j');
end
s = s / (T-M);


%% calculate the HAC standard errors of the coefficients
inv_xx = (x'*x)\eye(M);
var_coef = T * inv_xx * s * inv_xx;
hac_std = sqrt(diag(var_coef));



%% calculate the p-values of the coefficients
%p_value = normcdf(coef, zeros(size(coef)), hac_std);
p_value = normcdf(coef./hac_std);
idx = find(p_value > 0.5);
p_value(idx) = 1 - p_value(idx);
p_value = p_value * 2;


