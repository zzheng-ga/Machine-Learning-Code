function [err_train, err_test] = ModelFull(train, test)
%You implment this function by assuming a full covariance matrix. 
%err_train is the error rate on the train data
%err_test is the error rate on the test data


[n, d] = size(train);

xtrain = double(train(:, 1: d-1));
ytrain = double(train(:, d));

xtest = double(test(:, 1: d-1));
ytest = double(test(:, d));

xtrain = double(xtrain./255);
xtest = double(xtest ./255);

pos_idx = ytrain == 1;
neg_idx = ytrain == 0;

% prior
len_pos = sum(pos_idx);
len_neg = sum(neg_idx);
pi_pos = len_pos/n;
pi_neg = len_neg/n;

% mean
pos_mean = mean(xtrain(pos_idx, :));
neg_mean = mean(xtrain(neg_idx, :));

% covariance matrix
epsilon = 1e-2;
pos_cov = cov(xtrain(pos_idx, :));
neg_cov = cov(xtrain(neg_idx, :));
cov_mat = (pos_cov*(len_pos-1) + neg_cov*(len_neg-1))/n + epsilon*eye(d-1);

% predict
pos_train = mvnpdf(xtrain, pos_mean, cov_mat) * pi_pos;
neg_train = mvnpdf(xtrain, neg_mean, cov_mat) * pi_neg;

pos_test = mvnpdf(xtest, pos_mean, cov_mat) * pi_pos;
neg_test = mvnpdf(xtest, neg_mean, cov_mat) * pi_neg;

ytrain_label = max(sign(pos_train - neg_train), 0);
ytest_label = max(sign(pos_test - neg_test), 0);

% error rate
err_train = sum(abs(ytrain_label - (ytrain)))/length(ytrain);
err_test = sum(abs(ytest_label - ytest))/length(ytest);

end