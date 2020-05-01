function [w,b] = stochastic_subgradient_descent( X,Y,c )
[N,n]= size(X);
w=zeros(n,1);
b=0;
optim_result=optim_obj(X,Y,c,w,b);
optim_result_old=optim_result+10;
tol=1e-10;
iter=0;
m=inf;
while abs(optim_result_old-optim_result)>=tol && m >0
    optim_result_old=optim_result;
    iter=iter+1;
    % according to rule of thumb, set learning rate to 1/i
    learning_step=1/iter;
    tmp=(1-Y.*(X*w+b))>0; %find the index of tmp>0, which makes contribution
    %do stochastic sampling
    Xs=X(tmp,:);
    Ys=Y(tmp,:);
    m=size(Xs,1);
    if m==0
        continue
    end
    random_index= randsample(m,1);
    
    random_X=Xs(random_index,:);
    random_Y=Ys(random_index,:);
    gradient_w = w-m*random_X'*random_Y*c;
    gradient_b = - m*random_Y*c;
    %update the parameters based on the subgradient
    b = b - learning_step * gradient_b ;
    w = w - learning_step * gradient_w ;
    % calculate the new value of objective optimization 
    optim_result = optim_obj(X,Y,c,w,b) ;
end
w=w';
fprintf('Stochastic Subgradient iteration is %f. \n', iter);
end


