function [w,b] = subgradient_descent( X,Y,c )
[N,n]= size(X);
w=zeros(n,1);
b=0;
optim_result=optim_obj(X,Y,c,w,b);
optim_result_old=optim_result+10;
tol=1e-10;
iter=0;
while abs(optim_result_old-optim_result)>=tol
    optim_result_old=optim_result;
    iter=iter+1;
    % according to rule of thumb, set learning rate to 1/i
    learning_step=1/iter;
    tmp=(1-Y.*(X*w+b))>0; %find the index of tmp>0, which makes contribution
    gradient_w=w-X(tmp,:)'*Y(tmp)*c;
    gradient_b= - sum(Y(tmp))*c;
    %update the parameters based on the subgradient
    b = b - learning_step * gradient_b ;
    w = w - learning_step * gradient_w ;
    % calculate the new value of objective optimization 
    optim_result = optim_obj(X,Y,c,w,b) ;
end
w=w';
fprintf('Subgradient iteration is %f. \n', iter);
end


