function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


htheta=X*theta;
theta_extra=theta;
theta_extra(1)=0;
J=(((htheta-y)'*(htheta-y))+(lambda*(theta_extra'*theta_extra)))/(2*m);
%fprintf('cost cost %d\n',J);

for i=1:length(grad)
  grad(i)=(1/m)*sum((htheta-y).*X(:,i))+((lambda/m)*theta_extra(i));
end





% =========================================================================

grad = grad(:);

end
