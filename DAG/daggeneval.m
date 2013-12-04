function [loss] = daggeneval(nn, loss, train_x, train_y, val_x, val_y)
%NNEVAL evaluates performance of neural network
% Returns a updated loss struct
assert(nargin == 4 || nargin == 6, 'Wrong number of arguments');

% training performance
nn                    = daggenff(nn, train_x, train_y);
loss.train.e(end + 1) = nn.L;

% validation performance
if nargin == 6
    nn                    = daggenff(nn, val_x, val_y);
    loss.val.e(end + 1)   = nn.L;
end

%calc misclassification rate if softmax
if strcmp(nn.output,'softmax')
    [er_train, ~]               = daggentest(nn, train_x, train_y);
    loss.train.e_frac(end+1)    = er_train;
    
    if nargin == 6
        [er_val, ~]             = daggentest(nn, val_x, val_y);
        loss.val.e_frac(end+1)  = er_val;
    end
end

end