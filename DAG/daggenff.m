function nn = daggenff(nn, x, y)
%NNFF performs a feedforward pass
% nn = nnff(nn, x, y) returns an neural network structure with updated
% layer activations, error and loss (nn.a, nn.e and nn.L)

    n = nn.n;
    m = size(x, 1);
    
    x = [ones(m,1) x];
    for j = 2 : n
        nn.a{1}{j} = x;
    end
    %nn.a{1}{3} = zeros(size(x));

    %feedforward pass
    for i = 2 : n-1
        for j = i+1 : n
            switch nn.activation_function
                case 'sigm'
                    % Calculate the unit's outputs (including the bias term)
                    addTerm = 0;
                    for k = 1 : i-1
                        addTerm = addTerm + nn.a{k}{i}*nn.W{k}{i}';
                    end
                    nn.a{i}{j} = sigm(addTerm);
                case 'tanh_opt'
                    addTerm = 0;
                    for k = 1 : i-1
                        addTerm = addTerm + nn.a{k}{i}*nn.W{k}{i}';
                    end
                    nn.a{i}{j} = tanh_opt(addTerm);
            end
            
            %dropout
            if(nn.dropoutFraction > 0)
                if(nn.testing)
                    nn.a{i}{j} = nn.a{i}{j}.*(1 - nn.dropoutFraction);
                else
                    nn.dropOutMask{i}{j} = (rand(size(nn.a{i}{j}))>nn.dropoutFraction);
                    nn.a{i}{j} = nn.a{i}{j}.*nn.dropOutMask{i}{j};
                end
            end
            
            %calculate running exponential activations for use with sparsity
            if(nn.nonSparsityPenalty>0)
                nn.p{i} = 0.99 * nn.p{i} + 0.01 * mean(nn.a{i}, 1);
            end
            
            %Add the bias term
            nn.a{i}{j} = [ones(m,1) nn.a{i}{j}];
        end
    end
    switch nn.output 
        case 'sigm'
            addTerm = 0;
            for k = 1 : n - 1
                addTerm = addTerm + nn.a{k}{n}*nn.W{k}{n}'; 
            end
            nn.a{n} = sigm(addTerm);
        case 'linear'
            addTerm = 0;
            for k = 1 : n - 1
                addTerm = addTerm + nn.a{k}{n}*nn.W{k}{n}'; 
            end
            nn.a{n} = addTerm;
        case 'softmax'
            addTerm = 0;
            for k = 1 : n - 1
                addTerm = addTerm + nn.a{k}{n}*nn.W{k}{n}'; 
            end
            nn.a{n} = addTerm;
            nn.a{n} = exp(bsxfun(@minus, nn.a{n}, max(nn.a{n},[],2)));
            nn.a{n} = bsxfun(@rdivide, nn.a{n}, sum(nn.a{n}, 2)); 
    end

    %error and loss
    nn.e = y - nn.a{n};
    
    switch nn.output
        case {'sigm', 'linear'}
            nn.L = 1/2 * sum(sum(nn.e .^ 2)) / m; 
        case 'softmax'
            nn.L = -sum(sum(y .* log(nn.a{n}))) / m;
    end
end
