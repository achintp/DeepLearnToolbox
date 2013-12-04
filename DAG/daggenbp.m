function nn = daggenbp(nn)
%NNBP performs backpropagation
% nn = nnbp(nn) returns an neural network structure with updated weights 
    
    n = nn.n;
    sparsityError = 0;
    switch nn.output
        case 'sigm'
            d{n} = - nn.e .* (nn.a{n} .* (1 - nn.a{n}));
        case {'softmax','linear'}
            d{n} = - nn.e;
    end
    
    for i = (n - 1) : -1 : 2
        for j = n : -1: i+1
            % Derivative of the activation function
            switch nn.activation_function
                case 'sigm'
                    d_act{i}{j} = nn.a{i}{j} .* (1 - nn.a{i}{j});
                case 'tanh_opt'
                    d_act{i}{j} = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn.a{i}{j}.^2);
            end
            
            if(nn.nonSparsityPenalty>0)
                pi = repmat(nn.p{i}, size(nn.a{i}, 1), 1);
                sparsityError = [zeros(size(nn.a{i},1),1) nn.nonSparsityPenalty * (-nn.sparsityTarget ./ pi + (1 - nn.sparsityTarget) ./ (1 - pi))];
            end
            
            % Backpropagate first derivatives
            if j==n % in this case in d{n} there is not the bias term to be removed
                d_seg{i}{j} = (d{j} * nn.W{i}{j} + sparsityError) .* d_act{i}{j}; % Bishop (5.56)
            else % in this case in d{i} the bias term has to be removed
                d_seg{i}{j} = (d{j}(:,2:end) * nn.W{i}{j} + sparsityError) .* d_act{i}{j};
            end
            
            if(nn.dropoutFraction>0)
                d{i}{j} = d{i}{j} .* [ones(size(d{i}{j},1),1) nn.dropOutMask{i}{j}];
            end
            if (isempty(d{i}))
                d{i} = d_seg{i}{j};
            else
                d{i} = d{i} + d_seg{i}{j};
            end
            
        end
        
    end
    
    for i = 1 : (n - 1)
        for j = (i + 1) : n
            if j==n
                nn.dW{i}{j} = (d{j}' * nn.a{i}{j}) / size(d{j}, 1);
            else
                nn.dW{i}{j} = (d{j}(:,2:end)' * nn.a{i}{j}) / size(d{j}, 1);
            end
        end
    end
end
