function nn = daggenapplygrads(nn)
%NNAPPLYGRADS updates weights and biases with calculated gradients
% nn = nnapplygrads(nn) returns an neural network structure with updated
% weights and biases
    
    for i = 1 : (nn.n - 1)
        for j = (i + 1) : nn.n
            
        if(nn.weightPenaltyL2>0)
            dW = nn.dW{i}{j} + nn.weightPenaltyL2 * [zeros(size(nn.W{i}{j},1),1) nn.W{i}{j}(:,2:end)];
        else
            dW = nn.dW{i}{j};
        end
        
        dW = nn.learningRate * dW;
        
        if(nn.momentum>0)
            nn.vW{i}{j} = nn.momentum*nn.vW{i}{j} + dW;
            dW = nn.vW{i}{j};
        end
            
        nn.W{i}{j} = nn.W{i}{j} - dW;
        end
    end
end
