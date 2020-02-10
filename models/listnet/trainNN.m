function omega = trainNN(list_id, X, y, T, e, quiet=false, omega=0)
    % load constants
    source models/listnet/global.m;
    %source topp.m;
    %pkg load parallel;

    m = size(X,1);
    n_features = size(X,2);
    n_lists = size(unique(list_id),1);
    %fprintf("m=%d, n_features=%d, n_lists=%d\n", m, n_features, n_lists)

    % linear neural network parameter initialization
    if omega == 0
      omega = rand(n_features,1)*INIT_VAR;
    end

    for t = 1:T
        if quiet == false
            fprintf("iteration %d: ", t)
        end
        
        % forward propagation
        z =  X * omega;
        %disp(z(1:10)')
         
        % cost
        if quiet == false
            fprintf("computing cost... ")
        end

        % with regularization
        J = listwise_cost(y,z, list_id) + ((z.*z)'.*LAMBDA);
        % without regularization
        %J = listwise_cost(y,z, list_id);
        
        % gradient
        if quiet == false
            fprintf("computing gradient...")
        end

        grad = listnet_gradient(X, y, z, list_id);
        %disp(grad(1:10))
        
        % parameter update
        omega = omega - (e .* sum(grad',2));
        
        if quiet == false
            fprintf("\n")
        end
    end
end

