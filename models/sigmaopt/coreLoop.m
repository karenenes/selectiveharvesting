function [mut, Ct, score, score_exploration] =  coreLoop(edges, mu0, omega0, all_inds, y, alpha, obs_sigma )

  n = length(y);
  A = sparse(edges(1,:), edges(2,:), 1, n, n);
  L = diag(sum(A, 2) + omega0) - A;
  L = full(L);
  Ct = inv(L);
  mut = mu0 * ones(n,1);

  %%% update to posterior
  if ~isempty(all_inds)
    X = sparse(1:length(all_inds), all_inds(:)', 1, length(all_inds), length(Ct));
    [mut, ~, Ct] = Gaussian_posterior(mut, [], Ct, X, y(all_inds, :), obs_sigma);
  end
    
  %%% sigma-optimality acquisition
  score_exploration = sum(Ct, 2) ./ sqrt(diag(Ct) + obs_sigma^2);

  score = mut + alpha * score_exploration;  

endfunction
