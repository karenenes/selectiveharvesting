function grad = listnet_gradient (lx, ly, lz, list_id)
    global CORES
    m = size(lx,1);
    p = size(lx,2);
    
    s1 =  -(lx' * topp(ly));

    %s2 =   (1 / sum(exp(lz)));
    log_s2 = - logsumexp(lz); 
    %fprintf('\nsum(exp(lz)): %e (true: %e)\n', sum(exp(lz)), exp(-log_s2))

    %s3 =   (lx' * exp(lz));
    log_s3 = logsumexp(log(lx) + repmat(lz,1,columns(lx)), 1)';
    %fprintf('s3:');
    %disp(s3(1:10)');
    %fprintf('exp(log_s3):');
    %disp(exp(log_s3(1:10)'));
    
    %g = (s1 + s2 * s3); % n(i) x 1
    %fprintf('g:');
    %disp(g(1:10)');
    g = (s1 + exp(log_s2+log_s3)); % n(i) x 1
    %fprintf('stable g:');
    %disp(g(1:10)');

    f = g';
    grad = f;

    %grad = reshape(pararrayfun(CORES, f, 1:m, "VerboseLevel", 0),p,m)';
    %grad = reshape(f,p,m)';
    %disp(grad)
end
