function result = msgPassing(edges, in_train, responses, iter)

datalen = length(in_train);
data = sparse(edges(1,:), edges(2,:), 1, datalen, datalen);
degree = sum(data, 2);
newdata=1/datalen*diag(degree);
degree = sum(data(in_train, :), 2);
newdata(in_train, in_train)=diag(degree);
newf=0.13*ones(datalen,1);
newf(in_train)=responses(in_train);
f=[zeros(datalen, 1); newf];
Wu=[data newdata];
Duu=sum(Wu,2);
for j=1:iter
    oldf=f;
    f(1:datalen)=bsxfun(@rdivide, Wu*f, Duu);
    if(norm(f-oldf,2)<1e-6)
        break;
    end
end
result=f(1:datalen);
end
