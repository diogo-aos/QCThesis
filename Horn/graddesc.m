function D=graddesc(xyData,q,steps)
% function graddesc(xyData,q,[steps])
% purpose: performing quantum clustering in and moving the 
%          data points down the potential gradient
% input: xyData - the data vectors
%        q=a parameter for the parsen window variance (q=1/(2*sigma^2))
%        steps=number of gradient descent steps (default=50)
% output: D=location of data o=point after GD 
if nargin<3 
    steps=50;
end

eta=0.1; 
D=xyData;
[V,P,E,dV] = qc(xyData,q,D);
for j=1:4
   for i=1:(steps/4)
       dV=normc(dV')';
       D=D-eta*dV;
       [V,P,E,dV] = qc (xyData,q,D);
   end;
   eta=eta*0.5;
end

