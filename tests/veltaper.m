function[r] = veltaper(d,x,m,p)
% VTAPERW.M 
%
%	Velasco's tapered local Whittle likelihood
%
%		Ixx(k) = w(x)*conj(w(x)),
%
%	where
%								  N
%		w(k) =  (2*pi*n)^(-1/2)  sum  x(t)*exp(i*2*pi*(k-1)*t/N), 1 <= k <= N.
%                  				 t=1
%
%										   Katsumi Shimotsu, July 2004
%
% 		INPUT	x: data (n*1 vector)
%				m: truncation number
%				d: parameter value
%               p: the order of the taper.
%                   p = 2: Bartlett
%                   p = 3: Kolmogorov 
%
%____________________________________________________________________________

[n,nn] = size(x);

t = (0:1:n-1)';
lambda = 2*pi*t/n;

if p == 2	

    mm = ceil(n/2);
    h = 1 - abs(t+1-mm)/mm;

else
    
    pp = fix((n+2)/3);
    h = ones(pp,1);
    h2 = [(1:1:pp)';(pp-1:-1:1)'];
    h3 = conv(h,h2);
    h = [h3; zeros(rem(n+2,3),1)];

end

x = x.*h;
wx = (2*pi*n)^(-1/2)*conj(fft(conj(x))).*exp(i*lambda);

ind = p:p:m;
lambda = lambda(1+ind);
wx = wx(1+ind);
Ix = wx.*conj(wx);

g = sum((lambda.^(2*d)).*Ix)*p/m;
r = log(g) - 2*d*sum(log(lambda))*p/m;
