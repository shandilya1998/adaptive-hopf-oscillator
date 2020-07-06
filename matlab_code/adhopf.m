% complex adaptive Hopf

clear all
niter = 1000000;
dt = 0.01;
mu = 1;
eps = 0.9;
omgos = 40;
omgip = 30;
x = 1;
y = 0;
amp_max = 1;

omgosarr = zeros(1,niter);
xarr = zeros(1,niter);
yarr = zeros(1,niter);
t = (0:1:niter-1)*dt;
inp_arr = amp_max*sin(omgip*t);

for k = 1:niter
    k
    xarr(k) = x;
    yarr(k) = y;
    omgosarr(k) = omgos;
    dx =dt*((mu - (x^2+y^2))*x - omgos*y + eps*inp_arr(k));
    dy =dt*((mu - (x^2+y^2))*y + omgos*x);
    domg = dt*((-1)*eps*inp_arr(k)*y/sqrt(x^2+y^2));   
    x = x + dx;
    y = y + dy;
    omgos = omgos + domg;
    
end

plot(omgosarr)