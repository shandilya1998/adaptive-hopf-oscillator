% complex adaptive Hopf

clear all
niter = 1000000;
dt = 0.001;
mu = 1;
eps = 0.9;
omgos = 40;
omgip = 30;
amp_max = 1;
z = exp(1i*rand*2*pi);

omgosarr = zeros(1,niter);
zarr = zeros(1,niter);
rarr = zeros(1,niter);
phiarr = zeros(1,niter);
t = (0:1:niter-1)*dt;
inp_arr = amp_max*exp(1i*omgip*t);
for k = 1:niter
    k
    phiarr(k) = angle(z);
    rarr(k) = abs(z);
    zarr(k) = z;
    omgosarr(k) = omgos;
    dz = dt*((mu - abs(z)^2)*z + 1i*omgos*z + eps*imag(inp_arr(k)));
    domg = dt*((-1)*eps*imag(inp_arr(k))*imag(z)/abs(z));   
    z = z + dz;
    
    omgos = omgos + domg;
    
end

plot(omgosarr)
figure(2)
plot(rarr)
figure(3)
plot(phiarr)
