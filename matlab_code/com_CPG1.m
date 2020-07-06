% complex adaptive Hopf

clear all
niter = 2000000;
dt = 0.001;
t = (0:1:niter-1)*dt;
mu = 1;
eps = 0.9;
etaalp = 0.5;

nos = 4;
omgos = rand(nos,1)*20;
z = exp(1i*rand(nos,1)*2*pi);
alpha = rand(nos,1);

% Pteach signal construction
pteach = zeros(1,niter);
omgip = zeros(1,4);
I_max = zeros(1,4);
ipph = zeros(1,4);
omgip(1) = 5;
omgip(2) = 10;
omgip(3) = 15;
omgip(4) = 20;
I_max(1) = 1.1;
I_max(2) = 0.8;
I_max(3) = 1.5;
I_max(4) = 0.7;
ipph(1) = pi/3;
ipph(2) = pi/4;
ipph(3) = pi/2;
ipph(4) = pi/5;
for j = 1:nos
    pteach = pteach + I_max(j)*exp(1i*(omgip(j)*t+ipph(j)));
end
pteach1 = real(pteach);
St = zeros(1,niter);
et = zeros(1,niter);

omgosarr = zeros(nos,niter);
alphaarr = zeros(nos,niter);
zarr = zeros(nos,niter);
rarr = zeros(nos,niter);
angarr = zeros(nos,niter);
dz = zeros(nos,1);
domg = zeros(nos,1);
dalpha = zeros(nos,1);

for i = 1:niter
    i
    rarr(:,i) = abs(z);
    angarr(:,i) = angle(z);
    zarr(:,i) = z;
    omgosarr(:,i) = omgos;
    alphaarr(:,i) = alpha;
    
    St(i) = alpha'*real(z);
    et(i) = pteach1(i) - St(i);
    
    for j = 1:nos
        dz(j) =dt*((mu - abs(z(j))^2)*z(j) + 1i*omgos(j)*z(j) + eps*et(i));
        domg(j) = dt*((-1)*eps*et(i)*imag(z(j))/abs(z(j)));  
        dalpha(j) = dt*(etaalp*real(z(j))*et(i));
    end
    
    z = z + dz;
    omgos = omgos + domg;
    alpha = alpha + dalpha;
end


figure(5)
for j=1:nos
    angarr(j,:) = unwrap(angarr(j,:))-omgosarr(j,:).*t;
    plot(t,angarr(j,:));
    hold on
end
legend('os1','os2','os3','os4')
ylabel('angle(z)-omega*t')
xlabel('time')

figure(1)
for j = 1:nos
    plot(t,omgosarr(j,:))
    hold on
end
legend('os1','os2','os3','os4')
ylabel('omega')
xlabel('time')


figure(2)
for j = 1:nos
    plot(t,rarr(j,:))
    hold on
end
legend('os1','os2','os3','os4')
ylabel('abs(z)/r')
xlabel('time')

figure(3)
for j = 1:nos
    plot(t,alphaarr(j,:))
    hold on
end
legend('os1','os2','os3','os4')
ylabel('alpha')
xlabel('time')

% figure(4)
% for j = 1:nos
%     plot(t,angarr(j,:))
%     hold on
% end
% legend('os1','os2','os3','os4')
% ylabel('angle(z)')
%                                                                                                                                                                    
% xlabel('time')