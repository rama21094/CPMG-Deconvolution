%multi (8 ) homonuclearspins simulation
%AGG in d2o: Ala 3 methyl protons, 1Ha; 2Glycines 4Ha
%to check run time of a code, type following in the command window: tic; multispins_proton1D; toc

function multispins_proton1D()

%Define pauli matrices
sigma_x= sparse([0 1/2; 1/2 0]);
sigma_y=sparse([0 -1i/2; 1i/2 0]);
sigma_z=sparse([1/2 0; 0 -1/2]);
unit=sparse([1 0; 0 1]);

%number of spins
nspins=8; 

% Cell arrays of operators 2^nspins elements ultimately
Lx=cell(1,nspins); Ly=cell(1,nspins); Lz=cell(1,nspins);    %make arrays of coulnms=no of spins
for n=1:nspins
 Lx_current=1; Ly_current=1; Lz_current=1;
 for k=1:nspins
     if k==n
         Lx_current=kron(Lx_current,sigma_x);
         Ly_current=kron(Ly_current,sigma_y);
         Lz_current=kron(Lz_current,sigma_z);
     else
         Lx_current=kron(Lx_current,unit);
         Ly_current=kron(Ly_current,unit);
         Lz_current=kron(Lz_current,unit);
     
     end
 end
 Lx{n}=Lx_current; Ly{n}=Ly_current; Lz{n}=Lz_current;
end

%E=eye(2^nspins);                                           %identity matrix for nspins
%spy(Ly{6}); or spy(Ly{6}*Lx{1})                            %this will give blue dots where the matrix is non zero
%sparse(Lx{6})                                              %prints only non zero elements with their indices
%nmuel(sparse(Lx{6}) will give total elements while nnz will give non zero element count
%multiplying sparse matrices will save time  

%spectrometer field in MHz
Bo=500;
carrier_ppm=4.773;
carrier=carrier_ppm*Bo*2*pi;

%Hamiltonian
zeeman_freqs=carrier-(2*pi*Bo*[1.47 1.47 1.47 4.1 3.8 3.72 3.6 3.58]) ;% CH3, Ala-Ha, 2GlyHa, 3GlyHa
scalar_couplings=2*pi*[0 0 0 7 0 0 0 0 ; 0 0 0 7 0 0 0 0 ; 0 0 0 7 0 0 0 0 ; 7 7 7 0 0 0 0 0 ; 0 0 0 0 0 24 0 0 ; 0 0 0 0 24 0 0 0 ; 0 0 0 0 0 0 0 15 ; 0 0 0 0 0 0 15 0];                          

% Preallocate Hamiltonian array
H=spalloc(2^nspins,2^nspins,(nspins^2)*(2^nspins));                    %sparse allocation, dimension 2^n, max non zero values possible: (nspins^2)*(2^nspins)

% Zeeman interactions in H
for n=1:nspins
 H=H+zeeman_freqs(n)*Lz{n};
end

% Scalar couplings in H
for n=1:nspins
 for k=1:nspins
    if n~=k                                                               %~= not equal to
        H=H+scalar_couplings(n,k)*(Lx{n}*Lx{k}+Ly{n}*Ly{k}+Lz{n}*Lz{k});
    end
 end
end

%spy(H)

% Initial state
rho=spalloc(2^nspins,2^nspins,(nspins^2)*(2^nspins));
for n=1:nspins
 rho=rho+Lz{n};
end

% Detection state
coil=spalloc(2^nspins,2^nspins,(nspins^2)*(2^nspins));
for n=1:nspins
 coil=coil+Lx{n}+1i*Ly{n};
end

% Pulse Hamiltonian
Hp=spalloc(2^nspins,2^nspins,(nspins^2)*(2^nspins));
for n=1:nspins
 Hp=Hp+Ly{n};                                             %y pulse
end

% Build propagators
P_pulse=sparse(expm(-1i*Hp*(pi/2)));                      %90 y
sw=14*Bo;
time_step=1/(2*sw);
%time_step=1/normest(H);                                   %normest is estimate of norm as H isreally large
P_evol=sparse(expm(-1i*H*time_step));

% Clean up propagators, this will destroy all the values less than 1e-6
P_pulse=P_pulse.*(abs(P_pulse)>1e-6);
P_evol=P_evol.*(abs(P_evol)>1e-6);

% Simulation, stage 1: pulse
rho=P_pulse*rho*P_pulse';

% Simulation, stage 2: evolution
nsteps=8196;                     % number of steps in the simulation
fid=zeros(nsteps,1);               % preallocate the array
for n=1:nsteps
 fid(n)=trace(coil'*rho);
 rho=P_evol*rho*P_evol';
 %disp(n);
end

% Apodization
window_function=exp(-5*linspace(0,1,nsteps))';
fid=fid.*window_function;

% Fourier transform with zerofill
spectrum = fftshift(fft(fid));


% Plotting
freq=linspace(sw/2,-sw/2,nsteps);
freq1=freq./Bo;
freq2=carrier_ppm-freq1;
plot(freq2,real(spectrum));
set(gca, 'XDir','reverse');


end

