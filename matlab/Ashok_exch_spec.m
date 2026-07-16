% 2 spin-1/2
% Propagates M+ to generate the NMR spectrum in the presence of exchange

% NOTE: If the acquisition time is not an integral multiple of the time
% period of the spectral resonance, the fft routine will generate a 
% phase error in the resonance that is equal to the average phases at the 
% start and end of the fft input vector. Applying a strong line broadening
% function wil get rid of this phase error.

clear

% Spectrum parameters
frq = 60;                     % spectrometer frequency in MHz
sw = 1000;                    % Sweep width (Hz)
at = 1;                       % Spectrum acquisition time (s)
dwellt=1/sw;                  % Dwell time (s)
np=at/dwellt;                 % Number of datapoints

% System parameters (thermodynamic, kinetic)
kex = 0.001;                     % Exchange rate constant (s-1)
pA = 0.66;                     % Population of state A
pB = 1-pA;                    % Population of state B
kAB = pB*kex;                 % Forward rate constant (s-1)
kBA = pA*kex;                 % Backward rate constant (s-1)

% System parameters (magnetic)
R2A = 7;                      % Transverse relaxtion rate of state A (s-1)
R2B = 7;                      % Transverse relaxtion rate of state B (s-1)
dwppm = 3.5;                 % Chemical shift difference (wB-wA, ppm)
%wAppm = 8-0.5*dwppm;           % Chemical shift of state A (ppm)
wAppm = 2.0;                  % Chemical shift of state A (ppm)
wBppm = wAppm + dwppm;        % Chemical shift of state B (ppm)

% Initialize magnetization
Mzero = [pA pB]';

% Initialize arrays
Mag = zeros(size(Mzero,1),size(Mzero,2),length(dwppm),np);
Magt = zeros(length(dwppm),np);
spec = zeros(length(dwppm),np);
time = zeros(np);

% Outer loop for different chemical shifts
for ii = 1:length(dwppm)

wA = wAppm(ii)*frq*2*pi;
wB = wBppm(ii)*frq*2*pi;


A = [-R2A-kAB+(1i*wA)         kBA 
           kAB        -R2B-kBA+(1i*wB)];    % The Bloch-McConnell matrix

% Inner loop for time evolution        
for inc=1:np

    tevol=dwellt*(inc-1);
    Mag(:,:,ii,inc) = (expm(A*tevol))*Mzero;          % Note use of expm
    Magt(ii,inc) = Mag(1,1,ii,inc)+Mag(2,1,ii,inc);
    time(inc) = tevol;
    
end

end

% Processing the fid
time=0:dwellt:(np-1)*dwellt;      % Time axis 
freq=linspace(0,sw*(np-1)/np,np); % Frequency axis (note use of linspace)

lb=3;                             % Line broadening for EM
EMF=exp(-lb.*time);               % Exponential multiplication function

for ii=1:length(dwppm)
Magt(ii,1) = 0.5*Magt(ii,1);      % First point multiplication by 0.5
spec(ii,:)=(real(fft(Magt(ii,:))))/np;  % Fourier Transform
end

% Plotting
plot(freq/60,5*spec(1,:),'m-')
%hold on
%plot(freq(np/2:np)/60,spec(1,np/2:np),'k-')
%hold on
%plot(freq/60,spec(2,:),'r-')
%plot(freq/60,spec(3,:),'g-')
%plot(freq/60,spec(4,:),'b-')
%plot(freq/60,spec(5,:),'m-')
%plot(freq/60,spec(6,:),'y-')
%plot(freq/60,spec(7,:),'c-')
%hold off