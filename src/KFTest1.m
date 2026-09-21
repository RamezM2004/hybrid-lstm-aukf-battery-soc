%% 2D Adaptive Kalman Filter Simulation (Akhlaghi-style)
% Ramez Masadeh project – Adaptive vs Classic KF
clc; clear; close all;

%% Simulation parameters
dt = 0.1;
N = 800;
t = (0:N-1) * dt;

%% State definition: [x vx y vy]'
x_true = zeros(4, N);
x_true(:,1) = [0; 1.5; 0; 0.8];

%% Process noise (true)
q_true = 0.1;
Q_true = q_true * [dt^4/4 dt^3/2   0        0;
                   dt^3/2 dt^2     0        0;
                   0       0     dt^4/4  dt^3/2;
                   0       0     dt^3/2   dt^2];

%% Measurement model
H = [1 0 0 0;
     0 0 1 0];
m = 2;
n = 4;

%% Time-varying measurement noise R_true
R1 = diag([1 1]);
R2 = diag([25 25]);
R3 = diag([4 4]);
R_true = zeros(2,2,N);

R_true(:,:,1:200) = repmat(R1,1,1,200);
R_true(:,:,201:500) = repmat(R2,1,1,300);
R_true(:,:,501:end) = repmat(R3,1,1,300);

%% State transition (constant velocity)
F = [1 dt 0  0;
     0  1 0  0;
     0  0 1 dt;
     0  0 0  1];

%% Simulate true trajectory
for k = 2:N
    w = mvnrnd(zeros(4,1), Q_true)';
    x_true(:,k) = F * x_true(:,k-1) + w;
end

%% Generate measurements
z = zeros(2,N);
for k = 1:N
    vk = mvnrnd(zeros(2,1), R_true(:,:,k))';
    z(:,k) = H * x_true(:,k) + vk;
end

%% Initial KF settings
Q_fixed = 0.01 * eye(4);   % wrong Q
R_fixed = diag([2 2]);     % wrong R

xKF = zeros(4,N); xKF(:,1) = [0;0;0;0];
PKF = 10 * eye(4);

xAKF = xKF; 
PAKF = PKF;

Q_adapt = Q_fixed;
R_adapt = R_fixed;

%% Adaptation parameters
beta = 0.6;
alpha = 0.95;
window = 40;

innovHist = zeros(2,N);

%% Storage for R history
R_adapt_hist = zeros(2,2,N);

%% Filtering loop
for k = 2:N
    % -------- Fixed KF --------
    xpred = F * xKF(:,k-1);
    Ppred = F * PKF * F' + Q_fixed;
    
    nu = z(:,k) - H*xpred;
    S = H*Ppred*H' + R_fixed;
    K = Ppred * H' / S;
    
    xKF(:,k) = xpred + K*nu;
    PKF = (eye(4)-K*H) * Ppred;

    % -------- Adaptive KF --------
    xpredA = F * xAKF(:,k-1);
    PpredA = F * PAKF * F' + Q_adapt;

    nuA = z(:,k) - H*xpredA;
    S_A = H*PpredA*H' + R_adapt;
    K_A = PpredA * H' / S_A;

    xAKF(:,k) = xpredA + K_A*nuA;
    PAKF = (eye(4)-K_A*H)*PpredA;

    innovHist(:,k) = nuA;
    
    % ------ Adaptive update every window ------
    if k > window
        windowInnov = innovHist(:,k-window+1:k);
        S_hat = cov(windowInnov');   % empirical
        
        HPHT = H * PpredA * H';
        R_update = S_hat - HPHT;

        R_adapt = beta*R_adapt + (1-beta)*R_update;

        % enforce positive semidefinite
        R_adapt = (R_adapt + R_adapt')/2;
        [V,D] = eig(R_adapt);
        D = max(D, 1e-6*eye(2));
        R_adapt = V*D*V';

        % update Q
        outer = (nuA*nuA') - S_A;
        Q_update = K_A * outer * K_A';
        Q_adapt = alpha*Q_adapt + (1-alpha)*Q_update;

        R_adapt_hist(:,:,k) = R_adapt;
    end
end

%% ---- Plot Results ----
figure;
plot(x_true(1,:), x_true(3,:), 'k', 'LineWidth',2); hold on;
plot(xKF(1,:), xKF(3,:), 'b--', 'LineWidth',1);
plot(xAKF(1,:), xAKF(3,:), 'g', 'LineWidth',1.5);
scatter(z(1,1:10:end), z(2,1:10:end),20,'r','filled');
legend('True','KF','AKF','Measurements');
title('2D Trajectory'); xlabel('x'); ylabel('y'); grid on;

%% Plot R evolution
figure;
plot(squeeze(R_adapt_hist(1,1,:)),'g','LineWidth',1.5); hold on;
plot(squeeze(R_true(1,1,:)),'k--','LineWidth',1.5);
plot(ones(1,N)*R_fixed(1,1),'b--');
legend('Adaptive R','True R','Fixed R'); title('Measurement Noise R Evolution');
grid on;
