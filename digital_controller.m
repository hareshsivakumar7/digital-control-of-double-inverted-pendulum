% ====================================================================
% Digital State-Space Control of a 2-DOF Inverted Pendulum
% ====================================================================
% This program designs, analyzes, and implements digital state-space
% controllers for the stabilization and trajectory tracking of a 
% 2-DOF inverted pendulum system.
% 
% The program includes:
% 1. System modeling and discretization
% 2. Controllers:
%    - State feedback design using pole placement
%    - LQR state feedback design
%    - LQR with integral action for reference tracking
% 3. Observers:
%    - Luenberger observer design
%    - Kalman filter design
% 4. Performance analysis:
%    - Stability analysis
%    - Robustness to disturbances and parameter variations
%    - Tracking performance
% ====================================================================

clear all; close all; clc;

%% 1. System Parameters and State-Space Model
disp('1. Defining system parameters and creating state-space model...');

% Define system parameters
m0 = 1;      % Mass of cart (kg)
m1 = 0.5;    % Mass of first pendulum (kg)
m2 = 0.3;    % Mass of second pendulum (kg)
l1 = 0.5;    % Length of first pendulum (m)
l2 = 0.3;    % Length of second pendulum (m)
g = 9.81;    % Gravity (m/s^2)
Ts = 0.1;    % Sampling time (s)

% Define continuous-time system matrices
A = [0 0 0 1 0 0;
     0 0 0 0 1 0;
     0 0 0 0 0 1;
     0 (m1+m2)*l1*g/(m0+m1+m2) m2*l2*g/(m0+m1+m2) 0 0 0;
     0 -((m0+m1+m2)*g/l1 + m2*l2*g/l1^2) -m2*l2*g/l1 0 0 0;
     0 (m1+m2)*l1*g/(m2*l2) -g/l2 0 0 0];

B = [0;
     0;
     0;
     1/(m0+m1+m2);
     -l1/(m0+m1+m2);
     -(m1+m2)*l1/(m2*l2)];

C = [1 0 0 0 0 0;  % Cart position
     0 1 0 0 0 0;  % Pendulum 1 angle
     0 0 1 0 0 0]; % Pendulum 2 angle

D = [0; 0; 0];

% Create continuous-time state-space system
sys_c = ss(A, B, C, D);

% Discretize using Zero-Order Hold (ZOH)
sys_d = c2d(sys_c, Ts, 'zoh');

% Extract discrete-time matrices
A_d = sys_d.A;
B_d = sys_d.B;
C_d = sys_d.C;
D_d = sys_d.D;

% Display system information
disp('System order:');
n = size(A_d, 1);
disp(n);

disp('System outputs:');
p = size(C_d, 1);
disp(p);

disp('Discrete-time state matrices:');
disp('A_d:'); disp(A_d);
disp('B_d:'); disp(B_d);
disp('C_d:'); disp(C_d);
disp('D_d:'); disp(D_d);

%% 2. System Analysis
disp('2. Analyzing system properties...');

% Check controllability
Co = ctrb(A_d, B_d);
rank_Co = rank(Co);
fprintf('Rank of controllability matrix: %d (should be %d)\n', rank_Co, n);

if rank_Co == n
    disp('The system is controllable.');
else
    disp('Warning: The system is not controllable. Control design may be limited.');
end

% Check observability
Ob = obsv(A_d, C_d);
rank_Ob = rank(Ob);
fprintf('Rank of observability matrix: %d (should be %d)\n', rank_Ob, n);

if rank_Ob == n
    disp('The system is observable.');
else
    disp('Warning: The system is not observable. Observer design may be limited.');
end

% Compute open-loop poles
open_loop_poles = eig(A_d);
disp('Open-loop poles:');
disp(open_loop_poles);

%% 3. State Feedback Design Using Pole Placement
disp('3. Designing state feedback controller using pole placement...');

% Define desired closed-loop pole locations
% Natural frequencies and damping ratios for continuous-time design
wn_cart = 2.0;    % Natural frequency for cart
zeta_cart = 0.8;  % Damping ratio for cart
wn_pend1 = 3.0;   % Natural frequency for first pendulum
zeta_pend1 = 0.7; % Damping ratio for first pendulum
wn_pend2 = 3.5;   % Natural frequency for second pendulum
zeta_pend2 = 0.7; % Damping ratio for second pendulum

% Convert to continuous-time poles
s_poles = [
    -zeta_cart*wn_cart + wn_cart*sqrt(1-zeta_cart^2)*1i;
    -zeta_cart*wn_cart - wn_cart*sqrt(1-zeta_cart^2)*1i;
    -zeta_pend1*wn_pend1 + wn_pend1*sqrt(1-zeta_pend1^2)*1i;
    -zeta_pend1*wn_pend1 - wn_pend1*sqrt(1-zeta_pend1^2)*1i;
    -zeta_pend2*wn_pend2 + wn_pend2*sqrt(1-zeta_pend2^2)*1i;
    -zeta_pend2*wn_pend2 - wn_pend2*sqrt(1-zeta_pend2^2)*1i;
];

% Convert to discrete-time poles using z = e^(s*Ts)
desired_poles = exp(s_poles * Ts);

fprintf('Desired discrete-time poles:\n');
disp(desired_poles);

% Compute the feedback gain matrix K using pole placement
K_pp = place(A_d, B_d, desired_poles);

fprintf('State feedback gain matrix K (pole placement):\n');
disp(K_pp);

% Verify the closed-loop poles
A_cl_pp = A_d - B_d*K_pp;  % Closed-loop A matrix
cl_poles_pp = eig(A_cl_pp);

fprintf('Actual closed-loop poles (pole placement):\n');
disp(cl_poles_pp);

%% 4. LQR State Feedback Design
disp('4. Designing LQR state feedback controller...');

% Define LQR Cost Function Matrices
Q_lqr = diag([10, 50, 50, 1, 1, 1]);  % State cost weights
R_lqr = 0.1;                          % Control effort weight

% Solve the discrete-time LQR problem
[K_lqr, S_lqr, e_lqr] = dlqr(A_d, B_d, Q_lqr, R_lqr);

fprintf('LQR state feedback gain matrix K:\n');
disp(K_lqr);

% Compute closed-loop poles with LQR
A_cl_lqr = A_d - B_d*K_lqr;
cl_poles_lqr = eig(A_cl_lqr);

fprintf('Closed-loop poles (LQR):\n');
disp(cl_poles_lqr);

%% 5. Luenberger Observer Design
disp('5. Designing Luenberger observer...');

% For the observer, we'll only use the cart position and first pendulum angle
C_obs = C_d(1:2,:);  % Use first two outputs

% Place observer poles to be faster than controller poles
controller_poles = cl_poles_lqr;  % Use LQR poles as reference
observer_poles = controller_poles-2;  % Make observer poles faster

% Compute observer gain using pole placement
L = place(A_d', C_obs', observer_poles)';

fprintf('Observer gain matrix L:\n');
disp(L);

%% 6. Kalman Filter Design
disp('6. Designing Kalman filter...');

% Define process and measurement noise covariances
Q_kf = 0.001 * eye(n);          % Process noise covariance
R_kf = 0.01 * eye(size(C_obs,1)); % Measurement noise covariance

% Compute steady-state Kalman gain
[~, L_kf, ~] = dlqe(A_d, eye(n), C_obs, Q_kf, R_kf);

fprintf('Kalman filter gain matrix L_kf:\n');
disp(L_kf);

%% 7. LQR with Integral Action for Reference Tracking
disp('7. Designing LQR with integral action for reference tracking...');

% Augment the system with an integrator for cart position tracking
C_track = C_d(1,:);  % Track only cart position
m_track = 1;         % Number of outputs to track

% Augmented system matrices
A_aug = [A_d, zeros(n, m_track); 
        -C_track, eye(m_track)];
B_aug = [B_d; 
        zeros(m_track, 1)];

% Augmented LQR cost function
Q_aug = blkdiag(Q_lqr, 100);  % Add weight for integrator state
R_aug = R_lqr;               % Keep the same control weight

% Solve the augmented LQR problem
[K_aug, S_aug, e_aug] = dlqr(A_aug, B_aug, Q_aug, R_aug);

% Extract the state feedback and integral gains
K_x = K_aug(:, 1:n);          % State feedback gain
K_i = K_aug(:, n+1:end);      % Integral gain

fprintf('Augmented LQR State Feedback Gain K_x:\n');
disp(K_x);
fprintf('Augmented LQR Integral Gain K_i:\n');
disp(K_i);

%% 8. Simulation of State Feedback Controller (PP)
disp('8. Simulating state feedback controller (pole placement)...');

% Initial conditions
x0 = [0.1; 0.05; 0.05; 0; 0; 0];  % Initial state

% Time settings
t_final = 5;  % Final simulation time (seconds)
num_steps = floor(t_final / Ts);
t_pp = (0:num_steps) * Ts;

% Initialize arrays
x_pp = zeros(n, length(t_pp));
x_pp(:,1) = x0;
u_pp = zeros(1, length(t_pp)-1);
y_pp = zeros(size(C_d,1), length(t_pp));
y_pp(:,1) = C_d * x0;

% Simulation loop
for k = 1:num_steps
    % Compute control input
    u_pp(k) = -K_pp * x_pp(:,k);
    
    % Update state
    x_pp(:,k+1) = A_d * x_pp(:,k) + B_d * u_pp(k);
    
    % Compute output
    y_pp(:,k+1) = C_d * x_pp(:,k+1);
end

% Performance evaluation
settling_time_pp = NaN;
for i = length(t_pp):-1:2
    if max(abs(x_pp(1:3,i))) > 0.02  % 2% criterion for settling
        settling_time_pp = t_pp(i);
        break;
    end
end

fprintf('Pole Placement Controller Performance:\n');
fprintf('  Settling time (2%% criterion): %.2f seconds\n', settling_time_pp);
fprintf('  Maximum control effort: %.2f N\n', max(abs(u_pp)));
fprintf('  Maximum cart displacement: %.2f m\n', max(abs(y_pp(1,:))));
fprintf('  Maximum pendulum 1 angle: %.2f degrees\n', max(abs(y_pp(2,:)))*180/pi);
fprintf('  Maximum pendulum 2 angle: %.2f degrees\n', max(abs(y_pp(3,:)))*180/pi);

%% 9. Simulation of LQR State Feedback Controller
disp('9. Simulating LQR state feedback controller...');

% Time settings
t_lqr = (0:num_steps) * Ts;

% Initialize arrays
x_lqr = zeros(n, length(t_lqr));
x_lqr(:,1) = x0;
u_lqr = zeros(1, length(t_lqr)-1);
y_lqr = zeros(size(C_d,1), length(t_lqr));
y_lqr(:,1) = C_d * x0;

% Simulation loop
for k = 1:num_steps
    % Compute control input
    u_lqr(k) = -K_lqr * x_lqr(:,k);
    
    % Update state
    x_lqr(:,k+1) = A_d * x_lqr(:,k) + B_d * u_lqr(k);
    
    % Compute output
    y_lqr(:,k+1) = C_d * x_lqr(:,k+1);
end

% Performance evaluation
settling_time_lqr = NaN;
for i = length(t_lqr):-1:2
    if max(abs(x_lqr(1:3,i))) > 0.02  % 2% criterion for settling
        settling_time_lqr = t_lqr(i);
        break;
    end
end

fprintf('LQR Controller Performance:\n');
fprintf('  Settling time (2%% criterion): %.2f seconds\n', settling_time_lqr);
fprintf('  Maximum control effort: %.2f N\n', max(abs(u_lqr)));
fprintf('  Maximum cart displacement: %.2f m\n', max(abs(y_lqr(1,:))));
fprintf('  Maximum pendulum 1 angle: %.2f degrees\n', max(abs(y_lqr(2,:)))*180/pi);
fprintf('  Maximum pendulum 2 angle: %.2f degrees\n', max(abs(y_lqr(3,:)))*180/pi);

%% 10. Simulation of Observer-Based Controller
disp('10. Simulating observer-based controller...');

% Initialize arrays
t_obs = (0:num_steps) * Ts;
x_true = zeros(n, length(t_obs));  % True system states
x_hat = zeros(n, length(t_obs));   % Estimated states
y_meas = zeros(2, length(t_obs));  % Measured outputs (position and angle1)
u_obs = zeros(1, length(t_obs)-1); % Control inputs

% Initial conditions
x_true(:,1) = [0.1; 0.05; 0.05; 0; 0; 0];  % True initial state
x_hat(:,1) = zeros(n,1);                    % Observer starts from zero
y_meas(:,1) = C_obs * x_true(:,1);          % First measurement

% Simulation loop
for k = 1:num_steps
    % Compute control input using estimated states
    u_obs(k) = -K_lqr * x_hat(:,k);
    
    % Update true system state
    x_true(:,k+1) = A_d * x_true(:,k) + B_d * u_obs(k);
    
    % Compute measurement
    y_meas(:,k+1) = C_obs * x_true(:,k+1);
    
    % Update observer state estimate
    x_hat(:,k+1) = A_d * x_hat(:,k) + B_d * u_obs(k) + L * (y_meas(:,k) - C_obs * x_hat(:,k));
end

% Performance evaluation
settling_time_obs = NaN;
for i = length(t_obs):-1:2
    if max(abs(x_true(1:3,i))) > 0.02  % 2% criterion for settling
        settling_time_obs = t_obs(i);
        break;
    end
end

fprintf('Observer-Based Controller Performance:\n');
fprintf('  Settling time (2%% criterion): %.2f seconds\n', settling_time_obs);
fprintf('  Maximum control effort: %.2f N\n', max(abs(u_obs)));
fprintf('  Maximum estimation error: %.4f\n', max(max(abs(x_true - x_hat))));

%% 11. Simulation of Kalman Filter Controller with Process Noise
disp('11. Simulating Kalman filter controller with process and measurement noise...');

% Initialize arrays
t_kf = (0:num_steps) * Ts;
x_true_kf = zeros(n, num_steps+1);  % True system states (6×num_steps+1)
x_hat_kf = zeros(n, num_steps+1);   % Kalman filter estimated states (6×num_steps+1)
y_meas_kf = zeros(2, num_steps+1);  % Noisy measurements (2×num_steps+1)
u_kf = zeros(1, num_steps);         % Control inputs (1×num_steps)

% Error covariance matrix
P = eye(n);  % Initial estimate covariance

% Initial conditions
x_true_kf(:,1) = [0.1; 0.05; 0.05; 0; 0; 0];  % True initial state
x_hat_kf(:,1) = zeros(n,1);                   % Kalman filter starts from zero
noise_std = sqrt([R_kf(1,1); R_kf(2,2)]);  % Extract standard deviations
y_meas_kf(:,1) = C_obs * x_true_kf(:,1) + noise_std .* randn(2,1);  % First noisy measurement

% Simulation loop
for k = 1:num_steps
    % Compute control input using estimated states
    u_kf(k) = -K_lqr * x_hat_kf(:,k);
    
    % Ensure process noise is correctly sampled
    process_noise = chol(Q_kf, 'lower') * randn(n,1); % Ensures correct covariance
    
    % Update true system state with process noise
    x_true_kf(:,k+1) = A_d * x_true_kf(:,k) + B_d * u_kf(k) + process_noise;

    % Compute noisy measurement
    measurement_noise = noise_std .* randn(2,1);
    y_meas_kf(:,k+1) = C_obs * x_true_kf(:,k+1) + measurement_noise;

    % Kalman filter prediction step
    x_hat_pred = A_d * x_hat_kf(:,k) + B_d * u_kf(k);
    P_pred = A_d * P * A_d' + Q_kf;
    
    % Kalman filter update step
    K_gain = P_pred * C_obs' / (C_obs * P_pred * C_obs' + R_kf);
    x_hat_kf(:,k+1) = x_hat_pred + K_gain * (y_meas_kf(:,k+1) - C_obs * x_hat_pred);
    P = (eye(n) - K_gain * C_obs) * P_pred;
end

% Performance evaluation
settling_time_kf = NaN;
for i = length(t_kf):-1:2
    if max(abs(x_true_kf(1:3,i))) > 0.02  % 2% criterion for settling
        settling_time_kf = t_kf(i);
        break;
    end
end

% Display results
fprintf('Kalman Filter Controller Performance:\n');
fprintf('  Settling time (2%% criterion): %.2f seconds\n', settling_time_kf);
fprintf('  Maximum control effort: %.2f N\n', max(abs(u_kf)));
fprintf('  Maximum estimation error: %.4f\n', max(max(abs(x_true_kf - x_hat_kf))));


%% 12. Simulation of LQR with Integral Action (Reference Tracking)
disp('12. Simulating LQR with integral action for reference tracking...');

% Reference signal
r = 0.5;  % Desired cart position

% Time settings
t_final_track = 10;  % Longer simulation to see tracking behavior
num_steps_track = floor(t_final_track / Ts);
t_track = (0:num_steps_track) * Ts;

% Initialize arrays
x_track = zeros(n, length(t_track));
x_track(:,1) = x0;
x_int = 0;  % Integrator state
x_int_history = zeros(1, length(t_track));
x_int_history(1) = x_int;
u_track = zeros(1, length(t_track)-1);
y_track = zeros(size(C_d,1), length(t_track));
y_track(:,1) = C_d * x0;
error_track = zeros(1, length(t_track)-1);

% Simulation loop
for k = 1:num_steps_track
    % Compute tracking error
    error_track(k) = r - C_track * x_track(:,k);
    
    % Update integrator state
    x_int = x_int + error_track(k);
    x_int_history(k+1) = x_int;
    
    % Compute control input
    u_track(k) = -K_x * x_track(:,k) - K_i * x_int;
    
    % Update state
    x_track(:,k+1) = A_d * x_track(:,k) + B_d * u_track(k);
    
    % Compute output
    y_track(:,k+1) = C_d * x_track(:,k+1);
end

% Performance evaluation
settling_time_track = NaN;
steady_state_threshold = 0.02 * abs(r);  % 2% criterion of reference
for i = length(t_track):-1:2
    if abs(y_track(1,i) - r) > steady_state_threshold
        settling_time_track = t_track(i);
        break;
    end
end

steady_state_error = abs(r - y_track(1,end));

fprintf('LQR with Integral Action Performance:\n');
fprintf('  Settling time (2%% criterion): %.2f seconds\n', settling_time_track);
fprintf('  Steady-state error: %.6f m\n', steady_state_error);
fprintf('  Maximum control effort: %.2f N\n', max(abs(u_track)));
fprintf('  Maximum cart displacement: %.2f m\n', max(abs(y_track(1,:))));
fprintf('  Maximum pendulum 1 angle: %.2f degrees\n', max(abs(y_track(2,:)))*180/pi);
fprintf('  Maximum pendulum 2 angle: %.2f degrees\n', max(abs(y_track(3,:)))*180/pi);

%% 13. Robustness Analysis: Parameter Variation
disp('13. Analyzing robustness to parameter variations...');

% Modify system parameters (increase first pendulum mass by 10%)
m1_var = m1 * 1.1;

% Recalculate continuous-time matrices
A_var = [0 0 0 1 0 0;
         0 0 0 0 1 0;
         0 0 0 0 0 1;
         0 (m1_var+m2)*l1*g/(m0+m1_var+m2) m2*l2*g/(m0+m1_var+m2) 0 0 0;
         0 -((m0+m1_var+m2)*g/l1 + m2*l2*g/l1^2) -m2*l2*g/l1 0 0 0;
         0 (m1_var+m2)*l1*g/(m2*l2) -g/l2 0 0 0];

B_var = [0;
         0;
         0;
         1/(m0+m1_var+m2);
         -l1/(m0+m1_var+m2);
         -(m1_var+m2)*l1/(m2*l2)];

% Create and discretize the modified system
sys_c_var = ss(A_var, B, C, D);
sys_d_var = c2d(sys_c_var, Ts, 'zoh');
A_d_var = sys_d_var.A;
B_d_var = sys_d_var.B;

% Simulate the modified system with the original LQR controller
t_var = (0:num_steps) * Ts;
x_var = zeros(n, length(t_var));
x_var(:,1) = x0;
u_var = zeros(1, length(t_var)-1);
y_var = zeros(size(C_d,1), length(t_var));
y_var(:,1) = C_d * x0;

% Simulation loop
for k = 1:num_steps
    % Compute control input using the original LQR gain
    u_var(k) = -K_lqr * x_var(:,k);
    
    % Update state using the modified system matrices
    x_var(:,k+1) = A_d_var * x_var(:,k) + B_d_var * u_var(k);
    
    % Compute output
    y_var(:,k+1) = C_d * x_var(:,k+1);
end

% Performance evaluation
settling_time_var = NaN;
for i = length(t_var):-1:2
    if max(abs(x_var(1:3,i))) > 0.02  % 2% criterion for settling
        settling_time_var = t_var(i);
        break;
    end
end

fprintf('LQR Controller Performance with Parameter Variation:\n');
fprintf('  Settling time (2%% criterion): %.2f seconds\n', settling_time_var);
fprintf('  Maximum control effort: %.2f N\n', max(abs(u_var)));
fprintf('  Maximum cart displacement: %.2f m\n', max(abs(y_var(1,:))));
fprintf('  Maximum pendulum 1 angle: %.2f degrees\n', max(abs(y_var(2,:)))*180/pi);
fprintf('  Maximum pendulum 2 angle: %.2f degrees\n', max(abs(y_var(3,:)))*180/pi);

%% 14. Robustness Analysis: External Disturbance
disp('14. Analyzing robustness to external disturbances...');

% Simulate the system with an impulse disturbance at t = 2 seconds
impulse_time = 20;  % Apply disturbance at 2 seconds (sample 20)
impulse_magnitude = 5;  % Disturbance magnitude

t_dist = (0:num_steps) * Ts;
x_dist = zeros(n, length(t_dist));
x_dist(:,1) = x0;
u_dist = zeros(1, length(t_dist)-1);
y_dist = zeros(size(C_d,1), length(t_dist));
y_dist(:,1) = C_d * x0;
dist = zeros(n, length(t_dist)-1);

% Add an impulse disturbance to the cart's position
if impulse_time < length(t_dist)-1
    dist(1, impulse_time) = impulse_magnitude;
end

% Simulation loop
for k = 1:num_steps
    % Compute control input
    u_dist(k) = -K_lqr * x_dist(:,k);
    
    % Update state with disturbance
    x_dist(:,k+1) = A_d * x_dist(:,k) + B_d * u_dist(k) + dist(:,k);
    
    % Compute output
    y_dist(:,k+1) = C_d * x_dist(:,k+1);
end

% Performance evaluation
recovery_time = NaN;
for i = impulse_time:length(t_dist)
    if max(abs(x_dist(1:3,i))) < 0.02  % System has recovered
        recovery_time = t_dist(i) - t_dist(impulse_time);
        break;
    end
end

fprintf('LQR Controller Disturbance Response:\n');
fprintf('  Recovery time: %.2f seconds\n', recovery_time);
fprintf('  Maximum control effort: %.2f N\n', max(abs(u_dist)));
fprintf('  Maximum cart displacement: %.2f m\n', max(abs(y_dist(1,:))));
fprintf('  Maximum pendulum 1 angle: %.2f degrees\n', max(abs(y_dist(2,:)))*180/pi);
fprintf('  Maximum pendulum 2 angle: %.2f degrees\n', max(abs(y_dist(3,:)))*180/pi);

%% 15. Plot Results: State Feedback Controllers Comparison
disp('15. Plotting controller performance comparisons...');

% Figure 1: Comparing State Feedback Controllers
figure('Name', 'State Feedback Controllers Comparison');

subplot(3,1,1);
plot(t_pp, y_pp(1,:), 'b-', t_lqr, y_lqr(1,:), 'r--', 'LineWidth', 1.5);
title('Cart Position');
xlabel('Time (s)');
ylabel('Position (m)');
legend('Pole Placement', 'LQR');
grid on;

subplot(3,1,2);
plot(t_pp, y_pp(2,:)*180/pi, 'b-', t_lqr, y_lqr(2,:)*180/pi, 'r--', 'LineWidth', 1.5);
title('Pendulum 1 Angle');
xlabel('Time (s)');
ylabel('Angle (degrees)');
legend('Pole Placement', 'LQR');
grid on;

subplot(3,1,3);
plot(t_pp(1:end-1), u_pp, 'b-', t_lqr(1:end-1), u_lqr, 'r--', 'LineWidth', 1.5);
title('Control Input');
xlabel('Time (s)');
ylabel('Force (N)');
legend('Pole Placement', 'LQR');
grid on;

%% 16. Plot Results: Observer-Based Controllers
figure('Name', 'Observer-Based Controllers');

subplot(3,1,1);
plot(t_obs, x_true(1,:), 'b-', t_obs, x_hat(1,:), 'r--', 'LineWidth', 1.5);
title('Cart Position');
xlabel('Time (s)');
ylabel('Position (m)');
legend('True State', 'Estimated State');
grid on;

subplot(3,1,2);
plot(t_obs, x_true(2,:)*180/pi, 'b-', t_obs, x_hat(2,:)*180/pi, 'r--', 'LineWidth', 1.5);
title('Pendulum 1 Angle');
xlabel('Time (s)');
ylabel('Angle (degrees)');
legend('True State', 'Estimated State');
grid on;

subplot(3,1,3);
plot(t_obs(1:end-1), u_obs, 'k-', 'LineWidth', 1.5);
title('Control Input');
xlabel('Time (s)');
ylabel('Force (N)');
grid on;

%% 17. Plot Results: Kalman Filter Performance
figure('Name', 'Kalman Filter Performance');

subplot(3,1,1);
plot(t_kf, x_true_kf(1,:), 'b-', t_kf, x_hat_kf(1,:), 'r--', 'LineWidth', 1.5);
title('Cart Position with Kalman Filter');
xlabel('Time (s)');
ylabel('Position (m)');
legend('True State', 'Kalman Estimate');
grid on;

subplot(3,1,2);
plot(t_kf, x_true_kf(2,:)*180/pi, 'b-', t_kf, x_hat_kf(2,:)*180/pi, 'r--', 'LineWidth', 1.5);
title('Pendulum 1 Angle with Kalman Filter');
xlabel('Time (s)');
ylabel('Angle (degrees)');
legend('True State', 'Kalman Estimate');
grid on;

subplot(3,1,3);
plot(t_kf(1:end-1), u_kf, 'k-', 'LineWidth', 1.5);
title('Control Input with Kalman Filter');
xlabel('Time (s)');
ylabel('Force (N)');
grid on;

%% 18. Plot Results: Reference Tracking
figure('Name', 'Reference Tracking Performance');

subplot(3,1,1);
plot(t_track, y_track(1,:), 'b-', 'LineWidth', 1.5);
hold on;
plot(t_track, r*ones(size(t_track)), 'r--', 'LineWidth', 1.5);
title('Cart Position Tracking');
xlabel('Time (s)');
ylabel('Position (m)');
legend('Actual Position', 'Reference');
grid on;

subplot(3,1,2);
plot(t_track, y_track(2,:)*180/pi, 'b-', 'LineWidth', 1.5);
title('Pendulum 1 Angle During Tracking');
xlabel('Time (s)');
ylabel('Angle (degrees)');
grid on;

subplot(3,1,3);
plot(t_track(1:end-1), u_track, 'k-', 'LineWidth', 1.5);
title('Control Input for Tracking');
xlabel('Time (s)');
ylabel('Force (N)');
grid on;

%% 19. Display Performance Summary
disp('Performance Summary:');
disp('====================');
fprintf('Pole Placement Controller:\n');
fprintf('  - Settling time: %.2f s\n', settling_time_pp);
fprintf('  - Max control effort: %.2f N\n', max(abs(u_pp)));
fprintf('\nLQR Controller:\n');
fprintf('  - Settling time: %.2f s\n', settling_time_lqr);
fprintf('  - Max control effort: %.2f N\n', max(abs(u_lqr)));
fprintf('\nObserver-Based Controller:\n');
fprintf('  - Settling time: %.2f s\n', settling_time_obs);
fprintf('  - Max estimation error: %.4f\n', max(max(abs(x_true - x_hat))));
fprintf('\nKalman Filter Controller:\n');
fprintf('  - Settling time: %.2f s\n', settling_time_kf);
fprintf('  - Max estimation error: %.4f\n', max(max(abs(x_true_kf - x_hat_kf))));
fprintf('\nReference Tracking Performance:\n');
fprintf('  - Settling time: %.2f s\n', settling_time_track);
fprintf('  - Steady-state error: %.6f m\n', steady_state_error);