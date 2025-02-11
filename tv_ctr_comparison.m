%% Numerical Example: Observability starts -- k=nr-1 (toooo large for matlab to process) --> we use k = 3

clc; clear; close all;

% Define varying tensor dimensions
p_values = [2];  % Rows
n_values = [2];  % Columns
%r_values = [2,3,4,5,6,7,8];  % Depth
r_values = [2,3,4,5,6,7,8, 9, 10, 11, 12, 13, 14, 15,16, 17, 18];
num_trials = 5;
%k = 3;  % Number of recursion steps
%%
numerical_values = [rand(1), rand(1),rand(1), rand(1), rand(1), rand(1), ...
                    rand(1), rand(1),rand(1), rand(1), rand(1), rand(1), ...
                    rand(1), rand(1), rand(1), rand(1), rand(1), rand(1), ...
                    rand(1), rand(1), rand(1), rand(1), rand(1), rand(1), 0.05,...
                    rand(1), rand(1), rand(1), rand(1)]./10; % Assign numbers as needed
%%
% Store computation times
time_bcirc = zeros(length(r_values), num_trials);
time_fourier = zeros(length(r_values), num_trials);

for trials = 1:num_trials
    for ni = 1:1
        for ri = 1:length(r_values)
        %for ri = 1:2
            p = p_values(1);
            n = n_values(1);
            r = r_values(ri);
            %k = n*r - 1;
            k = 3;
            
            syms t;
            C = sym('C', [p, n, r]);
            A = sym('A', [n, n, r]);
            A_t = arrayfun(@(x) x * t^2, A, 'UniformOutput', false);
            T = sym('T', [p, n, r]); % Symbolic tensor p*n*r N
            C_t = arrayfun(@(x) x * t^2, C, 'UniformOutput', false);
            %{
            syms A1_1_1 A1_1_2 A1_1_3 A1_2_1 A1_2_2 A1_2_3
            syms A2_1_1 A2_1_2 A2_1_3 A2_2_1 A2_2_2 A2_2_3
            syms C1_1_1 C1_1_2 C1_1_3 C1_2_1 C1_2_2 C1_2_3
            syms C2_1_1 C2_1_2 C2_1_3 C2_2_1 C2_2_2 C2_2_3 t
            syms C1_1_4 C1_2_4 C2_1_4 C2_2_4
            values = [A1_1_1, A1_1_2, A1_1_3, A1_2_1, A1_2_2, A1_2_3, A2_1_1, A2_1_2, A2_1_3, A2_2_1, A2_2_2, A2_2_3, C1_1_1, C1_1_2, C1_1_3, C1_2_1, C1_2_2, C1_2_3, C2_1_1, C2_1_2, C2_1_3, C2_2_1, C2_2_2, C2_2_3, ...
                 C1_1_4, C1_2_4, C2_1_4, C2_2_4, t];
            %}
            
            % 1.2. Define the Fourier matrix F_n
            omega = exp(-2 * pi * i / r); % Primitive n-th root of unity
            F_r = (zeros(r, r));
            for i = 1:r
                for j = 1:r
                    F_r(i, j) = omega^((i-1)*(j-1)) / sqrt(r); % Normalize by sqrt(n)
                end
            end
            F_r = dftmtx(r) / sqrt(r);
            I_p = eye(p);
            I_n = eye(n);

            % ----- Method 1: bcirc Approach -----
            bcirc_T_t = sym([]); % Start with an empty matrix
            bcirc_A_t = sym([]);
            T_t = C_t; %N0(t) = C(t)
            for j = 1:r
                row = [];
                rowA = [];
                for l = 1:r
                    % Shift the tensor slices cyclically
                    idx = mod(j - l, r) + 1;
                    row = [row, cell2sym(T_t(:, :, idx))]; % Convert back to symbolic matrix
                    idxA = mod(j - l, r) + 1;
                    rowA = [rowA, cell2sym(A_t(:, :, idxA))]; % Convert back to symbolic matrix
                end
                bcirc_T_t = [bcirc_T_t; row]; % bcirc(N0(t))
                bcirc_A_t = [bcirc_A_t; row];
            end
            
            if r <= 15
                tic;
                % Derivative of bcirc_T(t) with respect to t
                %d_bcirc_T_t_dt = diff(bcirc_T_t, t); %d(bcirc(N0(t))/dt)
                N_prev = bcirc_T_t; %N0(t)
                bcirc_stack = sym([]); % prepare for the large concatenated matrix bcirc(N_0(t))...
                for l = 1:k
                    bcirc_N_prev = N_prev;
                    bcirc_A = bcirc_A_t;
                    bcirc_N = bcirc_N_prev * bcirc_A + diff(bcirc_N_prev, t); % bcirc(N_{l+1}(t))
                    bcirc_stack = [bcirc_stack; bcirc_N];
                    N_prev = bcirc_N;
                end

                %D = subs(bcirc_stack, values, numerical_values);
                %rank_bcirc = rank(double(D));
                rank_bcirc = rank(bcirc_stack);
                time_bcirc(ri, trials) = toc;
            end
            
            % ----- Method 2: Fourier Block Approach -----
            F_n_kron_I_n = kron(F_r, I_n); %pnr nnr
            F_n_kron_I_n_conj = kron(inv(F_r), I_n); 
            A_blk = F_n_kron_I_n * bcirc_A_t * F_n_kron_I_n_conj;
            
            tic;
            % bcirc_T_t;
            F_n_kron_I_s = kron(F_r, I_p);
            F_n_kron_I_s_conj = kron(inv(F_r), I_n);
            T_blk = F_n_kron_I_s * bcirc_T_t * F_n_kron_I_s_conj; % F(bcirc(N_0(t)))
            d_T_blk_dt = sym(zeros(size(T_blk)));

            % Loop over each diagonal block
            %N_prev_blocks = T_diag_block; % Tensor representation
            rank_fourier = 0;
            for l = 1:k
                for j = 1:r
                    T_diag_block = T_blk((j-1)*p+1:j*p, (j-1)*n+1:j*n); % extract small blocks mathbf{N}_{l,j}(t)
                    d_T_blk_dt((j-1)*p+1:j*p, (j-1)*n+1:j*n) = diff(T_diag_block, t);  % Derivative of T_t(:, :, i) with respect to t
                    A_j = A_blk((j-1)*n+1:j*n, (j-1)*n+1:j*n); % A: nnr
                    N_new_j = T_diag_block * A_j + diff(T_diag_block, t);
                    %rank_fourier = rank_fourier + rank(T_diag_block);
                    %E = subs(T_diag_block, values, numerical_values);
                    %rank(double(E)); % print the rank for that block
                    rank(T_diag_block);
                    T_blk((i-1)*p+1:i*p, (i-1)*n+1:i*n) = N_new_j; % mathbf{N}_{l+1,j}(t)
                end
            end
            %rank_fourier = rank_fourier + rank(N_new_j);
            time_fourier(ri, trials) = toc;
            
            disp("p, n, r for this iteration:");
            disp([p,n,r]);
            disp([time_bcirc, time_fourier]);
        end
    end
end

%%
% ---- visualization ----
avg_time_bcirc = mean(time_bcirc(1:14,:), 2); % Average for each ri
avg_time_fourier = mean(time_fourier, 2);

r_values_bcirc = r_values(1:length(avg_time_bcirc));      % For bcirc
r_values_fourier = r_values(1:length(avg_time_fourier));  % For fourier

figure('Position', [100, 100, 400, 200]); 
plot(r_values_bcirc, (avg_time_bcirc), '-o', 'color','b','LineWidth', 1.7, 'DisplayName', 'Unfolding-based method');
hold on;
plot(r_values_fourier, (avg_time_fourier), '-x', 'color','r','LineWidth', 1.7, 'DisplayName', 'T-product-based method');
hold off;

xlim([2, 18]);
xticks([2, 6, 10, 14, 18]); 

xlabel('Dimension r', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Time (s)', 'FontSize', 11, 'FontWeight', 'bold');
title('Comparison for Observability Condition', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'northwest', 'FontSize', 10, 'FontWeight', 'bold', 'Box', 'off');
