%% ========================================================================
% 脚本名称: test_run_c09_prototype.m
% 功能:
%   1. 【Task 3 -> Task 2 跨任务跨表征神经解码原型验证】
%   2. 训练集: Task 3 纯物理红绿色块 (120 试次: 60 Red, 60 Green)
%   3. 测试集: Task 2 纯灰度水果记忆色 (240 试次: 120 Red Memory, 120 Green Memory)
%   4. 特征标准化: 严格基于 Task 3 计算均值与方差，零泄露投影 Task 2
%   5. 两个时间维度:
%      - 1D 对角线同步解码: 执行 200 次置换检验与 1D 时间簇显著性校正
%      - 2D TGM 时间泛化矩阵: 仅计算真实准确率热力图 (免置换检验，极速出图)
%   6. 规范学术绘图与单通道快速验证 (测试目标: sub001 - G13)
% ========================================================================

clear; clc; close all;

%% 1. 主参数配置 (置顶平铺，直观可控)
cfg = struct();
cfg.sub_id         = 'sub001';                       % 测试被试
cfg.elec_name      = 'G13';                          % 测试目标通道 (典型颜色电极)
cfg.test_state     = 'gray';                         % 测试集条件: 纯灰度水果 (无物理颜色)
cfg.split_mode     = 'full';                         % 'full': 100%全量外推

% 时间窗与时程参数
cfg.win_len        = 20;                             % 滑动窗长 20 ms
cfg.win_step       = 20;                             % 滑动步长 20 ms
cfg.t_range        = [-200, 800];                    % 时程范围 ms (共 51 个窗口)

% 平滑与统计参数
cfg.smooth_pts     = 5;                              % 平滑点数
cfg.smooth_typ     = 'gaussian';                     % 平滑方式: gaussian
cfg.n_perm         = 200;                            % 仅对角线进行 200 次置换检验 (TGM免置换)
cfg.svm_lambda     = 0.01;                           % 岭正则化参数 Lambda

% 频段定义与规范学术显示标签
cfg.bands          = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'};
cfg.bands_disp     = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low-Gamma', 'High-Gamma'};
cfg.n_bands        = numel(cfg.bands);

% 路径设置
script_dir = fileparts(mfilename('fullpath'));
proj_root  = fileparts(fileparts(script_dir));
data_root  = fullfile(proj_root, 'process_data_new');
res_root   = fullfile(proj_root, 'result');

fig_dir    = fullfile(res_root, 'figures', 'cross_decoding_prototype');
tab_dir    = fullfile(res_root, 'tables', 'cross_decoding_prototype');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
if ~exist(tab_dir, 'dir'), mkdir(tab_dir); end

fprintf('========================================================================\n');
fprintf('  【C09 原型测试: Task 3 (物理色块) -> Task 2 (灰度记忆色) 跨任务解码】\n');
fprintf('========================================================================\n');
fprintf('被试: %s | 通道: %s | 时间范围: [%d, %d] ms | 窗口: %d ms | 步长: %d ms\n', ...
    cfg.sub_id, cfg.elec_name, cfg.t_range(1), cfg.t_range(2), cfg.win_len, cfg.win_step);
fprintf('分折模式: %s (全量完全外推) | 对角线置换次数: %d | TGM: 真实经验热力图\n\n', ...
    cfg.split_mode, cfg.n_perm);

%% 2. 加载 Task 3 (训练集) 与 Task 2 (测试集) 数据
t_start = tic;

t3_file = fullfile(data_root, cfg.sub_id, 'task3_multiband_epoched.mat');
t2_file = fullfile(data_root, cfg.sub_id, 'task2_multiband_epoched.mat');

if ~isfile(t3_file) || ~isfile(t2_file)
    error('未找到被试 %s 的多频段数据文件！', cfg.sub_id);
end

fprintf('[1/5] 载入被试数据 ...\n');
d3 = load(t3_file, 'epoched_data'); ep3 = d3.epoched_data;
d2 = load(t2_file, 'epoched_data'); ep2 = d2.epoched_data;

time_ms = ep3.time_ms(:)';
t_centers = cfg.t_range(1) : cfg.win_step : cfg.t_range(2);
n_win = numel(t_centers);

% 查找目标通道索引
ch_idx3 = find(strcmp(ep3.channels, cfg.elec_name), 1);
ch_idx2 = find(strcmp(ep2.channels, cfg.elec_name), 1);
if isempty(ch_idx3) || isempty(ch_idx2)
    error('目标通道 %s 不在数据通道列表中！', cfg.elec_name);
end

% -------------------------------------------------------------------------
% 构建 Task 3 训练样本 (物理红绿色块)
% -------------------------------------------------------------------------
ti3 = ep3.trial_info;
tr_mask = strcmp(ti3.color, 'red') | strcmp(ti3.color, 'green');
ti3_use = ti3(tr_mask, :);
n_tr = height(ti3_use);

y_tr = zeros(n_tr, 1);
y_tr(strcmp(ti3_use.color, 'red')) = 1; % Red=1, Green=0

% -------------------------------------------------------------------------
% 构建 Task 2 测试样本 (灰度记忆水果)
% -------------------------------------------------------------------------
ti2 = ep2.trial_info;
te_mask = strcmp(ti2.state, cfg.test_state);
ti2_use = ti2(te_mask, :);
n_te = height(ti2_use);

y_te = zeros(n_te, 1);
y_te(strcmp(ti2_use.memory_color, 'red')) = 1; % Red Memory=1, Green Memory=0

fprintf('    [+] 训练集 Task 3 试次数: %d (Red: %d, Green: %d)\n', ...
    n_tr, sum(y_tr == 1), sum(y_tr == 0));
fprintf('    [+] 测试集 Task 2 灰度试次数: %d (Red Memory: %d, Green Memory: %d)\n', ...
    n_te, sum(y_te == 1), sum(y_te == 0));

%% 3. 提取滑动时间窗多频段特征张量 [N x n_bands x n_win]
fprintf('[2/5] 提取滑动时间窗特征 ...\n');

X3_3d = zeros(n_tr, cfg.n_bands, n_win, 'single');
X2_3d = zeros(n_te, cfg.n_bands, n_win, 'single');

for b = 1:cfg.n_bands
    b_name = cfg.bands{b};
    raw3 = squeeze(ep3.(b_name)(tr_mask, ch_idx3, :)); % [n_tr x 750]
    raw2 = squeeze(ep2.(b_name)(te_mask, ch_idx2, :)); % [n_te x 750]
    
    for w = 1:n_win
        tc = t_centers(w);
        t_m = (time_ms >= (tc - cfg.win_len / 2)) & (time_ms < (tc + cfg.win_len / 2));
        X3_3d(:, b, w) = mean(raw3(:, t_m), 2);
        X2_3d(:, b, w) = mean(raw2(:, t_m), 2);
    end
end

%% 4. 执行跨任务解码 (对角线同步 + 2D TGM 时间泛化)
fprintf('[3/5] 执行真实跨任务解码 (Multi-Band 与 6 个单频段) ...\n');

% 4.1 对角线同步解码 (Diagonal: w3 = w2 = w)
real_diag_joint = zeros(1, n_win);
real_diag_bands = zeros(cfg.n_bands, n_win);

for w = 1:n_win
    % Multi-Band 联合特征 [N x 6]
    X_tr_w = double(squeeze(X3_3d(:, :, w)));
    X_te_w = double(squeeze(X2_3d(:, :, w)));
    
    % 严格基于 Task 3 的均值与标准差进行 Out-of-Sample 标准化
    mu_tr  = mean(X_tr_w, 1);
    sig_tr = std(X_tr_w, 0, 1);
    sig_tr(sig_tr < 1e-6) = 1;
    
    X_tr_norm = (X_tr_w - mu_tr) ./ sig_tr;
    X_te_norm = (X_te_w - mu_tr) ./ sig_tr;
    
    % 训练 SVM 并测试
    mdl = fitclinear(X_tr_norm, y_tr, 'Learner', 'svm', 'Regularization', 'ridge', 'Lambda', cfg.svm_lambda);
    y_pred = predict(mdl, X_te_norm);
    
    sens = sum(y_te == 1 & y_pred == 1) / max(1, sum(y_te == 1));
    spec = sum(y_te == 0 & y_pred == 0) / max(1, sum(y_te == 0));
    real_diag_joint(w) = (sens + spec) / 2;
    
    % 6 个单频段独立特征
    for b = 1:cfg.n_bands
        X_tr_b = double(X3_3d(:, b, w));
        X_te_b = double(X2_3d(:, b, w));
        
        mu_b  = mean(X_tr_b);
        sig_b = std(X_tr_b);
        if sig_b < 1e-6, sig_b = 1; end
        
        X_tr_bn = (X_tr_b - mu_b) ./ sig_b;
        X_te_bn = (X_te_b - mu_b) ./ sig_b;
        
        mdl_b = fitclinear(X_tr_bn, y_tr, 'Learner', 'svm', 'Regularization', 'ridge', 'Lambda', cfg.svm_lambda);
        y_pred_b = predict(mdl_b, X_te_bn);
        
        sens_b = sum(y_te == 1 & y_pred_b == 1) / max(1, sum(y_te == 1));
        spec_b = sum(y_te == 0 & y_pred_b == 0) / max(1, sum(y_te == 0));
        real_diag_bands(b, w) = (sens_b + spec_b) / 2;
    end
end

% 4.2 计算 2D TGM 时间泛化矩阵 (w3 为 Task 3 训练时间, w2 为 Task 2 测试时间)
fprintf('    [+] 正在计算 51x51 TGM 时间泛化矩阵 (Multi-Band) ...\n');
tgm_matrix = zeros(n_win, n_win); % [w3 x w2]

% 预先标准化所有时间窗的 Task 3 与 Task 2
norm_X3 = cell(n_win, 1);
norm_X2_for_w3 = cell(n_win, 1); % 按 w3 的标准化参数缩放 Task 2
for w3 = 1:n_win
    X_tr_w3 = double(squeeze(X3_3d(:, :, w3)));
    mu_w3  = mean(X_tr_w3, 1);
    sig_w3 = std(X_tr_w3, 0, 1);
    sig_w3(sig_w3 < 1e-6) = 1;
    
    norm_X3{w3} = (X_tr_w3 - mu_w3) ./ sig_w3;
    
    % 对每个测试时间窗 w2 缩放
    X2_scaled = zeros(n_te, cfg.n_bands, n_win);
    for w2 = 1:n_win
        X_te_w2 = double(squeeze(X2_3d(:, :, w2)));
        X2_scaled(:, :, w2) = (X_te_w2 - mu_w3) ./ sig_w3;
    end
    norm_X2_for_w3{w3} = X2_scaled;
end

% 遍历训练与测试时间窗
for w3 = 1:n_win
    mdl_w3 = fitclinear(norm_X3{w3}, y_tr, 'Learner', 'svm', 'Regularization', 'ridge', 'Lambda', cfg.svm_lambda);
    X2_all_w2 = norm_X2_for_w3{w3};
    for w2 = 1:n_win
        y_pred = predict(mdl_w3, squeeze(X2_all_w2(:, :, w2)));
        sens = sum(y_te == 1 & y_pred == 1) / max(1, sum(y_te == 1));
        spec = sum(y_te == 0 & y_pred == 0) / max(1, sum(y_te == 0));
        tgm_matrix(w3, w2) = (sens + spec) / 2;
    end
end

%% 5. 对角线非参数标签置换检验 (仅 1D 曲线跑置换，TGM 免置换)
fprintf('[4/5] 执行对角线 200 次标签置换检验与时间簇检验 ...\n');
null_diag_dist = zeros(cfg.n_perm, n_win);
svm_lam = cfg.svm_lambda;

% 预提对角线标准化后的特征矩阵，加速置换循环
X3_diag_norm = cell(n_win, 1);
X2_diag_norm = cell(n_win, 1);
for w = 1:n_win
    X_tr_w = double(squeeze(X3_3d(:, :, w)));
    X_te_w = double(squeeze(X2_3d(:, :, w)));
    mu_w  = mean(X_tr_w, 1);
    sig_w = std(X_tr_w, 0, 1);
    sig_w(sig_w < 1e-6) = 1;
    X3_diag_norm{w} = (X_tr_w - mu_w) ./ sig_w;
    X2_diag_norm{w} = (X_te_w - mu_w) ./ sig_w;
end

% 置换循环: 打乱 Task 2 测试标签
rng(42);
for perm_i = 1:cfg.n_perm
    y_te_perm = y_te(randperm(n_te));
    perm_curve = zeros(1, n_win);
    for w = 1:n_win
        mdl_p = fitclinear(X3_diag_norm{w}, y_tr, 'Learner', 'svm', 'Regularization', 'ridge', 'Lambda', svm_lam);
        y_pred = predict(mdl_p, X2_diag_norm{w});
        sens = sum(y_te_perm == 1 & y_pred == 1) / max(1, sum(y_te_perm == 1));
        spec = sum(y_te_perm == 0 & y_pred == 0) / max(1, sum(y_te_perm == 0));
        perm_curve(w) = (sens + spec) / 2;
    end
    null_diag_dist(perm_i, :) = perm_curve;
end

% 平滑处理
diag_joint_s = smoothdata(real_diag_joint, cfg.smooth_typ, cfg.smooth_pts);
diag_bands_s = zeros(size(real_diag_bands));
for b = 1:cfg.n_bands
    diag_bands_s(b, :) = smoothdata(real_diag_bands(b, :), cfg.smooth_typ, cfg.smooth_pts);
end
null_diag_s  = smoothdata(null_diag_dist, 2, cfg.smooth_typ, cfg.smooth_pts);
tgm_matrix_s = imgaussfilt(tgm_matrix, 0.8); % 适度 2D 高斯平滑增强视觉可读性

% 计算逐点 p 值与时间簇质量检验 (Cluster-Mass)
p_pointwise = (1 + sum(null_diag_s >= diag_joint_s, 1)) / (1 + cfg.n_perm);
alpha_pt = 0.05;
sig_mask = (p_pointwise < alpha_pt) & (t_centers >= 0);

% 提取连续显著簇
clusters = [];
in_c = false;
c_start = 1;
for w = 1:n_win
    if sig_mask(w) && ~in_c
        in_c = true; c_start = w;
    elseif ~sig_mask(w) && in_c
        in_c = false; clusters = [clusters; c_start, w-1]; %#ok<AGROW>
    end
end
if in_c, clusters = [clusters; c_start, n_win]; end

% 计算簇质量与 FWE 校正
sig_clusters = [];
if ~isempty(clusters)
    n_cl = size(clusters, 1);
    cl_mass = zeros(n_cl, 1);
    for ci = 1:n_cl
        c_range = clusters(ci, 1) : clusters(ci, 2);
        cl_mass(ci) = sum(diag_joint_s(c_range) - 0.5);
    end
    
    % 零分布最大簇质量
    max_null_mass = zeros(cfg.n_perm, 1);
    for pi = 1:cfg.n_perm
        null_c_mask = (null_diag_s(pi, :) >= prctile(null_diag_s, 95, 1)) & (t_centers >= 0);
        null_masses = [0];
        in_nc = false; nc_start = 1;
        for w = 1:n_win
            if null_c_mask(w) && ~in_nc
                in_nc = true; nc_start = w;
            elseif ~null_c_mask(w) && in_nc
                in_nc = false;
                null_masses(end+1) = sum(null_diag_s(pi, nc_start:w-1) - 0.5); %#ok<AGROW>
            end
        end
        if in_nc, null_masses(end+1) = sum(null_diag_s(pi, nc_start:n_win) - 0.5); end %#ok<AGROW>
        max_null_mass(pi) = max(null_masses);
    end
    
    for ci = 1:n_cl
        p_cl = (1 + sum(max_null_mass >= cl_mass(ci))) / (1 + cfg.n_perm);
        if p_cl < 0.05
            sig_clusters = [sig_clusters; clusters(ci, :)]; %#ok<AGROW>
        end
    end
end

fprintf('    [+] 统计检验完成! 发现 %d 个逐点显著窗口，%d 个经簇校正显著时间段。\n', ...
    sum(sig_mask), size(sig_clusters, 1));

%% 6. 学术级规范可视化绘图 (1:1 左右双子图)
fprintf('[5/5] 绘制学术级跨任务解码双子图 ...\n');

fig = figure('Position', [100, 100, 1280, 520], 'Color', 'w', 'Visible', 'off');

% 配色定义 (遵循 Nature 标准)
col_joint = [0.85, 0.37, 0.01]; % 陶土暖橙 (Multi-Band)
col_null  = [0.88, 0.88, 0.88]; % 浅灰零分布阴影
col_clust = [1.00, 0.92, 0.70]; % 金黄显著簇背景
band_cols = [
    0.40, 0.40, 0.40;  % Delta: 灰
    0.95, 0.60, 0.20;  % Theta: 杏黄
    0.20, 0.45, 0.75;  % Alpha: 钴蓝
    0.10, 0.65, 0.45;  % Beta: 青绿
    0.50, 0.35, 0.75;  % Low_Gamma: 紫灰
    0.90, 0.15, 0.50   % High_Gamma: 玫红
];

% -------------------------------------------------------------------------
% 左子图: 1D 对角线解码时程
% -------------------------------------------------------------------------
subplot(1, 2, 1);
hold on;

% 绘制显著时间簇底色
if ~isempty(sig_clusters)
    for sci = 1:size(sig_clusters, 1)
        c_x1 = t_centers(sig_clusters(sci, 1));
        c_x2 = t_centers(sig_clusters(sci, 2));
        fill([c_x1, c_x2, c_x2, c_x1], [0.35, 0.35, 0.85, 0.85], col_clust, ...
            'EdgeColor', 'none', 'FaceAlpha', 0.6, 'HandleVisibility', 'off');
    end
end

% 绘制置换检验 95% 经验零分布置信区间
null_hi = prctile(null_diag_s, 97.5, 1);
null_lo = prctile(null_diag_s, 2.5, 1);
fill([t_centers, fliplr(t_centers)], [null_hi, fliplr(null_lo)], col_null, ...
    'EdgeColor', 'none', 'FaceAlpha', 0.6, 'DisplayName', 'Null 95% CI');

% 绘制 0.5 机会水平虚线与 0 ms 刺激线
yline(0.5, '--', 'Color', [0.55, 0.55, 0.55], 'LineWidth', 1.0, 'HandleVisibility', 'off');
xline(0, ':', 'Color', [0.4, 0.4, 0.4], 'LineWidth', 1.0, 'HandleVisibility', 'off');

% 绘制 6 个单频段细线
for b = 1:cfg.n_bands
    plot(t_centers, diag_bands_s(b, :), 'Color', [band_cols(b, :), 0.55], ...
        'LineWidth', 1.2, 'DisplayName', cfg.bands_disp{b});
end

% 绘制 Multi-Band 加粗陶土橙折线
plot(t_centers, diag_joint_s, 'Color', col_joint, 'LineWidth', 2.6, ...
    'DisplayName', 'Multi-Band');

% 标注逐点显著标记点
if any(sig_mask)
    plot(t_centers(sig_mask), repmat(0.38, 1, sum(sig_mask)), 's', ...
        'MarkerFaceColor', col_joint, 'MarkerEdgeColor', 'none', 'MarkerSize', 4, ...
        'HandleVisibility', 'off');
end

xlim([-200, 800]);
ylim([0.35, 0.80]);
xlabel('Time (ms)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Balanced Accuracy', 'FontSize', 11, 'FontWeight', 'bold');
title(sprintf('%s - %s  (Diagonal Sync)', cfg.sub_id, cfg.elec_name), ...
    'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 8, 'Box', 'off');
grid off; box off;
set(gca, 'TickDir', 'out', 'LineWidth', 1.0, 'FontSize', 10);

% -------------------------------------------------------------------------
% 右子图: 2D TGM 时间泛化矩阵热力图
% -------------------------------------------------------------------------
subplot(1, 2, 2);
imagesc(t_centers, t_centers, tgm_matrix_s);
set(gca, 'YDir', 'normal'); % 保持 Y 轴向上递增
colormap(gca, 'parula');
caxis([0.40, 0.70]); % 聚焦有效准确率对比区间
cb = colorbar;
ylabel(cb, 'Cross-Task Accuracy', 'FontSize', 10, 'FontWeight', 'bold');

hold on;
% 绘制对角线 (同步参考线)
plot([-200, 800], [-200, 800], 'w--', 'LineWidth', 1.2);
% 绘制 0 ms 刺激呈现刻度线
xline(0, 'w:', 'LineWidth', 1.0);
yline(0, 'w:', 'LineWidth', 1.0);

xlim([-200, 800]);
ylim([-200, 800]);
xlabel('Task 2 (Memory Fruit) Time [ms]', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Task 3 (Physical Patch) Time [ms]', 'FontSize', 11, 'FontWeight', 'bold');
title(sprintf('%s - %s  (Temporal Generalization)', cfg.sub_id, cfg.elec_name), ...
    'FontSize', 13, 'FontWeight', 'bold');
grid off; box off;
set(gca, 'TickDir', 'out', 'LineWidth', 1.0, 'FontSize', 10);

% 保存图表与结果
fig_png = fullfile(fig_dir, sprintf('%s_%s_cross_decoding.png', cfg.sub_id, cfg.elec_name));
exportgraphics(fig, fig_png, 'Resolution', 300);
close(fig);

out_mat = fullfile(tab_dir, sprintf('%s_%s_cross_decoding_results.mat', cfg.sub_id, cfg.elec_name));
save(out_mat, 'real_diag_joint', 'diag_joint_s', 'real_diag_bands', 'diag_bands_s', ...
    'null_diag_dist', 'tgm_matrix', 'tgm_matrix_s', 'p_pointwise', 'sig_clusters', ...
    't_centers', 'cfg');

t_total = toc(t_start);
fprintf('\n========================================================================\n');
fprintf('  【原型测试顺利完成! 总耗时: %.2f 秒】\n', t_total);
fprintf('  - 学术大图导出至: %s\n', fig_png);
fprintf('  - 数据结果导出至: %s\n', out_mat);
fprintf('  - 对角线最高准确率 (平滑): %.2f%% (发生在 %d ms)\n', ...
    max(diag_joint_s)*100, t_centers(diag_joint_s == max(diag_joint_s)));
[r_max, c_max] = find(tgm_matrix_s == max(tgm_matrix_s(:)), 1);
fprintf('  - TGM 全局最高准确率 (平滑): %.2f%% (Task3=%d ms, Task2=%d ms)\n', ...
    max(tgm_matrix_s(:))*100, t_centers(r_max), t_centers(c_max));
fprintf('========================================================================\n');
