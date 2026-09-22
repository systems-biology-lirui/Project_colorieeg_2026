% test_plot_within_shape_single.m
% 测试单通道的同形状内解码绘图与时程导出逻辑

clear; clc; close all;

sub_id = 'sub001';
ch_name = 'D15';

script_dir = fileparts(mfilename('fullpath'));
proj_root  = fileparts(fileparts(script_dir));
data_root  = fullfile(proj_root, 'process_data_new');
res_root   = fullfile(proj_root, 'result');

fig_dir = fullfile(res_root, 'figures', 'decoding_task3_within_shape');
tc_dir  = fullfile(res_root, 'tables',  'decoding_task3_within_shape_timecourses');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
if ~exist(tc_dir, 'dir'),  mkdir(tc_dir);  end

mat_file = fullfile(data_root, sub_id, 'task3_multiband_epoched.mat');
d = load(mat_file, 'epoched_data');
ep = d.epoched_data;
ti = ep.trial_info;
rg_mask = strcmp(ti.color, 'red') | strcmp(ti.color, 'green');
ti_rg = ti(rg_mask, :);
ch_idx = find(strcmp(ep.channels, ch_name), 1);

bands = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'};
n_bands = numel(bands);
t_centers = -200:20:800; n_win = numel(t_centers);

X_3d = zeros(height(ti_rg), n_bands, n_win, 'single');
for b = 1:n_bands
    raw_b = squeeze(ep.(bands{b})(rg_mask, ch_idx, :));
    for w = 1:n_win
        tm = (ep.time_ms >= (t_centers(w) - 10)) & (ep.time_ms < (t_centers(w) + 10));
        X_3d(:, b, w) = mean(raw_b(:, tm), 2);
    end
end

shapes = unique(ti_rg.pic_id);
n_shapes = numel(shapes); n_folds = 5; lambda = 0.1;

shape_data = cell(n_shapes, 1);
for s = 1:n_shapes
    s_idx = find(ti_rg.pic_id == shapes(s));
    y_s = strcmp(ti_rg.color(s_idx), 'red');
    rng(42 + s);
    cv_r = cvpartition(sum(y_s==1), 'KFold', n_folds);
    cv_g = cvpartition(sum(y_s==0), 'KFold', n_folds);
    r_sub = find(y_s==1); g_sub = find(y_s==0);
    
    tr_m = false(numel(s_idx), n_folds);
    te_m = false(numel(s_idx), n_folds);
    for f = 1:n_folds
        te = [r_sub(test(cv_r, f)); g_sub(test(cv_g, f))];
        te_m(te, f) = true;
        tr_m(:, f) = ~te_m(:, f);
    end
    shape_data{s} = struct('s_idx', s_idx, 'y', double(y_s), 'tr_m', tr_m, 'te_m', te_m);
end

% 预计算投影矩阵 P: [n_shapes, n_win, n_folds]
P_cell = cell(n_shapes, n_win, n_folds);
Xte_cell = cell(n_shapes, n_win, n_folds);
for s = 1:n_shapes
    st = shape_data{s};
    X_s = double(X_3d(st.s_idx, :, :));
    for w = 1:n_win
        X_w = squeeze(X_s(:, :, w));
        for f = 1:n_folds
            mu = mean(X_w(st.tr_m(:, f), :), 1);
            sig = std(X_w(st.tr_m(:, f), :), 0, 1); sig(sig < 1e-6) = 1;
            Xtr = [(X_w(st.tr_m(:, f), :) - mu) ./ sig, ones(sum(st.tr_m(:, f)), 1)];
            Xte = [(X_w(st.te_m(:, f), :) - mu) ./ sig, ones(sum(st.te_m(:, f)), 1)];
            I_reg = eye(size(Xtr, 2)); I_reg(end, end) = 0;
            P_cell{s, w, f} = (Xtr' * Xtr + lambda * I_reg) \ Xtr';
            Xte_cell{s, w, f} = Xte;
        end
    end
end

% 真实解码
acc_shapes = zeros(n_shapes, n_win);
for s = 1:n_shapes
    st = shape_data{s};
    y_pm1 = 2 * st.y - 1;
    for w = 1:n_win
        f_acc = zeros(1, n_folds);
        for f = 1:n_folds
            w_vec = P_cell{s, w, f} * y_pm1(st.tr_m(:, f));
            pred = Xte_cell{s, w, f} * w_vec;
            y_te = y_pm1(st.te_m(:, f));
            sens = sum(y_te == 1 & pred > 0) / sum(y_te == 1);
            spec = sum(y_te == -1 & pred <= 0) / sum(y_te == -1);
            f_acc(f) = (sens + spec) / 2;
        end
        acc_shapes(s, w) = mean(f_acc);
    end
end
acc_avg = mean(acc_shapes, 1);

% 平滑
smooth_pts = 5; smooth_typ = 'gaussian';
acc_avg_s = smoothdata(acc_avg, smooth_typ, smooth_pts);
acc_shapes_s = zeros(size(acc_shapes));
for s = 1:n_shapes
    acc_shapes_s(s, :) = smoothdata(acc_shapes(s, :), smooth_typ, smooth_pts);
end

% 200 次置换检验
n_perm = 200;
null_dist = zeros(n_perm, n_win);
for p = 1:n_perm
    p_shapes = zeros(n_shapes, n_win);
    for s = 1:n_shapes
        st = shape_data{s};
        y_perm = 2 * st.y(randperm(numel(st.y))) - 1;
        for w = 1:n_win
            f_acc = zeros(1, n_folds);
            for f = 1:n_folds
                w_vec = P_cell{s, w, f} * y_perm(st.tr_m(:, f));
                pred = Xte_cell{s, w, f} * w_vec;
                y_te = y_perm(st.te_m(:, f));
                sens = sum(y_te == 1 & pred > 0) / sum(y_te == 1);
                spec = sum(y_te == -1 & pred <= 0) / sum(y_te == -1);
                f_acc(f) = (sens + spec) / 2;
            end
            p_shapes(s, w) = mean(f_acc);
        end
    end
    null_dist(p, :) = mean(p_shapes, 1);
end
null_dist_s = smoothdata(null_dist, 2, smooth_typ, smooth_pts);

% 聚类统计
p_pt = (1 + sum(null_dist_s >= acc_avg_s, 1)) / (1 + n_perm);
sig_pts = (p_pt < 0.05) & (t_centers >= 0);

clusters = []; in_c = false; c_s = 1;
for w = 1:n_win
    if sig_pts(w) && ~in_c, in_c = true; c_s = w;
    elseif ~sig_pts(w) && in_c, in_c = false; clusters = [clusters; c_s, w-1]; %#ok<AGROW>
    end
end
if in_c, clusters = [clusters; c_s, n_win]; end

n_cl = size(clusters, 1);
real_masses = zeros(n_cl, 1);
for c_i = 1:n_cl
    real_masses(c_i) = sum(acc_avg_s(clusters(c_i,1):clusters(c_i,2)) - 0.5);
end

null_max_m = zeros(n_perm, 1);
for p = 1:n_perm
    c_p = null_dist_s(p, :);
    p_p = (1 + sum(null_dist_s >= c_p, 1)) / (1 + n_perm);
    s_m = (p_p < 0.05) & (t_centers >= 0);
    cls = []; in_cc = false; cc_s = 1;
    for w = 1:n_win
        if s_m(w) && ~in_cc, in_cc = true; cc_s = w;
        elseif ~s_m(w) && in_cc, in_cc = false; cls = [cls; cc_s, w-1]; %#ok<AGROW>
        end
    end
    if in_cc, cls = [cls; cc_s, n_win]; end
    if isempty(cls), null_max_m(p) = 0;
    else
        m_arr = zeros(size(cls, 1), 1);
        for ci = 1:size(cls, 1), m_arr(ci) = sum(c_p(cls(ci,1):cls(ci,2)) - 0.5); end
        null_max_m(p) = max(m_arr);
    end
end
cl_pvals = zeros(n_cl, 1);
for c_i = 1:n_cl
    cl_pvals(c_i) = (1 + sum(null_max_m >= real_masses(c_i))) / (1 + n_perm);
end

% 导出时程 CSV
tc_csv = fullfile(tc_dir, sprintf('%s_%s_task3_within_shape_timecourse.csv', sub_id, ch_name));
tc_tbl = table(repmat(string(sub_id), n_win, 1), repmat(string(ch_name), n_win, 1), ...
    t_centers', acc_avg_s', p_pt', acc_shapes_s(1,:)', acc_shapes_s(2,:)', acc_shapes_s(3,:)', ...
    mean(null_dist_s, 1)', prctile(null_dist_s, 97.5, 1)', prctile(null_dist_s, 2.5, 1)', ...
    'VariableNames', {'subject', 'channel', 'time_ms', 'acc_avg_smoothed', 'p_pointwise', ...
                      'acc_shape1_smoothed', 'acc_shape2_smoothed', 'acc_shape3_smoothed', ...
                      'null_mean', 'null_ci_upper', 'null_ci_lower'});
writetable(tc_tbl, tc_csv);

% 绘图 (左图: 3种形状 + 平均曲线; 右图: 峰值柱状图)
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 1200, 480]);

% 子图 1: 时程解码曲线
subplot(1, 2, 1); hold on; grid off;
set(gca, 'Box', 'off', 'FontSize', 11, 'LineWidth', 1.1);

null_up  = prctile(null_dist_s, 97.5, 1);
null_low = prctile(null_dist_s, 2.5, 1);
fill([t_centers, fliplr(t_centers)], [null_low, fliplr(null_up)], ...
    [0.85, 0.85, 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.6, 'HandleVisibility', 'off');

yline(0.5, '--', 'Color', [0.6, 0.6, 0.6], 'LineWidth', 1.2, 'HandleVisibility', 'off');
xline(0, '-', 'Color', [0.3, 0.3, 0.3], 'LineWidth', 1.0, 'HandleVisibility', 'off');

% 显著簇高亮
for c_i = 1:n_cl
    if cl_pvals(c_i) < 0.05
        fill([t_centers(clusters(c_i,1)), t_centers(clusters(c_i,2)), ...
              t_centers(clusters(c_i,2)), t_centers(clusters(c_i,1))], ...
             [0.35, 0.35, 0.85, 0.85], [1.0, 0.9, 0.7], 'EdgeColor', 'none', ...
             'FaceAlpha', 0.35, 'HandleVisibility', 'off');
    end
end

% 逐点显著标记
sig_p_idx = find(p_pt < 0.05 & t_centers >= 0);
if ~isempty(sig_p_idx)
    plot(t_centers(sig_p_idx), repmat(0.38, 1, numel(sig_p_idx)), 's', ...
        'MarkerFaceColor', [0.85, 0.35, 0.01], 'MarkerEdgeColor', 'none', ...
        'MarkerSize', 4, 'HandleVisibility', 'off');
end

% 绘制 3 种形状的单条曲线
shape_cols = [0.20, 0.55, 0.85; 0.25, 0.70, 0.45; 0.65, 0.35, 0.75];
for s = 1:n_shapes
    plot(t_centers, acc_shapes_s(s, :), 'Color', shape_cols(s, :), ...
        'LineWidth', 1.3, 'LineStyle', ':', 'DisplayName', sprintf('Shape %d (pic0%d)', s, s));
end

% 绘制 3 种形状平均线 (加粗实线)
plot(t_centers, acc_avg_s, 'Color', [0.85, 0.35, 0.01], 'LineWidth', 2.8, ...
    'DisplayName', 'Shape-Averaged (Mean)');

xlim([-200, 800]); ylim([0.35, 0.85]);
xlabel('Time from stimulus onset (ms)', 'FontWeight', 'bold');
ylabel('Balanced Accuracy', 'FontWeight', 'bold');
legend('Location', 'northwest', 'Box', 'off', 'FontSize', 9, 'Interpreter', 'none');

% 子图 2: 峰值柱状对比图
subplot(1, 2, 2); hold on; grid off;
set(gca, 'Box', 'off', 'FontSize', 11, 'LineWidth', 1.1);

b_names = {'Shape-Averaged', 'Shape 1', 'Shape 2', 'Shape 3'};
b_vals  = [max(acc_avg_s), max(acc_shapes_s(1,:)), max(acc_shapes_s(2,:)), max(acc_shapes_s(3,:))];
b_cols  = [[0.85, 0.35, 0.01]; shape_cols];

bh = bar(1:4, b_vals, 0.65, 'FaceColor', 'flat', 'EdgeColor', 'k', 'LineWidth', 0.9);
for k = 1:4, bh.CData(k, :) = b_cols(k, :); end
yline(0.5, '--', 'Color', [0.6, 0.6, 0.6], 'LineWidth', 1.2);

for k = 1:4
    text(k, b_vals(k) + 0.015, sprintf('%.1f%%', b_vals(k) * 100), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 9.5);
end
set(gca, 'XTick', 1:4, 'XTickLabel', b_names, 'XTickLabelRotation', 20, 'TickLabelInterpreter', 'none');
ylim([0.40, 0.85]); ylabel('Peak Balanced Accuracy', 'FontWeight', 'bold');

sgtitle(sprintf('%s - %s (Task 3 Within-Shape Color Decoding & Averaged)', sub_id, ch_name), ...
    'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'none');

fig_png = fullfile(fig_dir, sprintf('%s_%s_task3_within_shape_decoding.png', sub_id, ch_name));
saveas(fig, fig_png);
close(fig);

fprintf('[+] 单通道测试成功！\n  时程 CSV: %s\n  图像 PNG: %s\n  平均峰值正确率: %.2f%%\n', ...
    tc_csv, fig_png, max(acc_avg_s)*100);
