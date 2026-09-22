%% replot_c08_figures.m
% 重新绘制 C08 图表：简化标题、纵坐标 [0.40, 0.70]、添加 95% 置换检验灰色阴影

clear; clc; close all;

script_dir = fileparts(mfilename('fullpath'));
proj_root  = fileparts(fileparts(fileparts(script_dir)));
res_root   = fullfile(proj_root, 'color_analyse_0825', 'result');
tab_dir    = fullfile(res_root, 'tables');
fig_dir    = fullfile(res_root, 'figures', 'multisite_decoding');
sub_fig_dir = fullfile(fig_dir, 'subjects');

ws_file = fullfile(tab_dir, 'multisite_decoding_task2_workspace.mat');
if ~isfile(ws_file)
    error('未找到工作区文件: %s', ws_file);
end

fprintf('>>> 正在载入工作区: %s ...\n', ws_file);
load(ws_file);

c04_table  = fullfile(res_root, 'tables', 'color_effects_summary.mat');
loaded_c04 = load(c04_table);
if isfield(loaded_c04, 'all_tbl'), sig_tbl = loaded_c04.all_tbl; else, sig_tbl = loaded_c04.res_table; end
sig_tbl    = sig_tbl(sig_tbl.is_significant == 1, :);

all_subs = {'sub001', 'sub002', 'sub003', 'sub004', 'sub005', 'sub006', 'sub007', 'sub008'};
n_subs   = numel(all_subs);
n_win    = numel(t_centers);

sub_elecs = struct();
for s_i = 1:n_subs
    sub = all_subs{s_i};
    st = sig_tbl(strcmp(sig_tbl.subject, sub), :);
    
    ch_pos = unique(st.channel(strcmp(st.concordance_type, 'Concordant_Positive')));
    ch_neg = unique(st.channel(strcmp(st.concordance_type, 'Concordant_Negative')));
    ch_concord = unique([ch_pos; ch_neg]);
    ch_diff_raw = unique(st.channel(strcmp(st.concordance_type, 'Category_Biased')));
    ch_diff = setdiff(ch_diff_raw, ch_concord);
    ch_all = unique(st.channel);
    
    sub_elecs.(sub).pos     = ch_pos;
    sub_elecs.(sub).neg     = ch_neg;
    sub_elecs.(sub).diff    = ch_diff;
    sub_elecs.(sub).concord = ch_concord;
    sub_elecs.(sub).all_sig = ch_all;
end

%% 1. 重新绘制 8 个被试的单被试图谱 (含 95% 置换灰色阴影与显著横条)
fprintf('>>> 正在重绘 8 个单被试图谱 ...\n');
for s_i = 1:n_subs
    sub_id = all_subs{s_i};
    sub_fig_png = fullfile(sub_fig_dir, sprintf('%s_multisite_decoding.png', sub_id));
    
    f_sub = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 1450, 840]);
    for b_idx = 1:cfg.n_bands
        subplot(2, 3, b_idx);
        hold on; grid off;
        set(gca, 'Box', 'off', 'FontSize', 10, 'LineWidth', 1.0);
        
        % 绘制 95% 置换检验零分布灰色阴影
        sub_null_b = reshape(sub_null_acc(s_i, :, b_idx, :, :), cfg.n_sets * cfg.n_perm, n_win);
        sub_null_b_sm = smoothdata(sub_null_b, 2, cfg.smooth_typ, cfg.smooth_pts);
        ci_up  = prctile(sub_null_b_sm, 97.5, 1);
        ci_low = prctile(sub_null_b_sm, 2.5, 1);
        fill([t_centers, fliplr(t_centers)], [ci_up, fliplr(ci_low)], ...
            [0.85, 0.85, 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
        
        yline(0.50, '--', 'Color', [0.55, 0.55, 0.55], 'LineWidth', 1.0, 'HandleVisibility', 'off');
        xline(0, '-', 'Color', [0.4, 0.4, 0.4], 'LineWidth', 0.8, 'HandleVisibility', 'off');
        
        h_lines = [];
        for s_idx = 1:cfg.n_sets
            n_elecs = numel(sub_elecs.(sub_id).(cfg.sets{s_idx}));
            if n_elecs == 0, continue; end
            
            c_raw = reshape(sub_real_acc(s_i, s_idx, b_idx, :), 1, n_win);
            c_sm  = smoothdata(c_raw, 2, cfg.smooth_typ, cfg.smooth_pts);
            lw = 1.8;
            if s_idx == 5, lw = 2.4; end
            hl = plot(t_centers, c_sm, 'Color', cfg.set_cols(s_idx, :), 'LineWidth', lw, ...
                'DisplayName', sprintf('%s (N=%d)', cfg.set_short{s_idx}, n_elecs));
            h_lines = [h_lines, hl]; %#ok<AGROW>
            
            % 置换检验显著性横条
            null_raw = reshape(sub_null_acc(s_i, s_idx, b_idx, :, :), cfg.n_perm, n_win);
            null_sm  = smoothdata(null_raw, 2, cfg.smooth_typ, cfg.smooth_pts);
            p_pt_sub = (1 + sum(null_sm >= c_sm, 1)) / (1 + cfg.n_perm);
            sig_m_sub = (p_pt_sub < cfg.cluster_alpha) & (t_centers >= 0);
            
            s_cls = []; in_sc = false; sc_start = 1;
            for w = 1:n_win
                if sig_m_sub(w) && ~in_sc
                    in_sc = true; sc_start = w;
                elseif ~sig_m_sub(w) && in_sc
                    in_sc = false; s_cls = [s_cls; sc_start, w-1]; %#ok<AGROW>
                end
            end
            if in_sc, s_cls = [s_cls; sc_start, n_win]; end
            
            n_sc = size(s_cls, 1);
            s_real_mass = zeros(n_sc, 1);
            for sc_i = 1:n_sc
                s_real_mass(sc_i) = sum(c_sm(s_cls(sc_i, 1) : s_cls(sc_i, 2)) - 0.50);
            end
            
            s_null_max = zeros(cfg.n_perm, 1);
            for p_i = 1:cfg.n_perm
                p_curve = null_sm(p_i, :);
                p_pt_null = (1 + sum(null_sm >= p_curve, 1)) / (1 + cfg.n_perm);
                p_sig_null = (p_pt_null < cfg.cluster_alpha) & (t_centers >= 0);
                c_null = []; in_nc = false; nc_s = 1;
                for w = 1:n_win
                    if p_sig_null(w) && ~in_nc
                        in_nc = true; nc_s = w;
                    elseif ~p_sig_null(w) && in_nc
                        in_nc = false; c_null = [c_null; nc_s, w-1]; %#ok<AGROW>
                    end
                end
                if in_nc, c_null = [c_null; nc_s, n_win]; end
                if isempty(c_null)
                    s_null_max(p_i) = 0;
                else
                    m_arr = zeros(size(c_null, 1), 1);
                    for ncl_i = 1:size(c_null, 1)
                        m_arr(ncl_i) = sum(p_curve(c_null(ncl_i, 1) : c_null(ncl_i, 2)) - 0.50);
                    end
                    s_null_max(p_i) = max([0; m_arr]);
                end
            end
            
            s_pvals = zeros(n_sc, 1);
            for sc_i = 1:n_sc
                s_pvals(sc_i) = (1 + sum(s_null_max >= s_real_mass(sc_i))) / (1 + cfg.n_perm);
            end
            
            % 在 0.406 ~ 0.422 绘制横条
            y_bar = 0.422 - (s_idx - 1) * 0.004;
            for sc_i = 1:n_sc
                t_s = t_centers(s_cls(sc_i, 1));
                t_e = t_centers(s_cls(sc_i, 2));
                dur = t_e - t_s;
                if s_pvals(sc_i) < 0.05
                    plot([t_s, t_e], [y_bar, y_bar], 'LineWidth', 3.2, ...
                        'Color', cfg.set_cols(s_idx, :), 'HandleVisibility', 'off');
                elseif dur >= 40
                    plot([t_s, t_e], [y_bar, y_bar], 'LineWidth', 1.5, 'LineStyle', ':', ...
                        'Color', cfg.set_cols(s_idx, :), 'HandleVisibility', 'off');
                end
            end
        end
        xlim(cfg.t_range);
        ylim([0.40, 0.70]);
        xlabel('Time (ms)', 'FontSize', 10);
        ylabel('Balanced Accuracy', 'FontSize', 10);
        title(cfg.bands{b_idx}, 'FontSize', 12, 'FontWeight', 'bold');
        if b_idx == 1
            legend(h_lines, 'Location', 'northwest', 'Box', 'off', 'FontSize', 8.5);
        end
    end
    sgtitle(sprintf('Subject %s', sub_id), 'FontSize', 14, 'FontWeight', 'bold');
    if isfile(sub_fig_png), try, delete(sub_fig_png); catch, end; end
    exportgraphics(f_sub, sub_fig_png, 'Resolution', 200);
    close(f_sub);
    fprintf('  [+] %s 重绘完成\n', sub_id);
end

%% 2. 重新绘制群体 6 频段时程图
fprintf('>>> 正在重绘群体 6 频段时程图 ...\n');
fig1 = figure('Visible', 'off', 'Color', 'w', 'Position', [80, 80, 1500, 880]);

for b = 1:cfg.n_bands
    subplot(2, 3, b);
    hold on; grid off;
    set(gca, 'Box', 'off', 'FontSize', 11, 'LineWidth', 1.1, ...
        'XColor', [0.15, 0.15, 0.15], 'YColor', [0.15, 0.15, 0.15]);
    
    % 绘制 95% 置换检验零分布灰色阴影
    g_null_b = reshape(group_null_smoothed(:, b, :, :), cfg.n_sets * cfg.n_perm, n_win);
    ci_up  = prctile(g_null_b, 97.5, 1);
    ci_low = prctile(g_null_b, 2.5, 1);
    fill([t_centers, fliplr(t_centers)], [ci_up, fliplr(ci_low)], ...
        [0.85, 0.85, 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
    
    yline(0.50, '--', 'Color', [0.55, 0.55, 0.55], 'LineWidth', 1.0, 'HandleVisibility', 'off');
    xline(0, '-', 'Color', [0.35, 0.35, 0.35], 'LineWidth', 0.8, 'HandleVisibility', 'off');
    
    h_lines = zeros(cfg.n_sets, 1);
    for set_i = 1:cfg.n_sets
        c_curve = squeeze(group_real_smoothed(set_i, b, :));
        c_col   = cfg.set_cols(set_i, :);
        lw = 1.8;
        if set_i == 5, lw = 2.4; end
        h_lines(set_i) = plot(t_centers, c_curve, 'Color', c_col, 'LineWidth', lw, ...
            'DisplayName', cfg.set_names{set_i});
        
        cl_info = group_cluster_info{set_i, b};
        if ~isempty(cl_info) && ~isempty(cl_info.clusters)
            for c_k = 1:size(cl_info.clusters, 1)
                if cl_info.pvals(c_k) < 0.05
                    c_start = t_centers(cl_info.clusters(c_k, 1));
                    c_end   = t_centers(cl_info.clusters(c_k, 2));
                    y_bar   = 0.422 - (set_i - 1) * 0.004;
                    plot([c_start, c_end], [y_bar, y_bar], 'LineWidth', 3.2, ...
                        'Color', c_col, 'HandleVisibility', 'off');
                end
            end
        end
    end
    
    xlim(cfg.t_range);
    ylim([0.40, 0.70]);
    xlabel('Time (ms)', 'FontWeight', 'bold', 'FontSize', 11);
    ylabel('Balanced Accuracy', 'FontWeight', 'bold', 'FontSize', 11);
    title(cfg.bands{b}, 'FontSize', 13, 'FontWeight', 'bold');
    
    if b == 1
        legend(h_lines, cfg.set_names, 'Location', 'northwest', 'Box', 'off', ...
            'FontSize', 9.0, 'Interpreter', 'none');
    end
end

sgtitle('Group Decoding (Task 2)', 'FontSize', 15, 'FontWeight', 'bold');
fig1_png = fullfile(fig_dir, sprintf('multisite_decoding_%s_timecourses_6bands.png', cfg.task_name));
if isfile(fig1_png), try, delete(fig1_png); catch, end; end
try
    exportgraphics(fig1, fig1_png, 'Resolution', 300);
catch
    saveas(fig1, fig1_png);
end
close(fig1);
fprintf('  [+] 群体时程图重绘完成: %s\n', fig1_png);

%% 3. 重新绘制峰值性能对比柱状图
fprintf('>>> 正在重绘峰值柱状图 ...\n');
fig2 = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 1200, 560]);
hold on; grid off;
set(gca, 'Box', 'off', 'FontSize', 12, 'LineWidth', 1.2, ...
    'XColor', [0.15, 0.15, 0.15], 'YColor', [0.15, 0.15, 0.15]);

peak_mat = zeros(cfg.n_bands, cfg.n_sets);
for b = 1:cfg.n_bands
    for set_i = 1:cfg.n_sets
        peak_mat(b, set_i) = max(squeeze(group_real_smoothed(set_i, b, :)));
    end
end

b_bars = bar(1:cfg.n_bands, peak_mat, 0.85, 'EdgeColor', 'k', 'LineWidth', 0.8);
for set_i = 1:cfg.n_sets
    b_bars(set_i).FaceColor = cfg.set_cols(set_i, :);
end

yline(0.50, '--', 'Color', [0.65, 0.65, 0.65], 'LineWidth', 1.2, 'HandleVisibility', 'off');
ylim([0.45, 0.60]);
set(gca, 'XTick', 1:cfg.n_bands, 'XTickLabel', cfg.bands, 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Peak Balanced Accuracy', 'FontSize', 13, 'FontWeight', 'bold');
title('Peak Decoding Accuracy across Bands', 'FontSize', 14, 'FontWeight', 'bold');
legend(cfg.set_names, 'Location', 'northeast', 'Box', 'off', 'FontSize', 10.5);

for b = 1:cfg.n_bands
    for set_i = 1:cfg.n_sets
        val = peak_mat(b, set_i);
        x_offset = b_bars(set_i).XEndPoints(b);
        text(x_offset, val + 0.005, sprintf('%.1f%%', val * 100), ...
            'HorizontalAlignment', 'center', 'FontSize', 8, 'FontWeight', 'bold', ...
            'Rotation', 45);
    end
end

fig2_png = fullfile(fig_dir, sprintf('multisite_decoding_%s_peak_comparison.png', cfg.task_name));
if isfile(fig2_png), try, delete(fig2_png); catch, end; end
try
    exportgraphics(fig2, fig2_png, 'Resolution', 300);
catch
    saveas(fig2, fig2_png);
end
close(fig2);
fprintf('  [+] 峰值对比柱状图重绘完成: %s\n', fig2_png);
fprintf('\n>>> 全部图表重绘完成！\n');
