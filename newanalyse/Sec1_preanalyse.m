%% ========================================================================
%  全自动化预处理管线 (EEGLAB Pipeline)
%  包含：提取Marker -> 凹陷滤波 -> 1Hz高通滤波 -> 自动坏道剔除(3SD) ->
%       局部重参考 -> ICA降噪 -> 250Hz低通滤波 ->
%       双分支(ERP/TFA)频带滤波 -> 分段 -> 基线矫正(-250~-50ms) -> 自动坏段剔除
%% ========================================================================
clear;
clc;

cfg = newanalyse_load_run_config(mfilename, {'matlab_defaults', 'sec1_defaults'});
subject = 'test001';
if isfield(cfg, 'subject')
    subject = char(string(cfg.subject));
end

paths = newanalyse_paths();
project_root = paths.base_path;
raw_subject_dir = regexprep(subject, '^test0*', 'test');
if isfield(cfg, 'raw_subject_dir')
    raw_subject_dir = char(string(cfg.raw_subject_dir));
end
raw_data_dir = fullfile(project_root, 'seegdata', raw_subject_dir);
if isfield(cfg, 'raw_data_dir')
    raw_data_dir = char(string(cfg.raw_data_dir));
end
output_dir = fullfile(project_root, 'processed_data', subject);
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

% 启动并重置 EEGLAB 环境
% [ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
for i = 1:3
    task_label = i;
    preprocess_analyse(task_label, raw_data_dir, output_dir)
end

function preprocess_analyse(task_label, raw_data_dir, output_dir)
clc;
% 启动并重置 EEGLAB 环境
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
% 设定文件路径和参数 (请替换为你的实际参数)
data_path    = raw_data_dir;   % 原始数据文件夹路径
save_path    = output_dir;
file_name    = sprintf('erp%d.set',task_label);         % 原始数据文件名


% ==================== 0. 加载数据 ====================
fprintf('--- 加载数据 ---\n');
EEG = pop_loadset('filename', file_name, 'filepath', data_path);
[ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, 0 );

% ==================== 0.5 降采样至 500Hz ====================
fprintf('--- 降采样至 500Hz ---\n');
EEG = pop_resample(EEG, 500);

target_event = unique({EEG.event.type});

% ==================== 1. 提取 Marker ====================
% (EEGLAB在加载数据时已自动解析marker至 EEG.event。此处确认marker是否存在)
% if ~any(strcmp({EEG.event.type}, target_event))
%     warning(['未能在数据中找到目标 marker: ', target_event, '，请检查命名！']);
% endi


% ==================== 2. 凹陷滤波 (50Hz 及其谐波) ====================
fprintf('--- 执行凹陷滤波 (50, 100, 150 Hz) ---\n');
% 利用 FIR 滤波器切除工频干扰点
EEG = pop_eegfiltnew(EEG, 'locutoff', 49, 'hicutoff', 51, 'revfilt', 1);
EEG = pop_eegfiltnew(EEG, 'locutoff', 99, 'hicutoff', 101, 'revfilt', 1);
EEG = pop_eegfiltnew(EEG, 'locutoff', 149, 'hicutoff', 151, 'revfilt', 1);


% ==================== 2.5 高通滤波 (1Hz) ====================
fprintf('--- 执行 1Hz 高通滤波 ---\n');
EEG = pop_eegfiltnew(EEG, 'locutoff', 1);  % 1Hz 高通滤波，去除慢漂移


% ==================== 3. 自动坏道剔除 (方差超 3SD) ====================
fprintf('--- 剔除并插值坏道 ---\n');
% 使用 EEGLAB 内置函数，基于方差(var)，阈值为3个标准差剔除坏道
% ==================== 3. 自动坏道剔除 (方差超 3SD) ====================
fprintf('--- 计算方差并标记坏道 ---\n');

% 1. 计算每个通道在时间维度上的方差
chan_vars = var(EEG.data, 0, 2);

% 2. 计算所有通道方差的总体均值和标准差
mean_var = mean(chan_vars);
std_var = std(chan_vars);

% 3. 找出方差偏离均值超过 3 倍标准差的坏道索引
indElec = find(chan_vars > (mean_var + 3 * std_var) | chan_vars < (mean_var - 3 * std_var));



% ==================== 4. 局部混合重参考 (SEEG 1D Laplace + Bipolar) ====================
fprintf('--- 执行 SEEG 局部重参考 ---\n');
%%
% 获取所有通道名称
chan_labels = {EEG.chanlocs.labels};
num_chans = length(chan_labels);

% 初始化变量来存储解析后的电极棒名称和触点编号
shaft_names = cell(1, num_chans);
contact_nums = nan(1, num_chans);

% 1. 解析通道名称 (例如 'A11' -> shaft: 'A', num: 11)
for i = 1:num_chans
    label = chan_labels{i};
    % 找到第一个数字的位置，以此为界分割字母和数字
    first_digit_idx = find(isstrprop(label, 'digit'), 1);
    
    if ~isempty(first_digit_idx)
        shaft_names{i} = label(1:first_digit_idx-1);
        contact_nums(i) = str2double(label(first_digit_idx:end));
    else
        % 如果没有数字（比如参考电极、心电或肌电），保持原名并不分配编号
        shaft_names{i} = label;
    end
end

% 找出所有唯一的电极棒名称 (排除非 SEEG 通道)
unique_shafts = unique(shaft_names(~isnan(contact_nums)));
temp_data = EEG.data;
local_ref_data = zeros(size(temp_data));

% 2. 按每根电极棒依次处理
for s = 1:length(unique_shafts)
    curr_shaft = unique_shafts{s};
    
    % 找到当前电极棒的所有可用通道索引
    shaft_idx = find(strcmp(shaft_names, curr_shaft));
    
    for c = 1:length(shaft_idx)
        curr_ch_idx = shaft_idx(c);
        curr_num = contact_nums(curr_ch_idx);
        
        % 寻找物理上紧邻的触点 (严格依据编号 -1 和 +1)
        neighbor_idx = [];
        
        % 找同一根棒上编号 -1 的触点
        prev_idx = shaft_idx(contact_nums(shaft_idx) == curr_num - 1);
        if ~isempty(prev_idx)
            neighbor_idx(end+1) = prev_idx;
        end
        
        % 找同一根棒上编号 +1 的触点
        next_idx = shaft_idx(contact_nums(shaft_idx) == curr_num + 1);
        if ~isempty(next_idx)
            neighbor_idx(end+1) = next_idx;
        end
        
        % 3. 根据邻居数量执行不同的重参考逻辑
        if length(neighbor_idx) == 2
            % 中间位点：减去两侧的平均值 (1D Laplace)
            local_ref_data(curr_ch_idx, :) = temp_data(curr_ch_idx, :) - mean(temp_data(neighbor_idx, :), 1);
        elseif length(neighbor_idx) == 1
            % 两端位点：减去唯一相邻的一个值 (Bipolar)
            local_ref_data(curr_ch_idx, :) = temp_data(curr_ch_idx, :) - temp_data(neighbor_idx, :);
        else
            % 孤立位点（物理上的相邻触点都被剔除且未插值）：保持原样
            local_ref_data(curr_ch_idx, :) = temp_data(curr_ch_idx, :);
        end
    end
end

% 4. 处理非 SEEG 通道（直接复制，不参与重参考计算）
non_seeg_idx = find(isnan(contact_nums));
if ~isempty(non_seeg_idx)
    local_ref_data(non_seeg_idx, :) = temp_data(non_seeg_idx, :);
end

EEG.data = local_ref_data;


% 在 ICA 之前备份数据，用于后续 ICA 效果对比绘图
EEG_preICA = EEG;

% ==================== 4.5 ICA 两阶段自动降噪 (KL 散度) ====================
% fprintf('--- 执行 ICA 降噪 ---\n');
% % 使用 runica 算法运行 ICA 分解
% EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1);

% fprintf('--- 基于 Kullback-Leibler 散度执行两阶段 ICA 自动剔除 ---\n');
% % 获取所有成分的逆权重矩阵 (Channels x Components)
% % 注意: EEG.icawinv 表示每个成分在各个通道上的空间分布
% num_comps = size(EEG.icaweights, 1);
% seeg_idx = find(~isnan(contact_nums)); % 提取有效 SEEG 通道的索引
% num_seeg = length(seeg_idx);

% reject_comps = [];
% reject_reasons = cell(num_comps, 1);

% for j = 1:num_comps
%     % 取出该成分在所有 SEEG 通道上的空间绝对权重
%     weights_abs = abs(EEG.icawinv(seeg_idx, j));

%     % --- 阶段1：全局系统噪声识别 ---
%     p_global = weights_abs / sum(weights_abs); % 归一化为概率分布 P
%     q_global = ones(num_seeg, 1) / num_seeg;   % 全局均匀分布 Q

%     % 计算 KL 散度 D_KL(P || Q) = sum( P * log(P / Q) )
%     dkl_global = sum(p_global(p_global>0) .* log(p_global(p_global>0) ./ q_global(p_global>0)));

%     if dkl_global < 0.2
%         reject_comps(end+1) = j;
%         reject_reasons{j} = sprintf('全局噪声 (D_kl = %.4f < 0.2)', dkl_global);
%         continue; % 已经是全局噪声，无需进一步检测局部
%     end

%     % --- 阶段2：局部系统噪声识别 ---
%     for s = 1:length(unique_shafts)
%         curr_shaft = unique_shafts{s};
%         % 当前电极棒中且为有效 SEEG 触点的索引
%         shaft_idx_global = find(strcmp(shaft_names, curr_shaft) & ~isnan(contact_nums));

%         % 如果电极棒上的触点太少（例如<=2），均匀分布没有区分度，跳过
%         num_local = length(shaft_idx_global);
%         if num_local > 2
%             % 获取该成分在该电极棒通道上的权重
%             weights_local = abs(EEG.icawinv(shaft_idx_global, j));
%             sum_local = sum(weights_local);

%             if sum_local > 0
%                 p_local = weights_local / sum_local; % 局部归一化
%                 q_local = ones(num_local, 1) / num_local; % 局部均匀分布

%                 dkl_local = sum(p_local(p_local>0) .* log(p_local(p_local>0) ./ q_local(p_local>0)));

%                 if dkl_local < 0.05
%                     reject_comps(end+1) = j;
%                     reject_reasons{j} = sprintf('局部噪声 (电极棒 %s, D_kl = %.4f < 0.05)', curr_shaft, dkl_local);
%                     break; % 只要在一根电极上表现极度均匀，即认为是局部弥散噪声
%                 end
%             end
%         end
%     end
% end

% % 打印剔除汇总并实际执行去除
% if ~isempty(reject_comps)
%     fprintf('\n识别并即将去除 %d 个 ICA 噪声成分：\n', length(reject_comps));
%     for k = 1:length(reject_comps)
%         fprintf('  - 成分 %d: %s\n', reject_comps(k), reject_reasons{reject_comps(k)});
%     end
%     EEG = pop_subcomp(EEG, reject_comps, 0);
% else
%     fprintf('根据 KL 散度阈值，未发现需要剔除的极度弥散噪声成分。\n');
% end


% ==================== 4.6 低通滤波 (200Hz) ====================
% 注意: 降采样至 500Hz 后 Nyquist 频率为 250Hz，截止频率必须低于此值
fprintf('--- 执行 200Hz 低通滤波 ---\n');
EEG = pop_eegfiltnew(EEG, 'hicutoff', 200);  % 200Hz 低通滤波


EEG_Ref = EEG; % 保存公共预处理节点
% ==============================================================================


%% ==================== 5-8. 分支 A: ERP 处理管线 ====================
fprintf('\n>>> 开始执行 ERP 分支管线 <<<\n');
EEG_ERP = EEG_Ref;

% 5. 分频带滤波 (1-30Hz)
EEG_ERP = pop_eegfiltnew(EEG_ERP, 'locutoff', 1, 'hicutoff', 30);

% 6. 分段 (-500 到 1000ms)
EEG_ERP = pop_epoch(EEG_ERP, target_event, [-0.5 1.0], 'newname', 'ERP_epochs', 'epochinfo', 'yes');

% 7. 基线矫正 (-200 到 0ms)
EEG_ERP = pop_rmbase(EEG_ERP, [-200 0]);

% 8. 去除坏段 (假设剔除振幅超过 ±300微伏 的 trial，需根据实际 LFP 幅值调整)
% EEG_ERP = pop_eegthresh(EEG_ERP, 1, 1:EEG_ERP.nbchan, -300, 300, -0.5, 1.0, 0, 1);

% 保存 ERP 数据
pop_saveset(EEG_ERP, 'filename', 'processed_ERP.set', 'filepath', save_path);


%% ==================== 5-8. 分支 B: TFA 处理管线 ====================
fprintf('\n>>> 开始执行 TFA 分支管线 <<<\n');
EEG_TFA = EEG_Ref;

% 5. 分频带滤波 (1-150Hz)
EEG_TFA = pop_eegfiltnew(EEG_TFA, 'locutoff', 1, 'hicutoff', 150);

% 6. 分段 (-500 到 1000ms)
EEG_TFA = pop_epoch(EEG_TFA, target_event, [-0.5 1.0], 'newname', 'TFA_epochs', 'epochinfo', 'yes');

% 7. 基线矫正 (-200 到 0ms)
EEG_TFA = pop_rmbase(EEG_TFA, [-200 0]);

% 8. 去除坏段 (SEEG 更适合基于试次异常程度做稳健剔除，而不是固定振幅阈值)
fprintf('--- 执行 SEEG 自动坏段剔除 ---\n');
EEG_QC = EEG_Ref;
EEG_QC = pop_epoch(EEG_QC, target_event, [-0.5 1.0], 'newname', 'QC_epochs', 'epochinfo', 'yes');
EEG_QC = pop_rmbase(EEG_QC, [-200 0]);
[bad_epochs, bad_epoch_stats] = detect_bad_epochs_seeg(EEG_QC);
if ~isempty(bad_epochs)
    fprintf('自动检测到 %d/%d 个坏段 (%.1f%%)，将从 ERP/TFA 两个分支同步剔除。\n', ...
        length(bad_epochs), EEG_QC.trials, 100 * length(bad_epochs) / EEG_QC.trials);
    fprintf('坏段索引: %s\n', mat2str(bad_epochs));
    fprintf('坏段中位坏通道比例: %.3f | 最大坏通道比例: %.3f\n', ...
        median(bad_epoch_stats.bad_channel_fraction(bad_epochs)), ...
        max(bad_epoch_stats.bad_channel_fraction(bad_epochs)));
    EEG_ERP = pop_select(EEG_ERP, 'notrial', bad_epochs);
    EEG_TFA = pop_select(EEG_TFA, 'notrial', bad_epochs);
else
    fprintf('未检测到需要剔除的坏段。\n');
end
clear EEG_QC bad_epoch_stats bad_epochs;

% 保存 TFA 数据
pop_saveset(EEG_TFA, 'filename', 'processed_TFA.set', 'filepath', save_path);

fprintf('\n✅ 所有预处理步骤执行完毕！\n');

%% ==================== 9. 提取 Epoch 数据并保存为 .mat ====================
fprintf('\n>>> 正在提取 3D 矩阵并保存为 .mat 文件 <<<\n');

% 1. 提取三维数据矩阵 [Channels x Times x Epochs]
erp_data3D = EEG_ERP.data;
tfa_data3D = EEG_TFA.data;

% 2. 提取时间轴向量 (单位: 毫秒)
% 方便后续画图或在 Python 中构建 MNE Epochs 对象时对齐时间
time_axis = EEG_ERP.times;
target_event = setdiff(target_event,'Trigger-In:99');
% 3. 提取每个 Trial 对应的 Marker 标签 (极其重要！)
% 由于剔除了坏段，剩余的 trial 数量和原始顺序已经改变。
% 我们需要遍历 EEG.epoch 结构体，把每个 trial 锁定时的 marker 记录下来。
num_epochs = EEG_ERP.trials;
trial_labels = cell(1, num_epochs);
epoch.data = cell(length(target_event), 1); % 使用 cell array 存储每个条件的数据，因为 trial 数可能不同
epoch.trigger = target_event;
epoch.ch = EEG_ERP.chanlocs;
epoch.name = 'erp1';
% 临时存储矩阵以便统一格式 (如果需要强制对齐 Trial 数，需截断或填充；通常分析时保持原始 Trial 数)
% 这里我们修改逻辑：epoch.data 应该是一个 Cell Array，或者如果 Trial 数一致，才是 4D 矩阵
% 但报错显示 "左侧 1x69x98x750, 右侧 70x98x750"，说明第 i 个条件的 Trial 数 (70) 与之前初始化的不一致 (69?)
% 或者初始化时默认为 0?
% 
% 更好的做法：先找出所有条件中 Trial 数的最小值，或者使用 Cell Array。
% 为了兼容后续分析代码 (preprocess_highgamma.m 读取 epoch.data 为 4D 矩阵 [Cond, Rep, Ch, Time])
% 我们必须保证每个条件的 Trial 数一致 (Rep 维度)。
%
% 方案：截断至最小 Trial 数
min_trials = Inf;
for i = 1:length(target_event)
    idx = strcmp(target_event{i}, {EEG_ERP.epoch.eventtype});
    current_trials = sum(idx);
    if current_trials < min_trials
        min_trials = current_trials;
    end
end
fprintf('  所有条件保留最小 Trial 数: %d\n', min_trials);

% 初始化 4D 矩阵 [Cond, Rep, Ch, Time]
% Rep = min_trials
n_chans = EEG_ERP.nbchan;
n_times = EEG_ERP.pnts;
epoch.data = zeros(length(target_event), min_trials, n_chans, n_times);

for i = 1:length(target_event)
    % 找到该条件的所有 Trial
    event_indices = find(strcmp(target_event{i}, {EEG_ERP.epoch.eventtype}));
    
    % 如果 Trial 数多于 min_trials，截取前 min_trials 个
    selected_indices = event_indices(1:min_trials);
    
    % 提取数据: [Ch, Time, Rep] -> permute -> [Rep, Ch, Time]
    trial_data = EEG_ERP.data(:, :, selected_indices);
    epoch.data(i, :, :, :) = permute(trial_data, [3, 1, 2]);
end

mat_filename = fullfile(save_path, sprintf('task%d_ERP_epoched.mat',task_label));
save(mat_filename, 'epoch');
fprintf('成功将干净的 Epoch 矩阵及标签保存至: %s\n', mat_filename);


%%
epoch.name = 'tfa1';
% 同样对 TFA 数据执行最小 Trial 数截断，以保证数据维度对齐
% 注意：EEG_TFA 的 Trial 数应该与 EEG_ERP 严格一致（如果剔除坏段逻辑相同）
% 但为了保险，重新计算 min_trials (或者直接复用上面的 min_trials 如果确定一致)

% 这里为了安全，针对 TFA 数据重新计算 min_trials
min_trials_tfa = Inf;
for i = 1:length(target_event)
    idx = strcmp(target_event{i}, {EEG_TFA.epoch.eventtype});
    current_trials = sum(idx);
    if current_trials < min_trials_tfa
        min_trials_tfa = current_trials;
    end
end
fprintf('  TFA 数据保留最小 Trial 数: %d\n', min_trials_tfa);

n_chans_tfa = EEG_TFA.nbchan;
n_times_tfa = EEG_TFA.pnts;
epoch.data = zeros(length(target_event), min_trials_tfa, n_chans_tfa, n_times_tfa);

for i = 1:length(target_event)
    event_indices = find(strcmp(target_event{i}, {EEG_TFA.epoch.eventtype}));
    selected_indices = event_indices(1:min_trials_tfa);
    trial_data = EEG_TFA.data(:, :, selected_indices);
    epoch.data(i, :, :, :) = permute(trial_data, [3, 1, 2]);
end
mat_filename = fullfile(save_path, sprintf('task%d_TFA_epoched.mat',task_label));
save(mat_filename, 'epoch');
fprintf('成功将干净的 Epoch 矩阵及标签保存至: %s\n', mat_filename);
end

%% ==================== 10. ICA 降噪前后对比绘图 ====================
% fprintf('\n>>> 开始绘制 ICA 降噪前后对比图 <<<\n');

% % --- 10.1 对 ICA 之前的数据执行与主管线完全相同的后续处理 ---
% % 低通滤波 (与主管线一致，200Hz)
% EEG_preICA_proc = pop_eegfiltnew(EEG_preICA, 'hicutoff', 200);
% % ERP 频带滤波 (1-30Hz)
% EEG_preICA_proc = pop_eegfiltnew(EEG_preICA_proc, 'locutoff', 1, 'hicutoff', 30);
% % 分段
% EEG_preICA_proc = pop_epoch(EEG_preICA_proc, target_event, [-0.5 1.0], ...
%     'newname', 'preICA_ERP_epochs', 'epochinfo', 'yes');
% % 基线矫正
% EEG_preICA_proc = pop_rmbase(EEG_preICA_proc, [-250 -50]);

% % 选取用于比较的通道数量 (最多展示 8 个通道)
% num_plot_chans = min(8, EEG_ERP.nbchan);
% plot_chan_idx = round(linspace(1, EEG_ERP.nbchan, num_plot_chans));

% % --- 10.2 图 1：多通道叠加平均 ERP 波形对比 (pre-ICA vs post-ICA) ---
% figure('Name', 'ICA 前后 ERP 对比', 'NumberTitle', 'off', ...
%     'Position', [50, 50, 1400, 900], 'Color', 'w');

% for p = 1:num_plot_chans
%     ch = plot_chan_idx(p);
%     subplot(ceil(num_plot_chans/2), 2, p);
%     hold on;

%     % ICA 之前的平均 ERP
%     erp_pre = mean(EEG_preICA_proc.data(ch, :, :), 3);
%     % ICA 之后的平均 ERP
%     erp_post = mean(EEG_ERP.data(ch, :, :), 3);

%     plot(EEG_ERP.times, erp_pre, 'Color', [0.7 0.2 0.2, 0.7], 'LineWidth', 1.2);
%     plot(EEG_ERP.times, erp_post, 'Color', [0.1 0.4 0.8, 0.9], 'LineWidth', 1.5);

%     xline(0, '--k', 'LineWidth', 0.8);
%     xlabel('Time (ms)');
%     ylabel('Amplitude (\muV)');
%     title(sprintf('Ch %s', EEG_ERP.chanlocs(ch).labels), 'Interpreter', 'none');
%     legend('Pre-ICA', 'Post-ICA', 'Location', 'best');
%     hold off;
%     box on;
% end
% sgtitle('ICA Denoising Comparison: Average ERP', 'FontSize', 14, 'FontWeight', 'bold');
% saveas(gcf, fullfile(data_path, 'ICA_comparison_ERP.png'));
% fprintf('图 1 已保存: ICA_comparison_ERP.png\n');

% % --- 10.3 图 2：单通道全试次热力图对比 ---
% demo_ch = plot_chan_idx(1); % 使用第一个展示通道
% figure('Name', 'ICA 前后单通道热力图对比', 'NumberTitle', 'off', ...
%     'Position', [100, 100, 1200, 500], 'Color', 'w');

% subplot(1,2,1);
% imagesc(EEG_ERP.times, 1:size(EEG_preICA_proc.data,3), ...
%     squeeze(EEG_preICA_proc.data(demo_ch, :, :))');
% colorbar; xlabel('Time (ms)'); ylabel('Trial');
% title(sprintf('Pre-ICA: %s', EEG_ERP.chanlocs(demo_ch).labels), 'Interpreter', 'none');
% xline(0, '--w', 'LineWidth', 1);

% subplot(1,2,2);
% imagesc(EEG_ERP.times, 1:size(EEG_ERP.data,3), ...
%     squeeze(EEG_ERP.data(demo_ch, :, :))');
% colorbar; xlabel('Time (ms)'); ylabel('Trial');
% title(sprintf('Post-ICA: %s', EEG_ERP.chanlocs(demo_ch).labels), 'Interpreter', 'none');
% xline(0, '--w', 'LineWidth', 1);

% % 统一色标
% cax1 = subplot(1,2,1); cax2 = subplot(1,2,2);
% clim1 = caxis(cax1); clim2 = caxis(cax2);
% common_clim = [min(clim1(1), clim2(1)), max(clim1(2), clim2(2))];
% caxis(cax1, common_clim); caxis(cax2, common_clim);

% sgtitle('ICA Denoising Comparison: Single-Trial Heatmap', 'FontSize', 14, 'FontWeight', 'bold');
% saveas(gcf, fullfile(data_path, 'ICA_comparison_heatmap.png'));
% fprintf('图 2 已保存: ICA_comparison_heatmap.png\n');

% % --- 10.4 图 3：全局 RMS 功率对比 ---
% figure('Name', 'ICA 前后全局 RMS 对比', 'NumberTitle', 'off', ...
%     'Position', [150, 150, 800, 400], 'Color', 'w');
% hold on;

% % 计算跨通道跨试次的 RMS
% rms_pre  = sqrt(mean(mean(EEG_preICA_proc.data.^2, 1), 3));
% rms_post = sqrt(mean(mean(EEG_ERP.data.^2, 1), 3));

% plot(EEG_ERP.times, rms_pre,  'Color', [0.7 0.2 0.2], 'LineWidth', 1.5);
% plot(EEG_ERP.times, rms_post, 'Color', [0.1 0.4 0.8], 'LineWidth', 1.5);
% xline(0, '--k', 'LineWidth', 0.8);
% xlabel('Time (ms)');
% ylabel('RMS Amplitude (\muV)');
% title('ICA Denoising Comparison: Global RMS Power', 'FontSize', 14, 'FontWeight', 'bold');
% legend('Pre-ICA', 'Post-ICA', 'Location', 'best');
% box on;
% hold off;
% saveas(gcf, fullfile(data_path, 'ICA_comparison_RMS.png'));
% fprintf('图 3 已保存: ICA_comparison_RMS.png\n');

% fprintf('\n✅ ICA 前后对比图绘制完毕！图片已保存至: %s\n', data_path);

% % 释放 ICA 前备份数据节省内存
% clear EEG_preICA EEG_preICA_proc;

function [bad_epochs, stats] = detect_bad_epochs_seeg(EEG)
% DETECT_BAD_EPOCHS_SEEG 使用稳健统计量识别 SEEG 中的异常试次。
% 规则:
%   1. 对每个通道 across trials 计算 peak-to-peak / RMS / line-length 的 robust z-score
%   2. 若单试次有较多通道在任一指标上异常，则标记该试次
%   3. 同时监控 trial-level 的跨道中位统计量，捕捉全局异常试次

data = double(EEG.data);  % [Ch, Time, Trial]
[n_ch, ~, n_trials] = size(data);

stats = struct( ...
    'bad_channel_fraction', zeros(1, n_trials), ...
    'bad_channel_count', zeros(1, n_trials), ...
    'trial_p2p_rz', zeros(1, n_trials), ...
    'trial_rms_rz', zeros(1, n_trials), ...
    'trial_line_rz', zeros(1, n_trials));

if n_trials < 5
    bad_epochs = [];
    return;
end

peak_to_peak = squeeze(max(data, [], 2) - min(data, [], 2));    % [Ch, Trial]
rms_val = squeeze(sqrt(mean(data .^ 2, 2)));                    % [Ch, Trial]
line_len = squeeze(mean(abs(diff(data, 1, 2)), 2));             % [Ch, Trial]

if isvector(peak_to_peak), peak_to_peak = reshape(peak_to_peak, n_ch, []); end
if isvector(rms_val), rms_val = reshape(rms_val, n_ch, []); end
if isvector(line_len), line_len = reshape(line_len, n_ch, []); end

z_p2p = robust_abs_zscore(peak_to_peak, 2);
z_rms = robust_abs_zscore(rms_val, 2);
z_line = robust_abs_zscore(line_len, 2);

chan_bad = (z_p2p > 5) | (z_rms > 5) | (z_line > 5);

bad_channel_fraction = mean(chan_bad, 1);
bad_channel_count = sum(chan_bad, 1);

trial_p2p = median(peak_to_peak, 1);
trial_rms = median(rms_val, 1);
trial_line = median(line_len, 1);

trial_p2p_rz = robust_abs_zscore(trial_p2p, 2);
trial_rms_rz = robust_abs_zscore(trial_rms, 2);
trial_line_rz = robust_abs_zscore(trial_line, 2);

min_bad_channels = max(2, ceil(0.05 * n_ch));
bad_trial_mask = ...
    (bad_channel_fraction >= 0.15) | ...
    (bad_channel_count >= min_bad_channels) | ...
    (trial_p2p_rz > 5) | ...
    (trial_rms_rz > 5) | ...
    (trial_line_rz > 5);

bad_epochs = find(bad_trial_mask);

stats.bad_channel_fraction = bad_channel_fraction;
stats.bad_channel_count = bad_channel_count;
stats.trial_p2p_rz = trial_p2p_rz;
stats.trial_rms_rz = trial_rms_rz;
stats.trial_line_rz = trial_line_rz;
end

function z = robust_abs_zscore(x, dim)
if nargin < 2
    dim = 1;
end
med_val = median(x, dim);
mad_val = mad(x, 1, dim);
z = 0.6745 * abs(x - med_val) ./ (mad_val + eps);
end
