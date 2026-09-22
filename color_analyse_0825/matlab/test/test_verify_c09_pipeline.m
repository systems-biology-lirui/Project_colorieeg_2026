%% test_verify_c09_pipeline.m
% 验证 C09 批处理流水线的完整执行与汇总表导出

clear; clc;
script_dir = fileparts(mfilename('fullpath'));
matlab_dir = fileparts(script_dir);

% 运行 C09，但通过临时参数限制只跑 2 个典型通道进行完整流水线审计
fprintf('>>> 正在验证 C09 跨任务解码流水线 ...\n');

% 直接读取 C09 代码逻辑，用 custom 模式验证两个典型通道
c09_script = fullfile(matlab_dir, 'C09_cross_decoding_task3_to_task2_0825.m');

% 创建测试运行副本
test_cfg = struct();
test_cfg.target_mode   = 'custom';
test_cfg.custom_subs   = {'sub001', 'sub007'};
test_cfg.custom_elecs  = {'G13', 'C4'};
test_cfg.max_elecs     = 2;
test_cfg.skip_existing = false;
test_cfg.n_perm        = 50; % 快速验证用 50 次置换

% 动态执行测试
c09_text = fileread(c09_script);
% 替换默认配置为测试配置
c09_text = regexprep(c09_text, "cfg\.target_mode\s*=\s*'concordant';", "cfg.target_mode = 'custom';");
c09_text = regexprep(c09_text, "cfg\.max_elecs\s*=\s*Inf;", "cfg.max_elecs = 2;");
c09_text = regexprep(c09_text, "cfg\.n_perm\s*=\s*200;", "cfg.n_perm = 50;");
c09_text = regexprep(c09_text, "cfg\.skip_existing\s*=\s*true;", "cfg.skip_existing = false;");

test_runner_file = fullfile(script_dir, 'scratch', 'run_c09_temp_test.m');
if ~exist(fullfile(script_dir, 'scratch'), 'dir')
    mkdir(fullfile(script_dir, 'scratch'));
end

fid = fopen(test_runner_file, 'w');
fwrite(fid, c09_text);
fclose(fid);

fprintf('>>> 启动临时运行测试脚本: %s\n', test_runner_file);
run(test_runner_file);
fprintf('>>> C09 流水线验证完成!\n');
