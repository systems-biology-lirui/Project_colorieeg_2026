% test_consec_5pts.m
clear; clc;
tab_dir = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/result/tables';

run('test_comprehensive_near_sig.m');

% 测试 >= 5 点 (100ms)
k_t2c_5 = t2c_keys(t2c_strict == 1 | t2c_max_pts >= 5);
k_t2d_5 = t2d_keys(t2d_strict == 1 | t2d_max_pts >= 5);
k_t3c_5 = t3c_keys(t3c_strict == 1 | t3c_max_pts >= 5);
k_t3d_5 = t3d_keys(t3d_strict == 1 | t3d_max_pts >= 5);
k_xt_5  = xt_keys(xt_trend == 1);

fprintf('\n========================================================================\n');
fprintf('  【以 >=5点 (100ms) 作为近显著门槛时的交集】\n');
fprintf('========================================================================\n');
fprintf('Fig 1 (Cross): A=%d, B=%d, C=%d\n', numel(k_t2c_5), numel(k_t3c_5), numel(k_xt_5));
fprintf('  A & B: %d\n', numel(intersect(k_t2c_5, k_t3c_5)));
fprintf('  A & C: %d\n', numel(intersect(k_t2c_5, k_xt_5)));
fprintf('  B & C: %d\n', numel(intersect(k_t3c_5, k_xt_5)));
fprintf('  A & B & C: %d\n', numel(intersect(intersect(k_t2c_5, k_t3c_5), k_xt_5)));

fprintf('Fig 2 (Direct): A=%d, B=%d, C=%d\n', numel(k_t2d_5), numel(k_t3d_5), numel(k_xt_5));
fprintf('  A & B: %d\n', numel(intersect(k_t2d_5, k_t3d_5)));
fprintf('  A & C: %d\n', numel(intersect(k_t2d_5, k_xt_5)));
fprintf('  B & C: %d\n', numel(intersect(k_t3d_5, k_xt_5)));
fprintf('  A & B & C: %d\n', numel(intersect(intersect(k_t2d_5, k_t3d_5), k_xt_5)));
