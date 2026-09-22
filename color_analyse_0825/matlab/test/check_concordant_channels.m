%% check_concordant_channels.m
clear; clc;
root_dir = 'e:\liulab_project\Project_colorieeg_2026\color_analyse_0825';
c04_file = fullfile(root_dir, 'result', 'tables', 'color_effects_summary.mat');
d = load(c04_file);
tbl = d.all_tbl;
concord_mask = (tbl.is_significant == 1) & ...
    (strcmp(tbl.concordance_type, 'Concordant_Positive') | ...
     strcmp(tbl.concordance_type, 'Concordant_Negative'));
c04_sub = tbl(concord_mask, :);
elec_keys = strcat(c04_sub.subject, '_', c04_sub.channel);
[u_keys, u_ia] = unique(elec_keys, 'stable');
subs = c04_sub.subject(u_ia);
elecs = c04_sub.channel(u_ia);
fprintf('Total concordant unique channels: %d\n', numel(subs));
u_subs = unique(subs, 'stable');
for i = 1:numel(u_subs)
    s = u_subs{i};
    n_s = sum(strcmp(subs, s));
    fprintf('  %s: %d channels\n', s, n_s);
end
