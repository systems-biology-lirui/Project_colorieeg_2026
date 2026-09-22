% test_lookup_sig_channels_anatomy.m
% Lookup MNI coordinates and anatomical labels for significant cross-decoding electrodes

t_sig = {
    'sub001', 'B3';
    'sub003', 'C13';
    'sub006', 'E8';
    'sub006', 'H10';
    'sub008', 'C2';
    'sub008', 'C10';
    'sub008', 'E8'
};

csv_path = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/result/tables/all_recorded_electrodes.csv';
t_all = readtable(csv_path);

fprintf('--- SIGNIFICANT ELECTRODES ANATOMICAL SUMMARY ---\n');
for i = 1:size(t_sig, 1)
    sub = t_sig{i, 1};
    ch = t_sig{i, 2};
    idx = strcmp(t_all.subject, sub) & strcmp(t_all.channel, ch);
    if any(idx)
        row = t_all(idx, :);
        fprintf('%s - %s: MNI [%.1f, %.1f, %.1f], DKT: %s, ASEG: %s, AAL: %s\n', ...
            sub, ch, row.mni_x(1), row.mni_y(1), row.mni_z(1), ...
            string(row.dkt_anatomy(1)), string(row.aseg_anatomy(1)), string(row.aal_anatomy(1)));
    else
        fprintf('%s - %s: Not found in all_recorded_electrodes.csv\n', sub, ch);
    end
end
