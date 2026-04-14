%% ============================================================
%  Analysis of orientation vs pattern distances
%  data size: [18 orientation, 6 pattern, 96 channel, 200 time]
%  Author: ChatGPT for your experiment
%% ============================================================

clear; clc;

%% ===================== 0. Input data ========================
% Replace this with your real data variable
% data: [nOri, nPat, nCh, nTime]
% Example:
% load('your_data.mat');   % should contain variable "data"

assert(exist('data', 'var') == 1, 'Variable "data" not found in workspace.');

[nOri, nPat, nCh, nTime] = size(data);
fprintf('Data size = [%d ori, %d pat, %d ch, %d time]\n', nOri, nPat, nCh, nTime);

%% ===================== 1. Reshape data ======================
% Convert each condition (orientation, pattern) into one row vector
% X: [nCond, nFeatures], where nCond = nOri*nPat, nFeatures = nCh*nTime

nCond = nOri * nPat;
nFeat = nCh * nTime;

X = zeros(nCond, nFeat);
oriLabel = zeros(nCond, 1);
patLabel = zeros(nCond, 1);

row = 0;
for o = 1:nOri
    for p = 1:nPat
        row = row + 1;
        tmp = squeeze(data(o, p, :, :));   % [96, 200]
        X(row, :) = tmp(:)';               % flatten to 1 x (96*200)
        oriLabel(row) = o;
        patLabel(row) = p;
    end
end

% Optional: z-score each feature across conditions
% This is often recommended so that large-variance features do not dominate.
Xz = zscore(X, 0, 1);
Xz(:, std(X,0,1)==0) = 0;   % avoid NaN if some features have zero variance

fprintf('Reshaped X size = [%d conditions, %d features]\n', size(Xz,1), size(Xz,2));

%% ===================== 2. All-condition distance matrix ======================
% Direct distance in the ORIGINAL high-dimensional space
D_raw = squareform(pdist(Xz, 'euclidean'));   % [108 x 108]

figure;
imagesc(D_raw);
axis image;
colorbar;
title('All-condition Euclidean distance matrix (raw high-dimensional space)');
xlabel('Condition');
ylabel('Condition');

%% ============================================================
% 3. Two ways to compare orientation vs pattern
%    A. "Centroid distance"  -> first average, then compute distance
%    B. "Controlled pairwise distance" -> do NOT average first
%% ============================================================

%% ===================== 3A. Centroid distance ======================
% ---- Orientation centroid: average across patterns ----
OriCent = zeros(nOri, nFeat);
for o = 1:nOri
    idx = oriLabel == o;
    OriCent(o, :) = mean(Xz(idx, :), 1);
end
D_ori_centroid = squareform(pdist(OriCent, 'euclidean'));

% ---- Pattern centroid: average across orientations ----
PatCent = zeros(nPat, nFeat);
for p = 1:nPat
    idx = patLabel == p;
    PatCent(p, :) = mean(Xz(idx, :), 1);
end
D_pat_centroid = squareform(pdist(PatCent, 'euclidean'));

% Mean upper triangle distances
mean_ori_centroid = mean_upper_triangle(D_ori_centroid);
mean_pat_centroid = mean_upper_triangle(D_pat_centroid);

fprintf('\n=== Centroid distances (first average, then distance) ===\n');
fprintf('Mean orientation-centroid distance = %.4f\n', mean_ori_centroid);
fprintf('Mean pattern-centroid distance     = %.4f\n', mean_pat_centroid);

figure;
subplot(1,2,1);
imagesc(D_ori_centroid);
axis image; colorbar;
title('Orientation centroid distance');
xlabel('Orientation'); ylabel('Orientation');

subplot(1,2,2);
imagesc(D_pat_centroid);
axis image; colorbar;
title('Pattern centroid distance');
xlabel('Pattern'); ylabel('Pattern');

%% ===================== 3B. Controlled pairwise distance ======================
% This is usually more interpretable:
%
% Orientation distance:
%   compare o1 vs o2 while MATCHING the same pattern
%   then average across patterns
%
% Pattern distance:
%   compare p1 vs p2 while MATCHING the same orientation
%   then average across orientations

D_ori_ctrl = nan(nOri, nOri, nPat);   % for each pattern, orientation distance matrix
for p = 1:nPat
    for o1 = 1:nOri
        idx1 = cond_index(o1, p, nPat);
        for o2 = 1:nOri
            idx2 = cond_index(o2, p, nPat);
            D_ori_ctrl(o1, o2, p) = norm(Xz(idx1,:) - Xz(idx2,:));
        end
    end
end
D_ori_ctrl_mean = mean(D_ori_ctrl, 3);   % average across patterns

D_pat_ctrl = nan(nPat, nPat, nOri);   % for each orientation, pattern distance matrix
for o = 1:nOri
    for p1 = 1:nPat
        idx1 = cond_index(o, p1, nPat);
        for p2 = 1:nPat
            idx2 = cond_index(o, p2, nPat);
            D_pat_ctrl(p1, p2, o) = norm(Xz(idx1,:) - Xz(idx2,:));
        end
    end
end
D_pat_ctrl_mean = mean(D_pat_ctrl, 3);   % average across orientations

mean_ori_ctrl = mean_upper_triangle(D_ori_ctrl_mean);
mean_pat_ctrl = mean_upper_triangle(D_pat_ctrl_mean);

fprintf('\n=== Controlled pairwise distances (NO averaging first) ===\n');
fprintf('Mean orientation distance (matched pattern) = %.4f\n', mean_ori_ctrl);
fprintf('Mean pattern distance (matched orientation) = %.4f\n', mean_pat_ctrl);

figure;
subplot(1,2,1);
imagesc(D_ori_ctrl_mean);
axis image; colorbar;
title('Orientation distance (matched pattern, averaged over pattern)');
xlabel('Orientation'); ylabel('Orientation');

subplot(1,2,2);
imagesc(D_pat_ctrl_mean);
axis image; colorbar;
title('Pattern distance (matched orientation, averaged over orientation)');
xlabel('Pattern'); ylabel('Pattern');

%% ===================== 3C. Distribution comparison ======================
% Extract all unique pairwise distances for comparison
ori_ctrl_vals = upper_triangle_values(D_ori_ctrl_mean);
pat_ctrl_vals = upper_triangle_values(D_pat_ctrl_mean);

ori_cent_vals = upper_triangle_values(D_ori_centroid);
pat_cent_vals = upper_triangle_values(D_pat_centroid);

figure;
subplot(1,2,1);
boxplot([ori_cent_vals(:), pat_cent_vals(:)], ...
    'Labels', {'Ori centroid', 'Pat centroid'});
ylabel('Distance');
title('First average, then compute distance');

subplot(1,2,2);
boxplot([ori_ctrl_vals(:), pat_ctrl_vals(:)], ...
    'Labels', {'Ori matched-pattern', 'Pat matched-orientation'});
ylabel('Distance');
title('Controlled pairwise distance');

%% ============================================================
% 4. PCA comparison
%    A. top 3 PCs
%    B. all PCs
%% ============================================================

% Since Xz is already z-scored, no need for extra centering inside PCA
[coeff, score, latent, ~, explained, mu] = pca(Xz, 'Centered', false);

cumExplained = cumsum(explained);

figure;
plot(cumExplained, 'o-', 'LineWidth', 1.5);
xlabel('Number of PCs');
ylabel('Cumulative explained variance (%)');
title('PCA cumulative explained variance');
grid on;

fprintf('\n=== PCA explained variance ===\n');
fprintf('PC1-3 cumulative explained variance = %.2f %%\n', cumExplained(min(3, numel(cumExplained))));
fprintf('PC1-10 cumulative explained variance = %.2f %%\n', cumExplained(min(10, numel(cumExplained))));

% ----- Distance in top 3 PCs -----
score3 = score(:, 1:min(3,size(score,2)));
D_pca3 = squareform(pdist(score3, 'euclidean'));

% ----- Distance in all PCs -----
scoreAll = score;    % all PCs
D_pcaAll = squareform(pdist(scoreAll, 'euclidean'));

% Check equivalence between raw space and all-PC space
diff_raw_allpc = max(abs(D_raw(:) - D_pcaAll(:)));
fprintf('Max abs difference between raw-space distance and all-PC distance = %.10f\n', diff_raw_allpc);

figure;
subplot(1,2,1);
imagesc(D_pca3);
axis image; colorbar;
title('Distance matrix in top 3 PCs');

subplot(1,2,2);
imagesc(D_pcaAll);
axis image; colorbar;
title('Distance matrix in all PCs');

%% ===================== 4A. Controlled distances in top 3 PCs ======================
D_ori_ctrl_pca3 = nan(nOri, nOri, nPat);
for p = 1:nPat
    for o1 = 1:nOri
        idx1 = cond_index(o1, p, nPat);
        for o2 = 1:nOri
            idx2 = cond_index(o2, p, nPat);
            D_ori_ctrl_pca3(o1,o2,p) = norm(score3(idx1,:) - score3(idx2,:));
        end
    end
end
D_ori_ctrl_pca3_mean = mean(D_ori_ctrl_pca3, 3);

D_pat_ctrl_pca3 = nan(nPat, nPat, nOri);
for o = 1:nOri
    for p1 = 1:nPat
        idx1 = cond_index(o, p1, nPat);
        for p2 = 1:nPat
            idx2 = cond_index(o, p2, nPat);
            D_pat_ctrl_pca3(p1,p2,o) = norm(score3(idx1,:) - score3(idx2,:));
        end
    end
end
D_pat_ctrl_pca3_mean = mean(D_pat_ctrl_pca3, 3);

mean_ori_ctrl_pca3 = mean_upper_triangle(D_ori_ctrl_pca3_mean);
mean_pat_ctrl_pca3 = mean_upper_triangle(D_pat_ctrl_pca3_mean);

fprintf('\n=== Controlled distances in top 3 PCs ===\n');
fprintf('Mean orientation distance in PC1-3 = %.4f\n', mean_ori_ctrl_pca3);
fprintf('Mean pattern distance in PC1-3     = %.4f\n', mean_pat_ctrl_pca3);

%% ===================== 4B. Controlled distances in ALL PCs ======================
D_ori_ctrl_pcaAll = nan(nOri, nOri, nPat);
for p = 1:nPat
    for o1 = 1:nOri
        idx1 = cond_index(o1, p, nPat);
        for o2 = 1:nOri
            idx2 = cond_index(o2, p, nPat);
            D_ori_ctrl_pcaAll(o1,o2,p) = norm(scoreAll(idx1,:) - scoreAll(idx2,:));
        end
    end
end
D_ori_ctrl_pcaAll_mean = mean(D_ori_ctrl_pcaAll, 3);

D_pat_ctrl_pcaAll = nan(nPat, nPat, nOri);
for o = 1:nOri
    for p1 = 1:nPat
        idx1 = cond_index(o, p1, nPat);
        for p2 = 1:nPat
            idx2 = cond_index(o, p2, nPat);
            D_pat_ctrl_pcaAll(p1,p2,o) = norm(scoreAll(idx1,:) - scoreAll(idx2,:));
        end
    end
end
D_pat_ctrl_pcaAll_mean = mean(D_pat_ctrl_pcaAll, 3);

mean_ori_ctrl_pcaAll = mean_upper_triangle(D_ori_ctrl_pcaAll_mean);
mean_pat_ctrl_pcaAll = mean_upper_triangle(D_pat_ctrl_pcaAll_mean);

fprintf('\n=== Controlled distances in ALL PCs ===\n');
fprintf('Mean orientation distance in all PCs = %.4f\n', mean_ori_ctrl_pcaAll);
fprintf('Mean pattern distance in all PCs     = %.4f\n', mean_pat_ctrl_pcaAll);

%% ===================== 5. Simple summary ======================
fprintf('\n================ SUMMARY ================\n');
fprintf('Raw space, centroid method:\n');
fprintf('  Orientation = %.4f\n', mean_ori_centroid);
fprintf('  Pattern     = %.4f\n', mean_pat_centroid);

fprintf('Raw space, controlled method:\n');
fprintf('  Orientation = %.4f\n', mean_ori_ctrl);
fprintf('  Pattern     = %.4f\n', mean_pat_ctrl);

fprintf('Top-3 PC, controlled method:\n');
fprintf('  Orientation = %.4f\n', mean_ori_ctrl_pca3);
fprintf('  Pattern     = %.4f\n', mean_pat_ctrl_pca3);

fprintf('All-PC, controlled method:\n');
fprintf('  Orientation = %.4f\n', mean_ori_ctrl_pcaAll);
fprintf('  Pattern     = %.4f\n', mean_pat_ctrl_pcaAll);

if mean_pat_ctrl > mean_ori_ctrl
    fprintf('\nResult in RAW space: pattern distance > orientation distance\n');
else
    fprintf('\nResult in RAW space: orientation distance >= pattern distance\n');
end

if mean_pat_ctrl_pca3 > mean_ori_ctrl_pca3
    fprintf('Result in top-3 PCA space: pattern distance > orientation distance\n');
else
    fprintf('Result in top-3 PCA space: orientation distance >= pattern distance\n');
end

if abs(mean_pat_ctrl_pcaAll - mean_pat_ctrl) < 1e-8 && abs(mean_ori_ctrl_pcaAll - mean_ori_ctrl) < 1e-8
    fprintf('All-PC distances are effectively equivalent to raw-space distances.\n');
else
    fprintf('All-PC distances differ slightly from raw-space distances (check preprocessing).\n');
end

%% ===================== local functions ======================
function idx = cond_index(o, p, nPat)
    % Map (orientation, pattern) -> row index in X
    idx = (o - 1) * nPat + p;
end

function m = mean_upper_triangle(D)
    % Mean of upper triangle excluding diagonal
    mask = triu(true(size(D)), 1);
    vals = D(mask);
    m = mean(vals);
end

function vals = upper_triangle_values(D)
    mask = triu(true(size(D)), 1);
    vals = D(mask);
end
