function preprocess_luminance_alignment(input_folder)
% PREPROCESS_LUMINANCE_ALIGNMENT Performs luminance alignment on images.
%   PREPROCESS_LUMINANCE_ALIGNMENT(INPUT_FOLDER) processes images in the
%   specified folder. If INPUT_FOLDER is not provided, a dialog box allows
%   selection.
%
%   The function performs the following steps:
%   1. Calculates Global Mean Luminance across all images.
%   2. Generates luminance-aligned color images.
%   3. Generates corresponding grayscale images (with 'Color' -> 'Gray' in filename).
%   4. Generates isoluminant patches (Red, Yellow, Blue, Green, Gray).
%
%   Output:
%   Results are saved in a 'Normalized_Stimuli' folder within the input directory.

    % --- 0. Input Handling ---
    if nargin < 1 || isempty(input_folder)
        input_folder = uigetdir('', 'Select Input Folder containing Images');
        if input_folder == 0
            fprintf('User cancelled. Exiting.\n');
            return;
        end
    end

    % Define supported extensions
    exts = {'*.bmp', '*.jpg', '*.png'};
    image_files = [];
    for i = 1:length(exts)
        files = dir(fullfile(input_folder, exts{i}));
        image_files = [image_files; files]; %#ok<AGROW>
    end

    if isempty(image_files)
        error('No images found in the specified folder.');
    end

    num_images = length(image_files);
    fprintf('Found %d images in %s\n', num_images, input_folder);

    % Prepare Output Directory
    output_dir = fullfile(input_folder, 'Normalized_Stimuli');
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
        fprintf('Created output directory: %s\n', output_dir);
    else
        fprintf('Output directory already exists: %s\n', output_dir);
    end

    % --- Step 1: Calculate Global Mean Luminance ---
    fprintf('\n--- Step 1: Calculating Global Mean Luminance ---\n');
    
    total_L_sum = 0;
    total_pixel_count = 0;

    % Pre-allocate cell array to store Lab images to avoid reading twice if memory allows?
    % For robustness with large datasets, we will read twice (or read once and process if we calculated global mean differently).
    % Requirement says: "1. Read all images... 4. Calculate... Global_Mean_L". 
    % Then "Step 2... Traverse each image".
    % To be memory efficient, we loop twice.
    
    for k = 1:num_images
        filename = fullfile(input_folder, image_files(k).name);
        try
            img_rgb = imread(filename);
            img_double = im2double(img_rgb); % Convert to [0, 1]
            
            % Handle grayscale images if any (convert to RGB for consistency)
            if size(img_double, 3) == 1
                img_double = cat(3, img_double, img_double, img_double);
            end
            
            img_lab = rgb2lab(img_double);
            L_channel = img_lab(:, :, 1);
            
            total_L_sum = total_L_sum + sum(L_channel(:));
            total_pixel_count = total_pixel_count + numel(L_channel);
            
        catch ME
            fprintf('Error reading %s: %s\n', image_files(k).name, ME.message);
        end
        
        if mod(k, 10) == 0
            fprintf('Step 1 Progress: Scanned %d/%d images...\n', k, num_images);
        end
    end

    Global_Mean_L = total_L_sum / total_pixel_count;
    fprintf('Global Mean Luminance calculated: %.4f\n', Global_Mean_L);

    % --- Step 2 & 3: Generate Aligned Color & Grayscale Images ---
    fprintf('\n--- Step 2 & 3: Generating Aligned Images ---\n');

    for k = 1:num_images
        filename = image_files(k).name;
        full_path = fullfile(input_folder, filename);
        
        try
            img_rgb = imread(full_path);
            img_double = im2double(img_rgb);
             if size(img_double, 3) == 1
                img_double = cat(3, img_double, img_double, img_double);
            end
            
            img_lab = rgb2lab(img_double);
            L_old = img_lab(:, :, 1);
            img_mean_L = mean(L_old(:));
            
            % Step 2.3: Scale Factor
            if img_mean_L < 1e-6 % Avoid division by zero for black images
                ScaleFactor = 1;
            else
                ScaleFactor = Global_Mean_L / img_mean_L;
            end
            
            % Step 2.4: New L Channel
            L_new = L_old * ScaleFactor;
            
            % Construct aligned Lab image
            img_lab_aligned = img_lab;
            img_lab_aligned(:, :, 1) = L_new;
            
            % Step 2.5: Convert back to RGB
            img_rgb_aligned = lab2rgb(img_lab_aligned); % Output is double [0, 1] due to im2double input logic usually, check lab2rgb behavior. 
            % lab2rgb returns doubles in [0, 1] if input is double.
            
            % Step 2.6: Clip
            img_rgb_aligned(img_rgb_aligned > 1) = 1;
            img_rgb_aligned(img_rgb_aligned < 0) = 0;
            
            % Save Color Aligned Image
            output_filename_color = fullfile(output_dir, filename);
            imwrite(img_rgb_aligned, output_filename_color);
            
            % --- Step 3: Generate Grayscale Image ---
            % Step 3.2: Set a and b to 0
            img_lab_gray = img_lab_aligned;
            img_lab_gray(:, :, 2) = 0; % a channel
            img_lab_gray(:, :, 3) = 0; % b channel
            
            % Step 3.3: Convert to RGB
            img_rgb_gray = lab2rgb(img_lab_gray);
            img_rgb_gray(img_rgb_gray > 1) = 1;
            img_rgb_gray(img_rgb_gray < 0) = 0;
            
            % Step 3.4: Modify Filename
            [~, name, ext] = fileparts(filename);
            % Case-insensitive replacement of 'Color' with 'Gray'
            new_name = regexprep(name, 'Color', 'Gray', 'ignorecase');
            
            % If 'Color' wasn't in the name, append '_Gray' to avoid overwriting or confusion? 
            % User rule: "replace... 'Color'... with 'Gray'". 
            % If 'Color' is missing, the name remains same. Let's assume user inputs follow convention or accept unchanged name (but risk overwrite if logic was different).
            % However, if filename doesn't have 'Color', we should probably append '_Gray' to distinguish?
            % The prompt specifically says: "Face_Color_01.bmp -> Face_Gray_01.bmp".
            % If the user provides "Face_01.bmp", regexprep returns "Face_01". 
            % Saving "Face_01.bmp" (grayscale) would overwrite "Face_01.bmp" (color) if we were saving in same folder, 
            % but we are saving in `Normalized_Stimuli`.
            % However, we just saved the Color version there too!
            % So if "Color" is NOT in the filename, we have a collision.
            % Let's add a safety check: if name hasn't changed, append '_Gray'.
            
            if strcmpi(name, new_name)
                 % Pattern 'Color' not found. 
                 % To ensure distinction, let's append _Gray.
                 % (User prompt didn't strictly specify this case, but it's good practice).
                 % STRICT ADHERENCE: "文件名修改规则：将原文件名中的字符串 'Color' (不区分大小写) 替换为 'Gray'。"
                 % If I strictly follow, I do nothing else. 
                 % But "Face_01.bmp" -> Color Aligned saved as "Face_01.bmp".
                 % Then Grayscale saved as "Face_01.bmp". This overwrites the color one.
                 % I will append _Gray if replacement didn't happen to prevent data loss.
                 new_name = [name, '_Gray'];
            end
            
            output_filename_gray = fullfile(output_dir, [new_name, ext]);
            imwrite(img_rgb_gray, output_filename_gray);
            
            fprintf('Processed %d/%d: %s\n', k, num_images, filename);
            
        catch ME
            fprintf('Error processing %s: %s\n', filename, ME.message);
        end
    end

    % --- Step 4: Generate Isoluminant Patches ---
    fprintf('\n--- Step 4: Generating Isoluminant Patches ---\n');
    
    % Define colors: Name -> RGB Base (Approximation)
    % Using standard primaries.
    patch_names = {'blue_color_01.bmp', 'red_color_01.bmp', 'green_color_01.bmp', 'yellow_color_01.bmp', 'gray_color_01.bmp'};
    
    % RGB Bases (0-1)
    % Blue: [0 0 1], Red: [1 0 0], Green: [0 1 0], Yellow: [1 1 0], Gray: [0.5 0.5 0.5] (will be overridden)
    base_colors_rgb = {
        [0, 0, 1];   % Blue
        [1, 0, 0];   % Red
        [0, 1, 0];   % Green
        [1, 1, 0];   % Yellow
        [0.5, 0.5, 0.5] % Gray (placeholder)
    };
    
    % Image size for patches (e.g., 256x256 or average of input images?)
    % User didn't specify size. Let's pick a standard reasonable size, e.g., 512x512.
    patch_size = [512, 512]; 
    
    for i = 1:length(patch_names)
        base_rgb = base_colors_rgb{i};
        
        % Convert base RGB to Lab to get target a, b
        % Note: We create a 1x1 pixel to calculate
        pixel_rgb = reshape(base_rgb, [1, 1, 3]);
        pixel_lab = rgb2lab(pixel_rgb);
        
        % Algorithm Step 4.2
        if i == 5 % Gray
             % For gray: Lab = [Global_Mean_L, 0, 0]
             target_L = Global_Mean_L;
             target_a = 0;
             target_b = 0;
        else
             % For colors: Force L to Global_Mean_L, keep a, b
             target_L = Global_Mean_L;
             target_a = pixel_lab(1, 1, 2);
             target_b = pixel_lab(1, 1, 3);
        end
        
        % Create the full image
        % Construct Lab image
        patch_lab = zeros([patch_size, 3]);
        patch_lab(:, :, 1) = target_L;
        patch_lab(:, :, 2) = target_a;
        patch_lab(:, :, 3) = target_b;
        
        % Convert back to RGB
        patch_rgb = lab2rgb(patch_lab);
        
        % Check for gamut warning
        % lab2rgb output is [0, 1]. If values were clipped by lab2rgb automatically?
        % MATLAB's lab2rgb might output values outside [0,1] if not clipped?
        % Actually lab2rgb documentation says: "Values in RGB are in the range [0, 1] or [0, 255]..."
        % But if the Lab value is out of gamut, the RGB values might be <0 or >1 before casting if we are careful?
        % Wait, if I use `lab2rgb` without specifying output type, it returns double.
        % Does it clip?
        % Let's check bounds.
        
        max_val = max(patch_rgb(:));
        min_val = min(patch_rgb(:));
        
        gamut_warning = false;
        if max_val > 1.0001 || min_val < -0.0001
             gamut_warning = true;
             fprintf('Warning: Gamut clipping occurred for %s (Range: %.2f to %.2f)\n', patch_names{i}, min_val, max_val);
        end
        
        % Clip
        patch_rgb(patch_rgb > 1) = 1;
        patch_rgb(patch_rgb < 0) = 0;
        
        % Save
        output_path = fullfile(output_dir, patch_names{i});
        imwrite(patch_rgb, output_path);
        fprintf('Generated patch: %s\n', patch_names{i});
    end

    fprintf('\nProcessing Complete! Results saved in: %s\n', output_dir);
    fprintf('Final Global Mean Luminance: %.4f\n', Global_Mean_L);

end
