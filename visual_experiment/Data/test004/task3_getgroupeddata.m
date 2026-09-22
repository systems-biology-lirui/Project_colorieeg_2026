
% 1. 设置文件夹路径（如果是当前目录则设为 '.'）
folderPath = './'; 



%%%%%%%%%%---------重要，记得改数字-----------%%%%%%%%%%%%%%%%%%%%%%%
% 1. 设置文件匹配模式 (先宽泛地抓取)
filePattern = fullfile(folderPath, '*_task2*.mat');
rawFileList = dir(filePattern);

% 2. 严格过滤：只保留“数字_task1”格式的文件
fileList = [];
for k = 1:length(rawFileList)
    fileName = rawFileList(k).name;
    
    % 使用正则表达式检查：^\d+ 表示以一个或多个数字开头，接着是 _task1
    % 去掉开头的 ^ 符号
    if ~isempty(regexp(fileName, '\d+_Task2', 'once'))
        % 如果匹配成功，加入到正式的 fileList 中
        fileList = [fileList; rawFileList(k)]; 
    end
end

% 接下来继续你原来的 第3步...

% 3. 初始化结果变量
allStimData = []; 

% 4. 循环处理
for k = 1:length(fileList)
    % 构建完整文件路径
    baseFileName = fileList(k).name;
    fullFileName = fullfile(fileList(k).folder, baseFileName);
    
    fprintf('正在处理文件: %s\n', baseFileName);
    
    % 5. 加载变量
    % 使用 'stimData' 指定只加载该变量，提高速度
    dataStruct = load(fullFileName, 'stimData');
    
    % 检查变量是否存在于文件中
    if isfield(dataStruct, 'stimData')
        currentData = dataStruct.stimData;
        
        % 6. 拼接结构体 (垂直拼接)
        % 假设每个文件都是 308x1，拼接后将变成 (308*k)x1
        allStimData = [allStimData; currentData]; 
    else
        warning('文件 %s 中未找到 stimData 变量。', baseFileName);
    end
end

% 7. 检查最终结果
disp('拼接完成！');
size(allStimData)
% ==========================================
% 第二部分：数据分类 (6x3 的 cell 数组)
% ==========================================

% 1. 定义行和列的类别
% 6种颜色作为行 (注意首字母大小写，后面用了 strcmpi 可以忽略大小写差异)
category_colors = {'Red', 'Yellow', 'Blue', 'Green', 'Black', 'White'}; 
% 3种形状作为列 (直接用数字)
category_shapes = [1, 2, 3]; 

% 2. 初始化一个 6x3 的 cell 数组
groupedData = cell(length(category_colors), length(category_shapes));

% 3. 安全提取文件名列表 (兼容结构体数组和普通的 cell 数组)
if isstruct(allStimData) && length(allStimData) > 1
    fileList = {allStimData.filename}; 
else
    fileList = cellstr(allStimData.filename);
end

% 4. 遍历文件名进行分类
for i = 1:numel(fileList)
    fname = fileList{i}; 
    
    % 使用正则表达式提取信息
    % '^([a-zA-Z]+)_([a-zA-Z]+)_(\d+)' 会将 'Blue_Color_03.bmp' 拆分为：
    % tokens{1} = 'Blue', tokens{2} = 'Color', tokens{3} = '03'
    tokens = regexp(fname, '^([a-zA-Z]+)_([a-zA-Z]+)_(\d+)', 'tokens', 'once');
    
    if ~isempty(tokens)
        colorStr = tokens{1};      % 颜色部分 (例如 'Blue')
        % middleStr = tokens{2};   % 中间部分 (例如 'Color'，这里不需要做分类)
        shapeNum = str2double(tokens{3}); % 数字部分 (例如 '03' 转为数字 3)
        
        % 查找索引位置 
        % 使用 strcmpi 忽略大小写，避免 'Blue' 和 'blue' 匹配失败
        rowIdx = find(strcmpi(category_colors, colorStr));
        % 查找数字对应的列索引
        colIdx = find(category_shapes == shapeNum);
        
        % 分配数据
        if ~isempty(rowIdx) && ~isempty(colIdx)
            % 【重要修改】：这里改为存入索引 i。
            % 这样 groupedData{rowIdx, colIdx} 里面就是属于这个类别所有数据的序号。
            % 后续如果你要提取完整的结构体数据，只需使用: allStimData(groupedData{row, col}) 即可。
            groupedData{rowIdx, colIdx}(end+1) = i; 
        end
    end
end

% 5. 升序排序 (对存入的索引进行排序，保证按时间或读取顺序排列)
for r = 1:size(groupedData, 1)
    for c = 1:size(groupedData, 2)
        groupedData{r, c} = sort(groupedData{r, c});
    end
end

% 6. 保存数据
save('task3groupedData.mat', 'groupedData');
disp('分类并保存完成！');