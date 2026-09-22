
% 1. 设置文件夹路径（如果是当前目录则设为 '.'）
folderPath = './'; 



%%%%%%%%%%---------重要，记得改数字-----------%%%%%%%%%%%%%%%%%%%%%%%
% 1. 设置文件匹配模式 (先宽泛地抓取)
filePattern = fullfile(folderPath, '*_Task3*.mat');
rawFileList = dir(filePattern);

% 2. 严格过滤：只保留“数字_task1”格式的文件
fileList = [];
for k = 1:length(rawFileList)
    fileName = rawFileList(k).name;
    
    % 使用正则表达式检查：^\d+ 表示以一个或多个数字开头，接着是 _task1
    if ~isempty(regexp(fileName, '\d+_Task3', 'once'))
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
% 1. 定义两个部分的类别字典

% task1
category1 = {'face', 'object', 'body', 'place'}; 
category2 = {'color', 'gray'};

% task2
% category1 = {'Red', 'Yellow', 'Blue', 'Green', 'Black','White'}; 
% category2 = {'color', 'gray'};

% 2. 初始化一个 4x2 的 cell 数组
groupedData = cell(length(category1), length(category2));

% 3. 安全提取文件名列表 (兼容结构体数组和普通的 cell 数组)
if isstruct(allStimData) && length(allStimData) > 1
    % 如果 allStimData 是结构体数组，用大括号 {} 将其包裹成元胞数组
    fileList = {allStimData.filename}; 
else
    % 如果已经是 cell 或者字符矩阵，强制转为标准 cellstr
    fileList = cellstr(allStimData.filename);
end

% 4. 遍历文件名进行分类
for i = 1:numel(fileList)
    % 注意：这里改用了大括号 {} 来提取 cell 里面的字符向量
    fname = fileList{i}; 
    
    % 使用正则表达式提取信息
    tokens = regexp(fname, '^([a-zA-Z]+)_([a-zA-Z]+)_(\d+)', 'tokens', 'once');
    
    if ~isempty(tokens)
        part1 = tokens{1}; 
        part2 = tokens{2}; 
        numSeq = str2double(tokens{3}); 
        
        % 查找索引位置
        rowIdx = find(strcmp(category1, part1));
        colIdx = find(strcmp(category2, part2));
        
        % 分配数据
        if ~isempty(rowIdx) && ~isempty(colIdx)
            groupedData{rowIdx, colIdx}(end+1) = numSeq; 
        end
    end
end

% 5. 升序排序
for r = 1:size(groupedData, 1)
    for c = 1:size(groupedData, 2)
        groupedData{r, c} = sort(groupedData{r, c});
    end
end
save('groupedData.mat','groupedData')