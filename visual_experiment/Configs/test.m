% for i = 1:80
%     color_value(i,:) = squeeze(data.stimuli{i}(100,100,:));
% end
% color_value = double(color_value);
% save('Liulab_colorvalue.mat','color_value','intensities');
%% 1. 数据录入
figure;
% 这里输入你的16个点，x为RGB(0-255)，y为亮度值
color_type = {'red','green','blue','yellow','gray'};
for c = 1:length(color_type)
    fprintf('Fitting %s:\n',color_type{c});
    switch c
        case 1
            m = 1;
        case 2
            m = 2;
        case 3
            m = 3;
        case 4
            m = 1;
        case 5
            m = 1;
    end
    x = color_value([1:16]+(c-1)*16,m);
    y = intensities([1:16]+(c-1)*16,1); % 示例数据
    
    %% 2. 定义自定义拟合模型
    % 我们定义公式 a * (x/255)^g + b
    % 这样拟合出来的 a 直接就是最大亮度，g 就是 gamma 值
    ft = fittype('a * (x/255)^g + b', 'independent', 'x', 'dependent', 'y');
    
    %% 3. 设置初始猜测值 (StartPoints)
    % 这一步很重要，给算法一个起点，防止拟合失败
    % a 猜一个最大亮度 (比如 y的最大值)
    % g 猜一个标准gamma值 (比如 2.2)
    % b 猜一个最小亮度 (比如 y的最小值)
    opts = fitoptions(ft);
    opts.StartPoint = [max(y), 2.2, min(y)];
    opts.Lower = [0, 0, 0]; % 设置下限，参数不能为负
    
    %% 4. 执行拟合
    [fitresult, gof] = fit(x, y, ft, opts);
    
    %% 5. 查看结果
    disp('拟合参数：');
    disp(fitresult); % 这里会显示 a, g, b 的值
    disp(['R-square: ', num2str(gof.rsquare)]); % 检查拟合优度
    
    %% 6. 绘图验证
    subplot(1,5,c)
    plot(fitresult, x, y);
    title('Gamma Curve Fitting');
    xlabel('RGB Input (0-255)');
    ylabel('Luminance (cd/m^2)');
    grid on;
    
    %% --- 关键步骤：反向计算 ---
    % 假设你确定的 Target Luminance 是 30
    L_target = 8;
    
    % 获取拟合参数
    a_fit = fitresult.a;
    g_fit = fitresult.g;
    b_fit = fitresult.b;
    
    % 公式逆运算： Input = 255 * ((L - b) / a)^(1/g)
    required_RGB = round(255 * ((L_target - b_fit) / a_fit)^(1/g_fit));
    color_RGB(c) = required_RGB;
    disp(['为了达到亮度 ', num2str(L_target), color_type{c},'需要的 RGB 值为: ', num2str(required_RGB)]);
end
