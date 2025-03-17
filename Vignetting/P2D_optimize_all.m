clc; clear; close all;

%% 读取 EXR 图像
img = exrread('E:\sony_pictures\Vignetting_2_merge\48_52_56_merge_focus_distance_100cm_real_distance_20cm.exr');  % 读取 HDR 图像
[H, W, C] = size(img);

if C ~= 3
    error('输入的 EXR 文件必须是三通道 (RGB) 图像');
end

%% **设定多项式阶数**
S = 5;  % 2 阶双变量多项式拟合

%% **构造参数文件名称**
param_file = sprintf('bivariate_polyfit_S%d_P2D.mat', S);

if exist(param_file, 'file')
    fprintf('找到已有的优化参数文件 (%s)，直接加载...\n', param_file);
    load(param_file, 'models_RGB', 'error_RGB', 'max_values');
else
    %% **生成 x, y 网格坐标**
    [x_grid, y_grid] = meshgrid(1:W, 1:H);
    x_flat = x_grid(:);  % 展平为列向量
    y_flat = y_grid(:);

    %% **对 R/G/B 通道分别进行拟合**
    models_RGB = cell(1,3);  % 存储 R/G/B 通道的拟合模型
    error_RGB = zeros(1,3);  % 存储 R/G/B 通道的 MSE 误差
    max_values = zeros(3,3); % 存储 (x_max, y_max, V_max)

    for c = 1:3
        fprintf('拟合通道 %d 中...\n', c);

        % 获取当前通道的像素值，并展平
        V = img(:,:,c);
        V = double(V(:));

        % **多元多项式拟合**
        model = polyfitn([x_flat, y_flat], V, S);
        models_RGB{c} = model;

        % **计算拟合误差**
        V_fit = polyvaln(model, [x_flat, y_flat]);
        error_RGB(c) = mean((V - V_fit).^2);

        % **找到最大值及其位置**
        [V_max, idx_max] = max(V_fit);
        x_max = x_flat(idx_max);
        y_max = y_flat(idx_max);
        max_values(c, :) = [x_max, y_max, V_max];

        fprintf('通道 %d: 最大值 V_max = %.4f, (x_max, y_max) = (%.2f, %.2f), MSE: %e\n', ...
                c, V_max, x_max, y_max, error_RGB(c));
    end

    %% **优化完成后保存参数**
    save(param_file, 'models_RGB', 'error_RGB', 'max_values');
    fprintf('优化完成，参数已保存至 %s。\n', param_file);
end

%% **绘制 3D 曲面图**
fprintf('开始绘制 3D 结果...\n');

figure;
colors = {'r', 'g', 'b'}; % R/G/B 颜色
rgb_colors = [1 0 0;  % 红色
              0 1 0;  % 绿色
              0 0 1]; % 蓝色
titles = {'Red Channel', 'Green Channel', 'Blue Channel'};

[x_grid, y_grid] = meshgrid(1:W, 1:H);  % 重新生成完整坐标网格
x_flat = x_grid(:);
y_flat = y_grid(:);

for c = 1:3
    subplot(1,3,c);
    
    % **原始 HDR 数据采样 (每 500 个点采一个)**
    sample_idx_hdr = 1:500:length(x_flat);
    x_sample_hdr = x_flat(sample_idx_hdr);
    y_sample_hdr = y_flat(sample_idx_hdr);
    V_sample_hdr = double(img(:,:,c));
    V_sample_hdr = V_sample_hdr(:);
    V_sample_hdr = V_sample_hdr(sample_idx_hdr);

    % **创建均匀稀疏网格**
    gap = 1;
    x_fit = linspace(1, W, round(W/gap));  % 在 W 方向均匀采样
    y_fit = linspace(1, H, round(H/gap));  % 在 H 方向均匀采样
    [x_grid_fit, y_grid_fit] = meshgrid(x_fit, y_fit);

    % **计算稀疏采样的拟合曲面**
    V_sample_fit = polyvaln(models_RGB{c}, [x_grid_fit(:), y_grid_fit(:)]);
    V_grid_fit = reshape(V_sample_fit, size(x_grid_fit));

    % **绘制 HDR 采样数据 (淡灰色小点)**
    scatter3(x_sample_hdr, y_sample_hdr, V_sample_hdr, 3, rgb_colors(c, :), 'filled');
    hold on;

    % **绘制拟合曲面 (200 采样点)**
    surf(x_grid_fit, y_grid_fit, V_grid_fit, 'EdgeColor', 'none', 'FaceAlpha', 0.8);
    
    % **绘制最大值点**
    scatter3(max_values(c,1), max_values(c,2), max_values(c,3), 100, 'k', 'p', 'filled');

    title(sprintf('%s (MSE: %.2e)', titles{c}, error_RGB(c)));
    xlabel('X'); ylabel('Y'); zlabel('Intensity');
    view(10,10); % 3D 视角
    axis tight;
    grid on;

    % **调整 legend 到图的下方**
    legend({'HDR Sampled Data', 'Fitted Surface', 'Max Value'}, 'Location', 'southoutside');
end

fprintf('绘制完成！\n');