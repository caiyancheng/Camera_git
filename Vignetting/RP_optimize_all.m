clc; clear; close all;

%% 读取 EXR 图像
img = exrread('E:\sony_pictures\Vignetting/merged_16_20_ARQ_mtf_focus_distance_100cm_real_distance_40cm.exr');  % 读取 HDR 图像
[H, W, C] = size(img);

if C ~= 3
    error('输入的 EXR 文件必须是三通道 (RGB) 图像');
end

%% 设定多项式阶数
S = 2; % 先拟合二阶多项式

%% 初始化参数
[x_grid, y_grid] = meshgrid(1:W, 1:H); % 生成坐标网格
eta = 1;  % 固定 η

%% 预分配参数存储
theta_opt_RGB = zeros(3, S + 3); % 存储 (x_c, y_c, p) 的优化参数

%% **1. 粗略搜索 (x_c, y_c) 初始值**
search_range = -100:20:100; % 在 (x_c_init, y_c_init) 附近进行搜索
best_xc = W/2;
best_yc = H/2;
best_err = Inf;

for dx = search_range
    for dy = search_range
        xc_try = W/2 + dx;
        yc_try = H/2 + dy;

        r_eta = sqrt((x_grid - xc_try).^2 + (y_grid - yc_try).^2);
        r_eta = double(r_eta(:));

        for c = 1:3
            V = img(:,:,c);
            V = double(V(:));

            % 拟合二阶多项式
            p_try = polyfit(r_eta, V, S);
            V_fit = polyval(p_try, r_eta);

            err = mean((V - V_fit).^2);
            if err < best_err
                best_err = err;
                best_xc = xc_try;
                best_yc = yc_try;
            end
        end
    end
end

fprintf('粗略搜索最佳 x_c = %.2f, y_c = %.2f\n', best_xc, best_yc);

%% **2. 精细优化 (x_c, y_c, p)**
for c = 1:3
    fprintf('优化通道 %d 中...\n', c);
    
    % 获取当前通道的像素值
    V = img(:,:,c);
    V = double(V(:));

    % 计算初始径向坐标 r_eta
    r_eta = sqrt((x_grid - best_xc).^2 + (y_grid - best_yc).^2);
    r_eta = double(r_eta(:));

    % 初始二阶多项式拟合
    p_init = polyfit(r_eta, V, S);
    p_init = double(p_init);

    % **初始参数**
    theta0 = double([best_xc, best_yc, p_init]);

    % **使用 lsqnonlin 进行优化**
    options = optimoptions('lsqnonlin', 'Display', 'iter', 'Algorithm', 'trust-region-reflective');
    theta_opt = lsqnonlin(@(theta) objective_function(theta, x_grid, y_grid, V, S), theta0, [], [], options);
    
    % 存储优化参数
    theta_opt_RGB(c, :) = theta_opt;
end

%% **计算最终拟合的暗角校正函数**
V_fitted_RGB = zeros(H, W, 3);
for c = 1:3
    x_c_opt = theta_opt_RGB(c, 1);
    y_c_opt = theta_opt_RGB(c, 2);
    eta_opt = 1;  % 固定 η=1
    p_opt = theta_opt_RGB(c, 3:end);

    r_eta_opt = sqrt((x_grid - x_c_opt).^2 + (eta_opt * (y_grid - y_c_opt)).^2);
    
    V_fitted_RGB(:,:,c) = reshape(polyval(p_opt, r_eta_opt(:)), H, W);
end

%% **计算暗角校正后的图像**
img_corrected = img ./ V_fitted_RGB;
img_corrected = max(img_corrected, 0);

%% **显示结果**
figure;
subplot(1, 3, 1);
imshow(img, []); title('原始图像');

subplot(1, 3, 2);
imshow(V_fitted_RGB, []); title('拟合的暗角校正函数');

subplot(1, 3, 3);
imshow(img_corrected, []); title('暗角校正后');

%% **保存优化参数**
save('vignetting_correction_params.mat', 'theta_opt_RGB');
fprintf('优化完成，参数已保存。\n');

%% **目标函数**
function err = objective_function(theta, x_grid, y_grid, V, S)
    x_c = double(theta(1));
    y_c = double(theta(2));
    eta = 1;
    p = double(theta(3:end));

    r_eta = sqrt((x_grid - x_c).^2 + (eta * (y_grid - y_c)).^2);
    r_eta = double(r_eta(:));

    V_fit = polyval(p, r_eta);
    V_fit = double(V_fit);

    err = V - V_fit; % 误差向量，lsqnonlin 最小化误差平方和
end
