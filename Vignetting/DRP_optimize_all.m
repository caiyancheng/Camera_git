clc; clear; close all;

%% 读取 EXR 图像
img = exrread('E:\sony_pictures\Vignetting/merged_16_20_ARQ_mtf_focus_distance_100cm_real_distance_40cm.exr');  % 读取 HDR 图像
[H, W, C] = size(img);
if C ~= 3
    error('输入的 EXR 文件必须是三通道 (RGB) 图像');
end

%% 设定多项式阶数
S = 2; % 多项式阶数

%% 初始化参数
[x_grid, y_grid] = meshgrid(1:W, 1:H); % 生成坐标网格
x_c = 3343; %W / 2;  % 初始 x_c
y_c = 2535; %H / 2;  % 初始 y_c
eta = 1;      % 初始 η

%% 预分配参数存储
theta_opt_RGB = zeros(3, S + 4); % 存储三个通道的优化参数（x_c, y_c, eta, S+1个多项式系数）

%% 目标优化函数
function err = objective_function(theta, x_grid, y_grid, V, S)
    x_c = theta(1);
    y_c = theta(2);
    eta = theta(3);
    p = theta(4:end);
    
    % 计算新的 r_eta
    r_eta = sqrt((x_grid - x_c).^2 + (eta * (y_grid - y_c)).^2);
    r_eta = r_eta(:);
    
    % 计算拟合值
    V_fit = polyval(p, r_eta);
    
    % 计算误差
    err = mean((V - V_fit).^2);
    err = double(err);
    fprintf('当前误差: %e\n', err);
end

%% 对 R, G, B 三个通道分别进行拟合
for c = 1:3
    fprintf('优化通道 %d 中...\n', c);
    
    % 获取当前通道的像素值
    V_matrix = img(:,:,c);
    V = V_matrix(:);  % 转换为列向量

    % 计算初始径向坐标 r_eta
    r_eta_matrix = sqrt((x_grid - x_c).^2 + (eta * (y_grid - y_c)).^2);
    r_eta = r_eta_matrix(:);
    
    % 初始多项式拟合
    p_init = polyfit(r_eta, V, S); % S+1 个系数

    % 初始参数组合
    theta0 = double([x_c, y_c, eta, p_init]); 

    % 进行优化
    options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'Display', 'iter');
    theta_opt = fminunc(@(theta) objective_function(theta, x_grid, y_grid, V, S), theta0, options);
    
    % 存储优化参数
    theta_opt_RGB(c, :) = theta_opt;
end

%% 计算最终拟合的暗角校正函数
V_fitted_RGB = zeros(H, W, 3);
for c = 1:3
    x_c_opt = theta_opt_RGB(c, 1);
    y_c_opt = theta_opt_RGB(c, 2);
    eta_opt = theta_opt_RGB(c, 3);
    p_opt = theta_opt_RGB(c, 4:end); % S+1 个多项式系数

    % 计算 r_eta
    r_eta_opt = sqrt((x_grid - x_c_opt).^2 + (eta_opt * (y_grid - y_c_opt)).^2);
    
    % 计算拟合的暗角校正值
    V_fitted_RGB(:,:,c) = reshape(polyval(p_opt, r_eta_opt(:)), H, W);
end

%% 计算暗角校正后的图像
img_corrected = img ./ V_fitted_RGB;
img_corrected = max(img_corrected, 0); % 避免负值

%% 显示结果
figure;
subplot(1, 3, 1);
imshow(img, []); title('原始图像');

subplot(1, 3, 2);
imshow(V_fitted_RGB, []); title('拟合的暗角校正函数');

subplot(1, 3, 3);
imshow(img_corrected, []); title('暗角校正后');

%% 保存优化后的参数
save('vignetting_correction_params.mat', 'theta_opt_RGB');

fprintf('优化完成，参数已保存至 vignetting_correction_params.mat。\n');