function compute_nrgb2xyz_specbos( base_name, options )
arguments
    base_name char
    options.compensate_for_beamsplitter logical = false
    options.compensate_for_black_level logical = false
    options.ciede_loss_func logical = false
    options.root_polynomial logical = false
end
% Compute native RGB->XYZ matrix based on Specbos measurements

% The specbos measurements will be adjusted to compensate for the
% beamsplitter
% compensate_for_beamsplitter = false;
% compensate_for_black_level = false;
% ciede_loss_func = false;
% root_polynomial = false;

M = readtable( [base_name '_xyz.csv'] );

% Native RGBs from the RAW image generated with extract_color_checker();
R = readtable( [base_name '_rgb.csv'] ); %(24*9)

J = join( M, R, 'Key', 'color_id' );

XYZ = cat( 2, J.X, J.Y, J.Z ); % Specbos measurements
RGB = cat( 2, J.R, J.G, J.B );
%RGB_std = cat( 2, J.R_std, J.G_std, J.B_std );

ind = find( strcmp( J.color_id, 'A4' ) ); % White patch
XYZ_wp = XYZ(ind,:);
RGB_wp = cm_xyz2rgb( XYZ_wp );

if options.root_polynomial
    RGB = max(0,RGB);
    RGB = rgb2prod_polynomial(RGB);
    RGB_wp = rgb2prod_polynomial(RGB_wp);
end

gray_patches = startsWith( J.color_id, 'A' ); % All gray patches

weights = ones(size(XYZ,1),1);
weights(gray_patches) = 10;

%col_metric = @(XYZ_a, XYZ_b) mean( (log(max(XYZ_a,1e-5)) - log(max(XYZ_b,1e-5))).^2 );
col_metric = @(XYZ_a, XYZ_b) mean((weights.*CIE2000deltaE_XYZ(XYZ_a, XYZ_b, XYZ_wp)).^2);

if options.ciede_loss_func
    use_metric = col_metric;
else
    use_metric = [];
end

if options.compensate_for_black_level
    M_nrgb2xyz = optimize_color_transform( RGB, XYZ, { 'black_level', true, 'metric', use_metric } );
    RGB_wp_cal = xyz2rgb( [RGB_wp 1] * M_nrgb2xyz' );
else
    M_nrgb2xyz = optimize_color_transform( RGB, XYZ, { 'black_level', false, 'metric', use_metric  } );
    RGB_wp_cal = cm_xyz2rgb( RGB_wp * M_nrgb2xyz' );
end


wb_coeffs = max(RGB_wp_cal)./RGB_wp_cal;

display(M_nrgb2xyz);

fprintf( 1, 'Use these coefficients to white ballance in rec709 colour space' )

display(wb_coeffs);

calibration_date = datestr( now );

if options.root_polynomial
    type = 'root_polynomial';
else
    type = 'linear';
end

save( [base_name '_color_matrix.mat'], 'M_nrgb2xyz', 'wb_coeffs', 'base_name', 'calibration_date', 'type' );

% Visualize the result of the calibration

cc_rows = 4;
cc_cols = 6;

I_vis = zeros(cc_rows,cc_cols*2,3);

if options.compensate_for_black_level
    XYZ_pred = [RGB ones(size(RGB,1),1)] * M_nrgb2xyz'; % Camera colours after the calibration
    %    XYZ_pred_std = [RGB+RGB_std ones(size(RGB,1),1)] * M_nrgb2xyz' - XYZ_pred;
else
    XYZ_pred = RGB * M_nrgb2xyz'; % Camera colours after the calibration
    %    XYZ_pred_std = (RGB+RGB_std) * M_nrgb2xyz' - XYZ_pred;
end

for rr=1:cc_rows
    for cc=1:cc_cols

        col_label = strcat( char( 'A'+cc-1 ), num2str( rr ) );
        ind = find( strcmp( J.color_id, col_label ) );
        assert( length(ind) == 1 );

        I_vis( rr, (cc-1)*2+1, : ) = cm_xyz2rgb( XYZ(ind,:) );
        I_vis( rr, (cc-1)*2+2, : ) = cm_xyz2rgb( XYZ_pred(ind,:) );

    end
end

gamma = 2.2;

upscale_factor=32;

figure(1);
clf
subplot( 2, 1, 1 );

V_max = max(I_vis(:));

imshow( imresize( max(0,I_vis/V_max).^(1/gamma), upscale_factor, 'box' ) );
title( 'No WB, left: specbos, right: RAW calibrated' );

add_delte_E_text( XYZ, XYZ_pred, XYZ_wp )

xlabel( 'The numbers are CIE DeltaE values for each colour pair' );

subplot( 2, 1, 2 );

imshow( imresize( max(0,I_vis./repmat(reshape(RGB_wp(1:3),[1 1 3]), [cc_rows cc_cols*2 1] )/1.2).^(1/gamma), upscale_factor, 'box' ) );


title( 'With WB, left: specbos, right: RAW calibrated' );


figure(2);
visualize_color_error( XYZ, XYZ_pred, XYZ_wp );


    function add_delte_E_text( XYZ_a, XYZ_b, XYZ_wp )

    for rr=1:cc_rows
        for cc=1:cc_cols
            col_label = strcat( char( 'A'+cc-1 ), num2str( rr ) );
            ind = find( strcmp( J.color_id, col_label ) );

            DeltaE = CIE2000deltaE_XYZ(XYZ_a(ind,:), XYZ_b(ind,:), XYZ_wp);

            text( ((cc-1)*2+1)*upscale_factor, ((rr-1)+0.5)*upscale_factor, sprintf( '%.3g', DeltaE ), 'HorizontalAlignment', 'Center' );

        end
    end

    end

end