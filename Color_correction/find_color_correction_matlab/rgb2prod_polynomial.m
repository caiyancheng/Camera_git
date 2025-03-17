function RGB_poly = rgb2prod_polynomial( RGB )
%RGB_poly = cat( 2, RGB, sqrt(prod(RGB(:,1:2),2)), sqrt(prod(RGB(:,2:3),2)), sqrt(prod(RGB(:,[1 3]),2)) );
PPs = [0.5]; %0.1:0.1:0.9; %[0.5 0.6 0.7 0.8 0.9];

RGB_poly = RGB;
for kk=1:length(PPs)
    p = PPs(kk);
    n = (1-p);
    RGB_poly = cat( 2, RGB_poly, RGB(:,1).^p .* RGB(:,2).^n, RGB(:,2).^p .* RGB(:,3).^n, RGB(:,1).^p .* RGB(:,3).^n );
end


end