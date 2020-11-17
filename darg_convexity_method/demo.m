% Normalized paraboloid
% x = -100:100;
% [X, Y] = meshgrid(x, x);
% Z = 5*X.^2 + Y.^2;
% Iin = Z / max(Z, [], 'all');

Iin = imread("../Images/Images_from_Liu_Bolin_s_site/Liu28.PNG");

Iout = darg(Iin, 5);
% Iout(Iout < .65) = 0;
imshow(Iout);
