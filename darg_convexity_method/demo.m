% Normalized paraboloid
% x = -100:100;
% [X, Y] = meshgrid(x, x);
% Z = 5*X.^2 + Y.^2;
% Iin = Z / max(Z, [], 'all');

Iin = imread("../Liu/Liu17.PNG");
I2 = rgb2gray(Iin);

Iout = darg(I2, 1);
imshow(Iout);