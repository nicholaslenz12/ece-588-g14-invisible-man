% Normalized paraboloid
x = -100:100;
[X, Y] = meshgrid(x, x);
Z = 5*X.^2 + Y.^2;
Iin = Z / max(Z, [], 'all');

Iout = dArg(Iin, 1);
imshow(Iout);
