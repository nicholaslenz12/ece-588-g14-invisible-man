function Iout = dArg(Iin, filtSz)

% Crops the image to square
[m, n, c] = size(Iin);
w = min(n, m);
Ic = Iin(1:w, 1:w, :);

% Convert to grayscale
if (c == 3)
    Ic = rgb2gray(Ic);
end

% Could do change of basis in here...
% Ic = rgb2lab(Ic);
% Ic = Ic(:, :, 1);

if (filtSz > 1)
    Ic = medfilt2(Ic, [filtSz filtSz]);
end

dArg = zeros(w, w);
for i = 1:4
    % Rotate, then compute Yarg (partial derivative in y direc of argument of
    % gradient.
    Ir = rot90(Ic, i-1);
    [FX, FY] = gradient(double(Ir));
    theta = atan2(FY, FX);
    
    % Not the most efficient, but works for now. Compute partial y
    [~, TY] = gradient(theta);
    
    % Rotates back
    dArg = dArg + rot90(TY, -(i-1));
end

% Squares the image pixels, normalizes to [0, 1]
Iout = dArg.^2;
largInt = max(Iout, [], 'all');
if ( largInt > 0)
    Iout = Iout / max(dArg.^2, [], 'all');
end

end
