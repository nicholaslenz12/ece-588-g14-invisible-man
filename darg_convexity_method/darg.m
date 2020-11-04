function Iout = darg(Iin, filtSz)

% Crops the image to square
[m, n, c] = size(Iin);
w = min(n, m);
Ic = Iin;

% Convert to grayscale
if (c == 3)
    Ic = rgb2gray(Ic);
end

% Could do change of basis in here...
% Ic = rgb2lab(Ic);
% Ic = Ic(:, :, 1);

dArg = zeros(m, n);
for i = 1:4
    % Rotate, then compute Yarg (partial derivative in y direc of argument of
    % gradient.
    Ir = rot90(Ic, i-1);
    [FX, FY] = gradient(double(Ir));
    theta = atan2(FY, FX);
    
    if (filtSz > 1)
        theta = imgaussfilt(theta, filtSz);
    end

    
    % Not the most efficient, but works for now. Compute partial y
    [~, TY] = gradient(theta);
    
    % Rotates back
    dArg = dArg + rot90(TY, -(i-1));
end

Iout = dArg.^2;

largInt = max(Iout, [], 'all');
if ( largInt > 0)
    Iout = Iout / max(dArg.^2, [], 'all');
end

end
