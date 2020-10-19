% IMPORT THE IMAGE
Iin = imread("../Images/Liu17.PNG");

% PROCESS HERE
% Iout = entropyBlkProc(Iin, 7, 3);
% Iout = entropyPixProc(Iin, 3, .8);
Iout = imGrad(Iin);

% SHOW THE IMAGE
imshow(Iout);

function Inorm = entropyBlkProc(I, filtSz, blkSz, thres)
% Process in distinct blocks
I = reduceChannels(I);

entropyBlkPrc = @(block_struct) entropy(block_struct.data);
Ifilt = blockproc(...
    medfilt2(I, [filtSz filtSz]),...
    [blkSz blkSz],...
    entropyBlkPrc);
Inorm = normIm(Ifilt);
if nargin > 3
    Inorm(Inorm < thres) = 0;
end
end

function Inorm = entropyPixProc(I, blkSz, thres)
% Process in block around each pixel
I = reduceChannels(I);

fun = @(x) entropy(x(:));
Ifilt = nlfilter(I, [blkSz blkSz], fun);
Inorm = normIm(Ifilt);
if nargin > 2
    Inorm(Inorm < thres) = 0;
end
end

function Inorm = imGrad(I)
% Return normalized image gradient magnitudes
I = reduceChannels(I);

[gradMag, ~] = imgradient(I);
Inorm = normIm(gradMag);
end

function I = reduceChannels(I)
% Reduce channels from 3->1 if 3 exist, using RGB mapping
[~,~,c] = size(I);
if c == 3
    I = rgb2gray(I);
end
end

function Inorm = normIm(I)
% Rescale intensities.
Inorm = I;
theMax = max(I, [], 'all');
if theMax > 0
    Inorm = Inorm / theMax;
end
end