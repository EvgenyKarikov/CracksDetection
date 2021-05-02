gTruth=open('gTruth.mat');
sz = size(gTruth.gTruth.LabelData);
savepath = 'pixelres/';

data = load('dl3.mat');
net = data.net;

for i=1:sz(1) 
    img_num = i;

    test_img = readimage(imds,img_num);

    [pxdsResults, scores] = semanticseg(test_img,net);

    detectedimg = labeloverlay(test_img,pxdsResults);
    imwrite(detectedimg,savepath+string(i)+'.jpg')
    %imshow(detectedimg);
end
