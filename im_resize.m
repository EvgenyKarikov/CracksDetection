clear all;
close all;
clc;

mkdir pics_resize/;

fl_list = string(ls('pics'));

for i=[1:length(fl_list)]
   fl = fl_list(i);
   res = true;
   try
       img = imread('pics/'+fl);
   catch
       res = false;
   end
   if (res)
       i
       dim = size(img);
       img_x = dim(1);
       img_y = dim(2);
       %img_crop = [ img_y*0.1 img_x*0.1  img_y*0.8 img_x*0.8];
       %img = imcrop(img,img_crop);
       img = imresize(img,[416 416]);
       imwrite(img,'pics_resize/'+string(i)+'.jpg');
   end
end