%latSubX lonSubX
temp = -90:1:90;
latSubX = transpose(temp);
temp = 0:1:359;
lonSubX = transpose(temp);
%latObs lonObs
temp = -89.7500:0.5:89.7500;
latObs = transpose(flip(temp));
temp = 0.2500:0.5:359.75000;
lonObs = transpose(temp);
%create grid
[latSubXG,lonSubXG]=meshgrid(double(latSubX),double(lonSubX));
[latObsG,lonObsG]=meshgrid(double(latObs),double(lonObs));
for year = 1979:2020
    file_name = append('tmean.', num2str(year), '.nc');
    target = ncread(file_name, 'tmax');
    num_days = size(target);
    num_days = num_days(3);
    regrid = [];
    for it = 1:num_days
        regrid(:, :, it) = griddata(latObsG, lonObsG, double(target(:,:,it)),latSubXG, lonSubXG);
    end
    mat_name = append('regrid_', num2str(year), '.mat');
    save(mat_name, 'regrid');
end
