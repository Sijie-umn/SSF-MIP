load('acm_climo.mat')
climo = zeros(23, 31, 428);
smoothwin = 31;
for i = 1:23
    for j = 1:31
        climo(i, j, :)=nanfastsmooth(climo_raw(i, j, :), smoothwin, 2, 1);
    end
end
save('climo_smooth.mat', 'climo')

% load('acm_climo_subx_verify.mat')
% climo_subx = zeros(23, 31, 428);
% smoothwin = 31;
% for i = 1:23
%     for j = 1:31
%         climo_subx(i, j, :)=nanfastsmooth(climo_raw_subx(i, j, :), smoothwin, 2, 1);
%     end
% end
% save('climo_smooth_subx_verify.mat', 'climo_subx')
