tic
%% %load txt file
weather=importdata('aol.txt');
dataset=[weather.textdata num2cell(weather.data)];
%convert cell to double matrix
temp=str2double(dataset);
%only access numeric variable as ground truth 
dataset_gt=temp(:,[4:9,11:12]);

%% %remove original nan value to create ground truth matrix
dataset_gt=dataset_gt(~any(isnan(dataset_gt),2),:);


% %sparsify the ground truth using Guassian Distribution
[r,c]=size(dataset_gt);

dataset_sparse=zeros(size(dataset_gt));
for i=1:size(dataset_gt,2)
A=dataset_gt(:,i);
pd=fitdist(A,'Normal');
range=[pd.mu-pd.sigma pd.mu+pd.sigma];
A(any(A>range(1)& A<range(2),2),:)=NaN;
dataset_sparse(:,i)=A;
end
location_miss=isnan(dataset_sparse);
fprintf('\nOriginal Missing data(percent): %f\n ', (length(find(location_miss==1))/(r*c) )* 100);

%% %recover the dataset 
%%%Mean
missing_values='NaN';
dataset_mean=Imputer(dataset_sparse,'mean',missing_values);
loation_mean=isnan(dataset_mean);
fprintf('\nMean:Current Missing data(percent): %f\n ', (length(find(loation_mean==1))/(r*c) )* 100);

MSE_mean=mean((dataset_mean(find(location_miss==1))-dataset_gt(find(location_miss==1))).^2);


%%%Most_frequent
dataset_most_frequent=Imputer(dataset_sparse,'most_frequent',missing_values);
loation_most_frequent=isnan(dataset_most_frequent);
fprintf('\nMost_frequent:Current Missing data(percent): %f\n ', (length(find(loation_most_frequent==1))/(r*c) )* 100);

MSE_most_frequent=mean((dataset_most_frequent(find(location_miss==1))-dataset_gt(find(location_miss==1))).^2);
%%%LOCF
dataset_locf=locf(dataset_sparse);
loation_locf=isnan(dataset_locf);
fprintf('\nLOCF:Current Missing data(percent): %f\n ', (length(find(loation_most_frequent==1))/(r*c) )* 100);


MSE_locf=mean((dataset_locf(find(location_miss==1))-dataset_gt(find(location_miss==1))).^2);



%%%Corr_semi_v3(linear model)
dataset_corr_semi_lm_3=corr_semi_v3(dataset_sparse,4,1,'linear');%using the highest correlation

% dataset_corr_semi_lm_2=corr_semi_v2(dataset_corr_semi_lm_2,5,2,'linear');
% dataset_corr_semi_lm_2=corr_semi_v2(dataset_corr_semi_lm_2,5,3,'linear');
% dataset_corr_semi_lm_2=corr_semi_v2(dataset_corr_semi_lm_2,5,4,'linear');
% dataset_corr_semi_lm_2=corr_semi_v2(dataset_corr_semi_lm_2,5,5,'linear');
% dataset_corr_semi_lm_2=corr_semi_v2(dataset_corr_semi_lm_2,5,6,'linear');
% dataset_corr_semi_lm_2=corr_semi_v2(dataset_corr_semi_lm_2,5,7,'linear');
% 
location_semi_3=isnan(dataset_corr_semi_lm_3);
dataset_corr_semi_lm_3(find(location_semi_3==1))=0;
data_copy_3=dataset_gt;
data_copy_3(find(location_semi_3==1))=0;


MSE_corr_semi_lm_3=mean((dataset_corr_semi_lm_3(find(location_miss==1))-data_copy_3(find(location_miss==1))).^2);

% resut_table=table(MSE_mean,MSE_most_frequent,MSE_locf,MSE_corr_semi_lm,MSE_corr_semi_knn,MSE_corr_semi_lm_2);


%plot the result
y=[MSE_mean MSE_most_frequent MSE_locf MSE_corr_semi_lm_3];
numberOfXTicks = 4;
h = plot(y);
xData = get(h,'XData');
set(gca,'Xtick',linspace(xData(1),xData(end),numberOfXTicks))
set(gca,'xticklabel',{'mean','most_frequent','locf','corr_semi_lm'})
xlabel('Mean Square Error','fontsize',20);
strValues = strtrim(cellstr(num2str( y(:),'(%f)')));
text(xData,y,strValues,'VerticalAlignment','bottom');

toc

