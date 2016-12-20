function out=corr_semi_v3(dataset,repeat_num,corr_order,model_name)
%record nan value location
location=isnan(dataset);
data=dataset;%copy a dataset
[r,c]=size(dataset);
total_size=r*c;
%calculate the correlation by col
[rho, pval] = corr(dataset, 'rows','pairwise');

weight=zeros(size(rho));
for ind_1=1:size(rho,1)
weight(ind_1,ind_1)=1;
s=sum(rho(ind_1,[ind_1+1:end]));
for ind_2=ind_1+1:size(rho,1)
    weight(ind_1,ind_2)=abs(rho(ind_1,ind_2))/s;
end
end


%%
%%%%%%%%%%%%%%%%%%%Check the status of missing data%%%%%%%%%%%%%%%%%%%%%%%%

%First, understand the missing data of original data file

miss_status=checkMissStatus(dataset);

%%
%%%%%%%%%%%%%%%%%%%%Calculate the max correlation%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

max_corr=checkMaxCorr(miss_status,rho,corr_order);


%sort the max_corr list based on the less missing data percent
sorted_max_corr= sortCorrTable(max_corr);


%%
%%%%%%%%%%%%%%%%%%%%%%%%Imputation Process%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%create linear model for every col based on the variable with the max
%correlationship

fprintf('\nOriginal Missing data(percent): %f\n', (length(find(location==1))/total_size) * 100);

imputation(dataset,sorted_max_corr);

%%


function [ out] = model(train_X,train_Y,model_name)
switch model_name
    case 'knn1'
        out=fitcknn(train_X,train_Y,'NumNeighbors',1);
    case 'knn2'
        out=fitcknn(train_X,train_Y,'NumNeighbors',2);
    case 'knn3'
        out=fitcknn(train_X,train_Y,'NumNeighbors',3);
    case 'knn4'
        out=fitcknn(train_X,train_Y,'NumNeighbors',4);
    case 'knn5'
        out=fitcknn(train_X,train_Y,'NumNeighbors',5);
    case'linear'
        out=fitlm(train_X,train_Y);
              
end
end

function [out]=checkMissStatus(dataset)
%First, understand the missing data of original data file
out=zeros(size(dataset,2),1);
for j=1:size(dataset,2)
    out(j,1)=1-(length(find(~location(:,j)))/length(location(:,j)));
end
[S,I]=sort(out);
Miss_percent=S;
Col_index=I;
out(:,1)=S;
out(:,2)=I;
miss_table=table(Miss_percent,Col_index);
writetable(miss_table,'Miss_status_table.csv');
    
end

function [max_corr]=checkMaxCorr(miss_status,rho,corr_order)

max_corr=zeros(size(rho,1),4);
count=1;
%for every col, calculate the index of max correlation col except itself 
for i=1:size(rho,1)
    temp=abs(rho(i,:));
    max_corr(i,1)=count;
   
    if(~all(isnan(temp)))
    %calculate the max correlation except itself
    [ind1,ind2]=sort(temp,'descend');
    ii=1;
    
    while isnan(ind1(ii))~=0
    ii=ii+1;
    end
    corr_value=ind1(ii+corr_order);%highest=1, second=2, third=3,...
    corr_ind=ind2(ii+corr_order);%highest=1, second=2, third=3,...
    max_corr(i,2)=corr_ind;
    max_corr(i,3)=corr_value;
    max_corr(i,4)=miss_status(find(miss_status(:,2)==corr_ind),1);
   %if the whole col is nan, record the index as 0
    else
    max_corr(i,2:4)=0;
    end
    count=count+1;
end 
end

function [sorted_max_corr]= sortCorrTable(max_corr)
[values, order] = sort(max_corr(:,4));
sorted_max_corr = max_corr(order,:);
CurrentVar=sorted_max_corr(:,1);
MaxCorrVar=sorted_max_corr(:,2);
Corr=sorted_max_corr(:,3);
MissStatus_MaxCorrVar=sorted_max_corr(:,4);
Sorted_Max_corr_table=table(CurrentVar,MaxCorrVar,Corr,MissStatus_MaxCorrVar);
% 
 writetable(Sorted_Max_corr_table,'Sorted_Max_corr_table.csv');
end

function imputation(dataset,corr_table)
    location_nan=isnan(dataset);
for k=1:size(corr_table,1)
    run_times=1;
    if corr_table(k,2)~=0   
    %trainning the model repeatly to improve the accuracy 
    while(run_times<=repeat_num)        
    %find the index of nan value
    nan_index=find(location_nan(:,corr_table(k,1)));
    %only update the index of train data
    location_new=isnan(data);
    train_index=intersect(find(~location_new(:,corr_table(k,1))),find(~location_new(:,corr_table(k,2))));
    %linear model
    
    mdl=model(data(train_index,corr_table(k,2)),data(train_index,corr_table(k,1)),model_name);   
    %input value is the value of same location of the highest correlation 
    x=data(nan_index,corr_table(k,2));
    %impute the predict value to the nan value location
    data(nan_index,corr_table(k,1))=predict(mdl,x); 
    run_times=run_times+1;
    end   
    end
end
    
end


fprintf('\nCorr_semi_v3:Current Missing data(percent): %f\n ', (length(find(location_new==1))/total_size )* 100);
%write the result dataset to the csv file
outid = fopen ('result_dataset.csv', 'w+');
dlmwrite ('result_dataset.csv', data, '-append' );
fclose(outid);


out=data;

end