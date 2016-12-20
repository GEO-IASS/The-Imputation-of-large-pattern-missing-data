function [out]=locf(miss_data)
location_nan=isnan(miss_data);
[r,c]=size(miss_data);
for i=1:r
    for j=1:c
        if location_nan(i,j)==1 %if this value is nan
            %find the last observation available value in the same column
            if i-1>0 && isnan(miss_data(i-1,j))~=1 
            miss_data(i,j)=miss_data(i-1,j);
            %if there is no availble value before this nan value, find the
            %next available value after this nan value
            else
                if isnan(miss_data(i+1,j))~=1
                miss_data(i,j)=miss_data(i+1,j);
                else 
                    z=i;
                    while isnan(miss_data(z,j))==1
                        z=z+1;
                    end
                    miss_data(i,j)=miss_data(z,j);
                end
            end
        end
    end
end
                    
 out=miss_data;               
            
end




