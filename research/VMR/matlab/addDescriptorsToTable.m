function [table names] = addDescriptorsToTable(table, names, descriptors, filename)
%Donnie Newell
%VMIR
[m n] = size(descriptors);

for i = 1:n
    [old_m old_n] = size (table);
    descriptor = descriptors([], i);
    already_present = false;
    for j = 1:old_n
        table_descriptor = table([], j);
        already_present = descriptor == table_descriptor;
        if already_present
           % descriptor already in table
           break;
        end
    end
    if already_present == false
       disp('inserting new descriptor into table.'); 
       table([], old_n + 1) = descriptor;
    else
        disp('adding filename to list for pre-existing descriptor.');
    end
end
end