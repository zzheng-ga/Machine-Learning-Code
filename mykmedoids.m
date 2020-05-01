function [ class, centroid ] = mykmedoids( pixels, K )
%
% Your goal of this assignment is implementing your own K-medoids.
% Please refer to the instructions carefully, and we encourage you to
% consult with other resources about this algorithm on the web.
%
% Input:
%     pixels: data set. Each row contains one data point. For image
%     dataset, it contains 3 columns, each column corresponding to Red,
%     Green, and Blue component.
%
%     K: the number of desired clusters. Too high value of K may result in
%     empty cluster error. Then, you need to reduce it.
%
% Output:
%     class: the class assignment of each data point in pixels. The
%     assignment should be 1, 2, 3, etc. For K = 5, for example, each cell
%     of class should be either 1, 2, 3, 4, or 5. The output should be a
%     column vector with size(pixels, 1) elements.
%
%     centroid: the location of K centroids in your result. With images,
%     each centroid corresponds to the representative color of each
%     cluster. The output should be a matrix with K rows and
%     3 columns. The range of values should be [0, 255].
%     
%
% You may run the following line, then you can see what should be done.
% For submission, you need to code your own implementation without using
% the kmeans matlab function directly. That is, you need to comment it out.

	%[class, centroid] = kmeans(pixels, K);
    
    % m: number of data pixel
    [m, ~] = size(pixels);

    
    % K centroid initialization, randomly pick K points as centroids
    new_centroid_idx = randi([1, m], K, 1);
    centroid_idx    = ones(K, 1) * (m + 1);

    itr = 0;
    % before get into the while loop, centroid_idx is just a dummy var
    while(~isequal(centroid_idx, new_centroid_idx) && itr <= 200)


        centroid_idx = new_centroid_idx;
        centroid     = pixels(centroid_idx, :);

        % compute distance between every point with every centroid
        d = zeros(m, K);
        for i = 1 : m
            for j = 1 : K
                d(i, j) = norm(pixels(i, :) - centroid(j, :))^2;
            end
        end


        % find the smallest index of each row, that's the cluster they belong
        % to
        [~, class] = min(d, [], 2);


        % start to find a new centroid for every cluster
        for i = 1 : K

            % information about group i
            group_idx           = find(class == i);
            group_data          = pixels(group_idx, :);
            num_this_group      = length(group_idx);


            min_value = sum(vecnorm(group_data - centroid(i), 2, 2).^2);
            for j = 1 : num_this_group


                candidate = group_data(j, :);

                % in this group, find, distance from every group memeber to this
                % candidate, then add them all
                can_dist = sum(vecnorm(group_data - candidate, 2, 2).^2);
                
                % as long as a new point is better, we use it as our new
                % centroid
                if can_dist < min_value
                    min_value           = can_dist;
                    new_centroid_idx(i) = group_idx(j);
                    break
                end

            end
            

        end
        itr = itr + 1;
    end
    
    centroid_idx = new_centroid_idx;
    centroid     = pixels(new_centroid_idx, :);
    disp(centroid_idx);
end



