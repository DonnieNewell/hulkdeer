function n_threshold = surfNThreshold(descriptors, n, threshold)
% Donnie Newell (den4gr)
% returns the n strongest descriptors above the threshold
[n_strongest, strength] = surfNStrongest(descriptors, n);
above_thresh = strength > threshold;
n_threshold = n_strongest(above_thresh, :);
end