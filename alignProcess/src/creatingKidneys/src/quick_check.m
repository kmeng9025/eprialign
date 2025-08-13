% Quick check of the latest file
latest_file = 'C:\Users\ftmen\Documents\mrialign\alignProcess\data\inference\kidney_slaves_20250813_132233\withoutROIwithMRI_FINAL_KIDNEY_SLAVES.mat';

if exist(latest_file, 'file')
    data = load(latest_file);
    img = data.images{1, 1};
    slave = img.slaves{1};
    
    fprintf('✅ Latest slave settings:\n');
    fprintf('   Name: "%s"\n', slave.Name);
    fprintf('   Type: "%s"\n', slave.ImageType);
    fprintf('   Visible: %d (should be 0)\n', slave.Visible);
    fprintf('   isLoaded: %d (should be 0)\n', slave.isLoaded);
    fprintf('   isStore: %d (should be 1)\n', slave.isStore);
    fprintf('   Data size: %s\n', mat2str(size(slave.data)));
end
