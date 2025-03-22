function [channel1, channel2, channel3, channel4] = computeCorrChannels(vector)
    %COMPUTECORRCHANNELS Given a 1×48 vector, compute its correlation matrix,
    % then take the square root, and extract angle, abs, real, and imag channels.
    %
    %   Inputs:
    %       vector  - 1×48 real-valued vector
    %
    %   Outputs:
    %       channel1 - angle of sqrt(corr_matrix)
    %       channel2 - abs    of sqrt(corr_matrix)
    %       channel3 - real   of sqrt(corr_matrix)
    %       channel4 - imag   of sqrt(corr_matrix)
    
        % 1) Subtract mean
        mean_x = mean(vector);
        centeredVec = vector - mean_x;  % 1×48
    
        % 2) Outer product for correlation matrix
        %    corr_matrix(i,j) = centeredVec(i) * centeredVec(j)
        corr_matrix = centeredVec(:) * centeredVec(:)';  % 48×48
        
        % 3) Take square root (watch for negative or zero entries in corr_matrix)
        log_corr = sqrt(corr_matrix);
    
        % 4) Extract 4 channels: angle, abs, real, imag
        channel1 = angle(log_corr);  % angle of complex matrix
        channel2 = abs(log_corr);    % magnitude
        channel3 = real(log_corr);   % real part
        channel4 = imag(log_corr);   % imaginary part
    end
    