function G = processChannel(H)
    persistent channelCont;
    if isempty(channelCont)
        channelCont = 1; % Start with the first channel
    end
    
    % Process the current channel
    G = H(:,:,channelCont);
    
    % Update the channel counter
    channelCont = channelCont + 1;
    
    % Reset the counter if it reaches 10000
    if channelCont == 10000
        channelCont = 1;
    end
end
