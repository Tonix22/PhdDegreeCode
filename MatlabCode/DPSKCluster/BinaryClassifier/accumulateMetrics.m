function metricsTable = accumulateMetrics(metricsCell, symbolVec)
    % Convert the cell array to a struct array.
    metricsStruct = vertcat(metricsCell{:});  % Ensure it's correctly structured
    
    % Convert the struct array to a table.
    metricsTable = struct2table(metricsStruct);
    
    % Ensure symbolVec matches the number of rows in metricsTable
    if height(metricsTable) ~= length(symbolVec)
        error("Mismatch between metricsTable rows (%d) and symbolVec length (%d).", ...
               height(metricsTable), length(symbolVec));
    end
    
    % Add the symbol vector as a new column at the beginning.
    metricsTable = addvars(metricsTable, symbolVec(:), 'Before', 1, 'NewVariableNames', 'Symbol');
end
