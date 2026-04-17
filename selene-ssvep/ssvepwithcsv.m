%   1 = top-center     (numpad 8 or key '1')
%   2 = left-middle    (numpad 4 or key '2')
%   3 = right-middle   (numpad 6 or key '3')
%   4 = bottom-center  (numpad 2 or key '4')

function ssvepwithcsv()

    frequencies = [12, 8, 15, 10];  
    duration    = 30;               
    squareSize  = 40;             
    gridN       = 5;            
    subjectID   = 'S01'; % u can change to ur names.        

    nPatches    = 4;
    halfPeriods = 0.5 ./ frequencies;
    patchSize   = gridN * squareSize;

    patchKeys = {'1','2','3','4','numpad1','numpad2','numpad3','numpad4'};
    keyToPatch = containers.Map(...
        {'1','2','3','4','numpad1','numpad2','numpad3','numpad4'}, ...
        [1, 2, 3, 4, 1, 2, 3, 4]);

    markerLog  = {}; 
    logFile    = sprintf('SSVEP_markers_%s_%s.csv', subjectID, datestr(now,'yyyymmdd_HHMMSS'));

    phaseA = makeCheckerboard(gridN, squareSize, false);
    phaseB = makeCheckerboard(gridN, squareSize, true);

    fig = figure('Color', [0.5 0.5 0.5], ...
                 'MenuBar', 'none', ...
                 'ToolBar', 'none', ...
                 'NumberTitle', 'off', ...
                 'Name', 'SSVEP — Keys 1-4 to mark attention | Q to quit', ...
                 'CloseRequestFcn', @(~,~) [], ...
                 'KeyPressFcn', @keyHandler);

    set(fig, 'Units', 'normalized', 'OuterPosition', [0 0 1 1]);
    drawnow;

    set(fig, 'Units', 'pixels');
    figPos  = get(fig, 'Position');
    screenW = figPos(3);
    screenH = figPos(4);

    margin  = 50;
    centers = [
        screenW*0.5,                    margin + patchSize/2;
        margin + patchSize/2,           screenH*0.5;
        screenW - margin - patchSize/2, screenH*0.5;
        screenW*0.5,                    screenH - margin - patchSize/2;
    ];

    patchLabels = {'TOP (1)', 'LEFT (2)', 'RIGHT (3)', 'BOTTOM (4)'};

    ax    = gobjects(nPatches, 1);
    img   = gobjects(nPatches, 1);
    txFreq = gobjects(nPatches, 1); 

    for i = 1:nPatches
        cx = centers(i,1);  cy = centers(i,2);
        left   = (cx - patchSize/2) / screenW;
        bottom = 1 - (cy + patchSize/2) / screenH;
        w      = patchSize / screenW;
        h      = patchSize / screenH;

        ax(i)  = axes('Parent', fig, ...
                      'Position', [left bottom w h], ...
                      'Visible', 'off');
        img(i) = imshow(phaseA, 'Parent', ax(i));

        txFreq(i) = uicontrol('Style', 'text', ...
            'Parent', fig, ...
            'Units', 'pixels', ...
            'Position', [cx - 40, screenH - (cy + patchSize/2 + 22), 80, 18], ...
            'String', sprintf('%s | %d Hz', patchLabels{i}, frequencies(i)), ...
            'BackgroundColor', [0.5 0.5 0.5], ...
            'ForegroundColor', [1 1 1], ...
            'FontSize', 9);
    end

    drawnow;

    state.quit      = false;
    state.markerLog = markerLog;
    state.startTime = [];      
    state.keyToPatch = keyToPatch;
    state.frequencies = frequencies;
    state.patchKeys = patchKeys;
    set(fig, 'UserData', state);

    fprintf('Recording started. Keys 1-4 = patches [top, left, right, bottom]. Q = quit.\n');
    fprintf('Marker log will be saved to: %s\n', logFile);

    startTime = tic;
    state.startTime = startTime;
    set(fig, 'UserData', state);

    phase    = zeros(1, nPatches);
    nextFlip = halfPeriods;

    while toc(startTime) < duration

        state = get(fig, 'UserData');
        if state.quit || ~isvalid(fig)
            break;
        end

        t       = toc(startTime);
        updated = false;

        for i = 1:nPatches
            if t >= nextFlip(i)
                phase(i) = 1 - phase(i);
                if phase(i) == 0
                    set(img(i), 'CData', phaseA);
                else
                    set(img(i), 'CData', phaseB);
                end
                nextFlip(i) = nextFlip(i) + halfPeriods(i);
                updated = true;
            end
        end

        if updated
            drawnow;
        end
    end

    state = get(fig, 'UserData');
    saveMarkers(state.markerLog, logFile);

    if isvalid(fig)
        set(fig, 'CloseRequestFcn', 'closereq');
        close(fig);
    end
    fprintf('Stimulus complete. Markers saved to %s\n', logFile);
end

function keyHandler(src, event)
    state = get(src, 'UserData');

    if strcmpi(event.Key, 'q')
        state.quit = true;
        set(src, 'UserData', state);
        return;
    end

    if isKey(state.keyToPatch, event.Key)
        patchNum  = state.keyToPatch(event.Key);
        t         = toc(state.startTime);
        freqHz    = state.frequencies(patchNum);

        state.markerLog{end+1} = {t, patchNum, freqHz, event.Key};
        fprintf('  Marker @ %.3f s — Patch %d (%d Hz)\n', t, patchNum, freqHz);
    end

    set(src, 'UserData', state);
end

function saveMarkers(markerLog, filename)
    fid = fopen(filename, 'w');
    fprintf(fid, 'time_sec,patch,frequency_hz,key\n');
    for i = 1:length(markerLog)
        row = markerLog{i};
        fprintf(fid, '%.4f,%d,%d,%s\n', row{1}, row{2}, row{3}, row{4});
    end
    fclose(fid);
    fprintf('Saved %d markers to %s\n', length(markerLog), filename);
end

function img = makeCheckerboard(gridN, squareSize, invert)
    patchSize = gridN * squareSize;
    gray = zeros(patchSize, patchSize, 'uint8');
    for row = 1:gridN
        for col = 1:gridN
            isWhite = mod(row + col, 2) == 0;
            if invert, isWhite = ~isWhite; end
            val = uint8(isWhite) * 255;
            rIdx = (row-1)*squareSize+1 : row*squareSize;
            cIdx = (col-1)*squareSize+1 : col*squareSize;
            gray(rIdx, cIdx) = val;
        end
    end
    img = repmat(gray, 1, 1, 3);
end