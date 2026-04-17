% ssvep_checkerboard_lsl.m
% seobin han: 2026/04/17

%% code explanation 
% keypresses are broadcast as an lsl stream so labrecorder captures them in
% a sync with the open bci cyton so that no manual alignment is needed.

%% set up
%   1. download liblsl-Matlab: https://github.com/labstreaminglayer/liblsl-Matlab
%   2. addpath('/path/to/liblsl-Matlab')

%% how you will record
%   1. start the openbci gui
%   2. run this script. should make a 2nd lsl stream called 'SSVEPMarkers'
%   3. open labrecorder (should see BOTH streams listed), and then record
%   4. u will press 1-4 during stimulus

%% key mapping:
%   1 = top center     
%   2 = left middle   
%   3 = righ middle   
%   4 = bottom center  

function ssvep_checkerboard_lsl()

    frequencies = [12, 8, 15, 10];   % top, left, right, bottom
    duration    = 30;              
    squareSize  = 40;              
    gridN       = 5;                 
    subjectID   = 's01';

    nPatches    = 4;
    halfPeriods = 0.5 ./ frequencies;
    patchSize   = gridN * squareSize;

    keyToPatch = containers.Map(...
        {'1','2','3','4','numpad1','numpad2','numpad3','numpad4'}, ...
        [1, 2, 3, 4, 1, 2, 3, 4]);

    lsl_available = true;
    try
        lib = lsl_loadlib();  

        info   = lsl_streaminfo(lib, 'SSVEPMarkers', 'Markers', 1, 0, 'cf_string', subjectID);
        outlet = lsl_outlet(info);

        fprintf('lsl stream "SSVEPMarkers" created. open labrecorder & u should see it.\n');
        fprintf('waiting 3 sec for labrecorder to detect the stream...\n');
        pause(3);   

    catch e
        lsl_available = false;
        warning('lsl not available (%s). using CSV-only logging.', e.message);
        warning('download liblsl-Matlab + add it to your path.');
        outlet = [];
    end

    phaseA = makeCheckerboard(gridN, squareSize, false);
    phaseB = makeCheckerboard(gridN, squareSize, true);

    fig = figure('Color', [0.5 0.5 0.5], ...
                 'MenuBar', 'none', 'ToolBar', 'none', ...
                 'NumberTitle', 'off', ...
                 'Name', 'keys 1-4 to mark and Q to quit', ...
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

    ax  = gobjects(nPatches, 1);
    img = gobjects(nPatches, 1);

    for i = 1:nPatches
        cx = centers(i,1);  cy = centers(i,2);
        left   = (cx - patchSize/2) / screenW;
        bottom = 1 - (cy + patchSize/2) / screenH;
        w      = patchSize / screenW;
        h      = patchSize / screenH;
        ax(i)  = axes('Parent', fig, 'Position', [left bottom w h], 'Visible', 'off');
        img(i) = imshow(phaseA, 'Parent', ax(i));
    end

    drawnow;

    state.quit          = false;
    state.markerLog     = {};
    state.startTime     = tic;
    state.keyToPatch    = keyToPatch;
    state.frequencies   = frequencies;
    state.outlet        = outlet;      
    state.lsl_available = lsl_available;
    set(fig, 'UserData', state);

    fprintf('\nStimulus running at [%s] Hz. Keys 1-4 = patches. Q = quit.\n\n', ...
            num2str(frequencies));

    startTime = state.startTime;
    phase     = zeros(1, nPatches);
    nextFlip  = halfPeriods;

    while toc(startTime) < duration
        state = get(fig, 'UserData');
        if state.quit || ~isvalid(fig), break; end

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

        if updated, drawnow; end
    end

    state = get(fig, 'UserData');

    % csv as a backup just in case your lsl crashes or doesn't save (it's
    % happened before)
    sessionTag = datestr(now, 'yyyymmdd_HHMMSS');
    csvFile    = sprintf('SSVEP_markers_%s_%s.csv', subjectID, sessionTag);
    saveMarkerCSV(state.markerLog, csvFile);

    if isvalid(fig)
        set(fig, 'CloseRequestFcn', 'closereq');
        close(fig);
    end

    fprintf('Done. %d markers recorded.\n', length(state.markerLog));
    if lsl_available
        fprintf('Markers were streamed live to LabRecorder via LSL.\n');
    end
    fprintf('CSV backup saved to: %s\n', csvFile);
end

function keyHandler(src, event)
    state = get(src, 'UserData');

    if strcmpi(event.Key, 'q')
        state.quit = true;
        set(src, 'UserData', state);
        return;
    end

    if isKey(state.keyToPatch, event.Key)
        patchNum = state.keyToPatch(event.Key);
        t        = toc(state.startTime);
        freqHz   = state.frequencies(patchNum);

        markerStr = sprintf('patch_%d_%dHz', patchNum, freqHz);

        if state.lsl_available && ~isempty(state.outlet)
            state.outlet.push_sample({markerStr});
        end

        state.markerLog{end+1} = {t, patchNum, freqHz, markerStr};

        fprintf('  [%.3f s] Marker sent: %s\n', t, markerStr);
    end

    set(src, 'UserData', state);
end

function saveMarkerCSV(markerLog, filename)
    fid = fopen(filename, 'w');
    fprintf(fid, 'time_relative_sec,patch,frequency_hz,marker\n');
    for i = 1:length(markerLog)
        row = markerLog{i};
        fprintf(fid, '%.4f,%d,%d,%s\n', row{1}, row{2}, row{3}, row{4});
    end
    fclose(fid);
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