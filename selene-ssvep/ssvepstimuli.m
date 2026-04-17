function ssvepstimuli()

    frequencies = [12, 8, 15, 10];  % top, left, right, bottom (change these)
    duration    = 30;               % seconds (800 for full run)
    squareSize  = 30;             
    gridN       = 4;               

    nPatches    = 4;
    halfPeriods = 0.5 ./ frequencies;
    patchSize   = gridN * squareSize;

    phaseA = makeCheckerboard(gridN, squareSize, false);
    phaseB = makeCheckerboard(gridN, squareSize, true);

    fig = figure('Color', [0.5 0.5 0.5], ...
                 'MenuBar', 'none', ...
                 'ToolBar', 'none', ...
                 'NumberTitle', 'off', ...
                 'Name', 'stim running (press q to quit)', ...
                 'CloseRequestFcn', @(~,~) [] , ...  
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

        ax(i)  = axes('Parent', fig, ...
                      'Position', [left bottom w h], ...
                      'Visible', 'off');
        img(i) = imshow(phaseA, 'Parent', ax(i));
    end

    drawnow;

    phase    = zeros(1, nPatches);
    nextFlip = halfPeriods;

    fprintf('runing ssvep @ [%s] Hz. Q to quit.\n', num2str(frequencies));

    set(fig, 'UserData', false);

    startTime = tic;

    while toc(startTime) < duration
        if get(fig, 'UserData')
            break;
        end
        if ~isvalid(fig)
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
    if isvalid(fig)
        set(fig, 'CloseRequestFcn', 'closereq');
        close(fig);
    end
    fprintf('Stimulus complete.\n');
end

function keyHandler(src, event)
    if strcmpi(event.Key, 'q')
        set(src, 'UserData', true);
    end
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