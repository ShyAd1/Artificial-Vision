function processImages()
% processImages - Cuenta objetos (ya lo haces) y calcula propiedades manualmente
%   Luego construye un dataset con las propiedades de todos los objetos de
%   todas las imágenes. Aplica k-means (implementado) con distancia
%   euclidiana y con distancia Mahalanobis para agrupar en 5 clases
%   (circulo, cuadrado, triangulo, estrella, flecha). Guarda CSV por
%   imagen y un CSV combinado.

    inputDir = 'FIGURAS';
    grayDir = 'FIGURAS_GRISES';
    binaryDir = 'FIGURAS_BINARIO';
    countDir = 'FIGURAS_CONTEO';

    if ~exist(grayDir, 'dir'), mkdir(grayDir); end
    if ~exist(binaryDir, 'dir'), mkdir(binaryDir); end
    if ~exist(countDir, 'dir'), mkdir(countDir); end

    files = dir(fullfile(inputDir, '*.*'));
    % prealloc para filas combinadas (se ampliará si hace falta)
    alloc = 2048;
    combined = cell(alloc, 15);
    combIdx = 0;

    % primer paso: etiquetado (ya lo hacía tu código). Aquí reutilizo tu
    % estrategia de etiquetado para asegurar compatibilidad.
    allFeatures = []; % fila por objeto, con columnas de características
    objectRefs = {}; % [imagen, objectID]

    for f = 1:length(files)
        [~, ~, ext] = fileparts(files(f).name);
        if ~ismember(lower(ext), {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'})
            continue;
        end

        imgPath = fullfile(inputDir, files(f).name);
        img = imread(imgPath);

        % Convertir a gris y binarizar con las funciones permitidas
        grayImg = rgb2gray(img);
        imwrite(grayImg, fullfile(grayDir, files(f).name));

        level = graythresh(grayImg);
        binaryImg = imbinarize(grayImg, level);
        invertedBinaryImg = ~binaryImg; % objeto = 1
        imwrite(invertedBinaryImg, fullfile(binaryDir, files(f).name));

        bw = logical(invertedBinaryImg);
        [rows, cols] = size(bw);

        % Etiquetado 8-conectividad (stack flood fill)
        labeled = zeros(rows, cols, 'int32');
        currentLabel = int32(0);
        offsets = [-1,-1; -1,0; -1,1; 0,-1; 0,1; 1,-1; 1,0; 1,1];
        stack = zeros(rows*cols,2,'int32');

        for r = 1:rows
            for c = 1:cols
                if bw(r,c) && labeled(r,c) == 0
                    currentLabel = currentLabel + 1;
                    top = int32(1);
                    stack(top,:) = [r,c];
                    while top > 0
                        pr = stack(top,1); pc = stack(top,2);
                        top = top - 1;
                        if labeled(pr,pc) ~= 0, continue; end
                        labeled(pr,pc) = currentLabel;
                        for o = 1:8
                            nr = pr + offsets(o,1); nc = pc + offsets(o,2);
                            if nr>=1 && nr<=rows && nc>=1 && nc<=cols && bw(nr,nc) && labeled(nr,nc)==0
                                top = top + 1;
                                stack(top,:) = [nr,nc];
                            end
                        end
                    end
                end
            end
        end

        numObjects = double(currentLabel);
        fprintf('Image: %s  Objects: %d\n', files(f).name, numObjects);

        % Guardar una imagen coloreada simple para ver etiquetas
        rgbOut = zeros(rows, cols, 3, 'uint8');
        if numObjects > 0
            colors = uint8(mod((1:numObjects)' * [61,97,137], 256));
            for L = 1:numObjects
                maskL = (labeled == L);
                if any(maskL(:))
                    rgbOut(:,:,1) = rgbOut(:,:,1) + uint8(maskL) * colors(L,1);
                    rgbOut(:,:,2) = rgbOut(:,:,2) + uint8(maskL) * colors(L,2);
                    rgbOut(:,:,3) = rgbOut(:,:,3) + uint8(maskL) * colors(L,3);
                end
            end
        end
        imwrite(rgbOut, fullfile(countDir, files(f).name));

        % Calcular propiedades manualmente por objeto
        for L = 1:numObjects
            [rr, cc] = find(labeled == L);
            nPix = numel(rr);
            area = nPix;
            % centroide (x=col, y=row)
            cx = mean(cc); cy = mean(rr);

            % perímetro (píxel con algún 4-vecino que no sea del mismo label)
            perim = 0;
            for p = 1:nPix
                r0 = rr(p); c0 = cc(p);
                isPerim = false;
                if r0-1 < 1 || labeled(r0-1,c0) ~= L, isPerim = true; end
                if r0+1 > rows || labeled(r0+1,c0) ~= L, isPerim = true; end
                if c0-1 < 1 || labeled(r0,c0-1) ~= L, isPerim = true; end
                if c0+1 > cols || labeled(r0,c0+1) ~= L, isPerim = true; end
                if isPerim, perim = perim + 1; end
            end

            % covarianza y ángulo de orientación + excentricidad
            if nPix > 1
                X = [cc - cx, rr - cy]; % n x 2 (x,y)=(col,row)
                C = (X' * X) / (nPix - 1);
                [V, D] = eig(C);
                [~, idxMax] = max(diag(D));
                v = V(:, idxMax);
                angle = atan2(v(2), v(1)); % radianes
                lambda = diag(D);
                a = sqrt(lambda(idxMax)); b = sqrt(lambda(3-idxMax));
                if a > 0
                    eccentricity = sqrt(max(0, 1 - (b^2)/(a^2)));
                else
                    eccentricity = 0;
                end
            else
                angle = 0; eccentricity = 0;
            end

            % contorno (traza Moore) y propiedades adicionales
            maskL = (labeled == L);
            boundary = traceBoundary(maskL); % Nx2 [x,y]
            if isempty(boundary)
                hullArea = area; hullV = 0; cornerCnt = 0; solidity = 1; bboxAspect = 1; concaves = 0;
            else
                hull = convexHull(boundary);
                hullArea = polygonArea(hull(:,1), hull(:,2));
                if hullArea <= 0, solidity = 1; else solidity = area / hullArea; end
                bboxW = max(boundary(:,1)) - min(boundary(:,1)) + 1;
                bboxH = max(boundary(:,2)) - min(boundary(:,2)) + 1;
                bboxAspect = max(bboxW/bboxH, bboxH/bboxW);
                cornerCnt = countCorners(boundary);
                concaves = max(0, cornerCnt - size(hull,1));
                hullV = size(hull,1);
            end

            % circularity
            circ = 4 * pi * area / ((perim^2) + eps);

            % almacenar fila de características
            feat = [area, perim, cx, cy, angle, eccentricity, circ, solidity, bboxAspect, cornerCnt, concaves];
            allFeatures = [allFeatures; feat]; %#ok<AGROW>
            objectRefs(end+1,1:2) = {files(f).name, L}; %#ok<AGROW>
        end
    end

    % Si no hay objetos, salir
    if isempty(allFeatures)
        fprintf('No objects found in dataset.\n');
        return;
    end

    % Normalizar características para clustering
    % usaremos columnas: area, perim, circ, eccentricity, solidity, bboxAspect, cornerCnt, concaves
    X = allFeatures(:, [1,2,7,6,8,9,10,11]);
    mu = mean(X,1); sigma = std(X,0,1) + eps;
    Xnorm = (X - mu) ./ sigma;

    k = 5; maxIter = 200;
    % kmeans euclidiano (implementado)
    rng(0); % reproducible
    [idxE, centE] = kmeans_euclid(Xnorm, k, maxIter);

    % kmeans con Mahalanobis: usar cov de X (no normalizada) para distancia
    Cx = cov(X) + eye(size(X,2))*1e-6;
    invCx = pinv(Cx);
    [idxM, centM] = kmeans_mahal(X, k, invCx, maxIter);

    % Mapear clusters a etiquetas usando heurísticas sobre medias
    labelsE = mapClustersToLabels(X, idxE, k, objectRefs);
    labelsM = mapClustersToLabels(X, idxM, k, objectRefs);

    % Guardar resultados por objeto e imágenes
    % Columnas CSV: Image,ObjectID,Area,Perimeter,CentroidX,CentroidY,AngleRad,Eccentricity,Circularity,Solidity,BBoxAspect,CornerCount,ConcaveCount,Label_Euclid,Label_Mahalanobis
    outDir = countDir;
    % Para construir combined CSV, agrupar por imagen
    combinedRows = {};
    for i = 1:size(allFeatures,1)
        imgname = objectRefs{i,1}; objid = objectRefs{i,2};
        row = [ {imgname}, objid, num2cell(allFeatures(i,1:2)), num2cell(allFeatures(i,3:5)), num2cell(allFeatures(i,6:11)), {labelsE{i}}, {labelsM{i}} ];
        combinedRows(end+1,:) = row; %#ok<AGROW>
    end

    % escribir per-image CSVs
    uniqueImages = unique(objectRefs(:,1));
    for ui = 1:length(uniqueImages)
        nm = uniqueImages{ui};
        rowsForImg = find(strcmp(objectRefs(:,1), nm));
        csvPath = fullfile(outDir, [nm '_properties.csv']);
        fid = fopen(csvPath, 'w');
        fprintf(fid, 'Image,ObjectID,Area,Perimeter,CentroidX,CentroidY,AngleRad,Eccentricity,Circularity,Solidity,BBoxAspect,CornerCount,ConcaveCount,Label_Euclid,Label_Mahalanobis\n');
        for j = 1:length(rowsForImg)
            r = rowsForImg(j);
            f = allFeatures(r,:);
            fprintf(fid, '%s,%d,%.0f,%.0f,%.4f,%.4f,%.6f,%.6f,%.6f,%.6f,%.4f,%.0f,%.0f,%s,%s\n', nm, objectRefs{r,2}, f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8), f(9), f(10), f(11), labelsE{r}, labelsM{r});
        end
        fclose(fid);
    end

    % escribir combined CSV
    finalCsv = fullfile(outDir, 'combined_results.csv');
    fid = fopen(finalCsv,'w');
    fprintf(fid, 'Image,ObjectID,Area,Perimeter,CentroidX,CentroidY,AngleRad,Eccentricity,Circularity,Solidity,BBoxAspect,CornerCount,ConcaveCount,Label_Euclid,Label_Mahalanobis\n');
    for r = 1:size(allFeatures,1)
        f = allFeatures(r,:);
        fprintf(fid, '%s,%d,%.0f,%.0f,%.4f,%.4f,%.6f,%.6f,%.6f,%.6f,%.4f,%.0f,%.0f,%s,%s\n', objectRefs{r,1}, objectRefs{r,2}, f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8), f(9), f(10), f(11), labelsE{r}, labelsM{r});
    end
    fclose(fid);

    fprintf('Finished. Combined CSV: %s\n', finalCsv);
end

% ------------------------
% Aux functions
% ------------------------
function boundary = traceBoundary(mask)
% Moore boundary trace - returns Nx2 [x,y] (col,row)
    [rows, cols] = size(mask);
    maxLen = rows * cols;
    boundary = zeros(maxLen,2);
    step = 0;

    % find start pixel: foreground with a background 4-neighbor
    found = false;
    for r = 1:rows
        for c = 1:cols
            if mask(r,c)
                if (r>1 && ~mask(r-1,c)) || (r<rows && ~mask(r+1,c)) || (c>1 && ~mask(r,c-1)) || (c<cols && ~mask(r,c+1))
                    start = [r,c]; found = true; break;
                end
            end
        end
        if found, break; end
    end
    if ~found
        boundary = zeros(0,2); return;
    end

    nbrs = [-1 -1; -1 0; -1 1; 0 1; 1 1; 1 0; 1 -1; 0 -1];
    cur = start; prev_dir = 8;
    visited = zeros(rows,cols,'uint16');

    while true
        step = step + 1;
        boundary(step,1) = cur(2);
        boundary(step,2) = cur(1);
        visited(cur(1), cur(2)) = visited(cur(1), cur(2)) + 1;
        if visited(cur(1), cur(2)) > 8, break; end

        foundNext = false;
        sdir = mod(prev_dir-2,8) + 1;
        for k = 0:7
            idx = mod(sdir-1+k,8) + 1;
            nr = cur(1) + nbrs(idx,1); nc = cur(2) + nbrs(idx,2);
            if nr>=1 && nr<=rows && nc>=1 && nc<=cols && mask(nr,nc)
                prev_dir = idx; cur = [nr,nc]; foundNext = true; break;
            end
        end
        if ~foundNext, break; end
        if step > 1 && cur(1)==start(1) && cur(2)==start(2), break; end
        if step >= maxLen, break; end
    end

    if step < maxLen, boundary = boundary(1:step, :); end
end

function H = convexHull(P)
% Andrew's monotone chain. P Nx2 [x,y]
    if isempty(P), H = zeros(0,2); return; end
    % unique rows
    [~, ia, ~] = unique(P, 'rows', 'stable');
    P = P(sort(ia), :);
    pts = sortrows(P, [1 2]);
    n = size(pts,1);
    if n <= 1, H = pts; return; end
    lower = zeros(n,2); m = 0;
    for i = 1:n
        while m >= 2 && cross2d(lower(m,:) - lower(m-1,:), pts(i,:) - lower(m,:)) <= 0
            m = m - 1;
        end
        m = m + 1; lower(m,:) = pts(i,:);
    end
    lower = lower(1:m,:);
    up = zeros(n,2); m2 = 0;
    for i = n:-1:1
        while m2 >= 2 && cross2d(up(m2,:) - up(m2-1,:), pts(i,:) - up(m2,:)) <= 0
            m2 = m2 - 1;
        end
        m2 = m2 + 1; up(m2,:) = pts(i,:);
    end
    up = up(1:m2,:);
    H = [lower; up(2:end-1,:)];
    if isempty(H), H = pts; end
end

function A = polygonArea(x,y)
    x = x(:); y = y(:); n = numel(x);
    if n < 3, A = 0; return; end
    x2 = [x(2:end); x(1)]; y2 = [y(2:end); y(1)];
    A = 0.5 * abs(sum(x .* y2 - x2 .* y));
end

function c = cross2d(a,b), c = a(1)*b(2) - a(2)*b(1); end

function cnt = countCorners(boundary)
% curvatura simple en boundary Nx2
    n = size(boundary,1);
    if n < 6, cnt = 0; return; end
    w = max(1, round(n * 0.02));
    angles = zeros(n,1);
    for i = 1:n
        im = mod(i-1-w, n) + 1; ip = mod(i-1+w, n) + 1;
        v1 = boundary(im,:) - boundary(i,:);
        v2 = boundary(ip,:) - boundary(i,:);
        nv1 = norm(v1); nv2 = norm(v2);
        if nv1*nv2 == 0, angles(i) = 0; else
            cosang = max(-1, min(1, (v1*v2')/(nv1*nv2)));
            angles(i) = acos(cosang);
        end
    end
    angles = movmean(angles, max(3, 2*w+1));
    thresh = 0.6; isPeak = (angles > thresh);
    cnt = 0; inP = false;
    for i = 1:n
        if isPeak(i) && ~inP, cnt = cnt + 1; inP = true; elseif ~isPeak(i), inP = false; end
    end
end

function [idx, centroids] = kmeans_euclid(X, k, maxIter)
% simple kmeans with euclidean distance
    [n, d] = size(X);
    if k > n
        k = n;
    end
    perm = randperm(n);
    cIdx = perm(1:k);
    centroids = X(cIdx, :);
    idx = zeros(n,1);
    for it = 1:maxIter
        % assign
        for i = 1:n
            D = sum((centroids - X(i,:)).^2, 2);
            [~, idx(i)] = min(D);
        end
        % update
        moved = false;
        for j = 1:k
            members = X(idx==j, :);
            if isempty(members), continue; end
            newC = mean(members,1);
            if any(abs(newC - centroids(j,:)) > 1e-6), moved = true; end
            centroids(j,:) = newC;
        end
        if ~moved, break; end
    end
end

function [idx, centroids] = kmeans_mahal(X, k, invC, maxIter)
% kmeans using Mahalanobis distance defined by invC (for raw X)
    [n, d] = size(X);
    if k > n
        k = n;
    end
    perm = randperm(n);
    cIdx = perm(1:k);
    centroids = X(cIdx, :);
    idx = zeros(n,1);
    for it = 1:maxIter
        for i = 1:n
            D = zeros(k,1);
            for j = 1:k
                diff = X(i,:) - centroids(j,:);
                D(j) = diff * invC * diff';
            end
            [~, idx(i)] = min(D);
        end
        moved = false;
        for j = 1:k
            members = X(idx==j, :);
            if isempty(members), continue; end
            newC = mean(members,1);
            if any(abs(newC - centroids(j,:)) > 1e-6), moved = true; end
            centroids(j,:) = newC;
        end
        if ~moved, break; end
    end
end

function labels = mapClustersToLabels(Xraw, idx, k, objectRefs)
% Prototype-based mapping with normalized features. This prints and saves
% cluster means to help tuning and assigns the best-matching prototype.
    labels = cell(size(Xraw,1),1);
    % columns: [area, perim, circ, eccentricity, solidity, bboxAspect, cornerCnt, concaves]
    % compute cluster means
    means = zeros(k, size(Xraw,2));
    for j = 1:k
        members = Xraw(idx==j, :);
        if isempty(members)
            means(j,:) = NaN;
        else
            means(j,:) = mean(members,1);
        end
    end

    % print cluster means for debugging
    fprintf('Cluster means (for mapping)\n');
    for j = 1:k
        fprintf('Cluster %d: ', j);
        if any(isnan(means(j,:)))
            fprintf('empty\n');
        else
            fprintf('area=%.1f perim=%.1f circ=%.3f ecc=%.3f sol=%.3f aspect=%.3f corners=%.1f concaves=%.1f\n', means(j,1), means(j,2), means(j,3), means(j,4), means(j,5), means(j,6), means(j,7), means(j,8));
        end
    end

    % save cluster means to CSV to help tuning
    try
        outdir = pwd();
        csvP = fullfile(outdir, 'cluster_means_debug.csv');
        fid = fopen(csvP,'w');
        fprintf(fid, 'Cluster,Area,Perim,Circularity,Eccentricity,Solidity,BBoxAspect,CornerCount,ConcaveCount\n');
        for j = 1:k
            if any(isnan(means(j,:)))
                fprintf(fid, '%d,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN\n', j);
            else
                m = means(j,:);
                fprintf(fid, '%d,%.3f,%.3f,%.6f,%.6f,%.6f,%.6f,%.3f,%.3f\n', j, m(1), m(2), m(3), m(4), m(5), m(6), m(7), m(8));
            end
        end
        fclose(fid);
        fprintf('Saved cluster_means_debug.csv for inspection.\n');
    catch
        % ignore write errors
    end

    % normalize cluster means and define prototypes in original feature space
    globalMu = mean(Xraw,1);
    globalSigma = std(Xraw,0,1) + eps;

    % Try to build prototypes from labeled image ranges you provided:
    % 1-5: cuadrado, 6-10: circulo, 11-15: triangulo, 16-20: estrella, 21-25: flecha
    prot = struct();
    % defaults (fallback)
    prot.circulo    = [globalMu(1), globalMu(2), 0.90, 0.10, 0.98, 1.00, 1, 0];
    prot.cuadrado   = [globalMu(1), globalMu(2), 0.70, 0.20, 0.95, 1.05, 4, 0];
    prot.triangulo  = [globalMu(1), globalMu(2), 0.55, 0.60, 0.90, 1.00, 3, 0];
    prot.estrella   = [globalMu(1), globalMu(2), 0.30, 0.40, 0.60, 1.00, 8, 4];
    prot.flecha     = [globalMu(1), globalMu(2), 0.45, 0.40, 0.90, 1.30, 6, 1];
    % attempt to override prototypes using data from the images ranges
    try
        N = size(Xraw,1);
        % build a numeric image index for each object
        imgIdx = nan(N,1);
        for t = 1:N
            nm = objectRefs{t,1};
            % extract leading number before extension
            [tok, ~] = strtok(nm, '.');
            val = str2double(tok);
            if ~isnan(val), imgIdx(t) = val; end
        end
        % helper to compute prototype mean for a set of image indices
        ranges = [1 5; 6 10; 11 15; 16 20; 21 25];
        p_sq = []; p_ci = []; p_tr = []; p_es = []; p_fl = [];
        for ri = 1:size(ranges,1)
            r0 = ranges(ri,:);
            mask = (~isnan(imgIdx)) & (imgIdx >= r0(1)) & (imgIdx <= r0(2));
            if any(mask)
                pm = mean(Xraw(mask,:), 1);
            else
                pm = [];
            end
            switch ri
                case 1, p_sq = pm;
                case 2, p_ci = pm;
                case 3, p_tr = pm;
                case 4, p_es = pm;
                case 5, p_fl = pm;
            end
        end
        if ~isempty(p_sq), prot.cuadrado = p_sq; end
        if ~isempty(p_ci), prot.circulo = p_ci; end
        if ~isempty(p_tr), prot.triangulo = p_tr; end
        if ~isempty(p_es), prot.estrella = p_es; end
        if ~isempty(p_fl), prot.flecha = p_fl; end
    catch
        % if anything fails, keep default prot values
    end

    protoNames = fieldnames(prot);
    P = zeros(length(protoNames), size(Xraw,2));
    for pi = 1:length(protoNames)
        P(pi,:) = prot.(protoNames{pi});
    end
    % normalize prototypes to z-space
    Pz = (P - globalMu) ./ globalSigma;

    % normalize cluster means to z-space
    Mz = (means - globalMu) ./ globalSigma;

    % weights emphasize useful features: circ, eccentricity, solidity, corners, concaves, aspect
    w = [0, 0, 2.0, 1.2, 1.5, 1.0, 1.5, 1.5];

    % assign each cluster to nearest prototype (weighted Euclidean in z-space)
    clusterLabel = cell(k,1);
    for j = 1:k
        if any(isnan(Mz(j,:)))
            clusterLabel{j} = 'otra';
            continue;
        end
        dmin = inf; best = 'otra';
        for pi = 1:size(Pz,1)
            diff = (Mz(j,:) - Pz(pi,:));
            dist = sqrt( sum( (w .* diff).^2 ) );
            if dist < dmin
                dmin = dist; best = protoNames{pi};
            end
        end
        clusterLabel{j} = best;
    end

    % map labels to all members
    for j = 1:k
        labels(idx==j) = {clusterLabel{j}};
    end
end

processImages();