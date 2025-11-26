clear;
clc;
%% Lectura del Dataset de Imagenes
datasetFolder = 'FIGURAS';
imagePattern = '*.png';
%Listar todas las imagenes que coinciden con el patrón en la carpeta
fileList = dir(fullfile(datasetFolder, imagePattern));
numImages = length(fileList);
%Extraer los números de imagen para el ordenamiento
imageNumbers = zeros(numImages, 1);
for i = 1:numImages
    fileName = fileList(i).name;
    
    % Extraer el primer número que aparezca en el nombre del archivo (p.ej. '10' en '10_rot90.png')
    tokens = regexp(fileName, '(\d+)', 'tokens', 'once');
    if ~isempty(tokens)
        imageNumbers(i) = str2double(tokens{1});
    else
        % Si no hay número, poner un valor grande para ordenar al final
        imageNumbers(i) = 1e9 + i;
    end
end
%Ordenar los números de forma ascendente
[~, sortIdx] = sort(imageNumbers);
fileListSorted = fileList(sortIdx);
disp(['Se encontraron y ordenaron ', num2str(numImages), ' imágenes.']);
imageList = cell(numImages, 1);
%Leer cada imagen y guardarla en la celda
for i = 1:numImages
    % Construir la ruta completa del archivo
    fullImagePath = fullfile(datasetFolder, fileListSorted(i).name);
    
    % Leer la imagen
    img = imread(fullImagePath);
    % Guardar la imagen leída en la celda
    imageList{i} = img;
    % Crear máscara recortada para cálculo de propiedades
    disp(['Imagen ', num2str(i), ' leída: ', fileListSorted(i).name]);
end
 
% %Obtencion de los prototipos para las clases (características extendidas)
numClases = 5;
% Preferir archivos base conocidos como prototipos según convención del dataset:
% 1-5: cuadrados, 6-10: circulos, 11-15: triangulos, 16-20: estrellas, 21-25: flechas
% Usaremos los ejemplares base: 1,6,11,16,21 (si existen) para cada clase.
basePrototypeNumbers = [1, 6, 11, 16, 21];
% Reconstruir array de números de archivo ordenado para buscar coincidencias
sortedNumbers = imageNumbers(sortIdx);
prototypeIndices = zeros(numClases,1);
% disp('--- DEBUG: Archivos elegidos explícitamente como prototipos ---');
for ii = 1:numClases
    target = basePrototypeNumbers(ii);
    idx = find(sortedNumbers == target, 1);
    if isempty(idx)
        % si no se encuentra el número exacto, buscar el más cercano por valor absoluto
        [~, idx] = min(abs(sortedNumbers - target));
        warning(['No se encontró prototipo con número ', num2str(target), '; usando número cercano: ', num2str(sortedNumbers(idx))]);
    end
    prototypeIndices(ii) = idx;
    fn = fileListSorted(idx).name;
    try
        info = imfinfo(fullfile(datasetFolder, fn));
        disp([num2str(ii), ': ', fn, '  (numero=', num2str(sortedNumbers(idx)), ') Size: ', num2str(info.Width), 'x', num2str(info.Height)]);
    catch
        disp([num2str(ii), ': ', fn, '  (numero=', num2str(sortedNumbers(idx)), ')']);
    end
end
prototipoFeatures = cell(numClases, 1);
disp('Iniciando cálculo de características extendidas para los prototipos...');
minAreaThreshold = 100;
conectividad = 8;
vecinos = [-1, 0; 1, 0; 0, -1; 0, 1; -1, -1; -1, 1; 1, -1; 1, 1];

% Definimos el orden de las características extendidas que usaremos internamente:
% 1) Circularity (4*pi*Area/Perimeter^2)
% 2) AspectRatio (width/height)
% 3) Excentricity (como antes)
% 4) Solidity (Area/ConvexArea)
% 5) Extent (Area/BoundingBoxArea)
% 6) Perimeter
% 7) Hu1 (primer momento de Hu, absoluto)
% 8) CornerCount (número de esquinas detectadas)
for i = 1:numClases
    % Usar el índice seleccionado explícitamente para cada prototipo
    selIdx = prototypeIndices(i);
    imgColor = imageList{selIdx};
    imgDouble = im2double(imgColor);

    % Segmentación por Color (igual que antes)
    isObject = (imgDouble(:,:,1) < 0.9) | (imgDouble(:,:,2) < 0.9) | (imgDouble(:,:,3) < 0.9);

    % Etiquetado (BFS) como antes
    [rows_img, cols_img] = size(isObject);
    L = zeros(rows_img, cols_img);
    currentLabel = 0;
    for r = 1:rows_img
        for c = 1:cols_img
            if isObject(r, c) == 1 && L(r, c) == 0
                currentLabel = currentLabel + 1;
                queue = [r, c];
                L(r, c) = currentLabel;
                while ~isempty(queue)
                    pixel_actual = queue(1, :);
                    pr = pixel_actual(1);
                    pc = pixel_actual(2);
                    queue(1, :) = [];
                    for k_vecino = 1:conectividad
                        nr = pr + vecinos(k_vecino, 1);
                        nc = pc + vecinos(k_vecino, 2);
                        if nr >= 1 && nr <= rows_img && nc >= 1 && nc <= cols_img
                            if isObject(nr, nc) == 1 && L(nr, nc) == 0
                                L(nr, nc) = currentLabel;
                                queue(end+1, :) = [nr, nc];
                            end
                        end
                    end
                end
            end
        end
    end

    features_list = [];
    for k = 1:currentLabel
        objetoMask = (L == k);
        area = nnz(objetoMask);
        if area < minAreaThreshold, continue; end

        % Regionprops para obtener varias propiedades útiles
        rp = regionprops(objetoMask, 'Perimeter', 'BoundingBox', 'ConvexArea', 'Extent', 'Eccentricity', 'Centroid', 'Orientation');
        if isempty(rp), continue; end
        perimeter = rp.Perimeter;
        bbox = rp.BoundingBox; % [x y width height]
        ancho = bbox(3);
        alto = bbox(4);
        areaBoundingBox = ancho * alto;
        convexArea = rp.ConvexArea;
        solidity = 1.0;
        if convexArea > 0, solidity = area / convexArea; end
        extent = rp.Extent;
        aspectRatio = ancho / max(alto, eps);

        % Circularidad estándar (más robusta)
        if perimeter > 0
            circularity = (4 * pi * double(area)) / (perimeter ^ 2);
        else
            circularity = 0;
        end

        % Excentricidad desde regionprops
        Excentricity = rp.Eccentricity;

        % Hu moment (primer invariante) — comprimir con log para estabilidad
        hu = computeHuMoments(objetoMask);
        hu1 = log(1 + abs(hu(1)));

        % Perímetro normalizado (adimensional)
        if area > 0
            perimeter_norm = perimeter / sqrt(double(area));
        else
            perimeter_norm = 0;
        end

        % Convex hull vertex count (use convhull on hull boundary)
        hullVertices = 0;
        try
            hullMask = bwconvhull(objetoMask);
            Bh = bwboundaries(hullMask, 'noholes');
            if ~isempty(Bh)
                bh = Bh{1}; xyh = [bh(:,2), bh(:,1)];
                % Compute convex hull indices and count vertices (convhull repeats first point)
                ch = convhull(xyh(:,1), xyh(:,2));
                hullVertices = max(0, length(ch) - 1);
            end
        catch
            hullVertices = 0;
        end

        % Contar picos radiales en la frontera (shape signature)
        radialPeaks = 0;
        try
            B = bwboundaries(objetoMask, 'noholes');
            if ~isempty(B)
                % tomar el contorno principal
                maxL = 0; idm = 1;
                for ii = 1:length(B)
                    if size(B{ii},1) > maxL
                        maxL = size(B{ii},1); idm = ii;
                    end
                end
                bd = B{idm}; xy = [bd(:,2), bd(:,1)];
                centroid = rp.Centroid;
                dists = sqrt((xy(:,1)-centroid(1)).^2 + (xy(:,2)-centroid(2)).^2);
                % suavizar con media móvil simple
                w = max(3, round(length(dists)/40));
                smoothd = movmean(dists, w);
                % contar máximos locales simples
                m = mean(smoothd) + 0.1 * std(smoothd);
                for ii = 2:(length(smoothd)-1)
                    if smoothd(ii) > smoothd(ii-1) && smoothd(ii) > smoothd(ii+1) && smoothd(ii) > m
                        radialPeaks = radialPeaks + 1;
                    end
                end
            end
        catch
            radialPeaks = 0;
        end

        % Orientation absolute (0..180)
        orientation_abs = abs(rp.Orientation);

        % Evitar valores extremos
        hullVertices = min(hullVertices, 20);
        radialPeaks = min(radialPeaks, 20);

        % Empaquetar nuevo vector de características (orden consistente)
        % [circularity, solidity, extent, aspectRatio, Excentricity, perimeter_norm, hullVertices, radialPeaks, hu1, orientation_abs]
        features_list = [features_list; double([circularity, solidity, extent, aspectRatio, Excentricity, perimeter_norm, hullVertices, radialPeaks, hu1, orientation_abs])];
    end

    if ~isempty(features_list)
        % Select medoid (la muestra real más representativa) en lugar de la mediana
        try
            Dloc = pdist2(features_list, features_list, 'euclidean');
            sumD = sum(Dloc, 2);
            [~, medoidIdx] = min(sumD);
            prototipoFeatures{i} = features_list(medoidIdx, :);
            protoSamples{i} = features_list;
            % Store medoid index for debug
            protoMedoidIndex{i} = medoidIdx;
        catch
            % Fallback a median si pdist2 no está disponible
            prototipoFeatures{i} = median(features_list, 1);
            protoSamples{i} = features_list;
            protoMedoidIndex{i} = 1;
        end
    else
        warning(['No se pudo calcular el prototipo para la Clase ', num2str(i)]);
    end
    disp(['  > Características extendidas de la Clase ', num2str(i), ' extraídas.']);
end

% Debug: report number of samples per class and a few sample rows to detect duplicates
% disp('--- DEBUG: protoSamples per clase (counts y primeras filas) ---');
for ci = 1:numClases
    if exist('protoSamples','var') && ci <= numel(protoSamples) && ~isempty(protoSamples{ci})
        cnt = size(protoSamples{ci}, 1);
        disp(['Clase ', num2str(ci), ' muestras: ', num2str(cnt)]);
        % show up to first 3 rows
        showN = min(3, cnt);
        for rr = 1:showN
            disp(protoSamples{ci}(rr, :));
        end
        % If medoid index known, show it
        if exist('protoMedoidIndex','var') && ci <= numel(protoMedoidIndex)
            disp(['  MedoidIndex: ', num2str(protoMedoidIndex{ci})]);
            disp(['  MedoidRow:']); disp(protoSamples{ci}(protoMedoidIndex{ci}, :));
        end
    else
        disp(['Clase ', num2str(ci), ' muestras: 0']);
    end
end

% Construir matriz final de prototipos RAW (numClases x numFeatures)
% Buscar el primer prototipo no vacío para determinar la dimensión de características
featCount = 0;
for t = 1:numClases
    if ~isempty(prototipoFeatures{t})
        featCount = numel(prototipoFeatures{t});
        break;
    end
end
if featCount == 0
    error('No se calcularon prototipos válidos. Revise las imágenes de prototipo.');
end
finalPrototypes_raw = zeros(numClases, featCount);
for i = 1:numClases
    features = prototipoFeatures{i};
    if ~isempty(features)
        finalPrototypes_raw(i, :) = features;
    end
end
disp('--- Prototipos RAW extendidos definidos ---');
disp(finalPrototypes_raw);
% Si varios prototipos son casi idénticos, aplicar pequeño jitter para evitar
% distancias inter-clase cero que bloqueen la clasificación.
try
    protoD = pdist(finalPrototypes_raw);
    if any(protoD < 1e-9)
        warning('Se detectaron prototipos duplicados o muy similares; aplicando jitter pequeño.');
        rng(1);
        for ii = 1:numClases
            finalPrototypes_raw(ii, :) = finalPrototypes_raw(ii, :) + 1e-6 * randn(1, featCount);
        end
    end
catch
    % Si pdist no disponible o falla, ignoramos
end
% Construir matriz de muestras de prototipo normalizadas para k-NN
protoSamplesAll_raw = [];
protoLabels = [];
for ci = 1:numClases
    if exist('protoSamples','var') && ci <= numel(protoSamples) && ~isempty(protoSamples{ci})
        protoSamplesAll_raw = [protoSamplesAll_raw; protoSamples{ci}];
        protoLabels = [protoLabels; repmat(ci, size(protoSamples{ci},1), 1)];
    end
end
%% NORMALIZACIÓN Y CLASIFICACIÓN (rangos dinámicos según prototipos extendidos)
% Construimos rangos por característica tomando el mínimo/máximo observado
featCount = size(finalPrototypes_raw, 2);
minVals = min(finalPrototypes_raw, [], 1);
maxVals = max(finalPrototypes_raw, [], 1);
RANGES = zeros(featCount, 2);
for f = 1:featCount
    mn = minVals(f);
    mx = maxVals(f);
    if mn == mx
        % Si todos los prototipos tienen el mismo valor, damos un margen fijo
        mn = mn - 0.1 * abs(mn + eps);
        mx = mx + 0.1 * abs(mx + eps) + 1.0;
    else
        delta = (mx - mn) * 0.2; % margen del 20%
        mn = mn - delta;
        mx = mx + delta;
    end
    RANGES(f, :) = [mn, mx];
end

% Normalizar prototipos (0..1)
finalPrototypes = zeros(numClases, featCount);
for c = 1:numClases
    for f = 1:featCount
        denom = (RANGES(f, 2) - RANGES(f, 1));
        if denom == 0
            finalPrototypes(c, f) = 0.0;
        else
            finalPrototypes(c, f) = (finalPrototypes_raw(c, f) - RANGES(f, 1)) / denom;
        end
    end
end
finalPrototypes(finalPrototypes > 1) = 1;
finalPrototypes(finalPrototypes < 0) = 0;
disp('--- Prototipos de Clase Normalizados (extendidos) ---');
disp(finalPrototypes);
% Normalizar las muestras de prototipo (si existen)
protoSamplesAll = [];
if exist('protoSamplesAll_raw','var') && ~isempty(protoSamplesAll_raw)
    protoSamplesAll = zeros(size(protoSamplesAll_raw));
    for f = 1:featCount
        denom = (RANGES(f,2) - RANGES(f,1));
        if denom == 0
            protoSamplesAll(:, f) = 0.0;
        else
            protoSamplesAll(:, f) = (protoSamplesAll_raw(:, f) - RANGES(f,1)) / denom;
        end
    end
    protoSamplesAll(protoSamplesAll > 1) = 1;
    protoSamplesAll(protoSamplesAll < 0) = 0;
end
% ---------------------------------------------------------------------
% Mejora: recolectar muestras de prototipo desde todo el dataset usando
% la convención numérica dada por el usuario (1-5 cuadrados, 6-10 círculos,
% 11-15 triángulos, 16-20 estrellas, 21-25 flechas) y recalcular prototipos
% como medoids sobre todas las muestras encontradas por clase.
disp('Recolectando muestras adicionales para prototipos según convención 1..25...');
protoSamples = cell(numClases,1);
for jj = 1:numImages
    num = imageNumbers(sortIdx(jj));
    if num >= 1 && num <= 25
        classIdx = min(numClases, ceil(num / 5));
        try
            img = imageList{jj};
            fv = computeFeatureVectorFromImage(img, minAreaThreshold);
            if ~isempty(fv)
                protoSamples{classIdx} = [protoSamples{classIdx}; fv];
            end
        catch
            % ignorar imagenes no procesables
        end
    end
end

% Si se encontraron muestras, recalcular los prototipos (medoid por clase)
for ci = 1:numClases
    if ~isempty(protoSamples{ci})
        X = protoSamples{ci};
        try
            D = pdist2(X, X, 'euclidean');
            s = sum(D, 2);
            medoidIdx = find(s == min(s), 1);
            finalPrototypes_raw(ci, :) = X(medoidIdx, :);
            protoMedoidIndex{ci} = medoidIdx;
        catch
            % si falla pdist2, usar la primera muestra
            finalPrototypes_raw(ci, :) = X(1, :);
            protoMedoidIndex{ci} = 1;
        end
    else
        warning(['No se encontraron muestras para la clase ', num2str(ci), ' en los rangos 1..25.']);
    end
end

% Reconstruir protoSamplesAll_raw y protoLabels a partir de protoSamples
protoSamplesAll_raw = [];
protoLabels = [];
for ci = 1:numClases
    if ~isempty(protoSamples{ci})
        protoSamplesAll_raw = [protoSamplesAll_raw; protoSamples{ci}];
        protoLabels = [protoLabels; repmat(ci, size(protoSamples{ci},1), 1)];
    end
end

% Actualizar normalización con los prototipos recalculados
minVals = min(finalPrototypes_raw, [], 1);
maxVals = max(finalPrototypes_raw, [], 1);
for f = 1:featCount
    mn = minVals(f);
    mx = maxVals(f);
    if mn == mx
        mn = mn - 0.1 * abs(mn + eps);
        mx = mx + 0.1 * abs(mx + eps) + 1.0;
    else
        delta = (mx - mn) * 0.2;
        mn = mn - delta;
        mx = mx + delta;
    end
    RANGES(f, :) = [mn, mx];
end
% Re-normalizar prototipos y muestras
for c = 1:numClases
    for f = 1:featCount
        denom = (RANGES(f, 2) - RANGES(f, 1));
        if denom == 0
            finalPrototypes(c, f) = 0.0;
        else
            finalPrototypes(c, f) = (finalPrototypes_raw(c, f) - RANGES(f, 1)) / denom;
        end
    end
end
finalPrototypes(finalPrototypes > 1) = 1;
finalPrototypes(finalPrototypes < 0) = 0;
if ~isempty(protoSamplesAll_raw)
    protoSamplesAll = zeros(size(protoSamplesAll_raw));
    for f = 1:featCount
        denom = (RANGES(f,2) - RANGES(f,1));
        if denom == 0
            protoSamplesAll(:, f) = 0.0;
        else
            protoSamplesAll(:, f) = (protoSamplesAll_raw(:, f) - RANGES(f,1)) / denom;
        end
    end
    protoSamplesAll(protoSamplesAll > 1) = 1;
    protoSamplesAll(protoSamplesAll < 0) = 0;
end
disp('Recalculo de prototipos completado (medoids sobre muestras 1..25).');
disp('--- Prototipos RAW recalculados ---');
disp(finalPrototypes_raw);
% DEBUG: mostrar cuántas muestras por clase se recopilaron tras la recolección 1..25
% disp('--- DEBUG: Conteo de muestras recolectadas por clase (post-1..25) ---');
for ci = 1:numClases
    if ~isempty(protoSamples) && ci <= numel(protoSamples) && ~isempty(protoSamples{ci})
        disp(['Clase ', num2str(ci), ' muestras recolectadas: ', num2str(size(protoSamples{ci},1))]);
    else
        disp(['Clase ', num2str(ci), ' muestras recolectadas: 0']);
    end
end
%% Extraccion y clasificacion de objetos, guardado de los vectores de caracteristicas en un txt
% Nombres de las 5 clases 
claseNombres = {'Cuadrado', 'Circulo', 'Triangulo', 'Estrella', 'Flecha', 'Otros'};
% Umbral para la clasificación estricta (ligeramente más permisivo)
% Ajustado: subir levemente para reducir 'Otros' en triángulos/variantes
umbralClasificacion = 0.22; 
% Pesos para cada característica (misma longitud que featCount)
% Orden: [circularity, solidity, extent, aspectRatio, Excentricity, perimeter_norm, hullVertices, radialPeaks, hu1, orientation_abs]
% Set tuned feature weights (adjusted to emphasize hull/radial and solidity)
% Order: [circularity, solidity, extent, aspectRatio, Excentricity, perimeter_norm, hullVertices, radialPeaks, hu1, orientation_abs]
featWeights = zeros(1, featCount);
% Tuned weights (empirically chosen)
% circularity: less important, solidity: important, hull/radial: high
baseWeights = [0.5, 1.5, 0.8, 1.0, 0.5, 0.6, 2.5, 2.0, 0.4, 0.0];
for iw = 1:min(length(baseWeights), featCount)
    featWeights(iw) = baseWeights(iw);
end
% If featCount > length(baseWeights), fill remainder with 1.0
if featCount > length(baseWeights)
    featWeights(length(baseWeights)+1:featCount) = 1.0;
end
% Abrir el archivo de texto para guardar los resultados
fileName = 'NoIndexedDB.txt';
fileID = fopen(fileName, 'w'); 
if fileID == -1
    error('No se pudo abrir o crear el archivo NoIndexedDB.txt');
end
% Encabezado actualizado (salida compatible: 3 métricas usadas para registro)
fprintf(fileID, 'ID_Objeto,Circularidad_Simplificada,Relacion_Aspecto,Excentricidad_Simplif,Clase_Asignada\n');
objetoCounter = 0; 
numImagenesProcesadas = 0;
for i = numClases + 1 : length(imageList) 
    
    imgColor = imageList{i}; 
    imgFileName = fileListSorted(i).name;
    imgDouble = im2double(imgColor);
    
    %Segmentación
    isObject = (imgDouble(:,:,1) < 0.9) | (imgDouble(:,:,2) < 0.9) | (imgDouble(:,:,3) < 0.9);
    
    %Etiquetado de Objetos Individuales (BFS)
    [rows, cols] = size(isObject);
    L = zeros(rows, cols); 
    currentLabel = 0;
    
    % Algoritmo BFS
    for r = 1:rows
        for c = 1:cols
            if isObject(r, c) == 1 && L(r, c) == 0
                currentLabel = currentLabel + 1; 
                queue = [r, c]; 
                L(r, c) = currentLabel;
                
                while ~isempty(queue)
                    pixel_actual = queue(1, :);
                    pr = pixel_actual(1); 
                    pc = pixel_actual(2);
                    queue(1, :) = []; 
                
                    for k_vecino = 1:conectividad
                        nr = pr + vecinos(k_vecino, 1);
                        nc = pc + vecinos(k_vecino, 2);
                    
                        if nr >= 1 && nr <= rows && nc >= 1 && nc <= cols
                            if isObject(nr, nc) == 1 && L(nr, nc) == 0
                                L(nr, nc) = currentLabel;
                                queue(end+1, :) = [nr, nc]; 
                            end
                        end
                    end
                end 
            end 
        end 
    end
    numObjects = currentLabel;
    
    if numObjects > 0
        numImagenesProcesadas = numImagenesProcesadas + 1;
        disp(['Procesando imagen de prueba #', num2str(numImagenesProcesadas), ': ', imgFileName, ' con ', num2str(numObjects), ' objetos.']);
    end
    for k = 1:numObjects
    
        objetoMask = (L == k);
        [rows_obj, cols_obj] = find(objetoMask);
    
        if length(rows_obj) < minAreaThreshold % Filtrar ruido
            continue;
        end
    
        objetoCounter = objetoCounter + 1;

        % Se utiliza la función local para la extracción y clasificación
        [~, claseNombreAsignado, circularityProxy, aspectRatio, Excentricity, ~, ~, ~, ~, ~, ~] = extract_and_classify_features(rows_obj, cols_obj, RANGES, finalPrototypes, umbralClasificacion, claseNombres, numClases, protoSamplesAll, protoLabels, featWeights);

        % Si quedó como 'Otros', re-evaluamos con heurística más permisiva
        if strcmp(claseNombreAsignado, 'Otros')
            % Reconstruir máscara local
            minR = min(rows_obj); maxR = max(rows_obj);
            minC = min(cols_obj); maxC = max(cols_obj);
            ancho = maxC - minC + 1; alto = maxR - minR + 1;
            mask_local = false(alto, ancho);
            rr = rows_obj - minR + 1; cc = cols_obj - minC + 1;
            mask_local(sub2ind(size(mask_local), rr, cc)) = true;

            rp2 = regionprops(mask_local, 'Perimeter', 'ConvexArea', 'Extent', 'Eccentricity');
            if isempty(rp2)
                perimeter2 = 0; convexArea2 = 0; extent2 = 0; Excentricity = 0;
            else
                perimeter2 = rp2.Perimeter; convexArea2 = rp2.ConvexArea; extent2 = rp2.Extent;
                if isfield(rp2, 'Eccentricity')
                    Excentricity = rp2.Eccentricity;
                else
                    Excentricity = 0;
                end
            end
            area2 = nnz(mask_local);
            if perimeter2 > 0
                circularity2 = (4 * pi * double(area2)) / (perimeter2 ^ 2);
            else
                circularity2 = 0;
            end
            if convexArea2 > 0
                solidity2 = double(area2) / double(convexArea2);
            else
                solidity2 = 1.0;
            end
                    % Calcular vértices usando contorno + reducepoly (fallback a corner)
            % Primero suavizamos/llenamos la máscara para reducir ruido en el contorno
            cornerCount2 = 0;
            try
                mask_local_s = imclose(mask_local, strel('disk', 1));
                mask_local_s = imfill(mask_local_s, 'holes');
                B2 = bwboundaries(mask_local_s, 'noholes');
                if ~isempty(B2)
                    maxLen2 = 0; idxMax2 = 1;
                    for bi2 = 1:length(B2)
                        if size(B2{bi2}, 1) > maxLen2
                            maxLen2 = size(B2{bi2}, 1);
                            idxMax2 = bi2;
                        end
                    end
                    boundary2 = B2{idxMax2}; xy2 = [boundary2(:,2), boundary2(:,1)];
                    % Usar tolerancia mayor para reducir vértices espurios (más agresiva)
                    tol2 = max(1, round(0.12 * size(xy2,1)));
                    try
                        reduced2 = reducepoly(xy2, tol2);
                        cornerCount2 = size(reduced2, 1);
                    catch
                        corners2 = corner(mask_local_s, 50);
                        cornerCount2 = size(corners2, 1);
                    end
                end
            catch
                try
                    corners2 = corner(mask_local, 50);
                    cornerCount2 = size(corners2, 1);
                catch
                    cornerCount2 = 0;
                end
            end
            % Evitar valores extremos en conteo de vértices (más restrictivo)
            cornerCount2 = min(cornerCount2, 8);

            % Calcular hullVertices local usando convhull sobre el hull de la máscara
            hullVertices2 = 0;
            try
                hullMask2 = bwconvhull(mask_local);
                Bh2 = bwboundaries(hullMask2, 'noholes');
                if ~isempty(Bh2)
                    bh2 = Bh2{1}; xyh2 = [bh2(:,2), bh2(:,1)];
                    tolH_local = max(1, round(0.08 * size(xyh2,1)));
                    try
                        reduced_h2 = reducepoly(xyh2, tolH_local);
                        ch2 = convhull(reduced_h2(:,1), reduced_h2(:,2));
                        hullVertices2 = max(0, length(ch2) - 1);
                    catch
                        ch2 = convhull(xyh2(:,1), xyh2(:,2));
                        hullVertices2 = max(0, length(ch2) - 1);
                    end
                end
            catch
                hullVertices2 = 0;
            end

            % Previous debug logging to file removed to keep output clean

            % Heurística TRIÁNGULO: usar hullVertices si está disponible (triángulo -> hullVertices ~= 0 y <=4)
            if ((hullVertices2 > 0 && hullVertices2 <= 4) || (cornerCount2 <= 4)) && solidity2 > 0.88 && circularity2 < 0.95
                claseNombreAsignado = 'Triangulo';
            end
        end

        % ESCRITURA EN ARCHIVO
        objectID = sprintf('Obj%d,%s', objetoCounter, imgFileName);
        fprintf(fileID, '%s,%f,%f,%f,%s\n', ...
                objectID, ...
                circularityProxy, ... 
                aspectRatio, ...      
                Excentricity, ...
                claseNombreAsignado);
    end
end
% 6. Cerrar el archivo de texto
fclose(fileID);
disp(' ');
disp('EXTRACCIÓN Y CLASIFICACIÓN FINALIZADA');
disp(['Total de objetos identificados y guardados: ', num2str(objetoCounter)]);
disp(['Archivo de resultados guardado en: ', fileName]);
%% Guardado de los vectores de caracteristicas por clases en un txt
disp(' ');
disp('INICIANDO ORDENAMIENTO POR CLASE');
inputFileName = 'NoIndexedDB.txt';
outputFileName = 'IndexedDB.txt';
% Abrir el archivo de entrada
fileID_in = fopen(inputFileName, 'r');
if fileID_in == -1
    error(['No se pudo abrir el archivo de entrada: ', inputFileName]);
end
% Leer la primera línea
headerLine = fgetl(fileID_in); 
% Leer el resto de las líneas
dataLines = textscan(fileID_in, '%s%s%f%f%f%s', 'Delimiter', ',', 'HeaderLines', 0, 'EndOfLine', '\n', 'CollectOutput', 0);
fclose(fileID_in);
numRecords = length(dataLines{1});
if numRecords == 0
    disp('ADVERTENCIA: No se pudo leer ninguna línea de datos válida desde NoIndexedDB.txt.');
    % Salir y cerrar el archivo de salida vacío si no hay datos
    fileID_out = fopen(outputFileName, 'w');
    fprintf(fileID_out, '%s\n', headerLine);
    fclose(fileID_out);
    return;
end
disp(['Se leyeron ', num2str(numRecords), ' registros para indexar.']);
% Los datos de la columna 5 son la Clase Asignada (string)
classColumn = dataLines{6}; 
%ORDENAR LOS DATOS POR CLASE
[~, sortIndices] = sort(classColumn);
%Abrir el archivo de salida (IndexedDB.txt)
fileID_out = fopen(outputFileName, 'w');
if fileID_out == -1
    error(['No se pudo crear el archivo de salida: ', outputFileName]);
end
fprintf(fileID_out, '%s\n', headerLine);
% Se recorre y escriben los datos ordenados
numRecords = length(sortIndices);
recordsWritten = 0;
for j = 1:numRecords
    idx = sortIndices(j); % El índice del registro ordenado
    
    % Se extraen los valores en el orden correcto
    id = [dataLines{1}{idx}, ',', dataLines{2}{idx}]; % Ej: Obj1 + , + Imagen6.png
    circ = dataLines{3}(idx);
    ra = dataLines{4}(idx);
    excent = dataLines{5}(idx);
    clase = dataLines{6}{idx};
    
    % Escribir la línea de datos en el archivo IndexedDB.txt
    fprintf(fileID_out, '%s,%f,%f,%f,%s\n', ...
            id, circ, ra, excent, clase);
    recordsWritten = recordsWritten + 1;
end
fclose(fileID_out);
disp(['Proceso de indexación completado. Total de ', num2str(recordsWritten), ' registros ordenados.']);
disp(['Archivo de resultados indexado guardado en: ', outputFileName]);
%% Seccion para las pruebas
[testFileName, testPathName] = uigetfile({'*.png;*.jpg;*.bmp','Archivos de Imagenes (*.png,*.jpg,*.bmp)'},...
                                        'Seleccione la imagen de prueba');
if isequal(testFileName, 0)
    disp('Prueba de clasificación cancelada por el usuario.');
    return;
end
fullImagePath = fullfile(testPathName, testFileName);
imgColor = imread(fullImagePath);
imgDouble = im2double(imgColor);
disp(['Imagen cargada para prueba: ', testFileName]);
%SEGMENTACIÓN Y ETIQUETADO (BFS)
isObject = (imgDouble(:,:,1) < 0.9) | (imgDouble(:,:,2) < 0.9) | (imgDouble(:,:,3) < 0.9);
[rows, cols] = size(isObject);
L = zeros(rows, cols); 
currentLabel = 0;
% Algoritmo BFS para etiquetar objetos
for r = 1:rows
    for c = 1:cols
        if isObject(r, c) == 1 && L(r, c) == 0
            currentLabel = currentLabel + 1; 
            queue = [r, c]; 
            L(r, c) = currentLabel;
            
            while ~isempty(queue)
                pixel_actual = queue(1, :);
                pr = pixel_actual(1); 
                pc = pixel_actual(2);
                queue(1, :) = []; 
            
                for k_vecino = 1:conectividad
                    nr = pr + vecinos(k_vecino, 1);
                    nc = pc + vecinos(k_vecino, 2);
                
                    if nr >= 1 && nr <= rows && nc >= 1 && nc <= cols
                        if isObject(nr, nc) == 1 && L(nr, nc) == 0
                            L(nr, nc) = currentLabel;
                            queue(end+1, :) = [nr, nc]; 
                        end
                    end
                end
            end 
        end 
    end
end
numObjects = currentLabel;

% Inicializar contadores de clases
classCounts = zeros(1, length(claseNombres)); 
objectCounter = 0;

% Matriz de etiquetas para la visualizacion
L_display = L; 

% Generación de mapa de colores (para L+1 índices: 1=negro, 2...N+1=colores)
num_required_colors = currentLabel + 1; 

cmap_base_full = [
    0.00 0.00 0.00; 
    0.00 0.45 0.74; 0.85 0.33 0.10; 0.93 0.69 0.13; 0.49 0.18 0.56; 0.47 0.67 0.19;
    0.30 0.75 0.93; 0.64 0.08 0.18; 0.10 0.60 0.60; 0.70 0.50 0.20; 0.50 0.50 0.50;
];
if num_required_colors > size(cmap_base_full, 1)
     cmap = [cmap_base_full; rand(num_required_colors - size(cmap_base_full, 1), 3)];
else
     cmap = cmap_base_full(1:num_required_colors, :);
end


for k = 1:numObjects
    objetoMask = (L == k);
    [rows_obj, cols_obj] = find(objetoMask);
    area = length(rows_obj);

    if area < minAreaThreshold % Filtrar ruido
        % Si es ruido (objeto no válido), se convierte en fondo 
        L_display(objetoMask) = 0; 
        continue; 
    end
    
    objectCounter = objectCounter + 1;
    
    % Extracción y clasificación centralizada
    [claseIdentificada, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~] = extract_and_classify_features(rows_obj, cols_obj, RANGES, finalPrototypes, umbralClasificacion, claseNombres, numClases, protoSamplesAll, protoLabels, featWeights);
    
    % Contar resultados
    classCounts(claseIdentificada) = classCounts(claseIdentificada) + 1;

end
%Muestra de resultados
disp(' ');
disp('RESULTADO FINAL DE CLASIFICACIÓN');
disp(['Total de objetos válidos encontrados: ', num2str(objectCounter)]);
if objectCounter > 0
    % Mostrar el conteo para cada clase
    for c = 1:length(claseNombres)
        count = classCounts(c);
        if count > 0
            className = claseNombres{c};
            if strcmp(className, 'Otros')
                disp(['->', num2str(count), ' objeto(s) fueron clasificados como DESCONOCIDOS.']);
            else
                disp(['->', num2str(count), ' objeto(s) clasificado(s) como: ', className]);
            end
        end
    end
else
    disp('No se encontraron objetos válidos (tamaño > 100 píxeles) en la imagen de prueba.');
end
%% Muestra de la Imagen
if objectCounter == 0
    disp('No se encontraron objetos válidos para graficar.');
    return; 
end

%Se crea la imagen con fondo negro y objetos en colores
colored_indices = L_display + 1;
segmented_img = zeros(rows, cols, 3); 

% Se asignan los colores
for channel = 1:3
    % Mapeado de los índices de color a los valores RGB en cada canal
    segmented_img(:,:,channel) = reshape(cmap(colored_indices, channel), rows, cols);
end

figure('Name', ['Clasificación de Objetos: ' testFileName], 'NumberTitle', 'off', 'Position', [100 100 1200 600]);
%IMAGEN ORIGINAL
subplot(1, 2, 1);
imshow(imgColor);
title(['Imagen Original: ' testFileName]);
axis on;
%SEGMENTACIÓN COLOREADA CON FONDO NEGRO
subplot(1, 2, 2);
imshow(segmented_img);
title('Identificación de Objetos (Fondo Negro)');
axis on;
axis ij;

%% Función Local para Extracción de Características y Clasificación
function [claseIdentificada, claseNombreAsignado, circularity, aspectRatio, Excentricity, minR, maxR, minC, maxC, ancho, alto] = extract_and_classify_features(rows_obj, cols_obj, RANGES, finalPrototypes, umbralClasificacion, claseNombres, numClases, protoSamplesAll, protoLabels, featWeights)
    % Reconstruir máscara a partir de los coords recibidos (uso local)
    area = length(rows_obj);
    minR = min(rows_obj); maxR = max(rows_obj);
    minC = min(cols_obj); maxC = max(cols_obj);
    ancho = maxC - minC + 1;
    alto = maxR - minR + 1;

    % Crear máscara recortada para cálculo de propiedades
    mask = false(alto, ancho);
    rr = rows_obj - minR + 1;
    cc = cols_obj - minC + 1;
    ind = sub2ind(size(mask), rr, cc);
    mask(ind) = true;

    % Obtener regionprops (si disponible) e incluir campos adicionales
    rp = regionprops(mask, 'Perimeter', 'ConvexArea', 'Extent', 'Eccentricity', 'Centroid', 'Orientation');
    if isempty(rp)
        perimeter = 0;
        convexArea = 0;
        extent = 0;
        Excentricity = 0;
        centroid = [0,0];
        orientation_abs = 0;
    else
        perimeter = rp.Perimeter;
        convexArea = rp.ConvexArea;
        extent = rp.Extent;
        if isfield(rp, 'Eccentricity')
            Excentricity = rp.Eccentricity;
        else
            Excentricity = 0;
        end
        if isfield(rp, 'Centroid')
            centroid = rp.Centroid;
        else
            centroid = [0,0];
        end
        if isfield(rp, 'Orientation')
            orientation_abs = abs(rp.Orientation);
        else
            orientation_abs = 0;
        end
    end

    % Características básicas
    if perimeter > 0
        circularity = (4 * pi * double(area)) / (perimeter ^ 2);
    else
        circularity = 0;
    end
    aspectRatio = ancho / max(alto, eps);

    % Solidity
    solidity = 1.0;
    if convexArea > 0
        solidity = double(area) / double(convexArea);
    end

    % Hu moments (comprimir con log para estabilidad)
    hu = computeHuMoments(mask);
    hu1 = log(1 + abs(hu(1)));

    % Perímetro normalizado
    if area > 0
        perimeter_norm = perimeter / sqrt(double(area));
    else
        perimeter_norm = 0;
    end

        % Convex hull vertex count: simplify hull boundary first then convhull
        hullVertices = 0;
        try
            hullMask = bwconvhull(mask);
            Bh = bwboundaries(hullMask, 'noholes');
            if ~isempty(Bh)
                bh = Bh{1}; xyh = [bh(:,2), bh(:,1)];
                % Simplify the hull boundary more agresively to reduce noisy vertices
                % (a mayor tolerancia, menos vértices espurios en formas simples)
                tolH = max(1, round(0.16 * size(xyh,1)));
                try
                    reducedH = reducepoly(xyh, tolH);
                    ch = convhull(reducedH(:,1), reducedH(:,2));
                    hullVertices = max(0, length(ch) - 1);
                catch
                    ch = convhull(xyh(:,1), xyh(:,2));
                    hullVertices = max(0, length(ch) - 1);
                end
            end
        catch
            hullVertices = 0;
        end

    % Radial peaks
    radialPeaks = 0;
    try
        B = bwboundaries(mask, 'noholes');
        if ~isempty(B)
            maxL = 0; idm = 1;
            for ii = 1:length(B)
                if size(B{ii},1) > maxL
                    maxL = size(B{ii},1); idm = ii;
                end
            end
            bd = B{idm}; xy = [bd(:,2), bd(:,1)];
            dists = sqrt((xy(:,1)-centroid(1)).^2 + (xy(:,2)-centroid(2)).^2);
            w = max(3, round(length(dists)/40));
            smoothd = movmean(dists, w);
            m = mean(smoothd) + 0.1 * std(smoothd);
            for ii = 2:(length(smoothd)-1)
                if smoothd(ii) > smoothd(ii-1) && smoothd(ii) > smoothd(ii+1) && smoothd(ii) > m
                    radialPeaks = radialPeaks + 1;
                end
            end
        end
    catch
        radialPeaks = 0;
    end

    hullVertices = min(hullVertices, 20);
    radialPeaks = min(radialPeaks, 20);

    % Calcular cornerCount para heurísticas (usa máscara suavizada y reducepoly)
    cornerCount = 0;
    try
        mask_c = imclose(mask, strel('disk', 1));
        mask_c = imfill(mask_c, 'holes');
        Bc = bwboundaries(mask_c, 'noholes');
        if ~isempty(Bc)
            maxLc = 0; idxc = 1;
            for ii = 1:length(Bc)
                if size(Bc{ii},1) > maxLc
                    maxLc = size(Bc{ii},1); idxc = ii;
                end
            end
            bd2 = Bc{idxc}; xy2 = [bd2(:,2), bd2(:,1)];
            % Use larger tolerance to aggressively reduce spurious vertices
            % (incrementado para evitar conteos inflados por ruido en el contorno)
            tolC = max(1, round(0.22 * size(xy2,1)));
            try
                reducedC = reducepoly(xy2, tolC);
                cornerCount = size(reducedC, 1);
            catch
                tmpc = corner(mask_c, 50);
                cornerCount = size(tmpc, 1);
            end
        end
    catch
        try
            tmpc = corner(mask, 50);
            cornerCount = size(tmpc, 1);
        catch
            cornerCount = 0;
        end
    end
    cornerCount = min(cornerCount, 8);

    % Concavity measure (1 - solidity) for heuristics (higher = more concave)
    concavity = 1.0 - solidity;

    % Empaquetar vector raw con el mismo orden que los prototipos extendidos
    featureVector_raw = double([circularity, solidity, extent, aspectRatio, Excentricity, perimeter_norm, hullVertices, radialPeaks, hu1, orientation_abs]);

    % (Nota: heurísticas de forma se aplicarán como fallback DESPUÉS de la
    % clasificación por prototipos para evitar sobrescribir clases simples)

    % Normalizar de acuerdo a RANGES (RANGES debe tener filas = num características)
    featCount = size(featureVector_raw, 2);
    featureVector = zeros(1, featCount);
    for f = 1:featCount
        denom = (RANGES(f, 2) - RANGES(f, 1));
        if denom == 0
            featureVector(f) = 0.0;
        else
            featureVector(f) = (featureVector_raw(f) - RANGES(f, 1)) / denom;
        end
    end
    featureVector(featureVector > 1) = 1;
    featureVector(featureVector < 0) = 0;

    % Distancia Euclideana a prototipos normalizados
    % Distancia Euclideana ponderada a prototipos normalizados
    if nargin < 11 || isempty(featWeights)
        w = ones(1, featCount);
    else
        w = featWeights;
        if length(w) ~= featCount
            w = ones(1, featCount);
        end
    end
    distancias = zeros(1, numClases);
    for c = 1:numClases
        diff = featureVector - finalPrototypes(c, :);
        distancias(c) = sqrt(sum((diff .^ 2) .* w));
    end
    [minDist, classIdx] = min(distancias);

    % Ajustar umbral según dimensión del espacio de características
    baseThreshold = umbralClasificacion * sqrt(featCount / 3);
    % Calcular distancia inter-prototipos (mediana) para ajustar umbral
    try
        pairwise = pdist(finalPrototypes);
        pairwise = pairwise(pairwise > 0);
        if isempty(pairwise)
            medianInter = baseThreshold;
        else
            medianInter = median(pairwise);
        end
    catch
        medianInter = baseThreshold;
    end
    adjThreshold = max(baseThreshold, 0.5 * medianInter) * 1.05; % pequeño margen extra
    if isnan(adjThreshold) || adjThreshold <= 0
        adjThreshold = baseThreshold;
    end
    if minDist < adjThreshold
        claseIdentificada = classIdx;
    else
        claseIdentificada = 6; % Otros (aún no identificado)
    end

    % Si quedó como 'Otros', imprimir distancias a prototipos y características para debug
    if claseIdentificada == 6
            % Debug prints removed for cleaner output
    end

    % Fallback heurístico: sólo cuando la clasificación inicial es 'Otros'
    if claseIdentificada == 6
        % Heurística mejorada (ordenada por prioridad):
           % 1) Flecha: detectar concavidad marcada + forma alargada y número moderado de vértices
           % Restringimos la segunda alternativa para evitar que estrellas (muy cóncavas y
           % con solidez muy baja) sean etiquetadas como flecha. Requerimos solidez >= 0.7
           % para la rama que permite concavidad alta.
           if ( (concavity > 0.04 && cornerCount >= 4 && cornerCount <= 10 && aspectRatio > 1.15 && radialPeaks <= 6) || ...
               (concavity > 0.12 && solidity >= 0.70 && solidity < 0.92 && cornerCount >= 3 && radialPeaks >= 2 && radialPeaks <= 8) )
              claseIdentificada = 5; % Flecha
              % debug removed
        % 2) Triangulo: permitir cierta tolerancia a ruido en el conteo de vértices
        % - Muchos triángulos en el dataset muestran alta convexidad (solidity alta)
        % - Pueden aparecer vértices espurios; por eso usamos OR con radialPeaks y hullVertices
        elseif (((cornerCount <= 6) || (radialPeaks <= 2) || (hullVertices > 0 && hullVertices <= 4)) && solidity > 0.95 && circularity < 0.95)
            claseIdentificada = 3; % Triangulo
            % debug removed
        % 3) Estrella: varios picos y concavidad significativa.
        % Algunos ejemplos del dataset muestran 4 picos detectables (debido a discretización),
        % por lo que bajamos el umbral de picos a 4 si la concavidad es alta (>0.3).
        elseif ( (cornerCount >= 6 && radialPeaks >= 5 && concavity > 0.02 && circularity < 0.9) || ...
                 (radialPeaks >= 4 && concavity > 0.30 && circularity < 0.6) )
            claseIdentificada = 4; % Estrella
            % debug removed
        end
    end

    % Fallback nearest-neighbor sobre muestras de prototipo (si se proveyeron)
    if claseIdentificada == 6
        try
            if exist('protoSamplesAll','var') && ~isempty(protoSamplesAll) && exist('protoLabels','var') && ~isempty(protoLabels)
                if size(protoSamplesAll,2) == length(featureVector)
                    diffs = protoSamplesAll - repmat(featureVector, size(protoSamplesAll,1), 1);
                    % aplicar mismos pesos a la distancia de muestras
                    if exist('w','var') && ~isempty(w)
                        distsSamples = sqrt(sum((diffs.^2) .* repmat(w, size(diffs,1), 1), 2));
                    else
                        distsSamples = sqrt(sum(diffs.^2, 2));
                    end
                    % Compute distances to proto-samples
                    [sortedD, sortedIdx] = sort(distsSamples);
                    minDS = sortedD(1);
                    % Print top-5 sample distances + labels for diagnostics
                    topK = min(5, length(sortedD));
                    topD = sortedD(1:topK);
                    topIdx = sortedIdx(1:topK);
                    topLabels = protoLabels(topIdx)';
                    % debug removed (top-k proto-sample distances)

                    % Immediate accept if very close match
                    if minDS < 1e-9 || minDS < 1e-6
                        lbl = protoLabels(topIdx(1));
                        if lbl >=1 && lbl <= numClases
                            claseIdentificada = lbl;
                            % debug removed for exact match
                        end
                    else
                        % Weighted voting: weight = 1/(d + eps)
                        epsd = 1e-6;
                        weights = 1 ./ (topD + epsd);
                        wvotes = zeros(1, numClases);
                        for vi = 1:topK
                            lbl = protoLabels(topIdx(vi));
                            if lbl >=1 && lbl <= numClases
                                wvotes(lbl) = wvotes(lbl) + weights(vi);
                            end
                        end
                        % Find best and second best weighted vote
                        [sortedW, idxW] = sort(wvotes, 'descend');
                        bestClass = idxW(1);
                        bestScore = sortedW(1);
                        secondScore = 0;
                        if length(sortedW) >= 2
                            secondScore = sortedW(2);
                        end
                        % Thresholds: require bestScore sufficiently larger than secondScore and minDS reasonably small
                        % Reducimos la tolerancia de distancia a muestras (antes 2.0*adj) porque
                        % hemos observado matches lejanos que dominan la votación y producen falsos positivos.
                        sampleThreshold = max(0.02, 1.2 * adjThreshold);
                        if bestScore > 0 && bestScore >= 1.4 * max(secondScore, 1e-12) && minDS <= sampleThreshold
                            claseIdentificada = bestClass;
                            % debug removed for weighted vote
                        end
                    end
                end
            end
        catch
            % If anything goes wrong, do nothing and leave as Otros
        end
    end

    claseNombreAsignado = claseNombres{claseIdentificada};
end

% Función auxiliar: calcula los 7 momentos invariantes de Hu para una máscara binaria
function hu = computeHuMoments(binImage)
    % binImage: imagen binaria lógica
    [rows, cols] = find(binImage);
    if isempty(rows)
        hu = zeros(1,7);
        return;
    end
    m00 = numel(rows);
    m10 = sum(cols);
    m01 = sum(rows);
    xbar = m10 / m00;
    ybar = m01 / m00;

    % Calcular momentos centrales hasta orden 3
    mu20 = sum((cols - xbar).^2);
    mu02 = sum((rows - ybar).^2);
    mu11 = sum((cols - xbar) .* (rows - ybar));
    mu30 = sum((cols - xbar).^3);
    mu03 = sum((rows - ybar).^3);
    mu12 = sum((cols - xbar) .* (rows - ybar).^2);
    mu21 = sum((cols - xbar).^2 .* (rows - ybar));

    % Momentos normalizados eta_pq = mu_pq / m00^(1 + (p+q)/2)
    eta20 = mu20 / (m00 ^ (1 + (2/2)));
    eta02 = mu02 / (m00 ^ (1 + (2/2)));
    eta11 = mu11 / (m00 ^ (1 + (2/2)));
    eta30 = mu30 / (m00 ^ (1 + (3/2)));
    eta03 = mu03 / (m00 ^ (1 + (3/2)));
    eta12 = mu12 / (m00 ^ (1 + (3/2)));
    eta21 = mu21 / (m00 ^ (1 + (3/2)));

    % Formulas de Hu
    hu = zeros(1,7);
    hu(1) = eta20 + eta02;
    hu(2) = (eta20 - eta02)^2 + 4 * eta11^2;
    hu(3) = (eta30 - 3*eta12)^2 + (3*eta21 - eta03)^2;
    hu(4) = (eta30 + eta12)^2 + (eta21 + eta03)^2;
    hu(5) = (eta30 - 3*eta12)*(eta30 + eta12)*((eta30 + eta12)^2 - 3*(eta21 + eta03)^2) + (3*eta21 - eta03)*(eta21 + eta03)*(3*(eta30 + eta12)^2 - (eta21 + eta03)^2);
    hu(6) = (eta20 - eta02)*((eta30 + eta12)^2 - (eta21 + eta03)^2) + 4*eta11*(eta30 + eta12)*(eta21 + eta03);
    hu(7) = (3*eta21 - eta03)*(eta30 + eta12)*((eta30 + eta12)^2 - 3*(eta21 + eta03)^2) - (eta30 - 3*eta12)*(eta21 + eta03)*(3*(eta30 + eta12)^2 - (eta21 + eta03)^2);
end  

% Función auxiliar: extrae el vector de características desde una imagen color
function fv = computeFeatureVectorFromImage(imgColor, minAreaThreshold)
    try
        imgDouble = im2double(imgColor);
        isObject = (imgDouble(:,:,1) < 0.9) | (imgDouble(:,:,2) < 0.9) | (imgDouble(:,:,3) < 0.9);
        isObject = bwareaopen(isObject, minAreaThreshold);
        CC = bwconncomp(isObject, 8);
        if CC.NumObjects == 0
            fv = [];
            return;
        end
        % Tomar componente más grande
        areas = cellfun(@numel, CC.PixelIdxList);
        [~, idx] = max(areas);
        mask = false(size(isObject));
        mask(CC.PixelIdxList{idx}) = true;

        area = nnz(mask);
        rp = regionprops(mask, 'Perimeter', 'BoundingBox', 'ConvexArea', 'Extent', 'Eccentricity', 'Centroid', 'Orientation');
        if isempty(rp)
            fv = [];
            return;
        end
        perimeter = rp.Perimeter;
        bbox = rp.BoundingBox;
        ancho = bbox(3);
        alto = bbox(4);
        areaBoundingBox = ancho * alto;
        convexArea = rp.ConvexArea;
        solidity = 1.0;
        if convexArea > 0, solidity = area / convexArea; end
        extent = rp.Extent;
        aspectRatio = ancho / max(alto, eps);
        if perimeter > 0
            circularity = (4 * pi * double(area)) / (perimeter ^ 2);
        else
            circularity = 0;
        end
        Excentricity = rp.Eccentricity;
        hu = computeHuMoments(mask);
        hu1 = log(1 + abs(hu(1)));
        if area > 0
            perimeter_norm = perimeter / sqrt(double(area));
        else
            perimeter_norm = 0;
        end
        hullVertices = 0;
        try
            hullMask = bwconvhull(mask);
            Bh = bwboundaries(hullMask, 'noholes');
            if ~isempty(Bh)
                hcoords = Bh{1};
                % usar tolerancia mayor para simplificar hull y evitar vértices espurios
                tolH2 = max(1, round(0.16 * size(hcoords,1)));
                try
                    reducedH2 = reducepoly(hcoords, tolH2);
                    ch = convhull(reducedH2(:,2), reducedH2(:,1));
                    hullVertices = max(0, length(ch) - 1);
                catch
                    ch = convhull(hcoords(:,2), hcoords(:,1));
                    hullVertices = max(0, length(ch) - 1);
                end
            end
        catch
            hullVertices = 0;
        end
        radialPeaks = 0;
        try
            B = bwboundaries(mask, 'noholes');
            if ~isempty(B)
                bcoords = B{1};
                cx = mean(bcoords(:,2)); cy = mean(bcoords(:,1));
                r = sqrt((bcoords(:,2)-cx).^2 + (bcoords(:,1)-cy).^2);
                % contar picos simples por diferencia sign
                dr = diff(r);
                radialPeaks = sum((dr(1:end-1) .* dr(2:end)) < 0);
            end
        catch
            radialPeaks = 0;
        end
        orientation_abs = abs(rp.Orientation);
        hullVertices = min(hullVertices, 20);
        radialPeaks = min(radialPeaks, 20);
        fv = double([circularity, solidity, extent, aspectRatio, Excentricity, perimeter_norm, hullVertices, radialPeaks, hu1, orientation_abs]);
    catch
        fv = [];
    end
end