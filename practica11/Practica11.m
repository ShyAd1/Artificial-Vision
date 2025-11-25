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
    
    %Se usa 'regexp' para encontrar el número entre 'imagen' y el punto de la extensión.
    tokens = regexp(fileName, 'Imagen(\d+)\.', 'tokens', 'once');
    
    if ~isempty(tokens)
        imageNumbers(i) = str2double(tokens{1});
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
    
    disp(['Imagen ', num2str(i), ' leída: ', fileListSorted(i).name]);
end
%% Obtencion de los prototipos para las clases
numClases = 5;
prototipoFeatures = cell(numClases, 1); 
disp('Iniciando cálculo de 3 características (Circ., R.A., Excentricidad) para los 5 prototipos...');
minAreaThreshold = 100; 
conectividad = 8;
vecinos = [-1, 0; 1, 0; 0, -1; 0, 1; -1, -1; -1, 1; 1, -1; 1, 1];
for i = 1:numClases
    
    imgColor = imageList{i}; 
    imgDouble = im2double(imgColor);
    
    % Segmentación por Color
    isObject = (imgDouble(:,:,1) < 0.9) | (imgDouble(:,:,2) < 0.9) | (imgDouble(:,:,3) < 0.9);
    
    %ALGORITMO BFS PARA ETIQUETADO DE PROTOTIPOS
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
    
    %EXTRACCIÓN DE 3 CARACTERÍSTICAS PARA CADA OBJETO DEL PROTOTIPO
    features_list = [];
    
    for k = 1:currentLabel
        objetoMask = (L == k);
        [rows_obj, cols_obj] = find(objetoMask);
        area = length(rows_obj);
        
        if area < minAreaThreshold, continue; end
        
        %CARACTERÍSTICAS BÁSICAS
        minR = min(rows_obj); maxR = max(rows_obj);
        minC = min(cols_obj); maxC = max(cols_obj);
        ancho = maxC - minC + 1;
        alto = maxR - minR + 1;
        areaBoundingBox = ancho * alto;
        circularityProxy = area / areaBoundingBox;
        aspectRatio = ancho / alto;
        r_mean = sum(rows_obj) / area;
        c_mean = sum(cols_obj) / area;
        
        % Momentos Centrales de segundo orden
        mu_20 = sum((rows_obj - r_mean).^2);
        mu_02 = sum((cols_obj - c_mean).^2);
        mu_11 = sum((rows_obj - r_mean) .* (cols_obj - c_mean));
        
        % Excentricidad Simplificada
        Excentricity = (mu_20 - mu_02)^2 + 4 * mu_11^2; 
        features_list = [features_list; [circularityProxy, aspectRatio, Excentricity]];
    end
    
    % El prototipo RAW es el promedio de todos los objetos válidos en la imagen
    if ~isempty(features_list)
        prototipoFeatures{i} = mean(features_list, 1);
    else
        warning(['No se pudo calcular el prototipo para la Clase ', num2str(i)]);
    end
    
    disp(['  > Características RAW de la Clase ', num2str(i), ' extraídas.']);
end
finalPrototypes_raw = zeros(numClases, 3);
for i = 1:numClases
    features = prototipoFeatures{i};
    if ~isempty(features)
        finalPrototypes_raw(i, :) = features;
    end
end
disp('--- Prototipos RAW de 3 Características Definidos ---');
disp(finalPrototypes_raw);
%% NORMALIZACIÓN Y CLASIFICACIÓN
% Definición de Rangos de Normalización
% Fila 1: Circularidad Simplificada [Min, Max]
% Fila 2: Relación de Aspecto [Min, Max]
% Fila 3: Excentricidad Simplificada [Min, Max]
RANGES = [0.30, 1.0;    % Rango de Circularidad
         0.40, 3.0;    % Rango de Relación de Aspecto
         min(finalPrototypes_raw(:, 3)), max(finalPrototypes_raw(:, 3)) * 1.5]; % Excentricidad: Mín-Máx observado + 50%
% Se normalizan los objetivos
finalPrototypes = zeros(numClases, 3);
for c = 1:numClases
    for f = 1:3
        finalPrototypes(c, f) = (finalPrototypes_raw(c, f) - RANGES(f, 1)) / (RANGES(f, 2) - RANGES(f, 1));
    end
end
finalPrototypes(finalPrototypes > 1) = 1;
finalPrototypes(finalPrototypes < 0) = 0;
disp('--- Prototipos de Clase Normalizados (3 Características) ---');
disp(finalPrototypes);
%% Extraccion y clasificacion de objetos, guardado de los vectores de caracteristicas en un txt
% Nombres de las 5 clases 
claseNombres = {'Cuadrado', 'Circulo', 'Triangulo', 'Estrella', 'Flecha', 'Otros'};
% Umbral para la clasificación estricta
umbralClasificacion = 0.15; 
% Abrir el archivo de texto para guardar los resultados
fileName = 'NoIndexedDB.txt';
fileID = fopen(fileName, 'w'); 
if fileID == -1
    error('No se pudo abrir o crear el archivo NoIndexedDB.txt');
end
% Encabezado actualizado para 3 características
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
        [~, claseNombreAsignado, circularityProxy, aspectRatio, Excentricity, ~, ~, ~, ~, ~, ~] = extract_and_classify_features(rows_obj, cols_obj, RANGES, finalPrototypes, umbralClasificacion, claseNombres, numClases);
        
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
    [claseIdentificada, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~] = extract_and_classify_features(rows_obj, cols_obj, RANGES, finalPrototypes, umbralClasificacion, claseNombres, numClases);
    
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
function [claseIdentificada, claseNombreAsignado, circularityProxy, aspectRatio, Excentricity, minR, maxR, minC, maxC, ancho, alto] = extract_and_classify_features(rows_obj, cols_obj, RANGES, finalPrototypes, umbralClasificacion, claseNombres, numClases)
    area = length(rows_obj); 
    minR = min(rows_obj); maxR = max(rows_obj);
    minC = min(cols_obj); maxC = max(cols_obj);
    ancho = maxC - minC + 1;
    alto = maxR - minR + 1; 
    areaBoundingBox = ancho * alto;
    circularityProxy = area / areaBoundingBox; 
    aspectRatio = ancho / alto;
    r_mean = sum(rows_obj) / area;
    c_mean = sum(cols_obj) / area;
    mu_20 = sum((rows_obj - r_mean).^2);
    mu_02 = sum((cols_obj - c_mean).^2);
    mu_11 = sum((rows_obj - r_mean) .* (cols_obj - c_mean));
    Excentricity = (mu_20 - mu_02)^2 + 4 * mu_11^2; 
    featureVector_raw = [circularityProxy, aspectRatio, Excentricity];
    featureVector = zeros(1, 3);
    for f = 1:3
         featureVector(f) = (featureVector_raw(f) - RANGES(f, 1)) / (RANGES(f, 2) - RANGES(f, 1));
    end
    featureVector(featureVector > 1) = 1;
    featureVector(featureVector < 0) = 0;
    distancias = zeros(1, numClases);
    %Distancia Euclideana
    for c = 1:numClases
        diff = featureVector - finalPrototypes(c, :);
        distancias(c) = sqrt(sum(diff.^2)); 
    end
    [minDist, classIdx] = min(distancias);
    if minDist < umbralClasificacion
        claseIdentificada = classIdx; 
    else
        claseIdentificada = 6;
    end
    claseNombreAsignado = claseNombres{claseIdentificada};
end