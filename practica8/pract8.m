% Valor maximo al que puede permanecer una clase
maxVal = 255;

[filename, pathname] = uigetfile({'*.jpg;*.png;*.bmp;*.tif', 'Image Files (*.jpg, *.png, *.bmp, *.tif)'; '*.*', 'All Files (*.*)'}, 'Selecciona una imagen');
if isequal(filename, 0) || isequal(pathname, 0)
    disp('El usuario ha cancelado la seleccion');
else
    fullImagePath = fullfile(pathname, filename);
    disp(['Imagen seleccionada: ', fullImagePath]);
end

img = imread(fullImagePath);
% Mostrar la imagen
imshow(img);
title('Imagen Cargada');
[m,n]=size(img);

% Preguntar al usuario cuantas clases quiere
numClases = input('¿Cuántas clases quiere? ');

% Inicializar un arreglo para almacenar las clases
clases = cell(1, numClases);

% Preguntar al usuario cuantos indices por clases 
numIndices = input('¿Cuántos índices por clase? ');

for i = 1:numClases
    % Seleccionar con el mouse los limites de la ventana
    [x,y]=ginput(2);
    
    % Calcular las dimensiones del rectangulo
    
    x_min=min(x);
    y_min=min(y);
    
    ancho=abs(x(2)-x(1));
    alto=abs(y(2)-y(1));
    
    colors = lines(numClases); % Generar colores para cada clase
    % Dibujando la ventana
    hold on
    rectangle('Position',[x_min y_min,ancho alto],'EdgeColor',colors(i,:),'LineWidth',2)
    hold off;
    
    % Generando los puntos sobre el rectángulo;
    x_rand = x_min + ancho * rand(1,numIndices);
    y_rand = y_min + alto * rand(1,numIndices);
    
    % Almacenar los valores R, G, B para cada punto
    rgb_values = zeros(numIndices, 3); % Inicializar matriz para valores RGB
    for j = 1:numIndices
        % Obtener el color del pixel en la posición aleatoria
        pixel = img(round(y_rand(j)), round(x_rand(j)), :);
        rgb_values(j, :) = squeeze(pixel)'; % Almacenar los valores R, G, B
    end
    clases{i} = rgb_values; % Almacenar los valores RGB en la celda correspondiente
    
    % Ploteando los puntos
    hold on
    plot(x_rand(1,:), y_rand(1,:), 'o', 'MarkerSize', 10, 'MarkerFaceColor', colors(i,:))
    hold off
end

% Bloque para pedir otro tipo de distancia
o = "si";
while (o == "si" || o == "Si" || o == "SI" || o == "sI" || o == "s" || o == "S")
    % Menu para escoger tipo de distancia, euclidiana o mahalanobis
    % Pedir al usuario el tipo de distancia
    fprintf('========== Menu ==========\n');
    fprintf('1.- Distancia Euclidiana\n');
    fprintf('2.- Distancia Mahalanobis\n');
    distanceType = input('Seleccione el tipo de distancia (1 o 2): ');

    % Validar la entrada del usuario para el tipo de distancia
    switch distanceType
        case 1 % Distancia euclidiana
            % Calcular centroides de las clases
            centroid = cell(1, numClases);
            for i = 1:numClases
                centroid{i} = round(mean(clases{i}, 1)); % Calcular el centroide de cada clase y redondear
            end

            accuracy_reconstruction = zeros(1, numClases);
            accuracy_cross_validation = zeros(1, numClases);
            accuracy_leave_one_out = zeros(1, numClases);
            
            for i = 1:numClases
                % Calcular distancias para el conjunto de prueba usando todos los datos
                numPixels = size(clases{i}, 1); % Asegurarse de que numPixels se define antes de usarlo
                distances_reconstruction = zeros(numPixels, 1); % Cambiar a longitud de numPixels
                for j = 1:numPixels % Cambiar a numPixels
                    distances_reconstruction(j) = norm(double(centroid{i}) - double(clases{i}(j, :))); % Calcular la distancia euclidiana
                end
                accuracy_reconstruction(i) = sum(distances_reconstruction < maxVal) / numPixels * 100; % Porcentaje de aciertos

                % Separar los datos en 50% para entrenamiento y 50% para prueba
                idx = randperm(numPixels); % Mezclar los índices
                trainIdx = idx(1:round(0.5 * numPixels)); % 50% para entrenamiento
                testIdx = idx(round(0.5 * numPixels) + 1:end); % 50% para prueba

                % Calcular distancias para el conjunto de prueba usando cross-validation
                distances_cross_validation = zeros(length(testIdx), 1);
                for j = 1:length(testIdx)
                    distances_cross_validation(j) = norm(double(centroid{i}) - double(clases{i}(testIdx(j), :))); % Calcular la distancia euclidiana
                end
                accuracy_cross_validation(i) = sum(distances_cross_validation < maxVal) / length(testIdx) * 100; % Porcentaje de aciertos

                % Calcular distancias para leave-one-out
                distances_leave_one_out = zeros(numPixels, 1);
                for j = 1:numPixels
                    % Usar el índice j como el índice de prueba
                    testIdx = j; 
                    trainIdx = setdiff(1:numPixels, testIdx); % Entrenar con todos los datos menos el de prueba
                    distances_leave_one_out(j) = norm(double(centroid{i}) - double(clases{i}(trainIdx, :))); % Calcular la distancia euclidiana
                end
                accuracy_leave_one_out(i) = sum(distances_leave_one_out < maxVal) / (numPixels - 1) * 100; % Porcentaje de aciertos
            end
            
            % Mostrar las precisiones por clase
            fprintf('Precisión por clase (Reconstrucción - Euclidiana):\n');
            for i = 1:numClases
                accuracy_reconstruction(i) = min(accuracy_reconstruction(i), 100); % Asegurarse de que no supere el 100%
                fprintf('Clase %d: %.2f%%\n', i, accuracy_reconstruction(i));
            end
            avg_accuracy_reconstruction = mean(accuracy_reconstruction); % Promedio de precisión reconstrucción

            fprintf('Precisión por clase (Cross Validation - Euclidiana):\n');
            for i = 1:numClases
                accuracy_cross_validation(i) = min(accuracy_cross_validation(i), 100); % Asegurarse de que no supere el 100%
                fprintf('Clase %d: %.2f%%\n', i, accuracy_cross_validation(i));
            end
            avg_accuracy_cross_validation = mean(accuracy_cross_validation); % Promedio de precisión cross-validation

            fprintf('Precisión por clase (Leave-One-Out - Euclidiana):\n');
            for i = 1:numClases
                accuracy_leave_one_out(i) = min(accuracy_leave_one_out(i), 100); % Asegurarse de que no supere el 100%
                fprintf('Clase %d: %.2f%%\n', i, accuracy_leave_one_out(i));
            end
            avg_accuracy_leave_one_out = mean(accuracy_leave_one_out); % Promedio de precisión leave-one-out

            % Promediar los 3 promedios
            overall_avg_accuracy = mean([avg_accuracy_reconstruction, avg_accuracy_cross_validation, avg_accuracy_leave_one_out]);
            fprintf('Precisión promedio general para la distancia Euclidiana: %.2f%%\n', overall_avg_accuracy);
            
            % Graficar las precisiones
            figure;
            hold on;
            plot(1:numClases, accuracy_reconstruction, '-o', 'DisplayName', 'Reconstrucción - Euclidiana');
            plot(1:numClases, accuracy_cross_validation, '-s', 'DisplayName', 'Cross Validation - Euclidiana');
            plot(1:numClases, accuracy_leave_one_out, '-d', 'DisplayName', 'Leave-One-Out - Euclidiana');
            hold off;
            title('Precisión por Clase');
            xlabel('Clase');
            ylabel('Precisión (%)');
            xticks(1:numClases);
            xticklabels(arrayfun(@num2str, 1:numClases, 'UniformOutput', false));
            legend show;
            grid on;

        case 2 % Distancia Mahalanobis
            % Calcular la matriz de covarianza y la inversa para la distancia Mahalanobis
            covMatrix = cell(1, numClases);
            for i = 1:numClases
                covMatrix{i} = cov(clases{i}); % Calcular la matriz de covarianza para cada clase
            end
            invCovMatrix = cell(1, numClases);
            for i = 1:numClases
                covMatrix{i} = cov(clases{i}); % Calcular la matriz de covarianza para cada clase
                if rcond(covMatrix{i}) < 1e-10 % Verificar si la matriz es casi singular
                    warning('La matriz de covarianza para la clase %d es casi singular. Se ajustará para mejorar la estabilidad.', i);
                    covMatrix{i} = covMatrix{i} + eye(size(covMatrix{i})) * 1e-10; % Ajustar la matriz de covarianza
                end
                invCovMatrix{i} = inv(covMatrix{i}); % Calcular la inversa de la matriz de covarianza
            end

            accuracy_reconstruction = zeros(1, numClases);
            accuracy_cross_validation = zeros(1, numClases);
            accuracy_leave_one_out = zeros(1, numClases);
            
            for i = 1:numClases
                % Calcular distancias para el conjunto de prueba usando todos los datos
                numPixels = size(clases{i}, 1); % Asegurarse de que numPixels se define antes de usarlo
                distances_reconstruction = zeros(numPixels, 1); % Cambiar a longitud de numPixels
                for j = 1:numPixels % Cambiar a numPixels
                    diff = double(clases{i}(j, :)') - double(mean(clases{i}, 1)'); % Vector dado menos la media de la clase
                    distances_reconstruction(j) = sqrt(diff' * invCovMatrix{i} * diff); % Calcular la distancia Mahalanobis
                end
                accuracy_reconstruction(i) = sum(distances_reconstruction < maxVal) / numPixels * 100; % Porcentaje de aciertos

                % Separar los datos en 50% para entrenamiento y 50% para prueba
                idx = randperm(numPixels); % Mezclar los índices
                trainIdx = idx(1:round(0.5 * numPixels)); % 50% para entrenamiento
                testIdx = idx(round(0.5 * numPixels) + 1:end); % 50% para prueba

                % Calcular distancias para el conjunto de prueba usando cross-validation
                distances_cross_validation = zeros(length(testIdx), 1);
                for j = 1:length(testIdx)
                    diff = double(clases{i}(testIdx(j), :)') - double(mean(clases{i}, 1)'); % Vector dado menos la media de la clase
                    distances_cross_validation(j) = sqrt(diff' * invCovMatrix{i} * diff); % Calcular la distancia Mahalanobis
                end
                accuracy_cross_validation(i) = sum(distances_cross_validation < maxVal) / length(testIdx) * 100; % Porcentaje de aciertos

                % Calcular distancias para leave-one-out
                distances_leave_one_out = zeros(numPixels, 1);
                for j = 1:numPixels
                    % Usar el índice j como el índice de prueba
                    testIdx = j; 
                    trainIdx = setdiff(1:numPixels, testIdx); % Entrenar con todos los datos menos el de prueba
                    diff = double(clases{i}(testIdx, :)') - double(mean(clases{i}(trainIdx, :), 1)'); % Vector dado menos la media de la clase
                    distances_leave_one_out(j) = sqrt(diff' * invCovMatrix{i} * diff); % Calcular la distancia Mahalanobis
                end
                accuracy_leave_one_out(i) = sum(distances_leave_one_out < maxVal) / (numPixels - 1) * 100; % Porcentaje de aciertos
            end
            
            % Mostrar las precisiones por clase
            fprintf('Precisión por clase (Reconstrucción - Mahalanobis):\n');
            for i = 1:numClases
                accuracy_reconstruction(i) = min(accuracy_reconstruction(i), 100); % Asegurarse de que no supere el 100%
                fprintf('Clase %d: %.2f%%\n', i, accuracy_reconstruction(i));
            end
            avg_accuracy_reconstruction = mean(accuracy_reconstruction); % Promedio de precisión reconstrucción

            fprintf('Precisión por clase (Cross Validation - Mahalanobis):\n');
            for i = 1:numClases
                accuracy_cross_validation(i) = min(accuracy_cross_validation(i), 100); % Asegurarse de que no supere el 100%
                fprintf('Clase %d: %.2f%%\n', i, accuracy_cross_validation(i));
            end
            avg_accuracy_cross_validation = mean(accuracy_cross_validation); % Promedio de precisión cross-validation

            fprintf('Precisión por clase (Leave-One-Out - Mahalanobis):\n');
            for i = 1:numClases
                accuracy_leave_one_out(i) = min(accuracy_leave_one_out(i), 100); % Asegurarse de que no supere el 100%
                fprintf('Clase %d: %.2f%%\n', i, accuracy_leave_one_out(i));
            end
            avg_accuracy_leave_one_out = mean(accuracy_leave_one_out); % Promedio de precisión leave-one-out

            % Promediar los 3 promedios
            overall_avg_accuracy = mean([avg_accuracy_reconstruction, avg_accuracy_cross_validation, avg_accuracy_leave_one_out]);
            fprintf('Precisión promedio general para la distancia Mahalanobis: %.2f%%\n', overall_avg_accuracy);
            
            % Graficar las precisiones
            figure;
            hold on;
            plot(1:numClases, accuracy_reconstruction, '-o', 'DisplayName', 'Reconstrucción - Mahalanobis');
            plot(1:numClases, accuracy_cross_validation, '-s', 'DisplayName', 'Cross Validation - Mahalanobis');
            plot(1:numClases, accuracy_leave_one_out, '-d', 'DisplayName', 'Leave-One-Out - Mahalanobis');
            hold off;
            title('Precisión por Clase (Mahalanobis)');
            xlabel('Clase');
            ylabel('Precisión (%)');
            xticks(1:numClases);
            xticklabels(arrayfun(@num2str, 1:numClases, 'UniformOutput', false));
            legend show;
            grid on;
        otherwise
            fprintf('Seleccione una opcion valida.\n');
    end
    o = input('¿Desea probar otro tipo de distancia? (si/no): ', 's');
end
fprintf('Gracias por usar el clasificador de clases.\n');