% Valor minimo al que puede permanecer una clase
minVal = 400;

% Preguntar al usuario cuantas clases quiere
numClases = input('¿Cuántas clases quiere? ');

% Inicializar un arreglo para almacenar las clases
clases = cell(1, numClases);

% Preguntar al usuario cuantos indices por clases 
numIndices = input('¿Cuántos índices por clase? ');

% Llenar el arreglo de clases con el total de indices de forma random
for i = 1:numClases
    % Pedir que coordenadas estará la clase
    fprintf('Coordenadas para la clase %d:\n', i);
    x = input('Coordenada en x: ');
    y = input('Coordenada en y: ');

    % Pedir que cantidad de dispersion quiere por clase
    disp_x = input('¿Cuánta dispersión quiere en x? ');
    disp_y = input('¿Cuánta dispersión quiere en y? ');

    % Calcular indices
    indices_x = (randn(1, numIndices)+x)*disp_x;
    indices_y = (randn(1, numIndices)+y)*disp_y;
    clases{i} = [indices_x; indices_y]; % Almacenar las coordenadas en el arreglo de clases
    % clases{1}
end

% Bloque para pedir n vectores
o = "si";
while (o == "si" || o == "Si" || o == "SI" || o == "sI")
    % Pedir vector al usuario
    fprintf('El valor minimo al que puede pertencer una clase es de %d en x y %d en y \n', minVal, minVal)
    vx = input('Dame el valor de la coordenada en x= ');
    vy = input('Dame el valor de la coordenada en y= ');
    vector = [vx; vy];
    o1 = "si";
    while (o1 == "si" || o1 == "Si" || o1 == "SI" || o1 == "sI")
        % Menu para escoger tipo de distancia, euclidiana o mahalanobis
        % Pedir al usuario el tipo de distancia
        fprintf('========== Menu ==========\n');
        fprintf('1.- Distancia Euclidiana\n');
        fprintf('2.- Distancia Mahalanobis\n');
        distanceType = input('Seleccione el tipo de distancia (1 o 2): ');

        % Validar la entrada del usuario para el tipo de distancia
        switch distanceType
            case 1 % Distancia ecuclidiana
                % Calcular centroides de las clases
                centroid = cell(1, numClases);
                for i = 1:numClases
                    centroid{i} = mean(clases{i}, 2); % Calcular el centroide de cada clase
                end
            
                % Calcular la clase más cercana al vector verificando que no pase de la
                % minima distancia
                % Calcular distancias y encontrar la clase más cercana
                distances = zeros(1, numClases);
                for i = 1:numClases
                    distances(i) = norm(centroid{i} - vector); % Calcular la distancia euclidiana
                end
            
                % Mostrar las distancias calculadas en formato de número con 2 decimales
                fprintf('Distancias del vector a cada clase:\n');
                for i = 1:numClases
                    fprintf('Distancia a la Clase %d: %.2f\n', i, distances(i)); % Mostrar la distancia calculada para cada clase
                end
                
                % Verificar si la distancia mínima es aceptable
                [minDistance, closestClass] = min(distances); % Encontrar la distancia mínima y su índice
                if minDistance < minVal
                    fprintf('La clase más cercana es la Clase %d con distancia euclidiana\n', closestClass);
                else
                    fprintf('Ninguna clase está dentro de la distancia mínima.\n');
                end
            case 2 % Distancia mahalanobis
                % Calcular la matriz de covarianza y la inversa para la distancia Mahalanobis
                covMatrix = cell(1, numClases);
                for i = 1:numClases
                    covMatrix{i} = cov(clases{i}'); % Calcular la matriz de covarianza para cada clase
                end
                invCovMatrix = cell(1, numClases);
                for i = 1:numClases
                    invCovMatrix{i} = inv(covMatrix{i}); % Calcular la inversa de la matriz de covarianza
                end

                % Calcular la media de cada clase
                meanVectors = cell(1, numClases);
                for i = 1:numClases
                    meanVectors{i} = mean(clases{i}, 2); % Calcular la media de cada clase
                end
                
                % Calcular distancias Mahalanobis
                distances = zeros(1, numClases);
                for i = 1:numClases
                    diff = vector - meanVectors{i}; % Vector dado menos la media de la clase
                    distances(i) = sqrt(diff' * invCovMatrix{i} * diff); % Calcular la distancia Mahalanobis
                end
                
                % Mostrar las distancias calculadas en formato de número con 2 decimales
                fprintf('Distancias del vector a cada clase:\n');
                for i = 1:numClases
                    fprintf('Distancia a la Clase %d: %.2f\n', i, distances(i)); % Mostrar la distancia calculada para cada clase
                end
                
                % Verificar si la distancia mínima es aceptable
                [minDistance, closestClass] = min(distances); % Encontrar la distancia mínima y su índice
                if minDistance < minVal
                    fprintf('La clase más cercana es la Clase %d con la distancia mahalanobis\n', closestClass);
                else
                    fprintf('Ninguna clase está dentro de la distancia mínima.\n');
                end
            otherwise
                fprintf('Seleccione una opcion valida.\n');
        end
        % Graficar todas las clases y el vector
        figure; % Crear una nueva figura
        hold on; % Mantener la gráfica actual
        colors = lines(numClases); % Generar colores para cada clase
        for i = 1:numClases
            scatter(clases{i}(1, :), clases{i}(2, :), 36, colors(i, :), 'filled'); % Graficar cada clase
        end
        scatter(vector(1), vector(2), 100, 'k', 'filled', 'MarkerEdgeColor', 'r'); % Graficar el vector
    
        % Poner la leyenda del vector ingresado y leyendas de las n clases
        legend([arrayfun(@(i) sprintf('Clase %d', i), 1:numClases, 'UniformOutput', false), 'Vector ingresado']);
    
        title('Clases y Vector'); % Título de la gráfica
        hold off; % Liberar la gráfica
        o1 = input('¿Desea probar el mismo vector con otro tipo de distancia? (si/no): ', 's');
    end
    o = input('¿Desea ingresar otro vector? (si/no): ', 's');
end
fprintf('Gracias por usar el clasificador de clases.');