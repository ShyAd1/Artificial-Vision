% pract13.m
% Clustering Aglomerativo con distancia Manhattan
% Genera 10 puntos aleatorios en 2D y los agrupa hasta formar un solo cluster
% Muestra el dendrograma del proceso de agrupamiento

clear; clc; close all;

%% 1. Generar 10 puntos aleatorios en 2D
rng(42); % semilla para reproducibilidad
numPuntos = 10;
puntos = rand(numPuntos, 2) * 10; % puntos en rango [0,10] x [0,10]

fprintf('Puntos generados:\n');
disp(array2table(puntos, 'VariableNames', {'X', 'Y'}));

%% 2. Visualizar puntos iniciales
figure('Name', 'Clustering Aglomerativo - Manhattan', 'Position', [100 100 1400 600]);

subplot(1,2,1);
scatter(puntos(:,1), puntos(:,2), 100, 'filled', 'MarkerFaceColor', [0.2 0.4 0.8]);
hold on;
for i = 1:numPuntos
    text(puntos(i,1)+0.2, puntos(i,2)+0.2, sprintf('P%d',i), 'FontSize', 10, 'FontWeight', 'bold');
end
grid on;
xlabel('X'); ylabel('Y');
title('Puntos Iniciales (10 puntos aleatorios)');
axis equal;
xlim([-0.5 10.5]); ylim([-0.5 10.5]);

%% 3. Calcular matriz de distancias Manhattan
% Distancia Manhattan: d(p,q) = |p1-q1| + |p2-q2|
distMatrix = zeros(numPuntos, numPuntos);
for i = 1:numPuntos
    for j = 1:numPuntos
        distMatrix(i,j) = sum(abs(puntos(i,:) - puntos(j,:)));
    end
end

fprintf('\nMatriz de distancias Manhattan:\n');
disp(round(distMatrix, 2));

%% 4. Clustering Aglomerativo Jerárquico (Implementación manual mejorada)
% Implementación sin Statistics Toolbox usando método 'average linkage'

% Inicializar clusters: cada punto es un cluster
clusters = cell(numPuntos, 1);
for i = 1:numPuntos
    clusters{i} = i; % cluster i contiene solo el punto i
end

% Matriz Z guardará el historial de fusiones (formato linkage)
% Columnas: [cluster1, cluster2, distancia, numElementos]
Z = zeros(numPuntos-1, 4);

% Matriz de distancias activa (se irá reduciendo)
D = distMatrix;
D(1:numPuntos+1:end) = Inf; % diagonal = Inf para evitar auto-fusión

% Tracking de posiciones para dendrograma
maxClusters = 2 * numPuntos - 1;
clusterPosX = zeros(maxClusters, 1);
clusterPosY = zeros(maxClusters, 1);
nextPos = 1;

% Mapa: índice original -> último cluster que lo representa
clusterMap = (1:maxClusters)';

% Proceso iterativo de fusión
for paso = 1:numPuntos-1
    % Encontrar par de clusters más cercano
    [minDist, idx] = min(D(:));
    [i, j] = ind2sub(size(D), idx);
    
    % Asegurar i < j
    if i > j
        temp = i; i = j; j = temp;
    end
    
    % Obtener clusters actuales
    actualI = clusterMap(i);
    actualJ = clusterMap(j);
    
    % Asignar posiciones si no tienen
    if clusterPosX(actualI) == 0
        clusterPosX(actualI) = nextPos;
        nextPos = nextPos + 1;
    end
    if clusterPosX(actualJ) == 0
        clusterPosX(actualJ) = nextPos;
        nextPos = nextPos + 1;
    end
    
    % Registrar fusión con clusters actuales
    Z(paso, 1) = actualI;
    Z(paso, 2) = actualJ;
    Z(paso, 3) = minDist;
    Z(paso, 4) = length(clusters{i}) + length(clusters{j});
    
    % Crear nuevo cluster
    newCluster = numPuntos + paso;
    clusterPosX(newCluster) = (clusterPosX(actualI) + clusterPosX(actualJ)) / 2;
    clusterPosY(newCluster) = minDist;
    
    % Actualizar mapa: i y j ahora apuntan al nuevo cluster
    clusterMap(i) = newCluster;
    clusterMap(j) = newCluster;
    
    % Fusionar clusters: i absorbe j
    clusters{i} = [clusters{i}, clusters{j}];
    clusters{j} = [];
    
    % Actualizar distancias usando average linkage
    numClusters = length(D);
    for k = 1:numClusters
        if k == i || k == j || isempty(clusters{k})
            continue;
        end
        
        % Calcular distancia promedio
        distSum = 0;
        count = 0;
        for pi = clusters{i}
            for pk = clusters{k}
                distSum = distSum + distMatrix(pi, pk);
                count = count + 1;
            end
        end
        avgDist = distSum / count;
        
        D(i, k) = avgDist;
        D(k, i) = avgDist;
    end
    
    % Eliminar fila/columna j
    D(j, :) = Inf;
    D(:, j) = Inf;
end

fprintf('\nMatriz de enlace (Linkage Matrix Z):\n');
fprintf('Columnas: [Cluster1, Cluster2, Distancia, NumElementos]\n');
disp(Z);

%% 5. Visualizar Dendrograma (usando posiciones calculadas durante clustering)
subplot(1,2,2);
hold on;
grid on;
box on;

% Dibujar cada fusión con colores según nivel
maxHeight = max(Z(:,3)) * 1.1;
colors = jet(size(Z,1)); % azul (temprano) a rojo (tardío)

for paso = 1:size(Z,1)
    c1 = Z(paso, 1);
    c2 = Z(paso, 2);
    dist = Z(paso, 3);
    
    % Obtener posiciones
    x1 = clusterPosX(c1);
    y1 = clusterPosY(c1);
    x2 = clusterPosX(c2);
    y2 = clusterPosY(c2);
    
    % Color según paso de fusión
    color = colors(paso,:);
    
    % Dibujar líneas con color
    plot([x1 x1], [y1 dist], '-', 'Color', color, 'LineWidth', 2);
    plot([x2 x2], [y2 dist], '-', 'Color', color, 'LineWidth', 2);
    plot([x1 x2], [dist dist], '-', 'Color', color, 'LineWidth', 2.5);
end

% Configurar ejes
xlabel('Puntos', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Distancia Manhattan', 'FontSize', 11, 'FontWeight', 'bold');
title('Dendrograma - Clustering Aglomerativo', 'FontSize', 12, 'FontWeight', 'bold');
xlim([0.5 numPuntos+0.5]);
ylim([0 maxHeight]);

% Crear etiquetas según el orden de aparición
labels = cell(numPuntos, 1);
labelPos = zeros(numPuntos, 1);
for i = 1:numPuntos
    labelPos(i) = clusterPosX(i);
    labels{i} = sprintf('P%d', i);
end

% Ordenar etiquetas por posición
[labelPos_sorted, sortIdx] = sort(labelPos);
labels_sorted = labels(sortIdx);

set(gca, 'XTick', labelPos_sorted);
set(gca, 'XTickLabel', labels_sorted, 'FontSize', 10);
set(gca, 'FontSize', 10);
grid on;

hold off;

%% 6. Mostrar proceso de agrupamiento paso a paso
fprintf('\n=== Proceso de Agrupamiento ===\n');
for paso = 1:size(Z,1)
    c1 = Z(paso,1);
    c2 = Z(paso,2);
    dist = Z(paso,3);
    numElem = Z(paso,4);
    
    % Nombres de clusters (usar índice directo sin restar numPuntos)
    if c1 <= numPuntos
        nombre1 = sprintf('P%d', c1);
    else
        nombre1 = sprintf('C%d', c1);
    end
    
    if c2 <= numPuntos
        nombre2 = sprintf('P%d', c2);
    else
        nombre2 = sprintf('C%d', c2);
    end
    
    fprintf('Paso %d: %s + %s → C%d (distancia=%.2f, %d elementos)\n', ...
        paso, nombre1, nombre2, numPuntos+paso, dist, numElem);
end

%% 7. Visualización animada del proceso (opcional)
figure('Name', 'Proceso de Agrupamiento Animado', 'Position', [150 150 800 600]);
colores = lines(numPuntos);

% Reconstruir asignación de clusters para cada paso
for paso = 0:size(Z,1)
    clf;
    
    if paso == 0
        % Estado inicial: cada punto es su propio cluster
        scatter(puntos(:,1), puntos(:,2), 100, colores, 'filled');
        for i = 1:numPuntos
            text(puntos(i,1)+0.2, puntos(i,2)+0.2, sprintf('P%d',i), ...
                'FontSize', 10, 'FontWeight', 'bold');
        end
        title('Estado Inicial - 10 clusters (1 punto cada uno)');
    else
        % Reconstruir clusters hasta el paso actual
        T = obtenerAsignacionClusters(Z, numPuntos, paso);
        numClustersActual = max(T);
        
        % Visualizar clusters (manual, sin gscatter)
        hold on;
        for k = 1:numClustersActual
            idx = (T == k);
            scatter(puntos(idx,1), puntos(idx,2), 100, colores(k,:), 'o', 'filled');
        end
        
        % Añadir labels
        for i = 1:numPuntos
            text(puntos(i,1)+0.2, puntos(i,2)+0.2, sprintf('P%d',i), ...
                'FontSize', 9);
        end
        
        % Mostrar centroides de clusters
        for k = 1:numClustersActual
            idx = (T == k);
            centroide = mean(puntos(idx,:), 1);
            plot(centroide(1), centroide(2), 'k+', 'MarkerSize', 15, 'LineWidth', 2);
        end
        
        title(sprintf('Paso %d/%d - %d clusters restantes (dist=%.2f)', ...
            paso, size(Z,1), numPuntos-paso, Z(paso,3)));
    end
    
    grid on;
    xlabel('X'); ylabel('Y');
    axis equal;
    xlim([-0.5 10.5]); ylim([-0.5 10.5]);
    legend('Location', 'best');
    
    pause(1.2); % pausa para visualizar cada paso
end

fprintf('\n✓ Agrupamiento completo: todos los puntos en un solo cluster\n');

%% Función para calcular orden óptimo de hojas en dendrograma
function orden = optimalLeafOrder(Z, numPuntos)
    % Construir árbol y ordenar hojas para minimizar cruces
    % Usamos un algoritmo simple basado en la estructura de fusión
    
    if isempty(Z)
        orden = 1:numPuntos;
        return;
    end
    
    % Inicializar: cada hoja en su posición original
    leafOrder = cell(2*numPuntos-1, 1);
    for i = 1:numPuntos
        leafOrder{i} = i;
    end
    
    % Procesar fusiones
    for paso = 1:size(Z,1)
        c1 = round(Z(paso, 1));
        c2 = round(Z(paso, 2));
        newCluster = numPuntos + paso;
        
        % Verificar índices válidos
        if c1 < 1 || c1 > length(leafOrder) || c2 < 1 || c2 > length(leafOrder)
            continue;
        end
        
        % Combinar hojas de ambos clusters
        if isempty(leafOrder{c1})
            leaves1 = [];
        else
            leaves1 = leafOrder{c1};
        end
        
        if isempty(leafOrder{c2})
            leaves2 = [];
        else
            leaves2 = leafOrder{c2};
        end
        
        % Ordenar para que el cluster más a la izquierda vaya primero
        leafOrder{newCluster} = [leaves1, leaves2];
    end
    
    % El último cluster contiene el orden final
    lastCluster = 2*numPuntos-1;
    if ~isempty(leafOrder{lastCluster})
        orden = leafOrder{lastCluster};
    else
        orden = 1:numPuntos;
    end
    
    % Asegurar que orden tenga exactamente numPuntos elementos
    if length(orden) ~= numPuntos
        orden = 1:numPuntos;
    end
end

%% Función auxiliar para obtener asignación de clusters en un paso dado
function T = obtenerAsignacionClusters(Z, numPuntos, numPasos)
    % Inicializar: cada punto en su propio cluster
    clusterID = (1:numPuntos)';
    
    % Aplicar fusiones hasta el paso indicado
    for paso = 1:numPasos
        c1 = Z(paso, 1);
        c2 = Z(paso, 2);
        newCluster = numPuntos + paso;
        
        % Encontrar todos los puntos que pertenecen a c1 y c2
        % c1 y c2 pueden ser puntos originales o clusters creados anteriormente
        
        % Obtener miembros de c1
        if c1 <= numPuntos
            members1 = c1;
        else
            % Es un cluster creado, buscar sus miembros
            members1 = find(clusterID == c1);
        end
        
        % Obtener miembros de c2
        if c2 <= numPuntos
            members2 = c2;
        else
            % Es un cluster creado, buscar sus miembros
            members2 = find(clusterID == c2);
        end
        
        % Asignar todos los miembros al nuevo cluster
        clusterID(members1) = newCluster;
        clusterID(members2) = newCluster;
    end
    
    % Renumerar clusters consecutivamente (1, 2, 3, ...)
    uniqueClusters = unique(clusterID);
    T = zeros(size(clusterID));
    for i = 1:length(uniqueClusters)
        T(clusterID == uniqueClusters(i)) = i;
    end
end
