% Perceptrón 3D en el cubo [0,1]^3 con plano separador

clear; clc; close all;

% ---------------------------------------------------------
% 1) Definir los puntos del cubo y las clases
% ---------------------------------------------------------
% Vértices del cubo (x,y,z)
X = [ ...
    0 0 0;  % Patrón 1
    0 0 1;  % Patrón 2
    0 1 0;  % Patrón 3
    0 1 1;  % Patrón 4
    1 0 0;  % Patrón 5
    1 0 1;  % Patrón 6
    1 1 0;  % Patrón 7
    1 1 1]; % Patrón 8

% t = 1 para todos los patrones (término de sesgo)
t = ones(8,1);

% Matriz de entradas extendida: [x y z t]
X_ext = [X t];   % 8x4

% Clases:
% Primeros 4 puntos -> Clase 1
% Últimos 4 puntos -> Clase 2
% Usamos salida binaria: Clase 1 -> 0, Clase 2 -> 1
d = [0 0 0 0 1 1 1 1]';   % vector columna 8x1

% ---------------------------------------------------------
% 2) Pedir pesos iniciales y parámetros al usuario
% ---------------------------------------------------------
fprintf('Perceptrón 3D en el cubo [0,1]^3\n');
fprintf('Puntos (X,Y,Z):\n');
disp(X);
fprintf('Primeros 4 -> Clase 1, últimos 4 -> Clase 2\n\n');

w1 = input('Peso w1 (para x): ');
w2 = input('Peso w2 (para y): ');
w3 = input('Peso w3 (para z): ');
w4 = input('Peso w4 (para t = 1, sesgo): ');
w  = [w1; w2; w3; w4];  % vector columna 4x1

r        = input('Tasa de aprendizaje r (ej. 0.1): ');
maxEpoch = input('Número máximo de épocas (ej. 100): ');

% ---------------------------------------------------------
% 3) Entrenamiento del perceptrón
% ---------------------------------------------------------
fprintf('\n=== INICIO DEL ENTRENAMIENTO ===\n');
for epoch = 1:maxEpoch
    errores = 0;
    fprintf('\nÉpoca %d\n', epoch);
    fprintf('---------------------------------\n');

    for i = 1:8
        xi = X_ext(i,:)';   % patrón como columna [x y z t]^T
        di = d(i);          % salida deseada (0 o 1)

        % Suma ponderada: u = w^T * x
        u  = w' * xi;
        % Función de activación escalón
        y  = u >= 0;        % y = 1 si u >= 0; y = 0 si u < 0

        % Error
        e  = di - y;

        fprintf('Patrón %d: x = [%d %d %d 1], u = %.3f, y = %d, d = %d, e = %d\n', ...
            i, X(i,1), X(i,2), X(i,3), u, y, di, e);

        % Regla de aprendizaje del perceptrón
        if e ~= 0
            w = w + r * e * xi;
            errores = errores + 1;
            fprintf('   -> Actualización de pesos: w = [%.3f %.3f %.3f %.3f]\n', ...
                    w(1), w(2), w(3), w(4));
        end
    end

    fprintf('Errores en la época %d: %d\n', epoch, errores);

    if errores == 0
        fprintf('\nTodos los patrones fueron clasificados correctamente.\n');
        break;
    end
end

fprintf('\nPesos finales:\n');
fprintf('  w1 = %.4f, w2 = %.4f, w3 = %.4f, w4 = %.4f\n', w(1), w(2), w(3), w(4));

% ---------------------------------------------------------
% 4) Plano de separación dentro del cubo
% ---------------------------------------------------------
% Ecuación del plano de decisión (en variables originales x,y,z):
%   w1*x + w2*y + w3*z + w4*1 = 0
w1 = w(1); w2 = w(2); w3 = w(3); w4 = w(4);

fprintf('\nEcuación del plano de decisión:\n');
fprintf('  %.4f·x + %.4f·y + %.4f·z + %.4f = 0\n', w1, w2, w3, w4);

% Malla en el cubo para visualizar el plano
[Xg, Yg] = meshgrid(linspace(0,1,30), linspace(0,1,30));

if abs(w3) > 1e-6
    % Despejamos z del plano: z = -(w1*x + w2*y + w4)/w3
    Zg = -(w1*Xg + w2*Yg + w4) ./ w3;
else
    % Si w3 ≈ 0, el plano es casi vertical; no lo ploteamos bien con z(x,y)
    warning('w3 es casi cero; el plano es casi vertical. La representación puede ser limitada.');
    Zg = nan(size(Xg));
end

% Solo queremos la parte del plano que cae dentro del cubo [0,1]^3
mask = (Zg < 0) | (Zg > 1);
Zg(mask) = NaN;

% ---------------------------------------------------------
% 5) Gráfica 3D: cubo + puntos + plano
% ---------------------------------------------------------
figure;
hold on; grid on;
title('Cubo [0,1]^3 con plano separador del perceptrón');
xlabel('x'); ylabel('y'); zlabel('z');
axis([0 1 0 1 0 1]);
view(135, 30);

% Puntos de la Clase 1 (primeros 4)
scatter3(X(1:4,1), X(1:4,2), X(1:4,3), 80, 'b', 'filled'); % azul
% Puntos de la Clase 2 (últimos 4)
scatter3(X(5:8,1), X(5:8,2), X(5:8,3), 80, 'r', 'filled'); % rojo

% Plano de decisión
surf(Xg, Yg, Zg, 'FaceAlpha', 0.5, 'EdgeColor', 'none');

% Aristas del cubo para que se vea acotado
dibujar_cubo_unitario();

legend({'Clase 1 (primeros 4 puntos)', ...
        'Clase 2 (últimos 4 puntos)', ...
        'Plano de decisión'}, ...
       'Location','bestoutside');

hold off;


% Función auxiliar: dibuja el cubo [0,1]^3
function dibujar_cubo_unitario()
    % Dibuja las aristas del cubo [0,1]^3
    v = [0 0 0;
         1 0 0;
         1 1 0;
         0 1 0;
         0 0 1;
         1 0 1;
         1 1 1;
         0 1 1];

    aristas = [1 2; 2 3; 3 4; 4 1;   % base z=0
               5 6; 6 7; 7 8; 8 5;   % base z=1
               1 5; 2 6; 3 7; 4 8];  % verticales

    for k = 1:size(aristas,1)
        p1 = v(aristas(k,1),:);
        p2 = v(aristas(k,2),:);
        plot3([p1(1) p2(1)], [p1(2) p2(2)], [p1(3) p2(3)], 'k-');
    end
end