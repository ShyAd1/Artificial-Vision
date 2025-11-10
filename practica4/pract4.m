%Definir clases

c1 = [2 4 2 3 4 5 3 4; 1 1 2 2 2 2 3 3];
c2 = [1 2 5 2 3 4 8 10 ; 3 3 3 4 4 4 1 1];
c3 = [6 7 8 9 10 11 12 13 ; 2 2 2 2 2 2 2 2];
c4 = [6 7 8 9 10 11 12 13 ; 3 4 5 6 7 8 9 10];
c5 = [6 7 8 9 10 11 12 13 ; 4 5 6 7 8 9 10 11];
c6 = [6 7 8 9 10 11 12 13 ; 5 6 7 8 9 10 11 12];
c7 = [6 7 8 9 10 11 12 13 ; 6 7 8 9 10 11 12 13];
c8 = [6 7 8 9 10 11 12 13 ; 7 8 9 10 11 12 13 14];

%Calcular parametros

media1=mean(c1,2);
media2=mean(c2,2);
media3=mean(c3,2);
media4=mean(c4,2);
media5=mean(c5,2);
media6=mean(c6,2);
media7=mean(c7,2);
media8=mean(c8,2);

o = "si";
while (o == "si")
    vx = input('dame el valor de la coordenada en x= ');
    vy = input('dame el valor de la coordenada en y= ');
    vector = [vx; vy];
    
    %calculando su centroide:
    dist1 = norm(media1-vector);
    dist2 = norm(media2-vector);
    dist3 = norm(media3-vector);
    dist4 = norm(media4-vector);
    dist5 = norm(media5-vector);
    dist6 = norm(media6-vector);
    dist7 = norm(media7-vector);
    dist8 = norm(media8-vector);
    dist_total = [dist1, dist2, dist3, dist4, dist5, dist6, dist7, dist8];

    disp(['Valor de los centroides: ', num2str(dist_total)]);

    T = input('Dame la minima distancia para que pertenezca a una clase: ');
    
    if all(dist_total > T)
        disp('El punto no pertenece a ninguna clase.');
    else
        clase = find(dist_total <= T);
        disp(['El punto pertenece a la clase: ', num2str(clase)]);
    end
    
    %Graficar todas las clases y el vector
    figure;
    hold on;
    plot(c1(1,:), c1(2,:), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'DisplayName', "clase 1");
    plot(c2(1,:), c2(2,:), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g', 'DisplayName', "clase 2");
    plot(c3(1,:), c3(2,:), 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b', 'DisplayName', "clase 3");
    plot(c4(1,:), c4(2,:), 'mo', 'MarkerSize', 10, 'MarkerFaceColor', 'm', 'DisplayName', "clase 4");
    plot(c5(1,:), c5(2,:), 'co', 'MarkerSize', 10, 'MarkerFaceColor', 'c', 'DisplayName', "clase 5");
    plot(c6(1,:), c6(2,:), 'yo', 'MarkerSize', 10, 'MarkerFaceColor', 'y', 'DisplayName', "clase 6");
    plot(c7(1,:), c7(2,:), 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'k', 'DisplayName', "clase 7");
    plot(c8(1,:), c8(2,:), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'DisplayName', "clase 8");
    plot(vx, vy, 'kx', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Input Vector');
    legend show;
    xlabel('X-axis');
    ylabel('Y-axis');
    title('Classes and Input Vector');
    grid on;
    hold off;

    o = input('¿Deseas probar con otro punto? (si/no): ', 's');
end
disp('Gracias por usar el clasificador de clases.');