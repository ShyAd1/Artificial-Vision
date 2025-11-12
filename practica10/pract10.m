function processImages()
    % Definir la carpeta de entrada y las carpetas de salida
    inputDir = 'FIGURAS';
    grayDir = 'FIGURAS_GRISES';
    binaryDir = 'FIGURAS_BINARIO';
    countDir = 'FIGURAS_CONTEO';
    
    % Crear las carpetas si no existen
    if ~exist(grayDir, 'dir')
        mkdir(grayDir);
    end
    if ~exist(binaryDir, 'dir')
        mkdir(binaryDir);
    end
    if ~exist(countDir, 'dir')
        mkdir(countDir);
    end
    
    % Obtener una lista de todas las imagenes en la carpeta
    imageFiles = dir(fullfile(inputDir, '*.*'));
    
    for k = 1:length(imageFiles)
        % Checar si el archivo es una imagen
        [~, ~, ext] = fileparts(imageFiles(k).name);
        if ismember(lower(ext), {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'})
            % Leer la imagen
            img = imread(fullfile(inputDir, imageFiles(k).name));
            
            % Converitr de RGB a gris
            grayImg = rgb2gray(img);
            imwrite(grayImg, fullfile(grayDir, imageFiles(k).name));
            
            % Convertir de gris a binaria utilizando el método de Otsu
            level = graythresh(grayImg);
            binaryImg = imbinarize(grayImg, level);
            
            % Invertir la imagen binaria
            invertedBinaryImg = ~binaryImg;
            imwrite(invertedBinaryImg, fullfile(binaryDir, imageFiles(k).name));

            % Conteo de objetos en la imagen binaria
            [labeledImage, numObjects] = bwlabel(invertedBinaryImg, 8);
            
            % Crear una imagen RGB para visualizar los objetos encontrados
            coloredImg = label2rgb(labeledImage, 'jet', 'k', 'shuffle');
            imwrite(coloredImg, fullfile(countDir, imageFiles(k).name));
            
            % Mostrar el numero de objetos encontrados
            fprintf('Imagen: %s, Numero de objetos: %d\n', imageFiles(k).name, numObjects);
        end
    end
end


processImages();