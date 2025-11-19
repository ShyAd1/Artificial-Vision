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
            % Inicializar la imagen etiquetada y el contador de objetos
            [rows, cols] = size(invertedBinaryImg);
            labeledImage = zeros(rows, cols);
            currentLabel = 0;
            
            offsets = [-1,-1; -1,0; -1,1; 0,-1; 0,1; 1,-1; 1,0; 1,1];
            nOff = 8;
            
            stack = zeros(rows * cols, 2);
            
            for r = 1:rows
                for c = 1:cols
                    if invertedBinaryImg(r, c) == 1 && labeledImage(r, c) == 0
                        currentLabel = currentLabel + 1;
            
                        top = 1;
                        stack(top,:) = [r,c];
            
                        while top > 0
                            pr = stack(top,1);
                            pc = stack(top,2);
                            top = top - 1;
            
                            if labeledImage(pr,pc) ~= 0
                                continue;
                            end
            
                            labeledImage(pr,pc) = currentLabel;
            
                            for i = 1:nOff
                                nr = pr + offsets(i,1);
                                nc = pc + offsets(i,2);
            
                                if nr >= 1 && nr <= rows && nc >= 1 && nc <= cols && ...
                                   invertedBinaryImg(nr,nc) == 1 && labeledImage(nr,nc) == 0
                                    top = top + 1;
                                    stack(top,:) = [nr,nc];
                                end
                            end
                        end
                    end
                end
            end
            
            numObjects = currentLabel; % Contar el número de objetos etiquetados
            
            % Crear una imagen RGB para visualizar los objetos encontrados
            coloredImg = label2rgb(labeledImage, 'jet', 'k', 'shuffle');
            imwrite(coloredImg, fullfile(countDir, imageFiles(k).name));
            
            % Mostrar el numero de objetos encontrados
            fprintf('Imagen: %s, Numero de objetos: %d\n', imageFiles(k).name, numObjects);
        end
    end
end


processImages();