function net = buildDenoiseUNet()

% buildDenoiseUNet Crea y devuelve un dlnetwork para denoising residual.
%
%   dlnet = buildDenoiseUNet([H W C])
%
%   - Entrada:  tamaño de la imagen [H W C]
%   - Salida:   dlnetwork listo para entrenamiento (custom training loop)
%
%   Autor: tonix
%   Fecha: 2025

    inputSize = [720 960 3];
    lgraph = layerGraph();

    % ---------------- Encoder ----------------
    blk = [
        imageInputLayer(inputSize,"Name","in","Normalization","none")

        convolution2dLayer(3,64,"Padding","same","WeightsInitializer","he","Name","enc1_conv1")
        reluLayer("Name","enc1_relu1")
        convolution2dLayer(3,64,"Padding","same","WeightsInitializer","he","Name","enc1_conv2")
        reluLayer("Name","enc1_relu2")
    ];
    lgraph = addLayers(lgraph, blk);

    blk = [
        averagePooling2dLayer(2,"Stride",2,"Name","enc1_pool")
        convolution2dLayer(3,128,"Padding","same","WeightsInitializer","he","Name","enc2_conv1")
        reluLayer("Name","enc2_relu1")
        convolution2dLayer(3,128,"Padding","same","WeightsInitializer","he","Name","enc2_conv2")
        reluLayer("Name","enc2_relu2")
    ];
    lgraph = addLayers(lgraph, blk);

    blk = [
        averagePooling2dLayer(2,"Stride",2,"Name","enc2_pool")
        convolution2dLayer(3,256,"Padding","same","WeightsInitializer","he","Name","enc3_conv1")
        reluLayer("Name","enc3_relu1")
        convolution2dLayer(3,256,"Padding","same","WeightsInitializer","he","Name","enc3_conv2")
        reluLayer("Name","enc3_relu2")
    ];
    lgraph = addLayers(lgraph, blk);

    blk = [
        averagePooling2dLayer(2,"Stride",2,"Name","enc3_pool")
        convolution2dLayer(3,512,"Padding","same","WeightsInitializer","he","Name","enc4_conv1")
        reluLayer("Name","enc4_relu1")
        convolution2dLayer(3,512,"Padding","same","WeightsInitializer","he","Name","enc4_conv2")
        reluLayer("Name","enc4_relu2")
        dropoutLayer(0.5,"Name","enc4_dropout")
    ];
    lgraph = addLayers(lgraph, blk);

    % ---------------- Bridge + Up ----------------
    blk = [
        averagePooling2dLayer(2,"Stride",2,"Name","enc4_pool")
        convolution2dLayer(3,1024,"Padding","same","WeightsInitializer","he","Name","bridge_conv1")
        reluLayer("Name","bridge_relu1")
        convolution2dLayer(3,1024,"Padding","same","WeightsInitializer","he","Name","bridge_conv2")
        reluLayer("Name","bridge_relu2")
        dropoutLayer(0.5,"Name","bridge_dropout")
        transposedConv2dLayer(2,512,"Stride",2,"WeightsInitializer","he","Name","dec1_up")
        reluLayer("Name","dec1_uprelu")
    ];
    lgraph = addLayers(lgraph, blk);

    % Crops para alinear (skip connections)
    lgraph = addLayers(lgraph, crop2dLayer("centercrop","Name","crop4"));
    lgraph = addLayers(lgraph, [
        depthConcatenationLayer(2,"Name","merge4")
        convolution2dLayer(3,512,"Padding","same","WeightsInitializer","he","Name","dec1_conv1")
        reluLayer("Name","dec1_relu1")
        convolution2dLayer(3,512,"Padding","same","WeightsInitializer","he","Name","dec1_conv2")
        reluLayer("Name","dec1_relu2")
        transposedConv2dLayer(2,256,"Stride",2,"WeightsInitializer","he","Name","dec2_up")
        reluLayer("Name","dec2_uprelu")
    ]);

    lgraph = addLayers(lgraph, crop2dLayer("centercrop","Name","crop3"));
    lgraph = addLayers(lgraph, [
        depthConcatenationLayer(2,"Name","merge3")
        convolution2dLayer(3,256,"Padding","same","WeightsInitializer","he","Name","dec2_conv1")
        reluLayer("Name","dec2_relu1")
        convolution2dLayer(3,256,"Padding","same","WeightsInitializer","he","Name","dec2_conv2")
        reluLayer("Name","dec2_relu2")
        transposedConv2dLayer(2,128,"Stride",2,"WeightsInitializer","he","Name","dec3_up")
        reluLayer("Name","dec3_uprelu")
    ]);

    lgraph = addLayers(lgraph, crop2dLayer("centercrop","Name","crop2"));
    lgraph = addLayers(lgraph, [
        depthConcatenationLayer(2,"Name","merge2")
        convolution2dLayer(3,128,"Padding","same","WeightsInitializer","he","Name","dec3_conv1")
        reluLayer("Name","dec3_relu1")
        convolution2dLayer(3,128,"Padding","same","WeightsInitializer","he","Name","dec3_conv2")
        reluLayer("Name","dec3_relu2")
        transposedConv2dLayer(2,64,"Stride",2,"WeightsInitializer","he","Name","dec4_up")
        reluLayer("Name","dec4_uprelu")
    ]);

    lgraph = addLayers(lgraph, crop2dLayer("centercrop","Name","crop1"));
    lgraph = addLayers(lgraph, [
        depthConcatenationLayer(2,"Name","merge1")
        convolution2dLayer(3,64,"Padding","same","WeightsInitializer","he","Name","dec4_conv1")
        reluLayer("Name","dec4_relu1")
        convolution2dLayer(3,64,"Padding","same","WeightsInitializer","he","Name","dec4_conv2")
        reluLayer("Name","dec4_relu2")
    ]);

    % ---- Head: predicción del RESIDUO (3 canales) ----
    predHead = convolution2dLayer(1,3,"Name","pred_residual","Padding","same","WeightsInitializer","zeros");
    lgraph = addLayers(lgraph, predHead);

    % === Conexiones ===
    lgraph = connectLayers(lgraph,"enc1_relu2","enc1_pool");
    lgraph = connectLayers(lgraph,"enc1_relu2","crop1/in");
    lgraph = connectLayers(lgraph,"enc2_relu2","enc2_pool");
    lgraph = connectLayers(lgraph,"enc2_relu2","crop2/in");
    lgraph = connectLayers(lgraph,"enc3_relu2","enc3_pool");
    lgraph = connectLayers(lgraph,"enc3_relu2","crop3/in");
    lgraph = connectLayers(lgraph,"enc4_dropout","enc4_pool");
    lgraph = connectLayers(lgraph,"enc4_dropout","crop4/in");

    lgraph = connectLayers(lgraph,"dec1_uprelu","crop4/ref");
    lgraph = connectLayers(lgraph,"dec1_uprelu","merge4/in2");
    lgraph = connectLayers(lgraph,"crop4","merge4/in1");

    lgraph = connectLayers(lgraph,"dec2_uprelu","crop3/ref");
    lgraph = connectLayers(lgraph,"dec2_uprelu","merge3/in2");
    lgraph = connectLayers(lgraph,"crop3","merge3/in1");

    lgraph = connectLayers(lgraph,"dec3_uprelu","crop2/ref");
    lgraph = connectLayers(lgraph,"dec3_uprelu","merge2/in2");
    lgraph = connectLayers(lgraph,"crop2","merge2/in1");

    lgraph = connectLayers(lgraph,"dec4_uprelu","crop1/ref");
    lgraph = connectLayers(lgraph,"dec4_uprelu","merge1/in2");
    lgraph = connectLayers(lgraph,"crop1","merge1/in1");

    % Conectar head (residuo) al último ReLU del decoder
    lgraph = connectLayers(lgraph,"dec4_relu2","pred_residual");

    net = dlnetwork(lgraph);
end