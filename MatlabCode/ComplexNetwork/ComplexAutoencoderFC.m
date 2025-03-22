classdef ComplexAutoencoderFC < handle
    properties
        encoder_real
        encoder_imag
        decoder_real
        decoder_imag
    end
    
    methods
        function obj = ComplexAutoencoderFC(input_dim, latent_dim)
            % Definir las capas del encoder para la parte real
            encoder_layers_real = layerGraph([
                featureInputLayer(input_dim, 'Name', 'input_real')
                fullyConnectedLayer(latent_dim, 'Name', 'encoder_real')
                reluLayer('Name', 'relu_real')
            ]);
            
            % Definir las capas del encoder para la parte imaginaria
            encoder_layers_imag = layerGraph([
                featureInputLayer(input_dim, 'Name', 'input_imag')
                fullyConnectedLayer(latent_dim, 'Name', 'encoder_imag')
                reluLayer('Name', 'relu_imag')
            ]);
            
            % Definir las capas del decoder para la parte real
            decoder_layers_real = layerGraph([
                featureInputLayer(latent_dim, 'Name', 'latent_real')
                fullyConnectedLayer(input_dim, 'Name', 'decoder_real')
            ]);
            
            % Definir las capas del decoder para la parte imaginaria
            decoder_layers_imag = layerGraph([
                featureInputLayer(latent_dim, 'Name', 'latent_imag')
                fullyConnectedLayer(input_dim, 'Name', 'decoder_imag')
            ]);
            
            % 🔥 INICIALIZAR dlnetwork CORRECTAMENTE 🔥
            obj.encoder_real = dlnetwork(encoder_layers_real);
            obj.encoder_imag = dlnetwork(encoder_layers_imag);
            obj.decoder_real = dlnetwork(decoder_layers_real);
            obj.decoder_imag = dlnetwork(decoder_layers_imag);
        end
        
        function [z_real, z_imag] = encode(obj, x_real, x_imag)
            % Propagar datos a través del encoder
            z_real = predict(obj.encoder_real, x_real) - predict(obj.encoder_imag, x_imag);
            z_imag = predict(obj.encoder_real, x_imag) + predict(obj.encoder_imag, x_real);
            
            % Aplicar ReLU compleja
            z_real = max(0, z_real);
            z_imag = max(0, z_imag);
        end
        
        function [reco_real, reco_imag] = decode(obj, z_real, z_imag)
            % Propagar datos a través del decoder
            reco_real = predict(obj.decoder_real, z_real) - predict(obj.decoder_imag, z_imag);
            reco_imag = predict(obj.decoder_real, z_imag) + predict(obj.decoder_imag, z_real);
        end
        
        function [reco_real, reco_imag] = forward(obj, x_real, x_imag)
            % Paso completo de encoding y decoding
            [z_real, z_imag] = obj.encode(x_real, x_imag);
            [reco_real, reco_imag] = obj.decode(z_real, z_imag);
        end
    end
end
