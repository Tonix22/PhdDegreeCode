% Parámetros de la red
input_dim = 100;
latent_dim = 50;
num_samples = 5000; % Número de muestras de entrenamiento
batch_size = 32;

% Crear datos de entrada (simulados, puedes usar tus datos reales)
x_real = dlarray(randn([input_dim, num_samples], 'single'), 'CB'); % Parte real
x_imag = dlarray(randn([input_dim, num_samples], 'single'), 'CB'); % Parte imaginaria

% Dividir en entrenamiento (80%) y validación (20%)
split_idx = round(0.8 * num_samples);
x_real_train = x_real(:, 1:split_idx);
x_imag_train = x_imag(:, 1:split_idx);
x_real_val = x_real(:, split_idx+1:end);
x_imag_val = x_imag(:, split_idx+1:end);


%% 2. Instanciar modelo
model = ComplexAutoencoderFC(input_dim, latent_dim);

%% 3. Definir la Función de Pérdida y Optimización

% Función de pérdida personalizada (MSE en ambas partes)
lossFcn = @(pred_real, pred_imag, target_real, target_imag) ...
    mean((pred_real - target_real).^2, 'all') + ...
    mean((pred_imag - target_imag).^2, 'all');

%% 4. Definir la Función de Forward y Backpropagation

% Función de forward + cálculo de pérdida + gradientes
function [loss, grad_real, grad_imag] = modelLoss(model, x_real, x_imag, target_real, target_imag)
    % Forward pass
    [pred_real, pred_imag] = model.forward(x_real, x_imag);

    % Calcular pérdida
    loss = lossFcn(pred_real, pred_imag, target_real, target_imag);

    % Backpropagation
    [grad_real, grad_imag] = dlgradient(loss, model.encoder_real.Learnables, model.encoder_imag.Learnables);
end

%% 5. Configurar el Entrenamiento
% Parámetros del optimizador
learning_rate = 0.001;
num_epochs = 50;

% Inicializar momentos del optimizador (necesarios para Adam)
velocity_real = [];
velocity_imag = [];
momentum_real = [];
momentum_imag = [];

% Loop de entrenamiento
for epoch = 1:num_epochs
    for i = 1:batch_size:split_idx
        % Seleccionar batch
        batch_real = x_real_train(:, i:min(i+batch_size-1, split_idx));
        batch_imag = x_imag_train(:, i:min(i+batch_size-1, split_idx));

        % 🔥 USAMOS `dlfeval` PARA TRAZAR GRADIENTES 🔥
        [loss, grad_real, grad_imag] = dlfeval(@modelLoss, model, batch_real, batch_imag, batch_real, batch_imag);

        % Actualizar parámetros con Adam
        [model.encoder_real.Learnables, velocity_real, momentum_real] = adamupdate(...
            model.encoder_real.Learnables, grad_real, velocity_real, momentum_real, epoch, learning_rate);

        [model.encoder_imag.Learnables, velocity_imag, momentum_imag] = adamupdate(...
            model.encoder_imag.Learnables, grad_imag, velocity_imag, momentum_imag, epoch, learning_rate);
    end

    % Mostrar pérdida cada época
    disp(['Epoch ', num2str(epoch), ': Loss = ', num2str(extractdata(loss))]);
end


%% 6. Evaluar el Modelo en Datos de Validación

% Forward pass en datos de validación
[pred_real_val, pred_imag_val] = model.forward(x_real_val, x_imag_val);

% Calcular pérdida en validación
val_loss = lossFcn(pred_real_val, pred_imag_val, x_real_val, x_imag_val);
disp(['Validation Loss: ', num2str(extractdata(val_loss))]);


%% 7. Probar el Modelo con Nuevos Datos

% Datos de prueba (ejemplo aleatorio)
new_real = dlarray(randn([input_dim, 1], 'single'), 'CB');
new_imag = dlarray(randn([input_dim, 1], 'single'), 'CB');

% Propagación en la red
[reco_real, reco_imag] = model.forward(new_real, new_imag);

% Mostrar resultados
disp('Entrada Real:'); disp(extractdata(new_real));
disp('Reconstrucción Real:'); disp(extractdata(reco_real));

disp('Entrada Imaginaria:'); disp(extractdata(new_imag));
disp('Reconstrucción Imaginaria:'); disp(extractdata(reco_imag));

