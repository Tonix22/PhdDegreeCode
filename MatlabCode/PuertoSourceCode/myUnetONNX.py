import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image

# 1) Cargar modelo ONNX y crear la sesión
onnx_model = onnx.load("/home/tonix/Documents/PhdDegreeCode/MatlabCode/PuertoSourceCode/unet_model.onnx")  
onnx.checker.check_model(onnx_model)
session = ort.InferenceSession("/home/tonix/Documents/PhdDegreeCode/MatlabCode/PuertoSourceCode/unet_model.onnx")

# 2) Preparar función para hacer inferencia de un solo canal
def inference_single_channel(channel_img, session, input_name):
    """
    channel_img: objeto PIL con escala de grises (1 canal)
    session: onnxruntime.InferenceSession
    input_name: nombre del tensor de entrada en el modelo
    """
    # Convertir a numpy float32 en [0,1]
    arr = np.asarray(channel_img, dtype=np.float32) / 255.0
    # Redimensionar a [1, 1, 256, 256]
    arr = np.expand_dims(arr, axis=0)  # (256,256) -> (1,256,256)
    arr = np.expand_dims(arr, axis=0)  # -> (1,1,256,256)

    # Inferencia
    output = session.run(None, {input_name: arr})[0]  # Primer (y único) output
    # output.shape = (1,1,256,256)

    # Quitar dimensiones extra
    output = output[0,0,:,:]  # (256,256)

    # Reescalar a [0,255] y convertir a uint8
    output = (output*255.0).clip(0,255).astype(np.uint8)

    # Convertir a imagen PIL
    return Image.fromarray(output)

# 3) Leer la imagen y separar canales
img_color = Image.open("/home/tonix/Documents/PhdDegreeCode/MatlabCode/PuertoSourceCode/noise_image.jpg").convert("RGB").resize((256,256), Image.BICUBIC)
r, g, b = img_color.split()  # 3 objetos PIL de 1 canal cada uno

# Nombre de la entrada en tu modelo
input_name = session.get_inputs()[0].name

# 4) Pasar cada canal por el modelo
denoised_r = inference_single_channel(r, session, input_name)
denoised_g = inference_single_channel(g, session, input_name)
denoised_b = inference_single_channel(b, session, input_name)

# 5) Recombinar canales
denoised_rgb = Image.merge("RGB", (denoised_r, denoised_g, denoised_b))
denoised_rgb.save("clean_image.jpg")
