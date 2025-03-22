from ComplexLinear import ComplexLinear
from ComplexRelu import complex_relu

class ComplexAutoencoderFC(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(ComplexAutoencoderFC, self).__init__()
        # Encoder: reduce dimensión de input_dim -> latent_dim
        self.encoder = ComplexLinear(input_dim, latent_dim)
        # Decoder: reconstruye de latent_dim -> input_dim
        self.decoder = ComplexLinear(latent_dim, input_dim)
    def forward(self, x_real, x_imag):
        # Paso de codificación
        z_real, z_imag = self.encoder(x_real, x_imag)
        z_real, z_imag = complex_relu(z_real, z_imag)  # activación no lineal
        # Paso de decodificación
        reco_real, reco_imag = self.decoder(z_real, z_imag)
        # (Opcional: aplicar activación a la salida dependiendo del caso de uso.
        #  Por ejemplo, sigmoid si se desea restringir [0,1], etc.)
        return reco_real, reco_imag
