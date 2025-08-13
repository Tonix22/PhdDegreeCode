import torch
from torch import nn
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights

def mobilenet_v3_48x48(num_outputs: int = 48,
                       pretrained: bool = True,
                       freeze_backbone: bool = False) -> nn.Module:
    """
    Devuelve una MobileNetV3-Small modificada para
    1 canal de entrada y `num_outputs` salidas.
    """
    # 1) Cargar el modelo base
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model   = models.mobilenet_v3_small(weights=weights)

    # 2) Cambiar la primera capa Conv2d (3 → 1 canal)
    first_conv: nn.Conv2d = model.features[0][0]
    new_first = nn.Conv2d(
        in_channels = 1,
        out_channels= first_conv.out_channels,
        kernel_size = first_conv.kernel_size,
        stride      = first_conv.stride,
        padding     = first_conv.padding,
        bias        = first_conv.bias is not None,
    )

    # Si usas pesos pre-entrenados: promediar sobre el eje de canales RGB
    if pretrained:
        with torch.no_grad():
            new_first.weight[:] = first_conv.weight.mean(dim=1, keepdim=True)

    model.features[0][0] = new_first

    # 3) Ajustar la cabeza de clasificación
    in_feats = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feats, num_outputs)

    # 4) (Opcional) Congelar el resto del backbone
    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False

    return model


if __name__ == "__main__":
    net = mobilenet_v3_48x48(num_outputs=48, pretrained=True)
    x   = torch.randn(4, 1, 48, 48)  # batch de 4 imágenes
    y   = net(x)
    print(y.shape)   # -> torch.Size([4, 48])
