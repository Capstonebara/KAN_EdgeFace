import torch
from torch import nn
from timm.layers import trunc_normal_
from layers.Edge_layers.custom_layers import LayerNorm, PositionalEncodingFourier
from layers.Edge_layers.sdta_encoder import SDTAEncoder_KANLinear
from layers.Edge_layers.conv_encoder import ConvEncoder_KANLinear
from layers.Edge_layers.LoRaLin import LoRaLin_KAN


class EdgeFace_KANLinear(nn.Module):
    def __init__(self, in_chans=3, num_classes=512,
                 depths=[3, 3, 9, 3], dims=[48, 96, 160, 304],
                 global_block=[0, 0, 0, 3], global_block_type=['None', 'None', 'None', 'SDTA'],
                 drop_path_rate=0., layer_scale_init_value=1e-6, head_init_scale=1., expan_ratio=4,
                 kernel_sizes=[7, 7, 7, 7], heads=[8, 8, 8, 8], use_pos_embd_xca=[False, False, False, False],
                 use_pos_embd_global=False, d2_scales=[2, 3, 4, 5], **kwargs):
        super().__init__()
        for g in global_block_type:
            assert g in ['None', 'SDTA']
        
        self.pos_embd = PositionalEncodingFourier(dim=dims[0]) if use_pos_embd_global else None

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        self.downsample_layers.append(nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        ))
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
            ))

        self.stages = nn.ModuleList()  # Feature resolution stages
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            blocks = []
            for j in range(depths[i]):
                if j > depths[i] - global_block[i] - 1:
                    if global_block_type[i] == 'SDTA':
                        blocks.append(SDTAEncoder_KANLinear(
                            dim=dims[i], drop_path=dp_rates[cur + j], expan_ratio=expan_ratio,
                            scales=d2_scales[i], use_pos_emb=use_pos_embd_xca[i], num_heads=heads[i]
                        ))
                    else:
                        raise NotImplementedError
                else:
                    blocks.append(ConvEncoder_KANLinear(
                        dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value,
                        expan_ratio=expan_ratio, kernel_size=kernel_sizes[i]
                    ))
            self.stages.append(nn.Sequential(*blocks))
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = LoRaLin_KAN(dims[-1], num_classes)

        # Apply weight initialization
        self.apply(self._init_weights)
        self.head_dropout = nn.Dropout(kwargs.get("classifier_dropout", 0.0))

        # Weight scaling
        if hasattr(self.head, 'weight') and hasattr(self.head, 'bias'):
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        if self.pos_embd:
            B, C, H, W = x.shape
            x = x + self.pos_embd(B, H, W)
        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # Global average pooling

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(self.head_dropout(x))
        return x
