import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet34, ResNet34_Weights
from typing import Tuple, List
from utils.logger import setup_logger

logger = setup_logger(__name__, "logs/model.log")

class ChannelSE(nn.Module):
    """Channel Squeeze and Excitation block."""
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SpatialSE(nn.Module):
    """Spatial Squeeze and Excitation block."""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = torch.sigmoid(y)
        return x * y

class scSE(nn.Module):
    """Concurrent Spatial and Channel Squeeze and Excitation block."""
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.cSE = ChannelSE(channels, reduction)
        self.sSE = SpatialSE(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cSE(x) + self.sSE(x)

class FPA(nn.Module):
    """Feature Pyramid Attention module."""
    def __init__(self, channels: int) -> None:
        super().__init__()
        
        # Global pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 7x7 branch
        self.branch_7 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 5x5 branch
        self.branch_5 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 branch
        self.branch_3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Fuse branch
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.size(2), x.size(3)
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=(h, w), mode='bilinear', align_corners=True)
        
        feat_7 = self.branch_7(x)
        feat_5 = self.branch_5(x)
        feat_3 = self.branch_3(x)
        
        concat = torch.cat([global_feat, feat_7, feat_5, feat_3], dim=1)
        output = self.fuse(concat)
        return output

class MultiScaleTransformerAttention(nn.Module):
    """Multi-scale transformer attention module."""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.query_convs = nn.ModuleList([
            nn.Conv2d(channels, channels // 8, 1)
            for _ in range(3)  # 1x, 1/2x, 1/4x scales
        ])
        self.key_convs = nn.ModuleList([
            nn.Conv2d(channels, channels // 8, 1)
            for _ in range(3)
        ])
        self.value_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, 1)
            for _ in range(3)
        ])
        self.fusion = nn.Conv2d(channels * 3, channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        scales = [(h, w), (h//2, w//2), (h//4, w//4)]
        outputs = []
        
        for i, (qconv, kconv, vconv) in enumerate(zip(self.query_convs, self.key_convs, self.value_convs)):
            # Scale input
            curr_x = F.interpolate(x, size=scales[i], mode='bilinear', align_corners=True) if i > 0 else x
            
            # Compute QKV
            q = qconv(curr_x).view(b, -1, scales[i][0] * scales[i][1]).permute(0, 2, 1)
            k = kconv(curr_x).view(b, -1, scales[i][0] * scales[i][1])
            v = vconv(curr_x).view(b, -1, scales[i][0] * scales[i][1]).permute(0, 2, 1)
            
            # Attention
            attn = torch.bmm(q, k) / ((c // 8) ** 0.5)
            attn = F.softmax(attn, dim=-1)
            out = torch.bmm(attn, v).permute(0, 2, 1)
            out = out.view(b, c, scales[i][0], scales[i][1])
            
            # Rescale back to original size
            if i > 0:
                out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
            outputs.append(out)
            
        return self.fusion(torch.cat(outputs, dim=1))

class ConvBlock(nn.Module):
    """(convolution => [BN] => ReLU) x 2 with scSE."""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.scse = scSE(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.double_conv(x)
        x = self.scse(x)
        return x

class UpBlock(nn.Module):
    """Upscale + ConvBlock.
       up_in: number of channels from the feature to be upsampled.
       skip_in: number of channels from the skip connection.
       out_channels: number of output channels (and channels from transposed conv output).
    """
    def __init__(self, up_in: int, skip_in: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(up_in, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_in, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ResNetEncoder(nn.Module):
    """Use pretrained ResNet as encoder backbone."""
    def __init__(self, backbone: str = 'resnet34') -> None:
        super().__init__()
        if backbone == 'resnet34':
            resnet = resnet34(weights=ResNet34_Weights.DEFAULT)
        else:
            raise NotImplementedError("Only resnet34 is supported in this demo.")
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )
        self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x0 = self.layer0(x)   # [64, H/2,  W/2]
        x1 = self.layer1(x0)  # [64, H/4,  W/4]
        x2 = self.layer2(x1)  # [128, H/8, W/8]
        x3 = self.layer3(x2)  # [256, H/16,W/16]
        x4 = self.layer4(x3)  # [512, H/32,W/32]
        return x0, x1, x2, x3, x4

class FeaturePyramidAttention(nn.Module):
    """Feature Pyramid Attention module."""
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.pyramid_levels = [1, 2, 4]  # Reduced from [1,2,4,8] to manage channels
        
        self.conv_1x1 = nn.Conv2d(in_channels, in_channels//4, kernel_size=1)
        
        self.pyramid_convs = nn.ModuleList([
            nn.Conv2d(in_channels//4, in_channels//4, kernel_size=1)
            for _ in self.pyramid_levels
        ])
        
        # Adjust fusion to handle correct number of channels
        # After concatenation we have in_channels channels (original_channels/4 * 4)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.size()[2:]
        feat = self.conv_1x1(x)
        pyramid_features = [feat]
        
        for level, conv in zip(self.pyramid_levels, self.pyramid_convs):
            pool_size = (size[0] // level, size[1] // level)
            pooled = F.adaptive_avg_pool2d(feat, pool_size)
            processed = conv(pooled)
            upsampled = F.interpolate(processed, size=size, mode='bilinear', align_corners=True)
            pyramid_features.append(upsampled)
        
        pyramid_features = torch.cat(pyramid_features, dim=1)
        output = self.fusion(pyramid_features)
        
        return output + x  # Skip connection

class UNetResNet(nn.Module):
    """U-Net with ResNet encoder for salt deposit segmentation."""
    
    def __init__(self, n_classes: int = 1, in_channels: int = 4) -> None:
        super().__init__()
        logger.info(f"Initializing UNetResNet with {in_channels} input channels")
        # Load pretrained ResNet34 as encoder
        self.encoder = ResNetEncoder(backbone='resnet34')

        # Update the first convolutional layer to accept in_channels
        self.encoder.layer0[0] = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Multi-scale transformer attention modules
        self.transformer_blocks = nn.ModuleList([
            MultiScaleTransformerAttention(64),  # After layer1
            MultiScaleTransformerAttention(128), # After layer2
            MultiScaleTransformerAttention(256), # After layer3
            MultiScaleTransformerAttention(512)  # After layer4
        ])

        # Feature Pyramid Attention modules
        self.fpa_blocks = nn.ModuleList([
            FeaturePyramidAttention(64),
            FeaturePyramidAttention(128),
            FeaturePyramidAttention(256),
            FeaturePyramidAttention(512)
        ])

        # Decoder blocks
        self.decoder4 = UpBlock(512, 256, 256)
        self.decoder3 = UpBlock(256, 128, 128)
        self.decoder2 = UpBlock(128, 64, 64)
        self.decoder1 = UpBlock(64, 64, 32)

        # Calculate total channels for hypercolumn
        # d1(32) + x1(64) + x2(128) + x3(256) + x4(512) = 992
        self.hypercolumn_fusion = nn.Sequential(
            nn.Conv2d(992, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, kernel_size=1)
        )

        # Final upsampling to match input size
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logger.debug(f"Forward pass shape: {x.shape}")
        # Get encoder features
        x0, x1, x2, x3, x4 = self.encoder(x)
        
        # Apply attention and FPA at each scale
        x1_att = self.transformer_blocks[0](self.fpa_blocks[0](x1))
        x2_att = self.transformer_blocks[1](self.fpa_blocks[1](x2))
        x3_att = self.transformer_blocks[2](self.fpa_blocks[2](x3))
        x4_att = self.transformer_blocks[3](self.fpa_blocks[3](x4))
        
        # Decoder path
        d4 = self.decoder4(x4_att, x3_att)
        d3 = self.decoder3(d4, x2_att)
        d2 = self.decoder2(d3, x1_att)
        d1 = self.decoder1(d2, x0)
        
        # Hypercolumn processing - concatenate features at same scale
        x1_up = F.interpolate(x1, size=d1.shape[2:], mode='bilinear', align_corners=True)
        x2_up = F.interpolate(x2, size=d1.shape[2:], mode='bilinear', align_corners=True)
        x3_up = F.interpolate(x3, size=d1.shape[2:], mode='bilinear', align_corners=True)
        x4_up = F.interpolate(x4, size=d1.shape[2:], mode='bilinear', align_corners=True)
        
        hypercolumn = torch.cat([d1, x1_up, x2_up, x3_up, x4_up], dim=1)
        output = self.hypercolumn_fusion(hypercolumn)
        
        # Final upsampling to match input size (128x128)
        output = self.final_upsample(output)
        
        return output
