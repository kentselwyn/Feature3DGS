from torch import nn
import torch.nn.functional as F


class RefinedScoreNet(nn.Module):
    def __init__(self, superpoint_model, sptrain=False):
        super().__init__()
        self.superpoint = superpoint_model
        if sptrain:
            for param in self.superpoint.parameters():
                param.requires_grad = True
        else:
            for param in self.superpoint.parameters():
                param.requires_grad = False
        self.refine_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )
    def forward(self, image):
        if image.shape[1] == 3:  # RGB
            scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image * scale).sum(1, keepdim=True)
        x = self.superpoint.relu(self.superpoint.conv1a(image))
        x = self.superpoint.relu(self.superpoint.conv1b(x))
        x = self.superpoint.pool(x)
        x = self.superpoint.relu(self.superpoint.conv2a(x))
        x = self.superpoint.relu(self.superpoint.conv2b(x))
        x = self.superpoint.pool(x)
        x = self.superpoint.relu(self.superpoint.conv3a(x))
        x = self.superpoint.relu(self.superpoint.conv3b(x))
        x = self.superpoint.pool(x)
        x = self.superpoint.relu(self.superpoint.conv4a(x))
        x = self.superpoint.relu(self.superpoint.conv4b(x))
        cPa = self.superpoint.relu(self.superpoint.convPa(x))
        pred_score_map = self.refine_head(cPa)  # size: [B, 1, H, W]
        upsampled = F.interpolate(pred_score_map, scale_factor=8, mode='bilinear', align_corners=False)
        return upsampled  # shape [B, 1, H*8, W*8]
    

class RefinedScoreNet2(nn.Module):
    def __init__(self, superpoint_model, sptrain=False):
        super().__init__()
        self.superpoint = superpoint_model
        if sptrain:
            for param in self.superpoint.parameters():
                param.requires_grad = True
        else:
            for param in self.superpoint.parameters():
                param.requires_grad = False

        self.refine_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 多一層
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(1, 1, kernel_size=8, stride=8),
            nn.ReLU()
        )

    def forward(self, image):
        if image.shape[1] == 3:  # RGB to grayscale
            scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image * scale).sum(1, keepdim=True)
        # ↓ SuperPoint backbone to convPa
        x = self.superpoint.relu(self.superpoint.conv1a(image))
        x = self.superpoint.relu(self.superpoint.conv1b(x))
        x = self.superpoint.pool(x)
        x = self.superpoint.relu(self.superpoint.conv2a(x))
        x = self.superpoint.relu(self.superpoint.conv2b(x))
        x = self.superpoint.pool(x)
        x = self.superpoint.relu(self.superpoint.conv3a(x))
        x = self.superpoint.relu(self.superpoint.conv3b(x))
        x = self.superpoint.pool(x)
        x = self.superpoint.relu(self.superpoint.conv4a(x))
        x = self.superpoint.relu(self.superpoint.conv4b(x))
        cPa = self.superpoint.relu(self.superpoint.convPa(x))  # shape: [B, 256, H/8, W/8]
        pred_score_map = self.refine_head(cPa)  # shape: [B, 1, H/8, W/8]
        upsampled = self.upsample(pred_score_map)  # shape: [B, 1, H, W]
        return upsampled
    

class RefinedScoreNet3(nn.Module):
    def __init__(self, superpoint_model, sptrain=False):
        super().__init__()
        self.superpoint = superpoint_model
        if sptrain:
            for param in self.superpoint.parameters():
                param.requires_grad = True
        else:
            for param in self.superpoint.parameters():
                param.requires_grad = False
        self.refine_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 多一層
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )
    def forward(self, image):
        if image.shape[1] == 3:  # RGB
            scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image * scale).sum(1, keepdim=True)
        x = self.superpoint.relu(self.superpoint.conv1a(image))
        x = self.superpoint.relu(self.superpoint.conv1b(x))
        x = self.superpoint.pool(x)
        x = self.superpoint.relu(self.superpoint.conv2a(x))
        x = self.superpoint.relu(self.superpoint.conv2b(x))
        x = self.superpoint.pool(x)
        x = self.superpoint.relu(self.superpoint.conv3a(x))
        x = self.superpoint.relu(self.superpoint.conv3b(x))
        x = self.superpoint.pool(x)
        x = self.superpoint.relu(self.superpoint.conv4a(x))
        x = self.superpoint.relu(self.superpoint.conv4b(x))
        cPa = self.superpoint.relu(self.superpoint.convPa(x))
        pred_score_map = self.refine_head(cPa)  # size: [B, 1, H, W]
        upsampled = F.interpolate(pred_score_map, scale_factor=8, mode='bilinear', align_corners=False)
        return upsampled  # shape [B, 1, H*8, W*8]
