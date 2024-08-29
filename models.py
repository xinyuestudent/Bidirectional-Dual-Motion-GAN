import torch.nn as nn
import torch.nn.functional as F
import torch
import coordatt as crt
from models import *

class CrossAttentionLayer(nn.Module):
    def __init__(self, depth_feature_dim, feature_map_dim, num_heads):
        super(CrossAttentionLayer, self).__init__()
        self.depth_feature_dim = depth_feature_dim
        self.feature_map_dim = feature_map_dim
        self.num_heads = num_heads
        self.head_dim = depth_feature_dim // num_heads
        
        assert self.head_dim * num_heads == self.depth_feature_dim, "Depth feature dimension must be divisible by number of heads"
        
        # 定义线性变换层
        self.q_proj = nn.Linear(depth_feature_dim, depth_feature_dim)
        self.k_proj = nn.Linear(feature_map_dim, depth_feature_dim)
        self.v_proj = nn.Linear(feature_map_dim, depth_feature_dim)
        self.out_proj = nn.Linear(depth_feature_dim, depth_feature_dim)

    def forward(self, depth_feature, feature_map):
        # depth_feature: (batch_size, depth_feature_dim)
        # feature_map:   (batch_size, channels, height, width)
        
        batch_size = depth_feature.size(0)
        height, width = feature_map.size(-2), feature_map.size(-1)
        
        # 将特征图展平成 (batch_size, channels, height * width)
        feature_map_flat = feature_map.view(batch_size, self.feature_map_dim, -1).permute(0, 2, 1)  # (batch_size, height * width, feature_map_dim)
        
        # 对查询、键、值进行线性变换
        q = self.q_proj(depth_feature)  # (batch_size, depth_feature_dim)
        k = self.k_proj(feature_map_flat)  # (batch_size, height * width, depth_feature_dim)
        v = self.v_proj(feature_map_flat)  # (batch_size, height * width, depth_feature_dim)
        
        # 将向量拆分为多头
        q = q.view(batch_size, 1, self.num_heads, self.head_dim).repeat(1, height * width, 1, 1).permute(0, 2, 1, 3)  # (batch_size, num_heads, height * width, head_dim)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (batch_size, num_heads, height * width, head_dim)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (batch_size, num_heads, height * width, head_dim)
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size, num_heads, height * width, height * width)
        
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch_size, num_heads, height * width, height * width)
        
        # 加权求和
        attn_output = torch.matmul(attn_weights, v)  # (batch_size, num_heads, height * width, head_dim)
        
        # 拼接多头
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.depth_feature_dim)  # (batch_size, height * width, depth_feature_dim)
        
        # 输出线性变换
        attn_output = self.out_proj(attn_output)  # (batch_size, height * width, depth_feature_dim)
        
        # 将输出重新塑形为特征图的形状
        attn_output = attn_output.view(batch_size, height, width, -1).permute(0, 3, 1, 2)  # (batch_size, depth_feature_dim, height, width)
        
        return attn_output, attn_weights


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

##############################
#           RESNET
##############################

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        # print(self.block)
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()
        channels = (input_shape[1]-1)*input_shape[2]
        # Initial convolution block
        
        out_features = 128
        layer = [
            nn.ReflectionPad2d(3),# (3,3,3,3)
            nn.Conv2d(channels, out_features, 7),
            # nn.InstanceNorm2d(out_features),
            # nn.ReLU(inplace=True),
        ]
        in_features = out_features
        out_features //= 2
        # Downsampling
        for _ in range(2):
            out_features *= 2
            layer += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
        #cross_attention
		# 创建交叉注意力层
        self.cross_attn_layer = CrossAttentionLayer(out_features, out_features, 8)
        # Residual blocks
        
        for _ in range(num_residual_blocks):
            layer += [ResidualBlock(out_features)]
        self.layer = nn.Sequential(*layer) 
        self.coord = crt.CoordAtt(in_features, out_features)
        model = [
            nn.ReflectionPad2d(1),# (3,3,3,3)
            nn.Conv2d(in_features, out_features, 3),
            # nn.InstanceNorm2d(out_features),
            # nn.ReLU(inplace=True),
        ]
        # Upsampling
        # for _ in range(2):
        out_features //= 2
        model += [
            # nn.Upsample(scale_factor=2),
            # nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
            nn.ConvTranspose2d(in_features, out_features, (3, 3), stride=(2, 2), padding=(1, 1),output_padding=(1,1)),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        out_features = in_features * 2
        model += [
            nn.ConvTranspose2d(in_features, out_features, (3, 3), stride=(2, 2), padding=(1, 1),output_padding=(1,1)),
            # nn.Upsample(scale_factor=2),
            # nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(3), nn.Conv2d(out_features, 3, 7), nn.Tanh()]
        self.blocks = nn.ModuleList([
                Block(
                        dim=embed_dim//4, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer, is_mask=0)
                 ])
				 
        self.model = nn.Sequential(*model)

        
    def forward(self, x,model):
        x_image_0 = self.layer(x)
        x_image_1 = self.coord(x_image_0)
		
		depth = model.predict_depth(x, num_steps=2, ensemble_size=4) # (h, w) in [0, 1]
        depth = depth.squeeze(0).squeeze(0).cpu().numpy()  

        x_depth_0 = self.layer(depth)
        x_depth_1 = self.coord(x_depth_0)
		
		output, attn_weights = self.cross_attn_layer(x_image_1,x_depth_1)

        x2 = self.model(output) 
        return x2


##############################
#        Discriminator
##############################


class DiscriminatorA(nn.Module):
    def __init__(self, input_shape):
        super(DiscriminatorA, self).__init__()

        channels, height, width = input_shape[2:]

        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)

class DiscriminatorB(nn.Module):
    def __init__(self, input_shape):
        super(DiscriminatorB, self).__init__()

        channels = input_shape[1]*input_shape[2]
        height, width = input_shape[3:]

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)

class DiscriminatorF(nn.Module):
    def __init__(self, input_shape):
        super(DiscriminatorF, self).__init__()

        channels = input_shape[1]*input_shape[2]
        height, width = input_shape[3:]

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(1, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)    

if __name__ == '__main__':
    img_height = img_width = 256

    input_shape = (1,5,3,img_height, img_width)

    G = GeneratorResNet(input_shape,9)

    test_T = torch.rand((2,12,img_height,img_width))
    from torchsummary import summary
    summary(G, input_size=(12,img_height, img_width) )
    output = G(test_T)
    print(output.size())
