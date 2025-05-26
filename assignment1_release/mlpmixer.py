from torch import nn
import torch
import math
import matplotlib.pyplot as plt
import os


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size, patch_size, in_chans=3, 
                 embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        
        # Uncomment this line and replace ? with correct values
        #self.proj = nn.Conv2d(?, ?, kernel_size=?, stride=?)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        :param x: image tensor of shape [batch, channels, img_size, img_size]
        :return out: [batch. num_patches, embed_dim]
        """
        _, _, H, W = x.shape
        assert H == self.img_size, f"Input image height ({H}) doesn't match model ({self.img_size})."
        assert W == self.img_size, f"Input image width ({W}) doesn't match model ({self.img_size})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks """
    def __init__(
            self,
            in_features,
            hidden_features,
            act_layer=nn.GELU,
            drop=0.,
    ):
        super(Mlp, self).__init__()
        out_features = in_features
        hidden_features = hidden_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MixerBlock(nn.Module):
    """ Residual Block w/ token mixing and channel MLPs
    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    def __init__(
            self, dim, seq_len, mlp_ratio=(0.5, 4.0),
            activation='gelu', drop=0., drop_path=0.):
        super(MixerBlock, self).__init__()
        act_layer = {'gelu': nn.GELU, 'relu': nn.ReLU}[activation]
        tokens_dim, channels_dim = int(mlp_ratio[0] * dim), int(mlp_ratio[1] * dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6) # norm1 used with mlp_tokens
        self.mlp_tokens = Mlp(seq_len, tokens_dim, act_layer=act_layer, drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6) # norm2 used with mlp_channels
        self.mlp_channels = Mlp(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        
        y = self.norm1(x)         # [B, N, C]
        y = y.transpose(1, 2)     # [B, C, N] -> prepare to mix tokens along N dimension
        y = self.mlp_tokens(y)    # [B, C, N]
        y = y.transpose(1, 2)     # [B, N, C] back to original order
        x = x + y               # residual connection
        
        # Channel mixing
        z = self.norm2(x)         # [B, N, C]
        z = self.mlp_channels(z)  # [B, N, C]
        x = x + z               # residual connection
        
        return x
    

class MLPMixer(nn.Module):
    def __init__(self, num_classes, img_size, patch_size, embed_dim, num_blocks, 
                 drop_rate=0., activation='gelu', mlp_ratio=(0.5, 4.0)):
        super(MLPMixer, self).__init__()
        self.patchemb = PatchEmbed(img_size=img_size, 
                                   patch_size=patch_size, 
                                   in_chans=3,
                                   embed_dim=embed_dim)
        self.blocks = nn.Sequential(*[
            MixerBlock(
                dim=embed_dim, seq_len=self.patchemb.num_patches, 
                activation=activation, drop=drop_rate,)
            for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.num_classes = num_classes
        self.apply(self.init_weights)


    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, images):
        """ MLPMixer forward
        :param images: [batch, 3, img_size, img_size]
        """
        # step1: Go through the patch embedding
        # step 2 Go through the mixer blocks
        # step 3 go through layer norm
        # step 4 Global averaging spatially
        # Classification
        # Step 1: Go through the patch embedding
        x = self.patchemb(images)         # [B, num_patches, embed_dim]
        # Step 2: Go through the mixer blocks
        x = self.blocks(x)
        # Step 3: Apply layer normalization
        x = self.norm(x)
        # Step 4: Global average pooling over tokens (patches)
        x = x.mean(dim=1)                 # [B, embed_dim]
        # Classification head
        x = self.head(x)                  # [B, num_classes]
        return x
    
    def visualize(self, logdir="/home/ethan/IFT6135/IFT6135-2025/HW1_2025/assignment1_release/plots/Q4_6"):
        """ Visualize the token mixer layer 
        in the desired directory """
        
        import matplotlib.pyplot as plt
        import os
        import torch

        os.makedirs(logdir, exist_ok=True)
        
        # 1) Retrieve the first MixerBlock.
        first_block = self.blocks[0]
        
        # 2) Get the token-mixing MLP (mlp_tokens) first Linear layer (fc1).
        #    Shape of fc1.weight is [hidden_features, seq_len].
        fc1_weights = first_block.mlp_tokens.fc1.weight.data.clone().cpu()
        
        # 3) Create a heatmap of these weights.
        plt.figure(figsize=(8, 6))
        plt.imshow(fc1_weights, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title("MLP-Mixer Token-Mixing (fc1) Weights\n(First Block, First Layer)")
        plt.xlabel("Token")
        plt.ylabel("Hidden Units")
        
        # 4) Save the figure.
        save_path = os.path.join(logdir, "mlpmixer_token_mixing_fc1.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Token-mixing MLP (fc1) weights visualized and saved to {save_path}")
    
