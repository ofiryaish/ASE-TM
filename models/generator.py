import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from .codec_module import DenseEncoder, HyperDenseEncoder, MagDecoder, PhaseDecoder


class GaussianMaskedMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Trainable t for each head (ensures σ = t² is positive)
        self.t = nn.Parameter(torch.randn(num_heads))  # (num_heads,)

    def forward(self, x, x_copy1=None, x_copy2=None):
        x_copy1, x_copy2 = None, None  # Unused copies
        batch_size, seq_len, _ = x.shape

        # Compute position indices (shape: [seq_len, 1] and [1, seq_len])
        j = torch.arange(seq_len, device=x.device).unsqueeze(0)
        k = j.T  # Transpose to create a distance matrix

        # Compute Gaussian mask for each head
        sigma = self.t ** 2  # Reparameterization: σ = t² (ensures positivity)
        sigma = sigma.view(self.num_heads, 1, 1)  # (num_heads, 1, 1) for broadcasting

        gaussian_mask = -(j - k) ** 2 / (2 * sigma ** 2)  # (num_heads, seq_len, seq_len)
        # Each head with different mask - https://stackoverflow.com/q/68205894
        gaussian_mask = gaussian_mask.repeat(batch_size, 1, 1)  # (batch_size * num_heads, seq_len, seq_len)
        # Apply attention with Gaussian mask
        return self.mha(x, x, x, attn_mask=gaussian_mask)


def sinusoids(length, channels, max_timescale=10000):
    """
    Returns sinusoids for positional embedding
    from https://github.com/openai/whisper/blob/517a43ecd132a2089d85f4ebc044728a71d49f6e/whisper/model.py#L62
    """
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class SEMamba(nn.Module):
    """
    SEMamba model for speech enhancement using Mamba blocks.

    This model uses a dense encoder, multiple Mamba blocks, and separate magnitude
    and phase decoders to process noisy magnitude and phase inputs.
    """
    def __init__(self, cfg):
        """
        Initialize the SEMamba model.

        Args:
        - cfg: Configuration object containing model parameters.
        """
        if cfg['model_cfg']['mamba_version'] == 1:
            from .mamba_block import TFMambaBlock
        elif cfg['model_cfg']['mamba_version'] == 2:
            from .mamba2_block import TFMambaBlock
        else:
            raise ValueError("Invalid value for Mamba version")
        super(SEMamba, self).__init__()
        self.cfg = cfg
        self.num_tscblocks = \
            cfg['model_cfg']['num_tfmamba'] if cfg['model_cfg']['num_tfmamba'] is not None else 4  # default tfmamba: 4

        # Initialize dense encoder
        self.dense_encoder = DenseEncoder(cfg)

        # Initialize Mamba blocks
        self.TSMamba = nn.ModuleList([TFMambaBlock(cfg) for _ in range(self.num_tscblocks)])
        print("Number of TFMamba blocks: ", self.num_tscblocks)

        # Initialize decoders
        self.mask_decoder = MagDecoder(cfg)
        self.phase_decoder = PhaseDecoder(cfg)

    def forward(self, noisy_mag, noisy_pha):
        """
        Forward pass for the SEMamba model.

        Args:
        - noisy_mag (torch.Tensor): Noisy magnitude input tensor [B, F, T].
        - noisy_pha (torch.Tensor): Noisy phase input tensor [B, F, T].

        Returns:
        - denoised_mag (torch.Tensor): Denoised magnitude tensor [B, F, T].
        - denoised_pha (torch.Tensor): Denoised phase tensor [B, F, T].
        - denoised_com (torch.Tensor): Denoised complex tensor [B, F, T, 2].
        """
        # Reshape inputs
        noisy_mag = rearrange(noisy_mag, 'b f t -> b t f').unsqueeze(1)  # [B, 1, T, F]
        noisy_pha = rearrange(noisy_pha, 'b f t -> b t f').unsqueeze(1)  # [B, 1, T, F]

        # Concatenate magnitude and phase inputs
        x = torch.cat((noisy_mag, noisy_pha), dim=1)  # [B, 2, T, F]

        # Encode input
        x = self.dense_encoder(x)

        # Apply Mamba blocks
        for block in self.TSMamba:
            x = block(x)

        # Decode magnitude and phase
        denoised_mag = rearrange(self.mask_decoder(x) * noisy_mag, 'b c t f -> b f t c').squeeze(-1)
        denoised_pha = rearrange(self.phase_decoder(x), 'b c t f -> b f t c').squeeze(-1)

        # Combine denoised magnitude and phase into a complex representation
        denoised_com = torch.stack(
            (denoised_mag * torch.cos(denoised_pha), denoised_mag * torch.sin(denoised_pha)),
            dim=-1
        )

        return denoised_mag, denoised_pha, denoised_com


class SEMambaCoDe2dReAt(SEMamba):
    """
    SEMambaAt model for speech enhancement using Mamba blocks with attention between.

    This model uses a dense encoder, multiple Mamba blocks with attention between blocks, and separate magnitude
    and phase decoders to process noisy magnitude and phase inputs.
    """
    def __init__(self, cfg):
        """
        Initialize the SEMamba model.

        Args:
        - cfg: Configuration object containing model parameters.
        """
        super().__init__(cfg)
        self.reduce = nn.Sequential(
            nn.Conv2d(
                in_channels=cfg['model_cfg']['hid_feature'],
                out_channels=cfg['model_cfg']['attention_reduce_out_channels'],
                kernel_size=cfg['model_cfg']['attention_reduce_kernel_size'],
                padding=cfg['model_cfg']['attention_reduce_padding'],
                stride=cfg['model_cfg']['attention_reduce_stride'],
                groups=cfg['model_cfg']['attention_reduce_group_size']),
            nn.InstanceNorm2d(cfg['model_cfg']['attention_reduce_out_channels'], affine=True),
            nn.PReLU(cfg['model_cfg']['attention_reduce_out_channels'])
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=cfg['model_cfg']['attention_embed_dim'],
            num_heads=cfg['model_cfg']['attention_num_heads'], batch_first=True)

        self.expand = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=cfg['model_cfg']['attention_reduce_out_channels'],
                out_channels=cfg['model_cfg']['hid_feature'],
                kernel_size=cfg['model_cfg']['attention_expand_kernel_size'],
                padding=cfg['model_cfg']['attention_expand_padding'],
                stride=cfg['model_cfg']['attention_expand_stride'],
                groups=cfg['model_cfg']['attention_expand_group_size']),
            nn.InstanceNorm2d(cfg['model_cfg']['hid_feature'], affine=True),
            nn.PReLU(cfg['model_cfg']['hid_feature'])
        )

        if cfg['model_cfg']['attention_positional_enccoding_len'] != "None":
            self.register_buffer("positional_embedding",
                                 sinusoids(cfg['model_cfg']['attention_positional_enccoding_len'],
                                           cfg['model_cfg']['attention_embed_dim']))

    def forward(self, noisy_mag, noisy_pha):
        """
        Forward pass for the SEMambaAt model.

        Args:
        - noisy_mag (torch.Tensor): Noisy magnitude input tensor [B, F, T].
        - noisy_pha (torch.Tensor): Noisy phase input tensor [B, F, T].

        Returns:
        - denoised_mag (torch.Tensor): Denoised magnitude tensor [B, F, T].
        - denoised_pha (torch.Tensor): Denoised phase tensor [B, F, T].
        - denoised_com (torch.Tensor): Denoised complex tensor [B, F, T, 2].
        """
        # Reshape inputs
        noisy_mag = rearrange(noisy_mag, 'b f t -> b t f').unsqueeze(1)  # [B, 1, T, F]
        noisy_pha = rearrange(noisy_pha, 'b f t -> b t f').unsqueeze(1)  # [B, 1, T, F]

        # Concatenate magnitude and phase inputs
        x = torch.cat((noisy_mag, noisy_pha), dim=1)  # [B, 2, T, F]

        # Encode input
        x = self.dense_encoder(x)
        # Apply Mamba blocks
        for block_i, block in enumerate(self.TSMamba):
            if block_i == (self.num_tscblocks // 2):
                # Apply attention mechanism between blocks
                x_pre_att = self.reduce(x)  # [B, C, T, F] where C and F (optionaly) are reduced
                channels_num = x_pre_att.shape[1]
                x_pre_att = rearrange(x_pre_att, 'b c t f -> b t (f c)')  # Reshape for attention: [B, T, F*C]
                if hasattr(self, "positional_embedding"):
                    # Repeat the positional embedding if T is greater than its length
                    if x_pre_att.size(1) > self.positional_embedding.size(0):
                        repeat_factor = (x_pre_att.size(1) // self.positional_embedding.size(0)) + 1
                        extended_positional_embedding = self.positional_embedding.repeat(repeat_factor, 1)
                    else:
                        extended_positional_embedding = self.positional_embedding

                    # Slice the extended positional embedding to match the length of x_pre_att
                    x_pre_att = x_pre_att + extended_positional_embedding[:x_pre_att.size(1), :].unsqueeze(0)
                x_att, _ = self.attention(x_pre_att, x_pre_att, x_pre_att)
                x_att = rearrange(x_att, 'b t (f c) -> b c t f', c=channels_num)  # [B, C, T, F]
                x_att = self.expand(x_att)  # [B, C, T, F] where C and F (optionaly) are expanded

                x = x + x_att  # Residual connection
            x = block(x)

        # Decode magnitude and phase
        denoised_mag = rearrange(self.mask_decoder(x) * noisy_mag, 'b c t f -> b f t c').squeeze(-1)
        denoised_pha = rearrange(self.phase_decoder(x), 'b c t f -> b f t c').squeeze(-1)

        # Combine denoised magnitude and phase into a complex representation
        denoised_com = torch.stack(
            (denoised_mag * torch.cos(denoised_pha), denoised_mag * torch.sin(denoised_pha)),
            dim=-1
        )

        return denoised_mag, denoised_pha, denoised_com


class SEMambaCoDe2dReGuMaAt(SEMambaCoDe2dReAt):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.attention = GaussianMaskedMultiheadAttention(
            embed_dim=cfg['model_cfg']['attention_embed_dim'],
            num_heads=cfg['model_cfg']['attention_num_heads'])


class SEHyperMambaCoDe2dReAt(SEMambaCoDe2dReAt):
    """
    SEMambaAt model for speech enhancement using Mamba blocks with attention between.

    This model uses a dense encoder, multiple Mamba blocks with attention between blocks, and separate magnitude
    and phase decoders to process noisy magnitude and phase inputs.
    """
    def __init__(self, cfg):
        """
        Initialize the SEMamba model.

        Args:
        - cfg: Configuration object containing model parameters.
        """
        super().__init__(cfg)

        self.hid_feature = cfg['model_cfg']['hid_feature']

        self.hyper_reduce = nn.Sequential(
            nn.Conv2d(
                in_channels=self.hid_feature,
                out_channels=cfg['model_cfg']['hyper_reduce_out_channels'],
                kernel_size=cfg['model_cfg']['hyper_reduce_kernel_size'],
                padding=cfg['model_cfg']['hyper_reduce_padding'],
                stride=cfg['model_cfg']['hyper_reduce_stride'],
                groups=cfg['model_cfg']['hyper_reduce_group_size']),
            nn.InstanceNorm2d(cfg['model_cfg']['hyper_reduce_out_channels'], affine=True),
            nn.PReLU(cfg['model_cfg']['hyper_reduce_out_channels'])
        )

        self.hyper_net = nn.Sequential(
            nn.Linear(cfg['model_cfg']['hyper_reduce_output_dim'], cfg['model_cfg']['hyper_reduce_output_dim'] * 2),
            nn.ReLU(),
            nn.Linear(cfg['model_cfg']['hyper_reduce_output_dim'] * 2, cfg['model_cfg']['hyper_reduce_output_dim'] * 2),
            nn.ReLU(),
            nn.Linear(cfg['model_cfg']['hyper_reduce_output_dim'] * 2, self.hid_feature * self.hid_feature)
        )

        self.dense_conv = nn.Sequential(
            nn.InstanceNorm2d(self.hid_feature, affine=True),
            nn.PReLU(self.hid_feature)
        )

    def forward(self, noisy_mag, noisy_pha):
        """
        Forward pass for the SEMambaAt model.

        Args:
        - noisy_mag (torch.Tensor): Noisy magnitude input tensor [B, F, T].
        - noisy_pha (torch.Tensor): Noisy phase input tensor [B, F, T].

        Returns:
        - denoised_mag (torch.Tensor): Denoised magnitude tensor [B, F, T].
        - denoised_pha (torch.Tensor): Denoised phase tensor [B, F, T].
        - denoised_com (torch.Tensor): Denoised complex tensor [B, F, T, 2].
        """
        # Reshape inputs
        noisy_mag = rearrange(noisy_mag, 'b f t -> b t f').unsqueeze(1)  # [B, 1, T, F]
        noisy_pha = rearrange(noisy_pha, 'b f t -> b t f').unsqueeze(1)  # [B, 1, T, F]

        # Concatenate magnitude and phase inputs
        x = torch.cat((noisy_mag, noisy_pha), dim=1)  # [B, 2, T, F]

        # Encode input
        x = self.dense_encoder(x)
        # Apply Mamba blocks
        for block_i, block in enumerate(self.TSMamba):
            if block_i == (self.num_tscblocks // 2):
                # Apply attention mechanism between blocks
                x_pre_att = self.reduce(x)  # [B, C, T, F] where C and F (optionaly) are reduced
                channels_num = x_pre_att.shape[1]
                x_pre_att = rearrange(x_pre_att, 'b c t f -> b t (f c)')  # Reshape for attention: [B, T, F*C]
                if hasattr(self, "positional_embedding"):
                    # Repeat the positional embedding if T is greater than its length
                    if x_pre_att.size(1) > self.positional_embedding.size(0):
                        repeat_factor = (x_pre_att.size(1) // self.positional_embedding.size(0)) + 1
                        extended_positional_embedding = self.positional_embedding.repeat(repeat_factor, 1)
                    else:
                        extended_positional_embedding = self.positional_embedding

                    # Slice the extended positional embedding to match the length of x_pre_att
                    x_pre_att = x_pre_att + extended_positional_embedding[:x_pre_att.size(1), :].unsqueeze(0)
                x_att, _ = self.attention(x_pre_att, x_pre_att, x_pre_att)
                x_att = rearrange(x_att, 'b t (f c) -> b c t f', c=channels_num)  # [B, C, T, F]
                x_att = self.expand(x_att)  # [B, C, T, F] where C and F (optionaly) are expanded

                x = x + x_att  # Residual connection
            # cov2d before TF-mamba
            batch_size, channels_num, time_len = x.size(0), x.size(1), x.size(2)
            x_reduced = self.hyper_reduce(x)
            generated_weights = self.hyper_net(
                x_reduced.permute(0, 2, 1, 3).reshape(
                    batch_size, time_len, -1))  # [batch, T, hid_feature * hid_feature]
            generated_weights = generated_weights.mean(dim=1)  # Shape: [batch, hid_feature * hid_feature]
            generated_weights = generated_weights.view(batch_size, channels_num, channels_num, 1, 1)
            # Process each sample in the batch separately
            output = []
            for i in range(batch_size):
                # TODO: Check if this is the correct way to apply convolution with different weights for each sample
                # https://discuss.pytorch.org/t/how-to-run-functional-conv2d-with-different-weights-for-each-sample-in-batch/136364/2

                # Use the generated weights for convolution
                sample = x[i:i+1]  # Keep batch dimension: [1, hid_feature, time, freq]
                weights = generated_weights[i]  # [hid_feature, hid_feature, 1, 1]

                # Apply convolution with the generated weights
                conv_out = nn.functional.conv2d(sample, weights, stride=(1, 1))
                output.append(conv_out)
            x = torch.cat(output, dim=0)  # Stack the results back into a batch
            x = self.dense_conv(x)
            # TF-Mamba block
            x = block(x)

        # Decode magnitude and phase
        denoised_mag = rearrange(self.mask_decoder(x) * noisy_mag, 'b c t f -> b f t c').squeeze(-1)
        denoised_pha = rearrange(self.phase_decoder(x), 'b c t f -> b f t c').squeeze(-1)

        # Combine denoised magnitude and phase into a complex representation
        denoised_com = torch.stack(
            (denoised_mag * torch.cos(denoised_pha), denoised_mag * torch.sin(denoised_pha)),
            dim=-1
        )

        return denoised_mag, denoised_pha, denoised_com


class SEMambaCo2dReAt(SEMamba):
    """
    SEMambaAt model for speech enhancement using Mamba blocks with attention between.

    This model uses a dense encoder, multiple Mamba blocks with attention between blocks, and separate magnitude
    and phase decoders to process noisy magnitude and phase inputs.
    """
    def __init__(self, cfg):
        """
        Initialize the SEMamba model.

        Args:
        - cfg: Configuration object containing model parameters.
        """
        super().__init__(cfg)
        self.reduce = nn.Sequential(
            nn.Conv2d(
                in_channels=cfg['model_cfg']['hid_feature'],
                out_channels=cfg['model_cfg']['attention_reduce_out_channels'],
                kernel_size=cfg['model_cfg']['attention_reduce_kernel_size'],
                padding=cfg['model_cfg']['attention_reduce_padding'],
                stride=cfg['model_cfg']['attention_reduce_stride'],
                groups=cfg['model_cfg']['attention_reduce_group_size']),
            nn.InstanceNorm2d(cfg['model_cfg']['attention_reduce_out_channels'], affine=True),
            nn.PReLU(cfg['model_cfg']['attention_reduce_out_channels'])
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=cfg['model_cfg']['attention_embed_dim'],
            num_heads=cfg['model_cfg']['attention_num_heads'], batch_first=True)

        self.expand = nn.Sequential(
            nn.Conv2d(
                in_channels=cfg['model_cfg']['attention_reduce_out_channels'],
                out_channels=cfg['model_cfg']['hid_feature'],
                kernel_size=cfg['model_cfg']['attention_expand_kernel_size'],
                padding=cfg['model_cfg']['attention_expand_padding'],
                stride=cfg['model_cfg']['attention_expand_stride'],
                groups=cfg['model_cfg']['attention_expand_group_size']),
            nn.InstanceNorm2d(cfg['model_cfg']['hid_feature'], affine=True),
            nn.PReLU(cfg['model_cfg']['hid_feature'])
        )

        if cfg['model_cfg']['attention_positional_enccoding_len'] != "None":
            self.register_buffer("positional_embedding",
                                 sinusoids(cfg['model_cfg']['attention_positional_enccoding_len'],
                                           cfg['model_cfg']['attention_embed_dim']))

    def forward(self, noisy_mag, noisy_pha):
        """
        Forward pass for the SEMambaAt model.

        Args:
        - noisy_mag (torch.Tensor): Noisy magnitude input tensor [B, F, T].
        - noisy_pha (torch.Tensor): Noisy phase input tensor [B, F, T].

        Returns:
        - denoised_mag (torch.Tensor): Denoised magnitude tensor [B, F, T].
        - denoised_pha (torch.Tensor): Denoised phase tensor [B, F, T].
        - denoised_com (torch.Tensor): Denoised complex tensor [B, F, T, 2].
        """
        # Reshape inputs
        noisy_mag = rearrange(noisy_mag, 'b f t -> b t f').unsqueeze(1)  # [B, 1, T, F]
        noisy_pha = rearrange(noisy_pha, 'b f t -> b t f').unsqueeze(1)  # [B, 1, T, F]

        # Concatenate magnitude and phase inputs
        x = torch.cat((noisy_mag, noisy_pha), dim=1)  # [B, 2, T, F]

        # Encode input
        x = self.dense_encoder(x)
        # Apply Mamba blocks
        for block_i, block in enumerate(self.TSMamba):
            if block_i == (self.num_tscblocks // 2):
                # Apply attention mechanism between blocks
                x_pre_att = self.reduce(x)  # [B, C, T, F] where C and F (optionaly) are reduced
                channels_num = x_pre_att.shape[1]
                x_pre_att = rearrange(x_pre_att, 'b c t f -> b t (f c)')  # Reshape for attention: [B, T, F*C]
                if hasattr(self, "positional_embedding"):
                    # Repeat the positional embedding if T is greater than its length
                    if x_pre_att.size(1) > self.positional_embedding.size(0):
                        repeat_factor = (x_pre_att.size(1) // self.positional_embedding.size(0)) + 1
                        extended_positional_embedding = self.positional_embedding.repeat(repeat_factor, 1)
                    else:
                        extended_positional_embedding = self.positional_embedding

                    # Slice the extended positional embedding to match the length of x_pre_att
                    x_pre_att = x_pre_att + extended_positional_embedding[:x_pre_att.size(1), :].unsqueeze(0)
                x_att, _ = self.attention(x_pre_att, x_pre_att, x_pre_att)
                x_att = rearrange(x_att, 'b t (f c) -> b c t f', c=channels_num)  # [B, C, T, F]
                x_att = self.expand(x_att)  # [B, C, T, F] where C and F (optionaly) are expanded

                x = x + x_att  # Residual connection
            x = block(x)

        # Decode magnitude and phase
        denoised_mag = rearrange(self.mask_decoder(x) * noisy_mag, 'b c t f -> b f t c').squeeze(-1)
        denoised_pha = rearrange(self.phase_decoder(x), 'b c t f -> b f t c').squeeze(-1)

        # Combine denoised magnitude and phase into a complex representation
        denoised_com = torch.stack(
            (denoised_mag * torch.cos(denoised_pha), denoised_mag * torch.sin(denoised_pha)),
            dim=-1
        )

        return denoised_mag, denoised_pha, denoised_com


class SEMambaReAt(SEMamba):
    """
    SEMambaAt model for speech enhancement using Mamba blocks with attention between.

    This model uses a dense encoder, multiple Mamba blocks with attention between blocks, and separate magnitude
    and phase decoders to process noisy magnitude and phase inputs.
    """
    def __init__(self, cfg):
        """
        Initialize the SEMamba model.

        Args:
        - cfg: Configuration object containing model parameters.
        """
        super().__init__(cfg)
        self.reduce = nn.Conv1d(in_channels=cfg['model_cfg']['attention_input_dim'],
                                out_channels=cfg['model_cfg']['attention_embed_dim'],
                                kernel_size=cfg['model_cfg']['attention_proj_kernel_size'],
                                padding=cfg['model_cfg']['attention_proj_kernel_size']//2,
                                stride=1,
                                bias=False,
                                groups=cfg['model_cfg']['attention_proj_group_size'])

        self.attention = nn.MultiheadAttention(
            embed_dim=cfg['model_cfg']['attention_embed_dim'],
            num_heads=cfg['model_cfg']['attention_num_heads'], batch_first=True)

        self.expand = nn.Conv1d(in_channels=cfg['model_cfg']['attention_embed_dim'],
                                out_channels=cfg['model_cfg']['attention_input_dim'],
                                kernel_size=cfg['model_cfg']['attention_proj_kernel_size'],
                                padding=cfg['model_cfg']['attention_proj_kernel_size']//2,
                                stride=1,
                                bias=False,
                                groups=cfg['model_cfg']['attention_proj_group_size'])

        if cfg['model_cfg']['attention_positional_enccoding_len'] != "None":
            self.register_buffer("positional_embedding",
                                 sinusoids(cfg['model_cfg']['attention_positional_enccoding_len'],
                                           cfg['model_cfg']['attention_embed_dim']))

    def forward(self, noisy_mag, noisy_pha):
        """
        Forward pass for the SEMambaAt model.

        Args:
        - noisy_mag (torch.Tensor): Noisy magnitude input tensor [B, F, T].
        - noisy_pha (torch.Tensor): Noisy phase input tensor [B, F, T].

        Returns:
        - denoised_mag (torch.Tensor): Denoised magnitude tensor [B, F, T].
        - denoised_pha (torch.Tensor): Denoised phase tensor [B, F, T].
        - denoised_com (torch.Tensor): Denoised complex tensor [B, F, T, 2].
        """
        # Reshape inputs
        noisy_mag = rearrange(noisy_mag, 'b f t -> b t f').unsqueeze(1)  # [B, 1, T, F]
        noisy_pha = rearrange(noisy_pha, 'b f t -> b t f').unsqueeze(1)  # [B, 1, T, F]

        # Concatenate magnitude and phase inputs
        x = torch.cat((noisy_mag, noisy_pha), dim=1)  # [B, 2, T, F]

        # Encode input
        x = self.dense_encoder(x)
        # Apply Mamba blocks
        for block_i, block in enumerate(self.TSMamba):
            if block_i == (self.num_tscblocks // 2):
                channels_num = x.shape[1]
                # Apply attention mechanism between blocks
                x_pre_att = rearrange(x, 'b c t f -> b t (f c)')  # Reshape for attention: [B, T, F*C]
                x_pre_att = self.reduce(x_pre_att.transpose(1, 2)).transpose(1, 2)  # [B, T, reduced_dim]

                if hasattr(self, "positional_embedding"):
                    # Repeat the positional embedding if T is greater than its length
                    if x_pre_att.size(1) > self.positional_embedding.size(0):
                        repeat_factor = (x_pre_att.size(1) // self.positional_embedding.size(0)) + 1
                        extended_positional_embedding = self.positional_embedding.repeat(repeat_factor, 1)
                    else:
                        extended_positional_embedding = self.positional_embedding

                    # Slice the extended positional embedding to match the length of x_pre_att
                    x_pre_att = x_pre_att + extended_positional_embedding[:x_pre_att.size(1), :].unsqueeze(0)
                x_att, _ = self.attention(x_pre_att, x_pre_att, x_pre_att)
                x_att = self.expand(x_att.transpose(1, 2)).transpose(1, 2)  # [B, T, F*C]
                x_att = rearrange(x_att, 'b t (f c) -> b c t f', c=channels_num)  # [B, C, T, F]

                x = x + x_att  # Residual connection
            x = block(x)

        # Decode magnitude and phase
        denoised_mag = rearrange(self.mask_decoder(x) * noisy_mag, 'b c t f -> b f t c').squeeze(-1)
        denoised_pha = rearrange(self.phase_decoder(x), 'b c t f -> b f t c').squeeze(-1)

        # Combine denoised magnitude and phase into a complex representation
        denoised_com = torch.stack(
            (denoised_mag * torch.cos(denoised_pha), denoised_mag * torch.sin(denoised_pha)),
            dim=-1
        )

        return denoised_mag, denoised_pha, denoised_com


class SEMambaAt(SEMamba):
    """
    SEMambaAt model for speech enhancement using Mamba blocks with attention between.

    This model uses a dense encoder, multiple Mamba blocks with attention between blocks, and separate magnitude
    and phase decoders to process noisy magnitude and phase inputs.
    """
    def __init__(self, cfg):
        """
        Initialize the SEMamba model.

        Args:
        - cfg: Configuration object containing model parameters.
        """
        super().__init__(cfg)
        self.attention = nn.MultiheadAttention(
            embed_dim=cfg['model_cfg']['attention_embed_dim'],
            num_heads=cfg['model_cfg']['attention_num_heads'])

        if cfg['model_cfg']['attention_positional_enccoding_len'] != "None":
            self.register_buffer("positional_embedding",
                                 sinusoids(cfg['model_cfg']['attention_positional_enccoding_len'],
                                           cfg['model_cfg']['attention_embed_dim']))

    def forward(self, noisy_mag, noisy_pha):
        """
        Forward pass for the SEMambaAt model.

        Args:
        - noisy_mag (torch.Tensor): Noisy magnitude input tensor [B, F, T].
        - noisy_pha (torch.Tensor): Noisy phase input tensor [B, F, T].

        Returns:
        - denoised_mag (torch.Tensor): Denoised magnitude tensor [B, F, T].
        - denoised_pha (torch.Tensor): Denoised phase tensor [B, F, T].
        - denoised_com (torch.Tensor): Denoised complex tensor [B, F, T, 2].
        """
        # Reshape inputs
        noisy_mag = rearrange(noisy_mag, 'b f t -> b t f').unsqueeze(1)  # [B, 1, T, F]
        noisy_pha = rearrange(noisy_pha, 'b f t -> b t f').unsqueeze(1)  # [B, 1, T, F]

        # Concatenate magnitude and phase inputs
        x = torch.cat((noisy_mag, noisy_pha), dim=1)  # [B, 2, T, F]

        # Encode input
        x = self.dense_encoder(x)

        # Apply Mamba blocks
        for block_i, block in enumerate(self.TSMamba):
            torch.cuda.empty_cache()
            if block_i == (self.num_tscblocks // 2):
                channels_num = x.shape[1]
                # Apply attention mechanism between blocks
                x_pre_att = rearrange(x, 'b c t f -> (b c) t f')  # Reshape for attention: [B*C, T, F]
                if hasattr(self, "positional_embedding"):
                    # Repeat the positional embedding if T is greater than its length
                    if x_pre_att.size(1) > self.positional_embedding.size(0):
                        repeat_factor = (x_pre_att.size(1) // self.positional_embedding.size(0)) + 1
                        extended_positional_embedding = self.positional_embedding.repeat(repeat_factor, 1)
                    else:
                        extended_positional_embedding = self.positional_embedding

                    # Slice the extended positional embedding to match the length of x_pre_att
                    x_pre_att = x_pre_att + extended_positional_embedding[:x_pre_att.size(1), :].unsqueeze(0)
                x_pre_att = x_pre_att.permute(1, 0, 2)  # Reshape for attention: [T, B*C, F]
                x_att, _ = self.attention(x_pre_att, x_pre_att, x_pre_att)
                x_att = rearrange(x_att, 't (b c) f -> b c t f', c=channels_num)
                x = x + x_att  # Residual connection
            x = block(x)

        # Decode magnitude and phase
        denoised_mag = rearrange(self.mask_decoder(x) * noisy_mag, 'b c t f -> b f t c').squeeze(-1)
        denoised_pha = rearrange(self.phase_decoder(x), 'b c t f -> b f t c').squeeze(-1)

        # Combine denoised magnitude and phase into a complex representation
        denoised_com = torch.stack(
            (denoised_mag * torch.cos(denoised_pha), denoised_mag * torch.sin(denoised_pha)),
            dim=-1
        )

        return denoised_mag, denoised_pha, denoised_com


class SEAtMamba(nn.Module):
    """
    SEAtMamba model for speech enhancement using attention and Mamba blocks.

    This model uses a dense encoder, attention head, multiple Mamba blocks, and separate magnitude
    and phase decoders to process noisy magnitude and phase inputs.
    """
    def __init__(self, cfg):
        """
        Initialize the SEMamba model.

        Args:
        - cfg: Configuration object containing model parameters.
        """
        if cfg['model_cfg']['mamba_version'] == 1:
            from .mamba_block import TFMambaBlock
        elif cfg['model_cfg']['mamba_version'] == 2:
            from .mamba2_block import TFMambaBlock
        else:
            raise ValueError("Invalid value for Mamba version")
        super().__init__()
        self.cfg = cfg
        self.num_tscblocks = \
            cfg['model_cfg']['num_tfmamba'] if cfg['model_cfg']['num_tfmamba'] is not None else 4  # default tfmamba: 4

        # Initialize dense encoder
        self.dense_encoder = DenseEncoder(cfg)

        # Initialize attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=cfg['model_cfg']['attention_embed_dim'],
            num_heads=cfg['model_cfg']['attention_num_heads'])

        # Initialize Mamba blocks
        self.TSMamba = nn.ModuleList([TFMambaBlock(cfg) for _ in range(self.num_tscblocks)])

        # Initialize decoders
        self.mask_decoder = MagDecoder(cfg)
        self.phase_decoder = PhaseDecoder(cfg)

    def forward(self, noisy_mag, noisy_pha):
        """
        Forward pass for the SEMamba model.

        Args:
        - noisy_mag (torch.Tensor): Noisy magnitude input tensor [B, F, T].
        - noisy_pha (torch.Tensor): Noisy phase input tensor [B, F, T].

        Returns:
        - denoised_mag (torch.Tensor): Denoised magnitude tensor [B, F, T].
        - denoised_pha (torch.Tensor): Denoised phase tensor [B, F, T].
        - denoised_com (torch.Tensor): Denoised complex tensor [B, F, T, 2].
        """
        # Reshape inputs
        noisy_mag = rearrange(noisy_mag, 'b f t -> b t f').unsqueeze(1)  # [B, 1, T, F]
        noisy_pha = rearrange(noisy_pha, 'b f t -> b t f').unsqueeze(1)  # [B, 1, T, F]

        # Concatenate magnitude and phase inputs
        x = torch.cat((noisy_mag, noisy_pha), dim=1)  # [B, 2, T, F]
        # Encode input
        x = self.dense_encoder(x)
        channels_num = x.shape[1]
        # Apply attention mechanism
        x = rearrange(x, 'b c t f -> t b (f c)')  # Reshape for attention: [T, B, F*C]
        # Note that the attention mechanism is applied to the time dimension, batch_first=False
        x, _ = self.attention(x, x, x)
        x = rearrange(x, 't b (f c) -> b c t f', c=channels_num)  # Reshape back: [B, C, T, F]
        # Apply Mamba blocks
        for block in self.TSMamba:
            x = block(x)

        # Decode magnitude and phase
        denoised_mag = rearrange(self.mask_decoder(x) * noisy_mag, 'b c t f -> b f t c').squeeze(-1)
        denoised_pha = rearrange(self.phase_decoder(x), 'b c t f -> b f t c').squeeze(-1)

        # Combine denoised magnitude and phase into a complex representation
        denoised_com = torch.stack(
            (denoised_mag * torch.cos(denoised_pha), denoised_mag * torch.sin(denoised_pha)),
            dim=-1
        )

        return denoised_mag, denoised_pha, denoised_com


class HyperSEMamba(nn.Module):
    """
    HyperSEMamba model for speech enhancement using Mamba blocks and hypernetworks.

    This model uses a dense encoder that uses hyper netework, multiple Mamba blocks, and separate magnitude
    and phase decoders to process noisy magnitude and phase inputs.
    """
    def __init__(self, cfg):
        """
        Initialize the SEMamba model.

        Args:
        - cfg: Configuration object containing model parameters.
        """
        if cfg['model_cfg']['mamba_version'] == 1:
            from .mamba_block import TFMambaBlock
        elif cfg['model_cfg']['mamba_version'] == 2:
            from .mamba2_block import TFMambaBlock
        else:
            raise ValueError("Invalid value for Mamba version")
        super(HyperSEMamba, self).__init__()
        self.cfg = cfg
        self.num_tscblocks = \
            cfg['model_cfg']['num_tfmamba'] if cfg['model_cfg']['num_tfmamba'] is not None else 4  # default tfmamba: 4

        # Initialize dense encoder
        self.dense_encoder = HyperDenseEncoder(cfg)

        # Initialize Mamba blocks
        self.TSMamba = nn.ModuleList([TFMambaBlock(cfg) for _ in range(self.num_tscblocks)])

        # Initialize decoders
        self.mask_decoder = MagDecoder(cfg)
        self.phase_decoder = PhaseDecoder(cfg)

    def forward(self, noisy_mag, noisy_pha):
        """
        Forward pass for the SEMamba model.

        Args:
        - noisy_mag (torch.Tensor): Noisy magnitude input tensor [B, F, T].
        - noisy_pha (torch.Tensor): Noisy phase input tensor [B, F, T].

        Returns:
        - denoised_mag (torch.Tensor): Denoised magnitude tensor [B, F, T].
        - denoised_pha (torch.Tensor): Denoised phase tensor [B, F, T].
        - denoised_com (torch.Tensor): Denoised complex tensor [B, F, T, 2].
        """
        # Reshape inputs
        noisy_mag = rearrange(noisy_mag, 'b f t -> b t f').unsqueeze(1)  # [B, 1, T, F]
        noisy_pha = rearrange(noisy_pha, 'b f t -> b t f').unsqueeze(1)  # [B, 1, T, F]

        # Concatenate magnitude and phase inputs
        x = torch.cat((noisy_mag, noisy_pha), dim=1)  # [B, 2, T, F]

        # Encode input
        x = self.dense_encoder(x)

        # Apply Mamba blocks
        for block in self.TSMamba:
            x = block(x)

        # Decode magnitude and phase
        denoised_mag = rearrange(self.mask_decoder(x) * noisy_mag, 'b c t f -> b f t c').squeeze(-1)
        denoised_pha = rearrange(self.phase_decoder(x), 'b c t f -> b f t c').squeeze(-1)

        # Combine denoised magnitude and phase into a complex representation
        denoised_com = torch.stack(
            (denoised_mag * torch.cos(denoised_pha), denoised_mag * torch.sin(denoised_pha)),
            dim=-1
        )

        return denoised_mag, denoised_pha, denoised_com
