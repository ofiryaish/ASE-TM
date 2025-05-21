import torch
import torch.nn as nn

from mamba_ssm.modules.mamba2 import Mamba2
from einops import rearrange

from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined


class HyperMamba2(Mamba2):
    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=64,
        d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        process_group=None,
        sequence_parallel=True,
        device=None,
        dtype=None,
    ):
        super().__init__(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            conv_init=conv_init,
            expand=expand,
            headdim=headdim,
            d_ssm=d_ssm,
            ngroups=ngroups,
            A_init_range=A_init_range,
            D_has_hdim=D_has_hdim,
            rmsnorm=rmsnorm,
            norm_before_gate=norm_before_gate,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init_floor=dt_init_floor,
            dt_limit=dt_limit,
            bias=bias,
            conv_bias=conv_bias,
            chunk_size=chunk_size,
            use_mem_eff_path=use_mem_eff_path,
            layer_idx=layer_idx,
            process_group=process_group,
            sequence_parallel=sequence_parallel,
            device=device,
            dtype=dtype,
        )
        # Hypernet
        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.conv1d_weight_dim = (conv_dim, 1, self.d_conv)
        self.hyper_net = nn.Sequential(
            nn.Linear(self.d_in_proj, self.d_in_proj * 2),
            nn.ReLU(),
            nn.Linear(self.d_in_proj * 2,
                      self.conv1d_weight_dim[0] * self.conv1d_weight_dim[1] * self.conv1d_weight_dim[2]),
            nn.Unflatten(1, self.conv1d_weight_dim),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None):
        """
        u: (batch, seqlen, hidden_dim) if seqlen=None.
            If seqlen is not None, u is (batch * seqlen, hidden_dim). This is so that when we
            split u during sequence parallel, we split the batch * seqlen dimension
            (in case batch is small).
        Returns: same shape as u
        """
        # In this version: seqlen = None, inference_params = None,
        # self.use_mem_eff_path = True, self.process_group = None
        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)

        # Generate weights for the convolution
        batch_size = zxbcdt.size(0)
        pooled_zxbcdt = self.adaptive_pool(zxbcdt.transpose(1, 2))  # [B, d_in_proj, 1]
        pooled_zxbcdt = pooled_zxbcdt.view(batch_size, -1)  # [B, d_in_proj]
        generated_weights = self.hyper_net(pooled_zxbcdt)  # [B, conv_dim, 1, self.d_conv]
        generated_weights = generated_weights.mean(dim=0)  # [conv_dim, 1, self.d_conv]  # TODO: this is not ideal

        # TODO: add here conv before mamba_split_conv1d_scan_combined

        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        
        out = mamba_split_conv1d_scan_combined(
            zxbcdt,
            rearrange(generated_weights, "d 1 w -> d w"),
            self.conv1d.bias,
            self.dt_bias,
            A,
            D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
            chunk_size=self.chunk_size,
            seq_idx=seq_idx,
            activation=self.activation,
            rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
            rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
            outproj_weight=self.out_proj.weight,
            outproj_bias=self.out_proj.bias,
            headdim=None if self.D_has_hdim else self.headdim,
            ngroups=self.ngroups,
            norm_before_gate=self.norm_before_gate,
            **dt_limit_kwargs,
        )

        return out



class HyperMamba2(Mamba2):
    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=64,
        d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        process_group=None,
        sequence_parallel=True,
        device=None,
        dtype=None,
    ):
        super().__init__(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            conv_init=conv_init,
            expand=expand,
            headdim=headdim,
            d_ssm=d_ssm,
            ngroups=ngroups,
            A_init_range=A_init_range,
            D_has_hdim=D_has_hdim,
            rmsnorm=rmsnorm,
            norm_before_gate=norm_before_gate,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init_floor=dt_init_floor,
            dt_limit=dt_limit,
            bias=bias,
            conv_bias=conv_bias,
            chunk_size=chunk_size,
            use_mem_eff_path=use_mem_eff_path,
            layer_idx=layer_idx,
            process_group=process_group,
            sequence_parallel=sequence_parallel,
            device=device,
            dtype=dtype,
        )
        # Hypernet
        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.conv1d_weight_dim = (conv_dim, 1, self.d_conv)
        self.hyper_net = nn.Sequential(
            nn.Linear(self.d_in_proj, self.d_in_proj * 2),
            nn.ReLU(),
            nn.Linear(self.d_in_proj * 2,
                      self.conv1d_weight_dim[0] * self.conv1d_weight_dim[1] * self.conv1d_weight_dim[2]),
            nn.Unflatten(1, self.conv1d_weight_dim),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None):
        """
        u: (batch, seqlen, hidden_dim) if seqlen=None.
            If seqlen is not None, u is (batch * seqlen, hidden_dim). This is so that when we
            split u during sequence parallel, we split the batch * seqlen dimension
            (in case batch is small).
        Returns: same shape as u
        """
        # In this version: seqlen = None, inference_params = None,
        # self.use_mem_eff_path = True, self.process_group = None
        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)

        # Generate weights for the convolution
        batch_size = zxbcdt.size(0)
        pooled_zxbcdt = self.adaptive_pool(zxbcdt.transpose(1, 2))  # [B, d_in_proj, 1]
        pooled_zxbcdt = pooled_zxbcdt.view(batch_size, -1)  # [B, d_in_proj]
        generated_weights = self.hyper_net(pooled_zxbcdt)  # [B, conv_dim, 1, self.d_conv]
        generated_weights = generated_weights.mean(dim=0)  # [conv_dim, 1, self.d_conv]  # TODO: this is not ideal

        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        out = mamba_split_conv1d_scan_combined(
            zxbcdt,
            rearrange(generated_weights, "d 1 w -> d w"),
            self.conv1d.bias,
            self.dt_bias,
            A,
            D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
            chunk_size=self.chunk_size,
            seq_idx=seq_idx,
            activation=self.activation,
            rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
            rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
            outproj_weight=self.out_proj.weight,
            outproj_bias=self.out_proj.bias,
            headdim=None if self.D_has_hdim else self.headdim,
            ngroups=self.ngroups,
            norm_before_gate=self.norm_before_gate,
            **dt_limit_kwargs,
        )

        return out