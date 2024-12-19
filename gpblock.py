class GPBlock(nn.Module):
    def __init__(self, embed_dims, depth, num_group_heads, num_ungroup_heads,
                 num_group_token, ffn_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., with_cp=False, group_att_cfg=dict(), fwd_att_cfg=dict(),
                 ungroup_att_cfg=dict(), **kwargs):
        super().__init__()

        self.embed_dims = embed_dims
        self.num_group_token = num_group_token
        self.with_cp = with_cp

        self.group_token = nn.Parameter(torch.zeros(1, num_group_token, embed_dims))
        trunc_normal_(self.group_token, std=.02)

        _group_att_cfg = dict(
            embed_dims=embed_dims,
            num_heads=num_group_heads,
            ffn_ratio=ffn_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=0.,
            key_is_query=False,
            value_is_key=True,
            with_cp=with_cp)
        _group_att_cfg.update(group_att_cfg)
        self.group_layer = LightGroupAttnBlock(**_group_att_cfg)

        # MLP Mixer for propagation
        _mixer_cfg = dict(
            num_patches=num_group_token,
            embed_dims=embed_dims,
            patch_expansion=0.5,
            channel_expansion=4.0,
            depth=depth,
            drop_path=drop_path)
        _mixer_cfg.update(fwd_att_cfg)
        self.mixer = MLPMixer(**_mixer_cfg)

        _ungroup_att_cfg = dict(
            embed_dims=embed_dims,
            num_heads=num_ungroup_heads,
            ffn_ratio=ffn_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            key_is_query=False,
            value_is_key=True,
            with_cp=with_cp)
        _ungroup_att_cfg.update(ungroup_att_cfg)
        self.un_group_layer = FullAttnCatBlock(**_ungroup_att_cfg)

        self.dwconv = nn.Sequential(
            nn.Conv1d(embed_dims, embed_dims, kernel_size=3, padding=1, bias=False, groups=embed_dims),
            nn.BatchNorm1d(embed_dims),
            nn.ReLU(True))

    def forward(self, x, sequence_length):
        """
        Args:
            x: input tokens, shape [B, L, C]
            sequence_length: length of the sequence (1D)
        Returns:
            proj_tokens: output tokens, shape [B, L, C]
        """
        B, L, C = x.size()
        group_token = self.group_token.expand(B, -1, -1)

        gt = self.group_layer(query=group_token, key=x, value=x)

        gt = self.mixer(gt)

        ungroup_tokens = self.un_group_layer(query=x, key=gt, value=gt)

        ungroup_tokens = ungroup_tokens.permute(0, 2, 1)  # [B, C, L]
        proj_tokens = self.dwconv(ungroup_tokens).permute(0, 2, 1)  # [B, L, C]

        return proj_tokens
