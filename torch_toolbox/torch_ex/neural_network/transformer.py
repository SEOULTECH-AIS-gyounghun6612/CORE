import torch
from torch.nn import (
    Module, Sequential,
    MultiheadAttention, Conv1d, LayerNorm, Dropout, Linear, ReLU
)


class Position_Embeder():
    class Embeder_Module(Module):
        def __init__(
            self, max_count: int, chnnels: int,
            term: int = 10000, group_ct: int = 2
        ):
            super().__init__()
            self.requires_grad = False

            self.max_count = max_count
            self.depth_size = chnnels

            self.encoding_array = torch.zeros(max_count, chnnels)

            self.group_ct = group_ct
            _depth_term = term ** (
                group_ct * (
                    torch.arange(0, chnnels) // group_ct
                ) / chnnels)
            _pos = torch.arange(0, max_count).unsqueeze(dim=1)
            self.pos = _pos / _depth_term

        def forward(self, x: torch.Tensor):
            _b, _token_count, _token_depth = x.shape
            _pos_emb = self.pos[:_token_count].expand(
                _b, _token_count, _token_depth)
            return _pos_emb + x

    class Trigonometric(Embeder_Module):
        def __init__(
            self, max_count: int, chnnels: int, term: int = 10000
        ):
            super().__init__(max_count, chnnels, term)
            _pos = self.pos
            _pos = torch.stack(
                [_pos[:, 0::2].sin(), _pos[:, 1::2].cos()],
                dim=-1
            ).flatten(1)
            self.pos = _pos

    class Gaussian(Embeder_Module):
        def __init__(self, max_count: int, chnnels, term: int = 10000):
            super().__init__(max_count, chnnels, term)
            raise NotImplementedError

    class Trainable(Embeder_Module):
        def __init__(self, max_count: int, chnnels, term: int = 10000):
            super().__init__(max_count, chnnels, term)
            raise NotImplementedError


class Attention():
    class Muiltihead(Module):
        def __init__(
            self,
            token_count: int,
            chnnels: int,
            num_heads: int,
            drop_rate: float = 0.5
        ):
            super().__init__()
            self.attention = MultiheadAttention(
                chnnels, num_heads, drop_rate)

            self.kv_attention = Conv1d(
                token_count, 2 * token_count, 1, groups=token_count)
            self.q_attention = Conv1d(
                token_count, 1 * token_count, 1, groups=token_count)

        def _Make_QKV(
            self,
            q_source: torch.Tensor,
            kv_source: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            _, _c, _ = kv_source.shape
            _kv = self.kv_attention(kv_source)
            _q = self.q_attention(q_source)
            return _q, _kv[:, :_c], _kv[:, _c:]

        def forward(
            self,
            q_source: torch.Tensor,
            kv_source: torch.Tensor,
            mask: torch.Tensor | None = None,
            key_padding_mask: torch.Tensor | None = None
        ):
            _q, _k, _v = self._Make_QKV(q_source, kv_source)
            _x, _map = self.attention(
                query=_q,
                key=_k,
                value=_v,
                attn_mask=mask,
                key_padding_mask=key_padding_mask
            )

            return _x, _map

    class Self_Muiltihead(Module):
        def __init__(
            self,
            token_count: int,
            chnnels: int,
            num_heads: int,
            drop_rate: float = 0.5
        ):
            super().__init__()
            self.attention = MultiheadAttention(
                chnnels, num_heads, drop_rate)

            self.qkv_attention = Conv1d(
                token_count, 3 * token_count, 1, groups=token_count)

        def _Make_QKV(
            self,
            qkv_source: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            _, _c, _ = qkv_source.shape
            _qkv = self.qkv_attention(qkv_source)
            return _qkv[:, :_c], _qkv[:, _c: 2 * _c], _qkv[:, 2 * _c: 3 * _c]

        def forward(
            self,
            qkv_source: torch.Tensor,
            mask: torch.Tensor | None = None,
            key_padding_mask: torch.Tensor | None = None
        ):
            _q, _k, _v = self._Make_QKV(qkv_source)
            _x, _map = self.attention(
                query=_q,
                key=_k,
                value=_v,
                attn_mask=mask,
                key_padding_mask=key_padding_mask
            )
            return _x, _map


class Transformer():
    class Encoder(Module):
        def __init__(
            self,
            token_count: int,
            chnnels: int,
            num_heads: int,
            feadforward_dim: int,
            drop_rate: float = 0.5,
            normalize_before: bool = False
        ):
            super().__init__()
            self._normalize_before = normalize_before

            self.front_norm = LayerNorm(chnnels)
            self.attention = Attention.Self_Muiltihead(
                token_count, chnnels, num_heads, drop_rate)
            self.back_norm = LayerNorm(token_count)

            # self.linear_block = Sequential(
            #     Conv1d(token_count, token_count, 1, groups=token_count),
            #     ReLU(),
            #     Dropout(drop_rate),
            #     Conv1d(token_count, token_count, 1, groups=token_count),
            #     ReLU(),
            #     Dropout(drop_rate)
            # )
            self.linear_block = Sequential(
                Linear(chnnels, feadforward_dim),
                ReLU(),
                Dropout(drop_rate),
                Linear(feadforward_dim, chnnels),
                ReLU(),
                Dropout(drop_rate)
            )

        def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor | None = None,
            key_padding_mask: torch.Tensor | None = None,
        ):
            if self._normalize_before:
                # attention
                _x = self.front_norm(x)
                _attention_out, _map = self.attention(
                    _x, mask, key_padding_mask)

                # fc
                _x = self.back_norm(_attention_out)
                _x = _attention_out + self.linear_block(_x)
            else:
                # attention
                _attention_out, _map = self.attention(
                    x, mask, key_padding_mask)
                _attention_out = self.front_norm(_attention_out)

                # fc
                _x = _attention_out + self.linear_block(_attention_out)
                _x = self.back_norm(_x)
            return _x, _map

    class Decoder(Module):
        def __init__(
            self,
            token_count: int,
            chnnels: int,
            num_heads: int,
            feadforward_dim: int,
            drop_rate: float = 0.5,
            normalize_before: bool = False
        ):
            super().__init__()
            self._normalize_before = normalize_before

            self._front_norm = LayerNorm(token_count)
            self._self_attention = Attention.Self_Muiltihead(
                token_count, chnnels, num_heads, drop_rate)
            self._mid_norm = LayerNorm(token_count)
            self._multi_attention = Attention.Muiltihead(
                token_count, chnnels, num_heads, drop_rate)
            self._back_norm = LayerNorm(token_count)

            # self.linear_block = Sequential(
            #     Conv1d(token_count, token_count, 1, groups=token_count),
            #     ReLU(),
            #     Dropout(drop_rate),
            #     Conv1d(token_count, token_count, 1, groups=token_count),
            #     ReLU(),
            #     Dropout(drop_rate)
            # )
            self.linear_block = Sequential(
                Linear(chnnels, feadforward_dim),
                ReLU(),
                Dropout(drop_rate),
                Linear(feadforward_dim, chnnels),
                ReLU(),
                Dropout(drop_rate)
            )

        def forward(
                self,
                q_source: torch.Tensor,
                kv_source: torch.Tensor,
                q_mask: torch.Tensor | None = None,
                kv_mask: torch.Tensor | None = None,
                q_key_padding_mask: torch.Tensor | None = None,
                kv_key_padding_mask: torch.Tensor | None = None,
        ):
            if self._normalize_before:
                # attention
                _x = self._front_norm(q_source)
                _attention_out, _self_attention_map = self._self_attention(
                    _x, q_mask, q_key_padding_mask)
                _x = self._mid_norm(_attention_out)
                _attention_out, _multi_attention_map = self._multi_attention(
                    _x, kv_source, kv_mask, kv_key_padding_mask)

                # fc
                _x = self._back_norm(_attention_out)
                _x = _attention_out + self.linear_block(_x)
            else:
                # attention
                _attention_out, _self_attention_map = self._self_attention(
                    q_source, q_mask, q_key_padding_mask)
                _x = self._front_norm(_attention_out)
                _attention_out, _multi_attention_map = self._multi_attention(
                    _x, kv_source, kv_mask, kv_key_padding_mask)
                _x = self._mid_norm(_attention_out)

                # fc
                _x = _attention_out + self.linear_block(_x)
                _x = self._back_norm(_x)
            return _x, _self_attention_map, _multi_attention_map
