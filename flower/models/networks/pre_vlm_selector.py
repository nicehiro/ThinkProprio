import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import RmsNorm

logger = logging.getLogger(__name__)


class ProprioTextTokenizer(nn.Module):
    def __init__(
        self,
        vlm_embeddings: nn.Module,
        vocab_size: int,
        num_bins: int = 256,
        min_value: float = -3.0,
        max_value: float = 3.0,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.min_value = min_value
        self.max_value = max_value
        self.vlm_embeddings = vlm_embeddings
        self.vocab_size = vocab_size

    def discretize(self, values: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(values, self.min_value, self.max_value)
        normalized = (clipped - self.min_value) / (self.max_value - self.min_value)
        return (normalized * (self.num_bins - 1)).long()

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        B, T, D = proprio.shape
        token_ids = self.vocab_size - 1 - self.discretize(proprio)
        embeddings = self.vlm_embeddings(token_ids)
        return embeddings.view(B, T * D, embeddings.shape[-1])


class PreVLMVisionSelector(nn.Module):
    def __init__(
        self,
        vision_dim: int = 1024,
        use_residual: bool = True,
        gumbel_noise_start: float = 1.0,
        gumbel_noise_end: float = 0.01,
        div_cos_rho: float = 0.75,
        div_temperature: float = 1.0,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.gumbel_noise_start = gumbel_noise_start
        self.gumbel_noise_end = gumbel_noise_end
        self.div_cos_rho = div_cos_rho
        self.div_temperature = div_temperature
        self.register_buffer('_training_progress', torch.tensor(0.0))

        if use_residual:
            self.global_proj = nn.Linear(vision_dim, vision_dim, bias=False)

        self.vision_norm = RmsNorm(vision_dim, eps=1e-6)
        self.query_norm = RmsNorm(vision_dim, eps=1e-6)
        self.query_out_norm = RmsNorm(vision_dim, eps=1e-6)
        self._logged_compression = False
        self._step_counter = 0
        self._diagnostics = {}
        self._last_num_selected = 0
        self._last_compression_ratio = 1.0
        self._last_div_cos = None
        self._last_div_loss = None

    def set_training_progress(self, progress: float):
        self._training_progress.fill_(min(1.0, max(0.0, progress)))

    @property
    def current_noise_scale(self) -> float:
        progress = self._training_progress.item()
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.gumbel_noise_end + (self.gumbel_noise_start - self.gumbel_noise_end) * cosine_decay

    def _generate_per_vision_queries(
        self,
        query_embeds: torch.Tensor,
        vision_normed: torch.Tensor,
        query_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        D = vision_normed.shape[-1]
        query_normed = self.query_norm(query_embeds)
        attn_logits = torch.einsum('bvd,bpd->bvp', vision_normed, query_normed) / (D ** 0.5)

        if query_attention_mask is not None:
            mask = query_attention_mask.to(device=attn_logits.device, dtype=torch.bool).unsqueeze(1)
            attn_logits = attn_logits.masked_fill(~mask, torch.finfo(attn_logits.dtype).min)

        attn_weights = F.softmax(attn_logits, dim=-1)
        queries = torch.einsum('bvp,bpd->bvd', attn_weights, query_normed)
        return self.query_out_norm(queries)

    def _soft_selection_indicator(self, soft_scores: torch.Tensor) -> torch.Tensor:
        dtype = soft_scores.dtype
        soft_scores = soft_scores.float().clamp(0.0, 1.0 - 1e-6)
        indicator = 1.0 - torch.prod(1.0 - soft_scores, dim=1)
        return indicator.to(dtype=dtype)

    def _probabilistic_or(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return 1.0 - (1.0 - first) * (1.0 - second)

    def _selection_mask_from_indices(self, hard_indices: torch.Tensor, num_tokens: int) -> torch.Tensor:
        B = hard_indices.shape[0]
        counts = torch.zeros(B, num_tokens, device=hard_indices.device, dtype=torch.int32)
        counts.scatter_add_(1, hard_indices, torch.ones_like(hard_indices, dtype=torch.int32))
        return counts > 0

    def _gumbel(self, scores: torch.Tensor) -> torch.Tensor:
        uniform = torch.rand_like(scores).clamp(1e-10, 1 - 1e-10)
        return -torch.log(-torch.log(uniform))

    def forward(
        self,
        vision_tokens: torch.Tensor,
        proprio_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        query_attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, V, D = vision_tokens.shape
        device = vision_tokens.device

        text_attention_mask = None
        proprio_attention_mask = None
        if query_attention_mask is not None:
            num_text = text_embeds.shape[1]
            text_attention_mask = query_attention_mask[:, :num_text]
            proprio_attention_mask = query_attention_mask[:, num_text:]
            if proprio_attention_mask.shape[1] == 0:
                proprio_attention_mask = None

        vision_normed = self.vision_norm(vision_tokens)
        text_queries = self._generate_per_vision_queries(text_embeds, vision_normed, text_attention_mask)
        proprio_queries = self._generate_per_vision_queries(proprio_embeds, vision_normed, proprio_attention_mask)

        text_scores = torch.einsum('bqd,bkd->bqk', text_queries, vision_normed) / (D ** 0.5)
        proprio_scores = torch.einsum('bqd,bkd->bqk', proprio_queries, vision_normed) / (D ** 0.5)

        text_div_probs = F.softmax(text_scores.sum(dim=1) / self.div_temperature, dim=-1)
        proprio_div_probs = F.softmax(proprio_scores.sum(dim=1) / self.div_temperature, dim=-1)
        self._last_div_cos = F.cosine_similarity(text_div_probs, proprio_div_probs, dim=-1).mean()
        self._last_div_loss = F.relu(self._last_div_cos - self.div_cos_rho).pow(2)

        if self.training:
            noise_scale = self.current_noise_scale
            text_scores = text_scores + self._gumbel(text_scores) * noise_scale
            proprio_scores = proprio_scores + self._gumbel(proprio_scores) * noise_scale

        text_mask = self._selection_mask_from_indices(text_scores.argmax(dim=-1), V)
        proprio_mask = self._selection_mask_from_indices(proprio_scores.argmax(dim=-1), V)
        hard_mask = text_mask | proprio_mask

        if self.training:
            text_indicator = self._soft_selection_indicator(F.softmax(text_scores, dim=-1))
            proprio_indicator = self._soft_selection_indicator(F.softmax(proprio_scores, dim=-1))
            soft_union = self._probabilistic_or(text_indicator, proprio_indicator)
            indicator = hard_mask.float() + soft_union - soft_union.detach()
            selected_tokens, selected_attention_mask = self._gather_selected(vision_tokens, hard_mask, indicator)
        else:
            selected_tokens, selected_attention_mask = self._gather_selected(vision_tokens, hard_mask)

        num_selected = int(hard_mask[0].sum().item())
        self._last_num_selected = num_selected
        self._last_compression_ratio = num_selected / V

        if self.training:
            self._step_counter += 1
            if self._step_counter % 100 == 0:
                with torch.no_grad():
                    overlap = (text_mask & proprio_mask).sum(dim=1).float().mean().item()
                    union_selected = hard_mask.sum(dim=1).float().mean().item()
                    self._diagnostics = {
                        "pre_vlm/text_num_selected": text_mask.sum(dim=1).float().mean().item(),
                        "pre_vlm/proprio_num_selected": proprio_mask.sum(dim=1).float().mean().item(),
                        "pre_vlm/num_selected": union_selected,
                        "pre_vlm/overlap_ratio": overlap / max(union_selected, 1.0),
                        "pre_vlm/selection_ratio": union_selected / V,
                        "pre_vlm/noise_scale": self.current_noise_scale,
                    }

        if not self._logged_compression:
            logger.info(
                f"[PreVLMVisionSelector] Vision: {V} -> Selected: {num_selected} "
                f"(compression: {V / max(num_selected, 1):.1f}x)"
            )
            self._logged_compression = True

        if self.use_residual:
            global_ctx = self.global_proj(vision_tokens.mean(dim=1, keepdim=True))
            selected_tokens = torch.cat([selected_tokens, global_ctx], dim=1)
            selected_attention_mask = torch.cat(
                [selected_attention_mask, torch.ones((B, 1), device=device, dtype=torch.long)],
                dim=1,
            )

        return selected_tokens, selected_attention_mask

    def _gather_selected(
        self,
        vision_tokens: torch.Tensor,
        hard_mask: torch.Tensor,
        indicator: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, V, D = vision_tokens.shape

        selected_list = []
        for b in range(B):
            tokens_b = vision_tokens[b][hard_mask[b]]
            if indicator is not None:
                tokens_b = tokens_b * indicator[b][hard_mask[b]].unsqueeze(-1)
            selected_list.append(tokens_b)

        max_len = max(s.shape[0] for s in selected_list)
        padded = torch.zeros(B, max_len, D, device=vision_tokens.device, dtype=vision_tokens.dtype)
        attention_mask = torch.zeros(B, max_len, device=vision_tokens.device, dtype=torch.long)
        for b, sel in enumerate(selected_list):
            padded[b, : sel.shape[0]] = sel
            attention_mask[b, : sel.shape[0]] = 1

        return padded, attention_mask
