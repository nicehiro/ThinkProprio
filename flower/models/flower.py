import logging
from typing import Dict, Tuple, List
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import einops
from timm.layers.mlp import Mlp
from transformers import AutoModelForCausalLM, AutoProcessor

from flower.models.networks.transformers import (
    TimestepEmbedder,
    SharedAdaLNController,
    RmsNorm,
    FreqEmbedder,
    ActionSpaceEmbedderParameter,
    FlowBlock,
    stateless_norm,
)
from flower.models.networks.pre_vlm_selector import (
    ProprioTextTokenizer,
    PreVLMVisionSelector,
)
from flower.utils.lr_schedulers.tri_stage_scheduler import TriStageLRScheduler
from flower.models.utils import ActionIndex, generate_policy_prompt

logger = logging.getLogger(__name__)


class FLOWERVLA(pl.LightningModule):
    def __init__(
        self,
        vlm_path: str = "microsoft/Florence-2-base",
        vlm_revision: str = None,
        freeze_florence: bool = False,
        freeze_vision_tower: bool = False,
        vlm_prompt_style: str = "default",
        token_dropout: float = 0.2,

        multistep: int = 10,
        num_sampling_steps: int = 5,
        lowdim_obs_dim: int = 7,
        action_dim: int = 7,
        act_window_size: int = 10,

        use_second_view: bool = False,
        second_view_key: str = 'image_wrist',
        action_type_adaln: bool = True,
        use_causal_attention: bool = True,
        use_cross_attn: bool = True,
        use_adaln_cond: bool = False,
        use_readout_token: bool = False,
        return_act_chunk: bool = False,

        pre_vlm_use_residual: bool = True,
        pre_vlm_num_bins: int = 256,
        pre_vlm_min_value: float = -3.0,
        pre_vlm_max_value: float = 3.0,
        pre_vlm_noise_start: float = 1.0,
        pre_vlm_noise_end: float = 0.01,
        pre_vlm_div_loss_weight: float = 5e-4,
        pre_vlm_div_cos_rho: float = 0.75,
        pre_vlm_div_temperature: float = 1.0,

        sampling_type: str = 'ln',
        dit_dim: int = 512,
        n_heads: int = 16,
        n_layers: int = 12,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        mlp_pdrop: float = 0.1,

        use_rope: bool = False,
        use_nope: bool = False,
        query_seq_len: int = 128,
        rope_theta: float = 32.0,

        optimizer_type: str = "adamw",
        optimizer: DictConfig = None,
        lr_scheduler: DictConfig = None,

        load_pretrained: bool = False,
        pretrained_model_path: str = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.action_space_index = ActionIndex()
        self._init_flags(
            use_second_view=use_second_view,
            use_causal_attention=use_causal_attention,
            use_cross_attn=use_cross_attn,
            use_adaln_cond=use_adaln_cond,
            use_readout_token=use_readout_token,
            use_rope=use_rope,
            use_nope=use_nope,
            vlm_prompt_style=vlm_prompt_style,
            token_dropout=token_dropout,
            action_type_adaln=action_type_adaln,
            sampling_type=sampling_type,
            return_act_chunk=return_act_chunk,
            second_view_key=second_view_key,
            pre_vlm_use_residual=pre_vlm_use_residual,
            pre_vlm_num_bins=pre_vlm_num_bins,
            pre_vlm_min_value=pre_vlm_min_value,
            pre_vlm_max_value=pre_vlm_max_value,
            pre_vlm_noise_start=pre_vlm_noise_start,
            pre_vlm_noise_end=pre_vlm_noise_end,
            pre_vlm_div_loss_weight=pre_vlm_div_loss_weight,
            pre_vlm_div_cos_rho=pre_vlm_div_cos_rho,
            pre_vlm_div_temperature=pre_vlm_div_temperature,
        )
        self.obs_modalities = []
        self._init_dimensions(
            dit_dim=dit_dim,
            n_heads=n_heads,
            lowdim_obs_dim=lowdim_obs_dim,
            action_dim=action_dim,
            act_window_size=act_window_size,
            multistep=multistep,
            num_sampling_steps=num_sampling_steps,
        )
        self.target_modality = "actions"
        self._setup_vlm(vlm_path, freeze_vision_tower, freeze_florence, vlm_revision)
        hidden_dim = self.vlm.config.text_config.d_model
        self.vlm_latent_dim = hidden_dim
        self.action_type_adaln = action_type_adaln
        self._setup_dit_components(
            dit_dim=dit_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            lowdim_obs_dim=lowdim_obs_dim,
            action_dim=action_dim,
            act_window_size=act_window_size,
            hidden_dim=hidden_dim,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            mlp_pdrop=mlp_pdrop,
            use_cross_attn=use_cross_attn,
            use_rope=use_rope,
            use_nope=use_nope,
            query_seq_len=query_seq_len,
            rope_theta=rope_theta,
        )

        self.rollout_step_counter = 0
        self.pred_action_seq = None
        self.modality_scope = "lang"

        self.optimizer_config = optimizer
        self.lr_scheduler_config = lr_scheduler
        self.optimizer_type = optimizer_type

        if load_pretrained and pretrained_model_path is not None:
            self._load_pretrained_weights(pretrained_model_path)

    def _load_pretrained_weights(self, pretrained_model_path: str, mean_resizing: bool = False):
        print(f"Loading pretrained weights from {pretrained_model_path}...")
        if pretrained_model_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(pretrained_model_path, device=str(self.device))
            checkpoint = {"state_dict": state_dict}
            print("Loaded safetensors file")
        else:
            checkpoint = torch.load(pretrained_model_path, map_location=self.device)
            state_dict = checkpoint.get("state_dict", checkpoint)

        state_dict = checkpoint.get("state_dict", checkpoint)

        if ("callbacks" in checkpoint and
                "EMA" in checkpoint["callbacks"] and
                "ema_weights" in checkpoint["callbacks"]["EMA"]):
            ema_weights = checkpoint["callbacks"]["EMA"]["ema_weights"]
            if len(ema_weights) != len(state_dict):
                raise ValueError("EMA tensor count does not match the checkpoint state dict")
            for (name, weight), ema_weight in zip(state_dict.items(), ema_weights):
                if weight.shape != ema_weight.shape:
                    raise ValueError(f"EMA shape mismatch for {name}: {ema_weight.shape} != {weight.shape}")
            state_dict = dict(zip(state_dict, ema_weights))
            print(f"Loading all {len(state_dict)} EMA tensors in checkpoint order")

        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("agent.", "")

            if "vlm.language_encoder." in new_key:
                new_key = new_key.replace("vlm.language_encoder.", "vlm.language_model.model.encoder.")
            new_key = new_key.replace(".mlp.c_fc1.", ".mlp.fc1.")
            new_key = new_key.replace(".mlp.c_fc2.", ".mlp.fc2.")
            new_key = new_key.replace(".mlp.c_proj.", ".mlp.proj.")
            new_state_dict[new_key] = value

        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=True)

        print(f"Pretrained weights loaded with the following issues:")
        if missing_keys:
            print(f"  ⚠️ Missing keys (not found in checkpoint, using default init): {len(missing_keys)}")
            print(f"    {missing_keys[:30]} ...")
        if unexpected_keys:
            print(f"  ⚠️ Unexpected keys (ignored): {len(unexpected_keys)}")
            print(f"    {unexpected_keys[:30]} ...")
        if not missing_keys and not unexpected_keys:
            print("  ✅ All keys matched successfully!")

        if mean_resizing:
            self._initialize_new_embeddings(new_state_dict)

        return missing_keys, unexpected_keys

    def _init_flags(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.vlm_prompt_style not in ["default", "feature_focused", "state_oriented"]:
            raise ValueError("Invalid VLM prompt style")

        if self.sampling_type not in ['ln', 'pi_zero', 'loglogistic', 'uniform', 'stratified']:
            raise ValueError(f"Invalid sampling type: {self.sampling_type}")

        self.format_instruction = functools.partial(
                             generate_policy_prompt,
                             robot_name="Franka Panda",
                             action_space="Delta End-Effector",
                             num_arms="1",
                             prompt_style='minimal')

        self.use_adaln_cond = self.use_adaln_cond
        self.use_readout_token = self.use_readout_token and self.use_adaln_cond
        self.use_second_view = self.use_second_view and self.second_view_key is not None
        self.use_cross_attn = self.use_cross_attn
        self.use_rope = self.use_rope and not self.use_nope
        self.use_nope = self.use_nope and not self.use_rope
        self.vlm_prompt_style = self.vlm_prompt_style
        self.return_act_chunk = False

    def _init_dimensions(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.dit_dim % self.n_heads != 0:
            raise ValueError(f"dit_dim ({self.dit_dim}) must be divisible by n_heads ({self.n_heads})")


    def _setup_vlm(self, vlm_path: str, freeze_vision_tower: bool, freeze_florence: bool, revision=None):
        print(f"Loading Florence-2 from {vlm_path}")

        self.vlm = AutoModelForCausalLM.from_pretrained(vlm_path, revision=revision, trust_remote_code=True)

        if freeze_florence:
            for param in self.vlm.parameters():
                param.requires_grad = False
        elif not freeze_vision_tower:
            for param in self.vlm.vision_tower.parameters():
                param.requires_grad = True

        self.processor = AutoProcessor.from_pretrained(vlm_path, revision=revision, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer

        self.prompt_embeds = self._create_prompt_embed("<Flow>")

        del self.vlm.language_model.model.decoder
        del self.vlm.language_model.lm_head

        self.vlm_token_dropout = nn.Dropout(self.token_dropout)

        self.proprio_text_tokenizer = ProprioTextTokenizer(
            vlm_embeddings=self.vlm.get_input_embeddings(),
            vocab_size=self.tokenizer.vocab_size,
            num_bins=self.pre_vlm_num_bins,
            min_value=self.pre_vlm_min_value,
            max_value=self.pre_vlm_max_value,
        )

        self.pre_vlm_selector = PreVLMVisionSelector(
            vision_dim=self.vlm.config.vision_config.projection_dim,
            use_residual=self.pre_vlm_use_residual,
            gumbel_noise_start=self.pre_vlm_noise_start,
            gumbel_noise_end=self.pre_vlm_noise_end,
            div_cos_rho=self.pre_vlm_div_cos_rho,
            div_temperature=self.pre_vlm_div_temperature,
        )

    def _setup_dit_components(self, **kwargs):
        dit_dim = kwargs['dit_dim']
        n_heads = kwargs['n_heads']
        n_layers = kwargs['n_layers']
        hidden_dim = kwargs['hidden_dim']
        use_cross_attn = kwargs['use_cross_attn']
        use_rope = kwargs['use_rope']
        use_nope = kwargs['use_nope']

        self.action_encoders = nn.ModuleDict()
        self.action_decoders = nn.ModuleDict()

        self.adaln = nn.ModuleDict() if self.action_type_adaln else None

        self.cond_linear = nn.Linear(hidden_dim, dit_dim, bias=False)
        self.t_embedder = TimestepEmbedder(dit_dim)
        self.cond_norm = RmsNorm(hidden_dim)
        self.frequency_embedder = FreqEmbedder(dit_dim)
        self.action_space_embedder = ActionSpaceEmbedderParameter(dit_dim, max_actions=len(self.action_space_index.action_spaces))


        if not use_rope and not use_nope:
            self.positional_encoding = nn.Parameter(torch.randn(1, kwargs['act_window_size'], dit_dim) * 0.1)

        self.dit = nn.ModuleList([
            FlowBlock(
                dit_dim, n_heads,
                attn_pdrop=kwargs['attn_pdrop'],
                resid_pdrop=kwargs['resid_pdrop'],
                mlp_pdrop=kwargs['mlp_pdrop'],
                use_cross_attn=use_cross_attn,
                use_rope=use_rope,
                query_seq_len=kwargs['query_seq_len'],
                rope_theta=kwargs['rope_theta'],

            ) for _ in range(n_layers)
        ])

        for action_name, action_idx in self.action_space_index.action_spaces.items():
            input_dim = self.action_space_index.get_action_dim(action_idx)

            self.action_encoders[action_name] =  Mlp(in_features=input_dim, hidden_features=dit_dim, out_features=dit_dim, bias=True)
            self.action_decoders[action_name] = nn.Linear(dit_dim, input_dim).to(self.device)

            if self.action_type_adaln:
                self.adaln[action_name] = SharedAdaLNController(dit_dim, global_conddim=dit_dim, use_cross_attn=use_cross_attn)

    def configure_optimizers(self):
        optim_groups = self._get_param_groups()

        optimizer = torch.optim.AdamW(
                optim_groups,
                lr=self.optimizer_config.learning_rate,
                betas=self.optimizer_config.betas
            )

        scheduler = TriStageLRScheduler(
            optimizer,
            OmegaConf.create(self.lr_scheduler_config)
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    def _get_param_groups(self):
        no_decay = ['bias', 'LayerNorm', 'layernorm', 'ln', 'norm']
        decay_group = []
        no_decay_group = []

        for name, param in self.named_parameters():
            if param.requires_grad:
                if any(nd in name.lower() for nd in no_decay):
                    no_decay_group.append(param)
                else:
                    decay_group.append(param)

        return [
            {"params": decay_group, "weight_decay": self.optimizer_config.transformer_weight_decay},
            {"params": no_decay_group, "weight_decay": 0.0}
        ]

    def training_step(self, batch: Dict[str, Dict], batch_idx: int) -> torch.Tensor:
        opt = self.optimizers()

        total_loss = torch.tensor(0.0, device=self.device)
        action_loss = torch.tensor(0.0, device=self.device)
        total_bs = 0

        for modality_scope, dataset_batch in batch.items():
            self.modality_scope = modality_scope
            obs_features = self.encode_observations(dataset_batch)
            act_loss, losses_dict = self.rf_loss(obs_features, dataset_batch["actions"])
            div_loss = self._pre_vlm_diversity_loss()
            loss = act_loss + div_loss
            action_loss = action_loss + act_loss
            total_loss = total_loss + loss
            total_bs = total_bs + len(dataset_batch["actions"])

        total_loss = total_loss / len(batch)

        self._log_training_metrics(total_loss, action_loss, total_bs)


        return total_loss

    def _pre_vlm_diversity_loss(self) -> torch.Tensor:
        if self.pre_vlm_div_loss_weight <= 0 or self.pre_vlm_selector._last_div_loss is None:
            return torch.tensor(0.0, device=self.device)
        return self.pre_vlm_selector._last_div_loss * self.pre_vlm_div_loss_weight

    def validation_step(self, batch: Dict[str, Dict], batch_idx: int) -> Dict[str, torch.Tensor]:
        output = {}
        with torch.no_grad():
            obs_features = self.encode_observations(batch)
            target_actions = batch[self.target_modality]

            noise_actions = torch.randn_like(target_actions, device=self.device)

            action_pred = self.sample_actions(noise_actions, obs_features, inference=True)

            val_loss = F.mse_loss(action_pred, target_actions)

            self._log_validation_metrics(val_loss, val_loss)

            output["validation_loss"] = val_loss / len(batch)
            return output

    def rf_loss(self, cond, actions, dataset_idx=None):
        default_dtype = next(self.parameters()).dtype

        if len(actions.shape) == 4:
            actions = actions.squeeze(1)
        b = actions.size(0)
        device = actions.device
        actions = actions.to(default_dtype)

        if self.sampling_type == "pi_zero":
            alpha, beta = 1.5, 1.0
            t = torch.distributions.Beta(alpha, beta).sample((b,)).to(device)
            t = t.clamp(max=0.999)
        elif self.sampling_type == "ln":
            t = torch.sigmoid(torch.randn((b,), device=device))
            t = t.clamp(max=0.999).to(default_dtype)
        elif self.sampling_type == "uniform":
            eps = 1e-5
            t = (torch.rand(1, device=device) + torch.arange(b, device=device) / b) % (1 - eps)
            t = t.to(default_dtype)
        else:
            raise NotImplementedError(f"Sampling type {self.sampling_type} not implemented")

        texp = t.view([b] + [1] * (actions.dim() - 1))
        z1 = torch.randn_like(actions, device=device).to(default_dtype)

        zt = (1 - texp) * actions + texp * z1

        vtheta = self.dit_forward(zt, t, cond)
        diff = (z1 - actions) - vtheta
        valid_diff = diff
        loss = (valid_diff ** 2).mean()

        losses_dict = {
            "diff_min": valid_diff.min().item(),
            "diff_max": valid_diff.max().item(),
            "diff_mean": valid_diff.mean().item(),
            "loss": loss.item(),
        }

        return loss, losses_dict

    def sample_actions(self, z: torch.Tensor, cond: Dict[str, torch.Tensor], inference: bool=False):
        steps = self.num_sampling_steps if inference else 5
        b = z.size(0)
        device = z.device

        dt = 1.0 / steps
        dt_tensor = torch.tensor([dt] * b, device=device).view([b] + [1]*(z.dim()-1))

        precomputed = self._prepare_dit_cond(cond)

        for i in range(steps, 0, -1):
            t_val = i / steps
            t_tensor = torch.full((b,), t_val, device=device)

            vc = self.dit_forward(z, t_tensor, cond, precomputed=precomputed)
            z = z - dt_tensor * vc

        return z.clamp(-1, 1)

    def _prepare_dit_cond(self, cond_dict: dict) -> dict:
        default_dtype = next(self.parameters()).dtype

        cond = cond_dict['features'].to(default_dtype)
        frequency_embeds = cond_dict['frequency_embeds'].squeeze(1).to(default_dtype)
        action_type = cond_dict['action_type'].to(self.device)

        cond_projected = self.cond_linear(self.cond_norm(cond))

        context = cond_projected if self.use_cross_attn else None

        if self.use_adaln_cond:
            vlm_token = cond_projected[:, 0, :] if self.use_readout_token else cond_projected.mean(dim=1)
        else:
            vlm_token = None

        return {
            'cond_projected': cond_projected,
            'frequency_embeds': frequency_embeds,
            'action_type': action_type,
            'context': context,
            'vlm_token': vlm_token,
        }

    def dit_forward(self, z: torch.Tensor, t: torch.Tensor, cond_dict: dict, precomputed: dict = None) -> torch.Tensor:
        default_dtype = next(self.parameters()).dtype
        B, t_seq, d = z.shape

        if precomputed is not None:
            cond = precomputed['cond_projected']
            frequency_embeds = precomputed['frequency_embeds']
            action_type = precomputed['action_type']
            context = precomputed['context']
            vlm_token = precomputed['vlm_token']
        else:
            cond = cond_dict['features'].to(default_dtype)
            frequency_embeds = cond_dict['frequency_embeds'].squeeze(1).to(default_dtype)
            action_type = cond_dict['action_type'].to(self.device)

            cond = self.cond_linear(self.cond_norm(cond))

            context = cond if self.use_cross_attn else None

            if self.use_adaln_cond:
                vlm_token = cond[:, 0, :] if self.use_readout_token else cond.mean(dim=1)
            else:
                vlm_token = None

        z, valid_dims = self.encode_actions(z, action_type)

        if not self.use_rope and not self.use_nope:
            z = z + self.positional_encoding

        t_emb = stateless_norm(self.t_embedder(t)) + \
                stateless_norm(frequency_embeds).squeeze(1)

        if self.use_adaln_cond:
            global_cond = vlm_token + t_emb
        else:
            global_cond = t_emb

        if not self.action_type_adaln:
            global_adaln = self.adaln(global_cond)
        else:
            global_adaln = self.action_specific_adaln(global_cond, action_type)

        cx = z
        for layer in self.dit:
            cx = layer(
                cx,
                global_cond,
                context=context,
                is_causal=True,
                global_adaln=global_adaln
            )

        return self.decode_actions(cx, action_type, valid_dims)

    def action_specific_adaln(self, global_cond: torch.Tensor, action_type: torch.Tensor) -> List[torch.Tensor]:
        default_type = next(self.parameters()).dtype
        batch_size = global_cond.shape[0]
        num_chunks = 9 if self.use_cross_attn else 6
        device = global_cond.device

        mod_signals = [
            torch.zeros(batch_size, self.dit_dim, device=device, dtype=default_type)
            for _ in range(num_chunks)
        ]

        for action_idx in range(len(self.action_space_index.action_spaces)):
            mask = (action_type == action_idx)
            if mask.any():
                action_name = self.action_space_index.get_action_name(action_idx)
                action_mod = self.adaln[action_name](global_cond)
                for i, signal in enumerate(action_mod):
                    mod_signals[i] = signal

        return mod_signals

    def _create_prompt_embed(self, prompt_text):
        self.tokenizer.add_special_tokens({'additional_special_tokens': [prompt_text]})
        self.vlm.resize_token_embeddings(len(self.tokenizer))

        prompt_token_id = self.tokenizer.convert_tokens_to_ids(prompt_text)
        prompt_embed = nn.Parameter(
            self.vlm.get_input_embeddings()(torch.tensor(prompt_token_id)),
            requires_grad=False
        )

        return prompt_embed.unsqueeze(0).unsqueeze(0)

    def encode_observations(self, batch: Dict) -> torch.Tensor:
        device = self.device
        default_type = next(self.parameters()).dtype

        embed_tensor = torch.zeros(len(batch["rgb_obs"]['rgb_static']), 1, 1)
        action_type_tensor = torch.ones(len(batch["rgb_obs"]['rgb_static']), self.act_window_size, 7)
        image_tensor = batch["rgb_obs"]['rgb_static']
        B, T, C, H, W = image_tensor.shape

        last_image = image_tensor[:, -1, :, :, :]
        image_features = self.vlm._encode_image(
            last_image.view(-1, C, H, W).to(device).to(default_type)
        ).to(default_type)
        image_features = image_features.view(B, 1 * image_features.shape[1], -1)

        if self.use_second_view:
            image2_tensor = batch["rgb_obs"]['rgb_gripper']
            last_image2 = image2_tensor[:, -1, :, :, :]
            image2_features = self.vlm._encode_image(
                last_image2.view(-1, C, H, W).to(device).to(default_type)
            ).to(default_type)
            image2_features = image2_features.view(B, 1 * image2_features.shape[1], -1)
            image_features = torch.cat([image_features, image2_features], dim=1)

        constructed_prompts = self.construct_prompts(batch)
        text_embeds, text_attention_mask = self._get_text_embeddings(constructed_prompts, device)

        proprio_history = batch['robot_obs'].to(device).to(default_type)
        proprio_embeds = self.proprio_text_tokenizer(proprio_history)
        proprio_attention_mask = torch.ones(
            (B, proprio_embeds.shape[1]),
            device=device,
            dtype=text_attention_mask.dtype,
        )

        image_features, vision_attention_mask = self.pre_vlm_selector(
            image_features,
            proprio_embeds=proprio_embeds,
            text_embeds=text_embeds,
            query_attention_mask=torch.cat([text_attention_mask, proprio_attention_mask], dim=1),
        )

        task_prompt = self.prompt_embeds.expand(B, -1, -1).to(image_features.device)

        merged_embeds = torch.cat(
            [
                image_features,
                proprio_embeds.to(image_features.device),
                task_prompt,
                text_embeds.to(image_features.device),
            ],
            dim=1,
        )

        task_prompt_attention_mask = torch.ones(
            (B, task_prompt.shape[1]),
            device=merged_embeds.device,
            dtype=text_attention_mask.dtype,
        )
        attention_mask = torch.cat(
            [
                vision_attention_mask.to(merged_embeds.device, dtype=text_attention_mask.dtype),
                proprio_attention_mask.to(merged_embeds.device),
                task_prompt_attention_mask,
                text_attention_mask.to(merged_embeds.device),
            ],
            dim=1,
        )

        features = self.vlm.get_encoder()(
            inputs_embeds=merged_embeds,
            attention_mask=attention_mask
        ).last_hidden_state

        features = self.vlm_token_dropout(features)

        frequency_embeds = self.frequency_embedder(
            torch.ones_like(embed_tensor).to(device) * 3
        )

        return {
            'features': features,
            'frequency_embeds': frequency_embeds,
            'action_space_embeds': None,
            'action_type': torch.ones_like(action_type_tensor),
            'attention_mask': attention_mask,
        }

    def encode_actions(self, z: torch.Tensor, action_type: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        default_dtype = next(self.parameters()).dtype
        action_type = action_type.to(self.device)
        batch_size = z.shape[0]
        encoded = torch.zeros(batch_size, z.shape[1], self.dit_dim, device=self.device).to(default_dtype)

        valid_dims = torch.zeros_like(z).to(default_dtype)

        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                encoded = self.action_encoders[action_name](z)

        return encoded, valid_dims

    def decode_actions(self, z: torch.Tensor, action_type: torch.Tensor, valid_dims: torch.Tensor) -> torch.Tensor:
        default_dtype = next(self.parameters()).dtype
        batch_size = z.shape[0]
        max_action_dim = self.action_dim
        decoded = torch.zeros(batch_size, z.shape[1], max_action_dim,
                        device=self.device).to(default_dtype)

        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                action_dim = self.action_space_index.get_action_dim(action_idx)

                pred = self.action_decoders[action_name](z)
                decoded = pred
        return decoded

    def forward(self, obs: Dict, goal: Dict) -> torch.Tensor:
        rgb_static = obs["rgb_obs"]['rgb_static']
        rgb_gripper = obs["rgb_obs"]['rgb_gripper']

        batch = {
            "rgb_obs": {
                "rgb_static": rgb_static,
                "rgb_gripper": rgb_gripper
            },
            "lang_text": [goal["lang_text"]]
        }
        robot_obs = obs["robot_obs"]
        if robot_obs.dim() == 2:
            robot_obs = robot_obs.unsqueeze(1)
        batch["robot_obs"] = robot_obs

        features = self.encode_observations(batch)

        noise = torch.randn(
            len(features['features']),
            self.act_window_size,
            self.action_dim,
            device=features['features'].device
        )

        actions = self.sample_actions(noise, features, inference=True)

        return actions

    def step(self, obs: Dict, goal: Dict) -> torch.Tensor:
        if self.rollout_step_counter % self.multistep == 0:
            self.pred_action_seq = self(obs, goal)

        if not self.return_act_chunk:
            current_action = self.pred_action_seq[0, self.rollout_step_counter]
            if len(current_action.shape) == 2:
                current_action = einops.rearrange(current_action, 'b d -> b 1 d')
        else:
            current_action = self.pred_action_seq

        self.rollout_step_counter += 1
        if self.rollout_step_counter == self.multistep:
            self.rollout_step_counter = 0

        return current_action

    def reset(self):
        self.rollout_step_counter = 0
        self.pred_action_seq = None
        self.eval()

    def reset_sequence(self):
        self.reset()

    def on_train_start(self):
        self.to(self.device)
        self.vlm.to(self.device)

    def on_train_batch_start(self, batch, batch_idx):
        if self.trainer.max_steps and self.trainer.max_steps > 0:
            progress = self.global_step / self.trainer.max_steps
        elif self.trainer.max_epochs and self.trainer.max_epochs > 0:
            steps_per_epoch = self.trainer.num_training_batches
            total_steps = self.trainer.max_epochs * steps_per_epoch
            progress = self.global_step / max(total_steps, 1)
        else:
            progress = 0.0

        self.pre_vlm_selector.set_training_progress(progress)

    def on_validation_start(self):
        self.eval()

    def on_validation_end(self):
        self.train()

    def construct_prompts(self, dataset_batch):
        language_instruction = dataset_batch["lang_text"]
        text_prompts = []

        for instruction in language_instruction:
            if self.vlm_prompt_style == "default":
                text_prompts.append(self.format_instruction(instruction))

            elif self.vlm_prompt_style == "feature_focused":
                prompt = f"<od>{instruction}</od><grounding>identify objects and spatial relationships for robotic manipulation</grounding>"
                text_prompts.append(prompt)

            elif self.vlm_prompt_style == "state_oriented":
                prompt = f"<od>{instruction}</od><referring_expression_segmentation>locate objects and regions for manipulation</referring_expression_segmentation>"
                text_prompts.append(prompt)

            else:
                raise ValueError(f"Unknown prompt style: {self.vlm_prompt_style}")


        return text_prompts

    def _get_text_embeddings(self, text, device):
        text_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(device)

        text_embeds = self.vlm.get_input_embeddings()(text_inputs["input_ids"])
        text_attention_mask = text_inputs["attention_mask"]
        return text_embeds, text_attention_mask

    def _log_training_metrics(self, total_loss, action_loss, total_bs):
        self.log("train/action_loss", action_loss, on_step=False, on_epoch=True,
                sync_dist=True, batch_size=total_bs)
        self.log("train/total_loss", total_loss, on_step=False, on_epoch=True,
                sync_dist=True, batch_size=total_bs)

        selector = self.pre_vlm_selector
        self.log("train/pre_vlm_compression_ratio", selector._last_compression_ratio,
                on_step=True, on_epoch=False, sync_dist=False)
        self.log("train/pre_vlm_tokens_selected", selector._last_num_selected,
                on_step=True, on_epoch=False, sync_dist=False)
        if selector._last_div_cos is not None:
            self.log("train/pre_vlm_div_cos", selector._last_div_cos.detach(),
                    on_step=True, on_epoch=False, sync_dist=False)
            self.log("train/pre_vlm_div_loss_unweighted", selector._last_div_loss.detach(),
                    on_step=True, on_epoch=False, sync_dist=False)
        for key, value in selector._diagnostics.items():
            self.log(f"train/{key}", value, on_step=True, on_epoch=False, sync_dist=False)
        selector._diagnostics = {}

    def _log_validation_metrics(self, pred_loss, val_total_act_loss_pp):
        self.log(
            f"val_act/{self.modality_scope}_act_loss_pp",
            pred_loss,
            sync_dist=True
        )

        try:
            n_modalities = len(self.trainer.datamodule.modalities)
        except AttributeError:
            n_modalities = 1

        self.log(
            "val_act/action_loss",
            val_total_act_loss_pp / n_modalities,
            sync_dist=True
        )
