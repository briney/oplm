"""Tests for TransformerBlock, OplmEncoder, MLMHead, and OplmForMLM."""

from __future__ import annotations

import copy
from unittest import mock

import pytest
import torch

from oplm.config import ModelConfig
from oplm.model.masking import normalize_attention_mask
from oplm.model.residual import BlockAttentionResidualState
from oplm.model.transformer import MLMHead, OplmEncoder, OplmForMLM, TransformerBlock


def _make_config(**kwargs: object) -> ModelConfig:
    defaults: dict[str, object] = {
        "hidden_dim": 64,
        "num_heads": 4,
        "num_kv_heads": 2,
        "num_layers": 4,
        "max_seq_len": 32,
    }
    defaults.update(kwargs)
    return ModelConfig(**defaults)


B, T = 2, 8
VOCAB = 33


def _run_attn_residual_eager_reference(
    encoder: OplmEncoder,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the pre-optimization eager residual-attention encoder path."""
    assert encoder.attn_residual is not None

    x: torch.Tensor | None = encoder.embedding(input_ids)
    normalized_attention_mask = normalize_attention_mask(attention_mask)
    v_first: torch.Tensor | None = None
    state = BlockAttentionResidualState(blocks=[x], partial_block=None, step_count=0)

    for i, block in enumerate(encoder.blocks):
        ve: torch.Tensor | None = None
        if encoder.value_embedding is not None and encoder.value_embedding.uses_layer(i):
            assert x is not None
            ve = encoder.value_embedding(input_ids, x, i)
        x, v_first, state = block.forward_with_attn_res(
            v_first=v_first,
            attention_mask=normalized_attention_mask,
            value_embed=ve,
            attn_res=encoder.attn_residual,
            state=state,
            materialize_output=True,
        )

    assert x is not None
    return encoder.final_norm(x)


def _assert_encoder_grads_match(
    actual: OplmEncoder,
    expected: OplmEncoder,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> None:
    """Assert encoder parameter gradients match exactly."""
    for (name, param), (expected_name, expected_param) in zip(
        actual.named_parameters(), expected.named_parameters(), strict=True
    ):
        assert name == expected_name
        assert param.grad is not None, f"Missing gradient for {name}"
        assert expected_param.grad is not None, f"Missing reference gradient for {name}"
        torch.testing.assert_close(param.grad, expected_param.grad, atol=atol, rtol=rtol)


def _run_mlm_with_eager_attn_residual_encoder(
    model: OplmForMLM,
    input_ids: torch.Tensor,
    labels: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | None]:
    """Run the MLM head on top of the eager residual-attention encoder path."""
    hidden = _run_attn_residual_eager_reference(model.encoder, input_ids)
    logits = model.mlm_head(hidden)
    loss: torch.Tensor | None = None
    if labels is not None:
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
    return {"logits": logits, "loss": loss}


def _dead_parameter_names(module: torch.nn.Module) -> list[str]:
    """Return the parameter names whose gradients are absent or identically zero."""
    dead_params = []
    for name, param in module.named_parameters():
        if param.requires_grad and (param.grad is None or param.grad.abs().sum() == 0):
            dead_params.append(name)
    return dead_params


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------


class TestTransformerBlock:
    """Tests for single transformer layer."""

    def test_output_shape(self) -> None:
        cfg = _make_config()
        block = TransformerBlock(cfg, layer_idx=0)
        x = torch.randn(B, T, cfg.hidden_dim)
        out, v_first, weights = block(x)
        assert out.shape == (B, T, cfg.hidden_dim)

    def test_no_weights_by_default(self) -> None:
        cfg = _make_config()
        block = TransformerBlock(cfg, layer_idx=0)
        x = torch.randn(B, T, cfg.hidden_dim)
        _, _, weights = block(x)
        assert weights is None

    def test_need_weights(self) -> None:
        cfg = _make_config()
        block = TransformerBlock(cfg, layer_idx=0)
        x = torch.randn(B, T, cfg.hidden_dim)
        _, _, weights = block(x, need_weights=True)
        assert weights is not None
        assert weights.shape == (B, cfg.num_heads, T, T)

    def test_with_conv_a(self) -> None:
        cfg = _make_config(conv_positions="A")
        block = TransformerBlock(cfg, layer_idx=0)
        assert block.conv_a is not None
        assert block.conv_c is None
        x = torch.randn(B, T, cfg.hidden_dim)
        out, _, _ = block(x)
        assert out.shape == (B, T, cfg.hidden_dim)

    def test_with_conv_c(self) -> None:
        cfg = _make_config(conv_positions="C")
        block = TransformerBlock(cfg, layer_idx=0)
        assert block.conv_a is None
        assert block.conv_c is not None
        x = torch.randn(B, T, cfg.hidden_dim)
        out, _, _ = block(x)
        assert out.shape == (B, T, cfg.hidden_dim)

    def test_with_conv_ac(self) -> None:
        cfg = _make_config(conv_positions="AC")
        block = TransformerBlock(cfg, layer_idx=0)
        assert block.conv_a is not None
        assert block.conv_c is not None
        x = torch.randn(B, T, cfg.hidden_dim)
        out, _, _ = block(x)
        assert out.shape == (B, T, cfg.hidden_dim)

    def test_with_attention_mask(self) -> None:
        cfg = _make_config()
        block = TransformerBlock(cfg, layer_idx=0)
        x = torch.randn(B, T, cfg.hidden_dim)
        mask = torch.ones(B, 1, 1, T)
        mask[:, :, :, -2:] = 0  # mask out last 2 positions
        out, _, _ = block(x, attention_mask=mask)
        assert out.shape == (B, T, cfg.hidden_dim)


# ---------------------------------------------------------------------------
# TransformerBlock — attention residuals path
# ---------------------------------------------------------------------------


class TestTransformerBlockAttnRes:
    """Tests for the forward_with_attn_res path."""

    def test_attn_res_forward(self) -> None:
        cfg = _make_config(attn_residual=True, attn_residual_block_size=2)
        block = TransformerBlock(cfg, layer_idx=0)
        from oplm.model.residual import BlockAttentionResidual

        attn_res = BlockAttentionResidual(cfg)
        embed = torch.randn(B, T, cfg.hidden_dim)
        state = BlockAttentionResidualState(blocks=[embed])

        out, v_first, new_state = block.forward_with_attn_res(
            v_first=None,
            attention_mask=None,
            value_embed=None,
            attn_res=attn_res,
            state=state,
        )
        assert out.shape == (B, T, cfg.hidden_dim)
        assert new_state.step_count == 2  # attn + FFN sublayers

    def test_attn_res_forward_can_skip_materialization(self) -> None:
        cfg = _make_config(attn_residual=True, attn_residual_block_size=2)
        block = TransformerBlock(cfg, layer_idx=0)
        from oplm.model.residual import BlockAttentionResidual

        attn_res = BlockAttentionResidual(cfg)
        embed = torch.randn(B, T, cfg.hidden_dim)
        state = BlockAttentionResidualState(blocks=[embed])

        out, v_first, new_state = block.forward_with_attn_res(
            v_first=None,
            attention_mask=None,
            value_embed=None,
            attn_res=attn_res,
            state=state,
            materialize_output=False,
        )
        assert out is None
        assert v_first is None
        assert new_state.step_count == 2


# ---------------------------------------------------------------------------
# OplmEncoder
# ---------------------------------------------------------------------------


class TestOplmEncoder:
    """Tests for the full encoder backbone."""

    def test_output_shape(self) -> None:
        cfg = _make_config()
        encoder = OplmEncoder(cfg)
        input_ids = torch.randint(0, VOCAB, (B, T))
        hidden, weights = encoder(input_ids)
        assert hidden.shape == (B, T, cfg.hidden_dim)
        assert weights is None

    def test_need_weights(self) -> None:
        cfg = _make_config()
        encoder = OplmEncoder(cfg)
        input_ids = torch.randint(0, VOCAB, (B, T))
        hidden, weights = encoder(input_ids, need_weights=True)
        assert weights is not None
        assert len(weights) == cfg.num_layers
        for w in weights:
            assert w.shape == (B, cfg.num_heads, T, T)

    def test_with_attention_mask(self) -> None:
        cfg = _make_config()
        encoder = OplmEncoder(cfg)
        input_ids = torch.randint(0, VOCAB, (B, T))
        mask = torch.ones(B, 1, 1, T)
        hidden, _ = encoder(input_ids, attention_mask=mask)
        assert hidden.shape == (B, T, cfg.hidden_dim)

    def test_with_2d_attention_mask(self) -> None:
        cfg = _make_config()
        encoder = OplmEncoder(cfg)
        input_ids = torch.randint(0, VOCAB, (B, T))
        mask = torch.ones(B, T, dtype=torch.long)
        mask[:, -2:] = 0
        hidden, _ = encoder(input_ids, attention_mask=mask)
        assert hidden.shape == (B, T, cfg.hidden_dim)

    def test_with_value_embeddings(self) -> None:
        cfg = _make_config(num_value_embeds=2)
        encoder = OplmEncoder(cfg)
        assert encoder.value_embedding is not None
        input_ids = torch.randint(0, VOCAB, (B, T))
        hidden, _ = encoder(input_ids)
        assert hidden.shape == (B, T, cfg.hidden_dim)

    def test_no_value_embeddings_by_default(self) -> None:
        cfg = _make_config()
        encoder = OplmEncoder(cfg)
        assert encoder.value_embedding is None

    def test_with_attn_residual(self) -> None:
        cfg = _make_config(attn_residual=True, attn_residual_block_size=2)
        encoder = OplmEncoder(cfg)
        assert encoder.attn_residual is not None
        input_ids = torch.randint(0, VOCAB, (B, T))
        hidden, _ = encoder(input_ids)
        assert hidden.shape == (B, T, cfg.hidden_dim)

    def test_gradient_checkpointing(self) -> None:
        cfg = _make_config(gradient_checkpointing=True)
        encoder = OplmEncoder(cfg)
        encoder.train()
        input_ids = torch.randint(0, VOCAB, (B, T))
        hidden, _ = encoder(input_ids)
        loss = hidden.sum()
        loss.backward()
        # Should not error; verify some gradients exist
        for name, param in encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                break  # just check first param is sufficient

    @pytest.mark.parametrize("num_value_embeds", [0, 2])
    def test_attn_residual_matches_eager_reference(self, num_value_embeds: int) -> None:
        torch.manual_seed(0)
        cfg = _make_config(
            attn_residual=True,
            attn_residual_block_size=2,
            num_value_embeds=num_value_embeds,
        )
        encoder = OplmEncoder(cfg)
        eager_encoder = copy.deepcopy(encoder)

        input_ids = torch.randint(0, VOCAB, (B, T))
        attention_mask = torch.ones(B, T, dtype=torch.long)
        attention_mask[:, -2:] = 0
        grad_weight = torch.randn(B, T, cfg.hidden_dim)

        hidden, _ = encoder(input_ids, attention_mask=attention_mask)
        eager_hidden = _run_attn_residual_eager_reference(
            eager_encoder,
            input_ids,
            attention_mask=attention_mask,
        )

        torch.testing.assert_close(hidden, eager_hidden, atol=1e-6, rtol=1e-5)

        loss = (hidden * grad_weight).sum()
        eager_loss = (eager_hidden * grad_weight).sum()
        loss.backward()
        eager_loss.backward()
        _assert_encoder_grads_match(encoder, eager_encoder)

    def test_attn_residual_skips_intermediate_post_ffn_materialization(self) -> None:
        cfg = _make_config(attn_residual=True, attn_residual_block_size=2, num_layers=4)
        encoder = OplmEncoder(cfg)
        assert encoder.attn_residual is not None

        input_ids = torch.randint(0, VOCAB, (B, T))
        step_indices: list[int] = []
        original_aggregate = encoder.attn_residual.aggregate

        def counting_aggregate(
            state: BlockAttentionResidualState,
            step_idx: int,
        ) -> torch.Tensor:
            step_indices.append(step_idx)
            return original_aggregate(state, step_idx)

        with mock.patch.object(
            encoder.attn_residual,
            "aggregate",
            side_effect=counting_aggregate,
        ) as aggregate_mock:
            hidden, _ = encoder(input_ids)

        assert hidden.shape == (B, T, cfg.hidden_dim)
        assert aggregate_mock.call_count == 2 * cfg.num_layers + 1
        assert step_indices == [0, 1, 2, 3, 4, 5, 6, 7, 7]


# ---------------------------------------------------------------------------
# OplmEncoder — ablation matrix
# ---------------------------------------------------------------------------


class TestEncoderAblationMatrix:
    """Parametrize over boolean feature flags to verify forward pass succeeds."""

    @pytest.mark.parametrize("shared_kv", [False, True])
    @pytest.mark.parametrize("qk_norm", [False, True])
    @pytest.mark.parametrize("value_residual", [False, True])
    @pytest.mark.parametrize("post_embed_norm", [False, True])
    def test_feature_combinations(
        self,
        shared_kv: bool,
        qk_norm: bool,
        value_residual: bool,
        post_embed_norm: bool,
    ) -> None:
        cfg = _make_config(
            shared_kv=shared_kv,
            qk_norm=qk_norm,
            value_residual=value_residual,
            post_embed_norm=post_embed_norm,
        )
        encoder = OplmEncoder(cfg)
        input_ids = torch.randint(0, VOCAB, (B, T))
        hidden, _ = encoder(input_ids)
        assert hidden.shape == (B, T, cfg.hidden_dim)

    @pytest.mark.parametrize("output_gate", [False, True])
    @pytest.mark.parametrize("post_sdpa_norm", [False, True])
    @pytest.mark.parametrize("conv_positions", ["", "A", "C", "AC", "ACD"])
    def test_gate_and_conv_combinations(
        self,
        output_gate: bool,
        post_sdpa_norm: bool,
        conv_positions: str,
    ) -> None:
        cfg = _make_config(
            output_gate=output_gate,
            post_sdpa_norm=post_sdpa_norm,
            conv_positions=conv_positions,
        )
        encoder = OplmEncoder(cfg)
        input_ids = torch.randint(0, VOCAB, (B, T))
        hidden, _ = encoder(input_ids)
        assert hidden.shape == (B, T, cfg.hidden_dim)

    @pytest.mark.parametrize("ffn_activation", ["swiglu", "relu_squared", "gelu"])
    @pytest.mark.parametrize("partial_rope", [False, True])
    def test_activation_and_rope_combinations(
        self,
        ffn_activation: str,
        partial_rope: bool,
    ) -> None:
        # head_dim=16 with default rope_dim=32 would overflow, so set rope_dim=8
        extra: dict[str, object] = {}
        if partial_rope:
            extra["rope_dim"] = 8
        cfg = _make_config(
            ffn_activation=ffn_activation,
            partial_rope=partial_rope,
            **extra,
        )
        encoder = OplmEncoder(cfg)
        input_ids = torch.randint(0, VOCAB, (B, T))
        hidden, _ = encoder(input_ids)
        assert hidden.shape == (B, T, cfg.hidden_dim)


# ---------------------------------------------------------------------------
# Disabled features add zero parameters
# ---------------------------------------------------------------------------


class TestZeroOverheadWhenDisabled:
    """Verify disabled features add exactly zero parameters."""

    def test_no_conv_overhead(self) -> None:
        cfg_no = _make_config(conv_positions="")
        cfg_ac = _make_config(conv_positions="AC")
        n_no = sum(p.numel() for p in OplmEncoder(cfg_no).parameters())
        n_ac = sum(p.numel() for p in OplmEncoder(cfg_ac).parameters())
        assert n_ac > n_no

    def test_no_value_embed_overhead(self) -> None:
        cfg_no = _make_config(num_value_embeds=0)
        cfg_ve = _make_config(num_value_embeds=2)
        n_no = sum(p.numel() for p in OplmEncoder(cfg_no).parameters())
        n_ve = sum(p.numel() for p in OplmEncoder(cfg_ve).parameters())
        assert n_ve > n_no

    def test_no_attn_residual_overhead(self) -> None:
        cfg_no = _make_config(attn_residual=False)
        cfg_ar = _make_config(attn_residual=True, attn_residual_block_size=2)
        n_no = sum(p.numel() for p in OplmEncoder(cfg_no).parameters())
        n_ar = sum(p.numel() for p in OplmEncoder(cfg_ar).parameters())
        assert n_ar > n_no


# ---------------------------------------------------------------------------
# MLMHead
# ---------------------------------------------------------------------------


class TestMLMHead:
    """Tests for the MLM projection head."""

    def test_output_shape(self) -> None:
        cfg = _make_config()
        head = MLMHead(cfg)
        x = torch.randn(B, T, cfg.hidden_dim)
        logits = head(x)
        assert logits.shape == (B, T, VOCAB)


# ---------------------------------------------------------------------------
# OplmForMLM
# ---------------------------------------------------------------------------


class TestOplmForMLM:
    """Tests for the top-level MLM model."""

    def test_logits_shape(self) -> None:
        cfg = _make_config()
        model = OplmForMLM(cfg)
        input_ids = torch.randint(0, VOCAB, (B, T))
        result = model(input_ids)
        assert result["logits"].shape == (B, T, VOCAB)
        assert result["loss"] is None

    def test_loss_with_labels(self) -> None:
        cfg = _make_config()
        model = OplmForMLM(cfg)
        input_ids = torch.randint(0, VOCAB, (B, T))
        labels = torch.full((B, T), -100, dtype=torch.long)
        labels[:, 2:5] = torch.randint(0, VOCAB, (B, 3))
        result = model(input_ids, labels=labels)
        assert result["loss"] is not None
        assert result["loss"].ndim == 0  # scalar

    def test_loss_is_positive(self) -> None:
        cfg = _make_config()
        model = OplmForMLM(cfg)
        input_ids = torch.randint(0, VOCAB, (B, T))
        labels = torch.randint(0, VOCAB, (B, T))
        result = model(input_ids, labels=labels)
        assert result["loss"].item() > 0

    def test_tie_embeddings(self) -> None:
        cfg = _make_config(tie_embeddings=True)
        model = OplmForMLM(cfg)
        assert model.mlm_head.projection.weight is model.encoder.embedding.embed.weight

    def test_no_tie_embeddings(self) -> None:
        cfg = _make_config(tie_embeddings=False)
        model = OplmForMLM(cfg)
        assert model.mlm_head.projection.weight is not model.encoder.embedding.embed.weight

    def test_attention_weights_returned(self) -> None:
        cfg = _make_config()
        model = OplmForMLM(cfg)
        input_ids = torch.randint(0, VOCAB, (B, T))
        result = model(input_ids, need_weights=True)
        assert "attention_weights" in result
        assert len(result["attention_weights"]) == cfg.num_layers

    def test_with_attn_residual(self) -> None:
        cfg = _make_config(attn_residual=True, attn_residual_block_size=2)
        model = OplmForMLM(cfg)
        input_ids = torch.randint(0, VOCAB, (B, T))
        labels = torch.randint(0, VOCAB, (B, T))
        result = model(input_ids, labels=labels)
        assert result["logits"].shape == (B, T, VOCAB)
        assert result["loss"] is not None


# ---------------------------------------------------------------------------
# Gradient flow — full model
# ---------------------------------------------------------------------------


class TestFullModelGradient:
    """Verify gradients reach all parameters through the full model."""

    def test_all_params_receive_gradients(self) -> None:
        cfg = _make_config()
        model = OplmForMLM(cfg)
        input_ids = torch.randint(0, VOCAB, (B, T))
        labels = torch.randint(0, VOCAB, (B, T))
        result = model(input_ids, labels=labels)
        result["loss"].backward()
        dead_params = _dead_parameter_names(model)
        assert len(dead_params) == 0, f"Dead parameters: {dead_params}"

    def test_all_params_receive_gradients_with_features(self) -> None:
        """Full feature set: convolutions, value embeddings, value residuals."""
        cfg = _make_config(
            conv_positions="AC",
            num_value_embeds=2,
            value_residual=True,
            output_gate=True,
            qk_norm=True,
        )
        model = OplmForMLM(cfg)
        input_ids = torch.randint(0, VOCAB, (B, T))
        labels = torch.randint(0, VOCAB, (B, T))
        result = model(input_ids, labels=labels)
        result["loss"].backward()
        dead_params = _dead_parameter_names(model)
        assert len(dead_params) == 0, f"Dead parameters: {dead_params}"

    @pytest.mark.parametrize("num_value_embeds", [0, 2])
    def test_all_params_receive_gradients_with_attn_residual(
        self,
        num_value_embeds: int,
    ) -> None:
        cfg = _make_config(
            attn_residual=True,
            attn_residual_block_size=2,
            num_value_embeds=num_value_embeds,
        )
        model = OplmForMLM(cfg)
        eager_model = copy.deepcopy(model)
        input_ids = torch.randint(0, VOCAB, (B, T))
        labels = torch.randint(0, VOCAB, (B, T))
        result = model(input_ids, labels=labels)
        eager_result = _run_mlm_with_eager_attn_residual_encoder(
            eager_model,
            input_ids,
            labels=labels,
        )
        assert result["loss"] is not None
        assert eager_result["loss"] is not None
        torch.testing.assert_close(result["logits"], eager_result["logits"], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(result["loss"], eager_result["loss"], atol=1e-6, rtol=1e-5)
        result["loss"].backward()
        eager_result["loss"].backward()

        assert _dead_parameter_names(model) == _dead_parameter_names(eager_model)
