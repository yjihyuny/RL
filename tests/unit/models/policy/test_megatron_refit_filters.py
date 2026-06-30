# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the Kimi refit filters and HF-architectures override helpers."""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.mcore


def _bare_worker(rank: int):
    """Build a MegatronPolicyWorkerImpl without running __init__."""
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        MegatronPolicyWorkerImpl,
    )

    worker = MegatronPolicyWorkerImpl.__new__(MegatronPolicyWorkerImpl)
    worker.rank = rank
    worker.cfg = {}
    worker._refit_ep_rank_filter_logged = False
    return worker


class TestGlobalExpertIdFromRefitName:
    def test_extracts_expert_id(self):
        from nemo_rl.models.policy.workers.megatron_policy_worker import (
            MegatronPolicyWorkerImpl,
        )

        fn = MegatronPolicyWorkerImpl._global_expert_id_from_refit_name
        assert fn("model.layers.3.mlp.experts.42.gate_proj.weight") == 42
        assert fn("decoder.layers.0.mlp.experts.0.down_proj.weight_packed") == 0
        assert fn("model.layers.3.self_attn.q_proj.weight") is None


class TestShouldSkipExpertForEpRankRefit:
    def test_disabled_when_flag_unset(self, monkeypatch):
        monkeypatch.delenv(
            "KIMI_FILTER_EXPERT_REFIT_BY_EP_RANK_FOR_SMOKE", raising=False
        )
        worker = _bare_worker(rank=1)
        assert (
            worker._should_skip_expert_for_ep_rank_refit(
                "model.layers.0.mlp.experts.300.gate_proj.weight"
            )
            is False
        )

    def test_keeps_owned_experts_and_skips_others(self, monkeypatch):
        # 384 global experts over EP=64 -> 6 local experts per rank.
        monkeypatch.setenv("KIMI_FILTER_EXPERT_REFIT_BY_EP_RANK_FOR_SMOKE", "true")
        monkeypatch.setenv("KIMI_REFIT_NUM_GLOBAL_EXPERTS", "384")
        monkeypatch.setenv("KIMI_REFIT_EXPERT_PARALLEL_SIZE", "64")

        worker0 = _bare_worker(rank=0)  # ep_rank 0 owns [0, 6)
        assert not worker0._should_skip_expert_for_ep_rank_refit(
            "model.layers.0.mlp.experts.0.gate_proj.weight"
        )
        assert not worker0._should_skip_expert_for_ep_rank_refit(
            "model.layers.0.mlp.experts.5.gate_proj.weight"
        )
        assert worker0._should_skip_expert_for_ep_rank_refit(
            "model.layers.0.mlp.experts.6.gate_proj.weight"
        )

        worker1 = _bare_worker(rank=1)  # ep_rank 1 owns [6, 12)
        assert worker1._should_skip_expert_for_ep_rank_refit(
            "model.layers.0.mlp.experts.5.gate_proj.weight"
        )
        assert not worker1._should_skip_expert_for_ep_rank_refit(
            "model.layers.0.mlp.experts.6.gate_proj.weight"
        )

    def test_no_filtering_when_not_divisible(self, monkeypatch):
        monkeypatch.setenv("KIMI_FILTER_EXPERT_REFIT_BY_EP_RANK_FOR_SMOKE", "true")
        monkeypatch.setenv(
            "KIMI_REFIT_NUM_GLOBAL_EXPERTS", "385"
        )  # not divisible by 64
        monkeypatch.setenv("KIMI_REFIT_EXPERT_PARALLEL_SIZE", "64")
        worker = _bare_worker(rank=3)
        assert (
            worker._should_skip_expert_for_ep_rank_refit(
                "model.layers.0.mlp.experts.300.gate_proj.weight"
            )
            is False
        )


class TestShouldSkipRefitParamName:
    def test_skip_all(self, monkeypatch):
        monkeypatch.setenv("KIMI_SKIP_ALL_REFIT_FOR_SMOKE", "true")
        worker = _bare_worker(rank=0)
        assert worker._should_skip_refit_param_name("model.embed_tokens.weight")
        assert worker._should_skip_refit_param_name(
            "model.layers.0.mlp.experts.0.gate_proj.weight"
        )

    def test_skip_expert_only(self, monkeypatch):
        monkeypatch.delenv("KIMI_SKIP_ALL_REFIT_FOR_SMOKE", raising=False)
        monkeypatch.delenv("KIMI_SKIP_NONEXPERT_REFIT_FOR_SMOKE", raising=False)
        monkeypatch.delenv(
            "KIMI_FILTER_EXPERT_REFIT_BY_EP_RANK_FOR_SMOKE", raising=False
        )
        monkeypatch.setenv("KIMI_SKIP_EXPERT_REFIT_FOR_SMOKE", "true")
        worker = _bare_worker(rank=0)
        assert worker._should_skip_refit_param_name(
            "model.layers.0.mlp.experts.0.gate_proj.weight"
        )
        assert not worker._should_skip_refit_param_name("model.embed_tokens.weight")


class TestHfArchitecturesOverride:
    def test_get_hf_architectures_override_normalizes(self):
        from nemo_rl.models.megatron.setup import _get_hf_architectures_override

        assert _get_hf_architectures_override({}) == []
        assert _get_hf_architectures_override({"architectures": None}) == []
        assert _get_hf_architectures_override({"architectures": "KimiForCausalLM"}) == [
            "KimiForCausalLM"
        ]
        assert _get_hf_architectures_override(
            {"architectures": ["KimiForCausalLM"]}
        ) == ["KimiForCausalLM"]

    def test_apply_override_only_fills_empty_architectures(self):
        from nemo_rl.models.megatron.setup import (
            _apply_hf_config_architectures_override,
        )

        # Nested config-like object with an empty architectures attribute.
        inner = SimpleNamespace(architectures=None, model_type="kimi")
        root = SimpleNamespace(architectures=None, config=inner, model_type="kimi")
        _apply_hf_config_architectures_override(
            root, {"architectures": ["KimiForCausalLM"]}, context="test"
        )
        assert root.architectures == ["KimiForCausalLM"]
        assert inner.architectures == ["KimiForCausalLM"]

    def test_apply_override_preserves_existing(self):
        from nemo_rl.models.megatron.setup import (
            _apply_hf_config_architectures_override,
        )

        root = SimpleNamespace(architectures=["LlamaForCausalLM"], model_type="llama")
        _apply_hf_config_architectures_override(
            root, {"architectures": ["KimiForCausalLM"]}, context="test"
        )
        assert root.architectures == ["LlamaForCausalLM"]

    def test_apply_override_noop_without_config(self):
        from nemo_rl.models.megatron.setup import (
            _apply_hf_config_architectures_override,
        )

        root = SimpleNamespace(architectures=None, model_type="kimi")
        _apply_hf_config_architectures_override(root, {}, context="test")
        assert root.architectures is None
