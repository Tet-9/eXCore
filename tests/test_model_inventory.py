from collections import Counter

from excore.model.qwen38_cpu import ATTN, GDN, LM_HEAD, MLP, build_inventory


def test_unit_counts(spec):
    units = build_inventory(spec)
    kinds = Counter(u.kind for u in units)
    assert len(units) == 273
    assert kinds == {GDN: 144, ATTN: 64, MLP: 64, LM_HEAD: 1}


def test_layer_layout(spec):
    assert len(spec.gdn_layers) == 48
    assert len(spec.attention_layers) == 16
    assert spec.attention_layers[:3] == (3, 7, 11)
    assert spec.layer_kind(63) == ATTN and spec.layer_kind(0) == GDN


def test_names_unique_and_stable(spec):
    names = [u.name for u in build_inventory(spec)]
    assert len(set(names)) == len(names)
    assert names[0] == "L0.gdn.qkv" and names[-1] == "lm_head"
    assert "L3.attn.q" in names and "L12.mlp" in names


def test_total_params_near_27b(spec):
    total = sum(u.params for u in build_inventory(spec))
    assert 24e9 < total < 28e9


def test_all_rows_block_aligned_for_k_quants(spec):
    for u in build_inventory(spec):
        for t in u.tensors:
            assert t.in_features % 256 == 0, t.gguf_name
