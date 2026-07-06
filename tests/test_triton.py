"""Triton cache discovery + metadata pairing."""

import json

from cuxray.triton import discover


def _kernel(d, name, meta=None):
    (d / f"{name}.cubin").write_bytes(b"\x7fELF")
    if meta is not None:
        (d / f"{name}.json").write_text(json.dumps(meta))


def test_pairs_cubin_with_metadata(tmp_path):
    sub = tmp_path / "AB12"; sub.mkdir()
    _kernel(sub, "triton_red_fused_native_layer_norm_0",
            {"shared": 192, "num_warps": 16, "warp_size": 32, "arch": 80})
    ks = discover(tmp_path)
    assert len(ks) == 1
    k = ks[0]
    assert k.name == "triton_red_fused_native_layer_norm_0"
    assert k.shared == 192
    assert k.num_warps == 16
    assert k.threads == 512          # num_warps * warp_size
    assert k.arch == "sm80"


def test_arch_from_target_dict(tmp_path):
    _kernel(tmp_path, "k", {"target": {"arch": 90}, "num_warps": 4})
    k = discover(tmp_path)[0]
    assert k.arch == "sm90"
    assert k.threads == 128          # default warp_size 32


def test_missing_metadata_is_tolerated(tmp_path):
    _kernel(tmp_path, "bare", meta=None)
    k = discover(tmp_path)[0]
    assert k.name == "bare"
    assert k.shared is None and k.threads is None


def test_group_index_files_ignored(tmp_path):
    _kernel(tmp_path, "k", {"shared": 0, "num_warps": 8})
    (tmp_path / "__grp__k.json").write_text("{}")  # not a cubin, must be skipped
    ks = discover(tmp_path)
    assert len(ks) == 1 and ks[0].name == "k"


def test_single_cubin_path(tmp_path):
    _kernel(tmp_path, "solo", {"shared": 1024, "num_warps": 2, "warp_size": 32})
    ks = discover(tmp_path / "solo.cubin")
    assert len(ks) == 1 and ks[0].shared == 1024 and ks[0].threads == 64
