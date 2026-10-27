"""Visual LoRA compatibility, forward parity, and learning through the scanned encoder."""

import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest

from openpi.models import lora
from openpi.models import model as model_lib
from openpi.models import siglip
from openpi.training import weight_loaders


@pytest.fixture(autouse=True)
def _cpu_device():
    # These small numerical and checkpoint tests don't need an accelerator.
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _tiny_encoder(*, scan, dtype="float32", lora_config=None):
    return siglip.Module(
        num_classes=8,
        patch_size=(4, 4),
        width=16,
        depth=2,
        mlp_dim=32,
        num_heads=2,
        pool_type="none",
        head_zeroinit=False,
        scan=scan,
        dtype_mm=dtype,
        lora_config=lora_config,
    )


@pytest.mark.parametrize("scan", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_visual_lora_loads_base_checkpoint_with_identical_initial_output(monkeypatch, scan, dtype):
    images = jax.random.normal(jax.random.key(1), (1, 8, 8, 3))
    base = _tiny_encoder(scan=scan, dtype=dtype)
    adapted = _tiny_encoder(scan=scan, dtype=dtype, lora_config=lora.LoRAConfig(rank=2, alpha=2.0))
    base_params = base.init(jax.random.key(0), images)["params"]
    template = adapted.init(jax.random.key(0), images)["params"]

    # Exercise the training weight loader: only the newly added LoRA leaves may be absent.
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(model_lib, "restore_params", lambda *args, **kwargs: base_params)
    loaded = weight_loaders.CheckpointWeightLoader("base").load(template)
    flat_base = traverse_util.flatten_dict(base_params)
    flat_loaded = traverse_util.flatten_dict(loaded)
    added = flat_loaded.keys() - flat_base.keys()
    assert added
    assert all("lora" in "/".join(path) for path in added)
    assert {path[-3] for path in added} == {"query", "key", "value", "out"}
    for path, value in flat_base.items():
        np.testing.assert_array_equal(flat_loaded[path], value)
    for path in added:
        if path[-1] == "lora_b":
            np.testing.assert_array_equal(flat_loaded[path], 0)
    apply = jax.jit(lambda params: adapted.apply({"params": params}, images)[0])
    expected = jax.jit(lambda params: base.apply({"params": params}, images)[0])(base_params)
    np.testing.assert_array_equal(apply(loaded), expected)
    assert apply(loaded).dtype == jnp.dtype(dtype)


def test_scanned_visual_lora_learns_and_restores_without_updating_backbone(tmp_path):
    model = _tiny_encoder(scan=True, lora_config=lora.LoRAConfig(rank=2, alpha=2.0))
    images = jax.random.normal(jax.random.key(1), (1, 8, 8, 3))
    params = model.init(jax.random.key(0), images)["params"]

    def loss(params):
        return jnp.mean(jnp.square(model.apply({"params": params}, images)[0] - 0.5))

    value_and_grad = jax.jit(jax.value_and_grad(loss))
    initial_loss, grads = value_and_grad(params)
    initial = traverse_util.flatten_dict(params)
    flat_grads = traverse_util.flatten_dict(grads)
    adapters = {path for path in initial if "lora" in "/".join(path)}
    b_paths = {path for path in adapters if path[-1] == "lora_b"}
    assert b_paths
    assert all(np.any(flat_grads[path] != 0) for path in b_paths)
    updated = traverse_util.unflatten_dict(
        {path: value - 0.01 * flat_grads[path] if path in adapters else value for path, value in initial.items()}
    )
    updated_loss, second_grads = value_and_grad(updated)
    assert updated_loss < initial_loss
    flat_second_grads = traverse_util.flatten_dict(second_grads)
    assert all(np.any(flat_second_grads[path] != 0) for path in adapters if path[-1] == "lora_a")
    for path, value in traverse_util.flatten_dict(updated).items():
        if path not in adapters:
            np.testing.assert_array_equal(value, initial[path])

    # Save a real checkpoint and restore through the same reader used by policy deployment.
    checkpoint = tmp_path / "params"
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(checkpoint, {"params": updated})
    restored = model_lib.restore_params(checkpoint, restore_type=np.ndarray)
    restored_flat = traverse_util.flatten_dict(restored)
    for path, value in traverse_util.flatten_dict(updated).items():
        np.testing.assert_array_equal(restored_flat[path], value)
    apply = jax.jit(lambda params: model.apply({"params": params}, images)[0])
    np.testing.assert_array_equal(apply(restored), apply(updated))
