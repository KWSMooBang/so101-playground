#!/usr/bin/env python3
"""Merge the bench's FLUX 3 LoRA adapter into the released SO-101 package without LeRobot/PEFT.

Equivalent to so101_bench ``scripts/FLUX3_fine_tuning/pod/merge_lora_package.py`` (PEFT merge_and_unload):
for every ``lora_A``/``lora_B`` pair W <- W + (lora_alpha / r) * B @ A (fp32, then bf16), and the
``modules_to_save`` heads (saved as plain weights) replace the base tensors. The output package takes
config + processor JSONs from the base package and the normalization state (processor safetensors) from the
adapter, which carries so101_bench_sim's own statistics ("so101_ourstats", the LoRA run's base).

  merge_flux3_lora.py ADAPTER_DIR BASE_PACKAGE OUT_PACKAGE
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

PREFIX = "base_model.model."


def main() -> None:
    adapter, base, out = (Path(a) for a in sys.argv[1:4])
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    scale = float(cfg["lora_alpha"]) / float(cfg["r"])
    lora = load_file(str(adapter / "adapter_model.safetensors"))
    state = load_file(str(base / "model.safetensors"))
    merged, replaced = 0, 0
    for key, a in lora.items():
        if ".lora_A." in key:
            target = key.removeprefix(PREFIX).replace(".lora_A.weight", ".weight")
            b = lora[key.replace(".lora_A.", ".lora_B.")]
            w = state[target]
            state[target] = (w.float() + scale * (b.float() @ a.float())).to(w.dtype)
            merged += 1
        elif ".lora_B." in key:
            continue
        else:
            target = key.removeprefix(PREFIX)
            if target not in state or state[target].shape != a.shape:
                raise SystemExit(f"head tensor {target} missing or shape mismatch")
            state[target] = a.to(state[target].dtype)
            replaced += 1
    for head in cfg["modules_to_save"]:
        extra = [k for k in state if k.startswith(head + ".") and PREFIX + k not in lora]
        if extra:
            raise SystemExit(f"base has tensors of saved module {head} the adapter does not carry: {extra[:3]}")
    print(f"merged {merged} LoRA pairs (scale {scale}), replaced {replaced} head tensors")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for item in base.iterdir():
        if item.is_file() and item.name != "model.safetensors":
            shutil.copy2(item, out / item.name)
    for item in adapter.glob("policy_*processor_step_*.safetensors"):
        shutil.copy2(item, out / item.name)
    save_file({k: v.contiguous() for k, v in state.items()}, str(out / "model.safetensors"),
              metadata={"format": "pt", "merged_adapter": str(adapter)})
    print(f"wrote {out}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
