"""Run the model, LoRA, and optimizer checks required before GRPO training."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from .train_grpo import DEFAULT_MODEL, _load_model_and_tokenizer
from .training import select_lora_target_suffixes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model, tokenizer, torch, device, revision = _load_model_and_tokenizer(
        args.model, args.revision, args.device
    )
    base_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    module_names = [name for name, _ in model.named_modules()]
    target_modules = select_lora_target_suffixes(module_names)
    inputs = tokenizer("Numbers: 8, 2, 3\nTarget: 7\nAnswer:", return_tensors="pt")
    inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.inference_mode():
        forward = model(**inputs)
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=1,
            pad_token_id=tokenizer.eos_token_id,
        )

    from peft import LoraConfig, get_peft_model

    lora_model = get_peft_model(
        model,
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        ),
    )
    trainable = [parameter for parameter in lora_model.parameters() if parameter.requires_grad]
    before = [parameter.detach().clone() for parameter in trainable]
    loss = lora_model(**inputs, labels=inputs["input_ids"]).loss
    loss.backward()
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    optimizer.step()
    changed = any(not torch.equal(old, new) for old, new in zip(before, trainable, strict=True))
    artifact = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model_id": args.model,
        "revision": revision,
        "device": str(device),
        "model_class": type(model).__name__,
        "base_parameter_count": base_parameter_count,
        "module_count": len(module_names),
        "lora_target_modules": target_modules,
        "matched_lora_modules": sum(
            any(name.endswith(target) for target in target_modules) for name in module_names
        ),
        "forward_logits_shape": list(forward.logits.shape),
        "one_token_generation": tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False
        ),
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "synthetic_backward_loss": float(loss.detach()),
        "adapter_weights_changed_after_optimizer_step": changed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
