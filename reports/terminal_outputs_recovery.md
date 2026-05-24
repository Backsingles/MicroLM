
## D1 preflight train_qwen_lora script scan - 2026-05-20 21:04:09 +08:00

```powershell
rg -n "TrainingArguments|save_steps|eval_steps|gradient|load_in|device|resume|from_pretrained|Peft|Lora|SFT|Trainer|num_train_epochs|max_steps|batch" scripts\train_qwen_lora.py
```

```text
21:from peft import LoraConfig, get_peft_model, PeftModel
113:def collate_fn(batch):
114:    """Pad sequences to the same length within a batch."""
115:    input_ids_list, labels_list = zip(*batch)
145:def evaluate(model, loader, device):
151:            input_ids = input_ids.to(device)
152:            labels = labels.to(device)
153:            attention_mask = attention_mask.to(device)
188:    device = cfg["training"]["device"]
198:    tokenizer = AutoTokenizer.from_pretrained(
206:    model = AutoModelForCausalLM.from_pretrained(
210:        device_map=device,
215:    peft_config = LoraConfig(
246:    batch_size = cfg["training"]["batch_size"]
248:        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0
251:        valid_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
262:    grad_accum = cfg["training"].get("gradient_accumulation_steps", 1)
263:    max_steps = cfg["training"]["max_steps"]
281:    print(f"\nStarting training: {max_steps} steps, grad_accum={grad_accum}")
283:    print(f"  Effective batch size: {batch_size * grad_accum}")
284:    print(f"  Device: {device}, FP16: {cfg['training'].get('fp16', False)}")
289:    for step in range(max_steps):
290:        # Accumulate gradients
301:            input_ids = input_ids.to(device)
302:            labels = labels.to(device)
303:            attention_mask = attention_mask.to(device)
319:        if completed_step % eval_interval == 0 or completed_step == max_steps:
320:            val_loss = evaluate(model, valid_loader, device)
323:                f"Step {completed_step}/{max_steps} | "
343:        if completed_step % save_interval == 0 or completed_step == max_steps:

```
