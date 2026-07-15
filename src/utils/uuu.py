import os, yaml

def ensure_dirs(paths: dict):
    for k,p in paths.items():
        if p is not None and k.endswith("_dir"):
            os.makedirs(p, exist_ok=True)

def model_summary(model: torch.nn.Module) -> str:
    """
    =================================================================
    == Pretty Print of Model Parameter Summary (GitHUB.com/HooM4N) ==
    =================================================================
    """
    rows = []
    for name, m in model.named_modules():
        if name == "":
            continue
        trainable = sum(p.numel() for p in m.parameters(recurse=False) if p.requires_grad)
        frozen = sum(p.numel() for p in m.parameters(recurse=False) if not p.requires_grad)
        rows.append((
            name,
            f"{m.__class__.__name__}({m.extra_repr()})",
            trainable + frozen,
            trainable,
        ))

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    mod_w = max([6] + [len(r[0]) for r in rows])
    cls_w = max([5] + [len(r[1]) for r in rows])
    num_w = max(10, len(f"{total_params:,}"))
    width = mod_w + cls_w + num_w * 2 + 10

    lines = [
        "=" * width,
        f"Model Parameter Summary for {model.__class__.__name__}".center(width),
        "=" * width,
        f"{'Module':{mod_w}} | {'Class':{cls_w}} | {'Total':>{num_w}} | {'Trainable':>{num_w}}",
        "-" * width,
    ]

    lines += [
        f"{n:{mod_w}} | {c:{cls_w}} | {t:{num_w},} | {tr:{num_w},}"
        for n, c, t, tr in rows
    ]

    lines += [
        "-" * width,
        f"{'TOTAL':{mod_w}} | {'':{cls_w}} | {total_params:{num_w},} | {total_trainable:{num_w},}",
        "=" * width,
    ]

    return "\n".join(lines)