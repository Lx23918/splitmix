import torch
import torch.nn.functional as F


def compute_retrieval_metrics(
    eeg_image_features: torch.Tensor,
    eeg_text_features: torch.Tensor,
    eeg_depth_features: torch.Tensor,
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    depth_features: torch.Tensor,
) -> dict[str, float]:
    eeg_image_features = F.normalize(eeg_image_features, dim=-1)
    eeg_text_features = F.normalize(eeg_text_features, dim=-1)
    eeg_depth_features = F.normalize(eeg_depth_features, dim=-1)
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    depth_features = F.normalize(depth_features, dim=-1)

    image_scores = eeg_image_features @ image_features.T
    text_scores = eeg_text_features @ text_features.T
    depth_scores = eeg_depth_features @ depth_features.T
    targets = torch.arange(image_scores.size(0), device=image_scores.device)

    top1 = {"image": 0, "text": 0, "depth": 0, "all": 0}
    top5 = {"image": 0, "text": 0, "depth": 0, "all": 0}

    for index in range(targets.numel()):
        image_top5 = torch.topk(image_scores[index], k=min(5, image_scores.size(1)), largest=True).indices.tolist()
        text_top5 = torch.topk(text_scores[index], k=min(5, text_scores.size(1)), largest=True).indices.tolist()
        depth_top5 = torch.topk(depth_scores[index], k=min(5, depth_scores.size(1)), largest=True).indices.tolist()

        image_top1 = image_top5[0]
        text_top1 = text_top5[0]
        depth_top1 = depth_top5[0]
        target = int(targets[index].item())

        image_top1_correct = image_top1 == target
        text_top1_correct = text_top1 == target
        depth_top1_correct = depth_top1 == target

        image_top5_correct = target in image_top5
        text_top5_correct = target in text_top5
        depth_top5_correct = target in depth_top5

        if image_top1_correct:
            top1["image"] += 1
        if text_top1_correct:
            top1["text"] += 1
        if depth_top1_correct:
            top1["depth"] += 1
        if image_top1_correct or text_top1_correct or depth_top1_correct:
            top1["all"] += 1

        if image_top5_correct:
            top5["image"] += 1
        if text_top5_correct:
            top5["text"] += 1
        if depth_top5_correct:
            top5["depth"] += 1
        if image_top5_correct or text_top5_correct or depth_top5_correct:
            top5["all"] += 1

    total = float(targets.numel())
    return {
        "top1_image": top1["image"] / total,
        "top1_text": top1["text"] / total,
        "top1_depth": top1["depth"] / total,
        "top1_all": top1["all"] / total,
        "top5_image": top5["image"] / total,
        "top5_text": top5["text"] / total,
        "top5_depth": top5["depth"] / total,
        "top5_all": top5["all"] / total,
    }
