import torch
import torch.distributed as dist


def Reduce_metrics(
    metrics: dict[str, float], device: torch.device, world_size: int
) -> dict[str, float]:
    """전체 rank의 metrics를 평균으로 동기화.

    단일 GPU 또는 분산 미초기화 상태에서는 float 변환만 수행함.
    all_reduce 블로킹이 발생하므로 모든 rank에서 동일 시점에 호출해야 함.
    """
    if world_size <= 1 or not dist.is_initialized() or not metrics:
        return {k: float(v) for k, v in metrics.items()}

    _keys = list(metrics.keys())
    _values = []

    for k in _keys:
        v = metrics[k]
        if isinstance(v, (int, float)):
            _values.append(float(v))
        elif isinstance(v, torch.Tensor):
            _values.append(v.detach().item())
        else:
            raise ValueError(
                f"[ERROR] 지원하지 않는 메트릭 타입: {type(v)} for key {k}"
            )

    _tensor = torch.tensor(_values, dtype=torch.float32, device=device)
    dist.all_reduce(_tensor, op=dist.ReduceOp.SUM)
    _tensor /= world_size

    return {k: v for k, v in zip(_keys, _tensor.cpu().tolist())}
