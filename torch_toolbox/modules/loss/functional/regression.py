import torch

def relative_error_by_dot(
    predict: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0
) -> torch.Tensor:
    """ ### 벡터 간 내적 기반 상대 오차를 계산하는 함수

    입력된 두 벡터를 정규화한 뒤, 내적(dot product)을 통해 각도 기반
    오차를 계산합니다. 결과는 입력된 스케일(alpha) 값만큼 조정됩니다.

    ------------------------------------------------------------------
    ### Args
    - predict: 예측 벡터 텐서 (torch.Tensor)
    - target: 정답 벡터 텐서 (torch.Tensor)
    - alpha: 결과 오차에 곱할 스케일 인자 (기본값 = 1.0)

    ### Returns
    - torch.Tensor: 각도 기반 상대 오차 (radian 단위)
    """
    _pre = predict / (predict.norm(dim=-1, keepdim=True) + 1e-8)
    _tgt = target / (target.norm(dim=-1, keepdim=True) + 1e-8)
    _dot = torch.sum(_pre * _tgt, dim=-1).abs().clamp(min = 0.0, max=1.0)
    return alpha * torch.acos(_dot)

def relative_error_by_mse(
    predict: torch.Tensor,  # 3 = tx, ty, tz
    target: torch.Tensor
) -> torch.Tensor:
    _diff = predict - target
    return torch.norm(_diff, dim=-1)

def relative_rotation_from_matrix(
    predict: torch.Tensor,  # matrix -> n, 3, 3
    target: torch.Tensor
) -> torch.Tensor:
    # TODO: This function is not tested.
    # Please write and run appropriate tests.
    _diff = torch.matmul(predict.transpose(-2, -1), target)
    _trace = _diff[:, 0, 0] + _diff[:, 1, 1] + _diff[:, 2, 2]
    _theta = ((_trace - 1) / 2).clamp(-1, 1)
    return torch.acos(_theta)
