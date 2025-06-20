import torch


class Regression():
    @staticmethod
    def Relative_error_by_dot(
        predict: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 1.0
    ):
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

    @staticmethod
    def Relative_error_by_mse(
        predict: torch.Tensor,  # 3 = tx, ty, tz
        target: torch.Tensor
    ):
        _diff = predict - target
        return torch.norm(_diff, dim=-1)

    @staticmethod
    def Relative_rotation_from_matrix(
        predict: torch.Tensor,  # matrix -> n, 3, 3
        target: torch.Tensor
    ):
        # TODO: This function is not tested.
        # Please write and run appropriate tests.
        _diff = torch.matmul(predict.transpose(-2, -1), target)
        _trace = _diff[:, 0, 0] + _diff[:, 1, 1] + _diff[:, 2, 2]
        _theta = ((_trace - 1) / 2).clamp(-1, 1)
        return torch.acos(_theta)


class Area_Under_Curve():
    @staticmethod
    def Get_accs(
        errors: torch.Tensor, max_threshold: float, step_ct: int,
        mask: torch.Tensor | None = None
    ):
        _ths = torch.linspace(
            0, max_threshold, step_ct + 1, device=errors.device)
        _acc_list = torch.stack([
            (
                (errors <= t) if mask is None else (errors <= t) * mask
            ).float().mean() for t in _ths
        ])
        return _acc_list, _ths

    @staticmethod
    def Cumpute(
        errors_list: list[torch.Tensor],
        max_thresholds: list[float],
        step_ct: int,
        mask: torch.Tensor | None = None
    ):
        assert len(errors_list) == len(max_thresholds)
        _auc_holder = []

        _this_e = errors_list[0]
        _this_th = max_thresholds[0]

        _list, _ths = Area_Under_Curve.Get_accs(
            _this_e, _this_th, step_ct, mask)

        _auc_holder.append(torch.trapz(_list, _ths) / _this_th)

        if len(errors_list) > 1:
            for _e, _th in zip(errors_list[1:], max_thresholds[1:]):
                _this_list, _ = Area_Under_Curve.Get_accs(
                    _e, _th, step_ct, mask)

                _auc_holder.append(torch.trapz(_this_list, _ths) / _this_th)

                # min
                _mask = _this_list < _list
                _list[_mask] = _this_list[_mask]

        return torch.trapz(_list, _ths) / _this_th, _auc_holder
