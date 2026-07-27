"""도메인 descriptor 의 공통 계약 — 각 도메인이 자기 native feature 를 self-describe 한다.

**``KIND`` 가 순서 유무를 가른다** (``Feature_Spec.axis`` 는 정규화 종류라 이것과 별개다):

- :data:`FEATURE` — 순서 없는 scalar 묶음 (size·ratio·moment·area …). 소비처는 유클리드 거리로 본다.
- :data:`TOKEN`   — 순서 있는 sequence (radial_rle 의 θ 프로파일). 회전 = 순환 shift 라, 소비처는
  회전정합(circular cross-correlation) 거리로 본다.

검증(cohort)이 이 ``KIND`` 로 **도메인마다** 거리·클러스터 방식을 고른다. 하나로 병합하면 이 구분이
사라지고 스케일 큰 도메인이 거리를 지배하므로, 도메인은 **native shape 그대로** 낸다(병합·패딩 없음).
조립이 필요한 소비처(학습 헤더 등)가 스스로 붙인다 — 조립은 도메인 생성의 책임이 아니다.

각 descriptor 클래스는 ``KIND`` 클래스 속성으로 자기 성질을 밝히고, ``Spec()`` 으로 차원·정규화를
밝힌다(:class:`spec.Feature_Spec`). registry(:mod:`feature`)가 그걸 모아 ``Features(mask)`` 를 만든다.
"""
from __future__ import annotations

#: 순서 없는 scalar 도메인 — 유클리드 거리로 비교한다.
FEATURE = "feature"
#: 순서 있는 sequence 도메인(θ 프로파일 등) — 회전(순환 shift) 정합 거리로 비교한다.
TOKEN = "token"
