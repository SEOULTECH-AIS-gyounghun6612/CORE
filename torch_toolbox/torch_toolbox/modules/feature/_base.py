"""도메인 descriptor 의 공통 계약 — 각 도메인이 자기 native feature 를 self-describe 한다.

**``KIND`` 가 순서 유무를 가른다** (``Feature_Spec.axis`` 는 정규화 종류라 이것과 별개다):

- :data:`FEATURE` — 순서 없는 scalar 묶음 (size·ratio·moment·area …).
- :data:`TOKEN`   — 순서 있는 sequence (radial_rle 의 θ 프로파일). θ 축이 **순환**이라
  극좌표로 그려야 형상이 보인다.

도메인은 **native shape 그대로** 낸다(병합·패딩 없음). 하나로 병합하면 이 구분이 사라지고 스케일
큰 도메인이 거리를 지배한다. 조립이 필요한 소비처(학습 헤더 등)가 스스로 붙인다 — 조립은 도메인
생성의 책임이 아니다.

거리에 대하여
-------------
**두 KIND 모두 유클리드 거리로 본다.** TOKEN 이라고 회전정합(circular cross-correlation) 거리를
쓰면 안 된다 — θ 축은 :class:`~torch_toolbox.modules.transform.mask.polar.Polar_Raster` 가
:class:`~torch_toolbox.modules.transform.mask.canonical.Centroid_Frame` 의 주축각만큼 이미 굴려
**정렬을 마친 상태**로 나오기 때문이다. 여기서 순환 정합을 다시 걸면 그 정렬을 통째로 버린다.

정렬은 실측상 신뢰할 만하다 — 실데이터 1400 마스크 x 8 회전 자기일관성 **99.79%**, 실패는 전량
180° 이분법이다. 남는 소수 표본은 ``Frame.flip_margin`` 이 낮은 쪽에 몰리므로, 회전 불변 거리
(:mod:`torch_toolbox.metric.functional.vector.circular`)는 **대체재가 아니라**

- 정렬 방식을 바꿀 때의 회귀 측정 도구
- ``flip_margin`` 이 낮은 표본의 대비책

으로만 쓴다.

Note:
    같은 클래스 표본끼리의 불일치율은 정렬 실패율보다 훨씬 크게 관측된다(실측 6%+ 대 0.2%).
    그 차이는 정렬이 아니라 **객체 분리 품질**에서 온다 — 이웃 조각이 붙거나 가려져 잘린
    마스크는 형상 자체가 다르다. 거리 방식을 바꿔 해결할 수 있는 문제가 아니다.

현재 배선
---------
``KIND`` 를 실제로 소비하는 곳은 **시각화뿐**이다(어느 도메인을 극좌표 밴드로 그릴지). cohort
판정은 도메인별 유클리드 거리²의 합을 쓰며 ``KIND`` 를 보지 않는다 — 위 이유로 그게 맞다.

각 descriptor 클래스는 ``KIND`` 클래스 속성으로 자기 성질을 밝히고, ``Spec()`` 으로 차원·정규화를
밝힌다(:class:`~torch_toolbox.modules.transform.mask.geometry.spec.Feature_Spec`).
도메인별 native feature dict 를 만드는 것은 지금
:meth:`~torch_toolbox.modules.transform.mask.geometry.Geometry_Embedding.Features` 다 —
이 패키지에 registry 는 아직 없다.
"""
from __future__ import annotations

#: 순서 없는 scalar 도메인.
FEATURE = "feature"
#: 순서 있는 순환 sequence 도메인(θ 프로파일 등). **이미 정렬돼 나온다** — 거리는 유클리드.
TOKEN = "token"
