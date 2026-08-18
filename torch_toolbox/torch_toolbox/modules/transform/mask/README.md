# transform/mask

실루엣 이진 마스크 → 형상 표현. **torch 단일 소스** — 학습·추론·배포·평가가 같은 그래프를
쓰므로 "같은 연산"이 라이브러리 버전 고정이 아니라 **그래프 구조**로 보장된다. 그래서 전처리를
export 그래프에서 떼지 않고 GPU 에서 함께 최적화한다.

모든 모듈은 레지스트리 등록 `Trainable_Model`(object_type) + `_Config`(config_type)이라
`Build_from_registry`(`modules/build.py`)로 **직접 조립**된다. 인자·반환은 각 심볼의 docstring 이
소유한다 — 여기엔 **역할과 심볼 사이 개념**만 둔다.

## 설계 규율 (왜 이렇게 짰나)

- **해석적이라 raster 를 안 돌린다.** 회전은 좌표(Frame)로 소비하고, 극좌표는 정수 원점 덕에
  `GridSample` 없이 gather/mul/add 로 끝난다 → TRT INT8 이 살아 있고 리샘플 손실이 없다.
- **방위는 harmonic 차수로 고정한다.** 축 자체는 k=2 (`Z2 = Σ m·r²·e^{i2θ}`)가 정하고 u 를
  major 로 둔다 — 여기까지는 π 주기라 180° 가 남는다. 그 한 비트는 **홀수** harmonic k=3
  (`Z3`)의 위상으로만 가를 수 있다(짝수는 π 회전에 불변, 원점 이동도 평행축 정리상 무력).
  `Z3` 의 어느 위상 성분(0°/90°)을 볼지는 **부품별 표**가 정한다 — 프레임마다 값을 보고 고르면
  경계에 걸린 형상이 회전마다 선택을 뒤집어 더 나빠진다(실측).
- **그래서 이 정렬은 `정준(canonical)` 이 아니다.** 대칭 형상은 각도가 원리적으로 미결정이다
  (n≥3 회전대칭 `Z2=0`, 2회 대칭 `Z3=0`). 두 harmonic 의 상대 크기가 그대로 신뢰도
  `anisotropy` / `flip_margin` 이라, 정렬을 믿을지 회전 불변 거리로 내려갈지는 소비처가 그 값으로 판단한다(`Centroid_Frame` ↔ `Region_Scalars`
  한 벌 — 후자는 major/minor 를 축 이름이 아니라 고유값 크기순으로 뽑아 안전측에 둔다).
- **입력 캔버스는 자유다 — 크기 인자는 전부 다른 뜻이다.** 모든 모듈이 입력 H·W 를 `forward`
  에서 읽으므로 앞단 crop 이 필수가 아니다(원본 프레임을 그대로 넣어도 된다). 그래서 config 의
  크기 인자는 이름으로 역할을 가른다: `sampling_size` = 샘플 반경 기본값·길이 단위·Spec 상한을
  뽑는 **기준 캔버스**(입력이 아니다), `output_size`/`target_size`/`crop_size` = **출력 캔버스**.
  `Frame_Coords` 와 `Region_Scalars` 의 `sampling_size` 는 같아야 한다 — 전자가 나눈 길이 단위를
  후자가 곱해 px 로 되돌린다.
- **차원은 하드코딩하지 않는다.** 각 서술자가 `Feature_Spec` 으로 자기 차원·범위를 선언하고
  조립체가 합산한다 → 설정을 바꿔도 슬라이스가 조용히 어긋나지 않는다.
- **구멍은 형상 정보다.** fill / 최대연결성분 / convex hull 을 쓰지 않는다(실측상 표본의 56%가
  관통 구멍). 입력 마스크는 정렬·fill 하지 **않은** 이진이어야 한다.
- **정렬 자유도 제거는 결정적이라 여기 산다.** 무작위 회전 증강(학습 전용)은 dataloader 소유.

## 파이프라인 (조립 순서)

```
원본 마스크
  └─ [전단] silhouette.py   Center_Crop → Resize_Binarize     정렬 전 {0,1} 실루엣
       ├─ geometry 분기 ──────────────────────────────────────────────────┐
       │   canonical.Centroid_Frame → Frame_Coords   (정준 좌표계 u,v)      │
       │   polar.Polar_Raster                        (극좌표 occupancy)     │
       │   occupancy.Radial_Profile / Occupancy      (θ별 반경·총량)        │
       │   geometry/ Region_Scalars·Chirality_Moments·Profile_Stats·        │
       │             Fourier_Descriptor              (서술자)               │
       │   geometry/spec Normalizer                  (정규화)  → (B, FEAT_DIM)
       │   geometry/__init__ Geometry_Embedding      = 위 전부의 조립체 ────┘
       └─ image 분기 (백본 입력)
           image.Align_Raster → Image_Channels       (주축 정렬 + mask/blur/edge 채널)
```

## 역할 지도

**전단 — `silhouette.py`**
- `Center_Crop` [center_crop] — 중심점 기준 정사각 창(경계 밖 0). numpy `Crop_centered` 등가.
- `Resize_Binarize` [resize_binarize] — target 캔버스 bilinear 리샘플 + 재이진화. numpy `resize_and_binarize` 등가.

**정준 좌표계 — `canonical.py`**
- `Centroid_Frame` [frame] — centroid + 주축각. **파라미터만, raster 안 돌림.**
- `Frame_Coords` [coords] — 주축 정렬 무차원 좌표 `(u, v)`.

**극좌표 — `polar.py`**
- `Polar_Raster` [polar] — 마스크 → `(NR, NT)` occupancy 분수. backward gather.

**프로파일·총량 — `occupancy.py`**
- `Radial_Profile` [radial] — 극좌표 → θ별 `(r_outer, r_inner, coverage)`.
- `Occupancy` [occupancy_totals] — 직교/극좌표 면적과 그 비(재료 중심몰림 신호).

**형상 서술자 — `geometry/`**
- `Region_Scalars` [region_scalars] — 크기·비율·위치 스칼라. skimage regionprops 대체.
- `Chirality_Moments` [chirality_moments] — 좌우/상하 비대칭 + 3차 모멘트. FP16 안전.
- `Profile_Stats` [profile_stats] — 주기 프로파일 통계(평균·표준편차·백분위).
- `Fourier_Descriptor` [fourier_descriptor] — Fourier 기술자. 위상정규화로 회전 불변.

**정규화 계약 — `geometry/spec.py`**
- `Feature_Spec`·`Apply_transform`·`Normalizer` — 그룹 선언 + 자릿수 압축 + export 직전 통계 교체.

**조립체 — `geometry/__init__.py`**
- `Geometry_Embedding` [geometry_embedding] — 위 geometry 분기 전부를 sub-module 로 Build →
  마스크 → `(B, FEAT_DIM)`. `Forward_with_aux` 로 `r_outer`·`frame_angle` 도 함께 낸다.

**image 분기 — `image.py`** (DINOv2 백본 입력)
- `Align_Raster` [align_raster] — `Frame.angle` 만큼 회전 정렬(정준 자세는 아니다 — 위 설계 규율).
  **수동 bilinear**(GridSample/warpAffine 아님).
- `Image_Channels` [image_channels] — `mask / blur{k} / edge` 채널 스택.

## 남은 것

- **전처리 사슬은 용접하지 않고 config 로 공유한다.** `Center_Crop`/`Resize_Binarize`/
  `Geometry_Embedding` 을 하나의 Module 로 묶지 **않는다** — 용접하면 특정 조합이 객체로 고정된다.
  대신 세 모듈 config 를 **공유 config 파일**에 선언하고, 소비처(학습 dataset·평가·추론)가 그
  파일을 읽어 `Build_from_registry`(`modules/build.py`)로 각자 빌드해 순서 적용한다. 단일 출처는
  고정 객체가 아니라 **데이터(config)** 다. config 파일의 거처·수명은 소비처가 정한다(소비처 TODO).
