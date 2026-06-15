# ROADMAP

> 최근 갱신: 2026-05-05

`spatial_toolbox`의 다음 목표는 "기능을 더 붙이는 라이브러리"가 아니라, `scene`, `render`, `simulation`을 명확한 계약 위에서 조합 가능한 코어로 고정하는 것임.

이 문서는 완료된 작업 목록이 아니라, 앞으로 어디를 어떤 순서로 정리할지에 대한 방향 문서임.

---

## 1. 목표

`spatial_toolbox`를 다음 성격의 라이브러리로 가져간다.

- `scene`: 런타임과 persistence가 분리된 순수 장면 모델 계층
- `render`: backend 차이를 숨기되 capability 차이는 명시하는 렌더 계약 계층
- `simulation`: 렌더 결과를 데이터셋으로 전환하는 조립 계층

핵심 방향은 세 가지다.

- 저장되는 데이터는 schema로 고정한다.
- backend 차이는 runtime contract와 persisted contract로 분리한다.
- 상위 애플리케이션이 기대할 수 있는 public API를 좁고 명확하게 유지한다.

---

## 2. 가장 중요한 방향

### A. metadata를 runtime object가 아니라 schema로 다룬다

지금 가장 큰 방향성은 여기에 있다.

현재 segmentation metadata는 backend가 scene node 객체를 직접 반환하고, exporter가 이를 나중에 JSON-friendly 구조로 억지 변환하는 형태다. 이건 임시 방어 로직으로는 동작하지만, 코어 계약으로 보기에는 맞지 않는다.

앞으로의 방향:

- channel metadata는 처음부터 persistence schema를 목표로 설계한다.
- `Base_Node` 같은 런타임 객체는 metadata 저장 경계 밖으로 밀어낸다.
- exporter는 임의 object를 정리하는 계층이 아니라, schema를 조립하는 계층이 된다.

즉, `_make_serializable()` 같은 함수는 최종 구조가 아니라 과도기적 흔적으로 본다.

### B. backend 추상화는 "동일함"보다 "차이의 명시"를 우선한다

OpenGL과 Blender는 모두 `Render(scene, camera_labels, request)` 계약을 따르지만, 실제 metadata와 context 전략은 다르다. 앞으로는 이 차이를 숨기기보다 명시하는 방향이 맞다.

앞으로의 방향:

- 공통 인터페이스는 유지한다.
- backend capability는 문서와 타입 계약에서 더 분명히 드러낸다.
- segmentation, depth, normal 같은 채널의 결과 포맷과 metadata 차이를 runtime 단계와 persisted 단계로 나눠 설명한다.

### C. simulation은 backend 호출기가 아니라 dataset contract 계층이 된다

`simulation`의 역할은 단순히 여러 번 Render를 호출하는 것이 아니라, 샘플링 규약, naming 규약, output schema를 정의하는 데 있다.

앞으로의 방향:

- output 디렉토리 구조를 더 명시적 규약으로 다듬는다.
- multi-camera, object grouping, sample naming 같은 데이터셋 관점의 기능을 확대한다.
- render backend 차이를 simulation output schema에서 흡수한다.

---

## 3. 우선순위

### 1순위: metadata schema 정리

이건 현재 구조에서 가장 중요한 리팩토링이다.

해야 할 일:

- `Render_Result.metadata`의 의미를 다시 정의
- channel별 metadata schema 정의
- `Node_Ref` 같은 persistence용 참조 스키마 도입
- exporter에서 ad-hoc serialization 제거
- segmentation metadata를 backend 공통 저장 구조로 수렴

완료 기준:

- exporter가 runtime object를 알지 못함
- 저장되는 metadata가 전부 schema 기반 dict로만 구성됨
- OpenGL/Blender 결과를 같은 저장 규약으로 설명 가능함

### 2순위: render capability 모델 정리

현재 `Renderer` 계약은 최소한의 호출 규약만 제공한다. 앞으로는 어떤 backend가 무엇을 보장하는지 더 명시해야 한다.

해야 할 일:

- backend별 지원 채널 명시
- context 전략 명시
- metadata 차이 명시
- unsupported behavior와 오류 메시지 정리

완료 기준:

- 상위 호출자가 backend 선택 시 기대 가능한 행동을 명확히 알 수 있음

### 3순위: simulation output contract 정리

데이터셋 생성 계층으로서의 `simulation`을 더 분명히 만든다.

해야 할 일:

- output layout 규약 확장
- rigid-body settle 같은 pre-capture stage를 simulation contract 안으로 편입
- physics preset과 per-scene override 구조 정의
- multi-camera 결과 구조 정리
- sample/object naming 강화
- metadata version 도입 검토

완료 기준:

- output 폴더만 보고도 샘플 단위, 카메라 단위, 객체 단위를 기계적으로 복원 가능함
- physics-driven capture도 config schema만으로 재현 가능함

### 4순위: public API 좁히기

코어가 커질수록 외부에서 기대하는 진입점은 오히려 줄여야 한다.

해야 할 일:

- `scene`, `render`, `simulation`의 공식 진입점 재확정
- 오래된 helper 패턴 제거
- 테스트와 문서 예제를 public API 기준으로만 유지

완료 기준:

- 상위 프로젝트가 비공식 내부 모듈에 덜 의존함

---

## 4. 모듈별 진행 방향

### scene

`scene`은 계속해서 가장 보수적인 계층이어야 한다.

방향:

- node / asset / file 계층 분리를 유지한다.
- runtime graph와 saved graph의 roundtrip 안정성을 높인다.
- geometry 포맷 확장보다 scene contract 안정성을 우선한다.

### render

`render`는 backend 구현체 묶음이 아니라 명시적 렌더 계약 계층으로 발전해야 한다.

방향:

- pass 결과와 metadata의 표준화에 집중한다.
- OpenGL/Blender 차이를 capability로 드러낸다.
- 향후 backend가 추가되어도 `render/core` 계약이 흔들리지 않게 한다.

### simulation

`simulation`은 가장 제품적인 계층이다.

방향:

- 데이터셋 생성 규약을 더 풍부하게 한다.
- 랜덤 pose 샘플링과 physics settle을 같은 capture pipeline 안에서 조합 가능하게 만든다.
- physics preset은 registry 기반으로 관리하고, preset 이름 위에 scene별 세부 override를 덧씌우는 구조를 지향한다.
- 랜덤화, 출력 구조, naming, metadata version을 명시적 자산으로 만든다.
- backend 차이를 소비하는 최종 조립 계층 역할을 강화한다.

---

## 5. 보류 중인 질문

아래 항목은 아직 방향을 확정하지 않은 질문들이다.

- segmentation metadata는 색상 키를 유지할지, 전부 정수 id 기반으로 정규화할지
- persisted metadata에서 `prim_path` 외에 `source_key`, `label`을 어디까지 표준 필드로 둘지
- `simulation` output schema version을 언제 도입할지
- physics preset registry를 `simulation` 내부 public API로 노출할지, 상위 앱 확장 포인트로 둘지
- OpenGL과 Blender의 결과 포맷 차이를 어디까지 공통화할지
- geometry 포맷 확장을 당장 할지, 현재 `.obj` 중심 경로를 먼저 더 고정할지

---

## 6. 중기 그림

중기적으로 `spatial_toolbox`는 아래 형태를 목표로 한다.

- `scene`: 저장 가능한 장면 모델
- `render`: backend capability가 명시된 렌더 계약
- `simulation`: schema가 고정된 데이터셋 생성 파이프라인

즉, "3D 유틸리티 모음"이 아니라 "scene -> render -> dataset" 흐름을 안정적으로 제공하는 코어 라이브러리로 수렴시키는 것이 목표다.
