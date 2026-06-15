# scene

`scene/`은 3D 장면 데이터의 모델 정의, 에셋 캐시, 장면 직렬화, 트리 조립을 담당하는 순수 데이터 레이어임.

`render/`, `ui/`를 포함한 상위 모듈이 이 레이어를 소비하며, `scene/` 자체는 렌더러나 UI 계층에 의존하지 않음.

```text
scene/ ←── render/
       ←── ui/
```

## 역할

- `asset/`은 파일에서 로드된 원시 geometry를 관리함
- `node/`는 씬 그래프 노드와 배치 정보를 관리함
- `file/`은 씬 상태의 JSON / USD import-export를 담당함
- `stage.py`의 `Controller`는 세 계층을 묶는 상위 진입점임

## 의존 방향

```text
stage.py     ──► asset/, node/, file/
asset/cache  ──► asset/type, asset/file
node/type    ──► node/register
file/*       ──► node/type, asset/cache
```

순환 의존은 없음. `scene/file`은 "원시 geometry 로더"가 아니라 장면 상태의 import/export 계층이고, 원시 3D 파일 로딩은 `scene/asset/file`이 담당함.

## 문서

- 사용 예제와 빠른 시작: [COOKBOOK.md](./COOKBOOK.md)
- 노드 타입과 트리 조작: [node/COOKBOOK.md](./node/COOKBOOK.md)
- 에셋 캐시와 geometry 로드: [asset/COOKBOOK.md](./asset/COOKBOOK.md)
- JSON / USD 장면 저장·복원: [file/COOKBOOK.md](./file/COOKBOOK.md)
