# TODO

미완료 작업 목록. 완료 이력은 [README 개발 로그](./README.md#개발-로그) 참조.

---

## 후속 작업


---

## 테스트

- [ ] `runner/assembler` — mode별 `Assemble_Metric` 빌드 및 `Update/Finalize` 흐름 테스트
- [ ] `metric/accumulator` — Scalar/Centroid/Assemble accumulator 단위 테스트
- [ ] `runner/supervised` — `_Iter_hook` → `_Forward` → `metric[mode].Update` → `log_batch` 흐름 통합 테스트

---

## dataset

- [ ] YAML 파일 읽기 (`_Load_id_map` 등) — `python_toolbox` IO 유틸리티로 이전
- [ ] data transform — `Get_transform()` if-else 분기를 registry 구조로 교체
- [ ] Object Detection 데이터셋 파이프라인 완성 (COCO, YOLO) — coco.py/yolo.py 스텁 상태
- [ ] `dataloader/classification/image.py` — 이미지 외 입력(포인트클라우드, 센서 데이터 등)을 포괄하는 일반화된 classification dataset 구조로 개선 필요

---

## modules

- [ ] Transformer 계열 모듈 고도화 (Attention, Embedder 리팩토링)

---

## 배포 파이프라인

- [ ] ONNX 추출 후 모델 무결성 검증 로직 추가 (`onnx.checker.check_model()`)

---

## 로깅 표준화

- [ ] `python_toolbox` 로거 도입 후 `definition.py` 내 `print` 교체 — python_toolbox 작업 완료 후 진행
