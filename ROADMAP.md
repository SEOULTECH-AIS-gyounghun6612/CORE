# ROADMAP

진행 현황 및 향후 계획을 관리하는 문서.

> **코드 분석 일자:** 2026-04-09

---

## 기능 개선

- [ ] `Utils.Read_from` — 확장자 기반 분기 시 `.json`/`.yaml` 외 포맷 지원 (예: `.toml`)
- [ ] `Project_Template` — `_Setup()` 이후 workspace 경로를 로그에 출력하는 기본 훅 추가
- [ ] `Registry` — `Get_all()` 메서드 추가 (등록된 키 목록 조회)
- [ ] `Base_Config.Write_to` — JSON/YAML 포맷 자동 선택 (현재는 `File_Utils.Write_to` 위임)

---

## 로깅 표준화

- [ ] `Handle_exp` — `print` 기반 출력을 표준 logger로 교체 (라이브러리 내 logger 도입 후 진행)
- [ ] torch_toolbox runner 내 `print` 교체 — 이 라이브러리의 logger 준비 완료 후 진행
