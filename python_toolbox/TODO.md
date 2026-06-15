# TODO

미완료 작업 목록. 완료 이력은 [README 개발 로그](./README.md#개발-로그) 참조.

---

## file

- [ ] `Write_to`와 `Base_Config.Write_to` 사용 경계 문서화
- [ ] `.toml` 등 포맷 추가 여부 결정
- [ ] `Handle_exp` 출력 방식 개선 검토 (현재 print)

---

## registry

- [ ] 등록된 키 목록 조회 public API 추가 검토 (현재 `_module_dict` 직접 접근만 가능)
- [ ] Callable 타입 검증 범위 확대 여부 검토 (현재 파라미터 개수만 검증)

---

## project

- [ ] `Base_Config` 읽기 보조 API 필요성 재검토 — `Write_to`만 있고 읽기 대응 메서드 없음. `Build_config_from_file`로 충분한지 판단
- [ ] `Build_sub_config` 사용 시나리오 예제 보강
- [ ] `Build_parser_from_config`의 bool/list/dict 규약 문서 보강
