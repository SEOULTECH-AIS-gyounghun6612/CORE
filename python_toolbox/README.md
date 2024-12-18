# Python EXTENTION utils in AIS

Python에서 자주 사용되는 기능을 별도로 정리하여 이후 모듈로 설치하기 용이하도록 구성한 레포지토리

## Update plan

### 공통 과정

- [ ] python naming 규칙 통일 및 docstring 작업 -> [참고 링크](https://github.com/SEOULTECH-AIS-gyounghun6612/AIS_lab_manual/blob/Publish/python/doc_string.md)
  - [ ] system.py
  - [ ] project.py
  - [ ] file.py

### system.py

- [x] Pathlib 모듈 적용을 통해 Path 및 File 관련 코드 정리

### project.py

- [ ] log 처리 기능 추가
- [x] argparse 모듈을 사용하여 프로젝트의 초기 설정값 기능 추가
  - [x] 사전에 준비된 초기 설정 파일 입력 처리
  - [x] 클래스 변수를 이용한 초기 설정값 처리 기능 추가

### file.py

- [ ] 주요 파일 포멧 지원
  - [x] json
  - [ ] xml
  - [ ] yaml
  - [ ] csv

### data

- [별도 페이지 참조](./python_ex/data/README.md)

## Install

- pip 사용

  ```bash
  # using pip with https
  pip install git+https://github.com/SEOULTECH-AIS-gyounghun6612/AIS_python_ex.git@ver_alpha
  
  # using pip with ssh
  pip install git+ssh://git@github.com/SEOULTECH-AIS-gyounghun6612/AIS_python_ex.git@ver_alpha
  ```

- 직접 설치

  ```bash
  # clone this repository
  git clone https://github.com/SEOULTECH-AIS-gyounghun6612/AIS_python_ex.git

  cd AIS_python_ex

  # install 
  pip install -e .

  ```
