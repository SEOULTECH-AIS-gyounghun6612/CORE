# Python EXTENTION utils in AIS

Python에서 자주 사용되는 기능을 별도로 정리하여 이후 모듈로 설치하기 용이하도록 구성한 레포지토리

## Update plan

### 공통 과정

- [x] python naming 규칙 통일 및 docstring 작업 -> [참고 링크](https://github.com/SEOULTECH-AIS-gyounghun6612/AIS_lab_manual/blob/Publish/python/doc_string.md)
  - [x] system.py
  - [x] project.py
  - [x] file.py

### system.py

- [x] Pathlib 모듈 적용을 통해 Path 및 File 관련 코드 정리
  - [x] 파일 입출력 관련 코드 정리 -> Path class를 이용한 단순한 코드를 제외하고, 모듈 단위 코드 정리 필요

### project.py

- [x] ~~log 처리 기능 추가~~ -> config class와 통합
- [ ] argparse 모듈을 사용하여 프로젝트의 초기 설정값 기능 추가
  - [ ] 프로그램 실행 시 입력된 인자 입력 처리
  - [ ] 사전에 준비된 초기 설정 파일(config)와 결합된 입력 처리
  - [ ] 클래스 변수를 이용한 초기 설정값 처리 기능 추가

### file.py

- [x] 주요 파일 포멧 지원
  - [x] json
  - [x] xml
  - [x] yaml
  - [x] csv

### data

- [별도 페이지 참조](./python_toolbox/data/README.md)

## Install

- pip 사용

  ```bash
  # using pip with https
  pip install git+https://github.com/SEOULTECH-AIS-gyounghun6612/AIS_python_toolbox.git
  
  # using pip with ssh
  pip install git+ssh://git@github.com/SEOULTECH-AIS-gyounghun6612/AIS_python_toolbox.git
  ```

- 직접 설치

  ```bash
  # clone this repository
  git clone https://github.com/SEOULTECH-AIS-gyounghun6612/AIS_python_toolbox.git

  cd AIS_python_toolbox

  # install 
  pip install -e .

  ```
