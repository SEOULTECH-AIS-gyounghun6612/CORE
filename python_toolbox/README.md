# Python EXTENTION utils in AIS

Python에서 자주 사용되는 기능을 별도로 정리하여 이후 모듈로 설치하기 용이하도록 구성한 레포지토리

## Update plan

### 공통 과정

- [ ] python naming 표준화 -> [참고 링크](https://github.com/SEOULTECH-AIS-gyounghun6612/SEOULTECH-AIS-gyounghun6612.github.io/blob/main/python_md/doc_string.md#Naming-예시)
  - [ ] system.py
  - [ ] project.py

- [x] python code 한 줄 최대값(=80) 준수
  - [x] system.py
  - [x] project.py
  - [x] vision.py

- [ ] docstring 작업 -> [참고 링크](https://github.com/SEOULTECH-AIS-gyounghun6612/SEOULTECH-AIS-gyounghun6612.github.io/blob/main/python_md/doc_string.md#Doc-string-예시)
  - [ ] system.py
  - [ ] project.py
  - [ ] vision.py

### system.py

- [x] yaml 파일 입력 관련 코드 수정 -> dict data를 사용하여 처리
- [ ] pathlib 모듈 적용을 통해 Path 및 File 관련 코드 정리

### project.py

- [ ] log 처리 기능 추가
- [ ] argparse 모듈을 사용하여 프로젝트의 초기 설정값 기능 추가
  - [x] 사전에 준비된 초기 설정 파일 입력 처리
  - [ ] 클래스 변수를 이용한 초기 설정값 처리 기능 추가

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
