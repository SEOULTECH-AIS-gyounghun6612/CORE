# Python EXTENTION utils in AIS

Python에서 자주 사용되는 기능을 별도로 정리하여 이후 모듈로 설치하기 용이하도록 구성한 레포지토리

## Update plan

### 공통 과정
- [ ] python code 표준화
   - [ ] python naming 표준화 -> [참고 링크](https://github.com/SEOULTECH-AIS-gyounghun6612/SEOULTECH-AIS-gyounghun6612.github.io/blob/main/python_md/doc_string.md#Naming-예시)         
      - [ ] system.py
      - [ ] project.py
      - [ ] vision.py
  
   - [x] python code 한 줄 최대값(=80) 준수
      - [x] system.py
      - [x] project.py
      - [x] vision.py

   - [ ] docstring 작업 -> [참고 링크](https://github.com/SEOULTECH-AIS-gyounghun6612/SEOULTECH-AIS-gyounghun6612.github.io/blob/main/python_md/doc_string.md#Doc-string-예시)
      - [ ] system.py
      - [ ] project.py
      - [ ] vision.py

### system.py
- [ ] yaml 파일 입력 관련 코드 수정 -> dict data를 사용하여 처리

### vision.py
- [ ] 카메라 클래스 생성
   - [ ] 카메라 정보를 생성하고, 해당 정보를 바탕으로 생성 가능한 모듈 구성
   - [ ] 카메라를 통해 촬영한 각 장면의 정보를 처리할 클래스 구성 (= Scene)
   - [ ] Scene 클래스와 Camera 클래스의 정보 범위와 처리 과정에 대하여, 명확한 계획 구성

- [ ] 이미지 데이터 처리 모듈

### project.py
- [ ] log 처리 기능 추가


## Install
1. pip 사용
   - https 버전 -> pip install git+https://github.com/SEOULTECH-AIS-gyounghun6612/AIS_python_ex.git@ver_alpha
   - ssh 버전   -> pip install git+ssh://git@github.com/SEOULTECH-AIS-gyounghun6612/AIS_python_ex.git@ver_alpha

2. 직접 설치
   - git clone https://github.com/SEOULTECH-AIS-gyounghun6612/AIS_python_ex.git

   - cd AIS_python_ex

   - pip install -e .