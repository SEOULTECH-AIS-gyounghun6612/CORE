# Python EXTENTION utils in AIS

Python에서 자주 사용되는 기능을 별도로 정리하여 이후 모듈로 설치하기 용이하도록 구성한 레포지토리

## Update plan

### 공통 과정
- 코드내 인자 명명법 통일
   ~~~
   module, class   : 모든 단어 블럭의 첫글자를 대문자로 변경
   변수            : 소문자로 구성
   임시 변수       : 밑줄("_")을 변수 앞에 달아 구분
   ~~~

   - [ ] System.py
   - [ ] Project.py
   - [ ] Vision.py

- docstring 작업
   ~~~
   ~~~

   - [ ] System.py
   - [ ] Project.py
   - [ ] Vision.py

### Vision.py
- [ ] 카메라 클래스 생성
   - [ ] 카메라 정보를 생성하고, 해당 정보를 바탕으로 생성 가능한 모듈 구성

- [ ] 이미지 데이터 처리 모듈




## Install
1. pip 사용
   - https 버전 -> pip install git+https://github.com/SEOULTECH-AIS-gyounghun6612/AIS_python_ex.git@ver_alpha
   - ssh 버전   -> pip install git+ssh://git@github.com/SEOULTECH-AIS-gyounghun6612/AIS_python_ex.git@ver_alpha

2. 직접 설치
   - git clone https://github.com/SEOULTECH-AIS-gyounghun6612/AIS_python_ex.git

   - cd AIS_python_ex

   - pip install -e .