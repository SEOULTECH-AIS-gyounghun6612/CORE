# Python EXTENTION utils in AIS

Python에서 자주 사용되는 기능을 별도로 정리하여 이후 모듈로 설치하기 용이하도록 구성한 레포지토리

## Update plan

### 공통 과정
- 신규 데이터 모듈 생성
   - [ ] _Custom_Debuger.py

- 코드내 인자 명명법 수정
   - [x] _base.py
   - [x] _numpy.py
   - [x] _result.py
   - [ ] _vision.py

- 함수 주석 작업
   - [ ] _base.py
   - [ ] _numpy.py
   - [ ] _result.py
   - [ ] _vision.py

#### _error.py
- 자체적인 에러 생성 코드 방법 구성
   - [ ] 오류 출력 내용 중 발생 지점 정보 포함

#### _numpy.py
- 데이터 입출력 코드 수정
   - [x] 데이터 저장
   - [ ] 데이터 읽기


## Install
1. pip 사용
   - https 버전 -> pip install git+https://github.com/SEOULTECH-AIS-gyounghun6612/AIS_python_ex.git@ver_alpha
   - ssh 버전   -> pip install git+ssh://git@github.com/SEOULTECH-AIS-gyounghun6612/AIS_python_ex.git@ver_alpha
