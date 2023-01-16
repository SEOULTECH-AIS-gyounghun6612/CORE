# Python EXTENTION utils in AIS

Python에서 자주 사용되는 기능을 별도로 정리하여 이후 모듈로 설치하기 용이하도록 구성한 레포지토리

## Update plan

### 공통 과정
- 코드내 인자 명명법 수정
   - [ ] _base.py
   - [ ] _error.py
   - [ ] _numpy.py
   - [ ] _result.py
   - [ ] _vision.py

#### _error.py
- 자체적인 에러 생성 코드 방법 구성
   - [ ] 오류 출력 내용 중 발생 지점 정보 포함

#### _numpy.py
- 배열 생성 코드 수정
   - [ ] 크기 기반
   - [ ] 샘플 기반
   - [ ] 랜덤 방식 명명을 위한 얄가형 인자 값 추가

- 데이터 입출력 코드 수정
   - [ ] 데이터 저장
   - [ ] 데이터 읽기


## Install
1. pip 사용
   - https 버전 -> pip install git+https://github.com/SEOULTECH-AIS-gyounghun6612/AIS_python_ex.git@ver_alpha
   - ssh 버전   -> pip install git+ssh://git@github.com/SEOULTECH-AIS-gyounghun6612/AIS_python_ex.git@ver_alpha
