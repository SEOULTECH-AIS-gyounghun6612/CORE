# Pytorch EXTENTION utils in AIS

기본적인 pytorch 활용법과 관련하여 자주 쓰는 구성을 정리하여 이후 작업에 활용하기 위하여 구성한 레포지토리

## Update plan

### 공통 과정
- 코드내 인자 명명법 수정
   - [x] _Base.py
   - [x] _Dataset.py
   - [x] _Learning.py
   - [x] _Model_n_Optim.py

### _Model_n_Optim.py
   - [ ] Multi GPU 과정 중 Learning_Process.Basement._Process에서 메인 프로세서 구분 방법 변경(인자값 -> DDP 함수)


## Install
1. pip 사용
   - https 버전 -> pip install git+https://github.com/SEOULTECH-AIS-gyounghun6612/AIS_torch_ex.git@ver_alpha
   - ssh 버전   -> pip install git+ssh://git@github.com/SEOULTECH-AIS-gyounghun6612/AIS_torch_ex.git@ver_alpha
