# base_utils_v2

기본적으로 자주 사용하는 Python 관련 개인 모듈 정리 및 향후 pip 명령어를 활용하여 설치하기 위한 레포지토리  
Basically, a repository for personal modules related to Python that are frequently used and for installation using pip commands in the future.

## Update plan

#### _base.py
- [x] 폴더명을 트리 구조 데이터로 입력 받아, 폴더를 생성
- [x] Receives the folder name as tree structure data and creates a folder

#### _cv.py
- [ ] nump를 이용한 일부 모듈 이동 (-> _numpy.py)
- [ ] Move some modules using numpy (-> numpy.py)

#### _label.py
- [ ] 계획 미정
- [ ] Not yet

#### _error.py
- [ ] 계획 미정
- [ ] Not yet

#### _numpy.py
- [ ] Neighbor_Confusion_Matrix 기능 복구 및 수정
- [ ] Neighbor_Confusion_Matrix function recovery and fix

## Install
1. SSH 키 등록
   - 해당 등록 방법의 경우 연구실 서버내 메뉴얼 참조
2. 명렁어 입력
   - pip install "git+ssh://git@github.com/gyounghun6612/base_utils_v2.git"
