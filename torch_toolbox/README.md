# Pytorch EXTENTION utils in AIS

기본적인 pytorch 활용법과 관련하여 자주 쓰는 구성을 정리하여 이후 작업에 활용하기 위하여 구성한 레포지토리

## Update plan

### 공통 과정
- [ ] python code 표준화
   - [x] python naming 표준화 -> [참고 링크](https://github.com/SEOULTECH-AIS-gyounghun6612/SEOULTECH-AIS-gyounghun6612.github.io/blob/main/python_md/doc_string.md#Naming-예시)      
      - [x] learning.py
      - [x] config.py
      - [x] Python code in dataset directory
  
   - [ ] python code 한 줄 최대값(=120) 준수
      - [x] learning.py
      - [x] config.py
      - [ ] Python code in dataset directory

   - [ ] docstring 작업 -> [참고 링크](https://github.com/SEOULTECH-AIS-gyounghun6612/SEOULTECH-AIS-gyounghun6612.github.io/blob/main/python_md/doc_string.md#Doc-string-예시)
      - [x] System.py
      - [ ] config.py
      - [ ] Python code in dataset directory

### layer.py
- [ ] 주요 모듈 추가

### dataset
- [ ] augmentation 기능 추가 구성

## requirments
- 공통 사항
   - python_ex [링크]()

- layer/3d_gaussian
   - open3d
   - plyfile
   - diff-gaussian-rasterization [링크](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
   - simple-knn [링크](https://gitlab.inria.fr/bkerbl/simple-knn.git)

## Install
1. pip 사용
   - https 버전 -> pip install git+https://github.com/SEOULTECH-AIS-gyounghun6612/AIS_torch_ex.git@ver_alpha
   - ssh 버전   -> pip install git+ssh://git@github.com/SEOULTECH-AIS-gyounghun6612/AIS_torch_ex.git@ver_alpha


