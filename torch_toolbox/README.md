# Pytorch EXTENTION utils in AIS

기본적인 pytorch 활용법과 관련하여 자주 쓰는 구성을 정리하여 이후 작업에 활용하기 위하여 구성한 레포지토리

## Update plan

### 공통 과정

- [ ] docstring 작업 -> [참고 링크](https://github.com/SEOULTECH-AIS-gyounghun6612/SEOULTECH-AIS-gyounghun6612.github.io/blob/main/python_md/doc_string.md#Doc-string-예시)
  - [ ] dataset/
  - [ ] newural_network/
  - [ ] learning.py

### dataset

- [ ] kitti dataset 처리 모듈 추가

### newural_network

- [ ] 주요 모듈 업데이트
  - [ ] Conv2D 기반 자주 사용되는 encoder-decoder 구조 추가

### learning.py

- [ ] 심층 신경망 강화학습 구조 코드 추가

## requirments

- 공통 사항
  - python >= 3.10
  - python_ex [링크](https://github.com/SEOULTECH-AIS-gyounghun6612/AIS_python_ex.git)
  - pytorch

- layer/3d_gaussian
  - open3d
  - plyfile
  - diff-gaussian-rasterization [링크](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
  - simple-knn [링크](https://gitlab.inria.fr/bkerbl/simple-knn.git)

## Install

- pip 사용

  ```bash
  # using pip with https
  pip install git+https://github.com/SEOULTECH-AIS-gyounghun6612/AIS_torch_ex.git@ver_alpha
  
  # using pip with ssh
  pip install git+ssh://git@github.com/SEOULTECH-AIS-gyounghun6612/AIS_torch_ex.git@ver_alpha
  ```

- 직접 설치

  ```bash
  # clone this repository
  git clone https://github.com/SEOULTECH-AIS-gyounghun6612/AIS_torch_ex.git

  cd AIS_torch_ex

  # install 
  pip install -e .

  ```
