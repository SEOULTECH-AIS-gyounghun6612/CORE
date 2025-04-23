# Pytorch toolbox in AIS

연구 과정 중 사용되는 PyTorch 기반의 딥러닝 모델 구현 시 자주 사용되는 구성 요소들을 정리한 레포지토리입니다.
기초적인 활용부터 고급 구성까지 재사용 가능한 구조로 정리하여 연구 및 개발 효율성을 높이는 것을 목표로 합니다.

## Update plan

### 공통 과정

### 📁 dataset

### 📁 neural_network

- [ ] 주요 backbone 사용 구조 업데이트

### 📄 learning.py

- [ ] test 구조와 train-validation 과정 분리

- [ ] 학습 구조 추가
  - [ ] 심층 신경망 기반 강화학습

## 🔧 Installation

### 📚 Requirements

✅ 공통

- [python_ex](https://github.com/SEOULTECH-AIS-gyounghun6612/AIS_python_ex.git)

✅ layer/3d_gaussian 관련

- open3d
- plyfile
- [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [simple-knn](https://gitlab.inria.fr/bkerbl/simple-knn.git)

### pip 사용 (HTTPS)

```bash
# using pip with https
pip install git+https://github.com/SEOULTECH-AIS-gyounghun6612/AIS_torch_toolbox.git@ver_alpha
```

### 직접 설치 (로컬 clone)

```bash
# clone this repository
git clone https://github.com/SEOULTECH-AIS-gyounghun6612/AIS_torch_toolbox.git

cd AIS_torch_toolbox

# install 
pip install -e .
```
