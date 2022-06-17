# AI_PBL Final Project

## Summary
- 반려동물 구분을 위한 동물 영상 데이터셋 [링크](https://aihub.or.kr/aidata/34146)
- AI Hub의 동물 표정 데이터를 이용하여 동물의 이미지를 입력하면 동물의 기분을 한국어로 표현하는 모델 개발

## Environment
- `python=3.7`
- `tensorflow=2.5`
- `scikit-learn=1.0.1`
- `numpy=1.19.5`
- `pandas=1.3.4`

## Training Scripts
```
python train.py --train ./data/train.csv --valid ./data/val.csv --batch_size 64 --resolution 224 --learning_rate 5e-6 --epochs 100 --train_base
```
### arguments
- ```train``` : path of training csv data
- ```valid``` : path of validation csv data
- ```batch_size``` : batch_size
- ```resolution``` : model's input resolution and preprocessing input image size (default=224)
- ```learning_rate``` : learning rate
- ```epochs``` : epochs, Earlystopping callback is using in thie code, so this arg don't have to use (default=100)
- ```train_base``` : True, if you want to train backbone

## Inference Scripts
```
python inference.py --weight_path "weight_path" --input_size 224 --classes 6 --label "label map path"
```
### arguments
- ```weight_path``` : model's weight path saved with using ModelCheckpoint callback
- ```input_size``` : inference image's size, must be same with training resolution(default=224)
- ```classes``` : the number of label
- ```label``` : path of label map

---
<br></br>

# Details

## 학습환경
- CPU : AMD Ryzen 5800x
- GPU : RTX3060 (Vram : 12G)
- Ram : 32G

## Data
- 개, 고양이로 구성, 학습 데이터 기준 약 200만장
- ai hub에서 받은 데이터를 cleaning하면서 csv로 생성
    - bbox 에러에 대해서 cleaning 진행, 음수 혹은 이미지 사이즈의 범위보다 큰 수를 갖는 bbox정보에 대한 어노테이션 제거

### 문제점
1. Data Imbalance
    ![Untitled](https://user-images.githubusercontent.com/48716219/173808845-94b47697-0b14-4c14-bb79-79f48bb07ffe.png)

    <aside>
    ✅ model.fit을 하면서 각 클래스에 대한 weight를 함께 학습 (class_weight)

    [불균형 데이터 분류 | TensorFlow Core](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data?hl=ko#%ED%81%B4%EB%9E%98%EC%8A%A4_%EA%B0%80%EC%A4%91%EC%B9%98%EB%A1%9C_%EB%AA%A8%EB%8D%B8_%EA%B5%90%EC%9C%A1)
    </aside>

2. 정제되지 않는 너무 많은 데이터

    실제로 200만장 중에서도 정말로 유의미한 이미지의 개수는 훨씬 더 적음.  
    기존에 원래 영상데이터를 사용하다보니 프레임단위로 이미지가 준비되어 있어서 정제될 수 있다면 데이터 개수는 충분히 줄어들 수 있을 것이다.

    <aside>
    ✅ 랜덤 샘플링을 통해서 데이터를 반으로 줄여보았지만 학습이 제대로 진행되지 않아서 결국 다 사용
    ⇒ 유의미한 이미지들만 샘플링하기가 힘들다고 판단
    
    </aside>

## Model
- Feature Extraction
    - `EfficientNetB0` (Input size : `224`)
- Classification
    - `GlobalAveragePooling2D`
    - `BatchNormlaization`
    - `Dense(activation='softmax')`

### 문제점
1. model의 가중치 freeze시키기

    약 200만장의 데이터 를 모두 학습시키다 보니 학습이 너무 오래걸림

    - backbone 전체 freeze, classification 부분만 학습

        ⇒ 학습 자체는 빨라졌지만, val acc기준 1퍼센트도 간신히 나오는 정도로 성능이 매우 낮음
    
    - backbone의 마지막 block + classification 부분 학습

        ⇒ 전체를 학습시키는 것보다는 빠르지만 위에서 시도한 것과 마찬가지로 큰 효과는 없었음

    <aside>
    💡 결국 모델을 전체다 학습시키는 방향으로 선택
    
    </aside>
    
## Train

<aside>
💡 학습은 한 에폭에 약 4시간 소요
현재 최고 val acc는 `0.498`

</aside>

### Hyper Parameter
- init lr : 5e-6
- input resolution : 224
- batch_size : 64

### callback
- Early Stopping
    - val loss기준, patience = 5
- ModelCheckpoint
    - val acc기준, 높은 정확도의 모델 가중치를 저장

### 추후 시도해볼 만한 것들
- Learning rate Scheduler
- Warm-up + CosineAnealing
- `categorical_crossentropy`의 label smoothing인자 사용