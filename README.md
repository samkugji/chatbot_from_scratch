## tensorflow 1.2, python 3.6


### setup.py
    raw 문장 데이터에서 데이터셋을 구축 하는 코드

### test.py
    임의의 문장을 학습 데이터 및 테스트 데이터로 변환하는 간단한 예제코드

### train.py
    실제 학습을 진행하는 코드

### prediction.py
    여러 문장을 테스트 해서 예측 결과를 확인하는 코드


### configs/model_config.py
    모델의 세부 사이즈를 정의하는 코드

### lib
    유틸리티 함수들과 실제 모델의 구조를 정의하는 chat_seq2seq_model.py 코드가 위치


### data
    학습 및 테스트 데이터가 위치


### nn_models
    학습된 데이터가 저장되는 곳,
    checkpoint 파일에 최종 학습 모델파일의 경로가 기록되며
    새로 학습을 하고자 할경우 해당 디렉토리의 파일을 필요한 경우 백업하고 삭제 필수


### tensorboard
	기본적인 기능의 그래프 시각화를 위한 코드가 포함
	train.py가 실행된 이후에 다음과 같은 인자로 tensorboard를 실행해야 함
	tensorboard --logdir=/tmp/test_logs
	설치 및 정보 https://www.tensorflow.org/get_started/summaries_and_tensorboard