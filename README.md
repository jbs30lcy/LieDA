# LieDA

## 0. 받아오기
빈 폴더 만들고 그 안에서 
```powershell
git clone https://github.com/jbs30lcy/LieDA
```

## 1. 터미널에서 해야 할 거

vscode로 방금 받은 이 코드들 들어있는 폴더 Open Folder하고 나서
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch numpy opencv-python matplotlib pillow tqdm
```
한줄씩 실행해보셈
이게 오류나면 그건 난 몰라 더이상 진행못함

## 2. 할 거
너가 실행할 코드는 train.py 하나면 되고,
코드를 고칠 부분도 train.py 맨 아래 몇 줄 뿐이다.

1. 일단 실행
실행하면 runs 디렉토리에 뭐가 생긴다
eval에 들어가서 steps_(제일큰숫자) 영상을 보셈
초록색이 진짜 center, 빨간색이 예측 center임
적당히 맘에 들면 다음

2. 레벨2
맨 마지막 줄을 아래처럼 바꿔서 다시 해보셈
```python
dataloader = my_dataloader(difficulty=2, batch=4)
train(model, dataloader, steps = 200, batch = 4, eval_step = 20, save_step = 20, use_distance_bias=True, gamma=-0.01)
```
이것도 확인 해보고 맘에 들면 다음

3. 레벨3
맨 마지막 줄을 아래처럼 바꿔서 다시 해보셈
```python
dataloader = my_dataloader(difficulty=3, batch=4)
train(model, dataloader, steps = 300, batch = 4, eval_step = 20, save_step = 20, use_distance_bias=True, gamma=-0.01)
```
이것도 확인 해보고 맘에 들면 끝

## 3. 뭔가 이상하다!
일단 바꿔볼 건 gamma 값
지금 -0.01로 해놓았는데, 이게 그냥 쳐 찍은거란 말임
-0.02, -0.05, -0.1, -0.005 정도 시도해 보셈
0도 해봐도 되는데 아마? 성능 안 좋을거임

그래도 뭔가 븅신같다
그러면 모델 자체를 바꿔야 하는데
급하게 바이브코딩 때려놓은 SkipLieDA라는 모델이 있음
```python
model = SkipLieDA()
```
이렇게만 바꾸고 해보셈

## 4. 만약 코드가 오류나면
니 잘못임
그래도 혹시 모르니까 코덱스에 물어봐서 고쳐보셈