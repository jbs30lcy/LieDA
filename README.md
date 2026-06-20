# LieDA

## 0. 받아오기
너 git clone으로 불러왔었냐

zip파일로 받은 건 git 버전이 꼬여서 머리가 아프기 때문에
```powershell
git clone https://github.com/jbs30lcy/LieDA
```
하셈

git clone으로 불러온거 맞으면
```powershell
git pull
```
하면 됨

그리고 연구실 컴퓨터면 모듈 버전 안꼬이게 해야 되니까 venv 안에서 하는 게 좋을거임. 코덱스한테 venv 세팅해 달라고 하고 필요한 모듈 다 깔아달라고 하면 깔아줄거임

## 1. 학습 순서
지금 train.py에 주석이 길게 있고 6단계로 나눠져 있음

일단 바로 실행해도 됨. 지금 1단계 실행하는 코드가 적혀 있음. 1단계 끝나면 common은 건드리지 말고, 방금 실행한 train 함수 부분 지우고 주석 하나씩 풀어보면서 실행해 보셈

하다가 도저히 못써먹는 결과가 나오면 일단 나한테 알려줘

## 2. 주의할 점

1. early stop = 5로 고정. 막 10 step 안에 끝나도 정상인 거 같음

2. difficulty 1은 실제 결과가 당연히 좀 이상함. good example.mp4 처럼 나올 거고, accuracy든 loss든 별로임

그게 왜그러냐면 last.png로부터 last_heatmap.png 같이 생긴 이미지를 맞추도록 훈련되서 다른 도형들도 다 맞춤

3. train 결과를 runs/뭐시기뭐시기/eval 에서 확인 할 수 있음

json 들어가보면 accuracy (도형 내부로 잘 맞출 확률(사실 아님)), loss 같은 것들 확인 가능

맨 처음 빼고 학습이 300스텝 전부 돌거나, 학습이 끝났는데 accuracy가 80 밑이면 조진거임

## 3. 결과 공유
```powershell
git add .
git commit -m "하고싶은말 but 사회적으로 논란거리가 되는 말은 안됨"
git push
```
git push가 실패하고 뭐라뭐라 나올 때가 있음.

1. git push origin main 같은 거 나오면 그냥 그거 다시 치면 됨

2. 깃허브 계정? 뭐 config? 없다고 하는 말 나오면
```powershell
git config --global user.name "니이름"
git config --global user.email "니이메일"
```
라고 하면 됨. 니 이메일은 구라쳐도 되는데 이름은 안 치는 걸 추천함