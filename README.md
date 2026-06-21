# LieDA

**실험 전에 이름을 입력할 수 있으니까, runs에서 구별되도록 이름 좀 붙여주셈**

loss 함수를 일단 손을 좀 봤음. 근데 이래도 구석탱이로 가는 걸 막을 수 있을지는 모르겠음

그리고 학습할 경우의 수를 꽤나 늘려놨음

일단 difficulty 1~6까지는 최대한 순서대로 하셈. heatmap_params="latest"로 받아오니까


경우의 수는 다음과 같음
1. PartialHeatUNet 자리
```python
PartialHeatUNet(in_channels=7)
ShortNet(in_channels=7)
TinyNet(in_channels=7)
```
TinyNet은 shape error 날 수도 있음. 코덱스 보고 고쳐달라 해 만약 그러면

2. LieDA 자리
```python
LieDA(
    
)
noHiddenLieDA(

)
```
즉 6가지 경우에서 6번 학습하면 최대 36번 학습을 돌려야 할 수도 있음!