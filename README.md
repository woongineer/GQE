# GQE하는거

GQE 헷갈리기 전에 정리


## NOTUSE
### my_legacy_data_fix_py_is_first_hope
여기서 data_fix.py가 맨 처음 뭔가 된거임. data 10개만 써서 random sampling대로 했을 때 training 되는거 맨 처음 보였음. 이외 파일은 그냥 시도한 것들. 이제 안봐도 될듯
### not_my_legacy
이거는 원래 코드나 석훈이 코드. 아예 안봐도 될듯.

## 20250729
이 폴더가 이제 메인임. 여기는
1. sampling
2. temperature
3. save & load 기능
4. small model(7M, 원래는 85M)
5. 그리고 이걸 기준으로 지금 서버에서 쓰고 있음.

## NQE
이 폴더는 GQE해서 만들어진 회로를 NQE 해보는거임.
pkl이랑 json 들고 와서 주피터 돌린 후 NQE 돌리면 얼추 됨.
