"""A–C 시추공으로 k-NN을 검증하고 D공 암상을 예측한다."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 스크립트와 노트북(저장소 루트 또는 실습5)에서 모두 실행 가능.
BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if not (BASE / "core_wells_ABC.csv").exists():
    BASE = BASE / "실습5"
FEATURES = ["밀도_gcc", "P파속도_mps", "공극률_pct", "감마선_API", "대자율_mSI"]
LABELS = ["사암", "셰일", "석회암", "현무암"]
abc = pd.read_csv(BASE / "core_wells_ABC.csv")
d = pd.read_csv(BASE / "core_well_D.csv")
for name, frame in [("ABC", abc), ("D", d)]:
    if not np.isfinite(frame[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError(f"{name}: 결측값 또는 비유한 물성값을 확인하세요.")
if abc["암상"].isna().any() or set(abc["암상"]) != set(LABELS):
    raise ValueError("학습 자료의 암상 라벨을 확인하세요.")

# 제공된 D 파일의 밀도는 kg/m³로 기록됨. 원자료는 수정하지 않는다.
print("원자료 밀도 범위:", {"ABC": (abc["밀도_gcc"].min(), abc["밀도_gcc"].max()),
                                  "D": (d["밀도_gcc"].min(), d["밀도_gcc"].max())})
d_fix = d.copy()
if not d_fix["밀도_gcc"].between(1900, 3100).all():
    raise ValueError("D 밀도 단위가 예상과 다릅니다. /1000 변환을 재확인하세요.")
d_fix["밀도_gcc"] = d_fix["밀도_gcc"] / 1000

# k=5는 사전에 고정한 기본값이며, 검증 점수로 최적화하지 않았다.
# Pipeline을 사용해 각 훈련 폴드에서만 표준화 계수를 학습한다.
model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
X, y, groups = abc[FEATURES], abc["암상"], abc["시추공"]
pred_well = cross_val_predict(model, X, y, cv=LeaveOneGroupOut(), groups=groups)
print(f"시추공 단위 검증 정확도: {accuracy_score(y, pred_well):.4%}")
for well in sorted(groups.unique()):
    mask = groups == well
    print(f"  {well}공: {accuracy_score(y[mask], pred_well[mask]):.4%}")
cm = pd.DataFrame(confusion_matrix(y, pred_well, labels=LABELS),
                  index=pd.Index(LABELS, name="실제"), columns=LABELS)
print("혼동행렬 (행=실제, 열=예측):\n", cm)

# 비교용: 같은 층의 이웃 시료가 양쪽에 섞여 점수가 낙관적일 수 있다.
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
model.fit(Xtr, ytr)
print(f"무작위 70/30 훈련 정확도: {model.score(Xtr, ytr):.4%}")
print(f"무작위 70/30 검증 정확도 (비교용): {model.score(Xte, yte):.4%}")

# 최종 모델은 A–C 전체로 학습. D공에는 transform과 predict만 적용한다.
model.fit(X, y)
pred_D = model.predict(d_fix[FEATURES])
result = d_fix.assign(예측_암상=pred_D)
output = BASE / "core_well_D_knn_predictions.csv"
result.to_csv(output, index=False, encoding="utf-8-sig")
print("D공 예측 개수:\n", result["예측_암상"].value_counts().reindex(LABELS, fill_value=0))
print("저장:", output)
print("D공에는 정답 라벨이 없어 실제 예측 정확도를 계산할 수 없습니다.")
