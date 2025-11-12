import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# 1. 데이터 불러오기 (UTF-8 CSV)
data = pd.read_csv("school_bacteria_data.csv", encoding="cp949")

# 2. 입력(X), 정답(y) 나누기
X = data[["place_type", "people_level", "touch_level",
          "clean_level", "humidity_level", "colonies"]]
y = data["risk_label"]

# 3. 범주형 / 수치형 컬럼 지정
cat_cols = ["place_type", "people_level", "touch_level",
            "clean_level", "humidity_level"]
num_cols = ["colonies"]

# 4. 전처리 + 모델 파이프라인
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("clf", LogisticRegression(max_iter=1000))
])

# 5. 학습
model.fit(X, y)

# 6. 모델 저장
with open("risk_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ risk_model.pkl 저장 완료")
