# advertising.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# 데이터 불러오기
df = pd.read_csv('data/advertising.csv')

# 변수 선택
X = df[['TV','Radio','Newspaper']]
y = df['Sales']

# 모델 생성 및 훈련
model = RandomForestRegressor()
model.fit(X, y)

# 모델 저장
import sklearn
joblib.dump({
    'model': model,
    'sklearn_version': sklearn.__version__
}, 'model/ad.pkl')

