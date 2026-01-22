# preprocessing.py
# 이미지 전처리 담당

from PIL import Image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import io


def preprocess_image(image_bytes: bytes):
    """
    이미지 바이트 데이터를 모델 입력 형태로 변환
    
    Args:
        image_bytes: 이미지 파일의 바이트 데이터
        
    Returns:
        numpy array: (1, 224, 224, 3) 형태의 전처리된 이미지
    """
    # 1. 바이트 데이터를 PIL 이미지로 변환
    image = Image.open(io.BytesIO(image_bytes))
    
    # 2. RGB로 변환 (PNG 등 알파 채널 제거)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 3. 224x224로 리사이즈 (MobileNetV2 입력 크기)
    image = image.resize((224, 224))
    
    # 4. numpy 배열로 변환
    img_array = np.array(image)
    
    # 5. 배치 차원 추가 (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 6. MobileNetV2 전처리 적용 (-1 ~ 1 범위로 정규화)
    img_array = preprocess_input(img_array)
    
    return img_array