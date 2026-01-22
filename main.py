from fastapi import FastAPI, WebSocket, UploadFile, File
from mobilenet.preprocessing import preprocess_image
from mobilenet.model import predict, load_model
# -------------------
# 모델 로드
# -------------------
import joblib
model = joblib.load('ml/model/ad.pkl')
print('✅모델 로드 완료!')
print(f"저장 당시 sklearn 버전: {model['sklearn_version']}")
ad_model = model['model']

# --------------------
# 입력 데이터 정의
# --------------------
from pydantic import BaseModel

# Base Model을 상속하여 데이터 모델을 정의합니다.
# 클래스 속성에 타입 힌트를 지정하여 필드를 정의합니다.
class AdvertisingInput(BaseModel):
    tv: float             # tv 광고비
    radio: float          # radio 광고비
    newspaper: float      # newspaper 광고비


app = FastAPI()


@app.get("/")
def home():
    return {"message": "여기가 home입니다."}
    
@app.get("/hello")
def hello():
    return {"greeting": "hello, world!"}

@app.post("/sales_predict")
def sales_predict(advertising: AdvertisingInput):
    
    # 모델에 입력할 데이터 준비
    import pandas as pd
    features = [[
        advertising.tv,
        advertising.radio,
        advertising.newspaper,
    ]]    
    
    features = pd.DataFrame(features, columns=['TV', 'Radio', 'Newspaper'])
    
    # 예측 수행
    predicted_sales = ad_model.predict(features)[0]

    # 결과 반환
    return {
        "tv":advertising.tv,
        "radio":advertising.radio,
        "newspaper":advertising.newspaper,
        "predicted_sales": predicted_sales 
    }



# 챗봇 응답 함수
def get_bot_response(message: str) -> str:
    """
    사용자 메시지를 분석하여 적절한 응답을 반환합니다.
    """
    # 소문자로 변환하여 비교 (대소문자 구분 없이)
    msg = message.lower().strip()
    
    # 인사
    if any(word in msg for word in ['안녕', '하이', '헬로', 'hello', 'hi']):
        return "안녕하세요! 🛒 쇼핑몰 고객센터입니다. 무엇을 도와드릴까요?"
    
    # 배송 관련
    elif any(word in msg for word in ['배송', '언제', '도착', '며칠']):
        return """📦 배송 안내

- 결제 완료 후 1~2일 내 출고됩니다.
- 출고 후 1~2일 내 배송 완료됩니다.
- 제주/도서산간 지역은 2~3일 추가 소요됩니다.

배송 조회는 마이페이지에서 확인하실 수 있습니다."""
    
    # 반품/교환
    elif any(word in msg for word in ['반품', '교환', '환불', '취소']):
        return """🔄 반품/교환 안내

- 수령 후 7일 이내 신청 가능합니다.
- 단순 변심: 왕복 배송비 고객 부담
- 상품 불량: 배송비 무료

반품 신청은 마이페이지 > 주문내역에서 가능합니다."""
    
    # 결제 관련
    elif any(word in msg for word in ['결제', '카드', '계좌', '페이', '포인트']):
        return """💳 결제 수단 안내

- 신용/체크카드 (모든 카드 가능)
- 무통장 입금
- 카카오페이 / 네이버페이
- 포인트 결제

결제 관련 문의: 1234-5678"""
    
    # 영업시간
    elif any(word in msg for word in ['영업', '운영', '시간', '언제까지', '몇시']):
        return """🕐 고객센터 운영시간

- 평일: 09:00 ~ 18:00
- 점심시간: 12:00 ~ 13:00
- 주말/공휴일: 휴무

카카오톡 상담은 24시간 가능합니다."""
    
    # 연락처
    elif any(word in msg for word in ['전화', '연락', '상담', '번호', '콜센터']):
        return """📞 고객센터 연락처

- 대표번호: 1234-5678
- 이메일: help@shop.com
- 카카오톡: @쇼핑몰

평일 09:00~18:00 운영합니다."""
    
    # 도움말
    elif any(word in msg for word in ['도움', '명령', '뭐', '뭘', '기능', '할 수']):
        return """📋 도움말

다음과 같은 질문에 답변드릴 수 있어요:

- 배송 - 배송 일정 안내
- 반품/교환 - 반품, 교환 정책
- 결제 - 결제 수단 안내
- 영업시간 - 고객센터 운영시간
- 연락처 - 고객센터 연락처

키워드를 포함해서 질문해 주세요!"""
    
    # 감사 인사
    elif any(word in msg for word in ['감사', '고마워', '땡큐', 'thank']):
        return "감사합니다! 다른 문의사항이 있으시면 말씀해 주세요. 😊"
    
    # 종료 인사
    elif any(word in msg for word in ['종료', '끝', '바이', 'bye', '안녕히']):
        return "이용해 주셔서 감사합니다. 좋은 하루 되세요! 👋"
    
    # 기본 응답 (매칭 실패)
    else:
        return """죄송합니다. 이해하지 못했어요. 😅

다음 키워드로 질문해 보세요:
- 배송, 반품, 결제, 영업시간, 연락처

또는 '도움말'을 입력해 주세요."""



# WebSocket 엔드포인트
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("클라이언트 연결됨!")
    
    # 연결 시 환영 메시지 전송
    welcome_message = "안녕하세요! 🛒 쇼핑몰 고객센터입니다.\n무엇을 도와드릴까요? (도움말 입력 시 사용법 안내)"
    await websocket.send_text(welcome_message)
    
    try:
        while True:
            # 메시지 받기
            data = await websocket.receive_text()
            print(f"받은 메시지: {data}")
            
            # 챗봇 응답 생성
            response = get_bot_response(data)
            
            # 응답 전송
            await websocket.send_text(response)
            print(f"보낸 응답: {response[:50]}...")
            
    except Exception as e:
        print(f"연결 종료: {e}")


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 미리 로드"""
    load_model()


@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """
    이미지를 받아서 분류 결과 반환
    
    - **file**: 분류할 이미지 파일 (jpg, png 등)
    
    Returns:
        - predictions: 상위 5개 분류 결과
    """
    # 1. 이미지 읽기
    image_bytes = await file.read()
    
    # 2. 전처리 (preprocessing.py)
    processed_image = preprocess_image(image_bytes)
    
    # 3. 예측 (model.py)
    results = predict(processed_image)
    
    return {
        "success": True,
        "predictions": results
    }