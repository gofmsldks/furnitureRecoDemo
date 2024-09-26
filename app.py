from flask import Flask, request, jsonify
import model  # model.py 파일을 import

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    # 텍스트와 이미지 데이터 받기
    user_preference = request.form.get('textMessage', '')
    image_base64 = request.form.get('imgMessage', '')

    if not image_base64:
        return jsonify({"error": "Image data is required"}), 400

    # AI 모델 함수 호출
    try:
        result = model.chat_interface(image_base64, user_preference)
        return jsonify({"textMessage": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)