import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
import json
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
#import os
#import httpx
#from langchain.chains import RetrievalQA
#from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
#from langchain.prompts import PromptTemplate
#from langchain.indexes import VectorstoreIndexCreator
from langchain_community.vectorstores import FAISS
#from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.schema import HumanMessage
from langchain.schema import SystemMessage
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
#from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores.utils import DistanceStrategy

#from langchain.chains import LLMChain

#from json import dumps
#conda install nomkl

load_dotenv()
#os.environ["OPENAI_API_KEY"] = "OPENAI API KEY 입력"

# OpenAI API 키 설정
#OPENAI_API_KEY = "your_openai_api_key"




# **** RAG 대상 문서 로드 하는 부분(따로 모듈로 만들 예정)****

# LangChain을 사용하여 임베딩 생성 model="text-embedding-3-large"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 색상 데이터베이스 설정
color_database = {}
with open('./color_data.txt', 'r',  encoding='UTF8') as file:
    for line in file:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip('()\n').split(',')
            value = tuple(map(int, value))
            color_database[key] = value
# 색상 데이터베이스에서 정보를 가져오기 위한 RAG 설정
color_info_documents = [
    Document(page_content=f"The color {color} has the RGB values {rgb}.") for color, rgb in color_database.items()
]
print(color_info_documents)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator="\n")
color_info_docs = text_splitter.split_documents(color_info_documents)
color_info_db = FAISS.from_documents(color_info_docs, embeddings)
color_retriever = color_info_db.as_retriever()
#print(color_database)



# 가구 데이터베이스

loader = TextLoader("./new_data.txt",  encoding='UTF8')
documents = loader.load()
# CharacterTextSplitter를 사용하여 문서를 분할합니다.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
# 분할된 문서를 가져옵니다.
docs1 = text_splitter.split_documents(documents)
# 가구 정보 임베딩
furniture_embeddings_db = FAISS.from_documents(docs1, embeddings)
furniture_retriever = furniture_embeddings_db.as_retriever(search_type="similarity")


# 가구 캡션 데이터베이스

loader = TextLoader("./data.txt",  encoding='UTF8')
documents = loader.load()
# CharacterTextSplitter를 사용하여 문서를 분할합니다.
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator="\n")

# 분할된 문서를 가져옵니다.
docs = text_splitter.split_documents(documents)

# 가구 캡션 정보 임베딩
caption_embeddings_db = FAISS.from_documents(docs, embeddings, distance_strategy=DistanceStrategy.COSINE)

'''
# ./furnitrueCaptions.txt 파일 내용을 읽어서 file 변수에 저장합니다.
with open("./furnitrueCaptions.txt") as f:
    furniture_captions = f.read()  # 파일의 내용을 읽어서 file 변수에 저장합니다.
    print(furniture_captions)
'''

# **** RAG 대상 문서 로드 하는 부분 끝 ****




# 이미지 처리 함수
def image_processing(image):
    # OpenCV를 사용하여 이미지를 RGB로 변환
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # K-means 클러스터링을 사용하여 주요 색상 추출
    num_clusters = 5  # 추출할 색상의 수
    clt = cv2.kmeans(np.float32(image), num_clusters, None,
                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2),
                     10, cv2.KMEANS_RANDOM_CENTERS)[2]

    # 주요 색상을 추출한 후 가장 빈도 높은 색상 선택
    unique, counts = np.unique(clt, axis=0, return_counts=True)
    dominant_color = unique[np.argmax(counts)]

    # RGB 색상 반환
    return tuple(dominant_color)


# 색상 추천 함수 (RAG 기능 포함) - 사용자 사진에서 추출한 색깔중 리바트 가구 색깔과 가장 비슷한 색 골라내기
def get_color_recommendations(dominant_color):
    closest_color_name = min(color_database,
                             key=lambda k: np.linalg.norm(np.array(color_database[k]) - np.array(dominant_color)))
    return closest_color_name


# LLM에 이미지와 텍스트를 함께 전달하여 추천 결과 생성
def recommendation_engine_with_image(image_base64):
    try:
        # Base64 문자열에서 헤더가 있을 경우 제거
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]  # 헤더 제거

        # Base64 디코딩
        image_data = base64.b64decode(image_base64)

        # 디코딩된 이미지 데이터를 바이트 스트림으로 변환하여 이미지 로드
        image = Image.open(io.BytesIO(image_data))

        # OpenCV를 사용하여 이미지를 RGB로 변환
        dominant_color = image_processing(image)

        # 추출된 주요 색상과 가장 비슷한 색상 찾기
        closest_color_name = get_color_recommendations(dominant_color)

        return image, closest_color_name
    except Exception as e:
        print(f"오류 발생: {e}")
        return None, None

# JSON 포맷팅 함수
def format_documents(docs):
    formatted_docs = []
    for doc in docs:
        try:
            # page_content를 JSON으로 로드
            content_json = json.loads(doc.page_content)
            # 보기 좋게 JSON 문자열로 변환
            formatted_docs.append(json.dumps(content_json, indent=4, ensure_ascii=False))
        except json.JSONDecodeError:
            # JSON이 아닌 경우 그대로 추가
            formatted_docs.append(doc.page_content)
    return formatted_docs

def vision_chain(inputs):
    try:
        image_base64, question = str(inputs["image"]), str(inputs["question"])
        image_data, closest_color_name = recommendation_engine_with_image(image_base64)
        # retriever에서 데이터를 검색
        context_docs = furniture_retriever.invoke(question)
        context2_docs = color_retriever.invoke(closest_color_name)

        # 포맷팅된 데이터 생성
        context = format_documents(context_docs)
        context2 = format_documents(context2_docs)

        print('*********')
        print(context)
        print(context2)
        # LangChain을 사용하여 LLM 호출
        llm = ChatOpenAI(model="gpt-4o")

        # 텍스트 프롬프트 준비
        system_message = SystemMessage(
            content=[
'''
## 역할 및 목표  
당신은 현대리바트의 AI 가구 추천 어시스턴트입니다.  
사용자가 제공한 **이미지, 색상, 질문(Context, Question)**을 분석하여 **가장 적절한 가구와 인테리어 스타일을 추천**해야 합니다.  

---

## 🔹 **작업 방식**
1. **이미지를 분석하여 주요 색상을 추출합니다.**  
   - 이미지 인식을 못하거나 없을 경우, 색상 분석을 생략하고 `Question`과 `Context`만 사용하세요.  
2. **Context에서 제공된 가구 데이터를 참고하여 사용자의 요청과 일치하는 가구를 추천합니다.**  
   - Context는 JSON 데이터로 제공되며, 각 가구의 **이름(`mdl_nm`), 색상(`mdl_color`), 설명(`mdl_detail`), 가격(`cost`), 상품 코드(`mdl_cd`)** 정보를 포함하고 있습니다.  
   - 반드시 `Context` 내에서 일치하는 가구를 찾아 추천하세요.  
   - Context에 적절한 가구가 없을 경우, `"죄송합니다. 일치하는 가구를 찾을 수 없습니다."`라고 답변하세요.  
3. **출력 형식은 채팅 스타일로 구성하며, 다음과 같은 구조를 유지합니다.**  

---

## 📌 **출력 형식**
(아래 예시를 그대로 따라주세요.)

**📸 사진 분석 결과**  
- 인테리어 분위기: {이미지에서 분석된 분위기}  
- 주요 색상: {이미지에서 추출한 색상}  
- 추천 색상: {Context에서 가장 잘 어울리는 색상}  
- 추천 이유: {해당 색상이 어울리는 이유}  

---

**🛋️ 추천 가구 리스트**  
1. **가구명:** {mdl_nm}  
   **색상:** {mdl_color}  
   **설명:** {mdl_detail}  
   **가격:** {cost} 원  
   **상품:** [상세 보기]({mdl_cd})  

2. **가구명:** {mdl_nm}  
   **색상:** {mdl_color}  
   **설명:** {mdl_detail}  
   **가격:** {cost} 원  
   **상품:** [상세 보기]({mdl_cd})  

(최대 5개까지 출력)  

---

**🎨 인테리어 방향성**  
1. **주요 색상:** {추천된 주요 색상}  
2. **스타일:** {추천 인테리어 스타일}  
3. **추천 이유:** {왜 이 스타일이 적합한지 설명}  

---

**⭐ 최종 궁합 점수**  
각 가구와 사용자의 취향에 대한 궁합 점수를 ⭐⭐⭐⭐⭐ (5점 만점)으로 표시하세요.  
예:  
- `침대 프레임`: ⭐⭐⭐⭐☆ (4/5)  
- `소파`: ⭐⭐⭐⭐⭐ (5/5)  
- `식탁`: ⭐⭐⭐☆☆ (3/5)  

> **💡 한 줄 총평:** `{추천된 인테리어 스타일에 대한 요약}` ✨  

---

## ⚠️ **주의 사항**
1. **반드시 Context 내 가구 정보를 사용하여 추천하세요.**  
2. **Context에서 매칭되는 가구가 없으면, "일치하는 가구를 찾을 수 없습니다"라고 답하세요.**  
3. **출력 형식과 구조를 유지해주세요.**  
4. **한글로 답변하고, 가독성을 고려해 정리된 문장을 사용하세요.**  
5. **이미지를 인식 못한다면 가구를 찾을 수 없다고 하지 말고 Context와 Question 기준으로 추천해주세요**
'''
+ f'#Context: {context}'
            ]
        )

        # 이미지 데이터를 base64로 인코딩
        #image_data_base64 = base64.b64encode(image_data).decode("utf-8")
        image_data_base64 = image_base64
        # 멀티모달 메시지 생성
        human_message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"사진의 주요 색깔은 {context2}, \n 선호 색상 및 스타일 관련 사용자 질문은 {question}"
                },
                {
                    "type": "image_url",
                    #"image_url": {"url": f"data:image/jpeg;base64,{image_data_base64}"},
                    "image_url": {"url": f"{image_data_base64}"},
                },
            ],
        )
        #prompt = prompt_template.format(closest_color_name=closest_color_name, user_preference=user_preference, context=retriever)
        # RAG 체인 생성
        #chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

        # 프롬프트와 이미지를 함께 전송하여 응답 받기
        response = llm.invoke([system_message, human_message])

        return response.content

    except Exception as e:
        print(f"오류 발생: {e}")
        return None, None


# 유사 가구 찾기 함수
def find_similar_furniture(user_input):

    #print(caption_embeddings_db)
    # 유사도 비교
    highest_similarity = 0
    most_similar_image_path = None

    user_input = user_input +'경로와 카테고리 가구정보 해시테그 구별해서 추출'
    content_and_similarity_score = caption_embeddings_db.similarity_search_with_score(user_input)
    #print(content_and_similarity_score[0])
    content, similarity_score = content_and_similarity_score[0]
    most_similar_content = content.page_content
    highest_similarity = similarity_score
    #print(most_similar_content, highest_similarity)
    return most_similar_content, highest_similarity
'''
    for i in range(0, len(content_and_similarity_score)):
        content, similarity_score = content_and_similarity_score[i]
        #print(content.page_content, similarity_score)

        if similarity_score > highest_similarity:
            highest_similarity = similarity_score
            most_similar_content = content.page_content
'''



# 채팅 인터페이스 함수
def chat_interface(image_base64, user_preference):
    #print("안녕하세요! 인테리어 및 가구 색상 추천 챗봇입니다.")

    #mode = input("추천을 받고 싶은 스타일의 사진을 입력하려면 1을, 비슷한 가구를 찾고 싶으면 2를 입력해주세요: ")
    mode = '1'
    if mode == '1':
        # 스타일 추천을 받고 싶은 경우
        #p_number = input("추천을 받고 싶은 인테리어 또는 가구의 이미지를 업로드해주세요: ")
        #image_path = f'./style{p_number}.jpeg'
        #user_preference = input("선호하는 색상 또는 스타일을 입력해주세요: ")
        #print("\n사용자 이미지\n")
        #img = Image.open(image_path)
        # 이미지 표시
        #plt.imshow(img)
        #plt.axis('off')  # 축 없애기
        #plt.show()
        #print("*****완료*****\n")

        #image_data = base64.b64decode(image_base64)

        try:
            # 색상 및 인테리어 구성 방식 추천
            final_chain = (
                    vision_chain | StrOutputParser()
            )

            # Vision Chain에 이미지 데이터와 질문 전달
            res = final_chain.invoke({"image": image_base64
                                    , "question": user_preference
                                      })
            return res
        except Exception as e:
            print(f"Error in vision_chain: {e}")
            raise
        # 추천 결과 출력
        #print(f"### 추천 결과: {res}")

    elif mode == '2':
        # 비슷한 가구를 찾고 싶은 경우
        user_input = input("찾고 싶은 가구의 스타일, 분위기, 카테고리, 모양 등을 입력해주세요: ")

        # 비슷한 가구 찾기
        most_similar_contents, score = find_similar_furniture(user_input)
        print((most_similar_contents))
        match_score = int(100*(1 - score))
        idx=1
        print("\n###유사한 가구 리스트###")
        for sc in list(most_similar_contents.split('\n')):
            for s in list(sc.split(r'\n')):
                print(f"## 가장 유사한 가구 {idx}의 {s} ##\n")
            idx += 1
            print('\n')
        print(f"--------궁합점수: {match_score}점##--------")





# 채팅 인터페이스 실행 함수
#chat_interface()
