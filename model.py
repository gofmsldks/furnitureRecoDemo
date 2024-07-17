import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64
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
with open('./color_data.txt', 'r') as file:
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
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=10,separator="\n")
color_info_docs = text_splitter.split_documents(color_info_documents)
color_info_db = FAISS.from_documents(color_info_docs, embeddings)
color_retriever = color_info_db.as_retriever()
#print(color_database)



# 가구 데이터베이스

loader = TextLoader("./new_data.txt")
documents = loader.load()
# CharacterTextSplitter를 사용하여 문서를 분할합니다.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
# 분할된 문서를 가져옵니다.
docs1 = text_splitter.split_documents(documents)
# 가구 정보 임베딩
furniture_embeddings_db = FAISS.from_documents(docs1, embeddings)
furniture_retriever = furniture_embeddings_db.as_retriever()
#doc = furniture_retriever.invoke('소파를 찾아주세먼')
#print(doc[0].page_content)

# 가구 캡션 데이터베이스

loader = TextLoader("./data.txt")
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
def recommendation_engine_with_image(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()

    # 이미지 처리
    image = Image.open(io.BytesIO(image_data))
    dominant_color = image_processing(image)
    closest_color_name = get_color_recommendations(dominant_color)

    return image_data, closest_color_name

def vision_chain(inputs):
    image_path, user_preference, context, context2 = str(inputs["image_path"]), str(inputs["user_preference"]), inputs["context"], inputs["context2"]
    image_data, closest_color_name = recommendation_engine_with_image(image_path)

    # LangChain을 사용하여 LLM 호출
    llm = ChatOpenAI(model="gpt-4o")





    # 텍스트 프롬프트 준비
    system_message = SystemMessage(
        content=[
            '''당신은 사용자가 제공한 사진, 그 사진의 주요색상, 주어진 색상표와 잘 조합해서 잘어울리는 가구와 인테리어를 골라줘, 최종적으로 각 가구와 사용자와의 궁합과 궁합점수를 나타내주는 AI 어시스턴트입니다.
            당신의 임무는 주어진 문맥(Context1, Context2)에서 사용자의 취향과 일치하는 색상 및 가구를 찾아서 추천해 주는 것입니다. 검색된 다음 문맥(Context1, Context2)을 사용하여 질문에 답하세요.
            *중요) Context1은 색상표이며 Context2는 추천해줄 가구 리스트로 이미지경로, 가구명, 색상, 가구정보, 해시태그가 있는 정보입니다.
            사용자가 제공한 사진, 그 사진의 색상, Context1과 비슷한 색상, 취향과 문맥(Context1, Context2)내에서 가구와 인테리어를 추천헤주세요,
            먼저, 사진의 분위기와 설명을 요약해주고 해당 사진의 컬러와 추천 색상과 추천 이유를 말해주세요. 그후 인테리어 방향성과 추천 가구 리스트를 보여주세요. 추천가구이름은 최대한 대분류로 나타내주세요.
            최종적으로는 각 가구와 사용자와의 궁합과 궁합점수를 나타내주세요.
            한글로 답해주시고 재치있고 눈에잘들어오도록 답변해주세요.
           
            '''
            + '문맥의 내용은 아래와 같습니다.'
            + f' #Context1: {context}'
            + f' #Context2: {context2}'

        ]
    )
    # 이미지 데이터를 base64로 인코딩
    image_data_base64 = base64.b64encode(image_data).decode("utf-8")

    # 멀티모달 메시지 생성
    human_message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": f"사진의 주요 색깔은 {closest_color_name},선호 색상 및 스타일은 {user_preference}"
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data_base64}"},
            },
        ],
    )
    #prompt = prompt_template.format(closest_color_name=closest_color_name, user_preference=user_preference, context=retriever)
    # RAG 체인 생성
    #chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

    # 프롬프트와 이미지를 함께 전송하여 응답 받기
    response = llm.invoke([system_message, human_message])

    return response.content


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
def chat_interface():
    print("안녕하세요! 인테리어 및 가구 색상 추천 챗봇입니다.")

    mode = input("추천을 받고 싶은 스타일의 사진을 입력하려면 1을, 비슷한 가구를 찾고 싶으면 2를 입력해주세요: ")

    if mode == '1':
        # 스타일 추천을 받고 싶은 경우
        p_number = input("추천을 받고 싶은 인테리어 또는 가구의 이미지를 업로드해주세요: ")
        image_path = f'./style{p_number}.jpeg'
        user_preference = input("선호하는 색상 또는 스타일을 입력해주세요: ")

        print("\n사용자 이미지\n")
        img = Image.open(image_path)
        # 이미지 표시
        plt.imshow(img)
        plt.axis('off')  # 축 없애기
        plt.show()
        print("*****완료*****\n")

        # 색상 및 인테리어 구성 방식 추천
        final_chain = (
                vision_chain | StrOutputParser()
        )
        res = final_chain.invoke({"image_path": image_path, "user_preference": user_preference, "context": color_retriever, "context2": furniture_retriever})

        # 추천 결과 출력
        print(f"### 추천 결과: {res}")

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
chat_interface()
