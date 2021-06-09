#!/usr/bin/env python
# coding: utf-8

# In[108]:


from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
#import PyPDF2


# In[109]:


#순서 문제 지문 완벽화 
def type_order(text, answer):
    first_english = False
    C_chunk = False
    temp_str = ""
    alphabet = ''
    paren = ''
    chunk = []
    perfect_text = ""
    index = 0
    
    # 문제에서 시작 문장, A,B,C 지문 chunk 추출하여 chunk에 저장. 
    while index < len(text):
        if ord('a') <= ord(text[index].lower()) <= ord('z'):
            first_english = True
        if first_english:
            if text[index] == '(':
                alphabet = text[index+1]
                paren = text[index+2]
                if alphabet == 'A' and paren == ')':
                    print(temp_str)
                    chunk.append(temp_str)
                    temp_str = ""
                    index += 3
                    continue
                elif(alphabet == 'B' and paren == ')'):
                    chunk.append(temp_str)
                    temp_str = ""
                    index += 3
                    continue
                elif(alphabet == 'C' and paren == ')'):
                    chunk.append(temp_str)
                    temp_str = ""
                    index += 3
                    C_chunk = True
                    continue
            elif text[index] == '\n':
                if(C_chunk):
                    if(text[index+1] == '\n'):
                        break
                index += 1
                continue
            temp_str += text[index]
            first_english = True
        index += 1
    chunk.append(temp_str)
    
    #시작 문장 insert
    pefect_text = chunk[0]
    #A-C-B
    if answer == 1:
        perfect_text += chunk[1]
        perfect_text += chunk[3]
        perfect_text += chunk[2]
    #B-A-C
    elif answer == 2:
        perfect_text += chunk[2]
        perfect_text += chunk[1]
        perfect_text += chunk[3]
    #B-C-A
    elif answer == 3:
        perfect_text += chunk[2]
        perfect_text += chunk[3]
        perfect_text += chunk[1]
    #C-A-B
    elif answer == 4:
        perfect_text += chunk[3]
        perfect_text += chunk[1]
        perfect_text += chunk[2]
    #C-B-A
    elif answer == 5:
        perfect_text += chunk[3]
        perfect_text += chunk[2]
        perfect_text += chunk[1]
        
    return perfect_text


# In[110]:


#삽입 문제 지문 완벽화
def type_insert(text, answer):
    index = 0; ans_match = 0
    perfect_text = ""; to_insert = ""

#삽입할 문장 시작점 찾기
    for i in range(len(text)):
        start_flag = True
        if text[i].encode().isalpha() == False:
            continue
        else:
            for j in range(1,10):
                if (ord('가') <= ord(text[j+i]) <= ord('힣')):
                    start_flag = False
                    break
                else:
                    continue
            if start_flag == True:
                break
#삽입할 문장 저장, 본문 시작점 찾기  
    for j in range(i,len(text)):
        if text[j] != '\n':
            to_insert += text[j]
            continue
        elif text[j] == '\n':
            if text[j+1] == '\n':
                for k in range(j+2,len(text)):
                    if text[k] != '\n':
                        break
                break
            else:
                continue   
    print("삽입할문장: ",to_insert+'\n') #삽입할 문장 맞는지 확인용 

    # k: 본문 시작점 인덱스, text에 본문만 다시 저장한다.
    text = text[k:] 
    # 지문 시작부분에 공백이 있으면 제거.
    for i in range(len(text)):
        if text[i].isalnum() == False:
            text = text[i+1:]
            if text[i+1].isalnum() == False:
                continue
            else:
                break
                    
#삽입해서 완성하기                
    tmp_text = ""
    cnt = 1 # 현재 처리 중인 (번호)를 저장

    i = 0
    while (i < len(text)):
     #(번호) 부분 처리   
        if text[i] == '(': 
            for j in range(i,i+10):
                if text[j] ==  ')': 
                    for k in range(j+1,j+10): # ')' 뒤에 또 ')'로 인식하는 경우
                        if text[k] == ')':
                            j = k # 뒤에 나온 ')'가 진짜 괄호이므로 그 까지 지워야 함
                            break #')' 다음에 또 바로 '('가 나올 때가 있을까?
                    if cnt == answer: # 현재 (번호) 가 정답일 경우
                        tmp_text = text[:i] + to_insert +text[j+1:]
                        text = tmp_text
                        cnt += 1 
                        break
                    else: #  현재 (번호)가정답이 아닐경우
                        tmp_text = text[:i] + text[j+1:]
                        text = tmp_text
                        cnt += 1
                        break
        i += 1
    #print(text)
    # 공백과 개행 제거
    text = text.strip().replace("\n","")
    return text


# In[111]:


#빈칸 문제 지문 완벽화 
def type_blank(text, answer):
    perfect_text = []
    return perfect_text


# In[112]:


'''
def scan_img(text,answer,total_img_num):
    i = 0
    for i in range(total_img_num):
        img = Image.open("input_insert"+str(i)+".png")
        one_text = pytesseract.image_to_string(img, lang='kor+eng')
        text.append(one_text)
        answer.append(2)
'''
def scan_img(text):
    img = Image.open("input_insert3.jpg")#이미지 파일 이름은 OCR 엔진과 연계해서 다시 생각해보기
    text_temp = pytesseract.image_to_string(img, lang='kor+eng')
    text += text_temp
    
    return text


# In[113]:


''' old version. 사용자로부터 문제 유형을 입력 받기 때문에 유형을 코드로 판별할 필요 없음.

#읽어들인 지문 정제. 다수의 텍스트에서 유형 판별하여 해당 함수 호출. 
def text_refinement(text, answer):
    i = 0
    blank_flag = False
    order_flag = False
    insert_flag = False
    for i in range(len(text)):
        print(text[i])
        for j in text[i] :
            #빈칸 확인
            if j == '빈':
                blank_flag = True
                continue
            if blank_flag:
                if(j == '칸'):
                    text[i] = type_blank(text[i],answer[i])
                    break
                else:
                    blank_flag = False
            #순서 확인
            if j == '어':
                order_flag = True
                continue
            if order_flag:
                if(j == '질'):
                    text[i] = type_order(text[i],answer[i])
                    print(text[i])
                    break
                else:
                    order_flag = False
            #삽입 확인
            if (j == '흐'):
                insert_flag = True
                continue
            if insert_flag:
                if(j == '름'):
                    text[i] = type_insert(text[i],answer[i])
                    print(text[i])
                    break
                else:
                    insert_flag = False
'''
def text_refinement(text, text_type, answer):
    if text_type == 0: # 순서
        text = type_order(text,answer) 
    elif text_type == 1: # 삽입
        text =  type_insert(text,answer) 
    elif text_type == 2: # 빈칸
        text =  type_blank(text,answer)
    else:
        print("text type input error\n")
    return text


# In[114]:


def summarize_T5(text):
    import torch
    import json 
    import time
    from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

    time0 = time.time()      
    
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    device = torch.device('cpu')

    preprocess_text = text.strip().replace("\n","")
    t5_prepared_Text = "summarize: "+preprocess_text
    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
    
    # summmarize 
    summary_ids = model.generate(tokenized_text,
                                    num_beams=4,
                                    no_repeat_ngram_size=2,
                                    min_length=30,
                                    max_length=100,
                                    early_stopping=True)

    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    print("\n\nelapsed time for summarize: \n", time.time() - time0, "seconds")  
    
    return output


# In[115]:


#main
text = ""
answer = 5
#for i range(): 여러 개 지문 scan 할 필요 있을 시 아래 전부 반복 시키면 됨.
text_type = 1 # 문제 유형 입력 scan
#print(scan_img(text)) #손대지 않은 OCR 결과만 출력
text = text_refinement(scan_img(text), text_type, answer)
print(text)

summarized_text = summarize_T5(text)
print ("\nSummarized text: \n" + summarized_text)


# In[116]:


jupyter nbconvert --to script filename.ipynb


# In[ ]:




