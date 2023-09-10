# -- coding: utf-8 --
"""
Created on Sat Sep  9 16:43:27 2023

@author: Kushal
"""

import flet as ft
import json
import requests
import re
import cv2
import face_recognition
import pickle
import sys
import requests
import time

import html
import numpy as np
import math
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import os
from roboflow import Roboflow
import PyPDF2
from PIL import Image


from twilio.rest import Client



def main(page: ft.Page):
    
        
    kv_model = Word2Vec.load("model/model-50d.h5").wv
    
    stop_words = set(stopwords.words('english'))
    
    
    client = Client(account_sid, auth_token)
    

    def make_call():
        global karen_text
        print(karen_text)
        client.calls.create(
            twiml=f'''<Response><Say>
{karen_text}
                                </Say></Response>''',
            to='+12404137915',
            from_='+19139386952'
        )

    
    
    def extract_pdf_contents(file_path):
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            contents = ""
            for page_num in range(len(pdf_reader.pages)-1):
                page = pdf_reader.pages[page_num]
                contents += page.extract_text()
    
        return contents
    
    
    def preprocess(file):
        current_directory = os.getcwd()
        folder_name = "insurance"
        folder_path = os.path.join(current_directory, folder_name)
        file_path = folder_path + '/' + file
        extracted_contents = extract_pdf_contents(file_path)
        return extracted_contents
    
    
    def remove_disclaimers(sentences):
        # Tokenize the text into sentences
        
        # List of common disclaimer keywords to check against
        disclaimer_keywords = ['disclaimer','herein','typically' ,'forward-looking statements', 'company','risk factors', 'legal notice', 'accounting policies', 'accounting principles','form 10-k', 'form 8-k' ,'this form 10-q', 'this quarterly report']
        
        # Initialize a new list to store non-disclaimer sentences
        non_disclaimer_sentences = []
        
        # Iterate through each sentence
        for sentence in sentences:
            # Convert the sentence to lowercase for case-insensitive matching
            lower_sentence = sentence.lower()
            
            # Check if any disclaimer keywords exist in the sentence
            if any(keyword in lower_sentence for keyword in disclaimer_keywords):
                continue  # Skip sentences containing disclaimers
            else:
                non_disclaimer_sentences.append(sentence)
        
        # Join the non-disclaimer sentences to form the final text
        
        return non_disclaimer_sentences
    
    def extract_keywords_sentences(sentences, stop_words):
        s1 = [[word.lower() for word in re.sub(r'[^\w\s]', '', sentence).split()] for sentence in sentences]
        s1 = [[word for word in tokens if word not in stop_words] for tokens in s1]
        return s1
    
    def extract_keywords_question(question, stop_words):
        tokens = word_tokenize(question)
        tokens = [word.lower() for word in tokens]
        tokens = [word for word in tokens if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return tokens
    
    def text_extractor(sentences, question, follow_up_count=3):
        ques = extract_keywords_question(question, stop_words)
        s1 = extract_keywords_sentences(sentences, stop_words)
        
        indexsen = 0
        dictScore = dict()
        
        # Calculate similarity scores for each sentence
        for sentence in s1:
            sum_score = 0
            for word in sentence:
                for word1 in ques:
                    try:
                        sum_score += (np.dot(kv_model[word], kv_model[word1])) / 0.5
                    except:
                        continue
            dictScore[indexsen] = sum_score
            indexsen += 1
        
        sorted_scores = sorted(dictScore.items(), key=lambda x: x[1], reverse=True)
        
        top_2_percent_index = math.ceil(len(sorted_scores) * 0.02)
        threshold_score = sorted_scores[top_2_percent_index - 1][1] if top_2_percent_index > 0 else 0
        
        wordCount = 0
        unique_sentences = set()
        final_text = ''
        
        for index, score in sorted_scores:
            if score < threshold_score:
                continue
                
            current_sentence = sentences[index]
            if current_sentence in unique_sentences:
                continue
                
            unique_sentences.add(current_sentence)
            final_text += current_sentence + " "
            wordCount += len(word_tokenize(current_sentence))
            
            for i in range(1, follow_up_count + 1):
                next_index = index + i
                if next_index < len(sentences):
                    next_sentence = sentences[next_index]
                    if next_sentence in unique_sentences:
                        continue
                    
                    unique_sentences.add(next_sentence)
                    final_text += next_sentence + " "
                    wordCount += len(word_tokenize(next_sentence))
            
            if wordCount >= 5000:
                break
    
        return final_text.strip()
    
    def pdf_converter(filename):
        file = preprocess(filename)
        decoded_text = html.unescape(file)
        new_string = re.sub(r"(.?\|.?\|.*?)\s+(\d+)", "", decoded_text)
        new_string = new_string.encode("utf-8").decode("utf-8").replace("\xa0", " ")
        new_string = new_string.replace("\r", "\n")
        new_string = new_string.replace(". ", ". \n")
        sentences = re.split(r'\n+', new_string)
        return  remove_disclaimers(sentences)
    
    def detect_damage(name):
    # load the image from disk and preprocess it
        current_directory = os.getcwd()
        folder_name = "insurance"
        folder_path = os.path.join(current_directory, folder_name)
        PATH = folder_path + '/' + name

        image = Image.open(PATH)
        try:
            image.verify()
        except:
            return False
        
        rf = Roboflow(api_key="tF51WFIm1hXPxzxz8KFZ")
        project = rf.workspace().project("car-damage-test-lqgll")
        model = project.version(1).model

        data = model.predict(PATH, confidence=14, overlap=30).json()
        # store the useful data in a list.
        classes = [prediction['class'] for prediction in data['predictions']]
        print("Damage is: ", classes[0])
        return ', '.join(classes)

        
    
    def claims_filer(policy, damage, notes):
    
        instructions = """
    I want you to act as an Insurance Claims filer. Your job is to now file the most detailed claim one has ever seen. Here are the derails of the incident
   
    Keeo your generation within 600 tokens
    
    The format you must follow is going to be "CLAIM\n\n:"
    
    """
        prompt = """
        Instructions:
        {instructions}
    
        Here is the Information about the damages :
        {damages}
    
        Here is some additional information about what happened:
        {notes}
    
        Here is the relevant parts of the policy:
        {policy}
    
        """.format(policy = policy, damages = damage, notes = notes, instructions=instructions)
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
          
        }
        data = {
            "model": "gpt-3.5-turbo-16k",
            "messages": [
                    {"role": "user", "content": prompt},
        
            ]
        }
        
        response = requests.post(url, headers=headers, data=json.dumps(data))
        result = response.json()
        print(result)
        summary = result['choices'][0]['message']['content']
        print(summary)
        print('-'*150)
        return summary
    
    
    
    def karen_generate(policy, damage, notes):
    
        instructions = """
        
        NO WRITING
        
        You are now Karen, one of the top Arguers in teh country. I am facing this massive issue where the insurance company is not giving me my claim. Write a stern reply to them like how karen would
        
        
    Follow the format :
    "KAREN \n\n: "
    
    This option will be mostly exercised because there was an unnacceptable delay or a problem with the insurance and they are not claiming. There is more information mentioned in the notes/ 
    
    """
        prompt = """
        Instructions:
        {instructions}
    
        Here is the Information about the damages :
        {damages}
    
        Here is some additional information about what happened:
        {notes}
    
        Here is the relevant parts of the policy:
        {policy}
    
        """.format(policy = policy, damages = damage, notes = notes, instructions=instructions)
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            
        }
        data = {
            "model": "gpt-3.5-turbo-16k",
            "messages": [
                    {"role": "user", "content": prompt},
        
            ]
        }
        
        response = requests.post(url, headers=headers, data=json.dumps(data))
        result = response.json()
        
        summary = result['choices'][0]['message']['content']
   
        return summary

    def pick_insurance(e: ft.FilePickerResultEvent):
        insurance_doc.value = (
            ", ".join(map(lambda f: f.name, e.files)) if e.files else "Cancelled!"
        )
        insurance_doc.update()
    
    def pick_photos(e: ft.FilePickerResultEvent):
        photo.value = (
            ", ".join(map(lambda f: f.name, e.files)) if e.files else "Cancelled!"
        )
        photo.update()
    
    test_name = ft.Text()
    api_txt = ft.Text()

    pick_insurance_dialog = ft.FilePicker(on_result=pick_insurance)
    insurance_doc = ft.Text()

    pick_photo_dialog = ft.FilePicker(on_result=pick_photos)
    photo = ft.Text()
    
    claim = False
    karen = False
    karen_text = ''
    
    def sign_in():
        cap = cv2.VideoCapture(1)
        try:
            with open('face_data.dat', 'rb') as f:
                known_face_data = pickle.load(f)
            known_face_encodings = known_face_data['encodings']
            known_face_labels = known_face_data['labels']
        except FileNotFoundError:
            known_face_encodings = []
            known_face_labels = []

        # Capture frame-by-frame
        while True:
            ret, frame = cap.read()
            if frame is not None:
                # Find all the faces and their encodings in the current frame
                face_locations = face_recognition.face_locations(frame)
                face_encodings = face_recognition.face_encodings(frame, face_locations)

                # Initialize matches as an empty list
                matches = []

                # Loop through each face found in the frame
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Compare the current face encoding with the known face encodings
                    matches = face_recognition.compare_faces(
                    known_face_encodings, face_encoding)

                # If a match is found, display a welcome message and stop capturing video
                if True in matches:
                    label = known_face_labels[matches.index(True)]
                    print("Authentication successful! Welcome, " + label + "!")
                    cv2.putText(frame, "Authentication successful! Welcome, " + label + "!",
                                (left-10, bottom+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cap.release()
                    cv2.destroyAllWindows()
                    return 1
                else:
                    print("Face Not Found!\nTry Again!")
                    time.sleep(0.5)

                
            else:
                print("Waiting for webcam...")
                continue


    def sign_up(name):
        cap = cv2.VideoCapture(0)
        try:
            with open('face_data.dat', 'rb') as f:
                known_face_data = pickle.load(f)
            known_face_encodings = known_face_data['encodings']
            known_face_labels = known_face_data['labels']
        except FileNotFoundError:
            known_face_encodings = []
            known_face_labels = []

        while True:
            ret, frame = cap.read()
            if frame is not None:
                # Find all the faces and their encodings in the current frame
                face_locations = face_recognition.face_locations(frame)
                face_encodings = face_recognition.face_encodings(frame, face_locations)

                # Prompt user to enter a label for the face
                if len(face_encodings) > 0:
                # cv2.imshow('Video', frame)
                    label = name
                    known_face_encodings.append(face_encodings[0])
                    known_face_labels.append(label)

                # Save known face encodings and corresponding labels to file
                with open('face_data.dat', 'wb') as f:
                    pickle.dump({'encodings': known_face_encodings,
                                'labels': known_face_labels}, f)
                    print("Saved the Face on file!")
                    f.close()
                    cap.release()
                    cv2.destroyAllWindows()
                    return 1
            else:
                print("Waiting for webcam...")
                continue


    # Verbwire
    def mint_custom_nft(data, username):
        url = "https://api.verbwire.com/v1/nft/mint/mintFromMetadata"

        payload = f"""\
    -----011000010111000001101001\r
    Content-Disposition: form-data; name="quantity"\r
    \r
    1\r
    -----011000010111000001101001\r
    Content-Disposition: form-data; name="chain"\r
    \r
    goerli\r
    -----011000010111000001101001\r
    Content-Disposition: form-data; name="contractAddress"\r
    \r
    0xc83E1Dad8fC1A872420154dFbb5b318aaf769940\r
    -----011000010111000001101001\r
    Content-Disposition: form-data; name="data"\r
    \r
    {data}\r
    -----011000010111000001101001\r
    Content-Disposition: form-data; name="recipientAddress"\r
    \r
    0x717aeB89048f10061C0dCcdEB2592a60bA4F1a79\r
    -----011000010111000001101001\r
    Content-Disposition: form-data; name="name"\r
    \r
    {username}\r
    -----011000010111000001101001--\r
    """
        headers = {
            "accept": "application/json",
            "content-type": "multipart/form-data; boundary=---011000010111000001101001",
            
        }

        response = requests.post(url, data=payload, headers=headers)

        return response.text

    # Verbwire
    def get_nft_attributes():
        url = "https://api.verbwire.com/v1/nft/data/owned?walletAddress=0x717aeB89048f10061C0dCcdEB2592a60bA4F1a79&chain=goerli"
        headers = {
            "accept": "application/json",
            
        }
        response = requests.get(url, headers=headers)

        data = response.json()
        token_attributes = []

        for nft in data['nfts']:
            contract_address = nft['contractAddress']
            if contract_address == "0xc83E1Dad8fC1A872420154dFbb5b318aaf769940".lower():
                token_id = nft["tokenID"]
                chain = "goerli"
                url_inner = f"https://api.verbwire.com/v1/nft/data/nftDetails?contractAddress={contract_address}&tokenId={token_id}&chain={chain}"
                headers = {
                    "accept": "application/json",
                    "X-API-Key": "sk_live_b7159a98-601c-455e-b0e8-fd8cb42b48b3"
                }
                url_resp = requests.get(url_inner, headers=headers)
                json_data = url_resp.json()
                token_uri = json_data['nft_details']['tokenURI']
                main_resp = requests.get(token_uri).json()
                token_attributes.appends(main_resp['attributes'])
        return token_attributes


    def claim_func(e):
        api_txt.value = "LOADING......"
        api_txt.update()

        global claim
        claim = True
        print(os.getcwd())
        print("one")
        policy = preprocess(insurance_doc.value)
        print("two")
        damage = detect_damage(photo.value)
        print(damage)
        print("three")
        notes = test_name.value 
        print(notes)
        print('four')
        decoded_text = html.unescape(policy)
        new_string = re.sub(r"(.?\|.?\|.*?)\s+(\d+)", "", decoded_text)
        new_string = new_string.encode("utf-8").decode("utf-8").replace("\xa0", " ")
        new_string = new_string.replace("\r", "\n")
        new_string = new_string.replace(". ", ". \n")
        print('four')

        # Split the management commentary section into sentences
        sentences = re.split(r'\n+', new_string)
        print('four')

        x = remove_disclaimers(sentences)
        summary = text_extractor(x, damage)
        print('four')
        claims_text = claims_filer(summary, damage, notes)
        print(claims_text)
        api_txt.value = claims_text
        api_txt.update()
    
    def karen_func(e):
        api_txt.value = "LOADING......"
        api_txt.update()
        print(os.getcwd())
        print("one")
        policy = preprocess(insurance_doc.value)
        print("two")
        damage = detect_damage(photo.value)
        print(damage)
        print("three")
        notes = test_name.value 
        print(notes)
        print('four')
        decoded_text = html.unescape(policy)
        new_string = re.sub(r"(.?\|.?\|.*?)\s+(\d+)", "", decoded_text)
        new_string = new_string.encode("utf-8").decode("utf-8").replace("\xa0", " ")
        new_string = new_string.replace("\r", "\n")
        new_string = new_string.replace(". ", ". \n")
        print('four')

        # Split the management commentary section into sentences
        sentences = re.split(r'\n+', new_string)
        print('four')

        x = remove_disclaimers(sentences)
        summary = text_extractor(x, damage)
        print('four')
        claims_text = karen_generate(summary, damage, notes)
        print(claims_text)
        api_txt.value = claims_text
        api_txt.update()        
        global karen_text
        karen_text = claims_text
        global claim
        claim = True
        global karen
        karen = True
    


    def file_claim(e):
        
        global claim
        if claim == False:
            return
        

        api_txt.value = ('Authenticating Your Face .... \n We will authenticate you securely using your face before letting you file the claim. \n The system uses the nft stored in your Verbwire account to authenticate your identity')
        api_txt.update()
        sign_in()
        api_txt.value = ('Sending Claim .... \n We will send your claim via a smart contract to the insurance company ')
        api_txt.update()
        damage = detect_damage(photo.value)
        print(damage)
        claim = False
   
        
    def call(e):
        global claim
        global karen
        print(claim)
        print(karen)
        if claim == False or karen == False:
            return 
        api_txt.value = ('Calling......')
        api_txt.update()
        make_call()
        claim = False
        karen = False


   

    def btn_click(e):
        if not txt_name.value:
            txt_name.error_text = "Further Details about the accident"
            page.update()
        else:
            name = txt_name.value
            test_name.value = (f"{name}")
            test_name.update()

    txt_name = ft.TextField(label="Add Notes")
    page.overlay.append(pick_insurance_dialog)
    page.overlay.append(pick_photo_dialog)


    page.add(
 ft.Row(
        [
            ft.Column(
                
                [
                ft.Row(
                    [
                        ft.ElevatedButton(
                            "Upload Insurance Document",
                            icon=ft.icons.UPLOAD_FILE,
                            on_click=lambda _: pick_insurance_dialog.pick_files(
                                allow_multiple=True
                            ),
                        ),
                        insurance_doc,
                    ]
                ),
                ft.Row(
                    [
                        ft.ElevatedButton(
                            "Upload Photos",
                            icon=ft.icons.UPLOAD_FILE,
                            on_click=lambda _: pick_photo_dialog.pick_files(
                                allow_multiple=True
                            ),
                        ),
                        photo,
                    ]
                ),
                
                ft.Column(
                    [
                       txt_name, 
                       ft.ElevatedButton("Upload Your Notes!", on_click=btn_click),
                       ft.Container(
                           width = 350,
                           height = 120,
                           content = test_name
                           ),
                    ]
                ),
                
                ft.Row(
                    
                    [
                        ft.ElevatedButton("KAREN!", on_click=karen_func),
                ft.ElevatedButton("CLAIM!", on_click=claim_func),

],
                    alignment=ft.MainAxisAlignment.CENTER,
   
                    
                    ),
                
                
                ft.Row(
                     
                    [

                        ft.ElevatedButton("File Claim!", on_click=file_claim, disabled = claim),
                    ft.ElevatedButton("Call!", on_click=call),

],
                    alignment=ft.MainAxisAlignment.CENTER,
   
                    
                    )
                
                
                ],
                

                ),
            
            ft.Container(
                width = 30
                ),
            
            ft.Container(
                    margin=10,
    padding=10,

                
                width = 800,
                height = 750,
                content = api_txt,
                ),
            ],
        alignment=ft.MainAxisAlignment.CENTER,

     ),
    )

ft.app(target=main)