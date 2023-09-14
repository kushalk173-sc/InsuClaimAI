# 🚗 InsuclaimAI: Simplifying Insurance Claims 📝

## 🌟 Inspiration
💡 After a frustrating experience with a minor fender-bender, I was faced with the overwhelming process of filing an insurance claim. Filling out endless forms, speaking to multiple customer service representatives, and waiting for assessments felt like a second job. That's when I knew that there needed to be a more streamlined process. Thus, InsuclaimAI was conceived as a solution to simplify the insurance claim maze.

## 🎓 What I Learned

### 🛠 Technologies
#### 📖 OCR (Optical Character Recognition)
- OCR technologies like OpenCV helped in scanning and reading textual information from physical insurance documents, automating the data extraction phase.

#### 🧠 Machine Learning Algorithms (CNN)
- Utilized Convolutional Neural Networks to analyze and assess damage in photographs, providing an immediate preliminary estimate for claims.

#### 🌐 API Integrations
- Integrated APIs from various insurance providers to automate the claims process. This helped in creating a centralized database for multiple types of insurance.

### 🌈 Other Skills
#### 🎨 Importance of User Experience
- Focused on intuitive design and simple navigation to make the application user-friendly.

#### 🛡️ Data Privacy Laws
- Learned about GDPR, CCPA, and other regional data privacy laws to make sure the application is compliant.

#### 📑 How Insurance Claims Work
- Acquired a deep understanding of the insurance sector, including how claims are filed, and processed, and what factors influence the approval or denial of claims.

## 🏗️ How It Was Built

### Step 1️⃣: Research & Planning
- Conducted market research and user interviews to identify pain points.
- Designed a comprehensive flowchart to map out user journeys and backend processes.

### Step 2️⃣: Tech Stack Selection
- After evaluating various programming languages and frameworks, Python, TensorFlow, and Flet (From Python) were selected as they provided the most robust and scalable solutions.

### Step 3️⃣: Development
#### 📖 OCR
- Integrated Tesseract for OCR capabilities, enabling the app to automatically fill out forms using details from uploaded insurance documents.

#### 📸 Image Analysis
- Exploited an NLP model trained on thousands of car accident photos to detect the damages on automobiles.

#### 🏗️ Backend
##### 📞 Twilio
- Integrated Twilio to facilitate voice calling with insurance agencies. This allows users to directly reach out to the Insurance Agency, making the process even more seamless.

##### ⛓️ Aleo
- Used Aleo to tokenize PDFs containing sensitive insurance information on the blockchain. This ensures the highest levels of data integrity and security. Every PDF is turned into a unique token that can be securely and transparently tracked.

##### 👁️ Verbwire
- Integrated Verbwire for advanced user authentication using FaceID. This adds an extra layer of security by authenticating users through facial recognition before they can access or modify sensitive insurance information.

#### 🖼️ Frontend
- Used Flet to create a simple yet effective user interface. Incorporated feedback mechanisms for real-time user experience improvements.

## ⛔ Challenges Faced
#### 🔒 Data Privacy
- Researching and implementing data encryption and secure authentication took longer than anticipated, given the sensitive nature of the data.

#### 🌐 API Integration
- Where available, we integrated with their REST APIs, providing a standard way to exchange data between our application and the insurance providers. This enhanced our application's ability to offer a seamless and centralized service for multiple types of insurance.

#### 🎯 Quality Assurance
- Iteratively improved OCR and image analysis components to reach a satisfactory level of accuracy. Constantly validated results with actual data.

#### 📜 Legal Concerns
- Spent time consulting with legal advisors to ensure compliance with various insurance regulations and data protection laws.

## 🚀 The Future
👁️ InsuclaimAI aims to be a comprehensive insurance claim solution. Beyond just automating the claims process, we plan on collaborating with auto repair shops, towing services, and even medical facilities in the case of personal injuries, to provide a one-stop solution for all post-accident needs.
