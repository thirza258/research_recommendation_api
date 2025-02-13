from django.shortcuts import render
import urllib
import urllib.request
import xml.etree.ElementTree as ET
import json
import os
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content
from mistralai import Mistral
import requests
import numpy as np
import faiss
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import time
import random
import re

# Create your views here.

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
mistral_client = Mistral(api_key=MISTRAL_API_KEY)

def fetch_arxiv_data(query):
    encoded_query = urllib.parse.quote(query)
    
    url = f'http://export.arxiv.org/api/query?search_query=all:{encoded_query}'
    
    with urllib.request.urlopen(url) as response:
        xml_string = response.read().decode('utf-8')
    
    return xml_string
    
def xml_to_json(xml_string):
    if not xml_string.strip():
        raise ValueError("Empty XML string provided.")
    
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError as e:
        raise ValueError(f"XML Parsing Error: {e}")

    entries = []
    
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        data = {
            "id": entry.find("{http://www.w3.org/2005/Atom}id").text,
            "updated": entry.find("{http://www.w3.org/2005/Atom}updated").text,
            "published": entry.find("{http://www.w3.org/2005/Atom}published").text,
            "title": entry.find("{http://www.w3.org/2005/Atom}title").text,
            "summary": entry.find("{http://www.w3.org/2005/Atom}summary").text.strip(),
            "authors": [author.find("{http://www.w3.org/2005/Atom}name").text for author in entry.findall("{http://www.w3.org/2005/Atom}author")],
            "doi": entry.find("{http://arxiv.org/schemas/atom}doi").text if entry.find("{http://arxiv.org/schemas/atom}doi") is not None else None,
            "journal_ref": entry.find("{http://arxiv.org/schemas/atom}journal_ref").text if entry.find("{http://arxiv.org/schemas/atom}journal_ref") is not None else None,
            "pdf_url": next((link.attrib["href"] for link in entry.findall("{http://www.w3.org/2005/Atom}link") if link.attrib.get("title") == "pdf"), None)
        }
        entries.append(data)
    
    return json.dumps(entries, indent=2)

def get_text_embedding(input):
    embeddings_batch_response = mistral_client.embeddings.create(
          model="mistral-embed",
          inputs=input
    )
    return embeddings_batch_response.data[0].embedding

def fetch_with_retries(query, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            xml_string = fetch_arxiv_data(query)
            return json.loads(xml_to_json(xml_string))
        except Exception as e:
            if "429" in str(e):  # Detect rate limit error
                wait_time = 2 ** retries  # Exponential backoff (2, 4, 8, ...)
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
            else:
                raise e

class DayRecommencation(APIView):
    def get(self, request):
        try:
            generation_config = {
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
                "response_schema": content.Schema(
                    type = content.Type.OBJECT,
                    properties = {
                    "keyword": content.Schema(
                        type = content.Type.ARRAY,
                        items = content.Schema(
                        type = content.Type.STRING,
                        ),
                    ),
                    },
                ),
                "response_mime_type": "application/json",
                }
            model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=generation_config,
            system_instruction="\n",
            )

            chat_session = model.start_chat(
            history=[
                {
                "role": "user",
                "parts": [
                    "generate 1 research keyword of the day",
                ],
                },
                {
                "role": "model",
                "parts": [
                    "```json\n{\n  \"keyword\": [\"quantum computing\"]\n}\n```",
                ],
                },
            ]
            )

            response = chat_session.send_message("generate 1 research keyword of the day")

            # Convert response.text from JSON string to dictionary
            response_data = json.loads(response.text)

            # Extract the first keyword
            keyword = response_data["keyword"][0]

            print(keyword)

            # Use the extracted keyword
            json_data = fetch_with_retries(keyword)
            
            return Response({
                    "status": 200,
                    "message": "Success",
                    "data": json_data,
                    "keyword": keyword
                }, status=status.HTTP_200_OK)
        except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
class ResearchRecommendation(APIView):
    def post(self, request):
        search_request = request.data.get("search_request")
        try:
            generation_config = {
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
                "response_schema": content.Schema(
                type = content.Type.OBJECT,
                properties = {
                    "keyword": content.Schema(
                    type = content.Type.ARRAY,
                    items = content.Schema(
                        type = content.Type.STRING,
                    ),
                    ),
                },
                ),
                "response_mime_type": "application/json",
            }

            model = genai.GenerativeModel(
                model_name="gemini-2.0-flash-exp",
                generation_config=generation_config,
                system_instruction="Based on the prompt generate 3 keywords correlated\n",
            )

            chat_session = model.start_chat(
                history=[
                {
                    "role": "user",
                    "parts": [
                    "How to create the AI",
                    ],
                },
                {
                    "role": "model",
                    "parts": [
                    "```json\n{\n\"keyword\": \"development\"\n}\n```",
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                    "How to create AI",
                    ],
                },
                {
                    "role": "model",
                    "parts": [
                    "```json\n{\n\"keyword\": [\"artificial intelligence\", \"machine learning\", \"neural networks\", \"algorithms\", \"data\", \"programming\", \"training\", \"models\", \"deep learning\"]\n}\n```",
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                    "How to create AI ",
                    ],
                },
                {
                    "role": "model",
                    "parts": [
                    "```json\n{\n\"keyword\": [\"artificial intelligence\", \"machine learning\", \"deep learning\", \"neural networks\", \"algorithms\"]\n}\n```",
                    ],
                },
                ]
            )

            response = chat_session.send_message(request.data.get("search_request"))

            day_recommendation = json.loads(response.text)
            
            data = []
          
            for i in day_recommendation["keyword"]:
                time.sleep(3) 
                json_data = fetch_with_retries(i)

                if isinstance(json_data, list):  
                    data.extend(json_data)  
                else:
                    data.append(json_data) 
            
            # text_inputs = [json.dumps(item) for item in data]  # Ensure input is a valid string

            # text_embeddings = np.array([get_text_embedding(i) for i in text_inputs])

            # d = text_embeddings.shape[1]
            # index = faiss.IndexFlatL2(d)
            # index.add(text_embeddings)

            # question_embedding = np.array([get_text_embedding(json.dumps(search_request))])  # Convert question to JSON string
            # print(question_embedding.shape)
            # D, I = index.search(question_embedding, k=5)

            # retrieved_data = [data[i] for i in I[0]]
            
            generation_config = {
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
                "response_schema": content.Schema(
                    type = content.Type.OBJECT,
                    properties = {
                    "response": content.Schema(
                        type = content.Type.ARRAY,
                        items = content.Schema(
                        type = content.Type.OBJECT,
                        properties = {
                            "title": content.Schema(
                            type = content.Type.STRING,
                            ),
                            "summary": content.Schema(
                            type = content.Type.STRING,
                            ),
                            "url": content.Schema(
                            type = content.Type.STRING,
                            ),
                        },
                        ),
                    ),
                    },
                ),
                "response_mime_type": "application/json",
                }

            model = genai.GenerativeModel(
                model_name="gemini-2.0-flash-exp",
                generation_config=generation_config,
                system_instruction="\n",
            )

            chat_session = model.start_chat(
                history=[
                    {
                    "role": "user",
                    "parts": [
                        "add the summary",
                    ],
                    },
                    {
                    "role": "model",
                    "parts": [
                        "```json\n{\n  \"response\": [\n    {\n      \"summary\": \"To fight climate change and accommodate the increasing population, global\\ncrop production has to be strengthened. To achieve the \\\"sustainable\\nintensification\\\" of agriculture, transforming it from carbon emitter to carbon\\nsink is a priority, and understanding the environmental impact of agricultural\\nmanagement practices is a fundamental prerequisite to that. At the same time,\\nthe global agricultural landscape is deeply heterogeneous, with differences in\\nclimate, soil, and land use inducing variations in how agricultural systems\\nrespond to farmer actions. The \\\"personalization\\\" of sustainable agriculture\\nwith the provision of locally adapted management advice is thus a necessary\\ncondition for the efficient uplift of green metrics, and an integral\\ndevelopment in imminent policies. Here, we formulate personalized sustainable\\nagriculture as a Conditional Average Treatment Effect estimation task and use\\nCausal Machine Learning for tackling it. Leveraging climate data, land use\\ninformation and employing Double Machine Learning, we estimate the\\nheterogeneous effect of sustainable practices on the field-level Soil Organic\\nCarbon content in Lithuania. We thus provide a data-driven perspective for\\ntargeting sustainable practices and effectively expanding the global carbon\\nsink.\",\n      \"title\": \"Personalizing Sustainable Agriculture with Causal Machine Learning\",\n      \"url\": \"http://arxiv.org/pdf/2211.03179v1\"\n    },\n    {\n      \"summary\": \"This survey investigates the transformative potential of various YOLO\\nvariants, from YOLOv1 to the state-of-the-art YOLOv10, in the context of\\nagricultural advancements. The primary objective is to elucidate how these\\ncutting-edge object detection models can re-energise and optimize diverse\\naspects of agriculture, ranging from crop monitoring to livestock management.\\nIt aims to achieve key objectives, including the identification of contemporary\\nchallenges in agriculture, a detailed assessment of YOLO's incremental\\nadvancements, and an exploration of its specific applications in agriculture.\\nThis is one of the first surveys to include the latest YOLOv10, offering a\\nfresh perspective on its implications for precision farming and sustainable\\nagricultural practices in the era of Artificial Intelligence and automation.\\nFurther, the survey undertakes a critical analysis of YOLO's performance,\\nsynthesizes existing research, and projects future trends. By scrutinizing the\\nunique capabilities packed in YOLO variants and their real-world applications,\\nthis survey provides valuable insights into the evolving relationship between\\nYOLO variants and agriculture. The findings contribute towards a nuanced\\nunderstanding of the potential for precision farming and sustainable\\nagricultural practices, marking a significant step forward in the integration\\nof advanced object detection technologies within the agricultural sector.\",\n      \"title\": \"YOLOv1 to YOLOv10: A comprehensive review of YOLO variants and their\\n  application in the agricultural domain\",\n      \"url\": \"http://arxiv.org/pdf/2406.10139v1\"\n    },\n    {\n      \"summary\": \"Agriculture is a vital sector that significantly contributes to the economy\\nand food security, particularly in regions like Varanasi, India. This paper\\nexplores various types of agriculture practiced in the area, including\\nsubsistence, commercial, intensive, extensive, industrial, organic,\\nagroforestry, aquaculture, and urban agriculture. Each type presents unique\\nchallenges and opportunities, necessitating innovative approaches to enhance\\nproductivity and sustainability. To address these challenges, the integration\\nof advanced technologies such as sensors and communication protocols is\\nessential. Sensors can provide real-time data on soil health, moisture levels,\\nand crop conditions, enabling farmers to make informed decisions. Communication\\ntechnologies facilitate the seamless transfer of this data, allowing for timely\\ninterventions and optimized resource management. Moreover, programming\\ntechniques play a crucial role in developing applications that process and\\nanalyze agricultural data. By leveraging machine learning algorithms, farmers\\ncan gain insights into crop performance, predict yields, and implement\\nprecision agriculture practices. This paper highlights the significance of\\ncombining traditional agricultural practices with modern technologies to create\\na resilient agricultural ecosystem. The findings underscore the potential of\\nintegrating sensors, communication technologies, and programming in\\ntransforming agricultural practices in Varanasi. By fostering a data-driven\\napproach, this research aims to contribute to sustainable farming, enhance food\\nsecurity, and improve the livelihoods of farmers in the region.\",\n      \"title\": \"Transforming Agriculture: Exploring Diverse Practices and Technological\\n  Innovations\",\n      \"url\": \"http://arxiv.org/pdf/2411.00643v1\"\n    },\n    {\n      \"summary\": \"This paper explores the transformative potential of artificial intelligence\\n(AI) in the context of sustainable agricultural development across diverse\\nregions in Africa. Delving into opportunities, challenges, and impact, the\\nstudy navigates through the dynamic landscape of AI applications in\\nagriculture. Opportunities such as precision farming, crop monitoring, and\\nclimate-resilient practices are examined, alongside challenges related to\\ntechnological infrastructure, data accessibility, and skill gaps. The article\\nanalyzes the impact of AI on smallholder farmers, supply chains, and inclusive\\ngrowth. Ethical considerations and policy implications are also discussed,\\noffering insights into responsible AI integration. By providing a nuanced\\nunderstanding, this paper contributes to the ongoing discourse on leveraging AI\\nfor fostering sustainability in African agriculture.\",\n      \"title\": \"Harnessing Artificial Intelligence for Sustainable Agricultural\\n  Development in Africa: Opportunities, Challenges, and Impact\",\n      \"url\": \"http://arxiv.org/pdf/2401.06171v1\"\n    },\n    {\n      \"summary\": \"Sustainable agriculture plays a crucial role in ensuring world food security\\nfor consumers. A critical challenge faced by sustainable precision agriculture\\nis weed growth, as weeds share essential resources with the crops, such as\\nwater, soil nutrients, and sunlight, which notably affect crop yields. The\\ntraditional methods employed to combat weeds include the usage of chemical\\nherbicides and manual weed removal methods. However, these could damage the\\nenvironment and pose health hazards. The adoption of automated computer vision\\ntechnologies and ground agricultural consumer electronic vehicles in precision\\nagriculture offers sustainable, low-carbon solutions. However, prior works\\nsuffer from issues such as low accuracy and precision and high computational\\nexpense. This work proposes EcoWeedNet, a novel model with enhanced weed\\ndetection performance without adding significant computational complexity,\\naligning with the goals of low-carbon agricultural practices. Additionally, our\\nmodel is lightweight and optimal for deployment on ground-based consumer\\nelectronic agricultural vehicles and robots. The effectiveness of the proposed\\nmodel is demonstrated through comprehensive experiments on the CottonWeedDet12\\nbenchmark dataset reflecting real-world scenarios. EcoWeedNet achieves\\nperformance close to that of large models yet with much fewer parameters.\\n(approximately 4.21% of the parameters and 6.59% of the GFLOPs of YOLOv4). This\\nwork contributes effectively to the development of automated weed detection\\nmethods for next-generation agricultural consumer electronics featuring lower\\nenergy consumption and lower carbon footprint. This work paves the way forward\\nfor sustainable agricultural consumer technologies.\",\n      \"title\": \"EcoWeedNet: A Lightweight and Automated Weed Detection Method for\\n  Sustainable Next-Generation Agricultural Consumer Electronics\",\n      \"url\": \"http://arxiv.org/pdf/2502.00205v1\"\n    },\n    {\n      \"summary\": \"The Internet of Things (IoT) and Artificial Intelligence (AI) have been\\nemployed in agriculture over a long period of time, alongside other advanced\\ncomputing technologies. However, increased attention is currently being paid to\\nthe use of such smart technologies. Agriculture has provided an important\\nsource of food for human beings over many thousands of years, including the\\ndevelopment of appropriate farming methods for different types of crops. The\\nemergence of new advanced IoT technologies has the potential to monitor the\\nagricultural environment to ensure high-quality products. However, there\\nremains a lack of research and development in relation to Smart Sustainable\\nAgriculture (SSA), accompanied by complex obstacles arising from the\\nfragmentation of agricultural processes, i.e. the control and operation of\\nIoT/AI machines; data sharing and management; interoperability; and large\\namounts of data analysis and storage. This study firstly, explores existing\\nIoT/AI technologies adopted for SSA and secondly, identifies IoT/AI technical\\narchitecture capable of underpinning the development of SSA platforms. As well\\nas contributing to the current body of knowledge, this research reviews\\nresearch and development within SSA and provides an IoT/AI architecture to\\nestablish a Smart, Sustainable Agriculture platform as a solution.\",\n      \"title\": \"Smart Sustainable Agriculture (SSA) Solution Underpinned by Internet of\\n  Things (IoT) and Artificial Intelligence (AI)\",\n      \"url\": \"http://arxiv.org/pdf/1906.03106v1\"\n    },\n    {\n      \"summary\": \"This article explores the use of drones in agriculture and discusses the\\nvarious types of drones employed for different agricultural applications.\\nDrones, also known as unmanned aerial vehicles (UAVs), offer numerous\\nadvantages in farming practices. They provide real-time and high-resolution\\ndata collection, enabling farmers to make informed irrigation, fertilization,\\nand pest management decisions. Drones assist in precision spraying and\\napplication of agricultural inputs, minimizing chemical wastage and optimizing\\nresource utilization. They offer accessibility to inaccessible areas, reduce\\nmanual labor, and provide cost savings and increased operational efficiency.\\nDrones also play a crucial role in mapping and surveying agricultural fields,\\naiding crop planning and resource allocation. However, challenges such as\\nregulations and limited flight time need to be addressed. The advantages of\\nusing drones in agriculture include precision agriculture, cost and time\\nsavings, improved data collection and analysis, enhanced crop management,\\naccessibility and flexibility, environmental sustainability, and increased\\nsafety for farmers. Overall, drones have the potential to revolutionize farming\\npractices, leading to increased efficiency, productivity, and sustainability in\\nagriculture.\",\n      \"title\": \"Employing Drones in Agriculture: An Exploration of Various Drone Types\\n  and Key Advantages\",\n      \"url\": \"http://arxiv.org/pdf/2307.04037v2\"\n    }\n  ]\n}\n```",
                    ],
                    },
                ]
            )
            

            random_data = random.sample(data, min(5, len(data)))
            response = chat_session.send_message(f"""
                Based on the provided context here are some research papers that most correlate with
                the provided prompt
                <prompt> 
                {search_request}
                </prompt>
                
                giving the context of the research papers
                <context>
                {random_data}
                </context>
                
                fill the title in title, summary in summary, and url in url
                
                add the summary to summary response
                
                add title to title 
                add url to url
                
                please 
                """
            )
            
            try:
                fixed_text = response.text

                response_data = json.loads(fixed_text)
                
            except json.JSONDecodeError:
                return Response({
                    "status": 500,
                    "message": "Failed to parse response from chat session",
                    "data": None
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            return Response({
                "status": 200,
                "message": "Success",
                "data": response_data
            }, status=status.HTTP_200_OK)
            
            
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
        
        

