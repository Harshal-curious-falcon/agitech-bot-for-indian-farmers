import random
import string
import nltk
from newspaper import Article
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator
import requests
import pyttsx3
import speech_recognition as sr
import winsound
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import OptionMenu


# Initialize the pyttsx3 engine
engine = pyttsx3.init()

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)

# News API access key (replace 'YOUR_NEWS_API_KEY' with your actual API key)
NEWS_API_KEY = 'fbf718f6b2134206ba8d0ff1aa1e3e4a'

# Unsplash API access key (replace 'YOUR_UNSPLASH_API_KEY' with your actual API key)
UNSPLASH_API_KEY = 'Dsg0gm2PV1ajlinh9MTxiUYej05kokgcZ7ijRAMRKYY'

# Dictionary of helpline numbers and contact details of agricultural specialists
agricultural_contacts = {
    "National Farmer's Helpline": "+91-1800-180-1551",
    "State Agricultural Department": {
        "udaipur": "+91-0294 249 0004",
        "jaipur": "+91-9459246644",
        # Add more states and their contact details as needed
    },
    "Agricultural Specialists": {
        "sita raam kashyap(pesticide formulation analysis)": "+91-9459246644",
        "Ishan Mehta(plant pathologist)": "+91-7018198472",
        # Add more specialists and their contact details as needed
    }
}
UPLOAD_FOLDER = 'uploaded_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to provide agricultural helpline numbers and contact details:
def get_agricultural_contacts():
    print("Here are some agricultural helpline numbers and contact details:")
    for category, contacts in agricultural_contacts.items():
        if isinstance(contacts, dict):
            print(f"\n{category}:")
            for location, number in contacts.items():
                print(f"{location}: {number}")
        else:
            print(f"\n{category}: {contacts}")

# Function to play notification sound
def play_notification_sound():
    winsound.PlaySound("SystemNotification", winsound.SND_ASYNC)


# Function to fetch agriculture news from News API
def agriculture_news():
    articles = fetch_agriculture_news()
    if articles:
        print("Latest Agriculture News Headlines:")
        for idx, article in enumerate(articles, start=1):
            print(f"{idx}. {article['title']} - {article['source']['name']}")
    else:
        print("No news articles available.")

# Function to fetch sentences from given URLs, including Wikipedia URLs
# Function to fetch sentences from given URLs, including Wikipedia URLs
def fetch_sentences(urls):
    all_sentence_list = []
    for url in urls:
        try:
            if 'wikipedia' in url:  # Check if URL is from Wikipedia
                page_title = url.split('/')[-1]  # Extract the page title from the URL
                article = wikipedia.page(page_title)
                text = article.content
                sentence_lists = sent_tokenize(text)
                all_sentence_list.extend(sentence_lists)
            else:
                article = Article(url)
                article.download()
                article.parse()
                article.nlp()
                text = article.text
                sentence_lists = sent_tokenize(text)
                all_sentence_list.extend(sentence_lists)
        except Exception as e:
            print(f"Error processing {url}: {e}")
            print("-----------------------------------------------------------------------------")
    return all_sentence_list



# Function to handle greetings
def greeting_response(text, language):
    text = text.lower()
    if language == 'english':
        bot_greetings = ['Howdy', 'Hi', 'Hey', 'Hello', 'Hola']
        user_greetings = ['hi', 'hello', 'greetings', 'wassup']
    elif language == 'hindi':
        bot_greetings = ['नमस्ते', 'हाय', 'नमस्कार', 'हेलो', 'कैसे हो']
        user_greetings = ['namaste', 'hello', 'hii' 'namaskar', 'aadab', 'kya haal ']
    else:
        return "I'm sorry, I couldn't understand your language choice."

    for word in text.split():
        if word in user_greetings:
            return random.choice(bot_greetings)


# Function to generate response based on user input
def agitech_response(user_input, sentence_list, language):
    user_input = user_input.lower()
    sentence_list.append(user_input)
    response = ''
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentence_list)
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    similar_sentence_index = cosine_similarities.argsort()[0][-2]
    similar_sentences = cosine_similarities.flatten()
    similar_sentences.sort()
    response_flag = 0
    if similar_sentences[-2] == 0:
        response = "I apologize, I don't understand."
    else:
        response = sentence_list[similar_sentence_index]
        if language == 'hindi':
            try:
                translator = Translator()
                response = translator.translate(response, src='en', dest='hi').text
            except Exception as e:
                print(f"Error translating: {e}")
    sentence_list.remove(user_input)
    return response

# Function to read out the response using text-to-speech with a soft and sweet voice
def read_out_response(response):
    # Set voice properties for a soft and sweet voice
    voices = engine.getProperty('voices')


def read_out_response(response):
    # Get all available voices
    voices = engine.getProperty('voices')

    # Randomly select a voice
    selected_voice = random.choice(voices)

    # Set voice properties
    engine.setProperty('voice', selected_voice.id)
    engine.setProperty('rate', 150)  # Adjust speech rate (words per minute)
    engine.setProperty('volume', 1.0)  # Adjust volume (0.0 to 1.0)

    # Speak the text
    engine.say(response)
    engine.runAndWait()


# Function to suggest crops based on temperature range
def suggest_crops(temperature):
    if 10 <= temperature <= 20:
        return ["Wheat", "Barley", "Oats"]
    elif 21 <= temperature <= 30:
        return ["Rice", "Maize", "Tomato"]
    elif temperature > 30:
        return ["Sugarcane", "Cotton", "Soybean"]
    else:
        return ["Potato", "Carrot", "Onion"]  # Default suggestions


# Function to provide crop requirements
def crop_requirements(crop):
    # You can define the crop requirements based on your dataset or API
    requirements = {
        "wheat": "Wheat requires well-drained soil, moderate temperature, and regular watering.",
        "rice": "Rice needs flooded fields, warm weather, and high humidity.",
        "sugarcane": "Sugarcane grows well in tropical climates, needs regular watering and fertile soil.",
        "maize": "A moderate temperature, adequate moisture are basic need of Maize crop",
        "tomato": "tomato can be grown on  well-drained, sandy or red loam soils rich in organic matter with a pH range of 6.0-7.0 are considered",
        "cotton": "Cotton cultivation requires frost-free conditions, 20 to 30 degree Celsius temperature, and a small amount of annual rainfall",
        "soyabean": "soyabean requires well-drained, fertile loamy soils with a pH range of 6.0 to 7.5"
        # Define requirements for other crops as well
    }
    return requirements.get(crop, "Requirements for this crop are not available.")


# Function to get temperature for a given city (You may replace this with your own temperature fetching mechanism)
def fetch_temperature():
    temperature = input("Please enter the temperature of your region in Celsius: ")
    return float(temperature)

def recomment_equipments(farm_size, specific_requirements):
    # Define a dictionary mapping tural_equipmenagricultural equipment to their respective categories and features
    equipment_catalog = {
        "Tractor": {"Category": "Machinery", "Features": ["Plowing", "Harvesting", "Tillage"]},
        "Seeder": {"Category": "Planting", "Features": ["Seed drilling", "Precision planting"]},
        "Sprayer": {"Category": "Crop Protection", "Features": ["Pesticide spraying", "Herbicide spraying"]},
        "Harvester": {"Category": "Harvesting", "Features": ["Crop harvesting", "Threshing"]},
        # Add more equipment with categories and features as needed
    }

    # Define equipment recommendations based on farm size and specific requirements
    recommended_equipment = []

    # Based on farm size, recommend appropriate machinery
    if farm_size == "Small":
        recommended_equipment.extend(["Seeder", "Sprayer"])
    elif farm_size == "Medium":
        recommended_equipment.extend(["Tractor", "Harvester"])
    elif farm_size == "Large":
        recommended_equipment.extend(["Tractor", "Harvester", "Sprayer"])

    # Based on specific requirements, refine equipment recommendations
    if "Precision Farming" in specific_requirements:
        recommended_equipment.append("GPS Guidance System")
    if "Organic Farming" in specific_requirements:
        recommended_equipment.append("No-till Planter")

    # Filter out redundant equipment and return the final recommendations
    recommended_equipment = list(set(recommended_equipment))  # Remove duplicates
    recommended_equipment_details = [(equipment, equipment_catalog[equipment]) for equipment in recommended_equipment]

    return recommended_equipment_details


def get_agricultural_contacts():
    print("Here are some agricultural helpline numbers and contact details:")
    for category, contacts in agricultural_contacts.items():
        if isinstance(contacts, dict):
            print(f"\n{category}:")
            for location, number in contacts.items():
                print(f"{location}: {number}")
        else:
            print(f"\n{category}: {contacts}")


def fetch_agriculture_news(count=2):
    url = f'https://newsapi.org/v2/everything?q=agriculture&apiKey={NEWS_API_KEY}'
    response = requests.get(url)
    if response.status_code == 200:
        news_data = response.json()
        articles = news_data['articles']
        random.shuffle(articles)  # Fetch only the top 2 news headlines
        return articles[:count]
    else:
        print('Failed to fetch news articles')
        return []


def recommend_agricultural_equipment(farm_size, specific_requirements):
    # Define a dictionary mapping agricultural equipment to their respective categories and features
    equipment_catalog = {
        "Tractor": {"Category": "Machinery", "Features": ["Plowing", "Harvesting", "Tillage"]},
        "Seeder": {"Category": "Planting", "Features": ["Seed drilling", "Precision planting"]},
        "Sprayer": {"Category": "Crop Protection", "Features": ["Pesticide spraying", "Herbicide spraying"]},
        "Harvester": {"Category": "Harvesting", "Features": ["Crop harvesting", "Threshing"]},
        # Add more equipment with categories and features as needed
    }

    # Define equipment recommendations based on farm size and specific requirements
    recommended_equipment = []

    # Based on farm size, recommend appropriate machinery
    if farm_size == "Small":
        recommended_equipment.extend(["Seeder", "Sprayer"])
    elif farm_size == "Medium":
        recommended_equipment.extend(["Tractor", "Harvester"])
    elif farm_size == "Large":
        recommended_equipment.extend(["Tractor", "Harvester", "Sprayer"])

    # Based on specific requirements, refine equipment recommendations
    if "Precision Farming" in specific_requirements:
        recommended_equipment.append("GPS Guidance System")
    if "Organic Farming" in specific_requirements:
        recommended_equipment.append("No-till Planter")

    # Filter out redundant equipment and return the final recommendations
    recommended_equipment = list(set(recommended_equipment))  # Remove duplicates
    recommended_equipment_details = [(equipment, equipment_catalog[equipment]) for equipment in recommended_equipment]

    return recommended_equipment_details


def get_agricultural_contacts():
    print("Here are some agricultural helpline numbers and contact details:")
    for category, contacts in agricultural_contacts.items():
        if isinstance(contacts, dict):
            print(f"\n{category}:")
            for location, number in contacts.items():
                print(f"{location}: {number}")
        else:
            print(f"\n{category}: {contacts}")


def fetch_agriculture_news(count=2):
    url = f'https://newsapi.org/v2/everything?q=agriculture&apiKey={NEWS_API_KEY}'
    response = requests.get(url)
    if response.status_code == 200:
        news_data = response.json()
        articles = news_data['articles']
        random.shuffle(articles)  # Fetch only the top 2 news headlines
        return articles[:count]
    else:
        print('Failed to fetch news articles')
        return []


def analyze_farm_health(image):
    # Placeholder: Analyze color distribution in the image
    # This is a simplified approach, actual implementation may require more advanced image processing techniques

    # Convert image to HSV color space for better color analysis
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define green and brown color ranges in HSV
    lower_green = np.array([36, 25, 25])
    upper_green = np.array([86, 255, 255])
    lower_brown = np.array([10, 100, 20])
    upper_brown = np.array([30, 255, 200])

    # Threshold the HSV image to get only green and brown areas
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    brown_mask = cv2.inRange(hsv_image, lower_brown, upper_brown)

    # Calculate the percentage of green and brown areas in the image
    total_pixels = image.shape[0] * image.shape[1]
    green_pixels = cv2.countNonZero(green_mask)
    brown_pixels = cv2.countNonZero(brown_mask)

    green_percentage = (green_pixels / total_pixels) * 100
    brown_percentage = (brown_pixels / total_pixels) * 100

    # Display the gauge for good health based on green percentage
    create_gauge("Good Health", green_percentage)

    # Display the gauge for bad health based on brown percentage
    create_gauge("Bad Health", brown_percentage)

    # You can also return the calculated percentages if needed
    return green_percentage, brown_percentage



def create_gauge(label, percentage):
    colors = ['#00cc00', '#99ff33', '#ffcc00', '#ff6600', '#ff0000']
    labels = ['100%', '75%', '50%', '25%', '0%']
    plt.pie([percentage, 100 - percentage], colors=colors[:len(labels) - 1] + ['#ffffff'], startangle=90)
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    plt.gca().add_artist(centre_circle)
    plt.text(0, 0, f'{percentage}%', fontsize=40, ha='center')
    plt.text(0, -0.3, label, fontsize=12, ha='center')
    plt.axis('equal')
    plt.axis('off')
    plt.show()


def process_image(image):
    # Placeholder function for image processing
    print("Processing image...")
    analyze_farm_health(image)


def display_required_equipment():
    # Placeholder function to display required equipment
    farm_size = input("Enter your farm size (Small/Medium/Large): ").capitalize()
    specific_requirements = input("Enter specific requirements (comma-separated): ").split(',')
    equipment_recommendations = recommend_agricultural_equipment(farm_size, specific_requirements)
    print("Recommended Agricultural Equipment:")
    for equipment, details in equipment_recommendations:
        print(f"Equipment: {equipment}")
        print(f"Category: {details['Category']}")
        print(f"Features: {', '.join(details['Features'])}")


def capture_image(cap):
    ret, frame = cap.read()
    return frame

def main():
    cap = cv2.VideoCapture(0)

    root = tk.Tk()
    root.title("Farm Health Monitor")

    def process_button_click():
        image = capture_image(cap)
        process_image(image)


    process_button = tk.Button(root, text="Capture Image and Analyze", command=process_button_click)
    process_button.pack()
    root.mainloop()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()





# Function to recognize speech input
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak something...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print("You (Speech):", text)
        return text
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand what you said.")
        return ""
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return ""


# Function to fetch random image from Unsplash API based on keyword
def fetch_random_image(keyword):
    url = f"https://api.unsplash.com/photos/random/?client_id={UNSPLASH_API_KEY}&query={keyword}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data:
            return data['urls']['regular']
    return None


# Main function to drive the chatbot
# Main function to drive the chatbot
def chatbot(urls, user_history=None):
    if user_history is None:
        user_history = []  # Initialize user history list

    print(
        'agitech bot: I am an agitech bot, here to find your best answer possible for your questions,how can i help you')
    language = input("Choose language (english/hindi): ").lower()
    while language not in ['english', 'hindi']:
        print("Invalid language choice! Please choose between English and Hindi.")
        language = input("Choose language (english/hindi): ").lower()

    # Option to choose between text or speech input
    input_method = input("Choose input method (text/speech): ").lower()
    while input_method not in ['text', 'speech']:
        print("Invalid input method! Please choose between 'text' and 'speech'.")
        input_method = input("Choose input method (text/speech): ").lower()

    exit_list = ['exit', 'see you later', 'bye', 'quit']
    sentence_list = fetch_sentences(urls)
    while True:
        if input_method == 'speech':
            user_input = recognize_speech()  # Recognize speech input
            if not user_input:  # If speech recognition fails, prompt for text input
                user_input = input('You: ')
        else:
            user_input = input('You: ')  # Text input from the user
        user_history.append(user_input)  # Add user input to history

        if user_input.lower() in exit_list:
            print('agitech bot: Chat with you later!')
            break
        else:
            greeting = greeting_response(user_input, language)
            if greeting:
                print('agitech bot: ' + greeting)
            elif user_input.lower() == 'news':  # Check if user wants to see news headlines
                articles = fetch_agriculture_news()
                if articles:
                    print("Latest Agriculture News Headlines:")
                    for idx, article in enumerate(articles, start=1):
                        print(f"{idx}. {article['title']} - {article['source']['name']}")
                else:
                    print("No news articles available.")
            elif user_input.lower() == 'crop requirements':  # Check if user wants crop requirements
                crop_name = input("Enter the crop name: ")
                requirements = crop_requirements(crop_name)
                print(f"Requirements for {crop_name}: {requirements}")
            elif user_input.lower() == 'suggest crops':  # Check if user wants crop suggestions based on temperature
                temperature = fetch_temperature()  # Fetch temperature directly from the user
                suggested_crops = suggest_crops(temperature)
                print(f"Suggested crops based on the entered temperature: {', '.join(suggested_crops)}")
                selected_crop = input("Enter the crop you want to know the requirements for: ")
                requirements = crop_requirements(selected_crop.lower())
                print(f"Requirements for {selected_crop}: {requirements}")
            elif user_input.lower().startswith('wikipedia'):  # Check if user wants to search on Wikipedia
                search_query = user_input[10:].strip()  # Extract the search query after 'wikipedia'
                try:
                    wikipedia_result = wikipedia.summary(search_query)
                    print('Wikipedia:', wikipedia_result)
                except wikipedia.exceptions.DisambiguationError as e:
                    print("Wikipedia DisambiguationError: Please provide more specific query.")
                except wikipedia.exceptions.PageError as e:
                    print("Wikipedia PageError: No Wikipedia page found for the given query.")
            elif user_input.lower() == 'agricultural help':  # New option for getting agricultural help
                    get_agricultural_contacts()  # Call the function to provide agricultural contacts
            else:
                response = agitech_response(user_input, sentence_list, language)
                print('agitech bot: ' + response)
                # Fetch and display a random image related to the question asked by the user
                image_keyword = user_input.lower().split()[0]  # Use the first word of the user input as keyword
                image_url = fetch_random_image(image_keyword)
                if image_url:
                    print("Random Image Related to Your Question:")
                    print(image_url)
                else:
                    print("No image found related to your question.")

                read_button = input("Would you like me to read out the response? (yes/no): ")
                if read_button.lower() == 'yes':
                    read_out_response(response)

    # Optionally, you can print or store the user history after the conversation ends
    print("User History:")
    for idx, interaction in enumerate(user_history, start=1):
        print(f"{idx}. {interaction}")



# URLs for fetching information
urls = [
    "https://www.ibef.org/blogs/high-demand-for-medicinal-plants-in-india",
    "https://eos.com/blog/crop-diseases/",
    "https://education.nationalgeographic.org/resource/crop/",
    "https://www.plugandplaytechcenter.com/resources/new-agriculture-technology-modern-farming/",
    "https://www.jiva.ag/blog/what-are-the-most-common-problems-and-challenges-that-farmers-face",
    "https://timesofindia.indiatimes.com/india/empowering-indias-farmers-list-of-schemes-for-welfare-of-farmers-in-india/articleshow/107854121.cms"
]

# Run the chatbot with user history
chatbot(urls)
