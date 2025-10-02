# import requests  # allows Python to make API calls

# # Make a GET request to the joke API
# response = requests.get("https://official-joke-api.appspot.com/random_joke")

# # Check if it worked (status code 200 = OK)
# if response.status_code == 200:
#     data = response.json()  # Convert JSON to Python dict
#     print("Here's a joke for you 😄\n")
#     print("🧩 " + data['setup'])
#     print("😂 " + data['punchline'])
# else:
#     print("⚠️ Failed to fetch joke. Try again later.")

#---------------------------------------------------------------------------------------------------

# import requests

# def get_weather(city):
#     url = f"http://api.weatherapi.com/v1/current.json?key=fa98197679e3495bbe7143909251007&q={city}&aqi=no"
#     response = requests.get(url)

#     if response.status_code == 200:
#         data = response.json()
#         location = data['location']['name']
#         country = data['location']['country']
#         temp_c = data['current']['temp_c']
#         condition = data['current']['condition']['text']

#         print(f"\n📍 Location: {location}, {country}")
#         print(f"🌡️ Temperature: {temp_c}°C")
#         print(f"⛅ Condition: {condition}")
#     else:
#         print("❌ Could not retrieve weather data. Check city name or try again later.")

# # --- Run the app ---
# print("🌍 Welcome to Weather App 🌦️")
# city = input("Enter city name: ")
# get_weather(city)

#---------------------------------------------------------------------------------------------------

# import tkinter as tk
# import requests

# API_KEY = "fa98197679e3495bbe7143909251007"  # Replace this with your real WeatherAPI key

# def get_weather():
#     city = entry.get()
#     if not city:
#         output_label.config(text="⚠️ Please enter a city name.")
#         return

#     url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}&aqi=no"

#     try:
#         response = requests.get(url)
#         data = response.json()

#         if "error" in data:
#             output_label.config(text="❌ Invalid city or API key.")
#             return

#         name = data["location"]["name"]
#         country = data["location"]["country"]
#         temp_c = data["current"]["temp_c"]
#         condition = data["current"]["condition"]["text"]

#         result = (
#             f"📍 {name}, {country}\n"
#             f"🌡️ Temperature: {temp_c}°C\n"
#             f"⛅ Condition: {condition}"
#         )
#         output_label.config(text=result)
#     except:
#         output_label.config(text="⚠️ Failed to fetch weather. Try again.")

# # --- GUI Setup ---
# window = tk.Tk()
# window.title("🌦️ Weather App")
# window.geometry("400x300")

# tk.Label(window, text="Enter City Name:", font=("Arial", 14)).pack(pady=10)
# entry = tk.Entry(window, font=("Arial", 14))
# entry.pack(pady=5)

# tk.Button(window, text="Get Weather", font=("Arial", 12), command=get_weather).pack(pady=10)

# output_label = tk.Label(window, text="", font=("Arial", 13), fg="blue", justify="left")
# output_label.pack(pady=15)

# window.mainloop()

#---------------------------------------------------------------------------------------------------

# import requests

# def convert_currency(from_currency, to_currency, amount):
#     url = f"https://api.exchangerate-api.com/v4/latest/{from_currency.upper()}"
#     response = requests.get(url)

#     if response.status_code != 200:
#         print("❌ Failed to get exchange rates.")
#         return

#     data = response.json()

#     rates = data.get("rates")
#     if to_currency.upper() not in rates:
#         print("❌ Invalid target currency.")
#         return

#     rate = rates[to_currency.upper()]
#     converted = round(amount * rate, 2)

#     print(f"\n✅ {amount} {from_currency.upper()} = {converted} {to_currency.upper()}")
#     print(f"💱 Rate: 1 {from_currency.upper()} = {rate} {to_currency.upper()}")

# # --- Run ---
# print("💱 Currency Converter")

# from_curr = input("Enter FROM currency (e.g., USD): ")
# to_curr = input("Enter TO currency (e.g., PKR): ")

# try:
#     amt = float(input("Enter amount: "))
#     convert_currency(from_curr, to_curr, amt)
# except ValueError:
#     print("❌ Please enter a valid amount.")

#---------------------------------------------------------------------------------------------------

# import requests

# url = "https://catfact.ninja/fact"

# response = requests.get(url)
# if response.status_code == 200:
#     data = response.json()
#     print("🐱 Cat Fact:")
#     print(data['fact'])
# else:
#     print("❌ Failed to get a cat fact.")

#---------------------------------------------------------------------------------------------------

# import requests

# response = requests.get("https://dog.ceo/api/breeds/image/random")
# if response.status_code == 200:
#     data = response.json()
#     print("🐶 Random Dog Image:")
#     print(data["message"])  # this is the image URL
# else:
#     print("❌ Failed to get a dog image.")

#---------------------------------------------------------------------------------------------------

# import requests

# response = requests.get("https://api.adviceslip.com/advice")
# if response.status_code == 200:
#     data = response.json()
#     print("🧠 Random Advice:")
#     print(data["slip"]["advice"])
# else:
#     print("❌ Could not fetch advice.")

#---------------------------------------------------------------------------------------------------

# import requests

# try:
#     response = requests.get("https://www.boredapi.com/api/activity", timeout=5)
#     response.raise_for_status()  # Raises HTTPError for bad status
#     data = response.json()
#     print("🤖 Suggested Activity:")
#     print(data["activity"])
# except requests.exceptions.RequestException as e:
#     print("❌ Failed to get an activity.")
#     print("🔎 Error:", e)
    
#---------------------------------------------------------------------------------------------------