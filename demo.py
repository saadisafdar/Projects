import random

# List of questions
questions = [
    {"question": "What is the capital of Pakistan?", "answer": "Islamabad"},
    {"question": "What is 5 + 7?", "answer": "12"},
    {"question": "What programming language are we learning?", "answer": "Python"},
    {"question": "Which planet is known as the Red Planet?", "answer": "Mars"}
]

# Pick a random question
q = random.choice(questions)

# Ask the user
print("🧠 Quiz Time!")
user_answer = input(q["question"] + " ")

# Check answer
if user_answer.strip().lower() == q["answer"].lower():
    print("✅ Correct!")
else:
    print(f"❌ Wrong. The correct answer is {q['answer']}.")
    
          
print(random.choice(["apple", "banana", "mango"]))  # Random choice

import datetime

today = datetime.date.today()
print("Today's date is:", today)



