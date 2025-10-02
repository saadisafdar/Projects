# file = open("data.txt", "w")   # "w" = write (overwrites)
# file.write("Hello Saadi!")
# file.close()


# file = open("data.txt", "r")
# content = file.read()
# print(content)
# file.close()


# with open("data.txt", "w") as file:
#     file.write("This is saved safely!")


# with open("data.txt", "r") as file:
#     content = file.read()
#     print(content)


# note = input("📝 Enter a note to save: ")
# with open("data.txt", "a") as file:
#     file.write(note + "\n")
# print("✅ Note saved to data.txt")


# from datetime import datetime
# note = input("📝 Enter a note to append: ")
# time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# with open("data.txt", "a") as file:
#     file.write(f"[{time}] {note}\n")
# print("✅ Note with timestamp appended to data.txt")


while True:
    print("\nNotes Menu:")
    print("1. Read Notes")
    print("2. Write to Notes")
    print("3. Append to Notes")
    print("4. Clear All Notes")
    print("5. Exit")
    choice = input("Choose an option (1-5): ")

    if choice == '1':
        with open("data.txt", "r") as file:
            content = file.read()
            print("\nCurrent Notes:")
            print(content if content else "No notes found.")
    
    elif choice == '2':
        note = input("📝 Enter a note to write (overwrites existing content): ")
        with open("data.txt", "w") as file:
            file.write(note + "\n")
        print("✅ Note written to data.txt")
    
    elif choice == '3':
        note = input("📝 Enter a note to append: ")
        with open("data.txt", "a") as file:
            file.write(note + "\n")
        print("✅ Note appended to data.txt")
    
    elif choice == '4':
        with open("data.txt", "w") as file:
            file.write("")  # Clears the file
        print("🗑️ All notes cleared.")

    elif choice == '5':
        print("Exiting the Notes Menu. Goodbye!")
        break
    else:
        print("❌ Invalid choice. Please select a valid option (1-4).") 


