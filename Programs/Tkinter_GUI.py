# import tkinter as tk

# # Create window
# window = tk.Tk()
# window.title("My First GUI")
# window.geometry("300x200")  # width x height

# # Add a label
# label = tk.Label(window, text="Hello, Saadi!", font=("Arial", 16))
# label.pack(pady=20)  # Center it with padding

# # Run the window
# window.mainloop()

#-------------------------------------------------------------------------------

# import tkinter as tk

# window = tk.Tk()
# window.title("Welcome App")
# window.geometry("500x350")

# label = tk.Label(window, text="👋Hello, I am Saadi Safder", font=("Times New Roman", 20, "bold italic"), fg="green", bg="lightgray")
# label.pack(pady=30)

# window.mainloop()

#-------------------------------------------------------------------------------

# import tkinter as tk

# def greet_user():
#     name = entry.get()
#     label_output.config(text=f"Hello, {name}! Welcome 👋")
#     entry.delete(0, tk.END)


# window = tk.Tk()
# window.title("Greeting App")
# window.geometry("400x250")

# label = tk.Label(window, text="Enter your name:", font=("Arial", 14))
# label.pack(pady=10)

# entry = tk.Entry(window, font=("Arial", 14))
# entry.pack(pady=5)

# button = tk.Button(window, text="Greet Me", font=("Arial", 12), command=greet_user)
# button.pack(pady=10)

# label_output = tk.Label(window, text="", font=("Arial", 14), fg="green")
# label_output.pack(pady=10)

# window.mainloop()

#-------------------------------------------------------------------------------

# import tkinter as tk

# def greet_user():
#     age = entry.get()
#     label_output.config(text=f"You are {age} years old.")
#     entry.delete(0, tk.END)


# window = tk.Tk()
# window.title("Age App")
# window.geometry("400x250")

# label = tk.Label(window, text="Enter your age:", font=("Arial", 14))
# label.pack(pady=10)

# entry = tk.Entry(window, font=("Times New Roman", 14))
# entry.pack(pady=5)

# button = tk.Button(window, text="Click To Get Age", font=("Courier", 12), command=greet_user)
# button.pack(pady=10)

# label_output = tk.Label(window, text="", font=("Helvetica", 28), fg="blue", bg="lightpink")
# label_output.pack(pady=10)

# window.mainloop()

#-------------------------------------------------------------------------------

# import tkinter as tk

# def greet_user():
#     name = entry_name.get()
#     label_output_name.config(text=f"Hello, {name}! Welcome 👋")
#     #entry_name.delete(0, tk.END)

# def age_user():
#     age = entry_age.get()
#     label_output_age.config(text=f"You are {age} years old.")
#     #entry_age.delete(0, tk.END)

# # Window setup
# window = tk.Tk()
# window.title("Greeting App")
# window.geometry("400x350")

# # --- Name input ---
# label_name = tk.Label(window, text="Enter your name:", font=("Arial", 14))
# label_name.pack(pady=5)

# entry_name = tk.Entry(window, font=("Arial", 14))
# entry_name.pack(pady=5)

# # --- Age input ---
# label_age = tk.Label(window, text="Enter your age:", font=("Arial", 14))
# label_age.pack(pady=5)

# entry_age = tk.Entry(window, font=("Arial", 14))
# entry_age.pack(pady=5)

# # --- Buttons ---
# button_greet = tk.Button(window, text="Greet Me", font=("Arial", 12), command=greet_user)
# button_greet.pack(pady=10)

# button_age = tk.Button(window, text="Show Age", font=("Arial", 12), command=age_user)
# button_age.pack(pady=10)

# # --- Output Labels (2) ---
# label_output_name = tk.Label(window, text="", font=("Helvetica", 14), fg="darkgreen")
# label_output_name.pack(pady=5)

# label_output_age = tk.Label(window, text="", font=("Helvetica", 14), fg="blue")
# label_output_age.pack(pady=5)

# # Start GUI
# window.mainloop()

#-------------------------------------------------------------------------------

# import tkinter as tk

# def submit_info():
#     name = entry_name.get()
#     age = entry_age.get()
#     gender = entry_gender.get()
#     email = entry_email.get()
    
#     output_label.config(text=f"👤 Name: {name}\n🎂 Age: {age}\n🚻 Gender: {gender}\nEm@il: {email}")
    
# # entry_name.delete(0, tk.END)
# # entry_age.delete(0, tk.END)
# # entry_gender.delete(0, tk.END)
# # entry_email.delete(0, tk.END)

# # Create window
# window = tk.Tk()
# window.title("Form Example with Grid")
# window.geometry("400x300")

# # --- Labels and Entries ---
# label_name = tk.Label(window, text="Name:", font=("Arial", 12))
# label_name.grid(row=0, column=0, padx=10, pady=5, sticky="e")

# entry_name = tk.Entry(window, font=("Arial", 12))
# entry_name.grid(row=0, column=1, padx=10, pady=5)

# label_age = tk.Label(window, text="Age:", font=("Arial", 12))
# label_age.grid(row=1, column=0, padx=10, pady=5, sticky="e")

# entry_age = tk.Entry(window, font=("Arial", 12))
# entry_age.grid(row=1, column=1, padx=10, pady=5)

# label_gender = tk.Label(window, text="Gender:", font=("Arial", 12))
# label_gender.grid(row=2, column=0, padx=10, pady=5, sticky="e")

# entry_gender = tk.Entry(window, font=("Arial", 12))
# entry_gender.grid(row=2, column=1, padx=10, pady=5)

# label_email = tk.Label(window, text="Email:", font=("Arial", 12))
# label_email.grid(row=3, column=0, padx=10, pady=5, sticky="e")

# entry_email = tk.Entry(window, font=("Arial", 12))
# entry_email.grid(row=3, column=1, padx=10, pady=5)

# # --- Submit Button ---
# button_submit = tk.Button(window, text="Submit", font=("Arial", 12), command=submit_info)
# button_submit.grid(row=4, column=0, columnspan=2, pady=10)

# # --- Output Label ---
# output_label = tk.Label(window, text="", font=("Arial", 12), fg="green", justify="left")
# output_label.grid(row=4, column=1, columnspan=2, pady=10)

# # Run window
# window.mainloop()

#-------------------------------------------------------------------------------

# import tkinter as tk

# def submit_form():
#     name = entry_name.get()
#     gender = gender_var.get()
#     language = lang_var.get()
#     subscribed = "Yes" if subscribe_var.get() else "No"
#     accepted = "Yes" if accept_var.get() else "No"

#     result = (
#         f"👤 Name: {name}\n"
#         f"🚻 Gender: {gender}\n"
#         f"💻 Language: {language}\n"
#         f"📬 Subscribed: {subscribed}\n"
#         f"✅ Accepted T&C: {accepted}"
#     )
#     output_label.config(text=result)

# # Window setup
# window = tk.Tk()
# window.title("Form with Choices")
# window.geometry("500x400")

# # --- Name Input ---
# tk.Label(window, text="Name:", font=("Arial", 12)).grid(row=0, column=0, sticky="e", padx=10, pady=5)
# entry_name = tk.Entry(window, font=("Arial", 12))
# entry_name.grid(row=0, column=1, padx=10, pady=5)

# # --- Gender Dropdown ---
# tk.Label(window, text="Gender:", font=("Arial", 12)).grid(row=1, column=0, sticky="e", padx=10, pady=5)
# gender_var = tk.StringVar(value="Select")
# gender_menu = tk.OptionMenu(window, gender_var, "Male", "Female", "Other")
# gender_menu.grid(row=1, column=1, padx=10, pady=5)

# # --- Language (Radio Buttons) ---
# tk.Label(window, text="Language:", font=("Arial", 12)).grid(row=2, column=0, sticky="e", padx=10, pady=5)
# lang_var = tk.StringVar(value="Python")
# tk.Radiobutton(window, text="Python", variable=lang_var, value="Python").grid(row=2, column=1, sticky="w")
# tk.Radiobutton(window, text="Java", variable=lang_var, value="Java").grid(row=3, column=1, sticky="w")
# tk.Radiobutton(window, text="C++", variable=lang_var, value="C++").grid(row=4, column=1, sticky="w")
# tk.Radiobutton(window, text="JavaScript", variable=lang_var, value="JavaScript").grid(row=5, column=1, sticky="w")

# # --- Checkbuttons ---
# subscribe_var = tk.BooleanVar()
# tk.Checkbutton(window, text="Subscribe to newsletter", variable=subscribe_var).grid(row=6, columnspan=2, pady=5)
# accept_var = tk.BooleanVar()
# tk.Checkbutton(window, text="Accept Terms & Conditions", variable=accept_var).grid(row=7, columnspan=2, pady=5)

# # --- Submit Button ---
# tk.Button(window, text="Submit", font=("Arial", 12), command=submit_form).grid(row=8, columnspan=2, pady=10)

# # --- Output ---
# output_label = tk.Label(window, text="", font=("Arial", 12), fg="green", justify="left")
# output_label.grid(row=9, columnspan=2, pady=10)

# # Run app
# window.mainloop()

#-------------------------------------------------------------------------------

# import tkinter as tk
# from tkinter import messagebox

# def register():
#     name = entry_name.get()
#     email = entry_email.get()
#     password = entry_password.get()
#     gender = gender_var.get()
#     language = lang_var.get()
#     subscribed = "Yes" if subscribe_var.get() else "No"

#     # --- Validation ---
#     if not name or not email or not password or gender == "Select":
#         messagebox.showwarning("Missing Info", "❗ Please fill all required fields.")
#         return

#     # --- Display Data ---
#     result = (
#         f"👤 Name: {name}\n"
#         f"📧 Email: {email}\n"
#         f"🔐 Password: {'*' * len(password)}\n"
#         f"🚻 Gender: {gender}\n"
#         f"💻 Language: {language}\n"
#         f"📬 Subscribed: {subscribed}"
#     )
#     output_label.config(text=result)

#     # Clear fields
#     entry_name.delete(0, tk.END)
#     entry_email.delete(0, tk.END)
#     entry_password.delete(0, tk.END)

# # --- Main Window ---
# window = tk.Tk()
# window.title("User Registration Form")
# window.geometry("500x500")

# # --- Name ---
# tk.Label(window, text="Full Name*", font=("Arial", 12)).grid(row=0, column=0, sticky="e", padx=10, pady=5)
# entry_name = tk.Entry(window, font=("Arial", 12))
# entry_name.grid(row=0, column=1, padx=10, pady=5)

# # --- Email ---
# tk.Label(window, text="Email*", font=("Arial", 12)).grid(row=1, column=0, sticky="e", padx=10, pady=5)
# entry_email = tk.Entry(window, font=("Arial", 12))
# entry_email.grid(row=1, column=1, padx=10, pady=5)

# # --- Password ---
# tk.Label(window, text="Password*", font=("Arial", 12)).grid(row=2, column=0, sticky="e", padx=10, pady=5)
# entry_password = tk.Entry(window, font=("Arial", 12), show="*")
# entry_password.grid(row=2, column=1, padx=10, pady=5)

# # --- Gender ---
# tk.Label(window, text="Gender*", font=("Arial", 12)).grid(row=3, column=0, sticky="e", padx=10, pady=5)
# gender_var = tk.StringVar(value="Select")
# tk.OptionMenu(window, gender_var, "Male", "Female", "Other").grid(row=3, column=1, padx=10, pady=5)

# # --- Language (Radio Buttons) ---
# tk.Label(window, text="Language", font=("Arial", 12)).grid(row=4, column=0, sticky="e", padx=10, pady=5)
# lang_var = tk.StringVar(value="Python")
# tk.Radiobutton(window, text="Python", variable=lang_var, value="Python").grid(row=4, column=1, sticky="w")
# tk.Radiobutton(window, text="Java", variable=lang_var, value="Java").grid(row=5, column=1, sticky="w")
# tk.Radiobutton(window, text="C++", variable=lang_var, value="C++").grid(row=6, column=1, sticky="w")

# # --- Subscribe ---
# subscribe_var = tk.BooleanVar()
# tk.Checkbutton(window, text="Subscribe to Newsletter", variable=subscribe_var).grid(row=7, columnspan=2, pady=5)

# # --- Register Button ---
# tk.Button(window, text="Register", font=("Arial", 12), command=register).grid(row=8, columnspan=2, pady=15)

# # --- Output Label ---
# output_label = tk.Label(window, text="", font=("Arial", 12), fg="green", justify="left")
# output_label.grid(row=9, columnspan=2, pady=10)

# # Run App
# window.mainloop()

#-------------------------------------------------------------------------------

# import tkinter as tk
# from tkinter import messagebox

# def open_dashboard():
#     username = entry_username.get()
#     password = entry_password.get()

#     if username == "admin" and password == "1234":
#         # Hide login window
#         window.withdraw()

#         # Create dashboard
#         dashboard = tk.Toplevel()
#         dashboard.title("Dashboard")
#         dashboard.geometry("400x200")

#         tk.Label(dashboard, text=f"Welcome, {username}!", font=("Arial", 16)).pack(pady=20)
#         tk.Button(dashboard, text="Logout", command=lambda: logout(dashboard)).pack(pady=10)
#     else:
#         messagebox.showerror("Login Failed", "Invalid username or password")

# def logout(dashboard):
#     dashboard.destroy()
#     window.deiconify()  # Show login window again

# # --- Login Window ---
# window = tk.Tk()
# window.title("Login Page")
# window.geometry("400x250")

# tk.Label(window, text="Username", font=("Arial", 12)).pack(pady=5)
# entry_username = tk.Entry(window, font=("Arial", 12))
# entry_username.pack(pady=5)

# tk.Label(window, text="Password", font=("Arial", 12)).pack(pady=5)
# entry_password = tk.Entry(window, font=("Arial", 12), show="*")
# entry_password.pack(pady=5)

# tk.Button(window, text="Login", font=("Arial", 12), command=open_dashboard).pack(pady=20)

# window.mainloop()

#-------------------------------------------------------------------------------

# import tkinter as tk
# from tkinter import messagebox

# # Valid login credentials
# valid_users = {
#     "saadi": "1234",
#     "ali": "abcd",
#     "admin": "admin"
# }

# # --- Functions ---
# def open_dashboard():
#     username = entry_username.get()
#     password = entry_password.get()

#     # Validate user
#     if username in valid_users and valid_users[username] == password:
#         window.withdraw()  # Hide login window

#         # Create Dashboard Window
#         dashboard = tk.Toplevel()
#         dashboard.title("Dashboard")
#         dashboard.geometry("400x200")

#         # Welcome message
#         tk.Label(dashboard, text=f"🎉 Welcome, {username}!", font=("Arial", 16), fg="green").pack(pady=20)

#         # Logout button
#         tk.Button(dashboard, text="Logout", font=("Arial", 12), command=lambda: logout(dashboard)).pack(pady=5)

#         # Quit button
#         tk.Button(dashboard, text="Quit", font=("Arial", 12), fg="red", command=window.quit).pack(pady=5)
#     else:
#         messagebox.showerror("Login Failed", "Invalid username or password.")

# def logout(dashboard_window):
#     dashboard_window.destroy()
#     window.deiconify()  # Show login window again

# # --- Login Window Setup ---
# window = tk.Tk()
# window.title("Login Page")
# window.geometry("400x250")

# tk.Label(window, text="Username", font=("Arial", 12)).pack(pady=5)
# entry_username = tk.Entry(window, font=("Arial", 12))
# entry_username.pack(pady=5)

# tk.Label(window, text="Password", font=("Arial", 12)).pack(pady=5)
# entry_password = tk.Entry(window, font=("Arial", 12), show="*")
# entry_password.pack(pady=5)

# tk.Button(window, text="Login", font=("Arial", 12), command=open_dashboard).pack(pady=15)
# tk.Button(window, text="Quit", font=("Arial", 12), fg="red", command=window.quit).pack()

# # --- Start App ---
# window.mainloop()

#-------------------------------------------------------------------------------

# import tkinter as tk
# from tkinter import messagebox
# import os

# # ------------------ Helper Functions ------------------

# def save_user(username, password):
#     with open("users.txt", "a") as file:
#         file.write(f"{username},{password}\n")

# def user_exists(username):
#     if not os.path.exists("users.txt"):
#         return False
#     with open("users.txt", "r") as file:
#         for line in file:
#             if line.startswith(f"{username},"):
#                 return True
#     return False

# def validate_user(username, password):
#     if not os.path.exists("users.txt"):
#         return False
#     with open("users.txt", "r") as file:
#         for line in file:
#             stored_user, stored_pass = line.strip().split(",")
#             if stored_user == username and stored_pass == password:
#                 return True
#     return False

# ------------------ Login Functions ------------------

# def login():
#     username = entry_login_user.get()
#     password = entry_login_pass.get()

#     if validate_user(username, password):
#         messagebox.showinfo("Login Successful", f"🎉 Welcome back, {username}!")
#     else:
#         messagebox.showerror("Login Failed", "❌ Incorrect username or password.")

# def open_signup():
#     window.withdraw()  # Hide login window
#     open_signup_window()

# # ------------------ Signup Functions ------------------

# def signup():
#     username = entry_signup_user.get()
#     password = entry_signup_pass.get()

#     if not username or not password:
#         messagebox.showwarning("Empty Fields", "❗ Please fill all fields.")
#         return

#     if user_exists(username):
#         messagebox.showerror("User Exists", "⚠️ Username already taken.")
#     else:
#         save_user(username, password)
#         messagebox.showinfo("Success", "✅ Registered Successfully!")
#         signup_window.destroy()
#         window.deiconify()

# # ------------------ Signup Window ------------------

# def open_signup_window():
#     global signup_window, entry_signup_user, entry_signup_pass
#     signup_window = tk.Toplevel()
#     signup_window.title("Sign Up")
#     signup_window.geometry("350x250")

#     tk.Label(signup_window, text="Create Username", font=("Arial", 12)).pack(pady=5)
#     entry_signup_user = tk.Entry(signup_window, font=("Arial", 12))
#     entry_signup_user.pack(pady=5)

#     tk.Label(signup_window, text="Create Password", font=("Arial", 12)).pack(pady=5)
#     entry_signup_pass = tk.Entry(signup_window, font=("Arial", 12), show="*")
#     entry_signup_pass.pack(pady=5)

#     tk.Button(signup_window, text="Register", font=("Arial", 12), command=signup).pack(pady=15)
#     tk.Button(signup_window, text="Back to Login", font=("Arial", 10), command=lambda: [signup_window.destroy(), window.deiconify()]).pack()

# # ------------------ Main Login Window ------------------

# window = tk.Tk()
# window.title("Login System")
# window.geometry("350x250")

# tk.Label(window, text="Username", font=("Arial", 12)).pack(pady=5)
# entry_login_user = tk.Entry(window, font=("Arial", 12))
# entry_login_user.pack(pady=5)

# tk.Label(window, text="Password", font=("Arial", 12)).pack(pady=5)
# entry_login_pass = tk.Entry(window, font=("Arial", 12), show="*")
# entry_login_pass.pack(pady=5)

# tk.Button(window, text="Login", font=("Arial", 12), command=login).pack(pady=10)
# tk.Button(window, text="Sign Up", font=("Arial", 10), command=open_signup).pack()

# window.mainloop()

#-------------------------------------------------------------------------------

# import tkinter as tk
# from tkinter import messagebox
# import os
# import hashlib

# ------------------ Helper Functions ------------------

# def hash_password(password):
#     return hashlib.sha256(password.encode()).hexdigest()

# def save_user(username, password):
#     hashed = hash_password(password)
#     with open("users.txt", "a") as file:
#         file.write(f"{username},{hashed}\n")

# def user_exists(username):
#     if not os.path.exists("users.txt"):
#         return False
#     with open("users.txt", "r") as file:
#         for line in file:
#             if line.startswith(f"{username},"):
#                 return True
#     return False

# def validate_user(username, password):
#     if not os.path.exists("users.txt"):
#         return False
#     hashed = hash_password(password)
#     with open("users.txt", "r") as file:
#         for line in file:
#             stored_user, stored_hash = line.strip().split(",")
#             if stored_user == username and stored_hash == hashed:
#                 return True
#     return False

# # ------------------ Login Functions ------------------

# def login():
#     username = entry_login_user.get()
#     password = entry_login_pass.get()

#     if validate_user(username, password):
#         messagebox.showinfo("Login Successful", f"🎉 Welcome back, {username}!")
#     else:
#         messagebox.showerror("Login Failed", "❌ Incorrect username or password.")

# def open_signup():
#     window.withdraw()
#     open_signup_window()

# # ------------------ Signup Functions ------------------

# def signup():
#     username = entry_signup_user.get()
#     password = entry_signup_pass.get()

#     if not username or not password:
#         messagebox.showwarning("Empty Fields", "❗ Please fill all fields.")
#         return

#     if user_exists(username):
#         messagebox.showerror("User Exists", "⚠️ Username already taken.")
#     else:
#         save_user(username, password)
#         messagebox.showinfo("Success", "✅ Registered Successfully!")
#         signup_window.destroy()
#         window.deiconify()

# # ------------------ Signup Window ------------------

# def open_signup_window():
#     global signup_window, entry_signup_user, entry_signup_pass
#     signup_window = tk.Toplevel()
#     signup_window.title("Sign Up")
#     signup_window.geometry("350x250")

#     tk.Label(signup_window, text="Create Username", font=("Arial", 12)).pack(pady=5)
#     entry_signup_user = tk.Entry(signup_window, font=("Arial", 12))
#     entry_signup_user.pack(pady=5)

#     tk.Label(signup_window, text="Create Password", font=("Arial", 12)).pack(pady=5)
#     entry_signup_pass = tk.Entry(signup_window, font=("Arial", 12), show="*")
#     entry_signup_pass.pack(pady=5)

#     tk.Button(signup_window, text="Register", font=("Arial", 12), command=signup).pack(pady=15)
#     tk.Button(signup_window, text="Back to Login", font=("Arial", 10), command=lambda: [signup_window.destroy(), window.deiconify()]).pack()

# # ------------------ Main Login Window ------------------

# window = tk.Tk()
# window.title("Login System with Encrypted Passwords")
# window.geometry("350x250")

# tk.Label(window, text="Username", font=("Arial", 12)).pack(pady=5)
# entry_login_user = tk.Entry(window, font=("Arial", 12))
# entry_login_user.pack(pady=5)

# tk.Label(window, text="Password", font=("Arial", 12)).pack(pady=5)
# entry_login_pass = tk.Entry(window, font=("Arial", 12), show="*")
# entry_login_pass.pack(pady=5)

# tk.Button(window, text="Login", font=("Arial", 12), command=login).pack(pady=10)
# tk.Button(window, text="Sign Up", font=("Arial", 10), command=open_signup).pack()

# window.mainloop()

#-------------------------------------------------------------------------------

# import tkinter as tk
# from tkinter import ttk

# # --- Create Main Window ---
# window = tk.Tk()
# window.title("Tabbed Interface Example")
# window.geometry("400x300")

# # --- Create Notebook (Tabs Container) ---
# notebook = ttk.Notebook(window)
# notebook.pack(expand=True, fill="both")

# # --- Tab 1: Home ---
# home_tab = tk.Frame(notebook)
# notebook.add(home_tab, text="🏠 Home")

# tk.Label(home_tab, text="Welcome to the Home Tab!", font=("Arial", 14)).pack(pady=20)
# tk.Button(home_tab, text="Click Me", font=("Arial", 12), command=lambda: print("Home Button Clicked")).pack()

# # --- Tab 2: Profile ---
# profile_tab = tk.Frame(notebook)
# notebook.add(profile_tab, text="👤 Profile")

# tk.Label(profile_tab, text="Name:", font=("Arial", 12)).grid(row=0, column=0, padx=10, pady=10, sticky="e")
# tk.Entry(profile_tab, font=("Arial", 12)).grid(row=0, column=1)

# tk.Label(profile_tab, text="Age:", font=("Arial", 12)).grid(row=1, column=0, padx=10, pady=10, sticky="e")
# tk.Entry(profile_tab, font=("Arial", 12)).grid(row=1, column=1)

# # --- Tab 3: Settings ---
# settings_tab = tk.Frame(notebook)
# notebook.add(settings_tab, text="⚙️ Settings")

# tk.Label(settings_tab, text="Choose Theme:", font=("Arial", 12)).pack(pady=10)
# tk.Radiobutton(settings_tab, text="Light", value=1).pack()
# tk.Radiobutton(settings_tab, text="Dark", value=2).pack()

# # --- Start App ---
# window.mainloop()

#-------------------------------------------------------------------------------

# import tkinter as tk
# from tkinter import ttk

# def submit_info():
#     name = entry_name.get()
#     password = entry_pass.get()
#     output_label.config(text=f"👤 Name: {name}\n🔐 Password: {password}")

# # --- Create Main Window ---
# window = tk.Tk()
# window.title("Tabbed Interface Form")
# window.geometry("450x350")

# # --- Create Notebook (Tabs Container) ---
# notebook = ttk.Notebook(window)
# notebook.pack(expand=True, fill="both")

# # --- Tab 1: Welcome ---
# home_tab = tk.Frame(notebook)
# notebook.add(home_tab, text="🏠 Welcome")

# tk.Label(home_tab, text="Welcome to the Home Tab!", font=("Arial", 14)).pack(pady=20)
# tk.Button(home_tab, text="Click Me", font=("Arial", 12), command=lambda: print("Home Button Clicked")).pack()

# # --- Tab 2: Login Form ---
# profile_tab = tk.Frame(notebook)
# notebook.add(profile_tab, text="👤 Login Form")

# tk.Label(profile_tab, text="Name:", font=("Arial", 12)).grid(row=0, column=0, padx=10, pady=10, sticky="e")
# entry_name = tk.Entry(profile_tab, font=("Arial", 12))
# entry_name.grid(row=0, column=1, padx=10, pady=10)

# tk.Label(profile_tab, text="Password:", font=("Arial", 12)).grid(row=1, column=0, padx=10, pady=10, sticky="e")
# entry_pass = tk.Entry(profile_tab, font=("Arial", 12), show="*")
# entry_pass.grid(row=1, column=1, padx=10, pady=10)

# tk.Button(profile_tab, text="Submit", font=("Arial", 12), command=submit_info).grid(row=2, columnspan=2, pady=15)

# # --- Tab 3: Output Tab ---
# settings_tab = tk.Frame(notebook)
# notebook.add(settings_tab, text="📋 Output")

# output_label = tk.Label(settings_tab, text="", font=("Arial", 12), fg="green", justify="left")
# output_label.pack(pady=30)

# # --- Start App ---
# window.mainloop()

#-------------------------------------------------------------------------------

# import tkinter as tk
# from tkinter import ttk

# def submit_profile():
#     name = entry_name.get()
#     age = entry_age.get()
#     gender = gender_var.get()
#     country = country_var.get()

#     result = f"👤 Name: {name}\n🎂 Age: {age}\n🚻 Gender: {gender}\n🌍 Country: {country}"
#     output_label.config(text=result)

# def clear_form():
#     entry_name.delete(0, tk.END)
#     entry_age.delete(0, tk.END)
#     gender_var.set(None)
#     country_var.set("Select")

# # --- Main Window ---
# window = tk.Tk()
# window.title("Profile Manager App")
# window.geometry("450x350")

# # --- Notebook ---
# notebook = ttk.Notebook(window)
# notebook.pack(expand=True, fill="both")

# # ---------------- Tab 1: Profile Input ----------------
# tab_profile = tk.Frame(notebook)
# notebook.add(tab_profile, text="📝 Profile")

# tk.Label(tab_profile, text="Name:", font=("Arial", 12)).grid(row=0, column=0, padx=10, pady=5, sticky="e")
# entry_name = tk.Entry(tab_profile, font=("Arial", 12))
# entry_name.grid(row=0, column=1, padx=10, pady=5)

# tk.Label(tab_profile, text="Age:", font=("Arial", 12)).grid(row=1, column=0, padx=10, pady=5, sticky="e")
# entry_age = tk.Entry(tab_profile, font=("Arial", 12))
# entry_age.grid(row=1, column=1, padx=10, pady=5)

# tk.Label(tab_profile, text="Gender:", font=("Arial", 12)).grid(row=2, column=0, padx=10, pady=5, sticky="e")
# gender_var = tk.StringVar()
# tk.Radiobutton(tab_profile, text="Male", variable=gender_var, value="Male").grid(row=2, column=1, sticky="w")
# tk.Radiobutton(tab_profile, text="Female", variable=gender_var, value="Female").grid(row=3, column=1, sticky="w")

# tk.Label(tab_profile, text="Country:", font=("Arial", 12)).grid(row=4, column=0, padx=10, pady=5, sticky="e")
# country_var = tk.StringVar(value="Select")
# ttk.Combobox(tab_profile, textvariable=country_var, values=["Pakistan", "USA", "UK", "Canada", "Germany"], font=("Arial", 12)).grid(row=4, column=1, padx=10, pady=5)

# tk.Button(tab_profile, text="Submit", font=("Arial", 12), command=submit_profile).grid(row=5, columnspan=2, pady=15)

# # ---------------- Tab 2: Profile Summary ----------------
# tab_summary = tk.Frame(notebook)
# notebook.add(tab_summary, text="📋 Summary")

# output_label = tk.Label(tab_summary, text="Fill out your profile to see summary here.", font=("Arial", 12), fg="green", justify="left")
# output_label.pack(pady=30)

# # ---------------- Tab 3: Actions ----------------
# tab_actions = tk.Frame(notebook)
# notebook.add(tab_actions, text="⚙️ Actions")

# tk.Button(tab_actions, text="🧹 Clear Form", font=("Arial", 12), command=clear_form).pack(pady=20)
# tk.Button(tab_actions, text="❌ Exit App", font=("Arial", 12), fg="red", command=window.quit).pack(pady=10)

# # --- Run App ---
# window.mainloop()

#-------------------------------------------------------------------------------

# Just open your terminal and type:
# "pip install pillow"
# Then hit Enter.

#-------------------------------------------------------------------------------

# import tkinter as tk
# from tkinter import filedialog
# from PIL import Image, ImageTk  # Pillow is now installed!

# def upload_image():
#     filepath = filedialog.askopenfilename(
#         filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif")]
#     )
#     if filepath:
#         # Open and resize the image
#         img = Image.open(filepath)
#         img = img.resize((350, 350))  # Resize image to fit window
#         photo = ImageTk.PhotoImage(img)

#         # Update the label with the image
#         image_label.config(image=photo)
#         image_label.image = photo  # Keep a reference!

# # Create window
# window = tk.Tk()
# window.title("🖼️ Image Uploader")
# window.geometry("400x400")

# # Upload button
# btn = tk.Button(window, text="📁 Upload Image", font=("Arial", 12), command=upload_image)
# btn.pack(pady=20)

# # Image display area
# image_label = tk.Label(window)
# image_label.pack(pady=20)

# window.mainloop()

#-------------------------------------------------------------------------------

# import tkinter as tk
# from tkinter import messagebox
# from PIL import Image, ImageTk
# import os

# # --- Load Image Paths ---
# image_folder = "."  # Current folder
# image_files = [file for file in os.listdir(image_folder) if file.lower().endswith((".jpg", ".jpeg", ".png", ".gif"))]

# if not image_files:
#     messagebox.showerror("No Images", "❌ No image files found in the folder!")
#     exit()

# current_index = 0

# def load_image(index):
#     global photo
#     img = Image.open(image_files[index])
#     img = img.resize((300, 300))  # Resize to fit the label
#     photo = ImageTk.PhotoImage(img)
#     image_label.config(image=photo)
#     image_label.image = photo
#     label_info.config(text=f"Showing {index + 1} of {len(image_files)}")

# def next_image():
#     global current_index
#     if current_index < len(image_files) - 1:
#         current_index += 1
#         load_image(current_index)

# def prev_image():
#     global current_index
#     if current_index > 0:
#         current_index -= 1
#         load_image(current_index)

# # --- Create Window ---
# window = tk.Tk()
# window.title("🖼️ Image Viewer")
# window.geometry("400x420")

# # --- Image Display ---
# image_label = tk.Label(window)
# image_label.pack(pady=10)

# # --- Image Counter Info ---
# label_info = tk.Label(window, text="", font=("Arial", 12))
# label_info.pack()

# # --- Buttons ---
# btn_frame = tk.Frame(window)
# btn_frame.pack(pady=10)

# btn_prev = tk.Button(btn_frame, text="⬅️ Previous", font=("Arial", 10), command=prev_image)
# btn_prev.grid(row=0, column=0, padx=10)

# btn_next = tk.Button(btn_frame, text="Next ➡️", font=("Arial", 10), command=next_image)
# btn_next.grid(row=0, column=1, padx=10)

# # --- Load First Image ---
# load_image(current_index)

# # --- Run ---
# window.mainloop()

#-------------------------------------------------------------------------------

# import tkinter as tk
# from tkinter import filedialog, messagebox
# from PIL import Image, ImageTk
# import os

# # --- Globals ---
# image_files = []
# current_index = 0
# slideshow_active = False

# def select_folder():
#     global image_files, current_index
#     folder = filedialog.askdirectory()
#     if not folder:
#         return

#     # Load images
#     image_files = [os.path.join(folder, f) for f in os.listdir(folder)
#                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".gif"))]

#     if not image_files:
#         messagebox.showerror("Error", "No images found in selected folder.")
#         return

#     current_index = 0
#     load_image(current_index)

# def load_image(index):
#     global photo
#     img_path = image_files[index]
#     img = Image.open(img_path)
#     img = img.resize((320, 280))
#     photo = ImageTk.PhotoImage(img)

#     image_label.config(image=photo)
#     image_label.image = photo

#     label_info.config(text=f"Showing {index + 1} of {len(image_files)}")
#     label_name.config(text=os.path.basename(img_path))

# def next_image():
#     global current_index
#     if current_index < len(image_files) - 1:
#         current_index += 1
#         load_image(current_index)

# def prev_image():
#     global current_index
#     if current_index > 0:
#         current_index -= 1
#         load_image(current_index)

# def start_slideshow():
#     global slideshow_active
#     slideshow_active = True
#     run_slideshow()

# def stop_slideshow():
#     global slideshow_active
#     slideshow_active = False

# def run_slideshow():
#     if slideshow_active:
#         next_image()
#         window.after(2000, run_slideshow)  # Change every 2 seconds

# # --- GUI Setup ---
# window = tk.Tk()
# window.title("🖼️ Advanced Image Viewer")
# window.geometry("450x480")

# # --- Folder Select ---
# tk.Button(window, text="📁 Select Image Folder", font=("Arial", 12), command=select_folder).pack(pady=10)

# # --- Image Display ---
# image_label = tk.Label(window)
# image_label.pack(pady=10)

# # --- File Info ---
# label_name = tk.Label(window, text="", font=("Arial", 10), fg="gray")
# label_name.pack()

# label_info = tk.Label(window, text="", font=("Arial", 12))
# label_info.pack(pady=5)

# # --- Navigation Buttons ---
# nav_frame = tk.Frame(window)
# nav_frame.pack(pady=10)

# tk.Button(nav_frame, text="⬅️ Prev", font=("Arial", 11), command=prev_image).grid(row=0, column=0, padx=5)
# tk.Button(nav_frame, text="Next ➡️", font=("Arial", 11), command=next_image).grid(row=0, column=1, padx=5)

# # --- Slideshow Controls ---
# slide_frame = tk.Frame(window)
# slide_frame.pack(pady=10)

# tk.Button(slide_frame, text="▶️ Start Slideshow", font=("Arial", 11), command=start_slideshow).grid(row=0, column=0, padx=10)
# tk.Button(slide_frame, text="⏹️ Stop", font=("Arial", 11), fg="red", command=stop_slideshow).grid(row=0, column=1, padx=10)

# # --- Run App ---
# window.mainloop()

#-------------------------------------------------------------------------------

# import tkinter as tk
# from tkinter import filedialog, messagebox
# from PIL import Image, ImageTk
# import os

# # --- Globals ---
# image_files = []
# current_index = 0
# slideshow_active = False
# zoom_level = 1.0

# def select_folder():
#     global image_files, current_index, zoom_level
#     folder = filedialog.askdirectory()
#     if not folder:
#         return

#     image_files = [os.path.join(folder, f) for f in os.listdir(folder)
#                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".gif"))]

#     if not image_files:
#         messagebox.showerror("Error", "No images found in selected folder.")
#         return

#     current_index = 0
#     zoom_level = 1.0
#     load_image(current_index)

# def load_image(index):
#     global photo, zoom_level
#     img_path = image_files[index]
#     img = Image.open(img_path)

#     # Apply zoom level
#     width, height = img.size
#     new_size = (int(width * zoom_level), int(height * zoom_level))
#     img = img.resize(new_size)

#     photo = ImageTk.PhotoImage(img)
#     image_label.config(image=photo)
#     image_label.image = photo

#     label_name.config(text=os.path.basename(img_path))
#     label_info.config(text=f"Image {index + 1} of {len(image_files)} — Zoom: {int(zoom_level * 100)}%")

# def next_image():
#     global current_index, zoom_level
#     if current_index < len(image_files) - 1:
#         current_index += 1
#         zoom_level = 1.0
#         load_image(current_index)

# def prev_image():
#     global current_index, zoom_level
#     if current_index > 0:
#         current_index -= 1
#         zoom_level = 1.0
#         load_image(current_index)

# def zoom_in():
#     global zoom_level
#     zoom_level += 0.1
#     load_image(current_index)

# def zoom_out():
#     global zoom_level
#     if zoom_level > 0.2:
#         zoom_level -= 0.1
#         load_image(current_index)

# def reset_zoom():
#     global zoom_level
#     zoom_level = 1.0
#     load_image(current_index)

# def start_slideshow():
#     global slideshow_active
#     slideshow_active = True
#     run_slideshow()

# def stop_slideshow():
#     global slideshow_active
#     slideshow_active = False

# def run_slideshow():
#     if slideshow_active:
#         next_image()
#         window.after(2000, run_slideshow)

# # --- GUI Setup ---
# window = tk.Tk()
# window.title("🔍 Image Viewer with Zoom")
# window.geometry("500x600")

# # --- Select Folder ---
# tk.Button(window, text="📁 Select Folder", font=("Arial", 12), command=select_folder).pack(pady=10)

# # --- Image Display ---
# image_label = tk.Label(window)
# image_label.pack(pady=10)

# # --- File Info ---
# label_name = tk.Label(window, text="", font=("Arial", 10), fg="gray")
# label_name.pack()

# label_info = tk.Label(window, text="", font=("Arial", 12))
# label_info.pack(pady=5)

# # --- Navigation Buttons ---
# nav_frame = tk.Frame(window)
# nav_frame.pack(pady=5)

# tk.Button(nav_frame, text="⬅️ Prev", font=("Arial", 11), command=prev_image).grid(row=0, column=0, padx=5)
# tk.Button(nav_frame, text="Next ➡️", font=("Arial", 11), command=next_image).grid(row=0, column=1, padx=5)

# # --- Zoom Controls ---
# zoom_frame = tk.Frame(window)
# zoom_frame.pack(pady=10)

# tk.Button(zoom_frame, text="➕ Zoom In", font=("Arial", 11), command=zoom_in).grid(row=0, column=0, padx=10)
# tk.Button(zoom_frame, text="➖ Zoom Out", font=("Arial", 11), command=zoom_out).grid(row=0, column=1, padx=10)
# tk.Button(zoom_frame, text="🔁 Reset Zoom", font=("Arial", 11), command=reset_zoom).grid(row=0, column=2, padx=10)

# # --- Slideshow Controls ---
# slide_frame = tk.Frame(window)
# slide_frame.pack(pady=10)

# tk.Button(slide_frame, text="▶️ Slideshow", font=("Arial", 11), command=start_slideshow).grid(row=0, column=0, padx=10)
# tk.Button(slide_frame, text="⏹️ Stop", font=("Arial", 11), fg="red", command=stop_slideshow).grid(row=0, column=1, padx=10)

# # --- Run App ---
# window.mainloop()

#-------------------------------------------------------------------------------

# import tkinter as tk
# from tkinter import filedialog, messagebox
# from PIL import Image, ImageTk
# import os

# # --- Globals ---
# image_files = []
# current_index = 0
# slideshow_active = False
# zoom_level = 1.0
# fit_to_window = False

# def select_folder():
#     global image_files, current_index, zoom_level
#     folder = filedialog.askdirectory()
#     if not folder:
#         return

#     image_files = [os.path.join(folder, f) for f in os.listdir(folder)
#                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".gif"))]

#     if not image_files:
#         messagebox.showerror("Error", "No images found in selected folder.")
#         return

#     current_index = 0
#     zoom_level = 1.0
#     load_image(current_index)

# def load_image(index):
#     global photo, zoom_level
#     img_path = image_files[index]
#     img = Image.open(img_path)

#     if fit_to_window:
#         width = image_label.winfo_width()
#         height = image_label.winfo_height()
#         if width > 0 and height > 0:
#             img = img.resize((width, height))
#     else:
#         width, height = img.size
#         new_size = (int(width * zoom_level), int(height * zoom_level))
#         img = img.resize(new_size)

#     photo = ImageTk.PhotoImage(img)
#     image_label.config(image=photo)
#     image_label.image = photo

#     label_name.config(text=os.path.basename(img_path))
#     label_info.config(text=f"Image {index + 1} of {len(image_files)} — Zoom: {int(zoom_level * 100)}%")

# def next_image():
#     global current_index, zoom_level
#     if current_index < len(image_files) - 1:
#         current_index += 1
#         zoom_level = 1.0
#         load_image(current_index)

# def prev_image():
#     global current_index, zoom_level
#     if current_index > 0:
#         current_index -= 1
#         zoom_level = 1.0
#         load_image(current_index)

# def zoom_in():
#     global zoom_level, fit_to_window
#     fit_to_window = False
#     zoom_level += 0.1
#     load_image(current_index)

# def zoom_out():
#     global zoom_level, fit_to_window
#     if zoom_level > 0.2:
#         fit_to_window = False
#         zoom_level -= 0.1
#         load_image(current_index)

# def reset_zoom():
#     global zoom_level, fit_to_window
#     zoom_level = 1.0
#     fit_to_window = False
#     load_image(current_index)

# def toggle_fit():
#     global fit_to_window
#     fit_to_window = not fit_to_window
#     load_image(current_index)

# def toggle_fullscreen():
#     window.attributes("-fullscreen", not window.attributes("-fullscreen"))

# def start_slideshow():
#     global slideshow_active
#     slideshow_active = True
#     run_slideshow()

# def stop_slideshow():
#     global slideshow_active
#     slideshow_active = False

# def run_slideshow():
#     if slideshow_active:
#         next_image()
#         window.after(2000, run_slideshow)

# def handle_scroll(event):
#     if event.delta > 0:
#         zoom_in()
#     else:
#         zoom_out()

# # --- GUI Setup ---
# window = tk.Tk()
# window.title("🖼️ Image Viewer (Advanced)")
# window.geometry("800x600")

# # --- Select Folder ---
# tk.Button(window, text="📁 Select Folder", font=("Arial", 12), command=select_folder).pack(pady=10)

# # --- Image Display ---
# image_label = tk.Label(window, bg="black")
# image_label.pack(expand=True, fill="both", padx=10, pady=10)
# image_label.bind("<MouseWheel>", handle_scroll)  # Mouse scroll zoom

# # --- File Info ---
# label_name = tk.Label(window, text="", font=("Arial", 10), fg="gray")
# label_name.pack()

# label_info = tk.Label(window, text="", font=("Arial", 12))
# label_info.pack(pady=5)

# # --- Navigation Buttons ---
# nav_frame = tk.Frame(window)
# nav_frame.pack(pady=5)

# tk.Button(nav_frame, text="⬅️ Prev", font=("Arial", 11), command=prev_image).grid(row=0, column=0, padx=5)
# tk.Button(nav_frame, text="Next ➡️", font=("Arial", 11), command=next_image).grid(row=0, column=1, padx=5)

# # --- Zoom Controls ---
# zoom_frame = tk.Frame(window)
# zoom_frame.pack(pady=10)

# tk.Button(zoom_frame, text="➕ Zoom In", font=("Arial", 11), command=zoom_in).grid(row=0, column=0, padx=10)
# tk.Button(zoom_frame, text="➖ Zoom Out", font=("Arial", 11), command=zoom_out).grid(row=0, column=1, padx=10)
# tk.Button(zoom_frame, text="🔁 Reset Zoom", font=("Arial", 11), command=reset_zoom).grid(row=0, column=2, padx=10)
# tk.Button(zoom_frame, text="🪟 Fit to Window", font=("Arial", 11), command=toggle_fit).grid(row=0, column=3, padx=10)

# # --- Slideshow Controls ---
# slide_frame = tk.Frame(window)
# slide_frame.pack(pady=10)

# tk.Button(slide_frame, text="▶️ Slideshow", font=("Arial", 11), command=start_slideshow).grid(row=0, column=0, padx=10)
# tk.Button(slide_frame, text="⏹️ Stop", font=("Arial", 11), fg="red", command=stop_slideshow).grid(row=0, column=1, padx=10)

# # --- Fullscreen Toggle ---
# tk.Button(window, text="🖥️ Toggle Fullscreen", font=("Arial", 11), command=toggle_fullscreen).pack(pady=10)

# # --- Run App ---
# window.mainloop()

#-------------------------------------------------------------------------------

# import tkinter as tk

# # --- Add Task ---
# def add_task():
#     task = entry.get().strip()
#     if task:
#         listbox.insert(tk.END, task)
#         entry.delete(0, tk.END)

# # --- Main Window ---
# window = tk.Tk()
# window.title("📝 To-Do List App")
# window.geometry("400x400")

# # --- Input Field ---
# entry = tk.Entry(window, font=("Arial", 14))
# entry.pack(pady=10)

# # --- Add Button ---
# add_button = tk.Button(window, text="Add Task", font=("Arial", 12), command=add_task)
# add_button.pack(pady=5)

# # --- Listbox to Show Tasks ---
# listbox = tk.Listbox(window, font=("Arial", 12), width=30, height=10)
# listbox.pack(pady=10)

# # --- Run the App ---
# window.mainloop()

#-------------------------------------------------------------------------------

# import tkinter as tk
# import os

# FILE_NAME = "tasks.txt"

# # --- Add Task ---
# def add_task():
#     task = entry.get().strip()
#     if task:
#         listbox.insert(tk.END, task)
#         entry.delete(0, tk.END)
#         save_tasks()

# # --- Save All Tasks to File ---
# def save_tasks():
#     with open(FILE_NAME, "w") as file:
#         for i in range(listbox.size()):
#             file.write(listbox.get(i) + "\n")

# # --- Load Tasks from File ---
# def load_tasks():
#     if os.path.exists(FILE_NAME):
#         with open(FILE_NAME, "r") as file:
#             for line in file:
#                 listbox.insert(tk.END, line.strip())

# # --- Main Window ---
# window = tk.Tk()
# window.title("📝 To-Do List App")
# window.geometry("400x400")

# # --- Entry ---
# entry = tk.Entry(window, font=("Arial", 14))
# entry.pack(pady=10)

# # --- Add Button ---
# add_button = tk.Button(window, text="Add Task", font=("Arial", 12), command=add_task)
# add_button.pack(pady=5)

# # --- Listbox ---
# listbox = tk.Listbox(window, font=("Arial", 12), width=30, height=10)
# listbox.pack(pady=10)

# # --- Load Existing Tasks ---
# load_tasks()

# # --- Run ---
# window.mainloop()

#-------------------------------------------------------------------------------

# import tkinter as tk
# import os

# FILE_NAME = "tasks.txt"

# # --- Add Task ---
# def add_task():
#     task = entry.get().strip()
#     if task:
#         listbox.insert(tk.END, task)
#         entry.delete(0, tk.END)
#         save_tasks()

# # --- Delete Selected Task ---
# def delete_task():
#     selected = listbox.curselection()
#     if selected:
#         listbox.delete(selected)
#         save_tasks()

# # --- Save All Tasks ---
# def save_tasks():
#     with open(FILE_NAME, "w") as file:
#         for i in range(listbox.size()):
#             file.write(listbox.get(i) + "\n")

# # --- Load Tasks ---
# def load_tasks():
#     if os.path.exists(FILE_NAME):
#         with open(FILE_NAME, "r") as file:
#             for line in file:
#                 listbox.insert(tk.END, line.strip())

# # --- GUI Setup ---
# window = tk.Tk()
# window.title("📝 To-Do List App")
# window.geometry("400x450")

# # --- Entry ---
# entry = tk.Entry(window, font=("Arial", 14))
# entry.pack(pady=10)

# # --- Buttons ---
# btn_frame = tk.Frame(window)
# btn_frame.pack()

# tk.Button(btn_frame, text="Add Task", font=("Arial", 12), command=add_task).grid(row=0, column=0, padx=5)
# tk.Button(btn_frame, text="Delete Task", font=("Arial", 12), command=delete_task, fg="red").grid(row=0, column=1, padx=5)

# # --- Listbox ---
# listbox = tk.Listbox(window, font=("Arial", 12), width=35, height=12)
# listbox.pack(pady=10)

# # --- Load existing tasks ---
# load_tasks()

# # --- Run App ---
# window.mainloop()

#-------------------------------------------------------------------------------

# import tkinter as tk
# import os

# FILE_NAME = "tasks.txt"

# # --- Add Task ---
# def add_task():
#     task = entry.get().strip()
#     if task:
#         listbox.insert(tk.END, task)
#         entry.delete(0, tk.END)
#         save_tasks()

# # --- Delete Selected Task ---
# def delete_task():
#     selected = listbox.curselection()
#     if selected:
#         listbox.delete(selected)
#         save_tasks()

# # --- Toggle Completed (✔️) on Double-Click ---
# def toggle_complete(event):
#     index = listbox.curselection()
#     if not index:
#         return
#     text = listbox.get(index)
#     if text.startswith("✔️ "):
#         text = text[2:]  # remove check
#     else:
#         text = "✔️ " + text  # add check
#     listbox.delete(index)
#     listbox.insert(index, text)
#     save_tasks()

# # --- Save All Tasks to File ---
# def save_tasks():
#     with open(FILE_NAME, "w", encoding="utf-8") as file:
#         for i in range(listbox.size()):
#             file.write(listbox.get(i) + "\n")

# # --- Load Tasks from File ---
# def load_tasks():
#     if os.path.exists(FILE_NAME):
#         with open(FILE_NAME, "r", encoding="utf-8") as file:
#             for line in file:
#                 listbox.insert(tk.END, line.strip())

# # --- GUI ---
# window = tk.Tk()
# window.title("📝 To-Do List App")
# window.geometry("400x480")

# entry = tk.Entry(window, font=("Arial", 14))
# entry.pack(pady=10)

# # --- Buttons ---
# btn_frame = tk.Frame(window)
# btn_frame.pack()

# tk.Button(btn_frame, text="Add Task", font=("Arial", 12), command=add_task).grid(row=0, column=0, padx=5)
# tk.Button(btn_frame, text="Delete Task", font=("Arial", 12), command=delete_task, fg="red").grid(row=0, column=1, padx=5)

# # --- Task List ---
# listbox = tk.Listbox(window, font=("Arial", 12), width=35, height=12)
# listbox.pack(pady=10)
# listbox.bind("<Double-Button-1>", toggle_complete)

# # --- Load tasks on startup ---
# load_tasks()

# window.mainloop()

#-------------------------------------------------------------------------------

# import tkinter as tk
# from tkinter import messagebox
# import os

# FILE_NAME = "expenses.txt"

# # --- Add Expense ---
# def add_expense():
#     name = entry_name.get().strip()
#     amount = entry_amount.get().strip()

#     if not name or not amount:
#         messagebox.showwarning("Input Error", "Please enter both name and amount.")
#         return

#     try:
#         float(amount)
#     except ValueError:
#         messagebox.showerror("Invalid Amount", "Amount must be a number.")
#         return

#     entry = f"{name} - ${amount}"
#     listbox.insert(tk.END, entry)
#     entry_name.delete(0, tk.END)
#     entry_amount.delete(0, tk.END)
#     save_expenses()
#     update_total()

# # --- Delete Selected Expense ---
# def delete_expense():
#     selected = listbox.curselection()
#     if selected:
#         listbox.delete(selected)
#         save_expenses()
#         update_total()

# # --- Save All Expenses to File ---
# def save_expenses():
#     with open(FILE_NAME, "w") as file:
#         for i in range(listbox.size()):
#             file.write(listbox.get(i) + "\n")

# # --- Load Expenses on Startup ---
# def load_expenses():
#     if os.path.exists(FILE_NAME):
#         with open(FILE_NAME, "r") as file:
#             for line in file:
#                 listbox.insert(tk.END, line.strip())

# # --- Calculate and Update Total ---
# def update_total():
#     total = 0.0
#     for i in range(listbox.size()):
#         line = listbox.get(i)
#         try:
#             amount = float(line.split("$")[-1])
#             total += amount
#         except:
#             continue
#     total_label.config(text=f"💰 Total Spent: ${total:.2f}")

# # --- Clear All Expenses ---
# def clear_all():
#     if messagebox.askyesno("Clear All", "Are you sure you want to delete all expenses?"):
#         listbox.delete(0, tk.END)
#         save_expenses()
#         update_total()

# # --- GUI ---
# window = tk.Tk()
# window.title("💵 Personal Finance Tracker")
# window.geometry("450x500")
# window.resizable(False, False)

# tk.Label(window, text="Expense Name:", font=("Arial", 12)).pack(pady=5)
# entry_name = tk.Entry(window, font=("Arial", 12), width=30)
# entry_name.pack(pady=5)

# tk.Label(window, text="Amount ($):", font=("Arial", 12)).pack(pady=5)
# entry_amount = tk.Entry(window, font=("Arial", 12), width=30)
# entry_amount.pack(pady=5)

# # --- Buttons ---
# btn_frame = tk.Frame(window)
# btn_frame.pack(pady=10)

# tk.Button(btn_frame, text="Add Expense", font=("Arial", 12), command=add_expense).grid(row=0, column=0, padx=5)
# tk.Button(btn_frame, text="Delete Selected", font=("Arial", 12), command=delete_expense, fg="red").grid(row=0, column=1, padx=5)
# tk.Button(btn_frame, text="Clear All", font=("Arial", 12), command=clear_all, fg="orange").grid(row=0, column=2, padx=5)

# # --- Listbox for Expenses ---
# listbox = tk.Listbox(window, font=("Arial", 12), width=40, height=12)
# listbox.pack(pady=10)

# # --- Total Display ---
# total_label = tk.Label(window, text="💰 Total Spent: $0.00", font=("Arial", 13, "bold"), fg="green")
# total_label.pack(pady=10)

# # --- Load Previous Expenses ---
# load_expenses()
# update_total()

# # --- Start App ---
# window.mainloop()

#-------------------------------------------------------------------------------

# import tkinter as tk
# from tkinter import messagebox

# # --- Questions and Answers ---
# questions = [
#     {
#         "question": "What is the capital of France?",
#         "options": ["Paris", "London", "Rome", "Berlin"],
#         "answer": "Paris"
#     },
#     {
#         "question": "Which language is used for web development?",
#         "options": ["Python", "C++", "JavaScript", "Java"],
#         "answer": "JavaScript"
#     },
#     {
#         "question": "Who developed Python?",
#         "options": ["Elon Musk", "Mark Zuckerberg", "Guido van Rossum", "Bill Gates"],
#         "answer": "Guido van Rossum"
#     },
#     {
#         "question": "Which of these is a Python framework?",
#         "options": ["React", "Vue", "Django", "Laravel"],
#         "answer": "Django"
#     }
# ]

# # --- Global Variables ---
# current_question = 0
# score = 0

# # --- Load Question ---
# def load_question():
#     selected_option.set(None)
#     q = questions[current_question]
#     question_label.config(text=f"Q{current_question + 1}: {q['question']}")
#     for i in range(4):
#         options[i].config(text=q['options'][i], value=q['options'][i])

# # --- Check Answer & Move Next ---
# def next_question():
#     global current_question, score

#     if not selected_option.get():
#         messagebox.showwarning("Warning", "Please select an option!")
#         return

#     if selected_option.get() == questions[current_question]['answer']:
#         score += 1

#     current_question += 1

#     if current_question < len(questions):
#         load_question()
#     else:
#         show_result()

# # --- Show Final Score ---
# def show_result():
#     messagebox.showinfo("Quiz Finished", f"🎉 You scored {score} out of {len(questions)}")
#     window.destroy()

# # --- GUI Setup ---
# window = tk.Tk()
# window.title("🧠 Python Quiz App")
# window.geometry("500x350")
# selected_option = tk.StringVar()


# question_label = tk.Label(window, text="", font=("Arial", 14), wraplength=450, justify="left")
# question_label.pack(pady=20)

# options = []
# for _ in range(4):
#     opt = tk.Radiobutton(window, text="", font=("Arial", 12), variable=selected_option, value="", anchor="w", padx=20)
#     opt.pack(fill="x", padx=20, pady=5)
#     options.append(opt)

# next_btn = tk.Button(window, text="Next", font=("Arial", 12), command=next_question)
# next_btn.pack(pady=20)

# load_question()
# window.mainloop()

#-------------------------------------------------------------------------------

# import tkinter as tk
# import random

# # List of quotes
# quotes = [
#     "Believe you can and you're halfway there.",
#     "Success is not final, failure is not fatal.",
#     "Do something today that your future self will thank you for.",
#     "Dream big and dare to fail.",
#     "The harder you work for something, the greater you’ll feel when you achieve it.",
#     "Great things never come from comfort zones.",
#     "Stay positive, work hard, make it happen.",
#     "Your limitation—it’s only your imagination."
# ]

# # Function to display a new quote
# def show_quote():
#     quote = random.choice(quotes)
#     quote_label.config(text=f"“{quote}”")

# # Create main window
# root = tk.Tk()
# root.title("🌟 Inspirational Quote Generator")
# root.geometry("500x300")
# root.configure(bg="#f0f0f0")

# # Heading
# title = tk.Label(root, text="Get Inspired!", font=("Helvetica", 20, "bold"), bg="#f0f0f0", fg="#333")
# title.pack(pady=10)

# # Quote display
# quote_label = tk.Label(root, text="Click the button below 👇", wraplength=400,
#                        font=("Helvetica", 14), bg="#f0f0f0", fg="#555", justify="center")
# quote_label.pack(pady=20)

# # Button to generate quote
# generate_btn = tk.Button(root, text="✨ New Quote", font=("Helvetica", 12, "bold"),
#                          bg="#4caf50", fg="white", padx=20, pady=10, command=show_quote)
# generate_btn.pack()

# # Run the GUI
# root.mainloop()

#--------------------------------------------------------------------------------------------

# import os
# import tkinter as tk
# from tkinter import filedialog, messagebox
# from PIL import Image, ImageTk

# class ImageViewerApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("📷 Smart Image Viewer")
#         self.root.geometry("900x600")
#         self.root.configure(bg="#2E3440")

#         self.image_files = []
#         self.current_index = 0
#         self.zoom_factor = 1.0
#         self.slideshow_running = False
#         self.fullscreen = False
#         self.fit_to_window = False

#         self.setup_ui()

#     def setup_ui(self):
#         # Menu
#         menubar = tk.Menu(self.root)
#         filemenu = tk.Menu(menubar, tearoff=0)
#         filemenu.add_command(label="Open Folder", command=self.open_folder)
#         filemenu.add_separator()
#         filemenu.add_command(label="Exit", command=self.root.quit)
#         menubar.add_cascade(label="File", menu=filemenu)

#         self.root.config(menu=menubar)

#         # Image display area
#         self.canvas = tk.Canvas(self.root, bg="#3B4252", highlightthickness=0)
#         self.canvas.pack(fill=tk.BOTH, expand=True)

#         # Controls
#         control_frame = tk.Frame(self.root, bg="#2E3440")
#         control_frame.pack(fill=tk.X, pady=10)

#         self.prev_btn = tk.Button(control_frame, text="⏮️ Previous", command=self.show_prev)
#         self.prev_btn.pack(side=tk.LEFT, padx=5)

#         self.next_btn = tk.Button(control_frame, text="Next ⏭️", command=self.show_next)
#         self.next_btn.pack(side=tk.LEFT, padx=5)

#         self.zoom_in_btn = tk.Button(control_frame, text="Zoom ➕", command=self.zoom_in)
#         self.zoom_in_btn.pack(side=tk.LEFT, padx=5)

#         self.zoom_out_btn = tk.Button(control_frame, text="Zoom ➖", command=self.zoom_out)
#         self.zoom_out_btn.pack(side=tk.LEFT, padx=5)

#         self.fit_btn = tk.Button(control_frame, text="🖼️ Fit to Window", command=self.toggle_fit)
#         self.fit_btn.pack(side=tk.LEFT, padx=5)

#         self.slideshow_btn = tk.Button(control_frame, text="▶️ Slideshow", command=self.toggle_slideshow)
#         self.slideshow_btn.pack(side=tk.LEFT, padx=5)

#         self.fullscreen_btn = tk.Button(control_frame, text="🖥️ Fullscreen", command=self.toggle_fullscreen)
#         self.fullscreen_btn.pack(side=tk.LEFT, padx=5)

#         self.status_label = tk.Label(self.root, text="No Image Loaded", fg="white", bg="#2E3440")
#         self.status_label.pack(fill=tk.X)

#         # Bind keys
#         self.root.bind("<Right>", lambda e: self.show_next())
#         self.root.bind("<Left>", lambda e: self.show_prev())
#         self.root.bind("<Escape>", lambda e: self.exit_fullscreen())
#         self.root.bind("<f>", lambda e: self.toggle_fullscreen())
#         self.root.bind("<space>", lambda e: self.toggle_slideshow())
#         self.root.bind("<plus>", lambda e: self.zoom_in())
#         self.root.bind("<minus>", lambda e: self.zoom_out())

#     def open_folder(self):
#         folder = filedialog.askdirectory()
#         if folder:
#             self.image_files = [os.path.join(folder, f) for f in os.listdir(folder)
#                                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
#             self.image_files.sort()
#             self.current_index = 0
#             if self.image_files:
#                 self.show_image()
#             else:
#                 messagebox.showwarning("No Images", "No supported image files found in the selected folder.")

#     def show_image(self):
#         image_path = self.image_files[self.current_index]
#         image = Image.open(image_path)

#         if self.fit_to_window:
#             width = self.canvas.winfo_width()
#             height = self.canvas.winfo_height()
#             image.thumbnail((width, height), Image.LANCZOS)
#         else:
#             image = image.resize((int(image.width * self.zoom_factor), int(image.height * self.zoom_factor)))

#         self.tk_img = ImageTk.PhotoImage(image)
#         self.canvas.delete("all")
#         self.canvas.create_image(self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2,
#                                  anchor=tk.CENTER, image=self.tk_img)

#         filename = os.path.basename(image_path)
#         self.status_label.config(text=f"{filename} ({self.current_index + 1} of {len(self.image_files)})")

#     def show_next(self):
#         if self.image_files:
#             self.current_index = (self.current_index + 1) % len(self.image_files)
#             self.show_image()

#     def show_prev(self):
#         if self.image_files:
#             self.current_index = (self.current_index - 1) % len(self.image_files)
#             self.show_image()

#     def zoom_in(self):
#         self.zoom_factor *= 1.1
#         self.fit_to_window = False
#         self.show_image()

#     def zoom_out(self):
#         self.zoom_factor /= 1.1
#         self.fit_to_window = False
#         self.show_image()

#     def toggle_fit(self):
#         self.fit_to_window = not self.fit_to_window
#         self.show_image()

#     def toggle_slideshow(self):
#         self.slideshow_running = not self.slideshow_running
#         if self.slideshow_running:
#             self.slideshow_btn.config(text="⏸️ Pause")
#             self.run_slideshow()
#         else:
#             self.slideshow_btn.config(text="▶️ Slideshow")

#     def run_slideshow(self):
#         if self.slideshow_running and self.image_files:
#             self.show_next()
#             self.root.after(2000, self.run_slideshow)

#     def toggle_fullscreen(self):
#         self.fullscreen = not self.fullscreen
#         self.root.attributes("-fullscreen", self.fullscreen)

#     def exit_fullscreen(self):
#         self.fullscreen = False
#         self.root.attributes("-fullscreen", False)


# if __name__ == "__main__":
#     root = tk.Tk()
#     app = ImageViewerApp(root)
#     root.mainloop()


# import os
# import tkinter as tk
# from tkinter import filedialog, messagebox
# from PIL import Image, ImageTk

# class ImageViewerApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Image Viewer")
#         self.root.geometry("800x600")
#         self.root.configure(bg="#2c3e50")

#         self.images = []
#         self.index = 0

#         self.setup_ui()

#     def setup_ui(self):
#         # Header
#         title = tk.Label(self.root, text="📸 Desktop Image Viewer", font=("Helvetica", 20, "bold"),
#                          bg="#2c3e50", fg="#ecf0f1")
#         title.pack(pady=10)

#         # Image display area
#         self.canvas = tk.Canvas(self.root, bg="white", width=700, height=400, bd=0, highlightthickness=0)
#         self.canvas.pack(pady=10)

#         # Image Info
#         self.info_label = tk.Label(self.root, text="", font=("Helvetica", 12), bg="#2c3e50", fg="#bdc3c7")
#         self.info_label.pack()

#         # Navigation buttons
#         btn_frame = tk.Frame(self.root, bg="#2c3e50")
#         btn_frame.pack(pady=10)

#         tk.Button(btn_frame, text="⏮ Prev", width=10, command=self.show_prev, bg="#34495e", fg="white").grid(row=0, column=0, padx=10)
#         tk.Button(btn_frame, text="Open Folder 📂", width=20, command=self.load_images, bg="#16a085", fg="white").grid(row=0, column=1, padx=10)
#         tk.Button(btn_frame, text="Next ⏭", width=10, command=self.show_next, bg="#34495e", fg="white").grid(row=0, column=2, padx=10)

#     def load_images(self):
#         folder_selected = filedialog.askdirectory()
#         if not folder_selected:
#             return

#         supported_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
#         self.images = [os.path.join(folder_selected, file)
#                        for file in os.listdir(folder_selected)
#                        if file.lower().endswith(supported_exts)]

#         if not self.images:
#             messagebox.showwarning("No Images", "No image files found in the selected folder.")
#             return

#         self.index = 0
#         self.display_image()

#     def display_image(self):
#         if not self.images:
#             return

#         image_path = self.images[self.index]
#         try:
#             img = Image.open(image_path)
#             img.thumbnail((700, 400))
#             self.tk_img = ImageTk.PhotoImage(img)
#             self.canvas.delete("all")
#             self.canvas.create_image(350, 200, image=self.tk_img)
#             filename = os.path.basename(image_path)
#             self.info_label.config(text=f"{filename} - {img.width}x{img.height}")
#         except Exception as e:
#             messagebox.showerror("Error", f"Cannot open image: {e}")

#     def show_prev(self):
#         if self.images:
#             self.index = (self.index - 1) % len(self.images)
#             self.display_image()

#     def show_next(self):
#         if self.images:
#             self.index = (self.index + 1) % len(self.images)
#             self.display_image()

# # Run the app
# if __name__ == "__main__":
#     root = tk.Tk()
#     app = ImageViewerApp(root)
#     root.mainloop()

#-----------------------------------------------------------------------------------------

# import tkinter as tk
# from tkinter import messagebox
# import os

# TASKS_FILE = "tasks.txt"

# class ToDoApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("📝 To-Do List Manager")
#         self.root.geometry("400x500")
#         self.root.resizable(False, False)
#         self.root.configure(bg="#f4f4f4")

#         self.task_var = tk.StringVar()

#         self.create_widgets()
#         self.load_tasks()

#     def create_widgets(self):
#         # Title
#         tk.Label(self.root, text="To-Do List", font=("Arial", 18, "bold"), bg="#f4f4f4").pack(pady=10)

#         # Entry + Add Button
#         frame = tk.Frame(self.root, bg="#f4f4f4")
#         frame.pack(pady=5)
#         entry = tk.Entry(frame, textvariable=self.task_var, width=25, font=("Arial", 12))
#         entry.pack(side=tk.LEFT, padx=5)
#         tk.Button(frame, text="Add", command=self.add_task, bg="#27ae60", fg="white", font=("Arial", 10, "bold")).pack(side=tk.LEFT)

#         # Task display (scrollable)
#         self.canvas = tk.Canvas(self.root, borderwidth=0, bg="#f4f4f4", height=350)
#         self.task_frame = tk.Frame(self.canvas, bg="#f4f4f4")
#         self.v_scroll = tk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
#         self.canvas.configure(yscrollcommand=self.v_scroll.set)

#         self.v_scroll.pack(side="right", fill="y")
#         self.canvas.pack(side="left", fill="both", expand=True)
#         self.canvas.create_window((0, 0), window=self.task_frame, anchor="nw")

#         self.task_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

#         # Save Button
#         tk.Button(self.root, text="Save Tasks", command=self.save_tasks, bg="#2980b9", fg="white", font=("Arial", 10, "bold")).pack(pady=10)

#     def add_task(self):
#         task_text = self.task_var.get().strip()
#         if not task_text:
#             return

#         self.display_task(task_text)
#         self.task_var.set("")

#     def display_task(self, text, checked=False):
#         frame = tk.Frame(self.task_frame, bg="#ecf0f1", pady=2)
#         frame.pack(fill="x", padx=5, pady=2)

#         var = tk.BooleanVar(value=checked)
#         check = tk.Checkbutton(frame, variable=var, bg="#ecf0f1")
#         check.pack(side=tk.LEFT)

#         label = tk.Label(frame, text=text, bg="#ecf0f1", font=("Arial", 11))
#         label.pack(side=tk.LEFT, padx=5)

#         del_btn = tk.Button(frame, text="🗑", command=lambda: self.remove_task(frame), bg="#c0392b", fg="white")
#         del_btn.pack(side=tk.RIGHT, padx=5)

#     def remove_task(self, frame):
#         frame.destroy()

#     def save_tasks(self):
#         tasks = []
#         for child in self.task_frame.winfo_children():
#             widgets = child.winfo_children()
#             if len(widgets) >= 2:
#                 text = widgets[1].cget("text")
#                 tasks.append(text)

#         with open(TASKS_FILE, "w", encoding="utf-8") as f:
#             for task in tasks:
#                 f.write(task + "\n")

#         messagebox.showinfo("Saved", "Tasks saved successfully!")

#     def load_tasks(self):
#         if os.path.exists(TASKS_FILE):
#             with open(TASKS_FILE, "r", encoding="utf-8") as f:
#                 for line in f.readlines():
#                     self.display_task(line.strip())


# # Run the app
# if __name__ == "__main__":
#     root = tk.Tk()
#     app = ToDoApp(root)
#     root.mainloop()



# import tkinter as tk
# from tkinter import messagebox, filedialog
# import os

# class TodoApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Elegant To-Do List Manager")
#         self.root.geometry("500x500")
#         self.root.configure(bg="#f5f5f5")

#         self.tasks = []

#         # Title
#         title = tk.Label(root, text="To-Do List Manager", font=("Helvetica", 18, "bold"), bg="#f5f5f5", fg="#2e3f4f")
#         title.pack(pady=10)

#         # Entry
#         self.entry = tk.Entry(root, font=("Arial", 14), width=30)
#         self.entry.pack(pady=10)

#         # Add Task Button
#         add_btn = tk.Button(root, text="Add Task", width=15, bg="#4caf50", fg="white", font=("Arial", 12), command=self.add_task)
#         add_btn.pack(pady=5)

#         # Listbox
#         self.listbox = tk.Listbox(root, font=("Arial", 13), width=45, height=10, selectbackground="#ff5722")
#         self.listbox.pack(pady=10)

#         # Buttons
#         btn_frame = tk.Frame(root, bg="#f5f5f5")
#         btn_frame.pack()

#         del_btn = tk.Button(btn_frame, text="Delete Selected", width=15, bg="#f44336", fg="white", command=self.delete_task)
#         del_btn.grid(row=0, column=0, padx=5)

#         save_btn = tk.Button(btn_frame, text="Save to File", width=15, bg="#2196f3", fg="white", command=self.save_tasks)
#         save_btn.grid(row=0, column=1, padx=5)

#         load_btn = tk.Button(btn_frame, text="Load from File", width=15, bg="#9c27b0", fg="white", command=self.load_tasks)
#         load_btn.grid(row=0, column=2, padx=5)

#         # Footer
#         footer = tk.Label(root, text="Made with ❤ by ChatGPT", font=("Arial", 9), bg="#f5f5f5", fg="gray")
#         footer.pack(side="bottom", pady=5)

#     def add_task(self):
#         task = self.entry.get().strip()
#         if task:
#             self.tasks.append(task)
#             self.update_listbox()
#             self.entry.delete(0, tk.END)
#         else:
#             messagebox.showwarning("Input Error", "Please enter a task.")

#     def delete_task(self):
#         selected = self.listbox.curselection()
#         if selected:
#             index = selected[0]
#             del self.tasks[index]
#             self.update_listbox()
#         else:
#             messagebox.showwarning("Selection Error", "Please select a task to delete.")

#     def save_tasks(self):
#         file_path = filedialog.asksaveasfilename(defaultextension=".txt",
#                                                  filetypes=[("Text files", "*.txt")])
#         if file_path:
#             try:
#                 with open(file_path, "w") as f:
#                     for task in self.tasks:
#                         f.write(task + "\n")
#                 messagebox.showinfo("Saved", "Tasks saved successfully!")
#             except Exception as e:
#                 messagebox.showerror("Error", f"Error saving file:\n{e}")

#     def load_tasks(self):
#         file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
#         if file_path:
#             try:
#                 with open(file_path, "r") as f:
#                     self.tasks = [line.strip() for line in f.readlines()]
#                 self.update_listbox()
#                 messagebox.showinfo("Loaded", "Tasks loaded successfully!")
#             except Exception as e:
#                 messagebox.showerror("Error", f"Error loading file:\n{e}")

#     def update_listbox(self):
#         self.listbox.delete(0, tk.END)
#         for task in self.tasks:
#             self.listbox.insert(tk.END, task)


# if __name__ == "__main__":
#     root = tk.Tk()
#     app = TodoApp(root)
#     root.mainloop()


#-------------------------------------------------------------------------