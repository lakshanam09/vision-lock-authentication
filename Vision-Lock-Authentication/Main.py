import tkinter as tk
from tkinter import messagebox
import os
import json
import hashlib

from Register import register_face
from Train_model import train_model

# ---------------- Database ---------------- #

user_db = "database.json"

if not os.path.exists(user_db):
    with open(user_db, "w") as f:
        json.dump({}, f)

# ---------------- GUI ---------------- #

root = tk.Tk()
root.title("Face Authentication System")
root.geometry("400x400")


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# ---------------- Registration ---------------- #

def register():
    user_id = entry_id.get()
    username = entry_name.get()
    password = entry_password.get()

    if not user_id or not username or not password:
        messagebox.showerror("Error", "All fields are required!")
        return

    with open(user_db, "r") as f:
        users = json.load(f)

    if user_id in users:
        messagebox.showerror("Error", "User ID already exists!")
        return

    users[user_id] = {
        "name": username,
        "password": hash_password(password)
    }

    with open(user_db, "w") as f:
        json.dump(users, f, indent=4)

    messagebox.showinfo(
        "Info",
        f"User '{username}' registered.\nCapture face now."
    )

    register_face(user_id)
    train_model()

    messagebox.showinfo(
        "Success",
        "Face Registered & Model Trained Successfully!"
    )


# ---------------- Login ---------------- #

def login():
    user_id = entry_login_id.get()
    password = entry_login_password.get()

    if not user_id or not password:
        messagebox.showerror("Error", "All fields required!")
        return

    with open(user_db, "r") as f:
        users = json.load(f)

    if user_id not in users:
        messagebox.showerror("Error", "Invalid User ID!")
        return

    if users[user_id]["password"] != hash_password(password):
        messagebox.showerror("Error", "Invalid Password!")
        return

    # 🔥 Lazy import (THIS FIXES YOUR CRASH)
    try:
        from Recognizer import recognize_face, model_loaded
    except RuntimeError as e:
        messagebox.showerror(
            "Model Error",
            "Dlib model not found!\n\n"
            "Download:\n"
            "shape_predictor_68_face_landmarks.dat\n"
            "and place it in your project folder."
        )
        return
    except Exception as e:
        messagebox.showerror("Error", str(e))
        return

    if not model_loaded:
        messagebox.showerror(
            "Error",
            "Face recognition model not trained yet!"
        )
        return

    status, _ = recognize_face(user_id)

    if status == "success":
        messagebox.showinfo(
            "Welcome",
            f"Welcome {users[user_id]['name']}!"
        )
    elif status == "idle":
        messagebox.showwarning("Warning", "User is idle!")
    else:
        messagebox.showerror("Error", "Face recognition failed!")


# ---------------- UI Layout ---------------- #

reg_frame = tk.LabelFrame(root, text="Registration", padx=10, pady=10)
reg_frame.pack(pady=10)

tk.Label(reg_frame, text="User ID:").grid(row=0, column=0)
entry_id = tk.Entry(reg_frame)
entry_id.grid(row=0, column=1)

tk.Label(reg_frame, text="Username:").grid(row=1, column=0)
entry_name = tk.Entry(reg_frame)
entry_name.grid(row=1, column=1)

tk.Label(reg_frame, text="Password:").grid(row=2, column=0)
entry_password = tk.Entry(reg_frame, show="*")
entry_password.grid(row=2, column=1)

tk.Button(reg_frame, text="Register", command=register)\
    .grid(row=3, columnspan=2, pady=10)


login_frame = tk.LabelFrame(root, text="Login", padx=10, pady=10)
login_frame.pack(pady=10)

tk.Label(login_frame, text="User ID:").grid(row=0, column=0)
entry_login_id = tk.Entry(login_frame)
entry_login_id.grid(row=0, column=1)

tk.Label(login_frame, text="Password:").grid(row=1, column=0)
entry_login_password = tk.Entry(login_frame, show="*")
entry_login_password.grid(row=1, column=1)

tk.Button(login_frame, text="Login", command=login)\
    .grid(row=2, columnspan=2, pady=10)

root.mainloop()
