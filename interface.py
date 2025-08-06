import tkinter as tk
from tkinter import ttk, messagebox
import joblib

# Model ve vektörleştiriciyi yükleme
try:
    model = joblib.load("logistic_regression_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    print("Model ve vektörleştirici başarıyla yüklendi.")
except FileNotFoundError as e:
    messagebox.showerror("Hata", f"Dosya bulunamadı: {e}")
    exit()
except Exception as e:
    messagebox.showerror("Hata", f"Model yükleme hatası: {e}")
    exit()

# Analiz fonksiyonu
def analyze_sentiment():
    user_input = entry.get()
    if not user_input.strip():
        messagebox.showwarning("Uyarı", "Lütfen bir yorum girin.")
        return

    try:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        result = "Olumlu" if prediction == 1 else "Olumsuz"
        result_label.config(text=f"Yorumun Analizi: {result}", foreground="green" if prediction == 1 else "red")
    except Exception as e:
        messagebox.showerror("Hata", f"Analiz sırasında bir hata oluştu: {e}")

# Arayüz sıfırlama fonksiyonu
def reset_interface():
    entry.delete(0, tk.END)
    result_label.config(text="Sonuç burada görünecek.", foreground="black")

# Tkinter arayüzü
root = tk.Tk()
root.title("Yorum Analiz Uygulaması")
root.geometry("500x300")
root.resizable(False, False)

# Başlık
title_label = tk.Label(root, text="Yorum Analiz Uygulaması", font=("Helvetica", 16, "bold"))
title_label.pack(pady=10)

# Kullanıcı girişi
frame = tk.Frame(root)
frame.pack(pady=20)

entry_label = tk.Label(frame, text="Yorumunuzu girin:", font=("Helvetica", 12))
entry_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

entry = tk.Entry(frame, width=50)
entry.grid(row=0, column=1, padx=5, pady=5)

# Sonuç etiketi
result_label = tk.Label(root, text="Sonuç burada görünecek.", font=("Helvetica", 12))
result_label.pack(pady=10)

# Butonlar
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

analyze_button = tk.Button(button_frame, text="Analiz Et", command=analyze_sentiment, bg="green", fg="white", font=("Helvetica", 10, "bold"))
analyze_button.grid(row=0, column=0, padx=10)

reset_button = tk.Button(button_frame, text="Sıfırla", command=reset_interface, bg="orange", fg="white", font=("Helvetica", 10, "bold"))
reset_button.grid(row=0, column=1, padx=10)

exit_button = tk.Button(button_frame, text="Çıkış", command=root.quit, bg="red", fg="white", font=("Helvetica", 10, "bold"))
exit_button.grid(row=0, column=2, padx=10)

# Main loop
root.mainloop()
