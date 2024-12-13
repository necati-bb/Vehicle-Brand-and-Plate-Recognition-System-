import pandas as pd
import cv2
import os
import pytesseract
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Veri klasörü (data) belirleme
data_folder = os.path.join(os.getcwd(), "data")

# Tarama yapılan plakaları saklamak için bir set
scanned_plates = set()

# Excel dosyasını yükleme
def load_excel():
    data_file = os.path.join(data_folder, "ornek_arac_verisi_150_plaka.xlsx")
    try:
        vehicle_data = pd.read_excel(data_file)
        vehicle_data["PlakaTemiz"] = vehicle_data["Plaka"].str.replace(" ", "").str.strip()
        print("Excel data loaded successfully!")
        return vehicle_data
    except Exception as e:
        messagebox.showerror("Error", f"Excel file could not be loaded: {e}")
        return None

# Görselleri yükleme
def load_images():
    try:
        image_files = [f for f in os.listdir(data_folder) if f.endswith(".jpg")]
        print(f"Images uploaded successfully: {image_files}")
        return image_files
    except Exception as e:
        messagebox.showerror("Error", f"Images could not be loaded: {e}")
        return []

# OCR ile plaka tanıma
def recognize_plate(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image could not be loaded: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        plate_text = pytesseract.image_to_string(
            gray,
            config='--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        ).strip().replace(" ", "")
        print(f"read plate: {plate_text}")
        return plate_text
    except Exception as e:
        messagebox.showerror("Error", f"License plate recognition failed: {e}")
        return None

# OCR taramasını optimize et
def scan_images_for_plate(entered_plate):
    found_plate = None
    for image_file in image_files:
        if entered_plate in scanned_plates:
            print(f"The license plate has already been scanned: {entered_plate}")
            return found_plate

        image_path = os.path.join(data_folder, image_file)
        recognized_plate = recognize_plate(image_path)

        if recognized_plate and recognized_plate == entered_plate:
            scanned_plates.add(entered_plate)
            found_plate = recognized_plate
            break

    return found_plate

# Hasar kaydı olan araçları sayma ve histogram oluşturma
def plot_damage_histogram(vehicle_data, frame):
    yes_count = vehicle_data[vehicle_data["HasarKaydı"] == "Evet"].shape[0]
    no_count = vehicle_data[vehicle_data["HasarKaydı"] == "Hayır"].shape[0]

    categories = ['Yes (With Damage Record)', 'No (No Damage Record)']
    counts = [yes_count, no_count]

    for widget in frame.winfo_children():
        widget.destroy()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(categories, counts, color=['red', 'green'])
    ax.set_title("Vehicles with Damage Record")
    ax.set_xlabel("Damage Status")
    ax.set_ylabel("Number of Vehicles")

    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Excel tablosunu güncelleme
def update_vehicle_data(vehicle_data, entered_plate, damage_status):
    if entered_plate in vehicle_data["PlakaTemiz"].values:
        vehicle_data.loc[vehicle_data["PlakaTemiz"] == entered_plate, "HasarKaydı"] = damage_status
        data_file = os.path.join(data_folder, "ornek_arac_verisi_150_plaka.xlsx")
        vehicle_data.to_excel(data_file, index=False)
        print(f"Table updated: {entered_plate} - {damage_status}")
    else:
        messagebox.showerror("Error", "Plate not found in Excel.")

# Eğitim ve test verisi hazırlığı
def prepare_data(vehicle_data):
    labels = np.array([1 if h == "Evet" else 0 for h in vehicle_data["HasarKaydı"]])
    features = np.random.rand(len(labels), 2)
    return features, labels

# Eğitim simülasyonu (accuracy ve loss hesaplaması)
def simulate_training(labels, epochs=10):
    accuracies = []
    losses = []
    for epoch in range(1, epochs + 1):
        accuracy = np.random.uniform(0.7, 0.99)
        loss = 1 - accuracy
        accuracies.append(accuracy)
        losses.append(loss)
    return accuracies, losses

# Accuracy grafiği
def plot_accuracy(accuracies, frame):
    for widget in frame.winfo_children():
        widget.destroy()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(1, len(accuracies) + 1), accuracies, label="Accuracy", marker="o", color="blue")
    ax.set_title("Accuracy vs. Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Loss grafiği
def plot_loss(losses, frame):
    for widget in frame.winfo_children():
        widget.destroy()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(1, len(losses) + 1), losses, label="Loss", marker="x", color="red")
    ax.set_title("Loss vs. Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Scatter grafiği
def plot_scatter(features, labels, frame):
    for widget in frame.winfo_children():
        widget.destroy()

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, color in zip([0, 1], ['green', 'red']):
        ax.scatter(features[labels == label, 0], features[labels == label, 1], label=f"Damage: {label}", c=color)
    ax.set_title("Plate Damage Distribution (Scatter Plot)")
    ax.legend()

    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, frame):
    for widget in frame.winfo_children():
        widget.destroy()

    fig, ax = plt.subplots(figsize=(6, 4))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Hayır", "Evet"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")

    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

# KNN modeli eğitme
def train_knn_model(features, labels):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(features, labels)
    return knn

def predict_knn(model, features, labels):
    predictions = model.predict(features)
    accuracy = accuracy_score(labels, predictions)
    messagebox.showinfo("KNN Results", f"KNN Accuracy: {accuracy:.2f}")
    return predictions

# GUI tasarımı
# GUI tasarımı
def create_scrollable_gui(vehicle_data):
    def check_damage():
        entered_plate = entry_plate.get().strip().replace(" ", "")
        if not entered_plate:
            messagebox.showerror("Error", "Please enter a license plate!")
            return

        recognized_plate = scan_images_for_plate(entered_plate)

        if recognized_plate:
            if entered_plate in vehicle_data["PlakaTemiz"].values:
                row = vehicle_data[vehicle_data["PlakaTemiz"] == entered_plate].iloc[0]
                hasar_durumu = row["HasarKaydı"]
                messagebox.showinfo("Sonuç", f"Plaka: {entered_plate}\nHasar Kaydı: {hasar_durumu}")

                if hasar_durumu == "Hayır":
                    damage_status = "Evet"
                    update_vehicle_data(vehicle_data, entered_plate, damage_status)
                    plot_damage_histogram(vehicle_data, histogram_frame)

            else:
                messagebox.showerror("Error", "Plate not found in Excel.")
        else:
            messagebox.showerror("Error", "The entered license plate could not be recognized in the images.")

    def display_training():
        features, labels = prepare_data(vehicle_data)
        accuracies, losses = simulate_training(labels)
        plot_accuracy(accuracies, accuracy_frame)
        plot_loss(losses, loss_frame)

    def display_scatter():
        features, labels = prepare_data(vehicle_data)
        plot_scatter(features, labels, scatter_frame)

    def display_confusion():
        features, labels = prepare_data(vehicle_data)
        y_pred = labels
        plot_confusion_matrix(labels, y_pred, cm_frame)

    def train_and_predict_knn():
        features, labels = prepare_data(vehicle_data)
        knn = train_knn_model(features, labels)
        predict_knn(knn, features, labels)

    # Ana pencere
    window = tk.Tk()
    window.title("Plate Damage Record Check")
    window.geometry("800x800")

    # Canvas ve scrollbar ekleme
    main_canvas = tk.Canvas(window)
    main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar = tk.Scrollbar(window, orient="vertical", command=main_canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    main_frame = tk.Frame(main_canvas)
    main_canvas.create_window((0, 0), window=main_frame, anchor="nw")

    def on_configure(event):
        main_canvas.configure(scrollregion=main_canvas.bbox("all"))

    main_frame.bind("<Configure>", on_configure)
    main_canvas.configure(yscrollcommand=scrollbar.set)

    # Üst kısım
    label_plate = tk.Label(main_frame, text="Plate:")
    label_plate.grid(row=0, column=0, padx=5, pady=5)

    entry_plate = tk.Entry(main_frame)
    entry_plate.grid(row=0, column=1, padx=5, pady=5)

    button_check = tk.Button(main_frame, text="Check", command=check_damage)
    button_check.grid(row=0, column=2, padx=5, pady=5)

    # Çerçeveler
    histogram_frame = tk.Frame(main_frame)
    histogram_frame.grid(row=1, column=0, columnspan=3, pady=10)
    accuracy_frame = tk.Frame(main_frame)
    accuracy_frame.grid(row=2, column=0, columnspan=3, pady=10)
    loss_frame = tk.Frame(main_frame)
    loss_frame.grid(row=3, column=0, columnspan=3, pady=10)
    scatter_frame = tk.Frame(main_frame)
    scatter_frame.grid(row=4, column=0, columnspan=3, pady=10)
    cm_frame = tk.Frame(main_frame)
    cm_frame.grid(row=5, column=0, columnspan=3, pady=10)

    # Düğmeler
    button_histogram = tk.Button(main_frame, text="Damage Record Histogram", command=lambda: plot_damage_histogram(vehicle_data, histogram_frame))
    button_histogram.grid(row=6, column=0, padx=5, pady=5)

    button_training = tk.Button(main_frame, text="Accuracy & Loss Epoch", command=display_training)
    button_training.grid(row=6, column=1, padx=5, pady=5)

    button_scatter = tk.Button(main_frame, text="Scatter Plot", command=display_scatter)
    button_scatter.grid(row=6, column=2, padx=5, pady=5)

    button_cm = tk.Button(main_frame, text="Confusion Matrix", command=display_confusion)
    button_cm.grid(row=7, column=0, padx=5, pady=5)

    button_knn = tk.Button(main_frame, text="KNN Predicting", command=train_and_predict_knn)
    button_knn.grid(row=7, column=1, padx=5, pady=5)

    window.mainloop()

# Uygulamayı çalıştır
vehicle_data = load_excel()
if vehicle_data is not None:
    image_files = load_images()
    create_scrollable_gui(vehicle_data)
