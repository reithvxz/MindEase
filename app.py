import os
from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load model yang sudah dilatih
model_path = os.path.join(app.root_path, 'static', 'model', 'stres_model.pkl')
model = joblib.load(model_path)

# Inisialisasi LabelEncoder (sama seperti di pelatihan model)
label_encoders = {}

# Fungsi untuk encode input pengguna
def encode_user_input(input_data):
    encoded_data = []
    for idx, value in enumerate(input_data):
        column_name = f"col_{idx}"  # Simpan encoder berdasarkan kolom
        if column_name not in label_encoders:
            label_encoders[column_name] = LabelEncoder()
            label_encoders[column_name].fit([value])  # Fit hanya sekali
        if value not in label_encoders[column_name].classes_:
            label_encoders[column_name].classes_ = np.append(label_encoders[column_name].classes_, value)
        encoded_data.append(label_encoders[column_name].transform([value])[0])
    return encoded_data

@app.route('/')
def home():
    return render_template('index.html')  # Halaman utama untuk input pengguna

@app.route('/page1')
def page1():
    return render_template('page1.html')  # Render halaman page1.html

@app.route('/page2')
def page2():
    return render_template('page2.html')  # Render halaman page2.html

@app.route('/page3')
def page3():
    return render_template('page3.html')  # Render halaman page3.html

@app.route('/tes')
def tes():
    return render_template('tes.html')

@app.route('/check_stress', methods=['POST'])
def check_stress():
    try:
        # Ambil data dari formulir
        gender = request.form['gender']
        age = float(request.form['age'])
        weight = float(request.form['weight'])
        sleep_hours = float(request.form['sleep_hours'])
        satisfaction_with_sleep = int(request.form['satisfaction_with_sleep'])
        sudden_events = request.form['sudden_events']
        control_over_life = request.form['control_over_life']
        feeling_anxious = request.form['feeling_anxious']
        managing_personal_issues = request.form['managing_personal_issues']
        expectations_met = request.form['expectations_met']
        overwhelmed = request.form['overwhelmed']
        too_many_problems = request.form['too_many_problems']
        quick_temper = request.form['quick_temper']
        worrying_about_future = request.form['worrying_about_future']
        lack_of_support = request.form['lack_of_support']

        # Encode nilai kategorikal
        encoded_data = encode_user_input([
            gender, sudden_events, control_over_life, feeling_anxious,
            managing_personal_issues, expectations_met, overwhelmed,
            too_many_problems, quick_temper, worrying_about_future, lack_of_support
        ])

        # Gabungkan data input dengan data numerik lainnya
        features = np.array([[
            encoded_data[0], age, weight, sleep_hours, satisfaction_with_sleep
        ] + encoded_data[1:]])

        # Debug input features
        print(f"Encoded Features: {features}")

        # Melakukan prediksi menggunakan model
        prediction = model.predict(features)
        probabilities = model.predict_proba(features)

        # Debug untuk memastikan prediksi berhasil
        print(f"Prediction Array: {prediction}")
        print(f"Prediction Probabilities: {probabilities}")

        # Konversi hasil prediksi ke kategori
        stress_categories = {0: 'Stres Tinggi', 1: 'Stres Sedang', 2: 'Stres Rendah'}
        result = stress_categories.get(prediction[0], "Hasil tidak diketahui")

        # Tambahkan saran berdasarkan klasifikasi
        advice_map = {
            'Stres Rendah': "Anda tampaknya berada dalam kondisi yang baik. Cobalah untuk memberi diri Anda waktu untuk beristirahat dan melakukan aktivitas yang Anda nikmati, seperti berjalan-jalan, berolahraga ringan, atau berbicara dengan teman dekat.",
            'Stres Sedang': "Anda mungkin merasa sedikit kewalahan. Cobalah untuk mengatur waktu lebih baik, berbicara dengan orang terpercaya, atau melakukan aktivitas relaksasi seperti meditasi, dan luangkan waktu untuk berolahraga secara teratur. Manajemen waktu yang lebih baik juga bisa membantu Anda merasa lebih terkendali dalam menghadapi tuntutan yang ada.",
            'Stres Tinggi': "Anda mungkin sedang mengalami tekanan yang berat. Sangat penting untuk segera mencari bantuan profesional untuk menangani stres ini, baik dengan berbicara kepada seorang psikolog, konselor, atau terapis. Jangan ragu untuk mencari dukungan dari teman atau keluarga, dan pastikan Anda memberi perhatian pada diri sendiri dengan memprioritaskan kesehatan mental. Luangkan waktu untuk aktivitas yang menenangkan dan coba praktikkan teknik relaksasi yang lebih intensif. Jangan biarkan stres ini berlarut-larut tanpa penanganan yang tepat."
        }
        advice = advice_map.get(result, "Tidak ada saran tersedia untuk klasifikasi ini.")

        # Debug probabilitas untuk setiap kelas
        for idx, prob in enumerate(probabilities[0]):
            print(f"Probabilitas {stress_categories.get(idx, idx)}: {prob:.2f}")

        # Langsung menampilkan hasil prediksi dan saran
        return render_template('result.html', result=result, advice=advice, probabilities=probabilities[0])

    except Exception as e:
        print(f"Error occurred: {e}")
        return render_template('result.html', error_message="Terjadi kesalahan dalam memproses data.")

if __name__ == "__main__":
    app.run(debug=True)
