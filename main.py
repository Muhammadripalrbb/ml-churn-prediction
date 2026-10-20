import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# 1. Simulasi Pembuatan Dataset (Representasi Data Historis Pelanggan)
np.random.seed(42)
n_samples = 1000

data = {
    'tenure_months': np.random.randint(1, 60, size=n_samples),
    'avg_monthly_spend': np.random.uniform(20, 500, size=n_samples),
    'support_tickets_count': np.random.poisson(lam=2, size=n_samples),
    'last_login_days': np.random.randint(0, 45, size=n_samples)
}

df = pd.DataFrame(data)

# Simulasi Target Churn berdasarkan aturan logis + sedikit noise
# Pelanggan lebih rentan churn jika jarang login, sering komplain, dan masa berlangganannya singkat
churn_prob = (
    (df['last_login_days'] > 20).astype(int) * 0.4 +
    (df['support_tickets_count'] > 3).astype(int) * 0.3 +
    (df['tenure_months'] < 6).astype(int) * 0.2
)
df['churn'] = (churn_prob + np.random.uniform(0, 0.2, size=n_samples) > 0.5).astype(int)

print("--- 5 Baris Pertama Data Mentah (Raw Data) ---")
print(df.head(), "\n")

# 2. Data Preprocessing (Prapemrosesan Data)
# Menghapus duplikasi jika ada
df = df.drop_duplicates()

# Memisahkan Fitur (X) dan Target/Label (y)
X = df.drop(columns=['churn'])
y = df['churn']

# Membagi Data: 80% Training, 20% Testing (Sesuai materi perkuliahan)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisasi / Standarisasi Fitur Numerik agar berada dalam skala yang seragam
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Pemodelan (Training Model menggunakan Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 4. Evaluasi Model pada Unseen Data (Testing Set)
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Akurasi Model pada Test Set: {accuracy * 100:.2f}%\n")
print("Confusion Matrix:")
print(conf_matrix, "\n")
print("Classification Report:")
print(class_report)