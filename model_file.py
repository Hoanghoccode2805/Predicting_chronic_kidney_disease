import pandas as pd
df = pd.read_excel('D:\Projet 1\data_input.xlsx')
print(df.shape)        # Số dòng, cột
print(df.columns)      # Danh sách tên cột
print(df.info())       # Kiểu dữ liệu và số lượng dữ liệu không null
print(df.describe())   # Thống kê cơ bản các cột dạng số
# Thay thế dấu '?' bằng NaN
df.replace('?', pd.NA, inplace=True)

# Ép kiểu về số cho các cột số
cols_num = ['age', 'blood pressure', 'blood glucose random', 'blood urea', 'serum creatinine', 
            'sodium', 'potassium', 'hemoglobin', 'packed cell volume', 
            'white blood cell count', 'red blood cell count']
for col in cols_num:
    df[col] = pd.to_numeric(df[col], errors='coerce')

#print(df.info())       # Để kiểm tra lại kiểu dữ liệu và missing values
#print(df.describe())   # Xem thống kê lại sau khi ép kiểu

# Xử lý missing values
# Với cột số
for col in df.select_dtypes(include=['float', 'int']).columns:
    df[col] = df[col].fillna(df[col].median())

# Với cột phân loại
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna('unknown')
#print(df.info())      # Kiểm tra lại sau khi xử lý missing values
df.to_excel('D:\Projet 1\data_output_sau_xu_li.xlsx', index=False)  # Lưu kết quả vào file mới

#label encoding cho các cột phân loại
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col].astype(str))  # ép kiểu về chuỗi

# Tách dữ liệu thành X và y
X = df.drop('class', axis=1)  # X là tất cả các cột trừ cột 'class'
y = df['class']  # y là cột 'class'

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# Dùng random_forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train,y_train)

# Đánh giá lại mô hình
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
y_pred = model.predict(X_test)

print(f"Độ chính xác: {accuracy_score(y_test,y_pred)}")
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

import joblib
joblib.dump(model,'model.pkl')