import json

# Mở file JSON
with open("file.json", "r") as f:
    data = json.load(f)

# Truy cập dữ liệu
print(data["name"]) # In ra tên
print(data["age"]) # In ra tuổi

# Lặp qua dữ liệu
for item in data["items"]:
    print(item["name"], item["price"])