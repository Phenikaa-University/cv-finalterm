import os
import re
import matplotlib.pyplot as plt

# Đường dẫn đến hai thư mục images và labels
images_folder = "dataset_yolov8x/images"
labels_folder = "dataset_yolov8x/labels"

# Lấy danh sách tệp tin trong hai thư mục
image_files = set(os.listdir(images_folder))
label_files = set(os.listdir(labels_folder))

# Tìm những tệp tin có tên khác nhau giữa hai thư mục
different_files = image_files.symmetric_difference(label_files)

print(len(different_files))
# In ra danh sách các tệp tin có tên khác nhau
class_counts = {}
for file_name in different_files:
    if file_name.endswith(".jpg"):
        label_name = re.search(r'_([A-Z]+-\d+)_', file_name)
        if label_name:
            label_name = label_name.group(1)
            class_counts[label_name] = class_counts.get(label_name, {"jpg": 0, "txt": 0})
            class_counts[label_name]["jpg"] += 1
    elif file_name.endswith(".txt"):
        label_name = re.search(r'_([A-Z]+-\d+)_', file_name)
        if label_name:
            label_name = label_name.group(1)
            class_counts[label_name] = class_counts.get(label_name, {"jpg": 0, "txt": 0})
            class_counts[label_name]["txt"] += 1

# Biểu đồ cột thể hiện số lượng .jpg và .txt cho mỗi nhãn
labels = class_counts.keys()
jpg_counts = [counts["jpg"] for counts in class_counts.values()]
txt_counts = [counts["txt"] for counts in class_counts.values()]

plt.figure(figsize=(12, 6))
plt.bar(labels, jpg_counts, label=".jpg")
plt.bar(labels, txt_counts, label=".txt", alpha=0.5)
plt.xlabel("Nhãn")
plt.ylabel("Số lượng")
plt.title("Thống kê số lượng .jpg và .txt cho mỗi nhãn")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# Lưu biểu đồ vào một tệp hình ảnh
plt.savefig("visualize/thongke.png")

# # Hiển thị biểu đồ
# plt.show()
# In ra thông tin thống kê
for label_name, counts in class_counts.items():
    print(f"Nhãn {label_name}: Số lượng .jpg = {counts['jpg']}, Số lượng .txt = {counts['txt']}")