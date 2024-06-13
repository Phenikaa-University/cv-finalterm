import os

# Đường dẫn đến thư mục chứa ảnh và nhãn của tập validation
images_val_folder = "custom_dataset/images/val"
labels_val_folder = "custom_dataset/labels/val"

# Lấy danh sách tên tệp ảnh và nhãn trong thư mục images/val
image_files = os.listdir(images_val_folder)
label_files = os.listdir(labels_val_folder)

# Chuyển danh sách tên tệp thành tập hợp (set) để kiểm tra trùng lặp
# Chỉ lấy phần tên của tệp (loại bỏ phần mở rộng)
image_set = set([os.path.splitext(file)[0] for file in image_files])
label_set = set([os.path.splitext(file)[0] for file in label_files])

# Tìm các tên tệp không trùng khớp bằng cách lấy phần giao của hai tập hợp
non_matching_files = image_set.symmetric_difference(label_set)

# In ra các tên tệp không trùng khớp
if non_matching_files:
    print("Các tên tệp không trùng khớp giữa images và labels:")
    for file in non_matching_files:
        print(file)
else:
    print("Tất cả các tên tệp đều trùng khớp giữa images và labels.")
