import os

# Đường dẫn đến thư mục 'labels' và 'images'
labels_folder = 'custom_dataset/labels'
images_folder = 'custom_dataset/images'

def remove_no_detection_files(labels_folder, images_folder):
    # Lấy danh sách tất cả các tên file trong thư mục 'labels' và 'images'
    labels_files = set(os.path.splitext(file)[0] for file in os.listdir(labels_folder) if file.endswith('.txt'))
    images_files = set(os.path.splitext(file)[0] for file in os.listdir(images_folder) if file.endswith('.jpg'))

    # Tìm các file trong 'images' không tồn tại trong 'labels'
    non_detected_files = images_files - labels_files

    # Xóa các file không được phát hiện
    for file in non_detected_files:
        os.remove(os.path.join(images_folder, file + '.jpg'))

remove_no_detection_files(labels_folder, images_folder)