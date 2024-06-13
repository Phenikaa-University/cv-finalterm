import matplotlib.pyplot as plt
import pandas as pd
import os


df = pd.read_excel("dataset/data_refine.xlsx")
# Tính số lượng mỗi nhãn ban đầu trong cột 'label_detail'
original_label_counts = df['label_detail'].value_counts()
image_folder = "custom_dataset/images"

# Tính số lượng nhãn sau khi được prelabel
image_files = os.listdir(image_folder)
prelabel_counts = {}

for image_file in image_files:
    label = df[df['img_url'].str.contains(image_file)]['label_detail'].values[0]
    if label in prelabel_counts:
        prelabel_counts[label] += 1
    else:
        prelabel_counts[label] = 1

# Chuyển số lượng nhãn sau khi được prelabel thành DataFrame
prelabel_counts_df = pd.Series(prelabel_counts)

# Kết hợp hai Series thành một DataFrame
combined_counts_df = pd.DataFrame({
    'Ban đầu': original_label_counts,
    'Sau khi prelabel': prelabel_counts_df
})

# Vẽ biểu đồ cột
plt.figure(figsize=(12, 6))
combined_counts_df.plot(kind='bar')
plt.xlabel("Nhãn")
plt.ylabel("Số lượng")
plt.title("So sánh số lượng nhãn ban đầu và sau khi prelabel")
plt.xticks(rotation=360)
plt.tight_layout()
plt.show()
plt.savefig('visualize/compare_label.png')