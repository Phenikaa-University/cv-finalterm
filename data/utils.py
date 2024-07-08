import cv2
import pandas as pd
import os
import random
import shutil

def pre_label(label_file: str, path: dict,  model):
    """
    Input parameters:
        label_file: str, path to the Excel file containing image URLs and labels
        path: dict, dictionary containing paths to input and label folders
        model: YOLO, YOLO model for pre-labeling
    Output:
        Return the pre-labeled images and labels
    """
    try:
        df = pd.read_excel(label_file)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return
    
    os.makedirs(path.get("input_folder"), exist_ok=True)
    os.makedirs(path.get("label_folder"), exist_ok=True)
    os.makedirs(path.get("outline_input_folder"), exist_ok=True)
    os.makedirs(path.get("outline_label_folder"), exist_ok=True)

    for index, row in df.iterrows():
        img_url = row['img_url']
        label_detail = row['label_detail']

        try:
            image_path = os.path.join(path.get("raw_folder"), os.path.basename(img_url))

            if os.path.exists(image_path):
                results = model(image_path, classes=53, save_txt=None)  # Ensure model is correctly defined

                if len(results[0].boxes.xywhn) == 0:
                    # Handle no object detected
                    txt_file = os.path.splitext(os.path.basename(img_url))[0] + '.txt'
                    txt_path = os.path.join(path.get("outline_label_folder"), txt_file)
                    img_dest_path = os.path.join(path.get("outline_input_folder"), os.path.basename(img_url))

                    os.rename(image_path, img_dest_path)

                    with open(txt_path, 'w') as f:
                        f.write(label_detail)
                else:
                    largest_box = max(results[0].boxes.xywhn, key=lambda x: x[3])

                    txt_file = os.path.splitext(os.path.basename(img_url))[0] + '.txt'
                    txt_path = os.path.join(path.get("label_folder"), txt_file)

                    x, y, w, h = largest_box[:4].tolist()

                    with open(txt_path, 'w') as file:
                        file.write(f"{label_detail} {x} {y} {w} {h}\n")
            else:
                print(f"Image not found: {img_url}")
        except Exception as e:
            print(f"Error processing {img_url}: {e}")
            
def refine_img(path: dict):
    """
    Input parameters:
        path: dict, dictionary containing paths to input and label folders
    Output:
        Return the cropped images
    """
    for label_file in os.listdir(path.get("label_folder")):
        if label_file.endswith(".txt"):
            with open(os.path.join(path.get("label_folder"), label_file), 'r') as file:
                content = file.readlines()
        
            if len(content) > 0:
                # Extract class_id, x, y, w, h from label file
                values = content[0].split()
                class_id = int(values[0])
                x, y, w, h = map(float, values[1:])

                image_path = os.path.join(path.get("raw_folder"), os.path.splitext(label_file)[0] + '.jpg')
                cropped_image_path = os.path.join(path.get("input_folder"), label_file.replace('.txt', '.jpg'))
                
                # Crop image
                image = cv2.imread(image_path)
                h_img, w_img, _ = image.shape
                x, y, w, h = int(x * w_img), int(y * h_img), int(w * w_img), int(h * h_img)
                x1, y1, x2, y2 = int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)
                cropped_image = image[y1:y2, x1:x2]
                
                # Save cropped image
                cv2.imwrite(cropped_image_path, cropped_image)
            

def split_data(path, split_ratio=0.8):
    """
    Input parameters:
        path: dict, dictionary containing paths to input and label folders
        split_ratio: float, ratio to split the data into training and testing
    Output:
        Return the split data
    """
    train_images_folder = os.path.join(path.get("input_folder"), "train")
    test_images_folder = os.path.join(path.get("input_folder"), "test")
    train_labels_folder = os.path.join(path.get("label_folder"), "train")
    test_labels_folder = os.path.join(path.get("label_folder"), "test")
    if not os.path.exists(train_images_folder) and not os.path.exists(test_images_folder) and not os.path.exists(train_labels_folder) and not os.path.exists(test_labels_folder):
        os.makedirs(train_images_folder)
        os.makedirs(test_images_folder)
        os.makedirs(train_labels_folder)
        os.makedirs(test_labels_folder)
        
    img_files = [f for f in os.listdir(path.get("input_folder")) if f.endswith('.jpg')]
    label_files = [f for f in os.listdir(path.get("label_folder")) if f.endswith('.txt')]
    
    random.shuffle(img_files)
    
    total_files = len(img_files)
    num_train = int(total_files * split_ratio)
    
    train_img_files = img_files[:num_train]
    test_img_files = img_files[num_train:]
    
    train_label_files = [f.replace(".jpg", ".txt") for f in train_img_files if f.replace(".jpg", ".txt") in label_files]
    test_label_files = [f.replace(".jpg", ".txt") for f in test_img_files if f.replace(".jpg", ".txt") in label_files]
    
    for img_file, lbl_file in zip(train_img_files, train_label_files):
        shutil.move(os.path.join(path.get("input_folder"), img_file), os.path.join(train_images_folder, img_file))
        shutil.move(os.path.join(path.get("label_folder"), lbl_file), os.path.join(train_labels_folder, lbl_file))