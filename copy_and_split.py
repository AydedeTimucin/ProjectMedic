import os
import shutil
import tqdm



input_dir_thumb = r"D:/AIN3007_project/train_test_splited/test_data/Breast3__ihc/thumb"
input_dir_real = r"D:/AIN3007_project/raw_data/Breast3__ihc"

out_test = r"D:/AIN3007_project/train_test_splited/test_data/Breast3__ihc/slides"
out_train = r"D:/AIN3007_project/train_test_splited/train_data/Breast3__ihc/slides"

os.makedirs(out_test, exist_ok=True)
os.makedirs(out_train, exist_ok=True)

test_slides = []
for file in os.listdir(input_dir_thumb):
    
    slide = file.split(".")[0]
    test_slides.append(slide)
print(test_slides)



for file in tqdm.tqdm(os.listdir(input_dir_real)):
    old_path = os.path.join(input_dir_real, file)
    
    print(old_path)
    slide = file.split(".")[0]
    if slide in test_slides:
        new_path = os.path.join(out_test, file)
    else:
        new_path = os.path.join(out_train, file)
    
    if os.path.isdir(old_path):
        print("Dir: ", file)
        shutil.move(old_path, new_path)
    else:
        print("File: ", file)
        shutil.move(old_path, new_path)