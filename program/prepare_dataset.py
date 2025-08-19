import os
import shutil
import random
from PIL import Image

# --- CONFIG ---
RAW_DIR = 'datasets/raw/UTKFace'
CLEAN_DIR = 'datasets/clean'
AGE_DIR = os.path.join(CLEAN_DIR, 'age')
#GENDER_DIR = os.path.join(CLEAN_DIR, 'gender')
IMG_SIZE = (128, 128)

# New Age groups (0–105, last bin 97–105)
AGE_GROUPS = [
    (0, 2),
    (3, 6),
    (7, 12),
    (13, 19),
    (20, 29),
    (30, 39),
    (40, 49),
    (50, 59),
    (60, 79),
    (80, 100),
]

# Gender labels
#GENDER_LABELS = {0: 'male', 1: 'female'}

# --- CREATE FOLDERS ---
for start, end in AGE_GROUPS:
    folder_name = f"{start}-{end}"
    os.makedirs(os.path.join(AGE_DIR, folder_name), exist_ok=True)

#for gender in GENDER_LABELS.values():
#    os.makedirs(os.path.join(GENDER_DIR, gender), exist_ok=True)

# --- GROUP FILES BY AGE AND GENDER ---
age_files = {f"{start}-{end}": [] for start, end in AGE_GROUPS}
#gender_files = {g: [] for g in GENDER_LABELS.values()}

files = [f for f in os.listdir(RAW_DIR) if f.endswith('.jpg')]

for fname in files:
    try:
        age, gender, race, _ = fname.split('_', 3)
        age = int(age)
        gender = int(gender)

        # Skip outliers (106–120)
        if age > 105:
            continue

        # Age
        for start, end in AGE_GROUPS:
            if start <= age <= end:
                age_files[f"{start}-{end}"].append(fname)
                break

        # Gender
        #gender_label = GENDER_LABELS.get(gender, 'unknown')
        #if gender_label != 'unknown':
            #gender_files[gender_label].append(fname)

    except Exception as e:
        print(f"Skipping {fname}: {e}")

# --- BALANCE AGE CLASSES (DOWN-SAMPLE) ---
min_count = min(len(files) for files in age_files.values() if len(files) > 0)
print(f"Balancing all age classes to {min_count} images each")

for age_group, files_list in age_files.items():
    if len(files_list) < min_count:
        print(f"Skipping {age_group}, not enough samples")
        continue
    selected_files = random.sample(files_list, min_count)
    dest_folder = os.path.join(AGE_DIR, age_group)
    for fname in selected_files:
        # Load, resize, save
        img_path = os.path.join(RAW_DIR, fname)
        img = Image.open(img_path).convert('RGB')
        img = img.resize(IMG_SIZE)
        img.save(os.path.join(dest_folder, fname))

# --- COPY AND RESIZE ALL GENDER FILES ---
#for gender, files_list in gender_files.items():
    #dest_folder = os.path.join(GENDER_DIR, gender)
    #for fname in files_list:
        #img_path = os.path.join(RAW_DIR, fname)
        #img = Image.open(img_path).convert('RGB')
        #img = img.resize(IMG_SIZE)
        #img.save(os.path.join(dest_folder, fname))

print("Finished organizing, resizing, and balancing dataset!")
