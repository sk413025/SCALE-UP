# 導入必要的庫
import os
import torchvision
import PIL

# 定義數據集的根目錄
root = "data"

# 定義儲存圖片的資料夾
train_dir = "train"
test_dir = "testset"

# 如果資料夾不存在，創建它們
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
if not os.path.exists(test_dir):
    os.mkdir(test_dir)

# 加載訓練集和測試集
train_dataset = torchvision.datasets.GTSRB(root, split="train", download=True)
test_dataset = torchvision.datasets.GTSRB(root, split="test", download=True)

# 遍歷訓練集，把每張圖片轉換成 RGB 格式，並且保存為 png 格式到對應的子資料夾
for i, (image, label) in enumerate(train_dataset):
    # 轉換成 RGB 格式
    image = image.convert("RGB")
    # 獲取圖片的文件名，例如 00000_00000.ppm
    filename = train_dataset._samples[i][0].split("/")[-1]
    # 把文件名的後綴改成 png，例如 00000_00000.png
    filename = filename.replace(".ppm", ".png")
    # 指定儲存的路徑，例如 train/0/00000_00000.png
    save_path = os.path.join(train_dir, str(label), filename)
    # 如果子資料夾不存在，創建它
    if not os.path.exists(os.path.dirname(save_path)):
        os.mkdir(os.path.dirname(save_path))
    # 保存圖片
    image.save(save_path)

# 遍歷測試集，把每張圖片轉換成 RGB 格式，並且保存為 png 格式到對應的子資料夾
for i, (image, label) in enumerate(test_dataset):
    # 轉換成 RGB 格式
    image = image.convert("RGB")
    # 獲取圖片的文件名，例如 00000.ppm
    filename = test_dataset._samples[i][0].split("/")[-1]
    # 把文件名的後綴改成 png，例如 00000.png
    filename = filename.replace(".ppm", ".png")
    # 指定儲存的路徑，例如 testset/0/00000.png
    save_path = os.path.join(test_dir, str(label), filename)
    # 如果子資料夾不存在，創建它
    if not os.path.exists(os.path.dirname(save_path)):
        os.mkdir(os.path.dirname(save_path))
    # 保存圖片
    image.save(save_path)
