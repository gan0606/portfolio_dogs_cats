import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# クラス名
# 日本語
classes_ja = ["猫", "犬"]
# 英語
classes_en = ["cats", "dogs"]
# クラスの数
n_class = len(classes_ja)
# ユーザーが入力する画像サイズ
# 訓練済みのmodelへの入力サイズに合わせる
img_size = 32

# モデルの構築
# 訓練済みにモデルと同じものを記載
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 畳み込み層
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )

        # 全結合層
        self.classifier = nn.Sequential(
            nn.Linear(in_features=400, out_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=2),
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x    

# 画像を予測する関数を定義
# imgはユーザーが入力した画像
def predict(img):
    # 画像をrgbに変換
    img = img.convert("RGB")
    # 画像を32x32に変換
    # 訓練したモデルと同じ画像サイズ
    img = img.resize((img_size, img_size))
    # 訓練したモデルの評価画像と同じ処理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])
    # 画像に上記の処理を行う
    img = transform(img)

    # 画像をモデルに入力可能な形に変換
    # batch_sizex, num_ch, w, h
    input = img.reshape(1, 3, img_size, img_size)

    # 訓練済みモデルの呼び出し
    net = Net()
    # 読み込み
    # streamlit環境はcpuなのでdeviceをcpuに設定
    net.load_state_dict(torch.load("model_cnn3.path", map_location=torch.device("cpu")))

    # 予測
    net.eval()
    pred = net(input)

    # 結果を確率で返す
    # squeezeでバッチの次元を取り除いている
    pred_prob = torch.nn.functional.softmax(torch.squeeze(pred), dim=0)
    # 降順に並び替える
    sorted_prob, sorted_idx = torch.sort(pred_prob, descending=True)
    return [(classes_ja[idx], classes_en[idx], prob.item()) for idx, prob in zip(sorted_idx, sorted_prob)]
