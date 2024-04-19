import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import os
# model.pyのpredict関数を使用
from model import predict

# pyplotを使用する際に注記が出ないようにする文
st.set_option("deprecation.showPyplotGlobalUse", False)

# 関数化する
def main():
    # タイトル
    st.title("オリジナルニューラルネットワークによる猫と犬の画像分類")
    st.write("最終更新日: 2024/4/19")

    # サイドバーのmenu
    menu = ["概要", "分類結果", "画像シュミレーター"]
    # サイドバーの作成
    chosen_menu = st.sidebar.selectbox(
        "menu選択", menu
    )

    # ファイルの設定
    # 訓練済みのモデルファイル

    # 犬の画像
    cat_file = "./cat.111.jpg"
    # 猫の画像
    dog_file = "./dog.2.jpg"
    # テストデータの結果
    result_file = "./test_result.jpg"


    # 読み込めているかを確認
    is_cat_file = os.path.isfile(cat_file)
    is_dog_file = os.path.isfile(dog_file)
    is_result_file = os.path.isfile(result_file)

    # printで出力すると、ターミナルに出る
    # st.writeだとブラウザ上に出る
    print(is_cat_file)
    print(is_dog_file)
    print(is_result_file)
    
    

    # menuの中身
    # 分析の概要
    if chosen_menu == "概要":
        st.subheader("概要")
        st.write("犬と猫の画像を分類する畳み込みニューラルネットワークを作成しました。")
        st.write("")
        st.write("畳み込みニューラルネットワークについて")
        st.write(
            """
            畳み込みニューラルネットワーク（CNN）は、画像認識分野で広く用いられるAI技術です。
            画像の特徴を効率的に抽出・分析できるため、顔認識、物体検出、画像分類など、様々なタスクに活用されています。
            """
        )
        st.write("")
        st.subheader("データセットの内容")
        st.write("訓練データには、25000枚の猫と犬の画像が含まれます。")
        st.write("テストデータにも25000枚の猫と犬の画像が含まれます。")
        st.write("訓練データを用いてモデルを構築して、テストデータの画像を分類しました。")
        st.write(" ")
        st.write(" ")
        st.text("訓練データの一部")
        # 画像の表示
        image_cat = Image.open(cat_file)
        image_dog = Image.open(dog_file)
        st.image([image_cat, image_dog], width=300)
       
    # 分類の結果
    elif chosen_menu == "分類結果":
        st.subheader("分類結果")
        # 結果の表示
        image_result = Image.open(result_file)
        st.image(image_result)

        # 結果についての説明
        st.write("構築したニューラルネットワークは未知のデータに対して77.5%の精度で分類を行うことができました。")
        st.write("上記の画像は25000枚のテストデータのうち16枚を可視化したものです。")
        st.write("一部誤差あり、猫の画像を'dogs'と予測していますが、概ね良好な精度を確認しました。")

    elif chosen_menu == "画像シュミレーター":
        st.subheader("画像シュミレーター")
        st.write("訓練データを使用して訓練したニューラルネットワークでアップロードされた画像が猫か犬か判定します。")
        # 空白行
        st.write("")
        # ラジオボタンの作成
        img_source = st.radio("画像のソースを選択してください", ("画像をアップロード", "カメラで撮影"))

        # 画像のアップロード
        if img_source == "画像をアップロード":
            # ファイルをアップロード
            img_file = st.file_uploader("画像を選択してください", type=["png", "jpg"])
        # カメラで撮影する場合
        elif img_source == "カメラで撮影":
            # カメラ撮影
            img_file = st.camera_input("カメラで撮影")

        # 推定の処理
        # img_fileが存在する場合に処理を進める
        if img_file is not None:
            # 特定の処理が行われていることを知らせる
            with st.spinner("推定中です..."):
                # 画像ファイルを開く
                img = Image.open(img_file)
                # 画面に画像を表示
                st.image(img, caption="予測対象画像", width=480)

                # 空白行
                st.write("")

                # 予測
                results = predict(img)

                # 結果の表示
                st.subheader("判定結果")
                for result in results:
                    st.write(f"{round(result[2]*100, 2)}%の確率で{result[0]}です。")

                # 円グラフの表示
                # 円グラフのラベル
                pie_labels = [result[1] for result in results]
                # 予測した確率
                pie_probs = [result[2] for result in results]

                fig, ax = plt.subplots()
                # グラフをドーナツ型にする設定
                wedgeprops = {"width":0.3, "edgecolor":"white"}
                # フォントサイズの設定
                text_props = {"fontsize":6}
                ax.pie(pie_probs, labels=pie_labels, counterclock=False, startangle=90, textprops=text_props, autopct="%.2f", wedgeprops=wedgeprops)
                st.pyplot(fig)

# streamlitを実行したときにmain()を実行するという表記
if __name__ == "__main__":
    main()
