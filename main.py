import streamlit as st
import cv2
import os
import numpy as np


def distort(image, angle):
    # # 画像の読み込み
    # image = cv2.imread("images/test.jpg")

    # すべての辺のピクセル数を半分にする
    height, width = image.shape[:2]
    image = cv2.resize(image, (int(width), int(height)))
    # image = cv2.resize(image, (int(width/4), int(height/4)))
    # height, width, channels = image.shape[:3]

    # 画像の左下の頂点を右に100ピクセル移動する\
    if angle == 0:
        y = 0
    else:
        # 底辺の圧縮割合を計算
        y = 0.006967*angle + 0.012167

    # 圧縮するピクセル数を計算
    translate_pixels = int(width*y)

    # translate_pixels = 5000

    # 左下始点の反時計回り
    # [左下][左上][右上][右下]
    src_pts = np.array([[0, height], [0, 0], [width, 0], [width, height]], dtype=np.float32)
    dst_pts = np.array([[translate_pixels, height], [0, 0], [width, 0], [width - translate_pixels, height]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    image = cv2.warpPerspective(image, M, (width, height))

    return image


def panorama(input_file, num_frames, images, angle):
    # ビデオファイルを読み込み、フレーム数を取得
    cap = cv2.VideoCapture(input_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 一定フレーム毎に画像を切り出す
    for i in range(num_frames):
        # ビデオファイルからフレームを取得
        frame_id = int(frame_count / (num_frames+1) * (i+1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            break
        
        # 歪み修正ここ？
        frame = distort(frame, angle)

        # 画像をリストに追加
        images.append(frame)

    # Stitcherを初期化し、パノラマ合成を行う
    stitcher = cv2.Stitcher.create(mode=1)
    ret, pano = stitcher.stitch(images)

    # パノラマ合成した画像を保存
    if ret == cv2.STITCHER_OK:
        cv2.imwrite('output_image.jpg', pano)
        # img = Image.open(pano)
        st.image('output_image.jpg', caption='Result')
    else:
        st.write('Error during stitching')



# レイアウト
st.set_option('deprecation.showfileUploaderEncoding', False)

st.title('Map Generate from UAV Movie')
st.sidebar.write('Resource and Parameta')

input_file = st.sidebar.file_uploader('Upload the movie', type=['mp4'])
num_frames = st.sidebar.slider('Frame number', min_value = 25, max_value = 100, value = 25)
angle = st.sidebar.slider('Angle', min_value = 0, max_value = 89, value = 0)


# 実行
if input_file is not None:
    with st.spinner('Processing...'):    
        # 保存
        # ファイル名の取得
        filename = 'input_movie.mp4'
        # ファイルの保存先
        save_path = os.path.join(os.getcwd(), filename)
        # ファイルの保存
        with open(save_path, 'wb') as f:
            f.write(input_file.getbuffer())

        # 切り出した画像を保存するリスト
        images = []

        # 合成
        panorama('input_movie.mp4', num_frames, images, angle)


