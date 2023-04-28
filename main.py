import streamlit as st
import cv2
import os
import numpy as np


def distort(image, angle):
    height, width = image.shape[:2]
    image = cv2.resize(image, (int(width), int(height)))

    # 圧縮率計算
    if angle == 0:
        y = 0
    else:
        # 底辺の圧縮割合を計算
        y = 0.006967*angle + 0.012167

    # 圧縮するピクセル数を計算
    compress_pixels = int(width*y)

    # 左下始点の反時計回り
    # [左下][左上][右上][右下]
    src_pts = np.array([[0, height], [0, 0], [width, 0], [width, height]], dtype=np.float32)
    dst_pts = np.array([[compress_pixels, height], [0, 0], [width, 0], [width - compress_pixels, height]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    image = cv2.warpPerspective(image, M, (width, height))

    return image


def panorama(input_file, interval, images, angle):
    # ビデオファイルを読み込み、フレーム数を取得
    cap = cv2.VideoCapture(input_file)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # フレームレートを取得
    frame_interval = int(frame_rate * interval)  # 一定秒数ごとのフレームの間隔を計算
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 0.x秒ごとに画像を切り出す
    for i in range(frame_count):
        # フレームを取得
        frame_id = int(frame_interval * (i + 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            break
        
        # 歪み修正
        frame = distort(frame, angle)

        # 画像をリストに追加
        images.append(frame)

    # 画像を25枚ずつに分割
    temp = [images[i:i+2] for i in range(0, len(images), 25)]
    # 元のリストを空に
    images.clear()

    # 分割したリストをそれぞれごうせい
    for i in temp:
        stitcher = cv2.Stitcher.create(mode=1)
        ret, pano = stitcher.stitch(i)

        # スキャンモードでパノラマ合成した画像をimagesに追加
        if ret == cv2.STITCHER_OK:
            images.append(pano)
            print("Stitch finished 1")

        # 失敗した場合はパノラマモードで合成を試みる
        else:
            stitcher = cv2.Stitcher.create(mode=0)
            ret, pano = stitcher.stitch(i)
            # パノラマ合成した画像を保存
            if ret == cv2.STITCHER_OK:
                print("Stitch finished 2")
                images.append(pano)
            else:
                print("Error during stitching")

    # Stitcherを初期化し、パノラマモードでパノラマ合成を行う
    stitcher = cv2.Stitcher.create(mode=0)
    ret, pano = stitcher.stitch(images)

    # 画像を表示
    if ret == cv2.STITCHER_OK:
        cv2.imwrite('output_image.jpg', pano)
        st.image('output_image.jpg', caption='Result')

    # 失敗した場合はスキャンモードでパノラマ合成を試みる
    else:
        stitcher = cv2.Stitcher.create(mode=1)
        ret, pano = stitcher.stitch(images)

        # 画像を表示
        if ret == cv2.STITCHER_OK:
            cv2.imwrite('output_image.jpg', pano)
            st.image('output_image.jpg', caption='Result')
        else:
            st.write('Error during stitching')



# レイアウト
st.set_option('deprecation.showfileUploaderEncoding', False)

st.title('Map Generate from UAV Movie')
st.sidebar.write('Resource and Parameta')

input_file = st.sidebar.file_uploader('Upload the movie', type=['mp4'])
interval = st.sidebar.slider('Interval seconds (0.x sec)', min_value = 1, max_value = 20, value = 5)/10
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
        panorama('input_movie.mp4', interval, images, angle)


