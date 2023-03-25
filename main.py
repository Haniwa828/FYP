import streamlit as st
import cv2
import os


def panorama(input_file, num_frames, images):
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
        panorama('input_movie.mp4', num_frames, images)


