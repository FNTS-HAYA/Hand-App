import streamlit as st
import cv2
import mediapipe as mp

# 部品を使いやすく整理
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

st.set_page_config(page_title="手話学習アプリ", layout="wide")
st.title("🤟 手話学習アプリ：Step 1 完了！")

# セッション状態を使って「実行中か」を管理
if 'run' not in st.session_state:
    st.session_state['run'] = False

col1, col2 = st.columns([3, 1])

with col2:
    start_button = st.button("カメラ開始")
    stop_button = st.button("カメラ停止")

if start_button:
    st.session_state['run'] = True
if stop_button:
    st.session_state['run'] = False
    st.rerun()

# 映像を表示する枠
frame_placeholder = col1.empty()

# AIの準備（with構文を使うと終了処理がきれいになります）
with mp_hands.Hands(
    model_complexity=0, # 0にすると動作が軽くなります
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    cap = cv2.VideoCapture(0)

    while cap.isOpened() and st.session_state['run']:
        success, image = cap.read()
        if not success:
            st.warning("カメラが見つかりません。")
            break

        # 左右反転（鏡のように見せる）して、色を変換
        image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # AIの判定
        results = hands.process(image)

        # 手が見つかったら描画
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # 画面に表示
        frame_placeholder.image(image, channels="RGB")

    cap.release()

if not st.session_state['run']:
    frame_placeholder.write("カメラが停止しています。「カメラ開始」を押してください。")