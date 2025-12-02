import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Kematangan Buah Berdasarkan Warna", page_icon="🍌", layout="centered")
st.title("🍌 Klasifikasi Kematangan Buah Berdasarkan Warna")
st.caption("Upload gambar pisang/mangga dengan latar sederhana untuk hasil terbaik.")

# Sidebar: pilih jenis buah dan parameter
fruit = st.sidebar.selectbox("Jenis buah", ["Pisang", "Mangga", "Tomat", "Apel"])
# Preset ambang warna untuk tiap buah
presets = {
    "Pisang": {"green": (40, 80), "yellow": (22, 33), "brown_v": 80},
    "Mangga": {"green": (40, 85), "yellow": (15, 40), "brown_v": 70},
    "Tomat": {"green": (40, 85), "yellow": (10, 25), "brown_v": 60},
    "Apel": {"green": (35, 75), "yellow": (20, 40), "brown_v": 65}
}

preset = presets.get(fruit)

st.sidebar.write("Ambang warna otomatis berdasarkan buah yang dipilih (bisa disesuaikan):")
hue_range_green = st.sidebar.slider("Hue hijau (OpenCV 0–179)", 0, 179, preset["green"])
hue_range_yellow = st.sidebar.slider("Hue kuning (OpenCV 0–179)", 0, 179, preset["yellow"])
value_min_brown = st.sidebar.slider("Ambang nilai (V) gelap untuk 'cokelat'", 0, 255, preset["brown_v"])

uploaded_file = st.file_uploader("Upload gambar buah (JPG/PNG)", type=["jpg", "jpeg", "png"])


if os.path.exists("riwayat_klasifikasi.csv"):
    st.subheader("📄 Riwayat Klasifikasi Sebelumnya")
    history = pd.read_csv("riwayat_klasifikasi.csv")
    st.dataframe(history)

    if st.button("🗑️ Hapus Riwayat Klasifikasi"):
        os.remove("riwayat_klasifikasi.csv")
        st.warning("Riwayat klasifikasi telah dihapus.")

def classify_fruit(hsv, fruit, h_green, h_yellow, v_brown):
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]

    # Mask area berwarna (hindari latar putih/abu)
    color_mask = (s > 40) & (v > 40)

    # Mask hijau, kuning, cokelat/gelap
    green_mask = color_mask & (h >= h_green[0]) & (h <= h_green[1])
    yellow_mask = color_mask & (h >= h_yellow[0]) & (h <= h_yellow[1])
    brown_mask = color_mask & (v < v_brown)  # piksel gelap dianggap 'cokelat/lewat matang'

    total = max(np.count_nonzero(color_mask), 1)
    pct_green = np.count_nonzero(green_mask) / total
    pct_yellow = np.count_nonzero(yellow_mask) / total
    pct_brown = np.count_nonzero(brown_mask) / total

    # Aturan sederhana
    if pct_brown > 0.35:
        label = "Lewat matang"
    elif pct_yellow > 0.45 and pct_green < 0.25:
        label = "Matang"
    elif pct_green > 0.45 and pct_yellow < 0.35:
        label = "Mentah"
    else:
        label = "Menuju matang"

    score = {
        "hijau": round(pct_green*100, 1),
        "kuning": round(pct_yellow*100, 1),
        "cokelat/gelap": round(pct_brown*100, 1)
    }
    return label, score, (green_mask, yellow_mask, brown_mask, color_mask)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    label, score, masks = classify_fruit(hsv, fruit, hue_range_green, hue_range_yellow, value_min_brown)

    st.subheader("Hasil klasifikasi")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Gambar asli", use_column_width=True)
    with col2:
        st.markdown(f"**Label:** {label}")
        st.markdown(f"- **Hijau:** {score['hijau']}%")
        st.markdown(f"- **Kuning:** {score['kuning']}%")
        st.markdown(f"- **Cokelat/gelap:** {score['cokelat/gelap']}%")
        import matplotlib.pyplot as plt


    # Tampilkan visualisasi mask
    green_mask, yellow_mask, brown_mask, color_mask = masks

    def mask_to_rgb(mask, color):
        out = np.zeros_like(img_np)
        out[mask] = color
        return out

    st.subheader("Visualisasi warna dominan")
    vcol1, vcol2, vcol3, vcol4 = st.columns(4)
    with vcol1:
        st.image(mask_to_rgb(color_mask, (200, 200, 200)), caption="Area berwarna", use_column_width=True)
    with vcol2:
        st.image(mask_to_rgb(green_mask, (80, 200, 80)), caption="Hijau", use_column_width=True)
    with vcol3:
        st.image(mask_to_rgb(yellow_mask, (240, 220, 50)), caption="Kuning", use_column_width=True)
    with vcol4:
        st.image(mask_to_rgb(brown_mask, (90, 60, 30)), caption="Cokelat/gelap", use_column_width=True)

    st.info("Tip: Jika latar belakang ikut terdeteksi, crop gambar atau turunkan ambang saturasi (S) di kode.")
else:
    st.write("Unggah gambar untuk mulai klasifikasi.")

st.markdown("---")
st.markdown("ℹ️ Aplikasi ini memakai pendekatan rule-based HSV. Untuk akurasi lebih tinggi di berbagai varietas dan pencahayaan, bisa ditambah segmentasi buah (GrabCut/DeepLab) atau model klasifikasi terlatih.")