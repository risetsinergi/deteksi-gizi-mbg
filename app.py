import streamlit as st
from PIL import Image
import pandas as pd
from ultralytics import YOLO
import numpy as np

# --- PENGATURAN DASAR & KONFIGURASI ---

st.set_page_config(
    page_title="Analisis Gizi Makan Siang",
    page_icon="🍱",
    layout="wide"
)

# --- KONFIGURASI SANDI ---
# Mengambil password dari st.secrets
CORRECT_PASSWORD = st.secrets["APP_PASSWORD"]

# --- DATA (DEFINISI KONSTANTA) ---
# Data diletakkan di luar fungsi agar tidak di-load ulang setiap interaksi

akg_profiles = {
    "Anak PAUD (1-3 Tahun)": {
        'Energi (kkal)': 1350, 'Protein (g)': 20, 'Lemak (g)': 45, 'Karbohidrat (g)': 215, 'Serat (g)': 19,
        'Kolin (mg)': 200, 'Folat (µg)': 160
    },
    "Anak TK (4-6 Tahun)": {
        'Energi (kkal)': 1400, 'Protein (g)': 25, 'Lemak (g)': 50, 'Karbohidrat (g)': 220, 'Serat (g)': 20,
        'Kolin (mg)': 250, 'Folat (µg)': 200
    },
    "Anak SD (7-9 Tahun)": {
        'Energi (kkal)': 1650, 'Protein (g)': 40, 'Lemak (g)': 55, 'Karbohidrat (g)': 250, 'Serat (g)': 23,
        'Kolin (mg)': 250, 'Folat (µg)': 300 
    },
    "Anak SD (10-12 Tahun)": {
        'Energi (kkal)': 1950, 'Protein (g)': 52, 'Lemak (g)': 65, 'Karbohidrat (g)': 290, 'Serat (g)': 27,
        'Kolin (mg)': 250, 'Folat (µg)': 300 
    },
    "Anak SMP (13-15 Tahun)": {
        'Energi (kkal)': 2200, 'Protein (g)': 67, 'Lemak (g)': 75, 'Karbohidrat (g)': 330, 'Serat (g)': 32,
        'Kolin (mg)': 375, 'Folat (µg)': 400 
    },
    "Anak SMA (16-18 Tahun)": {
        'Energi (kkal)': 2350, 'Protein (g)': 70, 'Lemak (g)': 77, 'Karbohidrat (g)': 350, 'Serat (g)': 33,
        'Kolin (mg)': 400, 'Folat (µg)': 400 
    },
    "Ibu Hamil (Trimester 2 & 3)": {
        'Energi (kkal)': 2500, 'Protein (g)': 70, 'Lemak (g)': 70, 'Karbohidrat (g)': 400, 'Serat (g)': 34,
        'Kolin (mg)': 450, 'Folat (µg)': 600 
    }
}

PORSI_MAKAN_SIANG = 0.30

data_gizi = {
    'nama_makanan': [
        'nasi_putih', 'ayam', 'nasi_kuning', 'nasi_liwet', 'buah_jeruk', 
        'buah_melon', 'buah_pisang', 'buah_duku', 'sayur_capcay', 
        'sayur_wortel_kacang', 'sayur', 'wortel', 'susu', 'tahu', 'tempe', 
        'tempe_bacem', 'ayam_kecap', 'buah_semangka',
        'roti', 'burger', 'omelet', 'nasi_labu_kuning', 'stik_singkong_labu',
        'oregano', 'ikan'
    ],
    'Energi (kkal)': [
        140, 250, 180, 190, 70, 34, 105, 70, 80, 60, 50, 41, 80, 80, 100, 110, 220, 45,
        130, 450, 150, 160, 180, 5, 180
    ],
    'Protein (g)': [
        3, 25, 4, 4.5, 1.5, 0.8, 1.3, 1, 4, 2.5, 2, 0.9, 4, 7, 9, 10, 23, 0.9,
        4.5, 25, 13, 3.5, 3, 0.1, 22
    ],
    'Lemak (g)': [
        0.3, 15, 5, 6, 0.2, 0.2, 0.4, 0.2, 4, 3, 2.5, 0.2, 4.5, 6, 6, 5, 10, 0.2,
        1.5, 20, 10, 3, 7, 0.1, 10
    ],
    'Karbohidrat (g)': [
        30, 2, 30, 31, 18, 8, 27, 17, 9, 7, 6, 10, 6, 2, 8, 10, 10, 11,
        25, 30, 2, 28, 25, 1, 0
    ],
    'Serat (g)': [
        0.5, 0, 1, 1, 3.5, 0.9, 3.1, 4, 3.5, 3, 3, 2.8, 0, 1, 1.5, 1.4, 0.5, 0.6,
        2, 4, 0.5, 2.5, 4, 0.2, 0
    ],
    'Kolin (mg)': [
        2, 75, 3, 3, 8, 7, 10, 5, 30, 20, 15, 6, 20, 25, 30, 30, 70, 4,
        10, 80, 290, 12, 15, 0.1, 60
    ],
    'Folat (µg)': [
        5, 8, 6, 6, 30, 20, 20, 10, 40, 45, 30, 15, 6, 20, 25, 25, 8, 3,
        40, 60, 50, 25, 30, 1, 10
    ]
}

df_gizi = pd.DataFrame(data_gizi)
all_known_foods = sorted(list(data_gizi['nama_makanan'])) 

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    return model

# --- FUNGSI UTAMA APLIKASI (MAIN APP) ---
def main_app():
    # Judul dan Deskripsi
    st.title('🍱 Analisis Gizi Seimbang Makan Bergizi Gratis')
    st.write("Unggah gambar nampan makanan Anda. Aplikasi akan menganalisis pemenuhan gizinya berdasarkan porsi 'Makan Siang' (30%) dari Angka Kecukupan Gizi (AKG) Kemenkes.")

    # Load Model di dalam main_app (atau panggil fungsi load yang sudah di-cache)
    try:
        model = load_model('best.pt')
    except Exception as e:
        st.error(f"Error memuat model: {e}. Pastikan file 'best.pt' ada di folder yang sama dengan app.py")
        st.stop()

    # --- TAMPILAN UTAMA APLIKASI ---

    st.subheader("Pilih Profil Gizi")
    profile_choice = st.selectbox(
        "Pilih profil AKG Harian:",
        options=list(akg_profiles.keys()),
        index=0,
        label_visibility="visible" 
    )

    akg_harian = akg_profiles[profile_choice]
    porsi_makan = PORSI_MAKAN_SIANG

    st.subheader("Unggah Gambar Makananmu")
    uploaded_file = st.file_uploader("Unggah gambar di sini...", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col_img, col_res = st.columns(2)

        with col_img:
            st.image(image, caption='Gambar yang diunggah.', use_container_width=True)

        with col_res:
            st.subheader("🔍 Hasil Analisis")
            with st.spinner('Model sedang menganalisis gambar...'):
                results = model(image)
                
                detected_objects = set()
                for r in results:
                    for box in r.boxes:
                        class_name = model.names[int(box.cls)]
                        detected_objects.add(class_name)
                
                if detected_objects:
                    st.success(f"**Otomatis terdeteksi:** {', '.join(list(detected_objects))}")
                else:
                    st.info("Model tidak mendeteksi item apapun. Silakan tambahkan secara manual.")

                st.write("---")
                st.subheader("Koreksi & Konfirmasi Manual")
                
                detected_list = list(detected_objects) 
                
                final_food_list = st.multiselect(
                    "Periksa hasil deteksi. Tambah/hapus item untuk konfirmasi manual:",
                    options=all_known_foods,
                    default=detected_list  
                )
                
                if final_food_list:
                    
                    final_food_set = set(final_food_list)
                    
                    estimasi = df_gizi[df_gizi['nama_makanan'].isin(final_food_set)]
                    total_gizi_makanan = estimasi.sum(numeric_only=True)
                    
                    st.write("---")
                    st.subheader("📊 Estimasi Kandungan Gizi")
                    st.dataframe(total_gizi_makanan.rename('Total Estimasi').to_frame())
                    
                    st.write("---")
                    st.subheader(f"📈 Analisis Pemenuhan Gizi Makan Bergizi Gratis")
                    st.write(f"Target gizi untuk **Makan Bergizi Gratis** (estimasi {porsi_makan*100:.0f}% dari total harian {profile_choice}).")
                    
                    komponen_kurang = [] 
                    
                    for gizi, nilai_harian in akg_harian.items():
                        target_gizi_porsi = nilai_harian * porsi_makan
                        nilai_aktual = total_gizi_makanan.get(gizi, 0)
                        persentase = (nilai_aktual / target_gizi_porsi) * 100 if target_gizi_porsi > 0 else 0
                        
                        st.write(f"**{gizi}:** {nilai_aktual:.1f} / **{target_gizi_porsi:.1f}** ({persentase:.1f}%)")
                        st.progress(int(min(persentase, 100)))
                        
                        if persentase < 100:
                            nama_gizi_bersih = gizi.split(' ')[0].lower() 
                            komponen_kurang.append(nama_gizi_bersih)

                    
                    st.write("---")
                    st.subheader("📜 Kesimpulan")
                    
                    if not komponen_kurang:
                        st.success("🎉 **Luar biasa!** Kebutuhan gizi untuk makan siang Anda sudah **Terpenuhi Sempurna** untuk semua komponen.")
                    else:
                        komponen_string = ", ".join(komponen_kurang)
                        st.warning(f"**Perhatian:** Porsi makan siang Anda masih **belum memenuhi target** untuk komponen: **{komponen_string}**.")
                        st.info("Pastikan untuk melengkapi kebutuhan gizi ini di waktu makan lainnya atau dengan menambahkan porsi.")
                
                else:
                    st.warning("Tidak ada makanan yang dipilih untuk dianalisis.")

# --- FUNGSI LOGIN ---
def show_login_page():
    with st.container():
        st.title("🔐 Halaman Terproteksi")
        st.write("Aplikasi Analisis Gizi ini diproteksi. Silakan masukkan sandi.")
        
        with st.form("login_form"):
            password = st.text_input("Sandi", type="password")
            submitted = st.form_submit_button("Masuk")

            if submitted:
                if password == CORRECT_PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Sandi yang Anda masukkan salah.")
                    st.session_state.authenticated = False

# --- LOGIKA SESSION STATE & EKSEKUSI ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Cek status autentikasi
if st.session_state.authenticated:
    main_app()
else:
    show_login_page()
