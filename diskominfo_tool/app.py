import streamlit as st
import pandas as pd
import plotly.express as px
from utils import loader, analysis, visualization

# Konfigurasi Halaman
st.set_page_config(
    page_title="Tools Analisis Data Diskominfo",
    page_icon="📊",
    layout="wide"
)

# --- HEADER & LOGO ---
col_h1, col_h2, col_h3 = st.columns([1, 6, 1])

with col_h1:
    try:
        st.image("logo_lamongan.png", width=120)
    except:
        st.empty()

with col_h2:
    # Judul Utama di Tengah
    st.markdown("<h1 style='text-align: center;'>Tools Analisis Data Diskominfo</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Aplikasi analisis deskriptif untuk data CSV dan Excel.</p>", unsafe_allow_html=True)

with col_h3:
    try:
        st.image("logo.png", width=120)
    except:
        st.write("*(Logo Diskominfo)*")

st.divider()

# --- UPLOAD SECTION (CENTER) ---
col_up_le, col_up_mid, col_up_ri = st.columns([1, 2, 1])
with col_up_mid:
    st.markdown("### 📂 Upload File")
    uploaded_file = st.file_uploader("Pilih file CSV atau Excel", type=['csv', 'xlsx', 'xls'])
    if not uploaded_file:
        st.info("Format yang didukung: .csv, .xlsx, .xls")

if uploaded_file is not None:
    # Memuat data
    df = loader.load_data(uploaded_file)
    
    if df is not None:
        # --- FITUR FILTER (EXPANDER) ---
        with st.expander("🔍 Filter Data"):
        
            # Opsi untuk filter berdasarkan kolom kategorikal
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            
            # Kita buat salinan df agar tidak merusak data asli saat filtering bertingkat
            df_filtered = df.copy()
            
            if len(cat_cols) > 0:
                for col in cat_cols:
                    # Ambil nilai unik
                    unique_vals = df[col].unique().tolist()
                    # Multiselect untuk filter
                    selected_vals = st.multiselect(
                        f"Filter: {col}",
                        options=unique_vals,
                        default=unique_vals # Default pilih semua
                    )
                    
                    # Terapkan filter
                    if selected_vals:
                        df_filtered = df_filtered[df_filtered[col].isin(selected_vals)]
            
            st.caption(f"Menampilkan {len(df_filtered)} dari {len(df)} baris data.")
        
        # Gunakan df_filtered untuk analisis selanjutnya
        df = df_filtered
        
        # Menampilkan pesan sukses
        # st.sidebar.success("File berhasil dimuat & Filter aktif!") # Optional, agar tidak terlalu ramai
        
        # Membuat Tabs dengan urutan baru
        tab_titles = [
            "📋 Overview Data", 
            "📈 Statistik Deskriptif", 
            "🧮 Grouping/Pivot",
            "📉 Regresi Sederhana", 
            "📈 Regresi Berganda",
            "🔮 Forecasting"
        ]
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_titles)
        
        # --- TAB 1: OVERVIEW DATA ---
        with tab1:
            st.header("Overview Data")
            col1, col2 = st.columns(2)
            info = analysis.get_basic_info(df)
            with col1:
                st.metric("Jumlah Baris", info['rows'])
                st.metric("Jumlah Kolom", info['columns'])
            with col2:
                st.metric("Total Missing Values", info['missing_values'])
            st.subheader("Data Lengkap (Terfilter)")
            st.dataframe(df, use_container_width=True)
            with st.expander("Lihat Tipe Data Kolom"):
                st.write(df.dtypes.astype(str))
            if info['missing_values'] > 0:
                st.subheader("Missing Values per Kolom")
                st.write(info['missing_per_column'])

        # --- TAB 2: STATISTIK DESKRIPTIF ---
        with tab2:
            st.header("Statistik Deskriptif")
            st.subheader("Ringkasan Statistik (Kolom Numerik)")
            stats = analysis.get_descriptive_stats(df)
            if not stats.empty:
                st.dataframe(stats.style.format("{:.2f}"), use_container_width=True)
                csv = stats.to_csv().encode('utf-8')
                st.download_button("📥 Unduh Tabel Statistik (CSV)", csv, "statistik_deskriptif.csv", "text/csv", key='download-csv')
            else:
                st.warning("Tidak ada kolom numerik ditemukan dalam data.")
            st.subheader("Distribusi Frekuensi (Kolom Kategorikal)")
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_cols:
                selected_cat_col = st.selectbox("Pilih Kolom Kategorikal:", categorical_cols)
                freq_dist = analysis.get_frequency_dist(df, selected_cat_col)
                st.dataframe(freq_dist, use_container_width=True)
            else:
                st.info("Tidak ada kolom kategorikal ditemukan.")

        # --- TAB 3: GROUPING/PIVOT ---
        with tab3:
            st.header("🧮 Grouping & Pivot Table")
            st.markdown("Lakukan pengelompokan data berdasarkan kategori dan hitung nilai statistiknya.")
            
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if cat_cols and num_cols:
                c1, c2, c3 = st.columns(3)
                with c1: 
                    g_cols = st.multiselect("Group By (Kategori):", cat_cols, default=[cat_cols[0]] if cat_cols else [], key='piv_g_multi')
                with c2: 
                    v_cols = st.multiselect("Value (Numerik):", num_cols, default=[num_cols[0]] if num_cols else [], key='piv_v_multi')
                with c3: 
                    agg_func = st.selectbox("Fungsi Agregasi:", ["mean", "sum", "count", "min", "max", "median"], key='piv_f_new')
                
                if st.button("Jalankan Grouping/Pivot", use_container_width=True):
                    if g_cols and v_cols:
                        # Melakukan grouping
                        res = df.groupby(g_cols)[v_cols].agg(agg_func).reset_index()
                        
                        # Menampilkan hasil
                        st.subheader(f"Hasil {agg_func.capitalize()}")
                        st.dataframe(res, use_container_width=True)
                        
                        # Download button
                        csv_res = res.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Unduh Hasil Grouping (CSV)", csv_res, "hasil_grouping.csv", "text/csv", key='download-piv')
                        
                        # Visualisasi sederhana (ambil kolom pertama jika milih banyak)
                        if len(g_cols) == 1 and len(v_cols) == 1:
                            st.subheader("Visualisasi")
                            fig_piv = visualization.plot_bar_chart(res, x_col=g_cols[0], y_col=v_cols[0])
                            st.plotly_chart(fig_piv, use_container_width=True)
                        elif len(g_cols) >= 1:
                            st.info("💡 Pilih tepat 1 kolom Group By dan 1 kolom Value untuk melihat visualisasi grafik.")
                    else:
                        st.warning("Pilih minimal satu kolom Group By dan satu kolom Value.")
            else:
                st.info("Butuh data kategori dan numerik untuk fitur grouping.")

        # --- TAB 4: REGRESI SEDERHANA ---
        with tab4:
            st.header("📉 Analisis Regresi Linear Sederhana")
            st.markdown("Memprediksi **1 Variabel Target (Y)** menggunakan **1 Variabel Bebas (X)**.")
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) >= 2:
                col_x, col_y = st.columns(2)
                with col_x: x_col = st.selectbox("Pilih Variabel (X) - Penyebab:", numeric_cols, key='reg_simple_x')
                with col_y: y_col = st.selectbox("Pilih Variabel (Y) - Akibat/Target:", numeric_cols, index=1 if len(numeric_cols)>1 else 0, key='reg_simple_y')
                if x_col != y_col:
                    if st.button("Jalankan Regresi Sederhana"):
                        result = analysis.perform_linear_regression(df, x_col, y_col)
                        if result:
                            # --- A. UJI ASUMSI KLASIK ---
                            st.subheader("A. Uji Asumsi Klasik")
                            
                            # 1. Normalitas
                            norm = analysis.check_normality(result['residuals'])
                            norm_msg = "✅ Normal" if norm['is_normal'] else "❌ Tidak Normal"
                            
                            # 4. Autokorelasi
                            dw_val = analysis.check_autocorrelation(result['residuals'])
                            dw_msg = "✅ Tidak ada autokorelasi" if 1.5 < dw_val < 2.5 else "⚠️ Terindikasi Autokorelasi"
                            
                            col_a1, col_a2, col_a3 = st.columns(3)
                            col_a1.metric("1. Normalitas (Shapiro)", f"{norm['p_value']:.4f}", norm_msg)
                            col_a2.metric("2. Autokorelasi (DW)", f"{dw_val:.3f}", dw_msg)
                            
                            # Interpretasi Asumsi
                            st.caption(f"💡 **Interpretasi**: Residual {'**berdistribusi normal**' if norm['is_normal'] else '**tidak normal**'} (P > 0.05). {'Tidak ada masalah autokorelasi' if 1.5 < dw_val < 2.5 else 'Ada indikasi autokorelasi'} pada data.")
                            
                            # Visualisasi Asumsi
                            with st.expander("Lihat Plot Asumsi"):
                                tab_viz_1, tab_viz_2 = st.tabs(["Normalitas", "Linieritas/Fit"])
                                with tab_viz_1:
                                    import plotly.figure_factory as ff
                                    fig_norm = ff.create_distplot([result['residuals']], ['Residuals'], bin_size=0.2, show_rug=False)
                                    st.plotly_chart(fig_norm, use_container_width=True)
                                with tab_viz_2:
                                    fig_reg = visualization.plot_regression(result['X'], result['y'], result['y_pred'], x_col, y_col)
                                    st.plotly_chart(fig_reg, use_container_width=True)

                            # --- B. UJI SIGNIFIKANSI ---
                            st.subheader("B. Uji Signifikansi")
                            
                            # 1. Uji F
                            f_msg = "✅ Model Layak (Signifikan)" if result['f_pvalue'] < 0.05 else "❌ Model Tidak Layak"
                            st.write(f"**1. Uji F (Kecocokan Model)**: F-Stat = {result['f_value']:.4f}, Prob(F) = {result['f_pvalue']:.4e} -> {f_msg}")
                            st.caption(f"💡 **Interpretasi**: Variabel independen **{'secara simultan berpengaruh signifikan' if result['f_pvalue'] < 0.05 else 'tidak berpengaruh signifikan'}** terhadap variabel dependen.")
                            
                            # 2. Uji t (Parameter)
                            st.write("**2. Uji t (Signifikansi Parameter)**")
                            t_df = pd.DataFrame({
                                "Koefisien": [result['intercept'], result['slope']],
                                "t-Stat": [result['t_values']['const'], result['t_values'][x_col]],
                                "P-Value": [result['p_values']['const'], result['p_values'][x_col]],
                                "Kesimpulan": ["Signifikan" if result['p_values']['const'] < 0.05 else "-", 
                                               "Signifikan" if result['p_values'][x_col] < 0.05 else "Tidak Signifikan"]
                            }, index=["Intercept (a)", f"Slope ({x_col})"])
                            st.dataframe(t_df, use_container_width=True)
                            if result['p_values'][x_col] < 0.05:
                                st.caption(f"💡 **Interpretasi**: Variabel **{x_col}** memiliki pengaruh yang **signifikan** terhadap **{y_col}**.")
                            else:
                                st.caption(f"💡 **Interpretasi**: Variabel **{x_col}** **tidak** berpengaruh signifikan terhadap **{y_col}**.")

                            # --- C. MODEL AKHIR ---
                            st.subheader("C. Model Akhir")
                            st.info(f"**Y = {result['intercept']:.4f} + {result['slope']:.4f} * {x_col}**")

                            # --- D. KOEFISIEN DETERMINASI ---
                            st.subheader("D. Koefisien Determinasi")
                            cd1, cd2 = st.columns(2)
                            cd1.metric("R-Squared", f"{result['r2']:.4f}", help="Menjelaskan seberapa besar pengaruh X terhadap Y")
                            cd2.metric("Adj. R-Squared", f"{result['adj_r2']:.4f}")
                            st.caption(f"💡 **Interpretasi**: Variabel independen mampu menjelaskan **{result['r2']*100:.2f}%** perubahan pada variabel **{y_col}**. Sisanya dijelaskan oleh faktor lain diluar model.")

                            # --- E. MSE ---
                            st.subheader("E. MSE (Mean Squared Error)")
                            st.metric("MSE", f"{result['mse']:.4f}")

                        else: st.error("Gagal melakukan analisis. Data mungkin kurang atau terjadi error numerik.")
                else: st.warning("Variabel X dan Y harus berbeda.")
            else: st.info("Butuh minimal 2 kolom numerik.")

        # --- TAB 5: REGRESI BERGANDA ---
        with tab5:
            st.header("📈 Analisis Regresi Linear Berganda")
            st.markdown("Memprediksi **1 Y** menggunakan **Banyak X**.")
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) >= 3:
                y_col_multi = st.selectbox("Pilih Variabel Target (Y):", numeric_cols, key='reg_multi_y')
                candidate_x = [c for c in numeric_cols if c != y_col_multi]
                x_cols_multi = st.multiselect("Pilih Variabel Bebas (X):", candidate_x, key='reg_multi_x')
                if x_cols_multi:
                    if st.button("Jalankan Regresi Berganda"):
                        result = analysis.perform_multiple_regression(df, x_cols_multi, y_col_multi)
                        if result:
                            # --- A. UJI ASUMSI KLASIK ---
                            st.subheader("A. Uji Asumsi Klasik")
                            
                            # Hitung Asumsi
                            resid = result['residuals']
                            norm = analysis.check_normality(resid)
                            bp = analysis.check_homoscedasticity(resid, result['X_with_const'])
                            dw_val = analysis.check_autocorrelation(resid)
                            vif_df = analysis.calculate_vif(result['X'])

                            # Tampilkan Ringkasan Asumsi
                            col_as1, col_as2, col_as3, col_as4 = st.columns(4)
                            col_as1.metric("1. Normalitas (P)", f"{norm['p_value']:.4f}", "Normal" if norm['is_normal'] else "Tidak Normal")
                            col_as2.metric("2. Linieritas", "Cek Plot", "Resid vs Pred")
                            col_as3.metric("3. Homoskedastis (P)", f"{bp['p_value']:.4f}", "Ya" if bp['is_homoscedastic'] else "Tidak")
                            col_as4.metric("4. Autokorelasi", f"{dw_val:.3f}", "Aman" if 1.5 < dw_val < 2.5 else "Warning")
                            
                            # Interpretasi Asumsi Lengkap
                            assump_text = []
                            assump_text.append(f"- Data {'**berdistribusi normal**' if norm['is_normal'] else '**tidak normal**'}.")
                            assump_text.append(f"- Varians error {'**konstan (Homoskedastis)**' if bp['is_homoscedastic'] else '**tidak konstan (Heteroskedastis)**'}.")
                            assump_text.append(f"- {'**Tidak ada**' if 1.5 < dw_val < 2.5 else '**Ada indikasi**'} masalah autokorelasi.")
                            st.info("💡 **Ringkasan Asumsi**:\n" + "\n".join(assump_text))
                            
                            with st.expander("5. Uji Multikolinieritas (VIF)"):
                                st.dataframe(vif_df, use_container_width=True)
                                st.caption("Jika VIF > 10, indikasi kuat multikolinearitas.")

                            # Evaluasi Visual
                            with st.expander("Plot Visualisasi Asumsi"):
                                import plotly.express as px
                                fig_resid_fit = px.scatter(x=result['y_pred'], y=resid, labels={'x':'Predicted', 'y':'Residuals'}, title="Uji Linieritas & Homoskedastisitas (Pattern)")
                                fig_resid_fit.add_hline(y=0, line_dash="dash", line_color="red")
                                st.plotly_chart(fig_resid_fit, use_container_width=True)

                            # --- B. UJI SIGNIFIKANSI ---
                            st.subheader("B. Uji Signifikansi")
                            
                            # 1. Uji F
                            f_msg = "✅ Model Layak (Signifikan)" if result['f_pvalue'] < 0.05 else "❌ Model Tidak Layak"
                            st.write(f"**1. Uji F (Simultan)**: F-Stat = {result['f_value']:.4f}, Prob(F) = {result['f_pvalue']:.4e} -> {f_msg}")
                            st.caption(f"💡 **Interpretasi**: Secara bersama-sama (simultan), variabel independen **{'berpengaruh signifikan' if result['f_pvalue'] < 0.05 else 'tidak berpengaruh signifikan'}** terhadap **{y_col_multi}**.")
                            
                            # 2. Uji t (Parsial)
                            st.write("**2. Uji t (Parsial/Parameter)**")
                            # Buat dataframe ringkasan t test
                            coefs_data = []
                            # Intercept
                            coefs_data.append({
                                "Variabel": "Intercept (Konstanta)",
                                "Koefisien": result['intercept'],
                                "t-Stat": result['t_values']['const'],
                                "P-Value": result['p_values']['const'],
                                "Kesimpulan": "Signifikan" if result['p_values']['const'] < 0.05 else "Tidak"
                            })
                            # Variabel Lain
                            for col in x_cols_multi:
                                coefs_data.append({
                                    "Variabel": col,
                                    "Koefisien": result['coefficients'][col],
                                    "t-Stat": result['t_values'][col],
                                    "P-Value": result['p_values'][col],
                                    "Kesimpulan": "Signifikan" if result['p_values'][col] < 0.05 else "Tidak"
                                })
                            st.dataframe(pd.DataFrame(coefs_data), use_container_width=True)
                            
                            # Interpretasi T
                            sig_vars = [col for col in x_cols_multi if result['p_values'][col] < 0.05]
                            if sig_vars:
                                st.caption(f"💡 **Interpretasi**: Variabel **{', '.join(sig_vars)}** memiliki pengaruh signifikan terhadap **{y_col_multi}**.")
                            else:
                                st.caption("💡 **Interpretasi**: Tidak ada variabel independen yang berpengaruh signifikan secara parsial.")

                            # --- C. MODEL AKHIR ---
                            st.subheader("C. Model Akhir")
                            equation = f"Y = {result['intercept']:.4f}"
                            for var, koef in result['coefficients'].items(): 
                                equation += f" {'+' if koef>=0 else '-'} {abs(koef):.4f}({var})"
                            st.info(f"**{equation}**")

                            # --- D. KOEFISIEN DETERMINASI ---
                            st.subheader("D. Koefisien Determinasi")
                            cd1, cd2 = st.columns(2)
                            cd1.metric("R-Squared", f"{result['r2']:.4f}")
                            cd2.metric("Adj. R-Squared", f"{result['adj_r2']:.4f}")
                            st.caption(f"💡 **Interpretasi**: Variabel independen yang digunakan mampu menjelaskan **{result['r2']*100:.2f}%** variasi pada variabel **{y_col_multi}**.")

                            # --- E. MSE ---
                            st.subheader("E. MSE (Mean Squared Error)")
                            st.metric("MSE", f"{result['mse']:.4f}")

                        else: st.error("Gagal melakukan analisis. Pastikan data mencukupi.")
                else: st.warning("Pilih minimal satu variabel X.")
            else: st.info("Butuh minimal 3 kolom numerik.")

        # --- TAB 6: FORECASTING ---
        with tab6:
            st.header("🔮 Time Series Forecasting")
            st.markdown("Memprediksi nilai masa depan berdasarkan tren historis.")
            
            # Identifikasi kolom potensial
            all_cols = df.columns.tolist()
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
             
            col_f1, col_f2 = st.columns(2)
            with col_f1: 
                time_col = st.selectbox("Pilih Kolom Waktu (Tahun/Periode):", all_cols, key='fc_time', index=all_cols.index('Tahun') if 'Tahun' in all_cols else 0)
            with col_f2: 
                target_col = st.selectbox("Pilih Target Data:", num_cols, key='fc_target')
            
            col_f3, col_f4 = st.columns(2)
            with col_f3:
                method = st.selectbox("Metode Forecasting:", ["Holt's Linear Trend", "Backpropagation (Neural Network)"], key='fc_method')
            with col_f4:
                steps = st.slider("Jumlah Periode Prediksi:", min_value=1, max_value=10, value=5)
            
            if st.button("Jalankan Forecasting"):
                if time_col and target_col:
                    with st.spinner(f"Menjalankan forecasting dengan {method}..."):
                        if method == "Holt's Linear Trend":
                            fc_res = analysis.perform_forecasting(df, time_col, target_col, periods=steps)
                        else:
                            fc_res = analysis.perform_backpropagation_forecasting(df, time_col, target_col, periods=steps)
                        
                    if fc_res:
                        st.success(f"Berhasil memprediksi {target_col} untuk {steps} periode ke depan menggunakan {method}.")
                        
                        # MSE Metric
                        st.metric("MSE (Mean Squared Error)", f"{fc_res['mse']:.4f}", help="Semakin kecil nilai MSE, semakin akurat model dalam mencocokkan data historis.")
                        
                        st.write(f"**Interpretasi MSE:** Nilai MSE sebesar **{fc_res['mse']:.4f}** menunjukkan rata-rata kuadrat selisih antara data aktual dan estimasi model. Dalam konteks ini, angka tersebut merupakan indikator seberapa besar penyimpangan prediksi terhadap data asli. **Semakin mendekati nol**, semakin presisi model tersebut menangkap pola data masa lalu.")
                        
                        # Plot
                        fig_fc = visualization.plot_forecast(fc_res['history'], fc_res['forecast'], time_col, target_col)
                        st.plotly_chart(fig_fc, use_container_width=True)
                        
                        # Tabel
                        st.subheader("Tabel Hasil Prediksi")
                        st.dataframe(fc_res['forecast'].style.format({'Forecast': "{:.2f}"}), use_container_width=True)
                        
                        # Insight Sederhana
                        end_hist = fc_res['history'][target_col].iloc[-1]
                        end_pred = fc_res['forecast']['Forecast'].iloc[-1]
                        
                        total_growth = ((end_pred - end_hist) / end_hist) * 100
                        st.info(f"💡 **Insight**: Dari data terakhir ({end_hist:.2f}) ke prediksi akhir ({end_pred:.2f}), diprediksi terjadi {'kenaikan' if total_growth > 0 else 'penurunan'} sebesar **{abs(total_growth):.2f}%**.")
                        
                    else:
                        st.error("Gagal melakukan forecasting. Pastikan kolom waktu valid (angka tahun) dan data cukup.")
                else:
                    st.warning("Pilih kolom waktu dan target.")

else:
    st.info("Silakan unggah file di atas untuk memulai analisis.")
