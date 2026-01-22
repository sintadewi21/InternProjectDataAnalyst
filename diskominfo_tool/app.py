import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
from utils import loader, analysis, visualization

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Diskominfo Data Tool",
    page_icon="📊",
    layout="wide"
)

# --- LOAD CUSTOM CSS ---
def local_css(file_name):
    with open(file_name, encoding='utf-8') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("assets/custom_style.css")

# --- INISIALISASI SESSION STATE ---
if 'df' not in st.session_state:
    st.session_state['df'] = None

# --- SIDEBAR ---
with st.sidebar:
    _, col_header, _ = st.columns([0.05, 0.9, 0.05])
    
    with col_header:
        c1, c2, c3 = st.columns([0.9, 1, 2.1])
        
        with c1:
            try:
                st.image("logo_lamongan.png", use_container_width=True)
            except:
                st.write("🏛️")
        
        with c2:
            try:
                st.image("logo.png", use_container_width=True)
            except:
                st.write("🌐")
        
        with c3:
            st.markdown("""
                <div style="line-height: 1.1; color: #1E3A8A; font-weight: 800; font-size: 13px; margin-top: 5px;">
                    DISKOMINFO<br>LAMONGAN
                </div>
            """, unsafe_allow_html=True)
    
    selected = option_menu(
        menu_title=None, 
        options=["Overview", "Descriptive Statistics", "Grouping", "Simple Regression", "Multiple Regression", "Forecasting", "Contact Info"],
        icons=["house", "clipboard-data", "people", "graph-up", "bar-chart-line", "clock-history", "key", "gear"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#FFFFFF"},
            "icon": {"color": "#1E3A8A", "font-size": "16px"}, 
            "nav-link": {
                "font-size": "14px", 
                "text-align": "left", 
                "margin":"0px", 
                "color": "#334155", 
                "--hover-color": "#EFF6FF"
            },
            "nav-link-selected": {
                "background-color": "#EFF6FF", 
                "color": "#2563EB", 
                "font-weight": "600", 
                "border-right": "3px solid #2563EB"
            },
        }
    )
   
# --- HALAMAN ANALISIS ---
# --- 1. OVERVIEW ---
if selected == "Overview":
    # Header
    col_h1, col_h2 = st.columns([2, 1])
    with col_h1:
        st.markdown('<div class="main-header"> OVERVIEW DATA ANALYST TOOLS</div>', unsafe_allow_html=True)
    with col_h2:
        st.markdown(""" """, unsafe_allow_html=True)

    col_upload, col_metrics = st.columns([1.8, 1])
    with col_upload:
        st.markdown('<div class="section-title">Upload File</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Data", type=['csv', 'xlsx', 'xls'], label_visibility="collapsed")
        st.markdown('<div class="upload-footer" style="font-size:12px; margin-top:5px; text-align:center;">Make Sure the Input Data is in .csv, .xlsx, .xls format</div>', unsafe_allow_html=True)
        if uploaded_file is not None:
            df_loaded = loader.load_data(uploaded_file)
            if df_loaded is not None:
                st.session_state['df'] = df_loaded

    # Metrics Box
    with col_metrics:
        st.markdown('<div class="section-title">Overview Data</div>', unsafe_allow_html=True)
        
        rows, cols, missing = 0, 0, 0
        if st.session_state['df'] is not None:
            info = analysis.get_basic_info(st.session_state['df'])
            rows, cols, missing = info['rows'], info['columns'], info['missing_values']
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-item">
                <div style="display:flex; align-items:center;">
                    <div class="metric-icon-circle bg-blue">📄</div>
                    <div class="metric-text">Count of Row</div>
                </div>
                <div class="metric-number">{rows}</div>
            </div>
            <div class="metric-item">
                <div style="display:flex; align-items:center;">
                    <div class="metric-icon-circle bg-blue">📊</div>
                    <div class="metric-text">Count of Column</div>
                </div>
                <div class="metric-number">{cols}</div>
            </div>
            <div class="metric-item">
                <div style="display:flex; align-items:center;">
                    <div class="metric-icon-circle bg-blue">💲</div>
                    <div class="metric-text">Missing Value</div>
                </div>
                <div class="metric-number">{missing}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # DataFrame Table
    if st.session_state['df'] is not None:
        df = st.session_state['df']
        
        st.markdown('<div class="statsdata-box" style="font-color: #FFFFFF;">Data Frame</div>', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, height=280)
        st.markdown('</div>', unsafe_allow_html=True)
        
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.markdown('<div class="statsdata-box" style="font-color: #FFFFFF;">Data Type</div>', unsafe_allow_html=True)
            type_df = df.dtypes.astype(str).reset_index().rename(columns={0:'Type', 'index':'Column'})
            st.dataframe(type_df, use_container_width=True, hide_index=True, height=250)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_b2:
            st.markdown('<div class="statsdata-box" style="font-color: #FFFFFF;">Missing Value</div>', unsafe_allow_html=True)
            info = analysis.get_basic_info(df)
            if info['missing_values'] > 0:
                missing_df = info['missing_per_column'].reset_index().rename(columns={0:'Count', 'index':'Column'})
                st.dataframe(missing_df, use_container_width=True, hide_index=True, height=250)
            else:
                st.info("✓ No missing values")
            st.markdown('</div>', unsafe_allow_html=True)

# --- GLOBAL CHECK FOR OTHER TABS ---
else:
    if st.session_state['df'] is None:
        st.warning("⚠️ Please Go to the **Overview** Menu and Upload Data First.")
    else:
        df = st.session_state['df']

        # --- 2. STATISTIK DESKRIPTIF ---
        if selected == "Descriptive Statistics":
            col_h1, col_h2 = st.columns([2, 1])
            with col_h1:
                st.markdown('<div class="main-header">DESCRIPTIVE STATISTICS</div>', unsafe_allow_html=True)

            st.markdown('<div class="statsdata-box" style="font-color: #FFFFFF;">Summary Statistics of Numerical Column</div>', unsafe_allow_html=True)
            stats = analysis.get_descriptive_stats(df)
            if not stats.empty:
                st.dataframe(stats.style.format("{:.2f}"), use_container_width=True)
            else:
                st.warning("Numerical Columns Not Found.")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="statsdata-box" style="font-color: #FFFFFF;">Frequency Distribution of Categorical Column</div>', unsafe_allow_html=True)
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if cat_cols:
                sel_cat = st.selectbox("Choose Column:", cat_cols)
                freq = analysis.get_frequency_dist(df, sel_cat)
                st.dataframe(freq, use_container_width=True)
            else:
                st.info("Categorical columns not found.")
            st.markdown('</div>', unsafe_allow_html=True)

            # --- OUTLIER DETECTION ---
            st.markdown('<div class="statsdata-box" style="font-color: #FFFFFF;">Outlier Detection</div>', unsafe_allow_html=True)
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            if num_cols:
                sel_outlier = st.selectbox("Select Numeric Column:", num_cols, key='outlier_col')
                
                col_o1, col_o2 = st.columns([1.5, 1])
                
                with col_o1:
                    st.plotly_chart(visualization.plot_box_chart(df, sel_outlier), use_container_width=True)
                    
                with col_o2:
                    outlier_info = analysis.get_outliers_iqr(df, sel_outlier)
                    st.write(f"**Outlier Summary:**")
                    st.write(f"- Lower Bound: `{outlier_info['lower_bound']:.2f}`")
                    st.write(f"- Upper Bound: `{outlier_info['upper_bound']:.2f}`")
                    st.write(f"- Outlier Count: `{outlier_info['count']}`")
                    
                    if outlier_info['count'] > 0:
                        with st.expander("View Outlier Data"):
                            st.dataframe(outlier_info['data'], use_container_width=True)
                    else:
                        st.success("No Outliers Found.")
            else:
                st.info("No Numeric Columns to Analyze.")
            st.markdown('</div>', unsafe_allow_html=True)

        # --- 3. GROUPING ---
        elif selected == "Grouping":
            col_h1, col_h2 = st.columns([2, 1])
            with col_h1:
                st.markdown('<div class="main-header">GROUPING & PIVOT</div>', unsafe_allow_html=True)
            with col_h2:
                st.markdown(""" """, unsafe_allow_html=True)

            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            g_cols = v_cols = agg = None
            if cat_cols and num_cols:
                c1, c2, c3 = st.columns(3)
                with c1:
                    g_cols = st.multiselect("Group By:", cat_cols)
                with c2:
                    v_cols = st.multiselect("Value:", num_cols)
                with c3:
                    agg = st.selectbox("Aggregation:", ["mean", "sum", "count", "min", "max"])
                run_group = st.button("Click here to Run!", key="group-btn", use_container_width=True)
                st.markdown("""
                <style>
                [data-testid="baseButton-secondary"][key="group-btn"] {
                    background: linear-gradient(135deg, #28166f 0%, #0093dd 100%) !important;
                    border: 1px solid #0c124e !important;
                    color: white !important;
                    border-radius: 12px !important;
                    font-weight: 700 !important;
                    font-size: 1.1rem !important;
                    width: 100% !important;
                    margin-top: 18px !important;
                    padding: 0.8rem 0 !important;
                    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
                }
                </style>
                """, unsafe_allow_html=True)
                if run_group:
                    if g_cols and v_cols:
                        res = df.groupby(g_cols)[v_cols].agg(agg).reset_index()
                        html_table = res.to_html(classes='blue-table', escape=False, index=False, float_format="{:.2f}".format)
                        st.markdown(html_table, unsafe_allow_html=True)
                        if len(g_cols)==1 and len(v_cols)==1:
                            st.plotly_chart(visualization.plot_bar_chart(res, g_cols[0], v_cols[0]), use_container_width=True)
                    else:
                        st.warning("Select Group and Value.")
            else:
                st.info("Data Can't Be Used for Grouping.")
            st.markdown('</div>', unsafe_allow_html=True)

        # --- 4. SIMPLE REGRESSION ---
        elif selected == "Simple Regression":
            col_h1, col_h2 = st.columns([2, 1])
            with col_h1:
                st.markdown('<div class="main-header">SIMPLE REGRESSION</div>', unsafe_allow_html=True)
            with col_h2:
                st.markdown(""" """, unsafe_allow_html=True)
                        
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            if len(num_cols) >= 2:
                st.info("💡 **Penjelasan Variabel:**\n- **Variabel X (Independen)**: Variabel yang *mempengaruhi* .\n- **Variabel Y (Dependen)**: Variabel yang *dipengaruhi*.")
                c1, c2 = st.columns(2)
                x = c1.selectbox("X Variable:", num_cols, key='sr_x')
                y = c2.selectbox("Y Variable:", num_cols, index=1, key='sr_y')
                
                if st.button("Calculate Simple Regression", use_container_width=True):
                    res = analysis.perform_linear_regression(df, x, y)
                    if res:
                        st.success(f"Model: Y = {res['intercept']:.2f} + {res['slope']:.2f}X")
                        
                        st.metric("R-Squared (Koefisien Determinasi)", f"{res['r2']:.4f}")
                        st.info(f"💡 *Interpretasi R-Squared:* Nilai *{res['r2']:.4f}* menunjukkan bahwa *{res['r2']*100:.2f}%* variasi dari **{y}** dapat dijelaskan oleh **{x}**. Sisanya sebesar *{100 - res['r2']*100:.2f}%* dijelaskan oleh faktor lain di luar model ini.")

                        st.markdown("### Interpretation of the Slope Coefficient")
                        direction = "naik" if res['slope'] > 0 else "turun"
                        st.write(f"- Setiap kenaikan 1 satuan **{x}**, maka **{y}** diperkirakan akan **{direction}** sebesar **{abs(res['slope']):.4f}** (dengan asumsi faktor lain tetap).")

                        st.plotly_chart(visualization.plot_regression(res['X'], res['y'], res['y_pred'], x, y), use_container_width=True)
                        
                        st.markdown("### Evaluation Model (Actual vs Predicted)")
                        st.plotly_chart(visualization.plot_actual_vs_predicted(res['y'], res['y_pred']), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # --- 5. MULTIPLE REGRESSION ---
        elif selected == "Multiple Regression":
            col_h1, col_h2 = st.columns([2, 1])
            with col_h1:
                st.markdown('<div class="main-header">MULTIPLE REGRESSION</div>', unsafe_allow_html=True)
            with col_h2:
                st.markdown(""" """, unsafe_allow_html=True)
                            
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            if len(num_cols) >= 3:
                st.info("💡 **Penjelasan Variabel:**\n- **Variabel X (Independen)**: Variabel yang *mempengaruhi* .\n- **Variabel Y (Dependen)**: Variabel yang *dipengaruhi*.")
                y = st.selectbox("Target (Y) - Yang Dipengaruhi:", num_cols, key='dr_y')
                xs = st.multiselect("Variables (X) - Yang Mempengaruhi:", [c for c in num_cols if c!=y], key='dr_x')
                
                if st.button("Calculate Multiple Regression", use_container_width=True):
                    if xs:
                        res = analysis.perform_multiple_regression(df, xs, y)
                        if res:
                            st.success(f"R-Squared (Koefisien Determinasi): {res['r2']:.4f}")
                            st.info(f"💡 *Interpretasi R-Squared:* Nilai *{res['r2']:.4f}* menunjukkan bahwa *{res['r2']*100:.2f}%* variasi dari *{y}* dapat dijelaskan oleh variabel independen yang dipilih ({', '.join(xs)}). Sisanya sebesar *{100 - res['r2']*100:.2f}%* dijelaskan oleh faktor lain di luar model ini.")
                            
                            st.markdown("### Interpretation of the Slope Coefficient:")
                            for col, val in res['coefficients'].items():
                                direction = "naik" if val > 0 else "turun"
                                st.write(f"- Setiap kenaikan 1 satuan *{col}, maka *{y}* diperkirakan akan *{direction}* sebesar *{abs(val):.4f}** (dengan asumsi variabel lain tetap).")
                                
                            st.markdown("### Evaluation Model (Actual vs Predicted)")
                            st.plotly_chart(visualization.plot_actual_vs_predicted(res['y_actual'], res['y_pred']), use_container_width=True)
                            
                            with st.expander("Check Coefficient Values"):
                                st.write(res['coefficients'])
            st.markdown('</div>', unsafe_allow_html=True)

        # --- 6. FORECASTING ---
        elif selected == "Forecasting":
            col_h1, col_h2 = st.columns([2, 1])
            with col_h1:
                st.markdown('<div class="main-header">FORECASTING</div>', unsafe_allow_html=True)
            with col_h2:
                st.markdown(""" """, unsafe_allow_html=True)

            all_cols = df.columns.tolist()
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            c1, c2 = st.columns(2)
            time_col = c1.selectbox("Time Column:", all_cols)
            target = c2.selectbox("Target:", num_cols)
            method = st.selectbox("Method:", ["Holt's Linear Trend", "Backpropagation"])
            steps = st.slider("Periods:", 1, 10, 5)
            
            if st.button("Forecast", use_container_width=True):
                if method == "Holt's Linear Trend":
                    res = analysis.perform_forecasting(df, time_col, target, periods=steps)
                else:
                    res = analysis.perform_backpropagation_forecasting(df, time_col, target, periods=steps)
                
                if res:
                    st.plotly_chart(visualization.plot_forecast(res['history'], res['forecast'], time_col, target), use_container_width=True)
                    st.dataframe(res['forecast'])
            st.markdown('</div>', unsafe_allow_html=True)

        # --- 7. CONTACT INFO ---
        elif selected == "Contact Info":
            st.markdown('<div class="main-header">CONTACT INFO & FAQ</div>', unsafe_allow_html=True)
            
            st.markdown("""
                <p style="color: #64748B; font-size: 16px; margin-bottom: 30px;">
                    If you have any questions or concerns, feel free to contact one of the following developers.
                </p>
            """, unsafe_allow_html=True)
            
            col_dev1, col_dev2 = st.columns([1, 1])
            
            subject = "Diskominfo Data Tool Inquiry"
            
            with col_dev1:
                mail_sinta = f"mailto:sinta@example.com?subject={subject.replace(' ', '%20')}"
                st.markdown(f"""
                <div class="contact-card">
                    <div style="font-size: 50px; text-align: center;">👩‍💻</div>
                    <h3 style="text-align: center; color: #1E3A8A; margin-top:7px; margin-bottom: 1px;">Sinta Dewi Rahmawati</h3>
                    <a href="{mail_sinta}" class="email-btn">Contact Here</a>
                </div>
                """, unsafe_allow_html=True)

            with col_dev2:
                mail_zaki = f"mailto:zaki@example.com?subject={subject.replace(' ', '%20')}"
                st.markdown(f"""
                <div class="contact-card">
                    <div style="font-size: 50px; text-align: center;">👨‍💻</div>
                    <h3 style="text-align: center; color: #1E3A8A; margin-top:7px; margin-bottom: 1px;">Ahmad Zaidan Ad Dimasyqie</h3>
                    <a href="{mail_zaki}" class="email-btn">Contact Here</a>
                </div>
                """, unsafe_allow_html=True)