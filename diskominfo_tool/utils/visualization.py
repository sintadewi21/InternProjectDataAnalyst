import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objects as go

def plot_actual_vs_predicted(y_actual, y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_actual, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(y=y_pred, mode='lines', name='Predicted'))
    fig.update_layout(title='Actual vs Predicted', xaxis_title='Index', yaxis_title='Value')
    return fig
def plot_histogram(df, column):
    """
    Membuat histogram interaktif menggunakan Plotly.
    
    Args:
        df: pandas DataFrame.
        column: Nama kolom numerik.
        
    Returns:
        plotly.graph_objects.Figure: Objek figure Plotly.
    """
    if df is None or column not in df.columns:
        return None
        
    fig = px.histogram(
        df, 
        x=column, 
        title=f'Distribusi Data: {column}',
        labels={column: column, 'count': 'Frekuensi'},
        template='plotly_white'
    )
    fig.update_layout(bargap=0.1)
    return fig

def plot_bar_chart(df, x_col, y_col=None):
    """
    Membuat bar chart interaktif. Jika y_col tidak diberikan, 
    akan menghitung frekuensi x_col (untuk data kategorikal).
    
    Args:
        df: pandas DataFrame.
        x_col: Nama kolom untuk sumbu X (kategori).
        y_col: Opsional, nama kolom numerik untuk sumbu Y.
        
    Returns:
        plotly.graph_objects.Figure: Objek figure Plotly.
    """
    if df is None or x_col not in df.columns:
        return None
        
    if y_col:
        fig = px.bar(
            df,
            x=x_col,
            y=y_col,
            title=f'{y_col} berdasarkan {x_col}',
            template='plotly_white'
        )
    else:
        count_data = df[x_col].value_counts().reset_index()
        count_data.columns = [x_col, 'Count']
        
        fig = px.bar(
            count_data,
            x=x_col,
            y='Count',
            title=f'Frekuensi per Kategori: {x_col}',
            text='Count',
            template='plotly_white'
        )
        fig.update_traces(textposition='outside')
        
    return fig



def plot_box_chart(df, column):
    """
    Membuat Box Plot untuk melihat distribusi dan outlier.
    """
    if df is None or column not in df.columns:
        return None
        
    fig = px.box(
        df, 
        y=column,
        title=f'Box Plot: {column}',
        template='plotly_white',
        points="all"
    )
    return fig
    return fig

def plot_regression(x_data, y_data, y_pred, x_label, y_label):
    """
    Membuat Scatter Plot dengan Garis Regresi.
    """
    fig = px.scatter(
        x=x_data.iloc[:,0], 
        y=y_data, 
        labels={'x': x_label, 'y': y_label},
        title=f"Regresi Linear: {y_label} vs {x_label}",
        template='plotly_white',
        opacity=0.65
    )
    
    fig.add_trace(
        go.Scatter(
            x=x_data.iloc[:,0],
            y=y_pred,
            mode='lines',
            name='Garis Prediksi',
            line=dict(color='red', width=3)
        )
    )
    return fig

def plot_forecast(history_df, forecast_df, time_col, target_col):
    """
    Membuat Plot Forecasting: Data Aktual (Solid) + Prediksi (Dash).
    """
    fig = go.Figure()
    
    # 1. Data Historis
    fig.add_trace(go.Scatter(
        x=history_df[time_col],
        y=history_df[target_col],
        mode='lines+markers',
        name='Data Aktual',
        line=dict(color='blue', width=2)
    ))
    
    # 2. Data Forecast
    fig.add_trace(go.Scatter(
        x=forecast_df[time_col],
        y=forecast_df['Forecast'],
        mode='lines+markers',
        name='Prediksi (Forecast)',
        line=dict(color='orange', width=2, dash='dash')
    ))
    
    # Layout
    fig.update_layout(
        title=f"Forecasting: {target_col} vs {time_col}",
        xaxis_title=time_col,
        yaxis_title=target_col,
        template='plotly_white',
        hovermode="x unified"
    )
    
    return fig
