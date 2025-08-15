import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
import base64
import io

# 既存の競馬AI予測システムをインポート
from horse_racing_ai_refactored import (
    HorseRacingPredictor,
    ResultsAnalyzer,
    list_saved_models,
    load_best_model
)

# ページ設定
st.set_page_config(
    page_title="🐎 Horse Racing AI Prediction System",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
    }
    
    .info-message {
        background: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #bee5eb;
    }
</style>
""", unsafe_allow_html=True)

# セッションステートの初期化
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'training_progress' not in st.session_state:
    st.session_state.training_progress = 0
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

def main():
    # ヘッダー
    st.markdown("""
    <div class="main-header">
        <h1>🐎 Horse Racing AI Prediction System</h1>
        <p>Machine Learning Horse Race Prediction System - Powered by LightGBM & Optuna</p>
    </div>
    """, unsafe_allow_html=True)
    
    # サイドバー
    with st.sidebar:
        st.markdown("## 🔧 System Settings")
        
        # モード選択
        mode = st.radio(
            "Select Operation Mode",
            ["🎯 Prediction Mode", "🏋️ Training Mode", "📊 Analysis Mode"],
            help="Prediction Mode: Predict with trained model\nTraining Mode: Train new model\nAnalysis Mode: Result analysis & comparison"
        )
        
        st.markdown("---")
        
        # システム状態表示
        st.markdown("### 📋 System Status")
        if st.session_state.model_loaded:
            st.success("✅ Model Loaded")
        else:
            st.warning("⚠️ Model Not Loaded")
        
        if st.session_state.predictor is not None:
            st.info(f"🤖 Prediction System: Ready")
        else:
            st.error("❌ Prediction System: Not Initialized")
    
    # メインコンテンツ
    if mode == "🎯 Prediction Mode":
        prediction_mode()
    elif mode == "🏋️ Training Mode":
        training_mode()
    elif mode == "📊 Analysis Mode":
        analysis_mode()

def prediction_mode():
    """予測モードのUI"""
    st.markdown("## 🎯 Prediction Mode")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🏇 Race Prediction</h3>
            <p>Predict race outcomes using trained models</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("🔄 Auto Load Model", type="primary", use_container_width=True):
            load_model_automatically()
    
    # モデル読み込み状態のチェック
    if not st.session_state.model_loaded:
        st.markdown("""
        <div class="info-message">
            <h4>📥 Model Loading Required</h4>
            <p>Please load a trained model before starting predictions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 利用可能なモデル一覧
        show_available_models()
        return
    
    # 予測実行UI
    st.markdown("### 🎯 Execute Race Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        race_id = st.text_input(
            "Race ID",
            value="202501010101",
            help="Enter netkeiba race ID (e.g., 202501010101)"
        )
    
    with col2:
        race_date = st.date_input(
            "Race Date",
            value=datetime(2025, 1, 1),
            help="Select race date"
        )
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Execute Prediction", type="primary", use_container_width=True):
            execute_prediction(race_id, race_date.strftime('%Y/%m/%d'))
    
    # 予測結果の表示
    if st.session_state.prediction_results is not None:
        display_prediction_results()

def training_mode():
    """学習モードのUI"""
    st.markdown("## 🏋️ Training Mode")
    
    st.markdown("""
    <div class="feature-card">
        <h3>🎓 New Model Training</h3>
        <p>Train new model using Optuna hyperparameter optimization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 訓練設定
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ⚙️ Training Settings")
        
        n_trials = st.slider(
            "Optuna Trials",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="Number of hyperparameter optimization trials (more = better accuracy, longer time)"
        )
        
        data_path = st.text_input(
            "Data File Path",
            value="data/data/results.pickle",
            help="Path to training data file"
        )
        
        use_optuna = st.checkbox(
            "Use Optuna Optimization",
            value=True,
            help="Enable automatic hyperparameter optimization"
        )
    
    with col2:
        st.markdown("### 📋 Data Information")
        
        if Path(data_path).exists():
            st.success("✅ Data File Found")
            file_size = Path(data_path).stat().st_size / (1024*1024)
            st.info(f"📦 File Size: {file_size:.1f} MB")
        else:
            st.error("❌ Data File Not Found")
        
        # 予想訓練時間
        estimated_time = estimate_training_time(n_trials)
        st.info(f"⏱️ Estimated Training Time: {estimated_time}")
    
    # 訓練実行
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            if Path(data_path).exists():
                execute_training(data_path, n_trials, use_optuna)
            else:
                st.error("❌ Data file does not exist")

def analysis_mode():
    """分析モードのUI"""
    st.markdown("## 📊 Analysis Mode")
    
    tabs = st.tabs(["📈 Model Comparison", "💰 Return Analysis", "📋 Prediction History", "🔍 System Diagnostics"])
    
    with tabs[0]:
        model_comparison_tab()
    
    with tabs[1]:
        return_analysis_tab()
    
    with tabs[2]:
        prediction_history_tab()
    
    with tabs[3]:
        system_diagnostics_tab()

def load_model_automatically():
    """モデルの自動読み込み"""
    with st.spinner("🔍 Searching for optimal model..."):
        try:
            # 利用可能なモデルを検索
            models_info = list_saved_models('.')
            
            if not models_info:
                st.error("❌ No saved models found")
                return
            
            # 最新のモデルを選択
            latest_model = models_info[0]
            model_path = latest_model['model_file'].replace('_model.pkl', '.pkl')
            
            st.info(f"📂 Loading: {latest_model['base_name']}")
            
            # 予測システムを初期化
            st.session_state.predictor = HorseRacingPredictor()
            
            # モデル読み込み
            success = st.session_state.predictor.load_model(model_path)
            
            if success:
                st.session_state.model_loaded = True
                st.success("✅ Model loaded successfully!")
                st.rerun()
            else:
                st.error("❌ Failed to load model")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

def show_available_models():
    """利用可能なモデル一覧表示"""
    st.markdown("### 📂 Available Models")
    
    models_info = list_saved_models('.')
    
    if not models_info:
        st.warning("No saved models available. Please create a new model in Training Mode.")
        return
    
    # モデル一覧をデータフレームで表示
    model_df = pd.DataFrame([
        {
            "Model Name": info['base_name'],
            "Created": info['modified'].strftime('%Y-%m-%d %H:%M'),
            "Size": f"{info['size']:,} bytes",
            "Trained": "✅" if info.get('is_trained', False) else "❌",
            "Integrity": "✅" if info['has_info'] and info['has_state'] else "⚠️"
        }
        for info in models_info
    ])
    
    st.dataframe(model_df, use_container_width=True)
    
    # 個別読み込みボタン
    if st.button("🔽 Select Individual Model"):
        selected_model = st.selectbox(
            "Select model to load",
            options=[info['base_name'] for info in models_info]
        )
        
        if st.button("📥 Load Selected Model"):
            selected_info = next(info for info in models_info if info['base_name'] == selected_model)
            model_path = selected_info['model_file'].replace('_model.pkl', '.pkl')
            
            with st.spinner("Loading..."):
                st.session_state.predictor = HorseRacingPredictor()
                success = st.session_state.predictor.load_model(model_path)
                
                if success:
                    st.session_state.model_loaded = True
                    st.success("✅ Model loaded successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to load model")

def execute_prediction(race_id, race_date):
    """予測実行"""
    if st.session_state.predictor is None:
        st.error("❌ Prediction system not initialized")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔍 Fetching race data...")
        progress_bar.progress(25)
        
        # 予測実行
        results = st.session_state.predictor.predict_race_live(race_id, race_date)
        progress_bar.progress(75)
        
        if results is not None:
            st.session_state.prediction_results = {
                'race_id': race_id,
                'race_date': race_date,
                'results': results,
                'timestamp': datetime.now()
            }
            
            progress_bar.progress(100)
            status_text.text("✅ Prediction completed!")
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
        else:
            st.error("❌ Prediction failed. Please check race ID or date.")
            progress_bar.empty()
            status_text.empty()
            
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        progress_bar.empty()
        status_text.empty()

def display_prediction_results():
    """予測結果の表示"""
    results_data = st.session_state.prediction_results
    
    st.markdown("## 🎯 Prediction Results")
    
    # レース情報
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🏇 Race ID</h4>
            <h2>{}</h2>
        </div>
        """.format(results_data['race_id']), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>📅 Race Date</h4>
            <h2>{}</h2>
        </div>
        """.format(results_data['race_date']), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>🐎 Runners</h4>
            <h2>{}</h2>
        </div>
        """.format(len(results_data['results'])), unsafe_allow_html=True)
    
    with col4:
        top_score = results_data['results'].iloc[0]['score']
        confidence = "High" if top_score > 0.6 else "Medium" if top_score > 0.4 else "Low"
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Confidence</h4>
            <h2>{}</h2>
        </div>
        """.format(confidence), unsafe_allow_html=True)
    
    # 予測結果テーブル
    st.markdown("### 📊 Prediction Ranking")
    
    results_df = results_data['results'].copy()
    results_df['Rank'] = range(1, len(results_df) + 1)
    results_df['Recommendation'] = results_df['score'].apply(lambda x: 
        "🔥 Strong Buy" if x > 0.6 else "📈 Buy" if x > 0.4 else "⚠️ Caution" if x > 0.2 else "❌ Avoid"
    )
    
    # 表示用データフレーム
    display_df = results_df[['Rank', '馬番', 'score', 'Recommendation']].copy()
    display_df['Score'] = display_df['score'].round(4)
    display_df = display_df[['Rank', '馬番', 'Score', 'Recommendation']]
    
    st.dataframe(
        display_df.head(10),
        use_container_width=True,
        hide_index=True
    )
    
    # 予測スコア分布グラフ
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Score Distribution")
        fig = px.histogram(
            results_df,
            x='score',
            nbins=15,
            title="Prediction Score Distribution",
            color_discrete_sequence=['#667eea']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Top Horses Score")
        top5 = results_df.head(5)
        fig = px.bar(
            top5,
            x='馬番',
            y='score',
            title="Top 5 Horses Prediction Scores",
            color='score',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # 投資推奨
    st.markdown("### 💰 Investment Recommendation")
    
    top_horse = results_df.iloc[0]
    if top_horse['score'] > 0.6:
        recommendation = "🔥 **Strong Recommendation**: High probability of good performance expected"
        color = "success"
    elif top_horse['score'] > 0.4:
        recommendation = "📈 **Recommendation**: Moderate probability of good performance expected"
        color = "info"
    else:
        recommendation = "⚠️ **Caution**: Low probability. Careful judgment recommended"
        color = "warning"
    
    if color == "success":
        st.success(recommendation)
    elif color == "info":
        st.info(recommendation)
    else:
        st.warning(recommendation)
    
    # 詳細情報の表示（展開可能）
    with st.expander("📋 Detailed Information"):
        st.markdown("#### 🐎 All Runners Data")
        st.dataframe(results_df, use_container_width=True)
        
        # 結果をJSONでダウンロード可能にする
        json_data = {
            'race_id': results_data['race_id'],
            'race_date': results_data['race_date'],
            'predictions': results_df.to_dict('records'),
            'timestamp': results_data['timestamp'].isoformat()
        }
        
        st.download_button(
            label="📥 Download Results as JSON",
            data=json.dumps(json_data, ensure_ascii=False, indent=2),
            file_name=f"prediction_{results_data['race_id']}.json",
            mime="application/json"
        )

def execute_training(data_path, n_trials, use_optuna):
    """訓練実行"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔧 Initializing prediction system...")
        st.session_state.predictor = HorseRacingPredictor()
        progress_bar.progress(10)
        
        status_text.text("📚 Loading data...")
        progress_bar.progress(20)
        
        if use_optuna:
            status_text.text(f"🚀 Starting Optuna optimization ({n_trials} trials)...")
            progress_bar.progress(30)
            
            # 訓練実行
            success = st.session_state.predictor.train_model(data_path, n_trials=n_trials)
        else:
            status_text.text("🏋️ Training with conventional method...")
            success = st.session_state.predictor.trainer.train(data_path)
            st.session_state.predictor.is_trained = True
        
        progress_bar.progress(80)
        
        if success:
            status_text.text("💾 Saving model...")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"trained_model_{timestamp}.pkl"
            
            save_success = st.session_state.predictor.save_model(model_name)
            progress_bar.progress(100)
            
            if save_success:
                st.session_state.model_loaded = True
                status_text.text("✅ Training completed!")
                
                st.success("🎉 Model training completed!")
                st.info(f"📁 Saved file: {model_name}")
                
                # 訓練結果の表示
                if hasattr(st.session_state.predictor.trainer, 'performance_metrics'):
                    metrics = st.session_state.predictor.trainer.performance_metrics
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("🎯 Validation Accuracy", f"{metrics.get('val_accuracy', 0):.4f}")
                    with col2:
                        st.metric("📈 AUC", f"{metrics.get('val_auc', 0):.4f}")
                    with col3:
                        st.metric("⚖️ F1 Score", f"{metrics.get('val_f1', 0):.4f}")
                    with col4:
                        st.metric("🎪 Recall", f"{metrics.get('val_recall', 0):.4f}")
                
            else:
                st.error("❌ Failed to save model")
        else:
            st.error("❌ Training failed")
        
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        st.error(f"❌ Error during training: {str(e)}")
        progress_bar.empty()
        status_text.empty()

def estimate_training_time(n_trials):
    """訓練時間の推定"""
    # 簡易的な時間推定（試行回数に基づく）
    base_time = 2  # 基本時間（分）
    trial_time = n_trials * 0.5  # 試行当たりの時間（分）
    total_minutes = base_time + trial_time
    
    if total_minutes < 60:
        return f"{total_minutes:.0f} minutes"
    else:
        hours = total_minutes // 60
        minutes = total_minutes % 60
        return f"{hours:.0f}h {minutes:.0f}m"

def model_comparison_tab():
    """モデル比較タブ"""
    st.markdown("### 📈 Model Performance Comparison")
    
    models_info = list_saved_models('.')
    
    if not models_info:
        st.warning("No models available for comparison")
        return
    
    # モデル選択
    selected_models = st.multiselect(
        "Select models to compare",
        options=[info['base_name'] for info in models_info],
        default=[info['base_name'] for info in models_info[:3]]  # 最大3つ
    )
    
    if len(selected_models) < 2:
        st.info("Please select at least 2 models for comparison")
        return
    
    # 比較実行
    if st.button("🔍 Execute Comparison"):
        with st.spinner("Comparing models..."):
            # ここで実際の比較ロジックを実装
            # デモ用のダミーデータ
            comparison_data = []
            for model_name in selected_models:
                comparison_data.append({
                    'Model Name': model_name,
                    'Accuracy': np.random.uniform(0.75, 0.85),
                    'AUC': np.random.uniform(0.80, 0.90),
                    'F1 Score': np.random.uniform(0.70, 0.80),
                    'Training Time (min)': np.random.randint(30, 120)
                })
            
            df = pd.DataFrame(comparison_data)
            
            # 比較結果表示
            st.dataframe(df, use_container_width=True)
            
            # グラフ表示
            fig = px.bar(
                df,
                x='Model Name',
                y=['Accuracy', 'AUC', 'F1 Score'],
                title="Model Performance Comparison",
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)

def return_analysis_tab():
    """回収率分析タブ"""
    st.markdown("### 💰 Return Rate Analysis")
    
    if not st.session_state.model_loaded:
        st.warning("Model loading required for return rate analysis")
        return
    
    if st.button("📊 Execute Return Rate Simulation"):
        with st.spinner("Running simulation..."):
            try:
                # 回収率シミュレーション実行
                simulation_results = st.session_state.predictor.simulate_returns(
                    save_results=True
                )
                
                if simulation_results:
                    st.success("✅ Simulation completed")
                    
                    # 結果表示
                    fukusho_gain = simulation_results['fukusho_gain']
                    tansho_gain = simulation_results['tansho_gain']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 🥉 Place Return Rate")
                        fig = px.line(
                            x=fukusho_gain.index,
                            y=fukusho_gain['return_rate'],
                            title="Place Return Rate Trend"
                        )
                        fig.add_hline(y=1.0, line_dash="dash", line_color="red")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### 🥇 Win Return Rate")
                        fig = px.line(
                            x=tansho_gain.index,
                            y=tansho_gain['return_rate'],
                            title="Win Return Rate Trend"
                        )
                        fig.add_hline(y=1.0, line_dash="dash", line_color="red")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # 最適閾値の表示
                    best_fukusho = fukusho_gain.loc[fukusho_gain['return_rate'].idxmax()]
                    best_tansho = tansho_gain.loc[tansho_gain['return_rate'].idxmax()]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "🥉 Best Place Return Rate",
                            f"{best_fukusho['return_rate']:.3f}",
                            f"Threshold: {best_fukusho.name:.3f}"
                        )
                    
                    with col2:
                        st.metric(
                            "🥇 Best Win Return Rate",
                            f"{best_tansho['return_rate']:.3f}",
                            f"Threshold: {best_tansho.name:.3f}"
                        )
                
            except Exception as e:
                st.error(f"❌ Simulation error: {str(e)}")

def prediction_history_tab():
    """予測履歴タブ"""
    st.markdown("### 📋 Prediction History")
    
    # 結果ディレクトリから履歴を読み込み
    results_dir = Path("simulation_results")
    
    if not results_dir.exists():
        st.info("No prediction history available")
        return
    
    # JSONファイルを検索
    json_files = list(results_dir.glob("prediction_*.json"))
    
    if not json_files:
        st.info("No prediction history found")
        return
    
    # 履歴一覧
    history_data = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            history_data.append({
                'File': json_file.name,
                'Race ID': data.get('race_id', 'N/A'),
                'Prediction Time': data.get('timestamp', 'N/A'),
                'Race Date': data.get('race_info', {}).get('date', 'N/A')
            })
        except:
            continue
    
    if history_data:
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No valid prediction history available")

def system_diagnostics_tab():
    """システム診断タブ"""
    st.markdown("### 🔍 System Diagnostics")
    
    # システム状態チェック
    checks = []
    
    # 予測システム
    if st.session_state.predictor is not None:
        checks.append({"Item": "Prediction System", "Status": "✅ Normal", "Details": "Initialized"})
    else:
        checks.append({"Item": "Prediction System", "Status": "❌ Error", "Details": "Not initialized"})
    
    # モデル読み込み
    if st.session_state.model_loaded:
        checks.append({"Item": "Model Loading", "Status": "✅ Normal", "Details": "Loaded"})
    else:
        checks.append({"Item": "Model Loading", "Status": "❌ Error", "Details": "Not loaded"})
    
    # データファイル
    data_files = [
        "data/data/results.pickle",
        "data/data/horse_results.pickle",
        "data/data/peds.pickle"
    ]
    
    for file_path in data_files:
        if Path(file_path).exists():
            checks.append({"Item": f"Data File ({file_path})", "Status": "✅ Exists", "Details": f"Size: {Path(file_path).stat().st_size:,} bytes"})
        else:
            checks.append({"Item": f"Data File ({file_path})", "Status": "⚠️ Missing", "Details": "File not found"})
    
    # 結果表示
    df = pd.DataFrame(checks)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # メモリ使用量などの追加情報
    st.markdown("#### 💻 System Information")
    
    import psutil
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("💾 Memory Usage", f"{psutil.virtual_memory().percent:.1f}%")
    
    with col2:
        st.metric("💽 Disk Usage", f"{psutil.disk_usage('.').percent:.1f}%")
    
    with col3:
        st.metric("🖥️ CPU Usage", f"{psutil.cpu_percent():.1f}%")

if __name__ == "__main__":
    main()