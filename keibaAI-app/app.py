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
    page_title="🐎 競馬AI予測システム",
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
        <h1>🐎 競馬AI予測システム</h1>
        <p>機械学習による競馬レース予測システム - Powered by LightGBM & Optuna</p>
    </div>
    """, unsafe_allow_html=True)
    
    # サイドバー
    with st.sidebar:
        st.markdown("## 🔧 システム設定")
        
        # モード選択
        mode = st.radio(
            "動作モードを選択",
            ["🎯 予測モード", "🏋️ 学習モード", "📊 分析モード"],
            help="予測モード: 学習済みモデルで予測\n学習モード: 新しいモデルを訓練\n分析モード: 結果分析・比較"
        )
        
        st.markdown("---")
        
        # システム状態表示
        st.markdown("### 📋 システム状態")
        if st.session_state.model_loaded:
            st.success("✅ モデル読み込み済み")
        else:
            st.warning("⚠️ モデル未読み込み")
        
        if st.session_state.predictor is not None:
            st.info(f"🤖 予測システム: 準備完了")
        else:
            st.error("❌ 予測システム: 未初期化")
    
    # メインコンテンツ
    if mode == "🎯 予測モード":
        prediction_mode()
    elif mode == "🏋️ 学習モード":
        training_mode()
    elif mode == "📊 分析モード":
        analysis_mode()

def prediction_mode():
    """予測モードのUI"""
    st.markdown("## 🎯 予測モード")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🏇 レース予測</h3>
            <p>学習済みモデルを使用してレースの勝敗を予測します</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("🔄 モデル自動読み込み", type="primary", use_container_width=True):
            load_model_automatically()
    
    # モデル読み込み状態のチェック
    if not st.session_state.model_loaded:
        st.markdown("""
        <div class="info-message">
            <h4>📥 モデル読み込みが必要です</h4>
            <p>予測を開始する前に、学習済みモデルを読み込んでください。</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 利用可能なモデル一覧
        show_available_models()
        return
    
    # 予測実行UI
    st.markdown("### 🎯 レース予測実行")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        race_id = st.text_input(
            "レースID",
            value="202501010101",
            help="netkeibaのレースIDを入力してください（例: 202501010101）"
        )
    
    with col2:
        race_date = st.date_input(
            "レース日付",
            value=datetime(2025, 1, 1),
            help="レース開催日を選択してください"
        )
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 予測実行", type="primary", use_container_width=True):
            execute_prediction(race_id, race_date.strftime('%Y/%m/%d'))
    
    # 予測結果の表示
    if st.session_state.prediction_results is not None:
        display_prediction_results()

def training_mode():
    """学習モードのUI"""
    st.markdown("## 🏋️ 学習モード")
    
    st.markdown("""
    <div class="feature-card">
        <h3>🎓 新規モデル訓練</h3>
        <p>Optunaによるハイパーパラメータ最適化を使用して新しいモデルを訓練します</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 訓練設定
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ⚙️ 訓練設定")
        
        n_trials = st.slider(
            "Optuna試行回数",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="ハイパーパラメータ最適化の試行回数（多いほど精度向上、時間増加）"
        )
        
        data_path = st.text_input(
            "データファイルパス",
            value="data/data/results.pickle",
            help="訓練用データファイルのパス"
        )
        
        use_optuna = st.checkbox(
            "Optuna最適化を使用",
            value=True,
            help="ハイパーパラメータの自動最適化を有効にする"
        )
    
    with col2:
        st.markdown("### 📋 データ情報")
        
        if Path(data_path).exists():
            st.success("✅ データファイル存在確認")
            file_size = Path(data_path).stat().st_size / (1024*1024)
            st.info(f"📦 ファイルサイズ: {file_size:.1f} MB")
        else:
            st.error("❌ データファイルが見つかりません")
        
        # 予想訓練時間
        estimated_time = estimate_training_time(n_trials)
        st.info(f"⏱️ 予想訓練時間: {estimated_time}")
    
    # 訓練実行
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 訓練開始", type="primary", use_container_width=True):
            if Path(data_path).exists():
                execute_training(data_path, n_trials, use_optuna)
            else:
                st.error("❌ データファイルが存在しません")

def analysis_mode():
    """分析モードのUI"""
    st.markdown("## 📊 分析モード")
    
    tabs = st.tabs(["📈 モデル比較", "💰 回収率分析", "📋 予測履歴", "🔍 システム診断"])
    
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
    with st.spinner("🔍 最適なモデルを検索中..."):
        try:
            # 利用可能なモデルを検索
            models_info = list_saved_models('.')
            
            if not models_info:
                st.error("❌ 保存されたモデルが見つかりませんでした")
                return
            
            # 最新のモデルを選択
            latest_model = models_info[0]
            model_path = latest_model['model_file'].replace('_model.pkl', '.pkl')
            
            st.info(f"📂 読み込み中: {latest_model['base_name']}")
            
            # 予測システムを初期化
            st.session_state.predictor = HorseRacingPredictor()
            
            # モデル読み込み
            success = st.session_state.predictor.load_model(model_path)
            
            if success:
                st.session_state.model_loaded = True
                st.success("✅ モデル読み込み完了！")
                st.rerun()
            else:
                st.error("❌ モデル読み込みに失敗しました")
                
        except Exception as e:
            st.error(f"❌ エラー: {str(e)}")

def show_available_models():
    """利用可能なモデル一覧表示"""
    st.markdown("### 📂 利用可能なモデル")
    
    models_info = list_saved_models('.')
    
    if not models_info:
        st.warning("保存されたモデルがありません。学習モードで新しいモデルを作成してください。")
        return
    
    # モデル一覧をデータフレームで表示
    model_df = pd.DataFrame([
        {
            "モデル名": info['base_name'],
            "作成日時": info['modified'].strftime('%Y-%m-%d %H:%M'),
            "サイズ": f"{info['size']:,} bytes",
            "学習済み": "✅" if info.get('is_trained', False) else "❌",
            "完整性": "✅" if info['has_info'] and info['has_state'] else "⚠️"
        }
        for info in models_info
    ])
    
    st.dataframe(model_df, use_container_width=True)
    
    # 個別読み込みボタン
    if st.button("🔽 個別モデル選択"):
        selected_model = st.selectbox(
            "読み込むモデルを選択",
            options=[info['base_name'] for info in models_info]
        )
        
        if st.button("📥 選択モデル読み込み"):
            selected_info = next(info for info in models_info if info['base_name'] == selected_model)
            model_path = selected_info['model_file'].replace('_model.pkl', '.pkl')
            
            with st.spinner("読み込み中..."):
                st.session_state.predictor = HorseRacingPredictor()
                success = st.session_state.predictor.load_model(model_path)
                
                if success:
                    st.session_state.model_loaded = True
                    st.success("✅ モデル読み込み完了！")
                    st.rerun()
                else:
                    st.error("❌ モデル読み込みに失敗しました")

def execute_prediction(race_id, race_date):
    """予測実行"""
    if st.session_state.predictor is None:
        st.error("❌ 予測システムが初期化されていません")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔍 レースデータ取得中...")
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
            status_text.text("✅ 予測完了！")
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
        else:
            st.error("❌ 予測に失敗しました。レースIDまたは日付を確認してください。")
            progress_bar.empty()
            status_text.empty()
            
    except Exception as e:
        st.error(f"❌ エラーが発生しました: {str(e)}")
        progress_bar.empty()
        status_text.empty()

def display_prediction_results():
    """予測結果の表示"""
    results_data = st.session_state.prediction_results
    
    st.markdown("## 🎯 予測結果")
    
    # レース情報
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🏇 レースID</h4>
            <h2>{}</h2>
        </div>
        """.format(results_data['race_id']), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>📅 レース日</h4>
            <h2>{}</h2>
        </div>
        """.format(results_data['race_date']), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>🐎 出走頭数</h4>
            <h2>{}</h2>
        </div>
        """.format(len(results_data['results'])), unsafe_allow_html=True)
    
    with col4:
        top_score = results_data['results'].iloc[0]['score']
        confidence = "高" if top_score > 0.6 else "中" if top_score > 0.4 else "低"
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 予測信頼度</h4>
            <h2>{}</h2>
        </div>
        """.format(confidence), unsafe_allow_html=True)
    
    # 予測結果テーブル
    st.markdown("### 📊 予測ランキング")
    
    results_df = results_data['results'].copy()
    results_df['順位'] = range(1, len(results_df) + 1)
    results_df['推奨度'] = results_df['score'].apply(lambda x: 
        "🔥 強推奨" if x > 0.6 else "📈 推奨" if x > 0.4 else "⚠️ 注意" if x > 0.2 else "❌ 非推奨"
    )
    
    # 表示用データフレーム
    display_df = results_df[['順位', '馬番', 'score', '推奨度']].copy()
    display_df['スコア'] = display_df['score'].round(4)
    display_df = display_df[['順位', '馬番', 'スコア', '推奨度']]
    
    st.dataframe(
        display_df.head(10),
        use_container_width=True,
        hide_index=True
    )
    
    # 予測スコア分布グラフ
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 スコア分布")
        fig = px.histogram(
            results_df,
            x='score',
            nbins=15,
            title="予測スコア分布",
            color_discrete_sequence=['#667eea']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 上位馬スコア")
        top5 = results_df.head(5)
        fig = px.bar(
            top5,
            x='馬番',
            y='score',
            title="上位5頭の予測スコア",
            color='score',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # 投資推奨
    st.markdown("### 💰 投資推奨")
    
    top_horse = results_df.iloc[0]
    if top_horse['score'] > 0.6:
        recommendation = "🔥 **強力推奨**: 高確率で好走が期待できます"
        color = "success"
    elif top_horse['score'] > 0.4:
        recommendation = "📈 **推奨**: 中程度の確率で好走が期待できます"
        color = "info"
    else:
        recommendation = "⚠️ **注意**: 確率は低めです。慎重な判断をお勧めします"
        color = "warning"
    
    if color == "success":
        st.success(recommendation)
    elif color == "info":
        st.info(recommendation)
    else:
        st.warning(recommendation)
    
    # 詳細情報の表示（展開可能）
    with st.expander("📋 詳細情報"):
        st.markdown("#### 🐎 全出走馬データ")
        st.dataframe(results_df, use_container_width=True)
        
        # 結果をJSONでダウンロード可能にする
        json_data = {
            'race_id': results_data['race_id'],
            'race_date': results_data['race_date'],
            'predictions': results_df.to_dict('records'),
            'timestamp': results_data['timestamp'].isoformat()
        }
        
        st.download_button(
            label="📥 結果をJSONダウンロード",
            data=json.dumps(json_data, ensure_ascii=False, indent=2),
            file_name=f"prediction_{results_data['race_id']}.json",
            mime="application/json"
        )

def execute_training(data_path, n_trials, use_optuna):
    """訓練実行"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔧 予測システム初期化中...")
        st.session_state.predictor = HorseRacingPredictor()
        progress_bar.progress(10)
        
        status_text.text("📚 データ読み込み中...")
        progress_bar.progress(20)
        
        if use_optuna:
            status_text.text(f"🚀 Optuna最適化開始 ({n_trials}試行)...")
            progress_bar.progress(30)
            
            # 訓練実行
            success = st.session_state.predictor.train_model(data_path, n_trials=n_trials)
        else:
            status_text.text("🏋️ 従来手法で訓練中...")
            success = st.session_state.predictor.trainer.train(data_path)
            st.session_state.predictor.is_trained = True
        
        progress_bar.progress(80)
        
        if success:
            status_text.text("💾 モデル保存中...")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"trained_model_{timestamp}.pkl"
            
            save_success = st.session_state.predictor.save_model(model_name)
            progress_bar.progress(100)
            
            if save_success:
                st.session_state.model_loaded = True
                status_text.text("✅ 訓練完了！")
                
                st.success("🎉 モデル訓練が完了しました！")
                st.info(f"📁 保存ファイル: {model_name}")
                
                # 訓練結果の表示
                if hasattr(st.session_state.predictor.trainer, 'performance_metrics'):
                    metrics = st.session_state.predictor.trainer.performance_metrics
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("🎯 検証精度", f"{metrics.get('val_accuracy', 0):.4f}")
                    with col2:
                        st.metric("📈 AUC", f"{metrics.get('val_auc', 0):.4f}")
                    with col3:
                        st.metric("⚖️ F1スコア", f"{metrics.get('val_f1', 0):.4f}")
                    with col4:
                        st.metric("🎪 再現率", f"{metrics.get('val_recall', 0):.4f}")
                
            else:
                st.error("❌ モデル保存に失敗しました")
        else:
            st.error("❌ 訓練に失敗しました")
        
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        st.error(f"❌ 訓練中にエラーが発生しました: {str(e)}")
        progress_bar.empty()
        status_text.empty()

def estimate_training_time(n_trials):
    """訓練時間の推定"""
    # 簡易的な時間推定（試行回数に基づく）
    base_time = 2  # 基本時間（分）
    trial_time = n_trials * 0.5  # 試行当たりの時間（分）
    total_minutes = base_time + trial_time
    
    if total_minutes < 60:
        return f"{total_minutes:.0f}分"
    else:
        hours = total_minutes // 60
        minutes = total_minutes % 60
        return f"{hours:.0f}時間{minutes:.0f}分"

def model_comparison_tab():
    """モデル比較タブ"""
    st.markdown("### 📈 モデル性能比較")
    
    models_info = list_saved_models('.')
    
    if not models_info:
        st.warning("比較可能なモデルがありません")
        return
    
    # モデル選択
    selected_models = st.multiselect(
        "比較するモデルを選択",
        options=[info['base_name'] for info in models_info],
        default=[info['base_name'] for info in models_info[:3]]  # 最大3つ
    )
    
    if len(selected_models) < 2:
        st.info("比較には2つ以上のモデルを選択してください")
        return
    
    # 比較実行
    if st.button("🔍 比較実行"):
        with st.spinner("モデル比較中..."):
            # ここで実際の比較ロジックを実装
            # デモ用のダミーデータ
            comparison_data = []
            for model_name in selected_models:
                comparison_data.append({
                    'モデル名': model_name,
                    '精度': np.random.uniform(0.75, 0.85),
                    'AUC': np.random.uniform(0.80, 0.90),
                    'F1スコア': np.random.uniform(0.70, 0.80),
                    '訓練時間': np.random.randint(30, 120)
                })
            
            df = pd.DataFrame(comparison_data)
            
            # 比較結果表示
            st.dataframe(df, use_container_width=True)
            
            # グラフ表示
            fig = px.bar(
                df,
                x='モデル名',
                y=['精度', 'AUC', 'F1スコア'],
                title="モデル性能比較",
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)

def return_analysis_tab():
    """回収率分析タブ"""
    st.markdown("### 💰 回収率分析")
    
    if not st.session_state.model_loaded:
        st.warning("回収率分析にはモデルの読み込みが必要です")
        return
    
    if st.button("📊 回収率シミュレーション実行"):
        with st.spinner("シミュレーション実行中..."):
            try:
                # 回収率シミュレーション実行
                simulation_results = st.session_state.predictor.simulate_returns(
                    save_results=True
                )
                
                if simulation_results:
                    st.success("✅ シミュレーション完了")
                    
                    # 結果表示
                    fukusho_gain = simulation_results['fukusho_gain']
                    tansho_gain = simulation_results['tansho_gain']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 🥉 複勝回収率")
                        fig = px.line(
                            x=fukusho_gain.index,
                            y=fukusho_gain['return_rate'],
                            title="複勝回収率推移"
                        )
                        fig.add_hline(y=1.0, line_dash="dash", line_color="red")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### 🥇 単勝回収率")
                        fig = px.line(
                            x=tansho_gain.index,
                            y=tansho_gain['return_rate'],
                            title="単勝回収率推移"
                        )
                        fig.add_hline(y=1.0, line_dash="dash", line_color="red")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # 最適閾値の表示
                    best_fukusho = fukusho_gain.loc[fukusho_gain['return_rate'].idxmax()]
                    best_tansho = tansho_gain.loc[tansho_gain['return_rate'].idxmax()]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "🥉 複勝最高回収率",
                            f"{best_fukusho['return_rate']:.3f}",
                            f"閾値: {best_fukusho.name:.3f}"
                        )
                    
                    with col2:
                        st.metric(
                            "🥇 単勝最高回収率",
                            f"{best_tansho['return_rate']:.3f}",
                            f"閾値: {best_tansho.name:.3f}"
                        )
                
            except Exception as e:
                st.error(f"❌ シミュレーションエラー: {str(e)}")

def prediction_history_tab():
    """予測履歴タブ"""
    st.markdown("### 📋 予測履歴")
    
    # 結果ディレクトリから履歴を読み込み
    results_dir = Path("simulation_results")
    
    if not results_dir.exists():
        st.info("予測履歴がありません")
        return
    
    # JSONファイルを検索
    json_files = list(results_dir.glob("prediction_*.json"))
    
    if not json_files:
        st.info("予測履歴が見つかりません")
        return
    
    # 履歴一覧
    history_data = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            history_data.append({
                'ファイル': json_file.name,
                'レースID': data.get('race_id', 'N/A'),
                '予測時刻': data.get('timestamp', 'N/A'),
                'レース日': data.get('race_info', {}).get('date', 'N/A')
            })
        except:
            continue
    
    if history_data:
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("有効な予測履歴がありません")

def system_diagnostics_tab():
    """システム診断タブ"""
    st.markdown("### 🔍 システム診断")
    
    # システム状態チェック
    checks = []
    
    # 予測システム
    if st.session_state.predictor is not None:
        checks.append({"項目": "予測システム", "状態": "✅ 正常", "詳細": "初期化済み"})
    else:
        checks.append({"項目": "予測システム", "状態": "❌ 異常", "詳細": "未初期化"})
    
    # モデル読み込み
    if st.session_state.model_loaded:
        checks.append({"項目": "モデル読み込み", "状態": "✅ 正常", "詳細": "読み込み完了"})
    else:
        checks.append({"項目": "モデル読み込み", "状態": "❌ 異常", "詳細": "未読み込み"})
    
    # データファイル
    data_files = [
        "data/data/results.pickle",
        "data/data/horse_results.pickle",
        "data/data/peds.pickle"
    ]
    
    for file_path in data_files:
        if Path(file_path).exists():
            checks.append({"項目": f"データファイル ({file_path})", "状態": "✅ 存在", "詳細": f"サイズ: {Path(file_path).stat().st_size:,} bytes"})
        else:
            checks.append({"項目": f"データファイル ({file_path})", "状態": "⚠️ 不在", "詳細": "ファイルが見つかりません"})
    
    # 結果表示
    df = pd.DataFrame(checks)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # メモリ使用量などの追加情報
    st.markdown("#### 💻 システム情報")
    
    import psutil
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("💾 メモリ使用率", f"{psutil.virtual_memory().percent:.1f}%")
    
    with col2:
        st.metric("💽 ディスク使用率", f"{psutil.disk_usage('.').percent:.1f}%")
    
    with col3:
        st.metric("🖥️ CPU使用率", f"{psutil.cpu_percent():.1f}%")

if __name__ == "__main__":
    main()