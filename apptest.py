import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
import joblib
from datetime import datetime
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Layer, TimeDistributed, Dense, LSTM, Dropout, Lambda, Input
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Bidirectional
import os
from streamlit_lottie import st_lottie
import json
from pytz import timezone
import pydeck as pdk
import random
from bs4 import BeautifulSoup
import re

LOOKBACK = 48
PRED_STEPS = 6

# ========== 外部モデル・スケーラーのダウンロード ==========
def download_file(url, local_path):
    if os.path.exists(local_path):
        return local_path
    try:
        st.info(f"{os.path.basename(local_path)} をダウンロード中...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        st.success(f"{os.path.basename(local_path)} をダウンロードしました。")
        return local_path
    except Exception as e:
        st.error(f"{os.path.basename(local_path)} のダウンロード失敗: {e}")
        st.stop()

# --- 外部ストレージの直リンクに置き換えてください ---
FEATURES = [
    'temp', 'pressure', 'humidity', 'wind', 'clouds', 'precipitation',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'temp_diff', 'pressure_diff', 'season', 'is_day'
]
MODEL_URL = "https://drive.google.com/uc?id=1-V5CaLAgd1Z8btpd4xUth3wtb5vW2HvJ&export=download"
SCALER_URL = "https://drive.google.com/uc?id=1VKiGQ8ONddH5lRwbpJsBfL2PztHw4ruU&export=download"
RF_MODEL_URLS = {'clouds': "https://drive.google.com/uc?id=1zjPRqc7NV6g283WaaBegEBkSMBaO9oJ4&export=download", 'hour_cos': "https://drive.google.com/uc?id=1xZ62zcxqCvjDCA20ELEa-Zs1kdr5iyFO&export=download", 'hour_sin': "https://drive.google.com/uc?id=1SNafTruumh4a-A3O3c0ahjrR9kCr5Z9r&export=download", 'humidity': "https://drive.google.com/uc?id=1rh5RNTkqNVWXh0y63J1DS6L6QeJqUbMV&export=download", 'is_day': "https://drive.google.com/uc?id=1JBdJvu5fslh7oCuGdzQ59EJAnpUPqN__&export=download", 'month_cos': "https://drive.google.com/uc?id=1CQPC9mUVO8vV1FlhBDMbl_xQT-JDTwil&export=download", 'month_sin': "https://drive.google.com/uc?id=1W2nJnWsoLDS-3Nqh5u85jXv0wJ84vhx6&export=download", 'pressure': "https://drive.google.com/uc?id=1GHb7krtQnkBAb8Gn_eInfQ5PwbrPIC9l&export=download", 'season': "https://drive.google.com/uc?id=1In_LdhgGapHMboTFRXrQGCVxe6WT6sfh&export=download", 'temp': "https://drive.google.com/uc?id=1KBoWjL34_kBqQWDUnxR0wCZZqiqyBlZY&export=download", 'wind': "https://drive.google.com/uc?id=1edVBoKSxZqp1nVy9nQvgC1ZT3QW66Tjz&export=download"}
XGB_MODEL_URLS = {'clouds': "https://drive.google.com/uc?id=1b6NyCe-UPfiu_qOQtLnq2s4YJqbdXXu3&export=download", 'hour_cos': "https://drive.google.com/uc?id=1BHE3X6OY3Wljjh4iD6dRN2RHDjCxadxj&export=download", 'hour_sin': "https://drive.google.com/uc?id=1HWzkuiMnhrnoRiejqmpFr0w_6OVGpAS6&export=download", 'humidity': "https://drive.google.com/uc?id=1r3ksKNe2GQbEXtvQIypONiFGeN8jIIS-&export=download", 'is_day': "https://drive.google.com/uc?id=1Lw7ONy0ZPjHOO_QLw29cqbig4OMlL-Jk&export=download", 'month_cos': "https://drive.google.com/uc?id=1wefCQFzMne15qywCvK3pOc1Y4c89y2P6&export=download", 'month_sin': "https://drive.google.com/uc?id=1OqGryLmq7dSUMXELk_bdFYrLBhtjmxm7&export=download", 'pressure': "https://drive.google.com/uc?id=1sXNi0rW0q3AYiogZ-Pyp3qVdl2y9G_Ps&export=download", 'season': "https://drive.google.com/uc?id=11aA13hKRj7X9sGQPutMtegCTr8MBsXp1&export=download", 'temp': "https://drive.google.com/uc?id=1tx-ugAbyulQioVTZCelsdp4uzThCMe3M&export=download", 'wind': "https://drive.google.com/uc?id=1jTBAj31cW2kkQOtha5LHF0sQbo1tAX15&export=download"}
LGBM_MODEL_URLS = {'clouds': "https://drive.google.com/uc?id=18WKCOghwSno2VfT_Jl6lmDqG1BrCG_x0&export=download", 'hour_cos': "https://drive.google.com/uc?id=1kfKoJmf0Y7Aue1V95NhuZ4UFTGO0kE34&export=download", 'hour_sin': "https://drive.google.com/uc?id=1GuXG9apdnqd0UmzoS9ODsQJUQKjZum2b&export=download", 'humidity': "https://drive.google.com/uc?id=1duRGQykJQW_L5pKrExZ_9kcQoLtKWpM4&export=download", 'is_day': "https://drive.google.com/uc?id=1Z99-QScx6BrKmguQtpsjbuBEgrvRKu7Y&export=download", 'month_cos': "https://drive.google.com/uc?id=1HCQZLCepmzxDmJqwpvrtifHsU7K1ZFlK&export=download", 'month_sin': "https://drive.google.com/uc?id=1ibL_fCrH16jmGdjqzs9NsDjEqKtZPLoV&export=download", 'pressure': "https://drive.google.com/uc?id=1KJR58EEAMPq2C3i3PlcKE8sAR62RM0aj&export=download", 'season': "https://drive.google.com/uc?id=1SUyvQrs6b_VqiB4vysl5rqH1PwSuotJ-&export=download", 'temp': "https://drive.google.com/uc?id=1lQStxmql8LWPK5inzedpBKj-5ast2sSs&export=download", 'wind': "https://drive.google.com/uc?id=1VZk8bQiDYPjPWF_mLWMoI2bIGE64h5S1&export=download"}

MODEL_DIR = './model'
os.makedirs(MODEL_DIR, exist_ok=True)
WEIGHTS_FILE = os.path.join(MODEL_DIR, 'my_weights_attention.weights.h5')
SCALER_FILE = os.path.join(MODEL_DIR, 'scaler_attention.save')

download_file(MODEL_URL, WEIGHTS_FILE)
download_file(SCALER_URL, SCALER_FILE)
for f in FEATURES:
    download_file(RF_MODEL_URLS[f], os.path.join(MODEL_DIR, f'rf_{f}.joblib'))
    download_file(XGB_MODEL_URLS[f], os.path.join(MODEL_DIR, f'xgb_{f}.joblib'))
    download_file(LGBM_MODEL_URLS[f], os.path.join(MODEL_DIR, f'lgbm_{f}.joblib'))

# ========== 記念日データの読み込み ==========
def load_anniversaries_from_csv(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8')
    anniversary_dict = {}
    for _, row in df.iterrows():
        key = f"{int(row['month']):02d}-{int(row['day']):02d}"
        if key not in anniversary_dict:
            anniversary_dict[key] = []
        anniversary_dict[key].append({
            'title': row['title'],
            'url': row['url'],
            'priority': row['priority'],
            'description': row.get('description', ''),
            'summary': row['summary']
        })
    return anniversary_dict

JST = timezone('Asia/Tokyo')
now = datetime.now(JST)

ANNIV_CSV_PATH = "anniversary_365_summary.csv"
anniversaries = load_anniversaries_from_csv(ANNIV_CSV_PATH)
today_key = f"{now.month:02d}-{now.day:02d}"

if today_key in anniversaries:
    events = anniversaries[today_key]
    events_sorted = sorted(events, key=lambda x: x['priority'], reverse=True)
    main_event = events_sorted[0]
    event_text = (
        f"{main_event['title']}<br>"
        f"URL: <a href='{main_event['url']}' target='_blank'>{main_event['url']}</a><br>"
        f"<span style='color:#888;'>{main_event.get('description', '')}</span>"
    )
else:
    event_text = f"{now.month}月{now.day}日：今日は特に記念日は登録されていません。"

# ========== モデル構築 ==========
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(input_shape[1], 1),
                                 initializer='zeros', trainable=True)
        super(AttentionLayer, self).build(input_shape)
    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1, keepdims=True)

def create_attention_lstm_model():
    inputs = Input(shape=(48, len(FEATURES)))
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = Dropout(0.3)(x)
    x = AttentionLayer()(x)
    x = Dropout(0.3)(x)
    outputs = [Dense(1)(x) for _ in range(len(FEATURES))]
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-4), loss=Huber())
    return model

# ========== モデル・スケーラーのロード ==========
try:
    model = create_attention_lstm_model()
    model.load_weights(WEIGHTS_FILE)
    scaler = joblib.load(SCALER_FILE)
    rf_models = []
    xgb_models = []
    lgbm_models = []
    for f in FEATURES:
        rf_path = os.path.join(MODEL_DIR, f'rf_{f}.joblib')
        xgb_path = os.path.join(MODEL_DIR, f'xgb_{f}.joblib')
        lgbm_path = os.path.join(MODEL_DIR, f'lgbm_{f}.joblib')
        rf_models.append(joblib.load(rf_path) if os.path.exists(rf_path) else None)
        xgb_models.append(joblib.load(xgb_path) if os.path.exists(xgb_path) else None)
        lgbm_models.append(joblib.load(lgbm_path) if os.path.exists(lgbm_path) else None)
except Exception as e:
    st.error(f"モデルまたはスケーラーの読み込みエラー: {e}")
    st.stop()

def load_lottie(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# アニメーション読み込み
lottie_rain = load_lottie("animation/rain3.json")
lottie_sun = load_lottie("animation/sun2.json")
lottie_thermometer = load_lottie("animation/Thermometer.json")
lottie_cloud = load_lottie("animation/cloud.json")


# CSS適用関数（ファイル名は実行環境に合わせて）
def load_css(css_file):
    with open(css_file, encoding="utf-8") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_css(os.path.join(os.path.dirname(__file__), "style.css"))


if lottie_thermometer:
    st_lottie(lottie_thermometer, height=200)

def load_css(css_file):
    with open(css_file, encoding="utf-8") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_css(os.path.join(os.path.dirname(__file__), "style.css"))

def missauner_apparent_temperature(temp_c, humidity, wind_speed_ms):
    T_a = temp_c
    H = humidity
    v = wind_speed_ms
    apparent = 37 - (37 - T_a) * (0.68 - 0.0014 * H + 1 / (1.76 + 1.4 * v**0.75)) - 0.29 * T_a * (1 - H / 100)
    return apparent

def calc_sentaku_index(temp, humidity):
    index = 0.81 * temp + 0.01 * humidity * (0.99 * temp - 14.3) + 46.3
    index = max(0, min(100, index))
    return index

def sentaku_index_comment(index):
    if index >= 80:
        return "厚手の洗濯物も短時間で乾きます"
    elif index >= 60:
        return "洗濯日和です"
    elif index >= 40:
        return "薄手のものは乾きますが、厚手は注意"
    elif index >= 20:
        return "室内干しを推奨します"
    else:
        return "ほとんど乾きません"

# UI入力
target_date = st.date_input("予測したい日付を選択")
target_time_only = st.time_input("予測したい時刻を選択")
target_time = JST.localize(datetime.combine(target_date, target_time_only).replace(minute=0, second=0, microsecond=0))

csv_path = "C:/Users/ktdi104-alt/webapp/streamlit_project/data/API4year.csv"
if not os.path.exists(csv_path):
    st.error(f"CSVファイルが見つかりません: {csv_path}")
    st.stop()

csv_path2 = "C:/Users/ktdi104-alt/webapp/streamlit_project/data/solar.csv"
if not os.path.exists(csv_path2):
    st.error(f"CSVファイルが見つかりません: {csv_path2}")
    st.stop()

try:
    # 先に読み込んでから列名を確認・変換
    solar_df = pd.read_csv(csv_path2, encoding='cp932')

    # 列名の変換（適宜調整）
    solar_df = solar_df.rename(columns={
        '年月日時': 'datetime',
        '日射量': 'solar_radiation'  # 明示的に必要な名前に変換
    })

    # datetime変換は rename 後に実行
    solar_df['datetime'] = pd.to_datetime(solar_df['datetime'], errors='coerce')

    # NaT（変換できなかった）を削除
    solar_df = solar_df.dropna(subset=['datetime'])

except UnicodeDecodeError:
    st.error("ファイルの文字エンコーディングに問題があります。Shift-JIS（cp932）以外の可能性があります。")
    st.stop()
except Exception as e:
    st.error(f"solar.csvの読み込み中にエラーが発生しました: {e}")
    st.stop()

try:
    jma_df = pd.read_csv(csv_path, encoding="cp932")
    jma_df = jma_df.rename(columns={
        '年月日時': 'datetime', '気温(℃)': 'temp', '現地気圧(hPa)': 'pressure',
        '相対湿度(％)': 'humidity', '風速(m/s)': 'wind',
        '雲量(10分比)': 'clouds', '降水量(mm)': 'precipitation'
    })
    jma_df['datetime'] = pd.to_datetime(jma_df['datetime'], errors='coerce').dt.tz_localize('Asia/Tokyo', ambiguous='NaT', nonexistent='shift_forward')
    jma_df['clouds'] = pd.to_numeric(jma_df['clouds'], errors='coerce') * 10
    for col in ['temp', 'pressure', 'humidity', 'wind', 'precipitation']:
        jma_df[col] = pd.to_numeric(jma_df[col], errors='coerce')
    jma_df['rain_flag'] = (jma_df['precipitation'] >= 0.5).astype(int)
    jma_df['hour'] = jma_df['datetime'].dt.hour
    jma_df['month'] = jma_df['datetime'].dt.month
    # --- 周期特徴量 ---
    jma_df['hour_sin'] = np.sin(2 * np.pi * jma_df['hour'] / 24)
    jma_df['hour_cos'] = np.cos(2 * np.pi * jma_df['hour'] / 24)
    jma_df['month_sin'] = np.sin(2 * np.pi * jma_df['month'] / 12)
    jma_df['month_cos'] = np.cos(2 * np.pi * jma_df['month'] / 12)

    jma_df['temp_diff'] = jma_df['temp'].diff()
    jma_df['pressure_diff'] = jma_df['pressure'].diff()
    jma_df['season'] = (jma_df['month'] % 12 // 3).astype(float)
    jma_df['is_day'] = ((jma_df['hour'] > 6) & (jma_df['hour'] < 18)).astype(float)

    jma_df = jma_df[['datetime'] + FEATURES].dropna().reset_index(drop=True)
except Exception as e:
    st.error(f"CSV読み込みエラー: {e}")
    st.stop()

API_KEY = "254dc748d2c9e3cb06a056223f6a9668"
url = f"https://api.openweathermap.org/data/2.5/forecast?lat=35.6895&lon=139.6917&units=metric&appid={API_KEY}"
try:
    response = requests.get(url)
    data = response.json()
    if 'list' not in data:
        raise ValueError("APIレスポンスにデータがありません。")
    owm_df = pd.DataFrame([{
        'datetime': pd.to_datetime(item['dt_txt']).tz_localize('UTC').tz_convert('Asia/Tokyo'),
        'temp': item['main']['temp'],
        'pressure': item['main']['pressure'],
        'humidity': item['main']['humidity'],
        'wind': item['wind']['speed'],
        'clouds': item['clouds']['all'],
        'precipitation': item.get('rain', {}).get('3h', 0)
    } for item in data['list']])
    owm_df['rain_flag'] = (owm_df['precipitation'] >= 0.5).astype(int)
    owm_df['hour'] = owm_df['datetime'].dt.hour
    owm_df['month'] = owm_df['datetime'].dt.month
    owm_df['hour_sin'] = np.sin(2 * np.pi * owm_df['hour'] / 24)
    owm_df['hour_cos'] = np.cos(2 * np.pi * owm_df['hour'] / 24)
    owm_df['month_sin'] = np.sin(2 * np.pi * owm_df['month'] / 12)
    owm_df['month_cos'] = np.cos(2 * np.pi * owm_df['month'] / 12)
except Exception as e:
    st.error(f"API取得エラー: {e}")
    st.stop()

df = pd.concat([jma_df, owm_df], ignore_index=True).sort_values('datetime').reset_index(drop=True)
df['clouds'] /= 10  # API値に合わせて調整

def to_1d_array(pred):
    if isinstance(pred, np.ndarray):
        if pred.ndim == 2 and pred.shape[0] == 1:
            return pred[0]
        elif pred.ndim == 1:
            return pred
    return np.array([pred])

api_start = owm_df['datetime'].min().strftime("%Y-%m-%d %H:%M")
api_end = owm_df['datetime'].max().strftime("%Y-%m-%d %H:%M")
st.caption(f"※APIとの比較可能範囲：{api_start} 〜 {api_end}")


if st.button("予測実行"):
    df = df.fillna(method='ffill').fillna(method='bfill')
    last_data = df[FEATURES].values[-LOOKBACK:]
    scaled_data = scaler.transform(last_data)
    input_seq = scaled_data.reshape(1, LOOKBACK, len(FEATURES))
    delta_sec = (target_time - now).total_seconds()


    steps = max(1, int(np.ceil(delta_sec / 3600)))

    predicted_seq = []
    last_dt = df['datetime'].max()

    for step in range(steps):
        preds = model.predict(input_seq, verbose=0)

        next_features = np.array([pred[0, -1, 0] for pred in preds])

        predicted_seq.append(next_features.copy())

        next_features = np.array([pred[0, -1, 0] for pred in preds]).reshape(-1)

        next_dt = last_dt + pd.Timedelta(hours=1)
        hour, month = next_dt.hour, next_dt.month
        next_features[FEATURES.index('hour_sin')] = np.sin(2 * np.pi * hour / 24)
        next_features[FEATURES.index('hour_cos')] = np.cos(2 * np.pi * hour / 24)
        next_features[FEATURES.index('month_sin')] = np.sin(2 * np.pi * month / 12)
        next_features[FEATURES.index('month_cos')] = np.cos(2 * np.pi * month / 12)

        input_seq = np.append(input_seq[:, 1:, :], next_features.reshape(1, 1, len(FEATURES)), axis=1)
        last_dt = next_dt

    # LSTM逆スケール（最新1時点）
    if len(predicted_seq) == 0:
        st.error("予測結果が空のため、逆変換できません。")
        st.stop()
    pred_full = np.zeros((1, len(FEATURES)))
    pred_full[0, :] = predicted_seq[-1].flatten()
    predicted_values_lstm = scaler.inverse_transform(pred_full)[0]
    predicted_values_lstm = np.maximum(predicted_values_lstm, 0)

    rf_input = last_data.flatten().reshape(1, -1)

    predicted_values_rf = np.zeros(len(FEATURES))
    predicted_values_xgb = np.zeros(len(FEATURES))
    predicted_values_lgbm = np.zeros(len(FEATURES))

    for i, feat in enumerate(FEATURES):
        if rf_models[i] is not None:
            pred_rf = rf_models[i].predict(rf_input)
            # 逆スケールしてから負値クリップ（逆変換前はクリップしない）
            pred_rf_orig = inverse_transform_feature(scaler, np.array([pred_rf[0]]), i)[0]
            predicted_values_rf[i] = max(pred_rf_orig, 0)
        else:
            predicted_values_rf[i] = np.nan

        if xgb_models[i] is not None:
            pred_xgb = xgb_models[i].predict(rf_input)
            pred_xgb_orig = inverse_transform_feature(scaler, np.array([pred_xgb[0]]), i)[0]
            predicted_values_xgb[i] = max(pred_xgb_orig, 0)
        else:
            predicted_values_xgb[i] = np.nan

        if lgbm_models[i] is not None:
            pred_lgbm = lgbm_models[i].predict(rf_input)
            pred_lgbm_orig = inverse_transform_feature(scaler, np.array([pred_lgbm[0]]), i)[0]
            predicted_values_lgbm[i] = max(pred_lgbm_orig, 0)
        else:
            predicted_values_lgbm[i] = np.nan

    predicted_values_ensemble = (
        np.nan_to_num(predicted_values_lstm) * 0.22 +
        np.nan_to_num(predicted_values_rf) * 0.25 +
        np.nan_to_num(predicted_values_xgb) * 0.205 +
        np.nan_to_num(predicted_values_lgbm) * 0.325
    )

    display_precip = max(0, predicted_values_ensemble[FEATURES.index('precipitation')])
    rain_flag_pred = 1 if display_precip >= 0.5 else 0
    rain_flag_str = str(rain_flag_pred)

    if display_precip >= 1:
        st_lottie(lottie_rain, height=200)
        st.info("雨が降る予報です☔")
    elif predicted_values_ensemble[FEATURES.index('clouds')] >= 70:
        st_lottie(lottie_cloud, height=200)
        st.info("曇りの予報です⛅")
    else:
        st_lottie(lottie_sun, height=200)
        st.info("晴れの予報です☀️")

    temp = predicted_values_ensemble[FEATURES.index('temp')]
    humidity = predicted_values_ensemble[FEATURES.index('humidity')]
    wind = predicted_values_ensemble[FEATURES.index('wind')]

    apparent_temp = missauner_apparent_temperature(temp, humidity, wind)

    def kasa_index_rank(display_precip):
        # 文字列ならfloatに変換
        if isinstance(display_precip, str):
            try:
                display_precip = float(display_precip)
            except ValueError:
                # 変換できない場合は0などにする
                display_precip = 0

        if display_precip >= 80:
            return "傘は忘れずに"
        elif display_precip >= 60:
            return "傘を持ってお出かけください"
        elif display_precip >= 30:
            return "折りたたみ傘があると安心"
        else:
            return "傘はほとんど必要なし"
        
    def calc_fukusou_index(temp):
        # 5℃以下→10、35℃以上→100、間は線形
        if temp >= 35:
            return 100
        elif temp <= 5:
            return 10
        else:
            return int(10 + (temp - 5) * (90 / 30))

    def fukusou_index_comment(index):
        if index >= 90:
            return "ノースリーブや半袖でOK！"
        elif index >= 80:
            return "半袖シャツで快適です"
        elif index >= 70:
            return "長袖シャツやカーディガンが快適"
        elif index >= 50:
            return "薄手の上着やセーターが必要です"
        elif index >= 30:
            return "コートや厚手の上着が必要です"
        else:
            return "ダウンコート・手袋・マフラー必須！"
        
    def estimate_apparent_temp(temp, humidity, wind, target_datetime, solar_df, solar_coeff=0):
        try:
            nearest = solar_df.iloc[(solar_df['datetime'] - target_datetime).abs().argsort().iloc[0]]
            solar_radiation = nearest['solar_radiation']
            if pd.isna(solar_radiation):
                solar_radiation = 0
            # 異常値対策で最大値制限（例：1000 W/m²）
            solar_radiation = min(solar_radiation, 1000)
        except Exception as e:
            solar_radiation = 0

        apparent_temp = 0.81 * temp + 0.01 * humidity * (0.99 * temp - 14.3) - 0.4 * wind
        apparent_temp += solar_coeff * solar_radiation

        return apparent_temp
    
    apparent_temp = estimate_apparent_temp(
        temp=predicted_values_ensemble[FEATURES.index('temp')],
        humidity=predicted_values_ensemble[FEATURES.index('humidity')],
        wind=predicted_values_ensemble[FEATURES.index('wind')],
        target_datetime=target_time,
        solar_df=solar_df
    )

    fukusou_index = calc_fukusou_index(temp)
    hukusou = fukusou_index_comment(fukusou_index)

    icons = {
        "temp": "https://img.icons8.com/?size=100&id=1414&format=png&color=000000",
        "apparent": "https://img.icons8.com/?size=100&id=XeaKlVFP2aRT&format=png&color=000000",
        "pressure": "https://img.icons8.com/?size=100&id=1418&format=png&color=000000",
        "humidity": "https://img.icons8.com/?size=100&id=6969&format=png&color=000000",
        "wind": "https://img.icons8.com/?size=100&id=31842&format=png&color=000000",
        "clouds": "https://img.icons8.com/?size=100&id=658&format=png&color=000000",
        "precip": "https://img.icons8.com/?size=100&id=2796&format=png&color=000000",
        "rainflag": "https://img.icons8.com/?size=100&id=3611&format=png&color=000000",
        "sentaku": "https://img.icons8.com/?size=100&id=3EdMvnDrrp4A&format=png&color=000000",
        "kasa":"https://img.icons8.com/?size=100&id=390&format=png&color=000000",
        "huku":"https://img.icons8.com/?size=100&id=9158&format=png&color=000000"
    }

    rain_flag = display_precip >= 0.5
    rainflag_icon = icons['rainflag'] if rain_flag else "https://img.icons8.com/?size=100&id=648&format=png&color=000000"

    # 予測値や観測値から計算
    temp = predicted_values_ensemble[FEATURES.index('temp')]
    humidity = predicted_values_ensemble[FEATURES.index('humidity')]

    sentaku_index = calc_sentaku_index(temp, humidity)

    comment = sentaku_index_comment(sentaku_index)

    kasa_index = kasa_index_rank(display_precip)
    rank = kasa_index_rank(kasa_index)

    # --- HTMLでカード表示 ---
    st.markdown(f"""
    <div style="font-size:22px;font-weight:bold;margin-bottom:16px;">
        {target_time.strftime('%Y-%m-%d %H:%M')} の予測
        <div style="font-size:20px;font-weight:bold;margin-bottom:4px;">
            今日は何の日：{now.month}月{now.day}日
        </div>
        <div style="font-size:16px;margin-bottom:16px;color:#444;">
            {event_text}
        </div>
    </div>

    <div class="result-container">
    <div class="result-card">
        <img src="{icons['temp']}" class="result-icon"/>
        <div>
        <div class="result-label">気温</div>
        <div class="result-value">{predicted_values_ensemble[FEATURES.index('temp')]:.2f} ℃</div>
        </div>
    </div>
    <div class="result-card">
        <img src="{icons['apparent']}" class="result-icon"/>
        <div>
        <div class="result-label">体感温度</div>
        <div class="result-value">{apparent_temp:.2f} ℃</div>
        </div>
    </div>
    <div class="result-card">
        <img src="{icons['pressure']}" class="result-icon"/>
        <div>
        <div class="result-label">気圧</div>
        <div class="result-value">{predicted_values_ensemble[FEATURES.index('pressure')]:.2f} hPa</div>
        </div>
    </div>
    <div class="result-card">
        <img src="{icons['humidity']}" class="result-icon"/>
        <div>
        <div class="result-label">湿度</div>
        <div class="result-value">{predicted_values_ensemble[FEATURES.index('humidity')]:.2f} %</div>
        </div>
    </div>
    <div class="result-card">
        <img src="{icons['wind']}" class="result-icon"/>
        <div>
        <div class="result-label">風速</div>
        <div class="result-value">{predicted_values_ensemble[FEATURES.index('wind')]:.2f} m/s</div>
        </div>
    </div>
    <div class="result-card">
        <img src="{icons['clouds']}" class="result-icon"/>
        <div>
        <div class="result-label">雲量</div>
        <div class="result-value">{predicted_values_ensemble[FEATURES.index('clouds')]:.2f} %</div>
        </div>
    </div>
    <div class="result-card">
        <img src="{icons['precip']}" class="result-icon"/>
        <div>
        <div class="result-label">降水量</div>
        <div class="result-value">{display_precip:.2f}mm</div>
        </div>
    </div>
    <div class="result-card">
        <img src="{icons['kasa']}" class="result-icon"/>
        <div>
        <div class="result-label">傘指数</div>
        <div class="result-value">{rank}</div>
        </div>
    </div>
        <div class="result-card">
        <img src="{icons['sentaku']}" class="result-icon"/>
        <div>
        <div class="result-label">洗濯指数</div>
        <div class="result-value">{comment}</div>
        </div>
    </div>
        <div class="result-card">
        <img src="{icons['huku']}" class="result-icon"/>
        <div>
        <div class="result-label">服装指数</div>
        <div class="result-value">{hukusou}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if (target_time >= owm_df['datetime'].min()) and (target_time <= owm_df['datetime'].max()):
        api_row = owm_df.iloc[(owm_df['datetime'] - target_time).abs().argsort()[:1]]
        api_temp = api_row['temp'].values[0]
        api_pressure = api_row['pressure'].values[0]
        api_humidity = api_row['humidity'].values[0]
        api_wind = api_row['wind'].values[0]
        api_clouds = api_row['clouds'].values[0]
        api_precip = api_row['precipitation'].values[0]
        api_rain_flag = api_row['rain_flag'].values[0]
        temp_err = abs(predicted_values_ensemble[FEATURES.index('temp')] - api_temp)
        pressure_err = abs(predicted_values_ensemble[FEATURES.index('pressure')] - api_pressure)
        humidity_err = abs(predicted_values_ensemble[FEATURES.index('humidity')] - api_humidity)
        wind_err = abs(predicted_values_ensemble[FEATURES.index('wind')] - api_wind)
        clouds_err = abs(predicted_values_ensemble[FEATURES.index('clouds')] - api_clouds)
        precip_err = abs(display_precip - api_precip)
        rain_flag_hit = int(rain_flag_pred) == int(api_rain_flag)

        st.markdown("### OpenWeather API値との比較")
        st.write(
            f"- 気温: {api_temp:.2f}℃（誤差: {temp_err:.2f}℃）\n"
            f"- 気圧: {api_pressure:.2f}hPa（誤差: {pressure_err:.2f}hPa）\n"
            f"- 湿度: {api_humidity:.2f}%（誤差: {humidity_err:.2f}%）\n"
            f"- 風速: {api_wind:.2f}m/s（誤差: {wind_err:.2f}m/s）\n"
            f"- 雲量: {api_clouds:.2f}%（誤差: {clouds_err:.2f}%）\n"
            f"- 降水量: {api_precip:.2f}mm（誤差: {precip_err:.2f}mm）\n"
            f"- 降水フラグ一致: {'〇' if rain_flag_hit else '×'}"
        )


        st.caption("※OpenWeather APIの値は、全国規模の気象モデルによる推定値です。指定した座標（東京など）のピンポイント観測値ではなく、東京周辺の広い範囲の平均的な天気を計算した値が返されるため、実際の観測値と大きく異なる場合があります。")
    else:
        st.info("選択日時はAPIの予報範囲外です。")
