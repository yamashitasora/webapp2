import streamlit as st
import json
import os
import pandas as pd
from datetime import datetime, time
from streamlit_lottie import st_lottie
from pytz import timezone

JST = timezone('Asia/Tokyo')
now = datetime.now(JST)

# --- UIタイトル ---
st.markdown(
    """
    <h1 style='text-align: center; font-size: 4em;'>
        気象予測アプリ
    </h1>
    """,
    unsafe_allow_html=True
)

def load_lottie(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# アニメーション読み込み
lottie_rain = load_lottie("animation/rain3.json")
lottie_sun = load_lottie("animation/sun2.json")
lottie_thermometer = load_lottie("animation/Thermometer.json")
lottie_cloud = load_lottie("animation/cloud.json")

if lottie_thermometer:
    st_lottie(lottie_thermometer, height=200)

# 日時指定
target_date = st.date_input("予測したい日付を選択", value=now.date())
target_time = st.time_input("予測したい時刻を選択", value=now.time().replace(minute=0, second=0, microsecond=0))
target_datetime = JST.localize(datetime.combine(target_date, target_time))
dt_key = target_datetime.strftime('%Y-%m-%d %H:%M')
today_key = f"{target_datetime.month:02d}-{target_datetime.day:02d}"




# CSS適用
def load_css(css_file):
    with open(css_file, encoding="utf-8") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
load_css(os.path.join(os.path.dirname(__file__), "style.css"))

# --- 記念日CSVの読み込みとevent_text生成 ---
anniversary_csv = "C:/Users/ktdi104-alt/webapp/streamlit_project/data/anniversary_365_summary.csv"  # パスは適宜修正
event_text = f"{target_datetime.month}月{target_datetime.day}日：今日は特に記念日は登録されていません。"
if os.path.exists(anniversary_csv):
    try:
        df_anniv = pd.read_csv(anniversary_csv, encoding='utf-8')
        anniversary_dict = {}
        for _, row in df_anniv.iterrows():
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
        if today_key in anniversary_dict:
            events = anniversary_dict[today_key]
            events_sorted = sorted(events, key=lambda x: x['priority'], reverse=True)
            main_event = events_sorted[0]
            event_text = (
                f"{main_event['title']}<br>"
                f"URL: <a href='{main_event['url']}' target='_blank'>{main_event['url']}</a><br>"
                f"<span style='color:#888;'>{main_event.get('description', '')}</span>"
            )
    except Exception as e:
        st.warning(f"記念日データの読み込みエラー: {e}")

# --- 予測実行ボタン ---
if st.button("予測実行"):
    with open('prediction_results.json', 'r', encoding='utf-8') as f:
        all_results = json.load(f)

    if dt_key not in all_results:
        st.warning(f"{dt_key} の予測結果がありません。")
        st.stop()

    result = all_results[dt_key]

    FEATURES = [
        'temp', 'pressure', 'humidity', 'wind', 'clouds', 'precipitation',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
        'temp_diff', 'pressure_diff', 'season', 'is_day'
    ]
    predicted_values_ensemble = [result[f] for f in FEATURES]
    display_precip = max(0, result['precipitation'])

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

    def missauner_apparent_temperature(temp_c, humidity, wind_speed_ms):
        T_a = temp_c
        H = humidity
        v = wind_speed_ms
        apparent = 37 - (37 - T_a) * (0.68 - 0.0014 * H + 1 / (1.76 + 1.4 * v**0.75)) - 0.29 * T_a * (1 - H / 100)
        return apparent

    def calc_sentaku_index(temp, humidity):
        index = 0.81 * temp + 0.01 * humidity * (0.99 * temp - 14.3) + 46.3
        return max(0, min(100, index))

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

    def kasa_index_rank(display_precip):
        if display_precip >= 80:
            return "傘は忘れずに"
        elif display_precip >= 60:
            return "傘を持ってお出かけください"
        elif display_precip >= 30:
            return "折りたたみ傘があると安心"
        else:
            return "傘はほとんど必要なし"

    def calc_fukusou_index(temp):
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

    temp = result["temp"]
    humidity = result["humidity"]
    wind = result["wind"]
    apparent_temp = missauner_apparent_temperature(temp, humidity, wind)
    sentaku_index = calc_sentaku_index(temp, humidity)
    comment = sentaku_index_comment(sentaku_index)
    rank = kasa_index_rank(display_precip)
    fukusou_index = calc_fukusou_index(temp)
    hukusou = fukusou_index_comment(fukusou_index)

    if display_precip >= 1:
        st_lottie(lottie_rain, height=200)
        st.info("雨が降る予報です☔")
    elif predicted_values_ensemble[FEATURES.index('clouds')] >= 70:
        st_lottie(lottie_cloud, height=200)
        st.info("曇りの予報です⛅")
    else:
        st_lottie(lottie_sun, height=200)
        st.info("晴れの予報です☀️")


    # --- HTMLでカード表示 ---
    st.markdown(f"""
    <div style="font-size:22px;font-weight:bold;margin-bottom:16px;">
        {target_datetime.strftime('%Y-%m-%d %H:%M')} の予測
        <div style="font-size:20px;font-weight:bold;margin-bottom:4px;">
            今日は何の日：{target_datetime.month}月{target_datetime.day}日
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
        <div class="result-value">{temp:.2f} ℃</div>
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
        <div class="result-value">{result['pressure']:.2f} hPa</div>
        </div>
    </div>
    <div class="result-card">
        <img src="{icons['humidity']}" class="result-icon"/>
        <div>
        <div class="result-label">湿度</div>
        <div class="result-value">{humidity:.2f} %</div>
        </div>
    </div>
    <div class="result-card">
        <img src="{icons['wind']}" class="result-icon"/>
        <div>
        <div class="result-label">風速</div>
        <div class="result-value">{wind:.2f} m/s</div>
        </div>
    </div>
    <div class="result-card">
        <img src="{icons['clouds']}" class="result-icon"/>
        <div>
        <div class="result-label">雲量</div>
        <div class="result-value">{result['clouds']:.2f} %</div>
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
