import io
import json
import textwrap
from typing import Dict, Any, List, Optional
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import datetime

# ---- OpenAI SDK 확인 ----
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


# =========================
# API 키 (코드 내 삽입)
# =========================
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")


# =========================
# [신규] 지식 파일 로드 헬퍼 (Simplified RAG)
# =========================
@st.cache_data # 앱 실행 시 한 번만 읽도록 캐시
def load_knowledge_file(file_path):
    """app.py와 동일한 위치에 있는 .txt 지식 파일을 읽습니다."""
    try:
        # GitHub 저장소의 루트에서 파일을 찾음
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        st.warning(f"경고: 지식 파일({file_path})을 찾을 수 없습니다. AI가 일반적인 답변만 할 수 있습니다.")
        return ""
    except Exception as e:
        st.error(f"지식 파일 로드 오류: {e}")
        return ""

# --- 앱 시작 시 지식 파일 로드 ---
KNOWLEDGE_CURRICULUM = load_knowledge_file("knowledge_curriculum.txt")
KNOWLEDGE_DISASTERS = load_knowledge_file("knowledge_disasters.txt")


# =========================
# 페이지 기본 설정
# =========================
st.set_page_config(
    page_title="AI 기반 빅데이터 탐구 (홈)", 
    page_icon="🛰️",
    layout="wide",
)

# =========================
# 세션 상태 초기화
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[Dict[str, str]] = []
if "df" not in st.session_state:
    st.session_state.df: Optional[pd.DataFrame] = None
if "api_key" not in st.session_state:
    st.session_state.api_key = OPENAI_API_KEY
if "model" not in st.session_state:
    st.session_state.model = "gpt-4o-mini"


# =========================
# 사이드바: AI 모델 설정
# =========================
with st.sidebar:
    st.markdown("## ⚙️ AI 모델 설정")
    if st.session_state.api_key == "YOUR_OPENAI_API_KEY_HERE" or not st.session_state.api_key:
        st.error("코드 상단의 OPENAI_API_KEY 변수에 실제 키를 입력하세요.")
    else:
        st.success("OpenAI API Key가 로드되었습니다.")
    st.session_state.model = st.selectbox(
        "모델 선택",
        options=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
        index=0,
        help="해석 정확도가 중요하면 상위 모델, 비용이 중요하면 mini 권장",
    )
    st.divider()
    st.info("데이터 다운로드는 'data' 페이지를 참고하세요.")


# =========================
# 상단 헤더
# =========================
st.title("🛰️ 재해·재난과 안전 빅데이터 탐구 지원 챗봇")
st.markdown(
    "중학생 과학 ‘재해·재난과 안전’ 수업에서 **빅데이터 탐구**를 돕는 챗봇입니다. "
    "데이터를 시각화하고, **AI에게 해석**을 요청해 보세요."
)
if st.session_state.api_key == "YOUR_OPENAI_API_KEY_HERE" or not st.session_state.api_key:
    st.error("분석을 시작하기 전에 Streamlit 코드의 `OPENAI_API_KEY` 변수에 실제 OpenAI API 키를 입력해야 합니다.")
    st.stop()


# =========================
# 1) 데이터 불러오기
# =========================
st.markdown("## 1) 데이터 불러오기 📥")
file = st.file_uploader(
    "CSV 또는 XLSX 파일 업로드",
    type=["csv", "xlsx"],
    accept_multiple_files=False,
    help="첫 번째 시트 기준(XLSX). 수업용 데이터는 'data' 페이지에서 다운로드 받으세요.",
)
def load_dataframe(_file) -> pd.DataFrame:
    if _file is None: return pd.DataFrame()
    if _file.name.lower().endswith(".csv"):
        try: df = pd.read_csv(_file, sep=",", low_memory=False, encoding='utf-8')
        except UnicodeDecodeError: df = pd.read_csv(_file, sep=",", low_memory=False, encoding='cp949')
    else: df = pd.read_excel(_file, engine="openpyxl")
    return df
def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["int64", "int32"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64", "float32"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df
if file:
    df = load_dataframe(file)
    df = optimize_dtypes(df)
    st.session_state.df = df
if st.session_state.df is not None and not st.session_state.df.empty:
    df = st.session_state.df
    st.success(f"불러온 데이터: {df.shape[0]:,}행 × {df.shape[1]:,}열")
    with st.expander("📋 데이터 미리보기(상위 100행)", expanded=True):
        st.dataframe(df.head(100), use_container_width=True)
    st.markdown("### 🔎 빠른 요약")
    col_meta1, col_meta2, col_meta3 = st.columns(3)
    with col_meta1: st.metric("행 수", f"{df.shape[0]:,}")
    with col_meta2: st.metric("열 수", f"{df.shape[1]:,}")
    with col_meta3:
        missing_total = int(df.isna().sum().sum())
        st.metric("결측치 총합", f"{missing_total:,}")
    with st.expander("🧮 기술통계(수치형)"):
        st.dataframe(df.describe().T, use_container_width=True)
    with st.expander("🧾 열 타입 정보"):
        info = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum(), "unique": df.nunique()})
        st.dataframe(info, use_container_width=True)
else:
    st.info("왼쪽 사이드바에서 **[data]** 페이지를 클릭해 CSV 파일을 다운로드 받거나, 가지고 있는 파일을 업로드하여 탐구를 시작하세요.")
    st.stop()


# =========================
# 2) 데이터 시각화
# =========================
st.markdown("## 2) 데이터 시각화 📊")
st.caption("핵심 차트 유형만 선택하고, AI와 함께 해석에 집중해 보세요.")
chart_type = st.selectbox(
    "차트 유형",
    ["선(line)", "막대(bar)", "산점도(scatter)", "원(pie)", "지도 (위도/경도)"]
)
if chart_type.startswith("원("):
    x_label = "이름 (범주 열)"; y_label = "값 (수치 열)"; size_label = "추가 범례 (선택)"
elif chart_type.startswith("지도"):
    x_label = "위도 (Latitude) 열"; y_label = "경도 (Longitude) 열"; size_label = "크기/강도 (Magnitude) 열"
else: 
    x_label = "X축"; y_label = "Y축 (필요시)"; size_label = "크기 (선택, 산점도용)"
viz_col1, viz_col2, viz_col3 = st.columns(3)
with viz_col1: x_col = st.selectbox(x_label, options=df.columns, index=0)
with viz_col2: y_col = st.selectbox(y_label, options=["- 선택 안함 -"] + df.columns.tolist(), index=0)
with viz_col3: size_col = st.selectbox(size_label, options=["- 선택 안함 -"] + df.columns.tolist(), index=0)
all_cols = df.columns.tolist()
hover_cols = st.multiselect(
    "💡 차트 툴팁(마우스 오버)에 표시할 추가 정보",
    options=all_cols, default=None
)
agg_fn = "count"
if chart_type.startswith("막대("):
    agg_fn = st.selectbox("집계 함수(막대)", ["count", "sum", "mean", "median"], help="Y축이 없으면 'count'가 자동 적용됩니다.")
def get_val(opt): return None if (opt == "- 선택 안함 -" or opt == "-") else opt
x = x_col; y = get_val(y_col); size = get_val(size_col); hover = hover_cols if hover_cols else None
fig = None; chart_spec = None
try:
    if chart_type.startswith("선("):
        if y is None: st.warning("선 그래프는 Y축이 필요합니다.")
        else:
            fig = px.line(df, x=x, y=y, hover_data=hover, height=500, title=f"{x}에 따른 {y} 변화")
            chart_spec = {"chart_type": "Line", "x": x, "y": y, "hover": hover}
    elif chart_type.startswith("막대("):
        if y is None: 
            tmp = df.groupby(x).size().reset_index(name="count")
            fig = px.bar(tmp, x=x, y="count", hover_data=hover, height=500, title=f"{x}별 개수(count)")
            chart_spec = {"chart_type": "Bar (Count)", "x": x, "y": "count", "hover": hover}
        else: 
            agg_map = {"count": "count", "sum": "sum", "mean": "mean", "median": "median"}
            tmp = df.groupby(x)[y].agg(agg_map[agg_fn]).reset_index()
            y_agg = f"{agg_fn}_{y}"; tmp = tmp.rename(columns={y: y_agg})
            fig = px.bar(tmp, x=x, y=y_agg, hover_data=hover, height=500, title=f"{x}별 {y}의 {agg_fn}")
            chart_spec = {"chart_type": "Bar (Aggregate)", "x": x, "y": y_agg, "function": agg_fn, "hover": hover}
    elif chart_type.startswith("산점도"):
        if y is None: st.warning("산점도는 Y축이 필요합니다.")
        else:
            fig = px.scatter(df, x=x, y=y, size=size, hover_data=hover, opacity=0.7, height=500, title=f"{x}와 {y}의 관계 (크기: {size})")
            chart_spec = {"chart_type": "Scatter", "x": x, "y": y, "size": size, "hover": hover}
    elif chart_type.startswith("원("):
        if y is None: st.warning("원 그래프는 '값 (수치 열)' (Y축)이 필요합니다.")
        else:
            fig = px.pie(df, names=x, values=y, hover_data=hover, height=500, title=f"{x}별 {y}의 비율")
            chart_spec = {"chart_type": "Pie", "names": x, "values": y, "hover": hover}
    elif chart_type.startswith("지도"): 
        if y is None: st.warning("지도 시각화는 '위도'와 '경도' 열이 모두 필요합니다.")
        else:
            fig = px.scatter_geo(df, lat=x, lon=y, size=size, hover_data=hover, projection="natural earth", height=600, title=f"지도 시각화 (위도:{x}, 경도:{y}, 크기:{size})")
            fig.update_geos(center={"lat": 36, "lon": 127.5}, lataxis_range=[33, 39], lonaxis_range=[124, 132], showcountries=True, showcoastlines=True)
            chart_spec = {"chart_type": "Map (Scatter Geo)", "lat": x, "lon": y, "size": size, "hover": hover}
except Exception as e:
    st.error(f"차트 생성 중 오류: {e}")
if fig is not None:
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("위의 옵션을 선택하여 시각화를 생성해 보세요.")


# =========================
# 3) 데이터 해석 챗봇
# =========================
st.markdown("## 3) 데이터 해석 챗봇 🤖")
st.caption("AI에게 데이터와 차트를 분석해 달라고 요청해 보세요.")

# [수정] summarize_dataframe: 통계 요약(describe)을 포함하도록 강화
def summarize_dataframe(df: pd.DataFrame, max_rows: int = 5) -> str:
    """데이터프레임을 AI가 이해하기 쉬운 상세한 JSON 요약으로 변환합니다."""
    
    # 1. 스키마 (데이터 타입)
    schema = {col: str(df[col].dtype) for col in df.columns}
    
    # 2. 미리보기 (Head)
    preview = df.head(max_rows).to_dict(orient="records")
    
    # 3. 통계 요약 (Numerical)
    try:
        numerical_summary = df.describe().to_dict()
    except Exception:
        numerical_summary = {} # 수치형 데이터가 없을 경우
        
    # 4. 범주형 요약 (Categorical)
    categorical_summary = {}
    for col in df.select_dtypes(include=['object', 'category']).columns:
        categorical_summary[col] = {
            "nunique": df[col].nunique(),
            "top_5_values": df[col].value_counts().head(5).to_dict()
        }

    summary = {
        "file_name": file.name if file else "N/A",
        "total_shape": [int(df.shape[0]), int(df.shape[1])],
        "schema": schema,
        "head_preview (5 rows)": preview,
        "numerical_summary (df.describe)": numerical_summary,
        "categorical_summary (top 5 values)": categorical_summary
    }

    # JSON 변환 시 ensure_ascii=False 로 한글 유지
    # indent=2를 넣어 가독성 향상
    return json.dumps(summary, ensure_ascii=False, indent=2, default=str)


# build_messages
def build_messages(prompt: str, data_brief: str) -> List[Dict[str, str]]:
    
    # --- RAG ---
    system_prompt = f"""
    [역할]
    너는 대한민국 중학교 과학 교사(장윤하 선생님)를 돕는 'AI 데이터 분석 전문가'이자 '과학 보조 교사'이다.

    [핵심 임무]
    중학생들이 '재해·재난과 안전' 단원을 탐구할 수 있도록, 제공된 [데이터 요약]과 [차트 정보]를 [교육과정 지식] 및 [과학 원리 지식]과 연결하여 **실질적이고 비판적인 해석**을 제공해야 한다.
    
    [규칙 1: 지식 기반 (RAG)]
    너의 모든 답변은 반드시 아래 제공된 두 가지 핵심 지식을 근거로 해야 한다.
    
    1. [교육과정 지식] (knowledge_curriculum.txt의 내용)
    {KNOWLEDGE_CURRICULUM if KNOWLEDGE_CURRICULUM else "N/A"}

    2. [과학 원리 지식] (knowledge_disasters.txt의 내용)
    {KNOWLEDGE_DISASTERS if KNOWLEDGE_DISASTERS else "N/A"}

    [규칙 2: 데이터 기반 (Grounded)]
    너의 분석은 **절대로** 너의 일반 상식이나 학습된 데이터를 기반으로 하면 안 된다.
    **오직** 아래 제공되는 [데이터 요약]과 [차트 정보]에서 관찰된 **구체적인 숫자, 경향, 패턴**만을 근거로 해석해야 한다.
    만약 데이터가 부족하면, "데이터에 따르면..."이라고 말하지 말고 "데이터가 부족하여 알 수 없지만..."이라고 명확히 밝혀야 한다.

    [규칙 3: 용도 제한 (Context Bound)]
    주어진 용도 (중학생 과학 수업)를 벗어난 대화에 대해서는 답변하지 말고, 반드시 "이 챗봇은 중학교 과학 수업 지원용입니다."라고 답변해라.

    [출력 형식]
    - 중학생이 이해할 수 있도록 명확하고 간결한 문장 사용
    - 전문가적이지만 친절한 어조 사용
    - 핵심 내용은 굵은 글씨(**)와 bullet points (•)를 사용해 정리
    """
    
    msgs: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    
    # --- 컨텍스트 ---
    ctx_parts = []
    if add_data_head: # 체크박스가 True일 때
        ctx_parts.append(f"[데이터 요약]\n{data_brief}")
    if add_context and chart_spec: # 체크박스가 True일 때
        ctx_parts.append(f"[현재 시각화된 차트 정보]\n{json.dumps(chart_spec, ensure_ascii=False, indent=2)}")
    
    ctx = "\n\n".join(ctx_parts) if ctx_parts else "(제공된 데이터 컨텍스트 없음)"

    user = f"{prompt}\n\n[참고할 컨텍스트]\n{ctx}"
    msgs.append({"role": "user", "content": user})
    return msgs


# call_openai
def call_openai(messages: List[Dict[str, str]], model: str, api_key: str) -> str:
    if not OPENAI_AVAILABLE:
        return "⚠️ openai 패키지를 찾을 수 없습니다. `pip install openai` 후 다시 시도하세요."
    if not api_key or api_key == "YOUR_OPENAI_API_KEY_HERE":
        return "⚠️ OpenAI API Key가 필요합니다. 코드 상단의 `OPENAI_API_KEY` 변수를 수정하세요."
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=1500
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ OpenAI 호출 오류: {e}"


# --- 챗봇 UI ---

# 데이터 요약
try:
    data_brief = summarize_dataframe(df, max_rows=5)
except Exception as e:
    data_brief = f"데이터 요약 생성 실패: {e}"
    st.warning(data_brief)

# 프롬프트
default_prompt = (
    "현재 업로드된 [데이터 요약]과 [차트 정보]를 분석해 주세요.\n\n"
    "1. 이 데이터에서 발견할 수 있는 가장 중요한 경향이나 사실은 무엇인가요? (데이터의 숫자를 근거로 들어주세요)\n"
    "2. 이 현상을 [과학 원리 지식]과 어떻게 연결할 수 있나요?\n"
    "3. 이 데이터를 [교육과정 지식]의 성취기준과 연결할 때, 어떤 비판적 질문을 토론해 볼 수 있을까요?"
)
user_prompt = st.text_area("질문 입력:", value=default_prompt, height=200)

col_chat1, col_chat2 = st.columns([1, 2])
with col_chat1:
    add_context = st.checkbox("그래프 메타데이터 포함", True, help="차트 유형, 축, 집계 방식 등 메타를 LLM에 전달")
with col_chat2:
    add_data_head = st.checkbox("데이터 요약(통계 포함) 포함", True, help="AI가 실제 데이터를 분석하도록 통계 요약본을 전달합니다.")

chat_cols = st.columns([1, 1, 6])
with chat_cols[0]:
    if st.button("AI 해석 요청", type="primary", use_container_width=True):
        with st.spinner("AI가 데이터를 분석 중입니다..."):
            # data_brief를 인자로 전달
            msgs = build_messages(user_prompt, data_brief)
            
            # (디버깅용) AI에게 보낸 최종 프롬프트 확인
            # with st.expander("[Debug] AI에게 전송된 최종 프롬프트"):
            #     st.json(msgs)

            answer = call_openai(msgs, st.session_state.model, st.session_state.api_key)
            st.session_state.chat_history.append({"role": "user", "content": user_prompt})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

with chat_cols[1]:
    if st.button("기록 지우기", use_container_width=True):
        st.session_state.chat_history = []

# --- 대화창 (변경 없음) ---
st.markdown("### 대화 기록")
if not st.session_state.chat_history:
    st.info("데이터를 업로드/시각화한 후, ‘AI 해석 요청’을 눌러보세요.")
else:
    for turn in st.session_state.chat_history:
        if turn["role"] == "user":
            st.markdown(f"**🧑 질문**\n\n{turn['content']}")
        else:
            st.markdown(f"**🤖 답변**\n\n{turn['content']}")

with st.expander("ℹ️ 도움말 / 주의"):
    st.markdown(
        """
- **교육 맥락**: AI는 '재해·재난과 안전' 단원 성취기준과 SSI 쟁점 토론을 유도하도록 설정되었습니다.
        """
    )