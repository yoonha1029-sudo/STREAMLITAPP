"""
[데이터 자료실 페이지]
- 주제별 수업용 데이터 다운로드
- 원본 데이터 출처 링크 제공
"""
import streamlit as st
import os

# =========================
# 로컬 파일 로드 헬퍼 함수 (필수)
# =========================
@st.cache_data # 캐시를 사용해 파일을 한 번만 읽어옵니다.
def load_local_file_bytes(file_path):
    """로컬 파일을 바이트(bytes)로 읽어옵니다."""
    try:
        # 이 코드가 app.py와 같은 폴더(루트)에 있는 파일을 찾습니다.
        with open(file_path, 'rb') as f:
            return f.read()
    except FileNotFoundError:
        # 파일이 없으면 None 반환
        st.warning(f"파일을 찾을 수 없습니다: {file_path}")
        return None
    except Exception as e:
        st.error(f"파일 로드 중 오류 발생: {e}")
        return None

# =========================
# 페이지 구성
# =========================
st.title("💾 수업용 데이터 자료실")
st.caption("주제를 선택하여 수업용 CSV 파일을 다운로드하거나 원본 출처를 확인하세요.")
st.info("이 페이지의 파일들은 선생님이 수업용으로 미리 정제한 데이터입니다.")

# --- 1. 기상 재해 ---
with st.expander("🌦️ 1. 기상 재해 (폭염, 장마 등)", expanded=True):
    st.markdown("**[수업용 CSV 파일]**")
    
    # [중요] GitHub 저장소 루트에 있는 실제 파일명과 일치해야 합니다.
    f_heatwave = load_local_file_bytes("STCS_폭염일수_20251106201850.csv")
    f_tropical = load_local_file_bytes("STCS_열대야일수_20251106225417.csv")
    f_rainy = load_local_file_bytes("STCS_장마_20251106224957.csv")
    f_temp = load_local_file_bytes("일시,평균기온,최고기온 평균,최저기온 평균,강수량.csv")
    f_temp = load_local_file_bytes("산불피해_현황.csv")
    f_temp = load_local_file_bytes("시도별_산불발생_현황_20251107033925.csv")

    if f_heatwave:
        st.download_button("1-1. 폭염일수 (전국, 연도별)", f_heatwave, "student_heatwave.csv", "text/csv")
    if f_tropical:
        st.download_button("1-2. 열대야일수 (전국, 연도별)", f_tropical, "student_tropical_night.csv", "text/csv")
    if f_rainy:
        st.download_button("1-3. 장마철 강수량 (전국, 연도별)", f_rainy, "student_rainy_season.csv", "text/csv")
    if f_temp:
        st.download_button("1-4. 월별 평균 기온/강수량 (서울)", f_temp, "student_monthly_temp.csv", "text/csv")
    if f_temp:
        st.download_button("1-5. 산불 현황 (전국, 연도별)", f_temp, "student_forest_fire.csv", "text/csv")
    if f_temp:
        st.download_button("1-6. 산불 현황 (지역별, 피해규모)", f_temp, "student_forest_fire_region.csv", "text/csv")

    st.markdown("**[원본 출처 링크]**")
    st.markdown("- [기상자료개방포털 (기후통계분석)](https://data.kma.go.kr/climate/RankState/selectRankStatisticsList.do)")
    st.markdown("- [산림임업플랫폼 산림통계시스템 (FoSS)](https://kfss.forest.go.kr/stat/ptl/main/main.do)")

# --- 2. 지진 재해 ---
with st.expander("🌍 2. 지진 재해"):
    st.markdown("**[수업용 CSV 파일]**")
    
    f_eq_list = load_local_file_bytes("EQK_지진정보_20251106234702.csv")
    f_eq_count = load_local_file_bytes("지역별_규모별_지진발생_횟수_20251106233407.csv")

    if f_eq_list:
        st.download_button("2-1. 지진 발생 목록 (2015~)", f_eq_list, "student_earthquake_list.csv", "text/csv")
    if f_eq_count:
        st.download_button("2-2. 지역/규모별 발생 횟수", f_eq_count, "student_earthquake_count_region.csv", "text/csv")

    st.markdown("**[원본 출처 링크]**")
    st.markdown("- [기상청 날씨누리 (지진 목록)](https://www.weather.go.kr/w/earthquake-volcano/list.do)")

# --- 3. 감염병 재해 ---
with st.expander("☣️ 3. 감염병 재해 (코로나19)"):
    st.markdown("**[수업용 CSV 파일]**")
    
    f_covid = load_local_file_bytes("코로나바이러스감염증-19_확진환자_발생현황_230904_최종v2.csv")
    
    if f_covid:
        st.download_button("3-1. 코로나19 발생 현황 (일별)", f_covid, "student_covid19.csv", "text/csv")

    st.markdown("**[원본 출처 링크]**")
    st.markdown("- [공공데이터포털 (코로나19)](https://www.data.go.kr/data/15079005/fileData.do)")

# --- 4. 화학/인적 재난 ---
with st.expander("🏭 4. 화학 및 인적 재난"):
    st.markdown("**[수업용 CSV 파일]**")
    
    # [중요] 이전에 정제한 화학물질 파일을 GitHub에 'my_data_chemical.csv'로 저장하세요.
    f_chemical = load_local_file_bytes("지역별_화학물질_배출량·위탁처리량_20251107022350.csv") 
    f_disaster_damage = load_local_file_bytes("자연재난_원인별_피해_20251106232109.csv")

    if f_chemical:
        st.download_button("4-1. 지역별 화학물질 배출량", f_chemical, "student_chemical_total.csv", "text/csv")
    if f_disaster_damage:
        st.download_button("4-2. 자연재난 원인별 피해 (2023)", f_disaster_damage, "student_disaster_damage.csv", "text/csv")

    st.markdown("**[원본 출처 링크]**")
    st.markdown("- [KOSIS 국가통계포털](https://kosis.kr/index/index.do)")
    st.markdown("- [국민재난안전포털 (재해연보)](https://www.safekorea.go.kr/idsiSFK/neo/main/main.html)")

st.divider()
st.caption("파일이 보이지 않을 경우, GitHub 저장소에 파일이 올바르게 업로드되었는지 확인하세요.")