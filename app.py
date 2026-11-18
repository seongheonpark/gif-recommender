"""감정 분석 + GIF 추천 Streamlit 데모 앱.

실행 (프로젝트 루트에서):
    streamlit run app.py
"""

import html
import os

import streamlit as st

from src import config
from src.giphy import recommend_gifs
from src.inference import analyze_sentence
from src.models import BACKENDS, load_finetuned_model
from src.utils import get_device, set_seed

set_seed(config.SEED)

DEFAULT_MAX_LEN = {"KoBERT": config.KOBERT_MAX_LEN, "KoELECTRA": config.KOELECTRA_MAX_LEN}
KEYBOARD_IMAGE = config.ASSETS_DIR / "keyboard.png"


############################################
# 모델 / 형태소 분석기 로드 (캐시)
############################################

@st.cache_resource(show_spinner=False)
def load_model(backend):
    return load_finetuned_model(backend, get_device())


@st.cache_resource(show_spinner=False)
def load_okt():
    from konlpy.tag import Okt

    return Okt()


############################################
# UI
############################################

def inject_chat_css():
    st.markdown("""
    <style>
    .chat-wrapper {
        max-width: 700px;
        margin: 0.5rem 0 1.0rem 0;
    }

    .chat-row {
        display: flex;
        margin-bottom: 0.8rem;
    }

    .chat-bubble {
        padding: 18px 24px;
        border-radius: 26px;
        max-width: 100%;
        font-size: 1.2rem;
        line-height: 1.6;
        word-wrap: break-word;
        word-break: keep-all;
    }

    .chat-bubble.bot {
        background: #f1f0f0;
        color: #111;
        border-bottom-left-radius: 8px;
    }

    .chat-bubble.user {
        margin-left: auto;
        background: linear-gradient(
            to bottom,
            #a855f7,
            #7c3aed
        );
        color: #fff;
        border-bottom-right-radius: 6px;
    }

    .chat-meta {
        font-size: 0.8rem;
        color: #999;
        margin-top: 0.1rem;
        margin-bottom: 0.9rem;
    }

    .keyboard-img {
        opacity: 0.9;
        margin-top: 0.6rem;
    }
    </style>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="감정 분석 + GIF 추천 서비스", layout="wide")
    inject_chat_css()

    # ===== 세션 상태 초기화 =====
    st.session_state.setdefault("sent_once", False)
    st.session_state.setdefault("last_result", None)

    # ===== 사이드바: 설정 =====
    backend = st.sidebar.radio("사용할 모델", list(BACKENDS), index=0)

    max_len = st.sidebar.slider(
        "최대 토큰 길이",
        min_value=32,
        max_value=256,
        value=DEFAULT_MAX_LEN[backend],
        step=4,
    )

    # API 키는 환경변수(또는 .env)에서 읽는다. 입력창은 다른 키로 바꿔 쓰고 싶을 때만 사용.
    env_api_key = os.getenv("GIPHY_API_KEY", "")
    user_api_key = st.sidebar.text_input(
        "GIPHY API Key",
        value="",
        type="password",
        placeholder="환경변수 GIPHY_API_KEY 사용 중" if env_api_key else "API Key를 입력하세요",
        help="비워두면 환경변수 GIPHY_API_KEY 값을 사용합니다.",
    )
    giphy_api_key = user_api_key.strip() or env_api_key

    # ===== 모델 로딩 =====
    with st.spinner("모델 로딩 중... (최초 1회, 가중치가 없으면 Hugging Face Hub에서 다운로드)"):
        try:
            model, tokenizer = load_model(backend)
            okt = load_okt()
        except Exception as e:
            st.error(
                f"{backend} 모델을 불러오지 못했습니다: {e}\n\n"
                f"`models/` 폴더에 가중치 파일이 있는지, 또는 Hugging Face Hub 저장소 "
                f"`{config.HF_MODEL_REPO}` 에 접근할 수 있는지 확인하세요. "
                f"(KoNLPy를 쓰려면 Java(JDK)도 설치되어 있어야 합니다.)"
            )
            st.stop()

    st.session_state["backend"] = backend

    # ===== 메인 영역 =====
    st.markdown("## 📩 감정 분석 + GIF 추천 서비스")

    chat_container = st.container()

    # ===== 입력창 & 버튼 =====
    default_text = "요즘 날씨도 춥고 기분도 좀 우울해."
    col_input, col_btn = st.columns([5, 1])

    with col_input:
        sentence = st.text_input("메시지 입력", value=default_text, label_visibility="collapsed")
    with col_btn:
        analyze_clicked = st.button("GIF 추천받기")

    # 1단계: 버튼 클릭 처리 & 감정 분석
    if analyze_clicked:
        if not sentence.strip():
            st.warning("문장을 입력해 주세요.")
            return

        st.session_state["sent_once"] = True

        with st.spinner("감정 분석 중..."):
            res = analyze_sentence(sentence, backend, model, tokenizer, okt, max_len)

        st.session_state["last_result"] = res

    # 2단계: 위쪽 채팅 영역 렌더
    with chat_container:
        # 1) 상대방 말풍선
        st.markdown("""
        <div class="chat-wrapper">
          <div class="chat-row">
            <div class="chat-bubble bot">
              오늘 어떤 일이 있었어?
            </div>
          </div>
        """, unsafe_allow_html=True)

        if st.session_state["last_result"] is not None:
            user_text = html.escape(st.session_state["last_result"]["sentence"])
        else:
            user_text = '<span style="opacity:0.4;">여기에 당신의 메시지가 들어갑니다</span>'

        st.markdown(f"""
          <div class="chat-row">
            <div class="chat-bubble user">
              {user_text}
            </div>
          </div>
        </div> <!-- chat-wrapper end -->
        """, unsafe_allow_html=True)

    # ===== 키보드 그림 =====
    if not st.session_state["sent_once"] and KEYBOARD_IMAGE.exists():
        st.image(str(KEYBOARD_IMAGE), width=1000)

    # ===== GIF 추천 =====
    if st.session_state["last_result"] is not None:
        result = st.session_state["last_result"]

        st.markdown("---")
        st.markdown("### 🎬 GIF 추천")

        with st.spinner("GIPHY에서 GIF 검색 중..."):
            gif_urls, msg = recommend_gifs(result, giphy_api_key, limit_per_kw=3, total_limit=12)

        st.write(msg)

        if gif_urls:
            cols = st.columns(3)
            for i, url in enumerate(gif_urls):
                with cols[i % 3]:
                    st.image(url, width=250)

    # ===== 사이드바: 감정 분석 결과 영역 =====
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 감정 분석 결과")

    if st.session_state["last_result"] is not None:
        res = st.session_state["last_result"]
        st.sidebar.markdown("**문장**")
        st.sidebar.write(res["sentence"])

        st.sidebar.markdown("**예측 감정**")
        st.sidebar.markdown(
            f"🧾 **{res['emotion_label']}**  "
            f"({res['emotion_prob'] * 100:.2f}% 확신)"
        )

        st.sidebar.markdown("**키워드**")
        if res["keywords"]:
            st.sidebar.write(", ".join(res["keywords"]))
        else:
            st.sidebar.write("_키워드가 추출되지 않았습니다._")
    else:
        st.sidebar.caption("아직 분석된 문장이 없습니다.")


if __name__ == "__main__":
    main()
