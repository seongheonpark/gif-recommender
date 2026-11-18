"""감정 + 키워드로 GIPHY에서 GIF를 검색해 추천한다."""

import logging
import random
import re
from typing import Dict, List, Optional, Tuple

import requests

from . import config

logger = logging.getLogger(__name__)

# 앞에서부터 순서대로 시도할 GIPHY 이미지 크기 (일부 결과는 downsized가 빈 값으로 온다)
RENDITION_PREFERENCE = ("downsized", "downsized_medium", "fixed_height", "original")


def gif_url(item: Dict) -> Optional[str]:
    """GIPHY 검색 결과 하나에서 쓸 수 있는 GIF URL을 고른다. 없으면 None."""
    images = item.get("images") or {}
    for name in RENDITION_PREFERENCE:
        url = (images.get(name) or {}).get("url")
        if url:
            return url
    return None


def build_search_keywords(result: Dict) -> List[str]:
    """Okt 키워드 + 한국어 감정 라벨 + 영어 감정 라벨을 순서를 유지한 채 중복 없이 합친다."""
    emotion_ko = result["emotion_label"]
    emotion_en = config.EMOTION_TO_EN.get(emotion_ko, "emotion")
    keywords = [*result["keywords"], emotion_ko, emotion_en]
    return list(dict.fromkeys(keywords))


def recommend_gifs(
    result: Dict,
    api_key: str,
    limit_per_kw: int = 3,
    total_limit: Optional[int] = 12,
    rating: str = "pg-13",
    timeout: float = 5,
) -> Tuple[List[str], str]:
    """키워드마다 GIF를 최대 limit_per_kw개씩 무작위로 골라 (URL 목록, 안내 메시지)를 반환한다."""
    if not api_key:
        return [], "GIPHY API Key가 설정되지 않았습니다."

    all_urls = []
    for kw in build_search_keywords(result):
        params = {"q": kw, "api_key": api_key, "limit": "25", "rating": rating}
        try:
            resp = requests.get(config.GIPHY_SEARCH_URL, params=params, timeout=timeout)
            if resp.status_code != 200:
                logger.warning("GIPHY 응답 오류 (kw=%s, status=%s)", kw, resp.status_code)
                continue
            results = resp.json().get("data", [])
        except (requests.RequestException, ValueError) as e:
            logger.warning("GIPHY 검색 중 에러 (kw=%s): %s", kw, e)
            continue

        urls = [url for url in map(gif_url, results) if url]
        if not urls:
            continue

        all_urls.extend(random.sample(urls, min(limit_per_kw, len(urls))))

    if not all_urls:
        return [], "GIF 검색 결과가 없습니다."

    all_urls = list(dict.fromkeys(all_urls))
    if total_limit is not None and len(all_urls) > total_limit:
        all_urls = random.sample(all_urls, total_limit)

    picked = [re.sub(r"media\d", "i", url) for url in all_urls]
    return picked, f"총 {len(picked)}개 GIF 추천"
