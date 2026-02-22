# utils/io_utils.py
from __future__ import annotations

import io
from typing import Optional, Tuple
import pandas as pd


def read_csv_from_upload(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    if uploaded_file is None:
        return None, "CSV 파일을 업로드한 뒤 Start를 눌러주세요."

    try:
        content = uploaded_file.read()
        df = pd.read_csv(io.BytesIO(content))
        return df, None
    except Exception as e:
        return None, f"CSV 읽기 실패: {e}"
