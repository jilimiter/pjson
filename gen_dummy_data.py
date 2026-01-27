import numpy as np
import pandas as pd

def generate_dummy_data(
    duration_s: float = 90.0,
    freq_hz: float = 20.0,
    track_radius_m: float = 200.0,
    n_laps: float = 3.0,
) -> pd.DataFrame:
    """
    duration_s 동안 freq_hz 샘플링으로
    time / brake / throttle / speed / drs / x / y 시계열을 더미로 생성.

    x, y 는 8자(∞) 모양 트랙을 n_laps 바퀴 도는 궤적.
    """
    n_samples = int(duration_s * freq_hz) + 1
    t = np.linspace(0, duration_s, n_samples)

    # Throttle: 전체적으로 40~100% 사이에서 Sine + 노이즈
    throttle = 70 + 30 * np.sin(2 * np.pi * t / 20.0) + np.random.normal(0, 5, size=t.shape)
    throttle = np.clip(throttle, 0, 100)

    # Brake: 특정 구간에서만 제동 (예: 코너 3개 있다고 가정)
    brake = np.zeros_like(t)
    for start, end in [(10, 13), (35, 38), (60, 63)]:
        mask = (t >= start) & (t <= end)
        segment_t = (t[mask] - start) / (end - start)
        brake[mask] = 100 * np.sin(np.pi * segment_t)
    brake += np.random.normal(0, 3, size=t.shape)
    brake = np.clip(brake, 0, 100)

    # Speed: 기본 속도 곡선 + 노이즈 (브레이크 구간에서 속도 하강)
    base_speed = 250 - 40 * np.sin(2 * np.pi * t / 45.0)
    speed = base_speed - 0.7 * brake + np.random.normal(0, 3, size=t.shape)
    speed = np.clip(speed, 60, 330)

    # DRS: 특정 구간에서만 ON (0/1)
    drs = np.zeros_like(t)
    for start, end in [(5, 9), (25, 32), (50, 58), (75, 85)]:
        mask = (t >= start) & (t <= end)
        drs[mask] = 1

    # === 8자(∞) 트랙 상의 (x, y) 궤적 생성 ===
    # theta 를 0 -> 2π * n_laps 로 선형 증가
    theta = 2.0 * np.pi * n_laps * (t / duration_s)

    # Lemniscate of Gerono 파라메트릭 곡선
    # x = R cos(theta)
    # y = (R/2) sin(2 theta)
    R = track_radius_m
    x = R * np.cos(theta)
    y = 0.5 * R * np.sin(2.0 * theta)

    df = pd.DataFrame(
        {
            "time_s": t,
            "brake_pct": brake,
            "throttle_pct": throttle,
            "speed_kph": speed,
            "drs": drs.astype(int),
            "x_m": x,
            "y_m": y,
        }
    )
    return df

if __name__ == "__main__":
    df = generate_dummy_data()
    out_path = "dummy_telemetry_2.csv"
    df.to_csv(out_path, index=False)
    print(f"Dummy telemetry written to {out_path}")