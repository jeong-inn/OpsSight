import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional

# ── 공정 파라미터 정의 (Digital Twin 물리 모델) ──────────
PROCESS_PARAMS = {
    "CVD": {
        "temp_range": (350, 420),      # 챔버 온도 (°C)
        "pressure_range": (0.1, 10.0), # 챔버 압력 (Torr)
        "flow_range": (50, 200),       # 가스 유량 (sccm)
        "drift_rate": 0.002,
    },
    "ETCH": {
        "temp_range": (20, 80),
        "pressure_range": (5, 50),
        "flow_range": (30, 150),
        "drift_rate": 0.003,
    },
    "CMP": {
        "temp_range": (20, 60),
        "pressure_range": (1, 8),
        "flow_range": (100, 300),
        "drift_rate": 0.001,
    },
    "LITHO": {
        "temp_range": (20, 25),
        "pressure_range": (0.01, 0.1),
        "flow_range": (10, 50),
        "drift_rate": 0.001,
    },
}

@dataclass
class ProcessState:
    process: str
    status: str          # normal / warning / critical
    temperature: float
    pressure: float
    flow_rate: float
    drift_factor: float
    anomaly_prob: float
    timestamp: str

class DigitalTwinSimulator:
    """
    반도체 공정 Digital Twin 시뮬레이터
    - 공정별 물리 파라미터 모델링
    - 센서 드리프트 시뮬레이션
    - 공정 상태 전이 (normal → warning → critical)
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series, window_size: int = 20):
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True)
        self.window_size = window_size
        self.pointer = 0

        # 공정별 상태 초기화
        self.process_states = {
            p: {"drift": 0.0, "status": "normal", "tick": 0}
            for p in PROCESS_PARAMS
        }

    # ── 공정 상태 시뮬레이션 ──────────────────────────────
    def simulate_process_state(self, process: str,
                                inject_anomaly: bool = False) -> ProcessState:
        params = PROCESS_PARAMS[process]
        state = self.process_states[process]

        # 드리프트 누적
        if inject_anomaly:
            state["drift"] = min(state["drift"] + 0.05, 1.0)
        else:
            state["drift"] = max(state["drift"] - 0.01, 0.0)

        drift = state["drift"]
        noise = np.random.normal(0, 0.02)

        # 파라미터 계산 (드리프트 반영)
        t_min, t_max = params["temp_range"]
        p_min, p_max = params["pressure_range"]
        f_min, f_max = params["flow_range"]

        temp = np.random.uniform(t_min, t_max) * (1 + drift * 0.3 + noise)
        pressure = np.random.uniform(p_min, p_max) * (1 + drift * 0.2 + noise)
        flow = np.random.uniform(f_min, f_max) * (1 - drift * 0.15 + noise)

        # 상태 전이
        if drift > 0.6:
            status = "critical"
            anomaly_prob = 0.85
        elif drift > 0.3:
            status = "warning"
            anomaly_prob = 0.45
        else:
            status = "normal"
            anomaly_prob = 0.05

        state["status"] = status
        state["tick"] += 1

        return ProcessState(
            process=process,
            status=status,
            temperature=round(temp, 2),
            pressure=round(pressure, 4),
            flow_rate=round(flow, 2),
            drift_factor=round(drift, 3),
            anomaly_prob=round(anomaly_prob, 3),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    def get_all_process_states(self, inject_anomaly_process: Optional[str] = None) -> dict:
        """전체 공정 상태 스냅샷 (Digital Twin 현재 상태)"""
        return {
            p: self.simulate_process_state(p, inject_anomaly=(p == inject_anomaly_process))
            for p in PROCESS_PARAMS
        }

    # ── 센서 스트림 (기존 호환) ───────────────────────────
    def get_next_window(self):
        end = min(self.pointer + self.window_size, len(self.X))
        X_window = self.X.iloc[self.pointer:end]
        y_window = self.y.iloc[self.pointer:end]
        self.pointer = end if end < len(self.X) else 0
        return X_window, y_window

    def get_random_sample(self, n: int = 20):
        idx = np.random.choice(len(self.X), size=n, replace=False)
        return self.X.iloc[idx], self.y.iloc[idx]

    def get_sensor_stream(self, n: int = 20, inject_anomaly_process: Optional[str] = None):
        """센서 스트림 + 공정 상태 동시 반환"""
        X_sample, y_sample = self.get_random_sample(n)
        process_states = self.get_all_process_states(inject_anomaly_process)
        return {
            "sensor_data": X_sample,
            "labels": y_sample,
            "process_states": process_states,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

# ── 기존 호환성 유지 ──────────────────────────────────────
SensorStreamSimulator = DigitalTwinSimulator
