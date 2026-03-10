# src/process_map.py
PROCESS_MAP = {
    31:  {"process": "CVD",   "param": "Chamber_Pressure",  "unit": "mTorr", "stage": "증착"},
    487: {"process": "ETCH",  "param": "Plasma_Power",       "unit": "W",     "stage": "식각"},
    545: {"process": "CMP",   "param": "Wafer_Temperature",  "unit": "°C",    "stage": "평탄화"},
    59:  {"process": "LITHO", "param": "Alignment_Offset",   "unit": "nm",    "stage": "노광"},
    419: {"process": "CVD",   "param": "Gas_Flow_Rate",      "unit": "sccm",  "stage": "증착"},
}

PROCESS_THRESHOLDS = {
    "CVD":   {"warning": 0.6, "critical": 0.8},
    "ETCH":  {"warning": 0.6, "critical": 0.8},
    "CMP":   {"warning": 0.5, "critical": 0.75},
    "LITHO": {"warning": 0.55, "critical": 0.78},
}

PROCESS_ORDER = ["LITHO", "CVD", "ETCH", "CMP"]

def get_sensor_label(sensor_id: int) -> str:
    if sensor_id in PROCESS_MAP:
        info = PROCESS_MAP[sensor_id]
        return f"{info['process']}_{info['param']}"
    return f"SENSOR_{sensor_id}"

def get_process_info(sensor_id: int) -> dict:
    return PROCESS_MAP.get(sensor_id, {
        "process": "UNKNOWN",
        "param": f"Sensor_{sensor_id}",
        "unit": "-",
        "stage": "미분류"
    })
