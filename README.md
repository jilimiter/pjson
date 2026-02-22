# 🏎️ pjson: F1 Telemetry Analysis (1차 통합 시안)

A high-tech telemetry dashboard developed during the GMTCK internship for analyzing F1 driving data. This version focuses on the first integration of modular components and UI optimization.

---

## 📁 Project Structure
The project is organized into modular components to ensure clean separation of concerns:

```text
SW-INTERNSHIP/
├── app.py                      # Main entry point & Global layout
├── core/                       # Backend & Data processing
│   ├── __init__.py
│   ├── io_utils.py             # CSV & FastF1 data loading
│   └── plotly_render.py        # Core rendering logic
├── viewers/                    # UI Display Modules
│   ├── __init__.py
│   ├── brake_throttle_viewer.py # Brake/Throttle overlay analysis
│   ├── drs_viewer.py           # DRS & Lap status UI
│   ├── score_viewer.py         # Match performance calculations
│   ├── speed_viewer.py         # Speed gauge & comparative charts
│   └── video_viewer.py         # Reference onboard streaming
└── venv/                       # Virtual environment
'''

Key Updates (Max Branch)
Modularization: Decoupled UI logic into individual .py files in viewers/ for better maintainability.

Adaptive UI: Implemented a balanced 3-column layout (col1:col2:col3) to stabilize data monitoring.

Enhanced Visuals:

Speed: Added Ref/Target legends and gradient area fills for intuitive flow analysis.

Brake: Applied red gradient overlays to clearly contrast with green throttle lines.

DRS/Lap: Neon-digital style UI for authentic racing dashboard aesthetics.

Ideas
1. Video Playback Integration
Option A (YouTube API): Leveraging the YouTube Data API to stream official F1 onboard footage directly.

Option B (Local Storage): Utilizing local .mp4 files for low-latency playback synced with telemetry data.

2. Filling the Gap (Expansion Ideas)
To utilize the remaining dashboard space, the following modules are under consideration:

Live Weather/Track Data: Real-time track temperature and wind speed via external APIs.

G-Force Vector Map: A dynamic 2D plot showing real-time longitudinal and lateral G-forces.

Sector-wise Analysis: Detailed timing breakdowns for Sectors 1, 2, and 3 to pinpoint performance gains.

Quick Start (Using Bash)
# Clone the repository
git clone [https://github.com/jilimiter/pjson.git](https://github.com/jilimiter/pjson.git)
git checkout Max

# Run the dashboard
streamlit run app.py
