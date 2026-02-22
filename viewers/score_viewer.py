import streamlit as st

def render_similarity_score(curr):
    st.markdown("""
    <style>
    @import url('https://fonts.cdnfonts.com/css/ds-digital');
    .digital-dashboard { background-color: #000; border: 4px solid #333; border-radius: 20px; padding: 15px; text-align: center; font-family: 'DS-Digital', sans-serif; color: white; margin-bottom: 10px; }
    .score-big { font-size: 80px; font-weight: bold; line-height: 80px; }
    </style>
    """, unsafe_allow_html=True)

    score = curr['score']
    color = "#00ff00" if score >= 90 else ("#ffffff" if score >= 70 else "#ff0000")
    
    st.markdown(f"""
    <div class="digital-dashboard">
        <div style="color: #888; font-size: 18px;">MATCH PERFORMANCE</div>
        <div class="score-big" style="color: {color}; text-shadow: 0 0 15px {color};">{int(score):02d}%</div>
        <div style="color: #666; font-size: 16px;">DISTANCE: {int(curr['dist'])} M</div>
    </div>
    """, unsafe_allow_html=True)