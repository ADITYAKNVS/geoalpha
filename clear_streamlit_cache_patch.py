import re

with open("app.py", "r") as f:
    content = f.read()

patch = """
# ── Sidebar Memory Tools ──
with st.sidebar:
    st.markdown("---")
    st.caption("Memory Management")
    if st.button("🧹 Clear cache & Reload"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
"""

content = content.replace("with st.sidebar:", patch)

with open("app.py", "w") as f:
    f.write(content)
