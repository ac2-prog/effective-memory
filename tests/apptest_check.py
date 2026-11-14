"""Headless check of dashboard.py with Streamlit's own AppTest (no browser needed)."""
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("dashboard.py", default_timeout=120).run()
assert not at.exception, at.exception
print("TABS:", [t.label for t in at.tabs])
print("SUBHEADERS:", [s.value for s in at.subheader])
print("WARNINGS:", [w.value for w in at.warning])
print("N markdown:", len(at.markdown), "N dataframes:", len(at.dataframe))
interp = [m.value for m in at.markdown if m.value.startswith("**Interpretation")]
limits = [m.value for m in at.markdown if m.value.startswith("**Limitation")]
print("Interpretation blocks:", len(interp), "Limitation blocks:", len(limits))
for m in interp + limits:
    print(" -", m[:400])

models = at.selectbox[0].options
for model in models:
    for thr in [0.05, 0.10, 0.50, 0.95]:
        at.selectbox[0].set_value(model)
        at.slider[0].set_value(thr)
        at.run()
        assert not at.exception, (model, thr, at.exception)
        vals = {m.label: m.value for m in at.metric}
        print(f"{model:20s} thr={thr:.2f}", vals)
print("ALL CHECKS PASSED")
