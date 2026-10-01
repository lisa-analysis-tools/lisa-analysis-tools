"""EMRIDirectWDM.batch == per-template EMRIDirectWDM.__call__ (laptop, CD1L EMRI 1, 16 d); prints max rel diff per row and ms/template. Env ROWS, NT."""
import os, sys, time
import numpy as np
sys.path.insert(0, "scripts/emri")
import emri_tof_xyz_threeway as W
import emri_batch_speed as B
NF, DT, NT = 180, 20.0, int(os.environ.get("NT", "384"))
W.N_WIN = NF * NT
from lisatools.domains import WDMLookupTable, WDMSettings
from lisatools.response.tdiconfig import TDIConfig
from lisatools.sources.emri.wdm_direct import EMRIDirectWDM
params, data, data_t0, orb = W.load(1)
wg, off, gen = W.legacy_wrapper(orb, data_t0)
table = WDMLookupTable.from_file("/Users/mkatz/Research/lisa_sprint_2026/wdm_lookup_emri_cx_NF180_DT20_TL32_fd8x0p01_nld2.h5", force_backend="cpu")
wdm = WDMSettings(NF, NT, DT, force_backend="cpu")
d = EMRIDirectWDM(gen, table, wdm, orbits=orb, tdi_config=TDIConfig("2nd generation", force_backend="cpu"), t_start=W.REF, data_t0=data_t0)
rows = B.batch_rows(params, int(os.environ.get("ROWS", "4")))
t0 = time.perf_counter(); single = [np.asarray(d(*r, mode_selection_threshold=1e-3).arr) for r in rows]; ts = time.perf_counter() - t0
t0 = time.perf_counter(); bat = np.asarray(d.batch(rows, mode_selection_threshold=1e-3)); tb = time.perf_counter() - t0
print("stats", d.last_stats)
for r, s_ in enumerate(single):
    print(f"row {r}: max|batch-single|/max|single| = {np.max(np.abs(bat[r]-s_))/np.max(np.abs(s_)):.2e}")
print(f"single {ts/len(rows)*1e3:.0f} ms/tmpl  batch {tb/len(rows)*1e3:.0f} ms/tmpl")
