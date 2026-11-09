from lisatools.globalfit.stock.l2d.io import load_run, dump_run, RunConfig
from lisatools.globalfit.stock.l2d.load_settings import load_blocks, load_general_setup, make_rank_info

PATH = "/Users/alessandrosantini/reps/globalift/erebor_org_setup/lisa-analysis-tools/gf_dev/interface/test"

run_cfg = load_run(PATH)

general_setup = load_general_setup(run_cfg)
blocks = load_blocks(run_cfg, general_setup)

breakpoint()

dump_run(run_cfg, "/Users/alessandrosantini/reps/globalift/erebor_org_setup/lisa-analysis-tools/gf_dev/interface/test_loading.yaml")
