"""Write the deferred posteriors of a finished global fit run into its existing submission folder.

A run with ``run.defer_posteriors: [gb]`` writes the submission metadata, the input data/residuals and
the posteriors of the other branches when sampling ends. This script produces the deferred ones later
(e.g. once a new GB clustering is ready), and can be re-run: the same files are overwritten.

    python scripts/postprocess_submission.py --config <run folder or run_config.yaml> \
        --submission <submission folder> [--branches gb ...] [--ess 20000]
"""

import argparse
import os
import sys

# importing run_global sets the CUDA device from --config before cupy is imported
import run_global  # noqa: F401


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-c", "--config", required=True, help="The run folder (global.yaml + blocks/) or its resolved run_config.yaml.")
    parser.add_argument("-s", "--submission", required=True, help="The submission folder written at the end of the run.")
    parser.add_argument("-b", "--branches", nargs="+", default=None, help="Branches to post-process (default: the run's defer_posteriors).")
    parser.add_argument("--ess", type=int, default=20_000, help="Number of independent samples to keep (the run uses 20000).")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)

    from mpi4py import MPI

    from lisatools.globalfit.hdfbackend import GFHDFBackend
    from lisatools.globalfit.postprocessing import RunMetadata, SubmissionWriter
    from lisatools.globalfit.run import GlobalFit
    from lisatools.globalfit.stock.l2d.io import load_run
    from lisatools.globalfit.stock.l2d.load_settings import build_curr_info

    cfg = load_run(args.config)
    branches = args.branches or list(cfg.run.defer_posteriors)
    if not branches:
        raise SystemExit("No branches to post-process: pass --branches or set run.defer_posteriors.")

    submission = os.path.abspath(args.submission)
    run_metadata = RunMetadata.from_submission(os.path.join(submission, "global_metadata.json"))
    # write next to the given folder, even if it was moved since the run
    run_metadata.submission_parent_folder = os.path.dirname(submission)
    expected = f"{run_metadata.run_type}_{run_metadata.global_fit_codename}_{run_metadata.run_id}_{run_metadata.submission_timestamp}"
    if os.path.basename(submission) != expected:
        raise SystemExit(f"{submission} does not match its global_metadata.json (expected a folder named {expected}).")

    curr = build_curr_info(cfg)
    engine_info = curr.engine_info
    state = GFHDFBackend(
        curr.general_info.main_file_path,
        sub_backend=engine_info.branch_backends,
        sub_state_bases=engine_info.branch_states,
    ).get_last_sample()
    # the end-of-run analysis containers, needed by the GB clustering
    acs = GlobalFit(curr, MPI.COMM_SELF).setup_acs(state, rebuild_residuals=True)

    writer = SubmissionWriter(curr=curr, ess=args.ess, run_metadata=run_metadata)
    writer.write_posteriors(acs, branches=branches)
    print(f"Posteriors of {branches} written to {writer.submission_folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
