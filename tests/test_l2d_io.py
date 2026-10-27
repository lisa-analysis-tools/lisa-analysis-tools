"""Tests for the L2D run-folder loader (``lisatools.globalfit.stock.l2d.io``)."""

# Pre-existing env shims (see tests/test_quintic_response.py): the installed
# scienceplots raises AttributeError at import, and eryn.utils.plot uses a bare
# ``typing`` name without importing it. Both must run before importing lisatools.
import sys as _sys

_sys.modules["scienceplots"] = None

import builtins as _builtins
import typing as _typing

_builtins.typing = _typing

import tempfile
import unittest
from pathlib import Path

import yaml
from pydantic import TypeAdapter, ValidationError

from lisatools.globalfit.stock.l2d.io import Duration, RunConfig, load_run, read_yaml
from lisatools.utils.constants import YRSID_SI

# One flow-style line per section, so a test can swap or drop a whole section.
SECTIONS = {
    "schema_version": "1",
    "run": "{name: smoke, output_dir: ./out/}",
    "sampler": "{nwalkers: 8, ntemps: 2}",
    "compute": "{gpus: [0], backend: cuda12x}",
    "data": "{path: /data/l1/, source_types: [gb, mbhb], Tobs: 9 months, dt: 5 s}",
    "domain": "{type: fd, min_freq: 1e-4, max_freq: 2.9e-2}",
    "sensitivity": "{class: CompositeSensitivityBackend}",
}


# A minimal valid psd block file: the prior is required.
PSD_BLOCK = "type: psd\npriors: {class: mojitolight.stochastic.NoisePrior, kwargs: {model_type: parametric-symmetric}}\n"


def global_yaml(**changes):
    """``global.yaml`` text with sections replaced (flow-style value) or dropped (``None``)."""
    sections = {**SECTIONS, **changes}
    return "".join(f"{key}: {value}\n" for key, value in sections.items() if value is not None)


def data_section(extra=""):
    """The fixture's ``data`` section with extra flow-style fields appended."""
    return SECTIONS["data"][:-1] + extra + "}"


def error_keys(exc: ValidationError):
    """``(type, dotted.loc)`` for every error pydantic reported."""
    return [(err["type"], ".".join(map(str, err["loc"]))) for err in exc.errors()]


class _RunFolder(unittest.TestCase):
    def make_run(self, global_text=None, blocks=None, blocks_dir=True):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = Path(tmp.name)
        (d / "global.yaml").write_text(global_yaml() if global_text is None else global_text)
        if blocks_dir:
            (d / "blocks").mkdir()
        for name, text in (blocks or {}).items():
            (d / "blocks" / f"{name}.yaml").write_text(text)
        return d

    def load(self, blocks=None, **changes):
        return load_run(self.make_run(global_yaml(**changes), blocks=blocks))

    def assertLoadError(self, err_type, loc, blocks=None, msg=None, **changes):
        """Loading fails with an ``err_type`` error at ``loc`` (whose message contains ``msg``)."""
        with self.assertRaises(ValidationError) as ctx:
            self.load(blocks=blocks, **changes)
        matches = [
            err for err in ctx.exception.errors()
            if err["type"] == err_type and ".".join(map(str, err["loc"])) == loc
        ]
        self.assertTrue(matches, f"no {err_type} at {loc!r}; got {error_keys(ctx.exception)}")
        if msg is not None:
            self.assertTrue(
                any(msg in err["msg"] for err in matches),
                f"{msg!r} not in {[err['msg'] for err in matches]}",
            )


class LoadRunTest(_RunFolder):
    def test_loads_a_typed_block(self):
        cfg = self.load(blocks={"psd": PSD_BLOCK})
        self.assertEqual(list(cfg.blocks), ["psd"])
        self.assertEqual(cfg.blocks["psd"].type, "psd")
        self.assertTrue(cfg.blocks["psd"].enabled)

    # The ordering of several blocks (alphabetical by file name) is tested once a second block
    # type is registered in ``blocks.BlockSpec``.

    def test_defaults(self):
        cfg = self.load(compute=None)
        self.assertIsNone(cfg.compute.gpus)
        self.assertEqual(cfg.compute.backend, "cpu")
        self.assertEqual(cfg.run.num_iterations, 500)
        self.assertIsNone(cfg.run.submission_dir)

    def test_submission_dir(self):
        cfg = self.load(run="{name: smoke, output_dir: ./out/, submission_dir: /work/results/}")
        self.assertEqual(cfg.run.submission_dir, Path("/work/results/"))

    def test_missing_blocks_folder_means_no_blocks(self):
        self.assertEqual(load_run(self.make_run(blocks_dir=False)).blocks, {})

    def test_required_sections(self):
        for section in ("run", "sampler", "data", "domain", "sensitivity"):
            with self.subTest(section=section):
                self.assertLoadError("missing", section, **{section: None})

    def test_unknown_top_level_key_is_an_error(self):
        self.assertLoadError("extra_forbidden", "samplr", samplr="{nwalkers: 4}")

    def test_unknown_nested_key_is_an_error(self):
        self.assertLoadError("extra_forbidden", "sampler.nwalker", sampler="{nwalker: 8, ntemps: 2}")

    def test_invalid_value_is_an_error(self):
        self.assertLoadError("literal_error", "compute.backend", compute="{gpus: [0], backend: cuda99x}")
        self.assertLoadError("greater_than", "sampler.nwalkers", sampler="{nwalkers: 0, ntemps: 2}")

    def test_gpus_with_cpu_backend_is_an_error(self):
        self.assertLoadError("value_error", "compute", compute="{gpus: [0], backend: cpu}")

    def test_cuda_backend_without_gpus_is_an_error(self):
        self.assertLoadError("value_error", "compute", compute="{backend: cuda12x}")

    def test_explicit_cpu_run(self):
        cfg = self.load(compute="{backend: cpu}")
        self.assertEqual(cfg.compute.backend, "cpu")
        self.assertIsNone(cfg.compute.gpus)

    def test_wrong_schema_version(self):
        self.assertLoadError("value_error", "schema_version", schema_version="2")

    def test_blocks_in_global_yaml_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "blocks/"):
            load_run(self.make_run(global_yaml() + "blocks: {psd: {type: psd}}\n"))

    def test_block_without_type_is_an_error(self):
        self.assertLoadError("union_tag_not_found", "blocks.psd", blocks={"psd": "enabled: true\n"})

    def test_unknown_block_type_is_an_error(self):
        self.assertLoadError("union_tag_invalid", "blocks.nonesuch", blocks={"nonesuch": "type: nonesuch\n"})

    def test_block_file_stem_must_equal_its_type(self):
        # Setups, priors and the recipe key branches by block name, so the name is the type.
        self.assertLoadError(
            "value_error", "", msg="name the file psd.yaml", blocks={"noise": PSD_BLOCK}
        )

    def test_unknown_block_field_is_an_error(self):
        self.assertLoadError(
            "extra_forbidden", "blocks.psd.psd.tempring", blocks={"psd": "type: psd\ntempring: {}\n"}
        )

    def test_non_mapping_file_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "must be a mapping"):
            self.load(blocks={"psd": "- just\n- a list\n"})


class DurationTest(unittest.TestCase):
    adapter = TypeAdapter(Duration)

    def test_units(self):
        self.assertAlmostEqual(self.adapter.validate_python("9 months"), 0.75 * YRSID_SI, delta=1e-6)
        self.assertEqual(self.adapter.validate_python("200 h"), 720_000.0)
        self.assertEqual(self.adapter.validate_python("1 day"), 86_400.0)
        self.assertEqual(self.adapter.validate_python("2 days"), 172_800.0)
        self.assertEqual(self.adapter.validate_python("1 yr"), YRSID_SI)
        self.assertEqual(self.adapter.validate_python("5 s"), 5.0)

    def test_plain_number_is_seconds(self):
        self.assertEqual(self.adapter.validate_python(5), 5.0)
        self.assertEqual(self.adapter.validate_python(2.5), 2.5)

    def test_bad_values(self):
        cases = {
            "3 parsecs": "value_error",  # unknown unit
            "9months": "value_error",  # no space between number and unit
            "five s": "value_error",  # not a number
            "0 s": "greater_than",
            -1: "greater_than",
        }
        for value, err_type in cases.items():
            with self.subTest(value=value), self.assertRaises(ValidationError) as ctx:
                self.adapter.validate_python(value)
            self.assertEqual([e["type"] for e in ctx.exception.errors()], [err_type])


class DataSectionTest(_RunFolder):
    def test_dt_and_tobs_in_seconds(self):
        data = self.load().data
        self.assertEqual(data.dt, 5.0)
        self.assertAlmostEqual(data.Tobs, 0.75 * YRSID_SI, delta=1e-6)
        self.assertEqual(self.load(data=data_section().replace("5 s", "2.5")).data.dt, 2.5)

    def test_fs_is_not_a_field(self):
        self.assertLoadError("extra_forbidden", "data.fs", data=data_section(", fs: 0.2"))

    def test_preprocessing_steps_default_to_off(self):
        data = self.load().data
        self.assertIsNone(data.highpass)
        self.assertIsNone(data.lowpass)
        self.assertIsNone(data.downsample)
        self.assertIsNone(data.trim)

    def test_preprocessing_steps(self):
        data = self.load(
            data=data_section(", highpass: {cutoff: 1e-5}, downsample: {}, trim: {duration: 200 h}")
        ).data
        self.assertEqual(data.highpass.cutoff, 1e-5)
        self.assertEqual(data.highpass.order, 2)
        self.assertEqual(tuple(data.downsample.window), ("kaiser", 31.0))
        self.assertEqual(data.trim.duration, 720_000.0)
        self.assertEqual(data.trim.trimming_type, "from_each_end")

    def test_bad_trimming_type(self):
        self.assertLoadError(
            "literal_error",
            "data.trim.trimming_type",
            data=data_section(", trim: {duration: 0.02, trimming_type: middle}"),
        )

    def test_filtering_needs_an_explicit_trim(self):
        # process() would otherwise trim 100 h from each end without it showing in the files.
        for step in ("highpass", "lowpass"):
            with self.subTest(step=step):
                self.assertLoadError(
                    "value_error", "data", msg="set data.trim", data=data_section(f", {step}: {{cutoff: 1e-5}}")
                )

    def test_filtering_with_trim(self):
        data = self.load(data=data_section(", highpass: {cutoff: 1e-5}, trim: {duration: 200 h}")).data
        self.assertEqual(data.trim.duration, 720_000.0)

    def test_no_filter_no_trim(self):
        data = self.load().data
        self.assertIsNone(data.highpass)
        self.assertIsNone(data.trim)

    def test_orbits_default_to_l1(self):
        orbits = self.load().data.orbits
        self.assertEqual(orbits.type, "l1")
        self.assertEqual(orbits.frame, "icrs")
        self.assertIsNone(orbits.armlength)

    def test_l1_orbits(self):
        orbits = self.load(
            data=data_section(", orbits: {type: l1, frame: ecliptic, armlength: 2493162305.42235}")
        ).data.orbits
        self.assertEqual(orbits.frame, "ecliptic")
        self.assertEqual(orbits.armlength, 2493162305.42235)

    def test_packaged_orbits_rejected_by_mojito_loader(self):
        # A section-level value_error at ``data`` comes only from the loader/orbits check.
        for kind in ("equal_armlength", "esa"):
            with self.subTest(kind=kind):
                self.assertLoadError(
                    "value_error", "data", data=data_section(f", orbits: {{type: {kind}}}")
                )

    def test_l1_only_field_on_packaged_orbits(self):
        self.assertLoadError(
            "extra_forbidden",
            "data.orbits.equal_armlength.frame",
            data=data_section(", orbits: {type: equal_armlength, frame: icrs}"),
        )

    def test_orbits_need_a_type(self):
        self.assertLoadError(
            "union_tag_not_found", "data.orbits", data=data_section(", orbits: {frame: icrs}")
        )


class DomainTest(_RunFolder):
    def test_fd(self):
        domain = self.load().domain
        self.assertEqual(domain.type, "fd")
        self.assertEqual((domain.min_freq, domain.max_freq), (1e-4, 2.9e-2))

    def test_stft(self):
        domain = self.load(
            domain="{type: stft, min_freq: 1e-4, max_freq: 2.9e-2, segment_duration: 1 day}"
        ).domain
        self.assertEqual(domain.type, "stft")
        self.assertEqual(domain.segment_duration, 86_400.0)

    def test_stft_needs_segment_duration(self):
        self.assertLoadError(
            "missing",
            "domain.stft.segment_duration",
            domain="{type: stft, min_freq: 1e-4, max_freq: 2.9e-2}",
        )

    def test_fd_rejects_segment_duration(self):
        self.assertLoadError(
            "extra_forbidden",
            "domain.fd.segment_duration",
            domain="{type: fd, min_freq: 1e-4, max_freq: 2.9e-2, segment_duration: 1 day}",
        )

    def test_unknown_domain_type(self):
        self.assertLoadError(
            "union_tag_invalid", "domain", domain="{type: wdm, min_freq: 1e-4, max_freq: 2.9e-2}"
        )

    def test_band_must_be_ordered(self):
        for band in ("min_freq: 1e-2, max_freq: 1e-3", "min_freq: 1e-3, max_freq: 1e-3"):
            with self.subTest(band=band):
                self.assertLoadError("value_error", "domain.fd", domain=f"{{type: fd, {band}}}")


class WindowTest(_RunFolder):
    def test_defaults(self):
        window = self.load().window
        self.assertEqual(window.type, "tukey")
        self.assertEqual(window.taper_duration, "auto")
        self.assertFalse(window.normalize)

    def test_taper_duration(self):
        self.assertEqual(self.load(window="{taper_duration: 1 day}").window.taper_duration, 86_400.0)
        self.assertEqual(self.load(window="{taper_duration: 1e4}").window.taper_duration, 1e4)

    def test_unknown_window_type(self):
        self.assertLoadError("literal_error", "window.type", window="{type: triangle}")

    def test_non_positive_taper(self):
        with self.assertRaises(ValidationError) as ctx:
            self.load(window="{taper_duration: 0}")
        self.assertTrue(
            any(t == "greater_than" and loc.startswith("window.taper_duration")
                for t, loc in error_keys(ctx.exception)),
            error_keys(ctx.exception),
        )

    def test_normalize_needs_a_backend_taking_window_values(self):
        cfg = self.load(window="{normalize: true}", sensitivity="{class: XYZSensitivityBackend}")
        self.assertTrue(cfg.window.normalize)
        self.assertLoadError(
            "value_error", "", msg="window.normalize has no effect", window="{normalize: true}"
        )


class SensitivityTest(_RunFolder):
    RUN_1_KWARGS = (
        "{tdi_generation: 2, mask_percentage: 0.02, average_transfer_functions: true, "
        "galactic_grid_kwargs: {R_d: 2.18, z_d: 0.48, N_lambda: 90, N_beta: 60}}"
    )

    def test_composite_without_kwargs(self):
        sens = self.load().sensitivity
        self.assertEqual(sens.class_name, "CompositeSensitivityBackend")
        self.assertEqual(sens.kwargs, {})
        self.assertIsNone(sens.fixed_psd_kwargs)

    def test_xyz_with_run_1_kwargs(self):
        sens = self.load(
            sensitivity=f"{{class: XYZSensitivityBackend, kwargs: {self.RUN_1_KWARGS}}}"
        ).sensitivity
        self.assertEqual(sens.kwargs["mask_percentage"], 0.02)
        self.assertEqual(sens.kwargs["galactic_grid_kwargs"]["R_d"], 2.18)

    def test_run_1_kwargs_on_composite_backend(self):
        # mojito_input/run_1 passes these to the (default) composite backend -> TypeError at setup.
        self.assertLoadError(
            "value_error",
            "sensitivity",
            msg="does not accept ['average_transfer_functions', 'galactic_grid_kwargs', 'mask_percentage']",
            sensitivity=f"{{class: CompositeSensitivityBackend, kwargs: {self.RUN_1_KWARGS}}}",
        )

    def test_misspelled_kwarg(self):
        self.assertLoadError(
            "value_error",
            "sensitivity",
            msg="mask_precentage",
            sensitivity="{class: XYZSensitivityBackend, kwargs: {mask_precentage: 0.02}}",
        )

    def test_engine_filled_kwarg(self):
        self.assertLoadError(
            "value_error",
            "sensitivity",
            msg="filled in by the engine",
            sensitivity="{class: XYZSensitivityBackend, kwargs: {settings: 1}}",
        )

    def test_unknown_class(self):
        self.assertLoadError(
            "value_error",
            "sensitivity",
            msg="has no class 'NoSuchBackend'",
            sensitivity="{class: NoSuchBackend}",
        )

    def test_class_is_required(self):
        self.assertLoadError("missing", "sensitivity.class", sensitivity="{kwargs: {}}")

    def test_fixed_psd_kwargs_pass_through(self):
        sens = self.load(
            sensitivity="{class: CompositeSensitivityBackend, "
            "fixed_psd_kwargs: {psd_params: [15e-12, 3e-15], galfor_params: null}}"
        ).sensitivity
        self.assertEqual(sens.fixed_psd_kwargs, {"psd_params": [15e-12, 3e-15], "galfor_params": None})


class MetadataTest(_RunFolder):
    def test_all_optional(self):
        meta = self.load().metadata
        self.assertEqual(meta.codename, "Erebor")
        self.assertIsNone(meta.version)
        self.assertIsNone(meta.comment)

    def test_fields(self):
        meta = self.load(
            metadata="{codename: erebor, version: CDL1run1_v4, contact: ereborl2d@googlegroups.com}"
        ).metadata
        self.assertEqual(meta.version, "CDL1run1_v4")
        self.assertEqual(meta.contact, "ereborl2d@googlegroups.com")

    def test_version_needs_run_type_and_id(self):
        self.assertLoadError("string_pattern_mismatch", "metadata.version", metadata="{version: v4}")

    def test_unknown_key(self):
        self.assertLoadError("extra_forbidden", "metadata.contakt", metadata="{contakt: me}")


class PathTest(_RunFolder):
    """Relative paths in the config are relative to the run folder."""

    def load_dir(self, **changes):
        d = self.make_run(global_yaml(**changes))
        return d, load_run(d)

    def test_relative_paths_resolve_against_the_run_folder(self):
        d, cfg = self.load_dir(
            run="{name: smoke, output_dir: ./out/, submission_dir: results}",
            data=data_section().replace("/data/l1/", "l1_data/"),
        )
        self.assertEqual(cfg.run.output_dir, d / "out")
        self.assertEqual(cfg.run.submission_dir, d / "results")
        self.assertEqual(cfg.data.path, d / "l1_data")

    def test_parent_references_are_normalized(self):
        d, cfg = self.load_dir(run="{name: smoke, output_dir: ../shared/./out}")
        self.assertEqual(cfg.run.output_dir, d.parent / "shared" / "out")

    def test_absolute_paths_are_kept(self):
        _, cfg = self.load_dir()
        self.assertEqual(cfg.data.path, Path("/data/l1/"))

    def test_home_is_expanded(self):
        _, cfg = self.load_dir(run="{name: smoke, output_dir: ~/gf_out}")
        self.assertEqual(cfg.run.output_dir, Path.home() / "gf_out")

    def test_no_run_folder_no_resolution(self):
        raw = yaml.safe_load(global_yaml())
        cfg = RunConfig.model_validate(raw)  # no validation context
        self.assertEqual(cfg.run.output_dir, Path("out"))


class DuplicateKeyTest(_RunFolder):
    def assertDuplicate(self, key, global_text=None, blocks=None):
        with self.assertRaises(yaml.YAMLError) as ctx:
            load_run(self.make_run(global_text, blocks=blocks))
        self.assertIn(f"duplicate key {key!r}", str(ctx.exception))
        return str(ctx.exception)

    def test_duplicate_nested_key(self):
        text = global_yaml(compute="{gpus: [0], backend: cuda12x, backend: cpu}")
        message = self.assertDuplicate("backend", text)
        self.assertIn("global.yaml", message)  # points at the file

    def test_duplicate_section(self):
        self.assertDuplicate("sampler", global_yaml() + "sampler: {nwalkers: 2, ntemps: 1}\n")

    def test_duplicate_in_block_file(self):
        message = self.assertDuplicate("type", blocks={"psd": "type: a\ntype: b\n"})
        self.assertIn("psd.yaml", message)

    def test_merge_keys_still_work(self):
        path = self.make_run(blocks_dir=False) / "merge.yaml"
        path.write_text("defaults: &d {x: 1, y: 2}\nsettings: {<<: *d, y: 3}\ntype: psd\n")
        self.assertEqual(read_yaml(path)["settings"], {"x": 1, "y": 3})


class ReadYamlTest(_RunFolder):
    def read(self, text):
        path = self.make_run(blocks_dir=False) / "x.yaml"
        path.write_text(text)
        return read_yaml(path)

    def test_exponent_without_dot_is_a_float(self):
        # PyYAML's own resolver reads `1e-4` as a string.
        data = self.read("a: 1e-4\nb: -2E5\nc: 1.5e3\nd: 7\n")
        for key, expected in dict(a=1e-4, b=-2e5, c=1.5e3).items():
            with self.subTest(key=key):
                self.assertIsInstance(data[key], float)
                self.assertEqual(data[key], expected)
        self.assertIsInstance(data["d"], int)

    def test_empty_file_is_an_empty_mapping(self):
        self.assertEqual(self.read(""), {})


class SnapshotTest(_RunFolder):
    """``dump_run`` writes one resolved file that ``load_run`` reads back unchanged."""

    FULL = dict(
        run="{name: full, output_dir: ./out/, submission_dir: results, seed: 7}",
        data=data_section(
            ", source_ids: {mbhb: [18, 5]}, orbits: {type: l1, armlength: 2493162305.42235}"
            ", highpass: {cutoff: 1e-5}, downsample: {}, trim: {duration: 0.02, is_percent: true}"
        ),
        domain="{type: stft, min_freq: 1e-4, max_freq: 2.9e-2, segment_duration: 1 day}",
        window="{taper_duration: auto, normalize: true}",
        sensitivity=(
            "{class: XYZSensitivityBackend, fixed_psd_kwargs: {psd_params: [15e-12, 3e-15]}, "
            "kwargs: {mask_percentage: 0.02, galactic_grid_kwargs: {R_d: 2.18, z_d: 0.48}}}"
        ),
        metadata="{version: CDL1run1_v4, comment: snapshot test}",
    )
    BLOCKS = {
        "psd": (
            "type: psd\n"
            "enabled: false\n"
            "num_prop_repeats: 10\n"
            "tempering: {ntemps: 3, ratio: 1.5}\n"
            "injection: {S_oms: 1.5e-11, S_tm: 3e-15}\n"
            "search: {search_type: prior, spread: [1e-4, 1e-3]}\n"
            "priors: {class: mojitolight.stochastic.NoisePrior, kwargs: {model_type: parametric-symmetric}}\n"
        )
    }

    def dump(self, cfg, path):
        from lisatools.globalfit.stock.l2d.io import dump_run

        return dump_run(cfg, path)

    def snapshot(self):
        cfg = load_run(self.make_run(global_yaml(**self.FULL), blocks=self.BLOCKS))
        out = Path(self.make_run(blocks_dir=False)) / "nested" / "dir" / "config.yaml"
        self.dump(cfg, out)
        return cfg, out

    def test_round_trip(self):
        cfg, out = self.snapshot()
        self.assertTrue(out.is_file())  # parent folders created
        self.assertEqual(load_run(out), cfg)

    def test_snapshot_is_fully_resolved(self):
        cfg, out = self.snapshot()
        raw = yaml.safe_load(out.read_text())
        self.assertEqual(raw["sensitivity"]["class"], "XYZSensitivityBackend")  # alias, not class_name
        self.assertEqual(raw["data"]["Tobs"], cfg.data.Tobs)  # durations in seconds
        self.assertEqual(raw["run"]["num_iterations"], 500)  # defaults written out
        self.assertEqual(raw["run"]["output_dir"], str(cfg.run.output_dir))  # absolute path
        self.assertTrue(Path(raw["run"]["output_dir"]).is_absolute())
        psd = raw["blocks"]["psd"]
        self.assertEqual(psd["priors"]["class"], "mojitolight.stochastic.NoisePrior")  # alias, not class_name
        self.assertEqual(psd["tempering"], {"ntemps": 3, "ratio": 1.5})
        self.assertEqual(psd["inner_moves"], None)  # defaults written out

    def test_single_file_may_contain_blocks(self):
        _, out = self.snapshot()
        psd = load_run(out).blocks["psd"]
        self.assertFalse(psd.enabled)
        self.assertEqual(psd.search.spread, [1e-4, 1e-3])
        self.assertEqual(psd.tempering.ratio, 1.5)

    def test_single_file_relative_paths_use_its_folder(self):
        d = self.make_run(blocks_dir=False)
        (d / "run.yaml").write_text(global_yaml())
        self.assertEqual(load_run(d / "run.yaml").run.output_dir, d / "out")


if __name__ == "__main__":
    unittest.main()
