import importlib.util
import io
import sys
import types
import unittest
from contextlib import contextmanager
from contextlib import redirect_stderr
from pathlib import Path


def _stub_module(tracked_modules, name, **attrs):
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
        tracked_modules.add(name)

    for attr_name, attr_value in attrs.items():
        setattr(module, attr_name, attr_value)

    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent = _stub_module(tracked_modules, parent_name)
        setattr(parent, child_name, module)

    return module


def _install_training_script_stubs(tracked_modules):
    class _DummyContextManagers:
        pass

    _stub_module(tracked_modules, "accelerate", Accelerator=object, FullyShardedDataParallelPlugin=object)
    _stub_module(tracked_modules, "accelerate.logging", get_logger=lambda *args, **kwargs: types.SimpleNamespace(info=lambda *a, **k: None))
    _stub_module(tracked_modules, "accelerate.state", AcceleratorState=object)
    _stub_module(
        tracked_modules,
        "accelerate.utils",
        ProjectConfiguration=object,
        set_seed=lambda *args, **kwargs: None,
    )

    _stub_module(tracked_modules, "diffusers", DDIMScheduler=object, FlowMatchEulerDiscreteScheduler=object)
    _stub_module(tracked_modules, "diffusers.optimization", get_scheduler=lambda *args, **kwargs: None)
    _stub_module(
        tracked_modules,
        "diffusers.training_utils",
        EMAModel=object,
        compute_density_for_timestep_sampling=lambda *args, **kwargs: None,
        compute_loss_weighting_for_sd3=lambda *args, **kwargs: None,
    )
    _stub_module(
        tracked_modules,
        "diffusers.utils",
        check_min_version=lambda *args, **kwargs: None,
        deprecate=lambda *args, **kwargs: None,
        is_wandb_available=lambda: False,
    )
    _stub_module(tracked_modules, "diffusers.utils.torch_utils", is_compiled_module=lambda *args, **kwargs: False)

    _stub_module(tracked_modules, "torch", nn=types.SimpleNamespace(Module=object))
    _stub_module(tracked_modules, "torch.nn", Module=object)
    _stub_module(tracked_modules, "torch.nn.functional")
    _stub_module(tracked_modules, "torch.utils")
    _stub_module(tracked_modules, "torch.utils.checkpoint")
    _stub_module(tracked_modules, "torch.utils.data", RandomSampler=object)
    _stub_module(tracked_modules, "torch.utils.tensorboard", SummaryWriter=object)

    _stub_module(tracked_modules, "torchvision")
    _stub_module(tracked_modules, "torchvision.transforms")
    _stub_module(tracked_modules, "torchvision.transforms.functional")

    _stub_module(tracked_modules, "transformers", AutoTokenizer=object)
    _stub_module(tracked_modules, "transformers.utils", ContextManagers=_DummyContextManagers)

    _stub_module(tracked_modules, "datasets")
    _stub_module(tracked_modules, "matplotlib")
    _stub_module(tracked_modules, "matplotlib.pyplot")
    _stub_module(tracked_modules, "numpy")
    _stub_module(tracked_modules, "einops", rearrange=lambda *args, **kwargs: None)
    _stub_module(tracked_modules, "omegaconf", OmegaConf=types.SimpleNamespace(to_container=lambda value: value))
    _stub_module(tracked_modules, "tqdm")
    _stub_module(tracked_modules, "tqdm.auto", tqdm=lambda *args, **kwargs: None)

    _stub_module(
        tracked_modules,
        "videox_fun.data.bucket_sampler",
        ASPECT_RATIO_512={},
        ASPECT_RATIO_RANDOM_CROP_512={},
        ASPECT_RATIO_RANDOM_CROP_PROB={},
        AspectRatioBatchImageVideoSampler=object,
        RandomSampler=object,
        get_closest_ratio=lambda *args, **kwargs: None,
    )
    _stub_module(
        tracked_modules,
        "videox_fun.data.dataset_image_video",
        ImageVideoControlDataset=object,
        ImageVideoCameraControlDataset=object,
        ImageVideoDataset=object,
        ImageVideoSampler=object,
        get_random_mask=lambda *args, **kwargs: None,
        process_pose_file=lambda *args, **kwargs: None,
        process_pose_params=lambda *args, **kwargs: None,
    )
    _stub_module(
        tracked_modules,
        "videox_fun.models",
        AutoencoderKLWan=object,
        CLIPModel=object,
        WanT5EncoderModel=object,
    )
    _stub_module(tracked_modules, "videox_fun.models.wan_transformer3d_world_model", WanTransformer3DModel=object)
    _stub_module(tracked_modules, "videox_fun.pipeline", WanFunControlPipeline=object)
    _stub_module(tracked_modules, "videox_fun.utils.discrete_sampler", DiscreteSampling=object)
    _stub_module(
        tracked_modules,
        "videox_fun.utils.lora_utils",
        create_network=lambda *args, **kwargs: None,
        merge_lora=lambda *args, **kwargs: None,
        unmerge_lora=lambda *args, **kwargs: None,
    )
    _stub_module(
        tracked_modules,
        "videox_fun.utils.utils",
        get_image_to_video_latent=lambda *args, **kwargs: None,
        get_video_to_video_latent=lambda *args, **kwargs: None,
        save_videos_grid=lambda *args, **kwargs: None,
    )
    _stub_module(tracked_modules, "videox_fun.utils.flowproxyloss", OnlineRAFTFlowProxy=object)


@contextmanager
def _temporary_training_script_module():
    tracked_modules = set()
    _install_training_script_stubs(tracked_modules)
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_magicworld_v1.5.py"
    spec = importlib.util.spec_from_file_location("scripts.train_magicworld_v1_5", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module

    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)
        for module_name in tracked_modules:
            sys.modules.pop(module_name, None)


class TrainMagicWorldEventArgsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._module_context = _temporary_training_script_module()
        cls.train_script = cls._module_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._module_context.__exit__(None, None, None)

    def test_parse_args_event_text_defaults(self):
        args = self.train_script.parse_args(
            [
                "--pretrained_model_name_or_path",
                "model",
                "--pretrained_transformer_path",
                "transformer",
                "--config_path",
                "config.yaml",
            ]
        )

        self.assertFalse(args.enable_event_text)
        self.assertEqual(args.text_composition_mode, "original")

    def test_parse_args_event_text_overrides(self):
        args = self.train_script.parse_args(
            [
                "--pretrained_model_name_or_path",
                "model",
                "--pretrained_transformer_path",
                "transformer",
                "--config_path",
                "config.yaml",
                "--train_data_meta",
                "train.csv",
                "--enable_event_text",
                "--text_composition_mode",
                "event_prefix",
            ]
        )

        self.assertTrue(args.enable_event_text)
        self.assertEqual(args.text_composition_mode, "event_prefix")

    def test_parse_args_requires_enable_event_text_for_non_original_composition(self):
        with self.assertRaises(SystemExit):
            self.train_script.parse_args(
                [
                    "--pretrained_model_name_or_path",
                    "model",
                    "--pretrained_transformer_path",
                    "transformer",
                    "--config_path",
                    "config.yaml",
                    "--text_composition_mode",
                    "event_prefix",
                ]
            )

    def test_parse_args_requires_train_data_meta_for_event_text(self):
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                self.train_script.parse_args(
                    [
                        "--pretrained_model_name_or_path",
                        "model",
                        "--pretrained_transformer_path",
                        "transformer",
                        "--config_path",
                        "config.yaml",
                        "--enable_event_text",
                    ]
                )

        self.assertIn("--enable_event_text requires --train_data_meta.", stderr.getvalue())

    def test_parse_args_requires_max_train_samples_for_smoke_run(self):
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                self.train_script.parse_args(
                    [
                        "--pretrained_model_name_or_path",
                        "model",
                        "--pretrained_transformer_path",
                        "transformer",
                        "--config_path",
                        "config.yaml",
                        "--smoke_run",
                    ]
                )

        self.assertIn("--smoke_run requires --max_train_samples.", stderr.getvalue())

    def test_parse_args_rejects_oversized_max_train_samples_for_smoke_run(self):
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                self.train_script.parse_args(
                    [
                        "--pretrained_model_name_or_path",
                        "model",
                        "--pretrained_transformer_path",
                        "transformer",
                        "--config_path",
                        "config.yaml",
                        "--smoke_run",
                        "--max_train_samples",
                        str(self.train_script.SMOKE_RUN_MAX_TRAIN_SAMPLES + 1),
                    ]
                )

        self.assertIn(
            f"--smoke_run requires --max_train_samples <= {self.train_script.SMOKE_RUN_MAX_TRAIN_SAMPLES}.",
            stderr.getvalue(),
        )

    def test_parse_args_requires_config_path(self):
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                self.train_script.parse_args(
                    [
                        "--pretrained_model_name_or_path",
                        "model",
                        "--pretrained_transformer_path",
                        "transformer",
                    ]
                )

        self.assertIn("the following arguments are required: --config_path", stderr.getvalue())

    def test_parse_args_rejects_unsupported_train_mode(self):
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                self.train_script.parse_args(
                    [
                        "--pretrained_model_name_or_path",
                        "model",
                        "--pretrained_transformer_path",
                        "transformer",
                        "--config_path",
                        "config.yaml",
                        "--train_mode",
                        "invalid",
                    ]
                )

        self.assertIn("invalid choice: 'invalid'", stderr.getvalue())

    def test_resolve_max_train_samples_defaults_to_full_dataset(self):
        self.assertIsNone(self.train_script.resolve_max_train_samples(args=types.SimpleNamespace(max_train_samples=None), dataset_length=12))

    def test_resolve_max_train_samples_bounds_smoke_run_sample_count(self):
        args = types.SimpleNamespace(max_train_samples=4)

        self.assertEqual(self.train_script.resolve_max_train_samples(args=args, dataset_length=12), 4)

    def test_resolve_max_train_samples_caps_to_dataset_length(self):
        args = types.SimpleNamespace(max_train_samples=20)

        self.assertEqual(self.train_script.resolve_max_train_samples(args=args, dataset_length=12), 12)
