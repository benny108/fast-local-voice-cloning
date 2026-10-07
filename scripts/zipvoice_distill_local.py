#!/usr/bin/env python3
"""Local ZipVoice-Distill wrapper: explicit assets, MPS model, CPU vocoder.

No cloud request, trained voice checkpoint or added inter-sentence silence.
Upstream ZipVoice is an external dependency and is not vendored here.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

PROFILE = Path(__file__).resolve().parents[1] / "profiles/a-1.05-4steps.example.json"
FIELDS = {
    "schema_version", "profile_name", "model_name", "steps", "speed",
    "guidance_scale", "seed", "threads", "device", "vocoder_device",
    "max_duration", "pre_clamp_peak_limit",
}


def validate_profile(profile):
    if not isinstance(profile, dict):
        raise ValueError("Profile must be a JSON object")
    if set(profile) != FIELDS:
        raise ValueError("Profile fields do not match schema version 1")
    if type(profile["schema_version"]) is not int or profile["schema_version"] != 1:
        raise ValueError("Only schema_version=1 is supported")
    if not isinstance(profile["profile_name"], str) or not profile["profile_name"]:
        raise ValueError("profile_name must be a non-empty string")
    if profile["model_name"] != "zipvoice_distill":
        raise ValueError("This wrapper only supports zipvoice_distill")
    if type(profile["steps"]) is not int or profile["steps"] not in (4, 8):
        raise ValueError("steps must be 4 or 8")
    for key in ("speed", "guidance_scale", "max_duration", "pre_clamp_peak_limit"):
        value = profile[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be a positive finite number")
    if not 0.5 <= profile["speed"] <= 2.0:
        raise ValueError("speed must be between 0.5 and 2.0")
    if profile["pre_clamp_peak_limit"] >= 1:
        raise ValueError("pre_clamp_peak_limit must be below 1")
    if type(profile["threads"]) is not int or not 1 <= profile["threads"] <= 16:
        raise ValueError("threads must be an integer between 1 and 16")
    if type(profile["seed"]) is not int or not 0 <= profile["seed"] <= 2**32 - 1:
        raise ValueError("seed must be an unsigned 32-bit integer")
    if profile["device"] not in ("mps", "cpu"):
        raise ValueError("device must be mps or cpu")
    if profile["vocoder_device"] != "cpu":
        raise ValueError("This tested profile requires the CPU vocoder")
    return profile


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROFILE)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--vocos-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-text-file", type=Path, required=True)
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without loading models or writing files")
    return parser.parse_args()


def main():
    started = time.perf_counter()
    args = parse_args()
    profile = validate_profile(json.loads(args.config.read_text(encoding="utf-8")))
    repo = args.repo_root.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    vocos_dir = args.vocos_dir.expanduser().resolve()
    reference = args.reference.expanduser().resolve()
    reference_text_file = args.reference_text_file.expanduser().resolve()
    text_file = args.text_file.expanduser().resolve()
    output = args.output.expanduser().resolve()
    metadata = output.with_suffix(".json")
    if output.suffix.lower() != ".wav":
        raise ValueError("Output must be WAV; normalize/encode MP3 separately")
    if output.exists() or metadata.exists():
        raise FileExistsError("Refusing to overwrite output or metadata")
    required = [
        repo / "zipvoice/bin/infer_zipvoice.py", model_dir / "model.json",
        model_dir / "tokens.txt", model_dir / "model.pt",
        vocos_dir / "config.yaml", vocos_dir / "pytorch_model.bin",
        reference, reference_text_file, text_file,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required input is missing: {path}")
    reference_text = reference_text_file.read_text(encoding="utf-8").strip()
    text = text_file.read_text(encoding="utf-8").strip()
    if not reference_text or not text:
        raise ValueError("Reference transcript and synthesis text must not be empty")
    if args.dry_run:
        print(json.dumps({"dry_run": True, "profile": profile, "text_characters": len(text), "input_files_verified": len(required)}, ensure_ascii=False, indent=2))
        return

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    sys.path.insert(0, str(repo))
    import torch
    import torchaudio
    from lhotse.utils import fix_random_seed
    from zipvoice.bin.infer_zipvoice import generate_sentence, get_vocoder
    from zipvoice.models.zipvoice_distill import ZipVoiceDistill
    from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
    from zipvoice.utils.checkpoint import load_checkpoint
    from zipvoice.utils.feature import VocosFbank

    torch.set_num_threads(profile["threads"])
    torch.set_num_interop_threads(profile["threads"])
    fix_random_seed(profile["seed"])
    if profile["device"] == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; choose a separate CPU profile explicitly")
    device = torch.device(profile["device"])
    config = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    sampling_rate = int(config["feature"]["sampling_rate"])
    if sampling_rate != 24000:
        raise ValueError("This tested profile requires the 24 kHz model")
    tokenizer = EmiliaTokenizer(token_file=model_dir / "tokens.txt")
    model = ZipVoiceDistill(**config["model"], vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id)
    load_checkpoint(filename=model_dir / "model.pt", model=model, strict=True)
    model = model.to(device).eval()
    base_vocoder = get_vocoder(str(vocos_dir)).to("cpu").eval()
    decode = base_vocoder.decode
    decoder_gains = []

    def safe_decode(features):
        audio = decode(features.to("cpu"))
        peak = float(audio.abs().max())
        gain = min(1.0, profile["pre_clamp_peak_limit"] / max(peak, 1e-8))
        decoder_gains.append(gain)
        return audio * gain

    vocoder = SimpleNamespace(decode=safe_decode)
    extractor = VocosFbank()
    loaded = time.perf_counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        metrics = generate_sentence(
            save_path=str(output), prompt_text=reference_text,
            prompt_wav=str(reference), text=text, model=model, vocoder=vocoder,
            tokenizer=tokenizer, feature_extractor=extractor, device=device,
            num_step=profile["steps"], speed=profile["speed"],
            guidance_scale=profile["guidance_scale"],
            sampling_rate=sampling_rate, max_duration=profile["max_duration"],
            remove_long_sil=False,
        )
    finished = time.perf_counter()
    audio, rate = torchaudio.load(str(output))
    if audio.numel() == 0 or not bool(torch.isfinite(audio).all()):
        raise RuntimeError("Generated audio is empty or non-finite")
    rms = float(audio.square().mean().sqrt())
    peak = float(audio.abs().max())
    if rms <= 0.001 or peak >= 1:
        raise RuntimeError("Generated audio is silent or clipped; inspect locally")
    report = {
        "profile": profile, "text_characters": len(text),
        "sample_rate": rate, "duration_seconds": audio.shape[-1] / rate,
        "initialization_seconds": loaded - started,
        "synthesis_seconds": finished - loaded,
        "total_seconds": finished - started,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "rms": rms, "peak": peak, "finite_audio": True,
        "decoder_pre_clamp_gains": decoder_gains,
        "upstream_metrics": metrics, "quantized": False,
        "added_inter_sentence_silence": False,
        "human_listening": "pending", "speaker_similarity": "not evaluated",
        "note": "No reference recording, transcript, speaker embedding or text content is stored in this report.",
    }
    metadata.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
