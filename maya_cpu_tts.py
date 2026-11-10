"""CPU text-to-speech runner for Maya 2 Native (named Hindi voices) on a plain GitHub runner.

Uses the model's own eager reference code (no CUDA graphs, no Triton) and the production
engine's sampling rules (controlled stop, presence penalty, per-piece length caps).

    python maya_cpu_tts.py --src /path/to/maya2-native --lines lines.json --out outdir

lines.json: [{"id": "b01", "voice": "Ritvik", "text": "..."}, ...]
Writes outdir/<id>.wav (24 kHz mono 16-bit) and outdir/result.json (times and real-time factor).
"""
import argparse
import json
import os
import sys
import time
import wave

import numpy as np

SR, FRAME, FPS = 24000, 1920, 12.5


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save_wav(path, w):
    with wave.open(path, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(SR)
        fh.writeframes((np.clip(w, -1, 1) * 32767).astype("<i2").tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="checkout of MayaResearch/maya2-native")
    ap.add_argument("--lines", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="maya-research/maya2-native")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--q", type=int, default=32, help="codebooks (32 = production engine)")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--max-frames", type=int, default=0, help="debug: cap frames per piece")
    ap.add_argument("--chunk-s", type=float, default=20.0)
    A = ap.parse_args()

    sys.path.insert(0, A.src)
    sys.path.insert(1, os.path.join(A.src, "scripts"))
    import torch
    torch.set_num_threads(A.threads)
    torch.manual_seed(A.seed)
    dt = torch.float32 if A.dtype == "fp32" else torch.bfloat16
    os.makedirs(A.out, exist_ok=True)
    lines = json.load(open(A.lines, encoding="utf-8"))

    from maya3 import hub
    from maya3 import guards as G
    from maya3.prompt import VoiceRegistry, load_named_voices
    from scripts import s2_serve_controls as S
    from tts.data.packing_v2 import PackerV2
    from tts.model.load_s3 import load_model_for_eval
    from tts.tokens import AEOS_CLASS
    from transformers import MimiModel

    t_start = time.perf_counter()
    r = hub.resolve(A.repo)
    log("resolved release", r["ckpt"])
    mcfg = {"model": {"backbone": r["backbone_dir"], "attn": "sdpa"}, "data": {"max_tokens": 8192}}
    model, prov = load_model_for_eval(r["ckpt"], device="cpu", dtype=dt, cfg=mcfg,
                                      config=os.path.join(A.src, "configs/s3_fresh_v1.yaml"),
                                      amortize_frac=1.0, require_anchor=True, verbose=False)
    model.depth.to(torch.float32)
    model.eval()
    mimi = MimiModel.from_pretrained(r["mimi_dir"]).eval()
    packer = PackerV2(model.backbone.tokenizer, model.backbone.token_ids, max_tokens=8192)
    prior = S.load_prior(os.path.join(A.src, "configs/text_frames_prior.json"))
    dur = G.DurationPrior()
    reg = VoiceRegistry()
    load_named_voices(reg, r["voices_json"])
    load_s = time.perf_counter() - t_start
    log(f"model loaded in {load_s:.0f}s dtype={A.dtype} q={A.q} threads={A.threads}")

    try:
        from maya3.text import normalise_text
    except Exception:
        normalise_text = lambda t, lang: t

    # speaker memory is nulled for named voices (same as the engine)
    refs = torch.zeros(1, 1, 192, dtype=torch.float32)
    mem = model.depth.spk_memory(refs, "ecapa", null_mask=torch.tensor([True]))
    rmask = torch.ones(1, 1, dtype=torch.bool)

    gen = torch.Generator().manual_seed(A.seed)

    @torch.inference_mode()
    def speak(prompt_emb, cap, minimum, tag):
        """Engine sampling: cb0 temp 0.9 / top-k 64 / presence penalty 0.8 over the last 12 frames,
        controlled stop (argmax AEOS, p >= 0.35, hazard >= 0.6 after `minimum` frames)."""
        past = None
        t0 = time.perf_counter()
        h, past = model.backbone(prompt_emb.to(dt), past_key_values=None, use_cache=True)
        h = h[:, -1]
        t_pre = time.perf_counter() - t0
        out, hist, hz = [], [], 0.0
        reason = "cap"
        t_bb = t_dp = 0.0
        for t in range(cap):
            logits = model.backbone.cb0_logits(h).float()[0] / 0.9
            if hist:
                pen = torch.zeros_like(logits)
                for c in set(hist):
                    pen[c] = 1.0
                pen[AEOS_CLASS] = 0
                logits = logits - 0.8 * pen
            kth = logits.topk(64).values[-1]
            logits = logits.masked_fill(logits < kth, -float("inf"))
            p = logits.softmax(-1)
            pa = float(p[AEOS_CLASS])
            if t >= minimum:
                hz = 1.0 - (1.0 - hz) * (1.0 - pa)
                if int(p.argmax()) == AEOS_CLASS:
                    reason = "aeos_argmax"; break
                if pa >= 0.35:
                    reason = "aeos_tau"; break
                if hz >= 0.6:
                    reason = "aeos_hazard"; break
            q = p.clone()
            q[AEOS_CLASS] = 0.0
            q = q / q.sum()
            cb0 = int(torch.multinomial(q, 1, generator=gen))
            hist = (hist + [cb0])[-12:]
            t1 = time.perf_counter()
            frame = S.depth_generate(model.depth, h.float(), torch.tensor([cb0]), mem, A.q,
                                     0.7, 64, 0.0, rmask)[0]
            t2 = time.perf_counter()
            out.append(frame)
            fb = model.audio_in.frames(frame[None, :], frame.shape[0])
            h, past = model.backbone(fb[None].to(dt), past_key_values=past, use_cache=True)
            h = h[:, -1]
            t3 = time.perf_counter()
            t_dp += t2 - t1
            t_bb += t3 - t2
            if (t + 1) % 25 == 0:
                log(f"  [{tag}] frame {t + 1}  depth {t_dp / (t + 1):.2f}s/f  backbone {t_bb / (t + 1):.2f}s/f  "
                    f"RSS {rss_gb():.1f} GB")
        if reason.startswith("aeos"):
            pass
        if not out:
            return torch.zeros(A.q, 0, dtype=torch.long), reason
        return torch.stack(out, -1), reason

    def rss_gb():
        try:
            for ln in open("/proc/self/status"):
                if ln.startswith("VmRSS"):
                    return int(ln.split()[1]) / 1e6
        except Exception:
            pass
        return 0.0

    results = []
    tot_wall = tot_audio = 0.0
    for ln in lines:
        lid, voice_name, text = ln["id"], ln["voice"], normalise_text(ln["text"], "hi")
        voice = reg.get(voice_name)
        voice.spk_refs = np.zeros((1, 192), np.float32)
        pieces = S.split_text(text, "hi", prior, A.chunk_s)
        t_line = time.perf_counter()
        wavs, stops = [], []
        for k, piece in enumerate(pieces):
            ex = voice.builder.example(packer, voice, piece, "hi")
            emb, _ = S.prompt_from_example(ex, model, "cpu", dt, 32)
            ef, er = S.expected_frames(prior, "hi", len(piece))
            cap = min(1500, int(2.0 * (ef + er) + 40))
            cap = min(cap, max(int(ef * 0.5) + 8, dur.cap_frames(piece, "hi", 2.2, 1.2)))
            if A.max_frames:
                cap = min(cap, A.max_frames)
            minimum = max(4, int(0.5 * ef))
            codes, reason = speak(emb, cap, minimum, f"{lid}.{k + 1}")
            t_dec = time.perf_counter()
            if codes.shape[-1]:
                with torch.inference_mode():
                    w = mimi.decode(codes[None].long())[0][0, 0].float().numpy()
            else:
                w = np.zeros(0, np.float32)
            log(f"  [{lid}.{k + 1}] mimi decode {time.perf_counter() - t_dec:.1f}s")
            wavs.append(w)
            stops.append({"reason": reason, "frames": int(codes.shape[-1]), "cap": cap})
            if k + 1 < len(pieces):
                wavs.append(np.zeros(int(0.15 * SR), np.float32))
        w = np.concatenate(wavs) if wavs else np.zeros(0, np.float32)
        wall = time.perf_counter() - t_line
        audio = len(w) / SR
        save_wav(os.path.join(A.out, f"{lid}.wav"), w)
        tot_wall += wall
        tot_audio += audio
        rec = {"id": lid, "voice": voice_name, "chars": len(text), "audio_s": round(audio, 2),
               "wall_s": round(wall, 1), "rtf": round(wall / max(audio, 1e-6), 2), "stops": stops}
        results.append(rec)
        log("LINE", json.dumps(rec, ensure_ascii=False))
        json.dump({"load_s": round(load_s, 1), "dtype": A.dtype, "q": A.q, "threads": A.threads,
                   "total_wall_s": round(tot_wall, 1), "total_audio_s": round(tot_audio, 2),
                   "rtf": round(tot_wall / max(tot_audio, 1e-6), 2), "lines": results},
                  open(os.path.join(A.out, "result.json"), "w"), ensure_ascii=False, indent=1)
    log(f"DONE lines={len(results)} wall={tot_wall:.0f}s audio={tot_audio:.1f}s rtf={tot_wall / max(tot_audio, 1e-6):.2f} "
        f"peakRSS~{rss_gb():.1f}GB")


if __name__ == "__main__":
    main()
