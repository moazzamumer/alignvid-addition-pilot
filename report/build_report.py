#!/usr/bin/env python3
"""Phase E: build the two deliverable PDFs from the repository state.

  PDF 1  report/pilot_report.pdf     header + versions, summary, panels (measure/panel.py output),
                                     tables T1 (human labels), T2 (human vs detector), T4 (runtime,
                                     memory), T5 (failures), "where it broke down", deviations (§9)
  PDF 2  report/revised_protocol.pdf report/revised_protocol.md with the measured cost table inserted
  CSV    report/tables/T1_labels.csv, T2_human_vs_detector.csv, T4_runtime_memory.csv, cost_table.csv

Prose comes from report/sections/*.md; everything numeric is read from labels/, outputs/, logs/ and
protocol/pilot_protocol.md.  HTML is printed to PDF with headless Chrome.

  python report/build_report.py            # both PDFs
  python report/build_report.py --no-pdf   # HTML and CSV only"""
import argparse, csv, datetime, glob, json, os, re, statistics, subprocess, sys
import markdown

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "report")
ARMS = ["vanilla", "alignvid"]
IDS = ["sample_240", "sample_188", "sample_208", "sample_245", "sample_116"]
LABEL_FIELDS = ["presence", "onset_frame", "persistence", "placement", "object_correct", "preservation",
                "primary_failure", "secondary_tags", "within_window", "scorable"]
CSS = """
@page { size: A4 landscape; margin: 12mm 14mm; }
body { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 10.5pt; color: #111; line-height: 1.35; }
h1 { font-size: 17pt; margin: 0 0 4px 0; } h2 { font-size: 13.5pt; margin: 14px 0 6px 0; border-bottom: 1px solid #999; }
h3 { font-size: 11.5pt; margin: 10px 0 4px 0; }
table { border-collapse: collapse; margin: 6px 0 10px 0; font-size: 9pt; width: 100%; } table.narrow { width: auto; } td.wide { min-width: 240px; } th, td { border: 1px solid #bbb; padding: 3px 6px; vertical-align: top; text-align: left; }
th { background: #eee; } code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.5pt; }
.small { font-size: 9pt; color: #333; } .page { page-break-before: always; } img.panel { width: 100%; }
.note { background: #f4f4f4; padding: 6px 8px; border-left: 3px solid #888; }
"""


def md(text):
    return markdown.markdown(text, extensions=["tables"])


def read(p):
    return open(os.path.join(ROOT, p), encoding="utf-8").read()


def section(name):
    p = os.path.join(REP, "sections", name)
    return md(open(p, encoding="utf-8").read()) if os.path.exists(p) else f"<p class='note'>section {name} missing</p>"


def protocol_info():
    t = read("protocol/pilot_protocol.md")
    on = re.search(r"\*\*Frozen on:\*\*\s*([^\n]+)", t); at = re.search(r"\*\*Frozen at commit:\*\*\s*`?([0-9a-f]{7,40})", t)
    rows = []
    sec9 = t.split("## 9. Deviations")[1] if "## 9. Deviations" in t else ""
    for ln in sec9.splitlines():
        if ln.startswith("| 20"):
            cells = [c.strip() for c in ln.strip().strip("|").split(" | ")]
            if len(cells) >= 3:
                rows.append((cells[0], " | ".join(cells[1:-1]), cells[-1]))
    sec2 = t.split("## 2. Versions")[1].split("## 3.")[0]
    return {"frozen_on": on.group(1).strip() if on else "?", "frozen_commit": at.group(1) if at else "?", "deviations": rows, "versions_md": sec2}


def blind_map():
    p = os.path.join(ROOT, "labels", "blind_map.csv")
    return {r["video_code"]: (r["arm"], r["id"]) for r in csv.DictReader(open(p))} if os.path.exists(p) else {}


def labels():
    """-> {labeller: {(arm,id): row}} for rows with a presence value."""
    bm = blind_map(); out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "labels", "labels_*.csv"))):
        if p.endswith("labels_template.csv"):
            continue
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r.get("presence", "").strip() and r.get("video_code") in bm:
                out.setdefault(r.get("labeller") or os.path.basename(p)[7:-4], {})[bm[r["video_code"]]] = r
    return out


def detector():
    out = {}
    for arm in ARMS:
        for p in glob.glob(os.path.join(ROOT, "outputs", arm, "detect", "sample_*.json")):
            d = json.load(open(p)); out[(arm, d["id"])] = d
    return out


def sidecars():
    out = {}
    for arm in ARMS:
        for sid in IDS:
            p = os.path.join(ROOT, "outputs", arm, f"{sid}.json")
            if os.path.exists(p):
                out[(arm, sid)] = json.load(open(p))
    return out


def env_info():
    p = os.path.join(ROOT, "logs", "kaggle", "sessionA", "env_kaggle.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def html_table(header, rows, cls="", wide_last=False):
    h = "".join(f"<th>{c}</th>" for c in header)
    def td(i, c):
        return f"<td class='wide'>{c}</td>" if wide_last and i == len(header) - 1 else f"<td>{'' if c is None else c}</td>"
    b = "".join("<tr>" + "".join(td(i, c) for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f"<table class='{cls}'><tr>{h}</tr>{b}</table>"


def write_csv(name, header, rows):
    with open(os.path.join(REP, "tables", name), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def kappa(a, b):
    """Cohen's kappa for two equal-length label lists."""
    n = len(a)
    if n == 0:
        return None
    cats = sorted(set(a) | set(b)); po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return None if pe == 1 else round((po - pe) / (1 - pe), 3)


def t1(lab):
    header = ["id", "arm", "labeller"] + LABEL_FIELDS + ["notes"]; rows = []
    for sid in IDS:
        for arm in ARMS:
            for lb in sorted(lab):
                r = lab[lb].get((arm, sid))
                if r:
                    rows.append([sid, arm, lb] + [r.get(f, "") for f in LABEL_FIELDS] + [r.get("notes", "")])
    return header, rows


def agreement(lab):
    if len(lab) < 2:
        return None
    (la, A), (lb, B) = sorted(lab.items())[:2]
    keys = [k for k in A if k in B]
    if not keys:
        return None
    out = {"labellers": (la, lb), "n": len(keys), "fields": {}}
    for f in LABEL_FIELDS:
        out["fields"][f] = round(sum(1 for k in keys if A[k].get(f, "").strip().lower() == B[k].get(f, "").strip().lower()) / len(keys), 2)
    out["kappa_presence"] = kappa([A[k]["presence"].strip().lower() for k in keys], [B[k]["presence"].strip().lower() for k in keys])
    return out


def t2(lab, det, notes):
    """Human (first labeller) vs detector, one row per video; disagreement column lists the fields."""
    ref = lab[sorted(lab)[0]] if lab else {}
    header = ["id", "arm", "field", "human", "detector", "detector facts", "note"]; rows = []
    for sid in IDS:
        for arm in ARMS:
            d = det.get((arm, sid)); h = ref.get((arm, sid))
            if not d:
                continue
            facts = (f"{d['n_detected']}/16 frames detected (k = {d['thresholds']['k']}); "
                     f"best score {d['best_box']['score'] if d.get('best_box') else '-'}; "
                     f"reference boxes {len(d['reference_dets'])}; pre-existing boxes flagged {sum(1 for p in d['per_frame'] for x in p['dets'] if x.get('pre_existing'))}")
            dp = "yes" if d["present"] else "no"
            # onset / persistence / placement are asserted by the detector only when it calls the object present
            pairs = [("presence", (h or {}).get("presence", "pending"), dp),
                     ("onset_frame", (h or {}).get("onset_frame", "pending"), (str(d["onset_index"]) if d.get("onset_index") is not None else "n.a.") if d["present"] else "n.a."),
                     ("persistence", (h or {}).get("persistence", "pending"), (d["persistence"] if d["persistence"] is not None else "n.a.") if d["present"] else "n.a."),
                     ("placement", (h or {}).get("placement", "pending"), (d["placement"] or "n.a.") if d["present"] else "n.a.")]
            for f, hv, dv in pairs:
                hv_s, dv_s = str(hv).strip().lower(), str(dv).strip().lower()
                same = hv_s == dv_s or (num(hv_s) is not None and num(dv_s) is not None and abs(num(hv_s) - num(dv_s)) < 0.13)
                if not same:
                    note = notes.get(f"{arm}/{sid}/{f}", "")
                    if f == "onset_frame" and num(hv_s) is not None and num(dv_s) is not None and abs(num(hv_s) - num(dv_s)) == 1:
                        note = ("one frame apart; " + note).rstrip("; ")
                    rows.append([sid, arm, f, hv, dv, facts if f == "presence" else "", note])
    return header, rows


def t4(side):
    header = ["id", "arm", "control_scale", "length (s)", "frames", "wall (s)", "wall (min)", "GPU peak (MiB, nvidia-smi)", "RAM peak (GB, RSS)", "dtype", "start (UTC)"]
    rows = []; frames = {}
    for arm in ARMS:
        for p in glob.glob(os.path.join(ROOT, "outputs", arm, "frames", "*", "frames.json")):
            j = json.load(open(p)); n = next((j[k] for k in ("frame_count_decoded", "n_frames", "frame_count") if isinstance(j.get(k), int)), "")
            frames[(arm, os.path.basename(os.path.dirname(p)))] = n
    for sid in IDS:
        for arm in ARMS:
            s = side.get((arm, sid))
            if s:
                rows.append([sid, arm, s["control_scale"], s["length"], frames.get((arm, sid), ""), s["wall_seconds"], round(s["wall_seconds"] / 60, 1),
                             s.get("peak_gpu_used_mb_nvidia_smi", ""), s.get("child_peak_rss_gb"),
                             s.get("versions", {}).get("transformer_dtype") or "fp16", s["start"]])
    return header, rows


def cost(side):
    walls = [s["wall_seconds"] for s in side.values()]
    per = statistics.median(walls) if walls else None
    per_v = statistics.median([s["wall_seconds"] for k, s in side.items() if k[0] == "vanilla"] or [0])
    per_a = statistics.median([s["wall_seconds"] for k, s in side.items() if k[0] == "alignvid"] or [0])
    h = per / 3600 if per else None
    rows = []
    def add(label, n, note=""):
        rows.append([label, n, round(n * h, 0) if h else "", round(n * h / 30, 1) if h else "", note])
    add("Pilot as run: 5 samples × 2 arms × 1 seed", 10, "measured: 9 h 49 min of Kaggle quota over two sessions")
    add("(a) 40 samples × 2 arms × 3 seeds", 240)
    add("(b) 129 addition samples × 2 arms × 1 seed", 258)
    add("(c) no-edit arm added to (a)", 120, "one no-edit video per sample and seed")
    add("(c') no-edit arm added to (b)", 129)
    header = ["Configuration", "videos", "GPU hours on the T4 (fp16, 6.0 s clips)", "weeks of the 30 h/week Kaggle quota", "note"]
    meta = {"median_wall_s": per, "median_vanilla_s": per_v, "median_alignvid_s": per_a, "n": len(walls)}
    return header, rows, meta


def build(no_pdf=False):
    info = protocol_info(); lab = labels(); det = detector(); side = sidecars(); env = env_info()
    notes_p = os.path.join(REP, "sections", "t2_notes.json")
    notes = json.load(open(notes_p)) if os.path.exists(notes_p) else {}
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    today = datetime.date.today().isoformat()
    # tables
    h1, r1 = t1(lab); write_csv("T1_labels.csv", h1, r1)
    h2, r2 = t2(lab, det, notes); write_csv("T2_human_vs_detector.csv", h2, r2)
    h4, r4 = t4(side); write_csv("T4_runtime_memory.csv", h4, r4)
    hc, rc, cmeta = cost(side); write_csv("cost_table.csv", hc, rc)
    agr = agreement(lab)
    # header / versions
    gpu = (env.get("gpu", {}).get("nvidia_smi") or [{}])[0]; v = env.get("versions", {})
    vers_rows = [["Hardware used", f"Kaggle free tier, {gpu.get('name', '?')} {int(gpu.get('vram_total_mb', 0)) // 1024} GB (one of two), {env.get('ram_gb', 31)} GB RAM, driver {gpu.get('driver', '?')}; two headless sessions"],
                 ["Generation env", f"Python {v.get('python')}, torch {v.get('torch')}, diffusers {v.get('diffusers')}, transformers {v.get('transformers')}, accelerate {v.get('accelerate')}, numpy {v.get('numpy')}"],
                 ["Transformer dtype", "fp16 with fp32 residual stream (protocol §9, 2026-11-22); all other models fp16 as upstream"],
                 ["Protocol", f"frozen {info['frozen_on']}, commit <code>{info['frozen_commit']}</code>; this report built from commit <code>{git}</code> on {today}"]]
    vers_html = md(info["versions_md"]).replace("<table>", "<table class='narrow'>") + html_table(["Item", "Value"], vers_rows, cls="narrow")
    labellers = ", ".join(sorted(lab)) or "none yet"
    agr_html = ""
    if agr:
        agr_html = "<h3>Agreement between labellers</h3>" + html_table(
            ["labellers", "n videos", "κ (presence)"] + list(agr["fields"]), [[" / ".join(agr["labellers"]), agr["n"], agr["kappa_presence"]] + list(agr["fields"].values())])
    panels = "".join(f"<div class='page'><img class='panel' src='panels/{sid}.png'></div>" for sid in IDS if os.path.exists(os.path.join(REP, "panels", f"{sid}.png")))
    dev_html = html_table(["Date", "Change", "Why"], [[d, md(c), md(w)] for d, c, w in info["deviations"]])
    body = f"""
<h1>Five-sample feasibility pilot: object addition, FramePack-F1, vanilla vs AlignVid (γ = 1.35)</h1>
<p class='small'>Protocol frozen {info['frozen_on']}, commit <code>{info['frozen_commit']}</code>. Report built {today} from commit <code>{git}</code>. Labellers: {labellers}.</p>
{section('01_summary.md')}
<h2>Versions and hardware</h2>{vers_html}
{panels}
<div class='page'><h2>T1. Human labels per video</h2>{html_table(h1, r1, wide_last=True)}{agr_html}
<h2>T2. Human vs detector disagreements</h2><p class='small'>Grounding DINO tiny, thresholds 0.45 / 0.35, presence = detected in ≥ 4 of 16 frames, fixed on dev images before the pilot (protocol §8). One row per disagreeing field; persistence counts as agreeing within 0.13 (two frames of 16).</p>{html_table(h2, r2)}
{section('02_t2_notes.md')}</div>
<div class='page'><h2>T4. Runtime and memory per video</h2><p class='small'>One FramePack process per video on one Tesla T4; wall time includes model loading from local disk. FramePack reported <code>Free VRAM: 14.46 GB, High-VRAM Mode: False</code>, i.e. its low-VRAM path with <code>--gpu_mem 6</code>. Median wall time: vanilla {round(cmeta['median_vanilla_s'])} s, AlignVid {round(cmeta['median_alignvid_s'])} s.</p>{html_table(h4, r4, cls="narrow")}
<h2>T5. Failures and unscorable cases</h2>{section('03_failures.md')}</div>
<div class='page'><h2>Where it broke down</h2>{section('04_breakdown.md')}</div>
<div class='page'><h2>Deviations from the frozen protocol</h2>{dev_html}
<p><b>Scope statement.</b> With n = 5 samples and one seed, no difference between arms can be estimated; the pilot validates the pipeline only.</p></div>
"""
    html1 = f"<!doctype html><html><head><meta charset='utf-8'><title>Pilot report</title><style>{CSS}</style></head><body>{body}</body></html>"
    open(os.path.join(REP, "pilot_report.html"), "w", encoding="utf-8").write(html1)
    # PDF 2
    rp = open(os.path.join(REP, "revised_protocol.md"), encoding="utf-8").read()
    cost_md = html_table(hc, rc) + f"<p class='small'>Per-video figure: median of the {cmeta['n']} pilot videos, {round(cmeta['median_wall_s'])} s ({round(cmeta['median_wall_s'] / 60, 1)} min) including model loading; 6.0 s setting = 181 frames at 24 fps. Session overhead on Kaggle (environment + 40 GB weight download) was about 6 min per session and is not included.</p>"
    html2 = f"<!doctype html><html><head><meta charset='utf-8'><title>Revised protocol</title><style>{CSS} body {{ font-size: 10pt; line-height: 1.3; }} @page {{ margin: 10mm 12mm; }}</style></head><body>{md(rp).replace('<p>{{COST_TABLE}}</p>', cost_md)}</body></html>"
    open(os.path.join(REP, "revised_protocol.html"), "w", encoding="utf-8").write(html2)
    json.dump({"built": datetime.datetime.now().isoformat(timespec="seconds"), "git": git, "labellers": sorted(lab), "n_labelled": {k: len(v) for k, v in lab.items()},
               "cost": cmeta, "agreement": agr}, open(os.path.join(REP, "report.sidecar.json"), "w"), indent=1)
    if not no_pdf:
        for name in ("pilot_report", "revised_protocol"):
            src = os.path.join(REP, f"{name}.html"); dst = os.path.join(REP, f"{name}.pdf")
            r = subprocess.run(["google-chrome", "--headless=new", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=15000",
                                f"--print-to-pdf={dst}", f"file://{src}"], capture_output=True, text=True)
            print(name, "->", dst, "ok" if os.path.exists(dst) else r.stderr[-400:])
    print(f"labellers {sorted(lab)}; T1 rows {len(r1)}; T2 disagreement rows {len(r2)}; T4 rows {len(r4)}; median wall {cmeta['median_wall_s']} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-pdf", action="store_true"); a = ap.parse_args()
    build(a.no_pdf)
