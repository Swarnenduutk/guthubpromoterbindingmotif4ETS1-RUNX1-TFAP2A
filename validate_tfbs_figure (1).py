#!/usr/bin/env python3
"""
validate_tfbs_figure.py
=======================
Bioinformatic validation of a promoter / TFBS figure claiming JASPAR-predicted
GATA2 and HOXB7 binding sites in six angiogenesis genes.

What it checks, independently of the figure:
  1. MATRIX IDENTITY  - queries the JASPAR REST API to confirm that the matrix
                        IDs printed on the figure actually belong to the TFs
                        named on the figure (this is the single most common
                        error in figures like this).
  2. TSS DEFINITION   - resolves each gene to its MANE Select transcript and
                        derives the TSS from GRCh38 coordinates.
  3. PROMOTER SEQ     - retrieves -2000 .. +500 relative to that TSS, on the
                        gene's own strand, from Ensembl REST.
  4. MOTIF SCAN       - scans both strands with a log-odds PSSM and reports
                        every hit with its position relative to the TSS.
  5. CLAIM COMPARISON - compares observed hit counts with the counts read off
                        the figure, and specifically tests the one site the
                        figure calls experimentally validated
                        (ANGPT2 GATA2 site at ~ -64 / -47).

Outputs
  results/motif_identity.tsv     what the matrix IDs really are
  results/tss_coordinates.tsv    MANE transcript + TSS used per gene
  results/promoters.fa           the exact sequences scanned
  results/motif_hits.tsv         every hit (gene, TF, pos vs TSS, strand, score, p)
  results/summary.txt            PASS / FAIL per figure claim
  results/<GENE>_promoter_map.png  optional per-gene maps (--plot)

Usage
  python validate_tfbs_figure.py --outdir results
  python validate_tfbs_figure.py --outdir results --rel-score 0.80 --pvalue 1e-4 --plot

Requirements
  python >= 3.9 ; pip install biopython requests matplotlib
Network
  Needs outbound HTTPS to rest.ensembl.org and jaspar.elixir.no.
  All downloads are cached under --cachedir, so re-runs are offline.
"""

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

import requests
from Bio import motifs
from Bio.motifs import jaspar as jaspar_tools
from Bio.Seq import Seq

# --------------------------------------------------------------------------
# WHAT THE FIGURE CLAIMS.  Edit this block if I misread any oval count.
# --------------------------------------------------------------------------
FIGURE = {
    # matrix IDs as printed in the figure legend
    "matrix_ids": {"GATA2": "MA0036.2", "HOXB7": "MA0090.2"},
    # promoter window as printed
    "window": (-2000, 500),
    "genome": "GRCh38",
    # number of ovals I counted in each promoter track of the figure.
    # purple = GATA2 (incl. the validated one), green = HOXB7.
    "site_counts": {
        "ANGPT1": {"GATA2": 2, "HOXB7": 2},
        "ANGPT2": {"GATA2": 3, "HOXB7": 2},   # 2 predicted + 1 validated
        "THBS1":  {"GATA2": 2, "HOXB7": 1},
        "VEGFA":  {"GATA2": 2, "HOXB7": 2},
        "PDGFA":  {"GATA2": 2, "HOXB7": 1},
        "ANG":    {"GATA2": 1, "HOXB7": 2},
    },
    # the one site the figure calls experimentally validated
    "validated": {"gene": "ANGPT2", "tf": "GATA2", "start": -64, "end": -47},
}

GENES = ["ANGPT1", "ANGPT2", "THBS1", "VEGFA", "PDGFA", "ANG"]

ENSEMBL = "https://rest.ensembl.org"
JASPAR = "https://jaspar.elixir.no/api/v1"
MANE_DIR = "https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/current/"

BACKGROUND = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}


# --------------------------------------------------------------------------
# small cached HTTP helper
# --------------------------------------------------------------------------
def fetch(url, cachedir, params=None, binary=False, tries=8):
    """GET with on-disk cache so reruns need no network."""
    import hashlib
    key = url + ("?" + "&".join(f"{k}={v}" for k, v in sorted((params or {}).items())))
    # NOTE: must NOT use builtin hash() here - it is salted per process, so the
    # cache would never be reused across runs.
    digest = hashlib.sha256(key.encode()).hexdigest()[:32]
    fn = Path(cachedir) / (digest + (".bin" if binary else ".txt"))
    if fn.exists():
        return fn.read_bytes() if binary else fn.read_text()
    last = None
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, timeout=60,
                             headers={"Content-Type": "application/json",
                                      "User-Agent": "tfbs-figure-validator/1.0"})
            if r.status_code == 429:                       # Ensembl rate limit
                time.sleep(float(r.headers.get("Retry-After", 2)) + 1)
                continue
            if r.status_code >= 500:                       # transient server fault
                time.sleep(min(2 ** attempt, 30) + 1)
                last = requests.HTTPError(f"{r.status_code} from {url}")
                continue
            r.raise_for_status()
            data = r.content if binary else r.text
            fn.parent.mkdir(parents=True, exist_ok=True)
            fn.write_bytes(data) if binary else fn.write_text(data)
            time.sleep(0.34)                               # be polite to Ensembl
            return data
        except Exception as exc:                           # noqa: BLE001
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed after {tries} tries: {url}\n{last}")


# --------------------------------------------------------------------------
# STEP 1 - are the matrix IDs on the figure really GATA2 and HOXB7?
# --------------------------------------------------------------------------
def jaspar_profiles_for(tf, cachedir):
    """All vertebrate JASPAR profiles whose name matches this TF exactly."""
    res = json.loads(fetch(f"{JASPAR}/matrix/", cachedir,
                           params={"search": tf, "tax_group": "vertebrates",
                                   "format": "json", "page_size": "100"}))
    return [m for m in res.get("results", [])
            if m.get("name", "").upper() == tf.upper()]


def split_id(mid):
    base, _, ver = mid.partition(".")
    return base, (int(ver) if ver.isdigit() else 0)


def latest_profile(hits):
    """Pick the primary profile for a TF, at its current version.

    A TF can have several unrelated base accessions in JASPAR - TFAP2A has
    MA0003, MA0810 and MA0872, which are different models (full site vs
    half site variants), not successive versions of one another. The lowest
    base accession is the TF's original, primary profile; higher ones are
    variant models. So: lowest base accession, then its highest version.
    Use --matrix TF=ID to override when you want a variant deliberately.
    """
    best_base = min(split_id(m["matrix_id"])[0] for m in hits)
    same = [m for m in hits if split_id(m["matrix_id"])[0] == best_base]
    return max(same, key=lambda m: split_id(m["matrix_id"])[1])["matrix_id"]


def resolve_matrices(tf_names, cachedir, outdir, prefer_figure_ids=True,
                     overrides=None):
    """Pick a JASPAR matrix per TF, and audit any matrix ID the figure printed."""
    rows, notes, usable = [], [], {}
    overrides = overrides or {}

    for tf in tf_names:
        hits = jaspar_profiles_for(tf, cachedir)
        fig_id = FIGURE["matrix_ids"].get(tf) if prefer_figure_ids else None

        if hits:
            ids = ", ".join(
                sorted(f"{m['matrix_id']} ({m.get('collection','')})" for m in hits))
            notes.append(f"[INFO] JASPAR profiles named {tf}: {ids}")
            newest = overrides.get(tf) or latest_profile(hits)
            if tf in overrides:
                notes.append(f"[INFO] {tf}: using {overrides[tf]} (--matrix override)")
            else:
                others = sorted({split_id(m["matrix_id"])[0] for m in hits}
                                - {split_id(newest)[0]})
                if others:
                    notes.append(
                        f"[INFO] {tf}: chose {newest} (primary profile). Variant "
                        f"models also exist under {', '.join(others)}; use "
                        f"--matrix {tf}=<ID> to scan one of those instead.")
        else:
            newest = None
            notes.append(f"[FAIL] JASPAR has NO vertebrate profile named '{tf}'.")

        # If the figure printed an ID for this TF, check that it is really this TF.
        if fig_id:
            try:
                rec = json.loads(fetch(f"{JASPAR}/matrix/{fig_id}/", cachedir,
                                       params={"format": "json"}))
                actual = rec.get("name", "?")
                ok = actual.upper() == tf.upper()
                rows.append({
                    "TF": tf, "matrix_id_checked": fig_id, "JASPAR_name": actual,
                    "collection": rec.get("collection", ""),
                    "class": ";".join(rec.get("class", []) or []),
                    "MATCH": "YES" if ok else "NO -- MISLABELLED",
                    "latest_profile_for_TF": newest or "(none)",
                    "matrix_used_for_scan": fig_id if ok else (newest or "(skipped)"),
                })
                if not ok:
                    notes.append(
                        f"[FAIL] Figure labels {fig_id} as {tf}, but JASPAR says "
                        f"{fig_id} is {actual}. Every '{tf}' site drawn from this "
                        f"matrix is actually a {actual} site.")
                if ok and newest and newest != fig_id:
                    notes.append(
                        f"[WARN] {fig_id} is not the current {tf} profile; "
                        f"JASPAR's latest is {newest}. Hit counts depend on version.")
                if ok:
                    usable[tf] = fig_id
                elif newest:
                    usable[tf] = newest
                    notes.append(f"[fix] scanning {tf} with {newest} instead of {fig_id}")
            except Exception as exc:                        # noqa: BLE001
                notes.append(f"[FAIL] {fig_id} could not be retrieved: {exc}")
                if newest:
                    usable[tf] = newest
        else:
            # No figure claim for this TF: just use the current profile.
            if newest:
                usable[tf] = newest
                rows.append({
                    "TF": tf, "matrix_id_checked": "(none claimed)",
                    "JASPAR_name": tf, "collection": next(
                        (m.get("collection", "") for m in hits
                         if m["matrix_id"] == newest), ""),
                    "class": ";".join(next(
                        (m.get("class", []) for m in hits
                         if m["matrix_id"] == newest), []) or []),
                    "MATCH": "n/a", "latest_profile_for_TF": newest,
                    "matrix_used_for_scan": newest,
                })

    write_tsv(Path(outdir) / "motif_identity.tsv", rows)
    return usable, rows, notes


def _legacy_verify(cachedir, outdir):
    rows, notes = [], []

    for claimed_tf, mid in FIGURE["matrix_ids"].items():
        try:
            rec = json.loads(fetch(f"{JASPAR}/matrix/{mid}/", cachedir,
                                   params={"format": "json"}))
            actual = rec.get("name", "?")
            ok = actual.upper() == claimed_tf.upper()
            rows.append({
                "figure_says_TF": claimed_tf, "matrix_id": mid,
                "JASPAR_name": actual, "collection": rec.get("collection", ""),
                "version_latest": rec.get("version", ""),
                "class": ";".join(rec.get("class", []) or []),
                "species": ";".join(s.get("name", "") for s in rec.get("species", []) or []),
                "MATCH": "YES" if ok else "NO -- MISLABELLED",
            })
            if not ok:
                notes.append(
                    f"[FAIL] Figure labels {mid} as {claimed_tf}, but JASPAR "
                    f"says {mid} is {actual}. Every '{claimed_tf}' site drawn "
                    f"from this matrix is actually a {actual} site.")
        except Exception as exc:                            # noqa: BLE001
            rows.append({"figure_says_TF": claimed_tf, "matrix_id": mid,
                         "JASPAR_name": f"LOOKUP FAILED ({exc})", "MATCH": "UNKNOWN"})
            notes.append(f"[FAIL] {mid} could not be retrieved from JASPAR: {exc}")

    # what matrices does JASPAR actually hold for these TFs?
    available = {}
    for tf in FIGURE["matrix_ids"]:
        try:
            res = json.loads(fetch(f"{JASPAR}/matrix/", cachedir,
                                   params={"search": tf, "tax_group": "vertebrates",
                                           "format": "json", "page_size": "100"}))
            hits = [m for m in res.get("results", [])
                    if m.get("name", "").upper() == tf.upper()]
            available[tf] = hits
            if not hits:
                notes.append(
                    f"[FAIL] JASPAR has NO vertebrate CORE/UN profile named "
                    f"'{tf}'. Sites for it cannot have come from JASPAR.")
            else:
                ids = ", ".join(f"{m['matrix_id']} ({m.get('collection','')})" for m in hits)
                notes.append(f"[INFO] JASPAR profiles named {tf}: {ids}")
        except Exception as exc:                            # noqa: BLE001
            notes.append(f"[WARN] could not list JASPAR profiles for {tf}: {exc}")

    write_tsv(Path(outdir) / "motif_identity.tsv", rows)
    return rows, notes, available


def load_pfm(matrix_id, cachedir):
    """Return a Bio.motifs Motif with pseudocounts and background set."""
    rec = json.loads(fetch(f"{JASPAR}/matrix/{matrix_id}/", cachedir,
                           params={"format": "json"}))
    pfm = rec["pfm"]                       # {'A':[...], 'C':[...], ...}
    counts = {b: [float(x) for x in pfm[b]] for b in "ACGT"}
    m = motifs.Motif(alphabet="ACGT", counts=counts)
    m.pseudocounts = jaspar_tools.calculate_pseudocounts(m)
    m.background = BACKGROUND
    m.name = rec.get("name", matrix_id)
    m.matrix_id = matrix_id
    return m


# --------------------------------------------------------------------------
# STEP 2 - MANE Select transcript and TSS
# --------------------------------------------------------------------------
def find_mane_url(cachedir):
    """Discover the current MANE summary filename; the version changes."""
    import re
    listing = fetch(MANE_DIR, cachedir)
    names = sorted(set(re.findall(
        r"MANE\.GRCh38\.v[\d.]+\.summary\.txt\.gz", listing)))
    if not names:
        raise RuntimeError(f"no MANE summary file found at {MANE_DIR}")
    # highest version last
    names.sort(key=lambda n: [int(x) for x in
                              re.search(r"v([\d.]+)\.summary", n).group(1).split(".")])
    return MANE_DIR + names[-1]


def load_mane(cachedir):
    """gene symbol -> Ensembl transcript ID (MANE Select). Empty dict on failure."""
    try:
        url = find_mane_url(cachedir)
        print(f"  using MANE summary: {url.rsplit('/', 1)[-1]}")
        raw = fetch(url, cachedir, binary=True)
        txt = gzip.decompress(raw).decode()
    except Exception as exc:                                # noqa: BLE001
        print(f"  [warn] MANE summary unavailable ({exc}); "
              f"falling back to Ensembl canonical transcripts", file=sys.stderr)
        return {}
    mane, header = {}, None
    for line in txt.splitlines():
        f = line.rstrip("\n").split("\t")
        if header is None:
            header = f
            continue
        row = dict(zip(header, f))
        if row.get("MANE_status") != "MANE Select":
            continue
        sym = row.get("symbol", "")
        enst = row.get("Ensembl_nuc", "").split(".")[0]
        if sym and enst:
            mane[sym] = enst
    return mane


def get_tss(gene, cachedir, mane):
    """Return dict with chrom, strand, tss, transcript, source."""
    info = json.loads(fetch(f"{ENSEMBL}/lookup/symbol/homo_sapiens/{gene}", cachedir,
                            params={"expand": "1", "content-type": "application/json"}))
    strand = int(info["strand"])
    chrom = info["seq_region_name"]
    want = mane.get(gene)

    chosen, source = None, None
    for tr in info.get("Transcript", []):
        if want and tr["id"] == want:
            chosen, source = tr, "MANE Select"
            break
    if chosen is None:
        for tr in info.get("Transcript", []):
            if tr.get("is_canonical"):
                chosen, source = tr, "Ensembl canonical (MANE unavailable)"
                break
    if chosen is None:
        raise RuntimeError(f"no transcript selected for {gene}")

    tss = int(chosen["start"]) if strand == 1 else int(chosen["end"])
    return {"gene": gene, "chrom": chrom, "strand": strand, "tss": tss,
            "transcript": chosen["id"], "biotype": chosen.get("biotype", ""),
            "transcript_source": source, "gene_id": info["id"]}


# --------------------------------------------------------------------------
# STEP 3 - promoter sequence, oriented 5'->3' on the gene strand
# --------------------------------------------------------------------------
def get_promoter(rec, cachedir, up, down):
    strand, tss = rec["strand"], rec["tss"]
    if strand == 1:
        start, end = tss + up, tss + down          # up is negative
    else:
        start, end = tss - down, tss - up
    region = f"{rec['chrom']}:{start}..{end}:{strand}"
    seq = json.loads(fetch(f"{ENSEMBL}/sequence/region/human/{region}", cachedir,
                           params={"content-type": "application/json"}))["seq"].upper()
    expected = down - up + 1
    if len(seq) != expected:
        raise RuntimeError(f"{rec['gene']}: got {len(seq)} bp, expected {expected}")
    # index 0 of seq == position `up` relative to TSS
    return seq, start, end


# --------------------------------------------------------------------------
# STEP 4 - scan
# --------------------------------------------------------------------------
def scan(seq, motif, up, rel_cut, pval):
    """Return hits as dicts with position relative to TSS."""
    pssm = motif.pssm
    dist = pssm.distribution(background=motif.background, precision=10 ** 4)
    thr_p = dist.threshold_fpr(pval)
    thr_rel = pssm.min + rel_cut * (pssm.max - pssm.min)
    threshold = min(thr_p, thr_rel)            # report the looser of the two, then flag
    L = motif.length
    s = Seq(seq)
    out = []
    for pos, score in pssm.search(s, threshold=threshold, both=True):
        idx = pos if pos >= 0 else len(s) + pos
        strand = "+" if pos >= 0 else "-"
        site = seq[idx:idx + L]
        if "N" in site:
            continue
        rel = (pssm.max - pssm.min)
        out.append({
            "start_vs_TSS": idx + up,
            "end_vs_TSS": idx + up + L - 1,
            "strand": strand,
            "site": site if strand == "+" else str(Seq(site).reverse_complement()),
            "genomic_context": site,
            "score": round(float(score), 3),
            "rel_score": round((float(score) - pssm.min) / rel, 4),
            "passes_rel_cut": "yes" if float(score) >= thr_rel else "no",
            "passes_pvalue": "yes" if float(score) >= thr_p else "no",
        })
    return out, thr_rel, thr_p


# --------------------------------------------------------------------------
# STEP 4b - how many hits would you get by chance?
# --------------------------------------------------------------------------
def dinuc_shuffle(seq, rng):
    """Altschul-Erikson style dinucleotide-preserving shuffle (Eulerian walk)."""
    if len(seq) < 3:
        return seq
    edges = {}
    for a, b in zip(seq, seq[1:]):
        edges.setdefault(a, []).append(b)
    last = seq[-1]
    # build a random last-edge tree rooted at `last` so the walk is Eulerian
    for _ in range(100):
        order = {k: v[:] for k, v in edges.items()}
        for k in order:
            rng.shuffle(order[k])
        last_edge = {k: order[k][-1] for k in order if k != last}
        ok, seen = True, set()
        for k in last_edge:
            node, path = k, set()
            while node != last:
                if node in path:
                    ok = False
                    break
                path.add(node)
                node = last_edge.get(node)
                if node is None:
                    ok = False
                    break
            if not ok:
                break
            seen |= path
        if ok:
            break
    out, node = [seq[0]], seq[0]
    pool = {k: v[:] for k, v in order.items()}
    while pool.get(node):
        nxt = pool[node].pop(0)
        out.append(nxt)
        node = nxt
    res = "".join(out)
    return res if len(res) == len(seq) else seq


def background_expectation(seq, motif, rel_cut, pval, n_shuffle, seed):
    """Mean +/- sd hits in dinucleotide-shuffled copies of the same promoter."""
    import random as _random
    import statistics as _stats
    rng = _random.Random(seed)
    pssm = motif.pssm
    dist = pssm.distribution(background=motif.background, precision=10 ** 4)
    thr_p = dist.threshold_fpr(pval)
    thr_rel = pssm.min + rel_cut * (pssm.max - pssm.min)
    counts_rel, counts_p = [], []
    for _ in range(n_shuffle):
        sh = Seq(dinuc_shuffle(seq, rng))
        hits = list(pssm.search(sh, threshold=min(thr_rel, thr_p), both=True))
        counts_rel.append(sum(1 for _, s in hits if float(s) >= thr_rel))
        counts_p.append(sum(1 for _, s in hits if float(s) >= thr_p))
    f = lambda v: (round(_stats.mean(v), 2),                       # noqa: E731
                   round(_stats.pstdev(v), 2) if len(v) > 1 else 0.0)
    return {"rel": f(counts_rel), "pval": f(counts_p)}


# --------------------------------------------------------------------------
# STEP 5 - the one "validated" site
# --------------------------------------------------------------------------
def check_validated(seq, up, hits, claim):
    lo, hi = claim["start"], claim["end"]
    pad = 15
    a, b = lo - pad - up, hi + pad - up + 1
    a, b = max(a, 0), min(b, len(seq))
    window = seq[a:b]
    overlapping = [h for h in hits
                   if not (h["end_vs_TSS"] < lo or h["start_vs_TSS"] > hi)]
    # literal WGATAR / canonical GATA core, both strands
    import re
    core = []
    for m in re.finditer(r"(?=([AT]GATA[AG]))", window):
        core.append(("+", m.start() + a + up, m.group(1)))
    rc = str(Seq(window).reverse_complement())
    for m in re.finditer(r"(?=([AT]GATA[AG]))", rc):
        gstart = a + (len(window) - m.start() - 6) + up
        core.append(("-", gstart, m.group(1)))
    return {"window_seq": window, "window_from": a + up, "window_to": b + up - 1,
            "pssm_hits_overlapping": overlapping, "WGATAR_cores": core}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def write_tsv(path, rows, cols=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("(no rows)\n")
        return
    cols = cols or list(rows[0].keys())
    with path.open("w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")


PALETTE = ["#7b3fa0", "#2e8b57", "#c1553b", "#2f6fb0", "#b8912a",
           "#8c4a6b", "#4a7d7d", "#6b6b2f"]


def plot_gene(gene, hits, up, down, outdir, tfs, matrix_ids=None):
    """One lane per TF actually scanned - never a hardcoded TF list."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tfs = list(tfs)
    n = max(len(tfs), 1)
    # lanes spread symmetrically above and below the promoter line
    span = 0.22 * n
    ys = [span - i * (2 * span / (n - 1)) for i in range(n)] if n > 1 else [0.25]

    fig, ax = plt.subplots(figsize=(10, 1.1 + 0.55 * n))
    ax.hlines(0, up, down, color="black", lw=1.5, zorder=1)
    ax.vlines(0, -span - 0.18, span + 0.18, color="red", lw=2, zorder=2)
    ax.text(0, span + 0.24, "TSS (+1)", color="red", ha="center", fontsize=8)

    for i, (tf, y) in enumerate(zip(tfs, ys)):
        xs = [h["start_vs_TSS"] for h in hits if h["tf"] == tf]
        ax.scatter(xs, [y] * len(xs), s=60,
                   facecolors=PALETTE[i % len(PALETTE)], edgecolors="k",
                   linewidths=0.4, alpha=0.85, zorder=3,
                   label=f"{tf} (n={len(xs)})")
        ax.text(up - (down - up) * 0.015, y, tf, ha="right", va="center",
                fontsize=8, color=PALETTE[i % len(PALETTE)])

    ax.set_xlim(up, down)
    ax.set_ylim(-span - 0.35, span + 0.42)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("position relative to TSS (bp)")
    sub = ("  [" + ", ".join(f"{t}:{matrix_ids[t]}" for t in tfs
                             if matrix_ids and t in matrix_ids) + "]"
           if matrix_ids else "")
    ax.set_title(f"{gene} - observed PSSM hits{sub}", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), fontsize=7,
              frameon=False, ncol=min(n, 4))
    fig.tight_layout()
    fig.savefig(Path(outdir) / f"{gene}_promoter_map.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--cachedir", default=".cache")
    ap.add_argument("--upstream", type=int, default=FIGURE["window"][0])
    ap.add_argument("--downstream", type=int, default=FIGURE["window"][1])
    ap.add_argument("--rel-score", type=float, default=0.80,
                    help="relative-score cutoff, JASPAR web default is 0.80")
    ap.add_argument("--pvalue", type=float, default=1e-4,
                    help="PSSM p-value (FPR) cutoff")
    ap.add_argument("--genes", nargs="*", default=GENES,
                    help="promoters to scan (the putative target genes)")
    ap.add_argument("--tfs", nargs="*", default=list(FIGURE["matrix_ids"]),
                    help="transcription factors to scan FOR. Each name is "
                         "resolved to its current JASPAR vertebrate profile; "
                         "if the figure printed an ID for that TF, the ID is "
                         "audited as well. e.g. --tfs ETS1 RUNX1 TFAP2A")
    ap.add_argument("--matrix", nargs="*", default=[], metavar="TF=MATRIX_ID",
                    help="force a specific JASPAR matrix for a TF, e.g. "
                         "--matrix TFAP2A=MA0810.2 ETS1=MA0098.3")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--n-shuffle", type=int, default=50,
                    help="dinucleotide-shuffled replicates for the background "
                         "expectation (0 to skip)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    Path(args.cachedir).mkdir(parents=True, exist_ok=True)
    log = []

    def say(msg):
        print(msg)
        log.append(msg)

    say("=" * 78)
    say("STEP 1  JASPAR matrix resolution and identity check")
    say("=" * 78)
    say(f"  scanning for: {', '.join(args.tfs)}")
    overrides = {}
    for spec in args.matrix:
        if "=" not in spec:
            sys.exit(f"--matrix expects TF=MATRIX_ID, got {spec!r}")
        k, _, val = spec.partition("=")
        overrides[k.strip()] = val.strip()
    usable, ident, notes = resolve_matrices(args.tfs, args.cachedir, out,
                                            overrides=overrides)
    for r in ident:
        say(f"  {r['TF']:>8}  claimed={r['matrix_id_checked']:<14} "
            f"JASPAR name={r['JASPAR_name']:<12} {r['MATCH']:<18} "
            f"using={r['matrix_used_for_scan']}")
    for n in notes:
        say("  " + n)
    for tf in args.tfs:
        if tf not in usable:
            say(f"  [skip] no usable JASPAR matrix for {tf}; omitted from scan")
    if not usable:
        sys.exit("No usable motifs; nothing to scan.")

    say("")
    say("=" * 78)
    say("STEP 2  MANE Select transcripts and TSS (GRCh38)")
    say("=" * 78)
    mane = load_mane(args.cachedir)
    tss_rows = []
    for g in args.genes:
        rec = get_tss(g, args.cachedir, mane)
        tss_rows.append(rec)
        say(f"  {g:<7} chr{rec['chrom']:<2} strand {rec['strand']:+d}  "
            f"TSS {rec['tss']:>12,}  {rec['transcript']}  [{rec['transcript_source']}]")
    write_tsv(out / "tss_coordinates.tsv", tss_rows)

    say("")
    say("=" * 78)
    say(f"STEP 3-4  Promoter retrieval and motif scan "
        f"({args.upstream:+d} .. {args.downstream:+d} bp)")
    say("=" * 78)
    pfms = {tf: load_pfm(mid, args.cachedir) for tf, mid in usable.items()}
    for tf, m in pfms.items():
        say(f"  {tf}: matrix {m.matrix_id}, length {m.length} bp, "
            f"PSSM range {m.pssm.min:.2f} .. {m.pssm.max:.2f}")

    all_hits, fasta, seqs, bg_rows = [], [], {}, []
    for rec in tss_rows:
        g = rec["gene"]
        seq, gstart, gend = get_promoter(rec, args.cachedir,
                                         args.upstream, args.downstream)
        seqs[g] = seq
        fasta.append(f">{g}|{rec['transcript']}|chr{rec['chrom']}:{gstart}-{gend}"
                     f"|strand{rec['strand']:+d}|TSS={rec['tss']}\n")
        fasta += [seq[i:i + 60] + "\n" for i in range(0, len(seq), 60)]
        gene_hits = []
        for tf, m in pfms.items():
            hits, thr_rel, thr_p = scan(seq, m, args.upstream,
                                        args.rel_score, args.pvalue)
            for h in hits:
                h.update({"gene": g, "tf": tf, "matrix_id": m.matrix_id,
                          "transcript": rec["transcript"]})
            gene_hits += hits
        all_hits += gene_hits
        if args.plot:
            plot_gene(g, gene_hits, args.upstream, args.downstream, out,
                      tfs=list(pfms), matrix_ids=usable)
        counts = {tf: sum(1 for h in gene_hits
                          if h["tf"] == tf and h["passes_rel_cut"] == "yes")
                  for tf in pfms}
        strict = {tf: sum(1 for h in gene_hits
                          if h["tf"] == tf and h["passes_pvalue"] == "yes")
                  for tf in pfms}
        say(f"  {g:<7} rel>={args.rel_score}: " +
            "  ".join(f"{tf}={counts[tf]}" for tf in pfms) +
            f"   |  p<={args.pvalue:g}: " +
            "  ".join(f"{tf}={strict[tf]}" for tf in pfms))

    if args.n_shuffle:
        say("")
        say("=" * 78)
        say(f"STEP 4b  Hits expected by chance ({args.n_shuffle} dinucleotide-"
            f"shuffled replicates per promoter)")
        say("=" * 78)
        say(f"  {'gene':<8}{'TF':<8}{'observed':>9}{'expected(sd)':>16}"
            f"{'ratio':>8}   [relative-score cutoff]")
        for g in args.genes:
            for tf, m in pfms.items():
                bg = background_expectation(seqs[g], m, args.rel_score,
                                            args.pvalue, args.n_shuffle, args.seed)
                obs = sum(1 for h in all_hits if h["gene"] == g and h["tf"] == tf
                          and h["passes_rel_cut"] == "yes")
                mu, sd = bg["rel"]
                ratio = f"{obs / mu:.2f}" if mu else "inf"
                say(f"  {g:<8}{tf:<8}{obs:>9}{mu:>11.2f} ({sd:.2f}){ratio:>8}")
                bg_rows.append({"gene": g, "tf": tf, "observed_rel": obs,
                                "expected_rel_mean": mu, "expected_rel_sd": sd,
                                "ratio": ratio,
                                "observed_pval": sum(
                                    1 for h in all_hits if h["gene"] == g
                                    and h["tf"] == tf and h["passes_pvalue"] == "yes"),
                                "expected_pval_mean": bg["pval"][0],
                                "expected_pval_sd": bg["pval"][1]})
        say("  A ratio near 1 means the promoter carries no more sites than a")
        say("  shuffled sequence of identical base and dinucleotide composition,")
        say("  i.e. the motif hits are not evidence of regulation by that TF.")
        write_tsv(out / "background_expectation.tsv", bg_rows)

    (out / "promoters.fa").write_text("".join(fasta))
    write_tsv(out / "motif_hits.tsv", all_hits,
              cols=["gene", "tf", "matrix_id", "transcript", "start_vs_TSS",
                    "end_vs_TSS", "strand", "site", "score", "rel_score",
                    "passes_rel_cut", "passes_pvalue"])

    say("")
    say("=" * 78)
    say("STEP 5  Comparison with the counts drawn on the figure")
    say("=" * 78)
    any_claim = any(tf in FIGURE["site_counts"].get(g, {})
                    for g in args.genes for tf in pfms)
    if not any_claim:
        say("  The figure makes no site-count claim for " +
            ", ".join(pfms) + ", so there is nothing to compare here.")
        say("  STEP 4b is the test that matters for these TFs.")
    else:
        say(f"  {'gene':<8}{'TF':<8}{'figure':>8}{'observed':>10}{'verdict':>12}")
        for g in args.genes:
            for tf in pfms:
                claimed = FIGURE["site_counts"].get(g, {}).get(tf)
                obs = sum(1 for h in all_hits if h["gene"] == g and h["tf"] == tf
                          and h["passes_rel_cut"] == "yes")
                verdict = ("n/a" if claimed is None
                           else ("MATCH" if claimed == obs else "DIFFERS"))
                say(f"  {g:<8}{tf:<8}{str(claimed):>8}{obs:>10}{verdict:>12}")
        say("  NOTE: oval counts were read off the image by eye. Correct the")
        say("        FIGURE['site_counts'] block if any count is wrong.")

    say("")
    say("=" * 78)
    say("STEP 6  The site the figure calls experimentally validated")
    say("=" * 78)
    v = FIGURE["validated"]
    if v["tf"] not in pfms:
        say(f"  Skipped: the validated claim concerns {v['tf']}, which is not "
            f"in this scan ({', '.join(pfms)}).")
    elif v["gene"] in seqs:
        gh = [h for h in all_hits if h["gene"] == v["gene"] and h["tf"] == v["tf"]]
        res = check_validated(seqs[v["gene"]], args.upstream, gh, v)
        say(f"  claim: {v['tf']} site in {v['gene']} at {v['start']} .. {v['end']}")
        say(f"  sequence {res['window_from']} .. {res['window_to']} (padded):")
        say(f"    5'-{res['window_seq']}-3'")
        if res["WGATAR_cores"]:
            for strand, pos, s in res["WGATAR_cores"]:
                inside = v["start"] <= pos <= v["end"]
                say(f"    WGATAR core {s} at {pos:+d} ({strand}) "
                    f"{'INSIDE' if inside else 'outside'} the claimed interval")
        else:
            say("    [FAIL] no WGATAR core anywhere in the padded window")
        if res["pssm_hits_overlapping"]:
            for h in res["pssm_hits_overlapping"]:
                say(f"    PSSM hit {h['start_vs_TSS']:+d}..{h['end_vs_TSS']:+d} "
                    f"({h['strand']}) {h['site']} rel={h['rel_score']}")
        else:
            say("    [FAIL] no PSSM hit overlaps the claimed interval")

    say("")
    say("Files written to " + str(out.resolve()))
    (out / "summary.txt").write_text("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
