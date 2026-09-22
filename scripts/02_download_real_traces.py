"""
Stage 1b — download and convert real ML-DPC LoadTraces.

Requires the ChampSim repo for the download links:
    git clone https://github.com/Quangmire/ChampSim.git
Usage:
    python scripts/02_download_real_traces.py --links ChampSim/download_links
    python scripts/02_download_real_traces.py --links ChampSim/download_links --pick 410.bwaves-s0 bfs-3
"""
import _path  # noqa: F401
import argparse, lzma, os, subprocess
from paging import config
from paging.traces import parse_download_links, convert

ap = argparse.ArgumentParser()
ap.add_argument('--links', default='ChampSim/download_links')
ap.add_argument('--pick', nargs='*', default=config.PICK)
ap.add_argument('--keep-raw', action='store_true', help='keep .txt.xz after converting')
args = ap.parse_args()

config.ensure_dirs()
if not os.path.exists(args.links):
    raise SystemExit(f'{args.links} not found. Run: git clone https://github.com/Quangmire/ChampSim.git')

urls = parse_download_links(args.links)
print(f'{len(urls)} LoadTraces listed')

for name in args.pick:
    out_csv = os.path.join(config.TRACE_DIR, f'{name}.csv')
    if os.path.exists(out_csv):
        print(f'{name:18s} already converted, skipping'); continue
    if name not in urls:
        print(f'{name:18s} not in download_links'); continue

    raw = os.path.join(config.RAW_DIR, f'{name}.txt.xz')
    if not os.path.exists(raw):
        print(f'downloading {name} ...')
        subprocess.run(['wget', '-q', '-O', raw, urls[name]], check=True)

    try:
        with lzma.open(raw, 'rt') as f:
            f.readline()
    except Exception as e:
        print(f'{name}: BAD DOWNLOAD ({e}) — Box may have returned an HTML page'); continue

    n, skipped = convert(raw, out_csv)
    mb = os.path.getsize(raw) / 1e6
    print(f'{name:18s} {mb:6.0f} MB -> {n} accesses, {skipped} skipped')
    if not args.keep_raw:
        os.remove(raw)

print(f'\nOK — traces in {config.TRACE_DIR}')
