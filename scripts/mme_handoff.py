"""Keep the human/agent handoff current without touching evaluation state or keys."""
from datetime import datetime, timezone
import argparse
import fcntl
import json
from pathlib import Path
import time

import mme_realworld as mme

ROOT = Path(__file__).resolve().parents[1]
START = '<!-- LIVE_MME_STATUS_START -->'
END = '<!-- LIVE_MME_STATUS_END -->'


def update():
    path = ROOT / 'TanishReadme.md'
    if not path.exists():
        return False
    campaign_path = ROOT / 'runs/mme_campaign/status.json'
    campaign = json.loads(campaign_path.read_text()) if campaign_path.exists() else {}
    now = datetime.now(timezone.utc).isoformat()
    lines = [START, '## Latest live status (automatic)', '', f'Updated **{now}**. This section supersedes older snapshots below.', '',
             '| Batch | Completed | Skipped >20 min | Errors awaiting repair | Never attempted |',
             '|---|---:|---:|---:|---:|']
    counts = {}
    for batch in (1, 2):
        run = ROOT / f'runs/mme_gemma4_reasoning_paid_32k_batch{batch}'
        rows = {r['sample_id']: mme.normalize_record(r) for r in mme.read_jsonl(run / 'responses.jsonl')}
        count = {s: sum(r['status'] == s for r in rows.values()) for s in ('completed', 'skipped', 'error')}
        count['pending'] = 1000 - len(rows)
        counts[str(batch)] = count
        lines.append(f"| {batch} | {count['completed']} | {count['skipped']} | {count['error']} | {count['pending']} |")
    lines += ['', f"Campaign stage: `{campaign.get('stage', 'unknown')}`; supervisor PID `{campaign.get('supervisor_pid', 'unknown')}`; child PID `{campaign.get('child_pid', 'none')}`.",
              f"Latest child log: `{campaign.get('log', 'see runs/mme_campaign/')}`.", '',
              '**Current policy:** paid Gemma, reasoning ON, 32,768 output tokens; skip individual requests over 1,200 seconds and retain them in `skipped_examples.json`. Do not retry skips. Finish each 1,000-selection batch when completed + skipped = 1,000.', '',
              'Counts are derived from normalized raw responses; error records may be in flight. Campaign counts can be round-start snapshots. Live controls and locks remain authoritative.', '',
              'Automatic status history: `runs/mme_campaign/handoff_status_history.jsonl`. Authored policy/fix updates remain below.', END]
    text = path.read_text()
    block = '\n'.join(lines)
    if START in text and END in text:
        before, rest = text.split(START, 1)
        _, after = rest.split(END, 1)
        text = before + block + after
    else:
        title, rest = text.split('\n', 1)
        text = title + '\n\n' + block + '\n' + rest
    temp = path.with_suffix('.md.live.tmp')
    temp.write_text(text)
    temp.replace(path)
    history = ROOT / 'runs/mme_campaign/handoff_status_history.jsonl'
    mme.append(history, {'timestamp': now, 'counts': counts, 'campaign': campaign})
    return campaign.get('stage') == 'complete'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    lock = (ROOT / 'runs/mme_campaign/.handoff.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Handoff monitor already running')
    while True:
        complete = update()
        if not args.watch or complete:
            return
        time.sleep(60)


if __name__ == '__main__':
    main()
