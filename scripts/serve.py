"""Local viewer server with one-at-a-time recomputation of a run with new detector parameters.

    python scripts/serve.py --port 8765        # then open http://127.0.0.1:8765/viewer/
"""
import argparse
import json
import sqlite3
import subprocess
import sys
import threading
import uuid
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit, parse_qs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from replay_bag import make_config  # noqa: E402
from source_clouds import SourceClouds  # noqa: E402
try:
    import detector_lab  # noqa: E402  (optional local experiment module)
except ImportError:
    detector_lab = None

JOBS = {}
LOCK = threading.Lock()
CLOUDS = SourceClouds()


def load_manifest(sequence):
    if not sequence or not sequence.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Unknown run')
    path = ROOT / 'viewer' / 'data' / sequence / 'manifest.json'
    return cached_manifest(path, path.stat().st_mtime_ns)


@lru_cache(maxsize=12)
def cached_manifest(path, modified):
    manifest = json.loads(path.read_text(encoding='utf-8'))
    manifest['summary']['source_bag'] = str((ROOT / manifest['summary']['source_bag']).resolve())
    return manifest


def run_job(job, summary, config):
    """Replay the run's bag in a fresh process, so edits to the detector are always picked up."""
    try:
        command = [sys.executable, '-u', str(ROOT / 'scripts' / 'replay_bag.py'), str(ROOT / summary['source_bag']),
                   '--id', summary['id'], '--config', json.dumps(config), '--start-s', str(summary.get('source_start_s', 0.))]
        if summary.get('source_duration_s') is not None:
            command += ['--duration-s', str(summary['source_duration_s'])]
        if summary.get('source_max_frames') is not None:
            command += ['--max-frames', str(summary['source_max_frames'])]
        if summary.get('source_every', 1) > 1:
            command += ['--every', str(summary['source_every'])]
        if summary.get('full_cloud_exported') is False:
            command += ['--preview-only']
        result, tail = None, []
        with subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True) as worker:
            for line in worker.stdout:
                tail = (tail + [line.rstrip()])[-12:]
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if isinstance(event, dict) and event.get('event') == 'progress':
                    with LOCK:
                        job.update(done=event['done'])
                elif isinstance(event, dict) and event.get('event') == 'complete':
                    result = event['summary']
            code = worker.wait()
        if code or result is None:
            raise RuntimeError('Recompute failed: ' + '\n'.join(tail)[-4000:])
        with LOCK:
            job.update(status='done', summary=result)
    except Exception as e:  # reported to the viewer
        with LOCK:
            job.update(status='error', error=str(e))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def json_response(self, value, status=200):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        if path == '/api/experiments':
            try:
                sequence=parse_qs(urlsplit(self.path).query).get('sequence',[''])[0]
                load_manifest(sequence)
                self.json_response({'runs':detector_lab.list_runs(sequence) if detector_lab else []})
            except (ValueError,OSError) as e:self.json_response({'error':str(e)},400)
            return
        if path.startswith('/api/cloud/'):
            try:
                _, _, _, sequence, frame = path.split('/')
                manifest = load_manifest(sequence)
                body = CLOUDS.read(manifest['summary'], int(frame))
            except (ValueError, KeyError, OSError, sqlite3.Error) as e:
                self.json_response({'error': str(e)}, 400)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        if path.startswith('/viewer/data/') and path.endswith('/manifest.json'):
            try:
                sequence = path.split('/')[3]
                manifest = load_manifest(sequence)
                if manifest['summary'].get('full_cloud_exported') is False:
                    manifest = dict(manifest, frames=[dict(frame, full_file=f'/api/cloud/{sequence}/{i}')
                                                     for i, frame in enumerate(manifest['frames'])])
                self.json_response(manifest)
            except (ValueError, OSError) as e:
                self.json_response({'error': str(e)}, 404)
            return
        if path == '/api/capabilities':
            self.json_response({'recompute': True})
            return
        if path.startswith('/api/jobs/'):
            with LOCK:
                job = dict(JOBS.get(path.rsplit('/', 1)[-1], {}))
            self.json_response(job or {'error': 'Unknown job'}, 200 if job else 404)
            return
        if path == '/':
            self.send_response(302)
            self.send_header('Location', '/viewer/')
            self.end_headers()
            return
        resolved = (ROOT / path.lstrip('/')).resolve()
        if not any(resolved.is_relative_to(ROOT / p) for p in ('viewer', 'docs')):
            self.send_error(403)
            return
        super().do_GET()

    def do_POST(self):
        if self.path == '/api/experiments' and detector_lab is None:
            self.json_response({'error': 'The variant lab is not part of this build'}, 404)
            return
        if self.path not in ('/api/recompute','/api/experiments'):
            self.json_response({'error': 'Unknown API'}, 404)
            return
        # only pages of this server may start a recompute
        origin = self.headers.get('Origin', '')
        port = self.server.server_port
        if origin not in (f'http://127.0.0.1:{port}', f'http://localhost:{port}'):
            self.json_response({'error': 'Same-origin request required'}, 403)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 16384 or self.headers.get_content_type() != 'application/json':
                raise ValueError('JSON body required')
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload,dict):raise ValueError('JSON object required')
            config = payload.get('config') or {}
            if self.path == '/api/experiments':
                version=payload.get('version','cpp-current')
                config=detector_lab.validate(version,payload.get('parameters') or {})
            else:make_config(config)
            sequence = str(payload.get('sequence', ''))
            manifest = ROOT / 'viewer' / 'data' / sequence / 'manifest.json'
            if not sequence.replace('_', '').replace('-', '').isalnum() or not manifest.is_file():
                raise ValueError('Unknown run')
            summary = json.loads(manifest.read_text(encoding='utf-8'))['summary']
            if not (ROOT / summary.get('source_bag', '') / 'metadata.yaml').is_file():
                raise ValueError('The bag of this run is not available locally')
        except (ValueError, KeyError, TypeError) as e:
            self.json_response({'error': str(e)}, 400)
            return
        with LOCK:
            if any(j['status'] == 'running' for j in JOBS.values()):
                self.json_response({'error': 'Другой пересчёт ещё выполняется'}, 409)
                return
            job = {'id': uuid.uuid4().hex, 'status': 'running', 'sequence': sequence, 'config': config, 'done': 0}
            if len(JOBS) >= 12:
                JOBS.pop(next(iter(JOBS)))
            JOBS[job['id']] = job
        if self.path == '/api/experiments':
            def update(**values):
                with LOCK:job.update(values)
            threading.Thread(target=detector_lab.run,args=(job,summary,version,config,CLOUDS,update),daemon=True).start()
        else:threading.Thread(target=run_job, args=(job, summary, config), daemon=True).start()
        self.json_response(dict(job), 202)

    def log_message(self, *args):
        pass


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8765)
    args = ap.parse_args()
    print(f'Open http://127.0.0.1:{args.port}/viewer/', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
