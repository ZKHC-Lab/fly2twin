"""Loopback-only, no-store static server for demo cache sites (standard library)."""
import argparse
import functools
import http.server
import json
from pathlib import Path


class NoCache(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, must-revalidate')
        super().end_headers()

    def send_head(self):
        for key in ('If-Modified-Since', 'If-None-Match'):
            if key in self.headers:
                del self.headers[key]
        return super().send_head()

    def do_GET(self):
        if self.path == '/__drone3d_demo__':
            data = json.dumps({'tool': 'drone3d-demo-v1', 'root': str(Path(self.directory).resolve())}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == '/favicon.ico':
            self.send_response(204)
            self.end_headers()
        else:
            super().do_GET()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--port', type=int, default=8091)
    args = parser.parse_args()
    handler = functools.partial(NoCache, directory=str(args.root.resolve()))
    with http.server.ThreadingHTTPServer(('127.0.0.1', args.port), handler) as server:
        server.serve_forever()


if __name__ == '__main__':
    main()
