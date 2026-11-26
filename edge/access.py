"""Invite codes for a hosted demo.

Each invitee gets a personal link, https://<site>/?code=<code>. The first visit
sets an HttpOnly cookie, so later page loads and the WebSockets need no code in
the URL. Codes live in a plain file, one per line: `<code> <label>`. Add or
revoke people by editing it (or with the commands below); the server rereads
the file when it changes, so no restart is needed. A revoked code stops working
on the next page load or reconnect. Without --codes the server is open, for
local development.

    python -m edge.access FILE add "Alice, Fund X"    # prints the invite path
    python -m edge.access FILE list
    python -m edge.access FILE revoke "Alice, Fund X"
"""

import hmac
import secrets
import sys
from pathlib import Path

from aiohttp import web

COOKIE = "teleop_access"
MAX_AGE = 30 * 24 * 3600  # s
OPEN_PATHS = {"/healthz"}
DENIED = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Invite only</title><body style="font:16px/1.5 system-ui;max-width:32rem;margin:15vh auto;padding:0 16px">
<h1 style="font-size:20px">This demo is invite-only</h1>
<p>Open the personal link you were sent. If it stopped working, ask for a new one.</p></body>"""


class Codes:
    def __init__(self, path):
        self.path = Path(path)
        self._mtime = None
        self._codes = {}  # code -> label

    def _reload(self):
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            self._mtime, self._codes = None, {}
            return
        if mtime == self._mtime:
            return
        codes = {}
        for line in self.path.read_text().splitlines():
            parts = line.strip().split(maxsplit=1)
            if parts and not parts[0].startswith("#"):
                codes[parts[0]] = parts[1] if len(parts) > 1 else parts[0][:4]
        self._mtime, self._codes = mtime, codes

    def lookup(self, code):
        """Label for a valid code, else None (constant-time compare)."""
        self._reload()
        found = None
        for c, label in self._codes.items():
            if hmac.compare_digest(c.encode(), (code or "").encode()):
                found = label
        return found


def middleware(codes):
    @web.middleware
    async def invite_only(request, handler):
        if request.path in OPEN_PATHS:
            return await handler(request)
        code = request.query.get("code")
        if code is not None and codes.lookup(code):
            # Swap the code in the URL for a cookie, so it doesn't sit in the address bar or history.
            rest = request.rel_url.with_query({k: v for k, v in request.query.items() if k != "code"})
            resp = web.Response(status=302, headers={"Location": str(rest)})
            https = request.secure or request.headers.get("X-Forwarded-Proto") == "https"
            resp.set_cookie(COOKIE, code, max_age=MAX_AGE, httponly=True, samesite="Lax", secure=https)
            return resp
        label = codes.lookup(request.cookies.get(COOKIE))
        if label is None:
            return web.Response(status=403, text=DENIED, content_type="text/html")
        request["who"] = label
        return await handler(request)

    return invite_only


def main(argv):
    if len(argv) < 2 or argv[1] not in ("add", "list", "revoke"):
        sys.exit(__doc__)
    path, cmd, args = Path(argv[0]), argv[1], argv[2:]
    lines = path.read_text().splitlines() if path.exists() else []
    if cmd == "list":
        for line in lines:
            parts = line.split(maxsplit=1)
            if parts and not parts[0].startswith("#"):
                print(parts[1] if len(parts) > 1 else "(no label)")
    elif cmd == "add":
        label = " ".join(args) or sys.exit("add needs a label, e.g. the person's name")
        code = secrets.token_urlsafe(12)
        path.write_text("\n".join(lines + [f"{code} {label}"]) + "\n")
        path.chmod(0o600)
        print(f"/?code={code}")
    else:
        label = " ".join(args)
        keep = [ln for ln in lines if ln.split(maxsplit=1)[1:] != [label]]
        if len(keep) == len(lines):
            sys.exit(f"no invite labelled {label!r}")
        path.write_text("\n".join(keep) + "\n" if keep else "")
        print(f"revoked {len(lines) - len(keep)} invite(s) for {label!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
