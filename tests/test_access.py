import asyncio
import os

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from edge import access


def make_app(codes_file):
    app = web.Application(middlewares=[access.middleware(access.Codes(codes_file))])
    app.router.add_get("/", lambda r: web.Response(text=f"hi {r['who']}"))
    app.router.add_get("/healthz", lambda r: web.json_response({"ok": True}))
    return app


def run(codes_file, scenario):
    async def go():
        async with TestClient(TestServer(make_app(codes_file))) as client:
            await scenario(client)
    asyncio.run(go())


def test_invite_link_swaps_the_code_for_a_cookie(tmp_path):
    f = tmp_path / "codes.txt"
    f.write_text("# comment\nsecret123 Alice, Fund X\n")

    async def scenario(client):
        assert (await client.get("/")).status == 403
        assert (await client.get("/?code=wrong")).status == 403
        assert (await client.get("/healthz")).status == 200  # monitoring needs no invite
        r = await client.get("/?code=secret123&x=1", allow_redirects=False)
        assert r.status == 302 and r.headers["Location"] == "/?x=1"  # code leaves the address bar
        assert "HttpOnly" in r.headers["Set-Cookie"]
        r = await client.get("/")  # the cookie jar now carries the code
        assert r.status == 200 and await r.text() == "hi Alice, Fund X"

    run(f, scenario)


def test_revoking_a_code_takes_effect_without_a_restart(tmp_path):
    f = tmp_path / "codes.txt"
    f.write_text("secret123 Alice\n")

    async def scenario(client):
        await client.get("/?code=secret123")
        assert (await client.get("/")).status == 200
        f.write_text("other456 Bob\n")
        os.utime(f, (1, 1))  # make sure the mtime changes even within one second
        assert (await client.get("/")).status == 403

    run(f, scenario)


def test_cli_adds_lists_and_revokes(tmp_path, capsys):
    f = tmp_path / "codes.txt"
    access.main([str(f), "add", "Alice,", "Fund", "X"])
    path = capsys.readouterr().out.strip()
    assert path.startswith("/?code=") and len(path) > 15
    assert access.Codes(f).lookup(path.split("=", 1)[1]) == "Alice, Fund X"
    assert oct(f.stat().st_mode)[-3:] == "600"
    access.main([str(f), "list"])
    assert capsys.readouterr().out.strip() == "Alice, Fund X"
    access.main([str(f), "revoke", "Alice, Fund X"])
    assert access.Codes(f).lookup(path.split("=", 1)[1]) is None
