#!/usr/bin/env bash
# One-time setup of the hosted demo on Ubuntu 24.04. Run as root, e.g. from EC2 user data:
#   SITE=demo.example.com REF=main bash setup.sh
# SITE is the public hostname Caddy gets a TLS certificate for. Safe to re-run (updates to REF).
set -euo pipefail
SITE=${SITE:?set SITE to the public hostname}
REF=${REF:-main}
REPO=${REPO:-https://github.com/ysonggit/openarm-teleop-sim.git}
APP=/opt/openarm-teleop-sim
CODES=/etc/openarm-teleop/codes.txt
export DEBIAN_FRONTEND=noninteractive

echo "== packages"
apt-get update -q
apt-get install -y -q git curl gnupg debian-keyring debian-archive-keyring apt-transport-https libosmesa6
if ! command -v caddy >/dev/null; then  # Caddy's official apt repository
  curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/gpg.key | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -q
  apt-get install -y -q caddy
fi
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
fi

echo "== app at $REF"
id teleop >/dev/null 2>&1 || useradd --system --create-home --shell /usr/sbin/nologin teleop
[ -d "$APP/.git" ] || git clone -q "$REPO" "$APP"
git -C "$APP" fetch -q origin
git -C "$APP" checkout -q "$REF"
git -C "$APP" pull -q --ff-only 2>/dev/null || true  # when REF is a branch
mkdir -p "$APP/logs"
chown -R teleop:teleop "$APP"
sudo -u teleop -H bash -c "cd $APP && uv sync --frozen -q && ./scripts/fetch_model.sh"

echo "== headless render check"
sudo -u teleop -H env MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa MESA_SHADER_CACHE_DISABLE=true \
  bash -c "cd $APP && .venv/bin/python -m edge.selftest"

echo "== service"
install -d -o teleop -g teleop -m 700 /etc/openarm-teleop
[ -f "$CODES" ] || install -o teleop -g teleop -m 600 /dev/null "$CODES"
cp "$APP/deploy/openarm-teleop.service" /etc/systemd/system/
sed "s/{SITE}/$SITE/" "$APP/deploy/Caddyfile" > /etc/caddy/Caddyfile
cat > /usr/local/bin/teleop-invite <<EOF
#!/usr/bin/env bash
# teleop-invite add "Name, Company" | list | revoke "Name, Company"
set -euo pipefail
out=\$(sudo -u teleop $APP/.venv/bin/python -m edge.access $CODES "\$@")
if [ "\${1:-}" = add ]; then echo "https://$SITE\$out"; else echo "\$out"; fi
EOF
chmod 755 /usr/local/bin/teleop-invite
systemctl daemon-reload
systemctl enable -q openarm-teleop
systemctl restart openarm-teleop
systemctl reload-or-restart caddy

for i in $(seq 1 60); do curl -fs http://127.0.0.1:8765/healthz >/dev/null && break; sleep 2; done
curl -fs http://127.0.0.1:8765/healthz && echo
echo "== done: https://$SITE  (create invites with: sudo teleop-invite add \"Name, Company\")"
