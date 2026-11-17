#!/usr/bin/env bash
# Allow the mail-lookup compose bridge to reach llama-swap on the host (tcp/9292).
# The host firewall has `policy drop` and only Docker-published ports are exempt,
# so a host-network listener like llama-swap is unreachable from containers.
# This rule is runtime-only (nft); rerun after a reboot.
set -euo pipefail

NET="${NET:-mailrag-backend_default}"
PORT="${PORT:-9292}"

SUBNET=$(docker network inspect "$NET" -f '{{(index .IPAM.Config 0).Subnet}}')
BR=$(docker network inspect "$NET" -f '{{.Id}}' | cut -c1-12)

if sudo nft list chain inet filter input | grep -q "br-$BR.*dport $PORT"; then
  echo "rule already present for br-$BR ($SUBNET)"
  exit 0
fi

sudo nft insert rule inet filter input iifname "br-$BR" ip saddr "$SUBNET" tcp dport "$PORT" accept
echo "allowed br-$BR $SUBNET -> tcp/$PORT"
