#!/usr/bin/env bash
# Delete everything ec2-launch.sh created: instance, Elastic IP, security group, key pair.
#   REGION=eu-central-1 NAME=openarm-teleop-v0 CONFIRM=yes ./deploy/ec2-teardown.sh
# Invite codes and session logs live on the instance and are deleted with it.
set -euo pipefail
REGION=${REGION:-$(aws configure get region)}
NAME=${NAME:-openarm-teleop-v0}
[ "${CONFIRM:-}" = yes ] || { echo "this deletes the $NAME instance and its data in $REGION; re-run with CONFIRM=yes" >&2; exit 1; }
ec2() { aws --region "$REGION" ec2 "$@"; }

IDS=$(ec2 describe-instances --filters Name=tag:Name,Values="$NAME" \
  Name=instance-state-name,Values=pending,running,shutting-down,stopping,stopped --query 'Reservations[].Instances[].InstanceId' --output text)
if [ -n "$IDS" ]; then
  ec2 terminate-instances --instance-ids $IDS >/dev/null
  echo "terminating ${IDS}..."
  ec2 wait instance-terminated --instance-ids $IDS
fi
for ALLOC in $(ec2 describe-addresses --filters Name=tag:Name,Values="$NAME" --query 'Addresses[].AllocationId' --output text); do
  ec2 release-address --allocation-id "$ALLOC" && echo "released Elastic IP $ALLOC"
done
SG=$(ec2 describe-security-groups --filters Name=group-name,Values="$NAME" --query 'SecurityGroups[0].GroupId' --output text)
[ "$SG" != None ] && ec2 delete-security-group --group-id "$SG" >/dev/null && echo "deleted security group $SG"
ec2 delete-key-pair --key-name "$NAME" >/dev/null && echo "deleted key pair $NAME (local copy kept at ~/.ssh/$NAME.pem)"
