#!/usr/bin/env bash
# Launch the hosted demo on one EC2 instance (Ubuntu 24.04), from your machine with the AWS CLI.
#   REGION=eu-central-1 ./deploy/ec2-launch.sh
# Options (env): REGION, INSTANCE_TYPE (c7g.xlarge), NAME (openarm-teleop-v0), REF (main),
# SITE (public hostname; default <elastic-ip>.sslip.io, which needs no DNS setup).
# Creates: key pair, security group (80/443 open, 22 from your current IP only),
# Elastic IP and the instance. Setup runs at first boot (deploy/setup.sh) and takes ~5 min.
set -euo pipefail
REGION=${REGION:-$(aws configure get region)}
TYPE=${INSTANCE_TYPE:-c7g.xlarge}
NAME=${NAME:-openarm-teleop-v0}
REF=${REF:-main}
REPO_RAW=${REPO_RAW:-https://raw.githubusercontent.com/ysonggit/openarm-teleop-sim}
KEY=~/.ssh/$NAME.pem
ec2() { aws --region "$REGION" ec2 "$@"; }

if [ -n "$(ec2 describe-instances --filters Name=tag:Name,Values="$NAME" \
      Name=instance-state-name,Values=pending,running,stopping,stopped --query 'Reservations[].Instances[].InstanceId' --output text)" ]; then
  echo "an instance named $NAME already exists in $REGION; tear it down first (deploy/ec2-teardown.sh)" >&2
  exit 1
fi

ARCH=$(ec2 describe-instance-types --instance-types "$TYPE" \
  --query 'InstanceTypes[0].ProcessorInfo.SupportedArchitectures[0]' --output text)
[ "$ARCH" = x86_64 ] && ARCH=amd64
AMI=$(aws --region "$REGION" ssm get-parameter --query Parameter.Value --output text \
  --name "/aws/service/canonical/ubuntu/server/24.04/stable/current/$ARCH/hvm/ebs-gp3/ami-id")
MYIP=$(curl -fs https://checkip.amazonaws.com)
echo "region $REGION, $TYPE ($ARCH), AMI $AMI, ssh allowed from $MYIP"

if ! ec2 describe-key-pairs --key-names "$NAME" >/dev/null 2>&1; then  # a local .pem from a torn-down launch is stale
  ec2 create-key-pair --key-name "$NAME" --query KeyMaterial --output text > "$KEY"
  chmod 600 "$KEY"
fi

VPC=$(ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
SG=$(ec2 create-security-group --group-name "$NAME" --description "$NAME web + ssh" --vpc-id "$VPC" \
  --query GroupId --output text)
for port in 80 443; do  # 80 only for the Let's Encrypt challenge and the redirect to https
  ec2 authorize-security-group-ingress --group-id "$SG" --ip-permissions \
    "IpProtocol=tcp,FromPort=$port,ToPort=$port,IpRanges=[{CidrIp=0.0.0.0/0}],Ipv6Ranges=[{CidrIpv6=::/0}]" >/dev/null
done
ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 --cidr "$MYIP/32" >/dev/null

read -r ALLOC EIP < <(ec2 allocate-address --domain vpc \
  --tag-specifications "ResourceType=elastic-ip,Tags=[{Key=Name,Value=$NAME}]" \
  --query '[AllocationId,PublicIp]' --output text)
SITE=${SITE:-${EIP//./-}.sslip.io}

USERDATA=$(mktemp)
cat > "$USERDATA" <<EOF
#!/bin/bash
curl -fsSL $REPO_RAW/$REF/deploy/setup.sh -o /root/setup.sh
SITE=$SITE REF=$REF bash /root/setup.sh > /var/log/openarm-setup.log 2>&1
EOF
ID=$(ec2 run-instances --image-id "$AMI" --instance-type "$TYPE" --key-name "$NAME" \
  --security-group-ids "$SG" --user-data "file://$USERDATA" \
  --metadata-options HttpTokens=required \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=20,VolumeType=gp3}' \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME}]" \
  --query 'Instances[0].InstanceId' --output text)
rm -f "$USERDATA"
echo "instance $ID starting…"
ec2 wait instance-running --instance-ids "$ID"
ec2 associate-address --instance-id "$ID" --allocation-id "$ALLOC" >/dev/null

cat <<EOF

Launched $ID at $EIP. Setup runs on first boot (~5 min).
  site:      https://$SITE
  watch:     ssh -i $KEY ubuntu@$EIP tail -f /var/log/openarm-setup.log
  invite:    ssh -i $KEY ubuntu@$EIP sudo teleop-invite add "Name, Company"
  stop:      aws --region $REGION ec2 stop-instances --instance-ids $ID   (start-instances to resume; same URL)
  teardown:  REGION=$REGION NAME=$NAME CONFIRM=yes ./deploy/ec2-teardown.sh
EOF
