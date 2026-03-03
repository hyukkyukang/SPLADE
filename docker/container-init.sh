#!/usr/bin/env bash
set -euo pipefail

SUDO_PASSWORD="user"
run_sudo() {
  echo "${SUDO_PASSWORD}" | sudo -S "$@"
}

bundle_dir="/home/user/.local/share/ca-certificates"
bundle_file="${bundle_dir}/combined-ca-bundle.crt"
mkdir -p "${bundle_dir}"
cp /etc/ssl/certs/ca-certificates.crt "${bundle_file}"

# Add org TLS inspection CAs to a user-level bundle.
if compgen -G "/host-ca-certificates/*.crt" > /dev/null; then
  echo "Appending host CA certificates from /host-ca-certificates"
  cat /host-ca-certificates/*.crt >> "${bundle_file}"
else
  echo "No host CA certificates found; appending built-in paloalto.lgresearch.ai root CA"
  cat >> "${bundle_file}" <<'EOF_CERT'
-----BEGIN CERTIFICATE-----
MIIC3TCCAcWgAwIBAgIEPY+mRDANBgkqhkiG9w0BAQsFADAhMR8wHQYDVQQDExZw
YWxvYWx0by5sZ3Jlc2VhcmNoLmFpMB4XDTI0MTExNzAzMjE1NFoXDTM0MTExNTAz
MjE1NFowITEfMB0GA1UEAxMWcGFsb2FsdG8ubGdyZXNlYXJjaC5haTCCASIwDQYJ
KoZIhvcNAQEBBQADggEPADCCAQoCggEBAMl4kohlOtqPiTx6FSXcnKJuT3iaa9ey
g+XusqGdCnXUP7C77kXU2jx5A7X4n2pBBsDwLjswhlbolUrMsCZXD/ky3slAWjEa
nijqlHkklZEd7gQubmw3j20bV/oPcPA6ZLvYM7Tp77BUNp89Oqv7aLu6tRBWTf7x
zaEiElAm9BjErIgJ7rrYdHzEbyG9jzd/LQNzAqLXviApQukw88tTuqDRkPByGY4X
XbiF2uRootQ5eNEyMKjqg/yb/O9Kh68U9ryN9V0uJfukuRfOoE31QSjMjfLVzD/T
HVLj/dcK7eRdzvP8Tf43YuiY2a8UT/zpTtqHKTPkip6VS/cbyYq4mkkCAwEAAaMd
MBswDAYDVR0TBAUwAwEB/zALBgNVHQ8EBAMCAgQwDQYJKoZIhvcNAQELBQADggEB
ALPXmTMaN5CvbVVDNX63BAoBZPd8R49GbrtK6z/LUFHNSSpFQSVahvVW1xHgsK3c
OlfqhX0YfJOM0V6KsWnoKFUsM4MK+ikzCvn3ddKURk7IfeLCAyLuaQpimTSt7qTQ
Ci855ODdzNuEk8sdPgN7pPjn93GjMO67U4A+HTViPkrv0PbdMc7R2bRoHJYEi97C
5V7l1UCpy8iHDSk2a/VJ75JO4cL622CRQ4GXPTzaQ2QC88dlZ3MD+6SbLAbMUOvR
WlmPebHhHkP9bLGUfJBmLL/I8kSWlhOkCNkmnOA7P1gQzoLje90SeFgcd47dkgRt
V/xYnEWN2+7rz57rxrOz1v0=
-----END CERTIFICATE-----
EOF_CERT
fi
chmod 644 "${bundle_file}"

if ! grep -qx 'export PATH=/home/user/.local/bin:$PATH' /home/user/.bashrc; then
  echo 'export PATH=/home/user/.local/bin:$PATH' >> /home/user/.bashrc
fi

# Keep permission and UID/GID setup behavior for mounted volumes.
run_sudo chmod 777 /home/user/
run_sudo chmod 777 /home/user/.*
run_sudo chmod 777 /tmp/

host_gid="${HOST_GID:-1000}"
host_uid="${HOST_UID:-1000}"
run_sudo groupmod -g "${host_gid}" user
run_sudo usermod -u "${host_uid}" -g "${host_gid}" user

tail -f /dev/null
