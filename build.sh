#!/usr/bin/env bash
set -e

docker build -f Dockerfile.serving -t ift6758-serving .
