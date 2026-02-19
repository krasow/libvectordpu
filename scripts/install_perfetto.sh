#!/bin/bash
set -e

DEST_DIR="${1:-/scratch/david/benchmark-upmem/opt/perfetto}"
PERFETTO_REPO="https://github.com/google/perfetto.git"
PERFETTO_TAG="v50.1"
TEMP_DIR=$(mktemp -d)

echo "Installing Perfetto SDK to $DEST_DIR..."

cleanup() {
    echo "Cleaning up..."
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

echo "Cloning Perfetto..."
git clone --depth 1 --branch "$PERFETTO_TAG" "$PERFETTO_REPO" "$TEMP_DIR"

cd "$TEMP_DIR"

mkdir -p "$DEST_DIR/include"
mkdir -p "$DEST_DIR/lib"

cp sdk/perfetto.h "$DEST_DIR/include/"

echo "Building static library..."
g++ -std=c++17 -O3 -fPIC -c sdk/perfetto.cc -o sdk/perfetto.o -I. -pthread
ar rcs "$DEST_DIR/lib/libperfetto.a" sdk/perfetto.o

echo "Perfetto SDK installed successfully to $DEST_DIR"
echo "You can now build with: make TRACE=1 PERFETTO_HOME=$DEST_DIR"
