#!/usr/bin/env bash

set -e

PYTHON_VERSION="3.12"
INSTALL_DIR="$HOME/.local/python-$PYTHON_VERSION"
LINK_DIR="$HOME/.local/bin"

# Check that a path was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 /absolute/path/to/Python-source"
    exit 1
fi

SOURCE_DIR="$1"

# Check that the directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: directory does not exist: $SOURCE_DIR"
    exit 1
fi

# Check that it looks like a Python source directory
if [ ! -f "$SOURCE_DIR/configure" ]; then
    echo "Error: configure script not found in: $SOURCE_DIR"
    exit 1
fi

echo "Python source: $SOURCE_DIR"
echo "Install location: $INSTALL_DIR"
echo

cd "$SOURCE_DIR"

echo "Configuring Python..."
./configure --prefix="$INSTALL_DIR"

echo "Building Python..."
make -j"$(nproc)"

echo "Installing Python..."
make altinstall

echo "Creating command..."
mkdir -p "$LINK_DIR"

ln -sf \
    "$INSTALL_DIR/bin/python$PYTHON_VERSION" \
    "$LINK_DIR/python$PYTHON_VERSION"

echo
echo "Installation complete!"
echo "Python installed to: $INSTALL_DIR"
echo "Run it with: python$PYTHON_VERSION"
