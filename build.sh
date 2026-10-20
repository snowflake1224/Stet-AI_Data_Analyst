#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# build.sh — Build and export the v5 Docker image for KDD Cup submission
#
# Usage:
#   chmod +x build.sh
#   ./build.sh
#
# Output: 1270_v5.tar.gz  (submit THIS file, NOT the source code)
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

IMAGE_NAME="1270"
IMAGE_TAG="v5"
OUTPUT_FILE="1270_v5.tar.gz"

echo "========================================"
echo " KDD Cup 2026 — Building Docker image"
echo " Image : ${IMAGE_NAME}:${IMAGE_TAG}"
echo " Output: ${OUTPUT_FILE}"
echo "========================================"

# Build for linux/amd64 (required by the evaluation cluster)
docker buildx build \
  --platform linux/amd64 \
  --load \
  -t "${IMAGE_NAME}:${IMAGE_TAG}" \
  .

echo ""
echo "Build complete. Exporting image..."

# Export as gzipped tar — THIS is what you submit
docker save "${IMAGE_NAME}:${IMAGE_TAG}" | gzip > "${OUTPUT_FILE}"

SIZE=$(du -sh "${OUTPUT_FILE}" | cut -f1)
echo ""
echo "✓ Done! Submit: ${OUTPUT_FILE}  (${SIZE})"
echo ""
echo "Quick sanity check:"
echo "  docker run --rm \\"
echo "    -v /path/to/input:/input \\"
echo "    -v /path/to/output:/output \\"
echo "    -v /tmp/logs:/logs \\"
echo "    -e MODEL_API_KEY=your_key \\"
echo "    ${IMAGE_NAME}:${IMAGE_TAG}"
