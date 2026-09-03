#!/usr/bin/env bash
# verify_report_artifacts.sh — post-run artifact gate for reportforge outputs.
# status=success is NOT proof the deliverable exists; this script is.
# usage: verify_report_artifacts.sh <output_dir> [min_charts]
# exit 0 = all gates pass; exit 1 = failure with reasons on stdout.
set -u
OUT="${1:?usage: $0 <output_dir> [min_charts]}"
MIN_CHARTS="${2:-1}"
fail=0
say() { printf '%s\n' "$*"; }

[ -f "$OUT/index.html" ] || { say "FAIL: index.html missing"; fail=1; }
[ -f "$OUT/index.pdf" ] || { say "FAIL: index.pdf missing"; fail=1; }
[ -f "$OUT/index.docx" ] || { say "FAIL: index.docx missing"; fail=1; }

if [ -f "$OUT/index.html" ]; then
  n=$(grep -c "<img" "$OUT/index.html")
  [ "$n" -ge "$MIN_CHARTS" ] || { say "FAIL: html has $n <img (min $MIN_CHARTS)"; fail=1; }
  say "html imgs: $n"
fi
if [ -f "$OUT/index.pdf" ]; then
  bad=$(pdftotext "$OUT/index.pdf" - 2>/dev/null | grep -c "Unable to display")
  [ "$bad" -eq 0 ] || { say "FAIL: pdf has $bad unrendered chunks"; fail=1; }
  pages=$(pdfinfo "$OUT/index.pdf" 2>/dev/null | awk '/Pages:/{print $2}')
  words=$(pdftotext "$OUT/index.pdf" - 2>/dev/null | wc -w)
  say "pdf pages: ${pages:-?} words: $words"
fi
if [ -f "$OUT/index.docx" ]; then
  media=$(unzip -l "$OUT/index.docx" | grep -c "word/media/")
  [ "$media" -ge "$MIN_CHARTS" ] || { say "FAIL: docx has $media media (min $MIN_CHARTS)"; fail=1; }
  say "docx media: $media"
fi
[ "$fail" -eq 0 ] && say "PASS: all artifact gates"
exit "$fail"
