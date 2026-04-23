#!/bin/bash
#set -euo pipefail

# Converts a mixed media file list into browser-friendly MP4 outputs for WISE ingestion.
#
# Inputs:
#   1) FILELIST          Null-delimited relative paths (e.g. `find . -type f -print0 > filelist.bin`)
#   2) SOURCE_MEDIA_DIR  Root folder containing the input files
#   3) OUTPUT_MEDIA_DIR  Root folder where output `.mp4` files are written
#
# Behavior:
#   - Images  -> static MP4 clips with exactly one frame at 2 fps (0.5s duration)
#   - Audio   -> audio-only MP4 (`-vn`) so video extractors do not see frames
#   - Videos  -> re-encoded MP4 with yuv420p + faststart for web playback compatibility
#
# Output paths preserve relative structure and base filenames from FILELIST.

start_time=$(date +%s)
echo "Script started at: $(date '+%Y-%m-%d %H:%M:%S')"

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 FILELIST SOURCE_MEDIA_DIR OUTPUT_MEDIA_DIR"
    echo "  - FILELIST: A null-delimited text file containing relative media filenames (e.g. find . -type f -print0 > filelist.bin)"
    echo "  - SOURCE_MEDIA_DIR: Base folder for input images/videos/audio files"
    echo "  - OUTPUT_MEDIA_DIR: Base folder for output mp4 files"
    exit 1
fi

FILELIST=$1
INDIR=$2
OUT_BASEDIR=$3
OUTDIR="${OUT_BASEDIR}/"

mkdir -p "$OUTDIR"

process_file() {
    file_rel="${1#./}"              # strip leading ./ if present
    infile="$INDIR/$file_rel"
    outfile_base="${file_rel%.*}"   # Remove the extension part
    outfile="${OUTDIR}/${outfile_base}.mp4"
    outdir=$(dirname "$outfile")
    tmpfile="${outfile}.inprogress"

    if [ ! -f "$infile" ]; then
        echo "Missing input: $infile" >&2
        return 1
    fi

    if [ -f "$outfile" ]; then
        echo "Skipping (already done): $outfile"
        return 0
    fi

    if [ -f "$tmpfile" ]; then
        echo "Skipping (ongoing by other process): $tmpfile"
        return 0
    fi
    mkdir -p "$outdir"
    echo "Converting: $infile → $outfile"
    lower_infile="${infile,,}"

    if [[ "$lower_infile" =~ \.(jpg|jpeg|png|bmp|webp|tif|tiff|gif)$ ]]; then
        # Image input:
        # - create one encoded frame at 2 fps => mp4 duration is exactly 0.5 sec
        # - keep frame visually identical (static)
        if ! </dev/null ffmpeg -hide_banner -loglevel warning \
            -loop 1 -framerate 2 \
            -i "$infile" \
            -frames:v 1 -r 2 \
            -vf "format=yuv420p" \
            -an \
            -c:v libx264 -preset slow -crf 23 \
            -movflags +faststart \
            -f mp4 \
            "$tmpfile"; then
            echo "Failed to convert image: $infile" >&2
            rm -f "$tmpfile"
            return 1
        fi
    elif [[ "$lower_infile" =~ \.(mp3|m4a|aac|wav|flac|ogg|opus|wma)$ ]]; then
        # Audio-only input:
        # - keep output in mp4 container
        # - force no video stream so video extractors do not see any frames
        if ! </dev/null ffmpeg -hide_banner -loglevel warning \
            -fflags +genpts+discardcorrupt -err_detect ignore_err \
            -i "$infile" \
            -vn \
            -c:a aac -b:a 128k \
            -movflags +faststart \
            -f mp4 \
            "$tmpfile"; then
            echo "Failed to convert audio: $infile" >&2
            rm -f "$tmpfile"
            return 1
        fi
    else
        # Video input.
        if ! </dev/null ffmpeg -hide_banner -loglevel warning \
            -fflags +genpts+discardcorrupt -err_detect ignore_err \
            -i "$infile" \
            -vf "format=yuv420p" \
            -ac 2 \
            -c:v libx264 -preset slow -crf 23 \
            -c:a aac -b:a 128k \
            -movflags +faststart \
            -f mp4 \
            "$tmpfile"; then
            echo "Failed to convert media: $infile" >&2
            rm -f "$tmpfile"
            return 1
        fi
    fi

    mv "$tmpfile" "$outfile"
}

export -f process_file
export INDIR OUTDIR

xargs -0 -n1 -P 16 bash -c 'process_file "$1"' _ < "$FILELIST"

end_time=$(date +%s)
echo "Script ended at:   $(date '+%Y-%m-%d %H:%M:%S')"

duration=$((end_time - start_time))
echo "Total time taken:  ${duration} seconds"
