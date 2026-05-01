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
#
# CPU control:
#   - MAX_JOBS controls parallel files processed via xargs (default: 16)
#   - FFMPEG_THREADS controls ffmpeg threads per file (default: 4)
#   - Approx max CPU threads used: MAX_JOBS * FFMPEG_THREADS

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
MAX_JOBS="${MAX_JOBS:-16}"
FFMPEG_THREADS="${FFMPEG_THREADS:-4}"

if ! [[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: MAX_JOBS must be a positive integer, got: $MAX_JOBS" >&2
    exit 1
fi

if ! [[ "$FFMPEG_THREADS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: FFMPEG_THREADS must be a positive integer, got: $FFMPEG_THREADS" >&2
    exit 1
fi

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
            -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p" \
            -an \
            -threads "$FFMPEG_THREADS" \
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
            -threads "$FFMPEG_THREADS" \
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
            -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p" \
            -ac 2 \
            -threads "$FFMPEG_THREADS" \
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
export INDIR OUTDIR FFMPEG_THREADS

xargs -0 -n1 -P "$MAX_JOBS" bash -c 'process_file "$1"' _ < "$FILELIST"

end_time=$(date +%s)
echo "Script ended at:   $(date '+%Y-%m-%d %H:%M:%S')"

duration=$((end_time - start_time))
echo "Total time taken:  ${duration} seconds"
