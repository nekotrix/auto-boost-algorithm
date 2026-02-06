#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "vsjetpack",
#   "rich",
# ]
# ///

# Requires manually installing:
# SVT-AV1-Essential: https://github.com/nekotrix/SVT-AV1-Essential/discussions/12
# in your system PATH or the script's directory, and:
# Vship (GPU):       https://github.com/Line-fr/Vship/releases
# or vs-zip (CPU):   https://github.com/dnjulek/vapoursynth-zip/releases/
# and FFMS2:         https://github.com/FFMS/ffms2/releases
# in the VapourSynth plugin directory

# Auto-Boost-Essential
# Copyright (c) Trix and contributors
# Thanks to the AV1 discord community members <3
#
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.

from vstools import vs, core, depth, DitherType, clip_async_render
try:
    from vstools.functions.progress import get_render_progress, FPSColumn
except:
    from vstools.functions.render.progress import get_render_progress, FPSColumn
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn
from rich.console import Console
from statistics import quantiles
from math import ceil, log10
from pathlib import Path
import subprocess
import argparse
import platform
import shutil
import struct
import glob
import sys
import gc
import os
import re

ver_str = "v2.0 (Pre-Release)"

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--stage", help = "Select stage: 1 = fast encode, 2 = calculate metrics, 3 = generate zones, 4 = final encode | Default: all", default=0)
parser.add_argument("-i", "--input", required=True, help = "Video input filepath (original source file)")
parser.add_argument("-t", "--temp", help = "The temporary directory for the script to store files in | Default: video input filename")
parser.add_argument("--fast-speed", help = "Fast encode speed (Allowed: medium, fast, faster) | Default: faster", default="faster")
parser.add_argument("--final-speed", help = "Final encode speed (Allowed: slower, slow, medium, fast, faster) | Default: slow", default="slow")
parser.add_argument("--quality", help = "Base encoder --quality (Allowed: low, medium, high) | Default: medium", default="medium")
parser.add_argument("-a", "--aggressive", action='store_true', help = "More aggressive boosting | Default: not active")
parser.add_argument("-u", "--unshackle", action='store_true', help = "Less restrictive boosting | Default: not active")
parser.add_argument("--fast-params", help="Custom fast encoding parameters")
parser.add_argument("--final-params", help="Custom final encoding parameters")
#parser.add_argument("-g", "--grain-format", help = "Select grain format: 1 = SVT-AV1 film-grain, 2 = Photon-noise table | Default: 1", default=1)
parser.add_argument(
    '--ssimu2',
    nargs='?',
    const='auto',
    choices=['auto', 'cpu', 'gpu'],
    help='SSIMU2 mode: auto (default when flag used), cpu (vs-zip), or gpu (Vship)'
)
parser.add_argument("--verbose", action='store_true', help = "Enable more verbosity | Default: not active")
parser.add_argument("-r", "--resume", action='store_true', help = "Resume the process from the last (un)completed stage | Default: not active")
parser.add_argument("-nb", "--no-boosting", action='store_true', help = "Runs the script without boosting (final encode only) | Default: not active")
parser.add_argument("-v", "--version", action='version', version = f"Auto-Boost-Essential {ver_str}")
parser.add_argument("--debug", action='store_true', help = "Checks the installation and provides relevant information for troubleshooting | Default: not active")

args = parser.parse_args()

stage = int(args.stage)
src_file = Path(args.input).resolve()
if platform.system() == 'Windows':
    src_file = type(src_file)(r"\\?" + rf"\{src_file}")
file_ext = src_file.suffix
output_dir = src_file.parent
if args.temp is not None:
    tmp_dir = Path(args.temp).resolve()
    if platform.system() == 'Windows':
        tmp_dir = type(tmp_dir)(r"\\?" + rf"\{tmp_dir}")
else:
    tmp_dir = output_dir / src_file.stem
vpy_file = tmp_dir / f"{src_file.stem}.vpy"
cache_file = tmp_dir / f"{src_file.stem}.ffindex"
fast_output_file = tmp_dir / f"{src_file.stem}_fastpass.ivf"
tmp_final_output_file = tmp_dir / f"{src_file.stem}.ivf"
final_output_file = output_dir / f"{src_file.stem}.ivf"
ssimu2_log_file = tmp_dir / f"{src_file.stem}_ssimu2.log"
xpsnr_log_file = tmp_dir / f"{src_file.stem}_xpsnr.log"
zones_file = tmp_dir / f"{src_file.stem}_zones.cfg"
stage_file = tmp_dir / f"{src_file.stem}_stage.txt"
stage_resume = 0
fast_speed = args.fast_speed
final_speed = args.final_speed
quality = args.quality
aggressive = args.aggressive
unshackle = args.unshackle
fast_params = args.fast_params if args.fast_params is not None else ""
final_params = args.final_params if args.final_params is not None else ""
#grain_format = args.grain_format # upcoming auto-FGS feature
ssimu2 = args.ssimu2 if args.ssimu2 is not None else ""
verbose = args.verbose
resume = args.resume
no_boosting = args.no_boosting

if args.debug:
    print("=" * 54)
    print("SYSTEM INFORMATION")
    print("=" * 54)

    print(f"System: {platform.platform()}")

    # Get current username for censoring
    env_vars = ['USERNAME', 'USER', 'LOGNAME']
    for var in env_vars:
        username = os.getenv(var)
        if username:
            break

    print(f"Python Version: {sys.version}")

    python_path = sys.executable
    censored_python_path = python_path.replace(username, '[USER]')
    print(f"Python Path: {censored_python_path}")

    current_dir = os.getcwd()
    censored_dir = current_dir.replace(username, '[USER]')
    print(f"Current Directory: {censored_dir}")

    print("=" * 54)
    print("SVT-AV1 VERSION INFORMATION")
    print("=" * 54)

    try:
        result = subprocess.run(['SvtAv1EncApp'], capture_output=True, text=True, timeout=5)

        lines = result.stderr.splitlines()
        info_lines = [line for line in lines if line.startswith('Svt[info]')]

        for line in info_lines[1:4]:
            print(line)
    except subprocess.TimeoutExpired:
        print("Command timed out")
    except FileNotFoundError:
        print("ERROR: SvtAv1EncApp not found. Make sure it's in your PATH")
    except PermissionError:
        print("ERROR: Permission denied when trying to execute SvtAv1EncApp")
    except Exception as e:
        print(f"An error occurred: {e}")

    print("=" * 54)
    print("SCRIPT INFORMATION")
    print("=" * 54)

    print(f"Auto-Boost-Essential {ver_str}")
    print(f"Vapoursynth Version: {vs.__version__[0]}.{vs.__version__[1]}")
    if hasattr(core, 'vszip'):
        print(f"Vs-zip Version: {core.vszip.version[0]}.{core.vszip.version[1]}")
    else:
        print("Vs-zip not present")
    if hasattr(core, 'vship'):
        print(f"Vship Version: {core.vship.version[0]}.{core.vship.version[1]}")
    else:
        print("Vship not present")

    censored_src_file = str(src_file).replace(username, '[USER]')
    print(f'Source File: "{censored_src_file}"')

    if fast_params:
        print(f'Fast Parameters: "{fast_params}"')
    if final_params:
        print(f'Final Parameters: "{final_params}"')

    print("=" * 54)
    print("=" * 54)
    raise SystemExit(1)

if not os.path.exists(src_file):
    print("The source input doesn't exist. Double-check the provided path.")
    raise SystemExit(1)

if "--preset" in fast_params.split():
    index = fast_params.split().index("--preset")
    fast_speed = int(fast_params.split()[index+1])
else:
    if fast_speed not in ["medium", "fast", "faster"]:
        print("The fast pass speed must be either medium, fast or faster.")
        raise SystemExit(1)

if "--preset" in final_params.split():
    index = final_params.split().index("--preset")
    final_speed = int(final_params.split()[index+1])
else:
    if final_speed not in ["slower", "slow", "medium", "fast", "faster"]:
        print("The final pass speed must be either slower, slow, medium, fast or faster.")
        raise SystemExit(1)

if "--crf" in fast_params:
    index = fast_params.index("--crf")
    try:
        quality = float(fast_params[index+6:index+11])
    except:
        try:
            quality = float(fast_params[index+6:index+10])
        except:
            try:
                quality = float(fast_params[index+6:index+8])
            except:
                print("CRF must have 0, 1 or 2 decimals.")
                raise SystemExit(1)
else:
    if quality not in ["low", "medium", "high"]:
        print("The quality preset must be either low, medium or high.")
        raise SystemExit(1)

if stage != 0 and resume:
    print("Resume will auto-resume from the last (un)completed stage. You cannot provide both stage and resume.")
    raise SystemExit(1)

if os.path.exists(tmp_dir):
    if resume and os.path.exists(stage_file): 
        with open(stage_file, "r") as file:
            lines = file.readlines()
            stage_resume = int(lines[0].strip())
            if stage_resume == 5:
                print('Final encode already finished. Nothing to resume.')
                raise SystemExit(0)
            else:
                print(f'Resuming from stage {stage_resume}.')

    if not resume and stage in [0, 1]:
        shutil.rmtree(tmp_dir)

if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

if not os.path.exists(vpy_file):
    with open(vpy_file, 'w') as file:          
        file.write(
f"""
from vstools import vs, core, depth, DitherType, set_output
core.max_cache_size = 1024
src = core.ffms2.Source(source=r"{src_file}", cachefile=r"{cache_file}")
bit_to_format = {{
    8: vs.YUV420P8,
    10: vs.YUV420P10,
    12: vs.YUV420P12
}}
bit_to_dither = {{
    8: DitherType.NONE,
    10: DitherType.NONE,
    12: DitherType.AUTO
}}
fmt = bit_to_format.get(src.format.bits_per_sample, vs.YUV420P16)
dt = bit_to_dither.get(src.format.bits_per_sample, DitherType.AUTO)
src = depth(src.resize.Bilinear(format=fmt), 10, dither_type=dt)
set_output(src)
"""
        )

core.max_cache_size = 1024
console = Console()

def read_from_offset(file_path: Path, offset: int, size: int) -> bytes:
    with open(file_path, 'rb') as file:
        file.seek(offset)
        data = file.read(size)
    return data

def merge_ivf_parts(base_path: Path, output_path: Path, fwidth: int, fheight: int, ffpsnum: int, ffpsden: int) -> None:
    # Collect ivf parts
    base = base_path.stem.split("__")[0]
    base_escaped = glob.escape(base)
    parts = sorted(
        base_path.parent.glob(f"{base_escaped}__*.ivf"),
        key=lambda p: int(p.stem.split("__")[-1])
    )
    final_part = base_path.parent / f"{base}.ivf"
    if final_part.exists():
        parts.append(final_part)

    if not parts:
        print("No parts found to merge. Muxing aborted.")
        return

    num_frames = 0
    framedata = b''
    for i, part in enumerate(parts):

        if not os.path.exists(part):
            print(f"Part {i} not found. Muxing aborted.")
            return

        num_frames += int.from_bytes(read_from_offset(part, 24, 4), 'little')
        framedata += read_from_offset(part, 32, -1)

    with open(output_path, "wb+") as file:
        header = struct.pack(
            '<4sHH4sHHIII4s',
            b'DKIF',        # Signature                             0x00
            0,              # Version                               0x04
            32,             # Header size (don't change this)       0x06
            b'AV01',        # Codec FourCC                          0x08
            fwidth,         # Width                                 0x0C
            fheight,        # Height                                0x0E
            ffpsnum,        # Framerate numerator                   0x10
            ffpsden,        # Framerate denominator                 0x14
            num_frames,     # Number of frames (can be 0 initially) 0x18
            b'\0\0\0\0'     # Reserved                              0x1C
            # Follows array of frame headers
        )
        file.write(header)
        file.write(framedata)
        offset = 32 # Frame data start
        for i in range(num_frames): # Rewrite timestamps
            file.seek(offset)                                # Jump to header
            size = int.from_bytes(file.read(4), 'little')    # Get size of frame data
            file.write(i.to_bytes(8, "little"))              # Rewrite the timestamp
            offset += 12 + size                              # Size of frame + size of frame header

    if verbose:
        console.print(f"Merged {len(parts)} chunks into {output_path} ({num_frames} total frames)")

def create_offset_zones_file(original_zones_path: Path, offset_zones_path: Path, offset_frames: int) -> None:
    """
    Creates a new zones file with frame ranges offset by the specified number of frames.
    Removes zones that become invalid (end <= 0).

    :param original_zones_path: path to original zones file
    :type original_zones_path: Path
    :param offset_zones_path: path to new offset zones file
    :type offset_zones_path: Path
    :param offset_frames: number of frames to subtract from zone ranges
    :type offset_frames: int
    """
    if no_boosting:
        return

    if not original_zones_path.exists():
        print(f"Original zones file {original_zones_path} not found!")
        return

    with original_zones_path.open("r") as file:
        zones_content = file.read().strip()

    if not zones_content.startswith("Zones :"):
        print(f"Invalid zones file format in {original_zones_path}")
        return

    zones_data = zones_content.replace("Zones :", "").strip()
    zone_parts = [zone.strip() for zone in zones_data.split(";") if zone.strip()]

    offset_zones = []
    for zone in zone_parts:
        parts = zone.split(",")
        if len(parts) not in [3, 4]:
            continue

        start = int(parts[0])
        end = int(parts[1])
        crf = parts[2]

        new_start = start - offset_frames
        new_end = end - offset_frames

        # Skip invalid zones
        if new_end <= 0:
            continue

        # Clamp start to 0 if it goes negative (though this shouldn't happen with keyframe boundaries)
        if new_start < 0:
            new_start = 0

        offset_zones.append(f"{new_start},{new_end},{crf}")

    if offset_zones:
        with offset_zones_path.open("w") as file:
            file.write(f"Zones : {';'.join(offset_zones)};")

        if verbose:
            console.print(f"Offset: {offset_frames} frames")
            console.print(f"Zones: {len(zone_parts)} -> {len(offset_zones)}")
    else:
        print(f"No valid zones remaining after offset of {offset_frames} frames")

def read_ivf_frames(path: Path) -> tuple[bytes, list]:
    frames = []
    with open(path, "rb") as file:
        header = file.read(32) # IVF header
        while True:
            frame_header = file.read(12)
            if len(frame_header) < 12:
                break
            size, timestamp = struct.unpack("<IQ", frame_header)
            frame_data = file.read(size)
            if len(frame_data) < size:
                break
            frames.append((size, timestamp, frame_data))
    return header, frames

def trim_ivf_from_last_keyframe(ivf_path: Path, ivf_out_path: Path, last_gop_start_index: int) -> None:
    header, frames = read_ivf_frames(ivf_path)
    trimmed_frames = frames[:last_gop_start_index]
    if verbose:
        console.print(f"Encode frame count: {len(frames)}")
        console.print(f"Keeping {len(trimmed_frames)} frames (removing last GOP starting at frame {last_gop_start_index})")

    with open(ivf_out_path, "wb") as file:
        new_header = bytearray(header)
        new_header[24:28] = struct.pack("<I", len(trimmed_frames))
        file.write(new_header)

        for size, timestamp, frame_data in trimmed_frames:
            file.write(struct.pack("<IQ", size, timestamp))
            file.write(frame_data)

def get_next_filename(base_path: Path) -> Path:
    """
    Gets the next available ivf or zones filename for resumed encodes.

    :param base_path: path to base file
    :type base_path: Path

    :return: path to next available file
    :rtype: Path
    """
    base = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent

    base_escaped = glob.escape(base)
    files = sorted(parent.glob(f"{base_escaped}__*{suffix}"), key=lambda x: int(x.stem.split("__")[-1]))
    if not files:
        return parent / f"{base}__1{suffix}"

    last_index = int(files[-1].stem.split("__")[-1])
    return parent / f"{base}__{last_index + 1}{suffix}"

def get_total_previous_frames(enc_file: Path) -> int:
    """
    Sum the frame counts of all previously trimmed encode files like encode__1.ivf + encode__2.ivf...

    :param enc_file: path to encode
    :type enc_file: Path

    :return: frame number
    :rtype: int
    """
    base = enc_file.stem.split("__")[0]
    base_escaped = glob.escape(base)
    ivf_files = sorted(enc_file.parent.glob(f"{base_escaped}__*.ivf"), key=lambda x: int(x.stem.split('__')[-1]))

    total = 0
    for ivf in ivf_files:
        with open(ivf, "rb") as file:
            file.seek(24)
            frame_count = int.from_bytes(file.read(4), "little")
            total += frame_count
    return total

def get_file_info(vfile: Path, mode: str) -> tuple[list[int], bool, int, int, int, int, int]:
    """
    Parse a video file for information including keyframes placement.

    :param file: path to file
    :type file: Path
    :param mode: informs the function what to do
    :type mode: str

    :return: list of frame numbers, high resolution switch, frame length, resolution and framerate
    :rtype: tuple[list[int], bool, int, int, int, int, int]
    """
    if mode == "src":
        kf_file = tmp_dir / "info_src.txt"
    else:
        kf_file = tmp_dir / "info.txt"

    if kf_file.exists() and mode == "src" and (stage != 0 or resume):
        with open(kf_file, "r") as file:
            print("Loading cached scene information...")
            lines = file.readlines()
            return [int(line.strip()) for line in lines[1:-3]], lines[0].strip() == "True", int(lines[-5].strip()) , int(lines[-4].strip()) , int(lines[-3].strip()), int(lines[-2].strip()), int(lines[-1].strip())
    try:
        if mode == "src":
            src = core.ffms2.Source(source=vfile, cachefile=f"{cache_file}")
        else:
            src = core.ffms2.Source(source=vfile, cache=False)
    except:
        console.print(f"[red]Cannot retrieve file information. Did you run the previous stages?")
        raise SystemExit(1)

    nframe = len(src)
    if mode == "len":
        return 0, 0, nframe, 0, 0, 0, 0

    fwidth, fheight = src[0].width, src[0].height
    hr = fwidth * fheight > 1920 * 1080
    with open(kf_file, "w") as file:
        file.write(str(hr)+"\n")

    # they're reversed for some reason
    ffpsnum = src.get_frame(0).props['_DurationDen']
    ffpsden = src.get_frame(0).props['_DurationNum']

    iframe_list = []

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            FPSColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:

        task = progress.add_task("[green]Finding scenes          " if ssimu2 == "" else "[green]Finding scenes                ", total=nframe)

        def progress_func(n: int, num_frames: int) -> None:
            progress.update(task, completed=n)

        def get_props(n: int, f: vs.VideoFrame) -> None:
            if f.props.get('_PictType') == 'I':
                iframe_list.append(n)

        clip_async_render(
            src, 
            outfile=None, 
            progress=progress_func,
            callback=get_props
        )

        progress.update(task, description="[cyan]Found scenes            " if ssimu2 == "" else "[cyan]Found scenes                  ", completed=nframe)

    if 'src' in locals():
        del src
        gc.collect()

    with open(kf_file, "a") as file:
        file.write("\n".join(map(str, iframe_list)))

    if verbose:
        print("I-Frames:", iframe_list)
        console.print("Total I-Frames:", len(iframe_list))

    with open(kf_file, "a") as file:
        file.write(f"\n{nframe}\n{fwidth}\n{fheight}\n{ffpsnum}\n{ffpsden}")

    return iframe_list, hr, nframe, fwidth, fheight, ffpsnum, ffpsden

def set_resuming_params(enc_file: Path, zones_file: Path, state: str) -> tuple[str, int, Path, int, int, int, int]:
    """
    Determines where to resume encoding by trimming the current encode at the last full GOP,
    summing previous trimmed chunks, creating offset zones file, and returning the skip/start options.

    :param enc_file: path to fast pass encode
    :type enc_file: Path
    :param zones_file: path to original zones file
    :type zones_file: Path
    :param state: 
    :type state: str

    :return: skip options, start options, offset zones file path, resolution and framerate
    :rtype: tuple[str, int, Path, int, int, int, int]
    """
    if not enc_file.exists():
        return "", "", zones_file, "", "", "", ""

    _, _, nframe_enc, _, _, _, _ = get_file_info(enc_file, "len")
    _, _, nframe_src, _, _, ffpsnum, ffpsden = get_file_info(src_file, "src")

    if verbose:
        console.print(f"Source: {nframe_src} frames\nEncode: {nframe_enc} frames")

    if nframe_enc > nframe_src:
        console.print(f"[red]Something wrong occurred with resume, report the issue and try re-running the {state} pass from scratch as a temporary workaround...")
        raise SystemExit(1)
    elif nframe_enc == nframe_src:
        print(f"Nothing to resume in the {state} pass. Continuing...")
        if state == "final":
            print('Stage 4 complete!')
            console.print("\n[bold]Auto-boost complete!")
            raise SystemExit(0)
        return "", "", zones_file, "", "", "", ""

    total_prev = get_total_previous_frames(enc_file)

    ranges, _, _, fwidth, fheight, _, _ = get_file_info(enc_file, "")
    last_gop_start = ranges[-1]

    if len(ranges) == 1:
        print(f"Not enough frames to resume in the {state} pass. Restarting...")
        os.remove(enc_file)
        return "", "", zones_file, "", "", "", ""

    resume_file = get_next_filename(enc_file)
    trim_ivf_from_last_keyframe(enc_file, resume_file, last_gop_start)

    total_resume_point = total_prev + last_gop_start
    print(f"Resuming the {state} pass from frame {total_resume_point}...")

    offset_zones_path = zones_file
    if state == "final" and zones_file.exists():
        offset_zones_path = get_next_filename(zones_file)
        create_offset_zones_file(zones_file, offset_zones_path, total_resume_point)

    return f"--skip {total_resume_point}", total_resume_point, offset_zones_path, fwidth, fheight, ffpsnum, ffpsden

def track_progress(vspipe_resume_value: int, svt_cmd: list[str], enc_pass: str):
    is_fast = " " if enc_pass == "fast" else ""

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        FPSColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:

        task = progress.add_task("[yellow]Initializing            " if ssimu2 == "" else "[yellow]Initializing                  ", total=None)

        try:
            vpy_vars = {}
            exec(open(vpy_file).read(), globals(), vpy_vars)
            clip_og = clip = vpy_vars["src"]
            if vspipe_resume_value:
                vspipe_resume_value_loc = int(vspipe_resume_value)
                clip = vpy_vars["src"][vspipe_resume_value_loc:]
            else:
                vspipe_resume_value_loc = 0

            svt_proc = subprocess.Popen(
                svt_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            progress.update(
                task,
                description=f"[green]Encoding {enc_pass} pass     {is_fast}" if ssimu2 == "" else f"[green]Encoding {enc_pass} pass           {is_fast}",
                completed=vspipe_resume_value_loc,
                total=clip_og.num_frames
            )

            def prog_func(current_frame, total_frames):
                progress.update(task, completed=vspipe_resume_value_loc+current_frame)

            clip.output(svt_proc.stdin, y4m=True, progress_update=prog_func)

            progress.update(
                task,
                description=f"[green]Finalizing {enc_pass} pass   {is_fast}" if ssimu2 == "" else f"[green]Finalizing {enc_pass} pass         {is_fast}",
                completed=vspipe_resume_value_loc + progress.tasks[task].total - 1
            )

            svt_proc.communicate()

            if svt_proc.returncode != 0:
                progress.stop()
                console.print(f"[red]The {enc_pass} pass encountered an error:[/red] SVT-AV1 exited with code {svt_proc.returncode}")
                raise SystemExit(1)

            progress.update(
                task,
                description=f"[cyan]Completed {enc_pass} pass    {is_fast}" if ssimu2 == "" else f"[cyan]Completed {enc_pass} pass          {is_fast}",
                completed=vspipe_resume_value_loc + progress.tasks[task].total
            )

        except KeyboardInterrupt:
            progress.stop()
            console.print("\n[yellow]Interrupted by user (Ctrl+C). Stopping...[/yellow]")
            svt_proc.terminate()
            raise SystemExit(1)
        except subprocess.CalledProcessError as e:
            progress.stop()
            console.print(f"[red]The {enc_pass} pass encountered an error:[/red]\n{e}")
            raise SystemExit(1)
        except Exception as e:
            progress.stop()
            console.print(f"[red]The {enc_pass} pass encountered an error:[/red]\n{e}")
            raise SystemExit(1)

def fast_pass() -> None:
    """
    Quick fast pass to gather scene complexity information.
    """
    encoder_params = f''
    if not isinstance(fast_speed, int):
        encoder_params += f'--speed {fast_speed} '
    if not isinstance(quality, (int, float)):
        encoder_params += f'--quality {quality} '
    if fast_params:
        encoder_params += f'{fast_params}'

    if verbose:
        console.print(f'Fast params: "{encoder_params}"')

    encoder_params_list = encoder_params.split()

    svt_resume_list = ""
    vspipe_resume_value = ""
    if resume:
        svt_resume_string, vspipe_resume_value, _, fwidth, fheight, ffpsnum, ffpsden = set_resuming_params(fast_output_file, "", "fast")
        svt_resume_list = svt_resume_string.split()

    if file_ext in [".y4m", ".yuv"]:

        fast_pass_command_y4m = [
            'SvtAv1EncApp',
            '-i', src_file,
            *svt_resume_list,
            '--progress', '2',
            *encoder_params_list,
            '-b', fast_output_file
        ]

        try:
            subprocess.run(fast_pass_command_y4m, text=True, check=True)

        except subprocess.CalledProcessError as e:
            console.print(f"[red]The fast pass encountered an error:\n{e}\nDid you make sure the source is 10-bit?")
            raise SystemExit(1)

    else:

        fast_svt_cmd = [
            'SvtAv1EncApp',
            '-i', '-',
            '--progress', '0',
            *encoder_params_list,
            '-b', fast_output_file
            ]

        track_progress(vspipe_resume_value, fast_svt_cmd, "fast")

    resume_file = tmp_dir / f"{fast_output_file.stem}__1.ivf"
    if resume and resume_file.exists():
        merge_ivf_parts(resume_file, fast_output_file, fwidth, fheight, ffpsnum, ffpsden)

def final_pass() -> None:
    """
    Final encoding pass with proper zone offsetting for resume functionality.
    """
    encoder_params = f''
    if not isinstance(final_speed, int):
        encoder_params += f'--speed {final_speed} '
    if not isinstance(quality, (int, float)):
        encoder_params += f'--quality {quality} '
    if final_params:
        encoder_params += f'{final_params}'

    if verbose:
        console.print(f'Final params: "{encoder_params}"')

    encoder_params_list = encoder_params.split()

    svt_resume_list = ""
    vspipe_resume_value = ""
    active_zones_path = zones_file
    if resume:
        svt_resume_string, vspipe_resume_value, active_zones_path, fwidth, fheight, ffpsnum, ffpsden = set_resuming_params(tmp_final_output_file, zones_file, "final")
        svt_resume_list = svt_resume_string.split()

    if file_ext in [".y4m", ".yuv"]:

        final_pass_command_y4m = [
            'SvtAv1EncApp',
            '-i', src_file,
            *svt_resume_list,
            '--progress', '2',
            *encoder_params_list
        ]

        if not no_boosting:
            final_pass_command_y4m.extend(['--config', str(active_zones_path)])

        final_pass_command_y4m.extend(['-b', tmp_final_output_file])

        try:
            subprocess.run(final_pass_command_y4m, text=True, check=True)

        except subprocess.CalledProcessError as e:
            console.print(f"[red]The final pass encountered an error:\n{e}\nDid you make sure the source is 10-bit?")
            raise SystemExit(1)

    else:

        final_svt_cmd = [
            'SvtAv1EncApp',
            '-i', '-',
            '--progress', '0',
            *encoder_params_list
            ]

        if not no_boosting:
            final_svt_cmd.extend(['--config', str(active_zones_path)])

        final_svt_cmd.extend(['-b', tmp_final_output_file])

        track_progress(vspipe_resume_value, final_svt_cmd, "final")

    resume_file = tmp_dir / f"{tmp_final_output_file.stem}__1.ivf"
    if resume and resume_file.exists():
        merge_ivf_parts(resume_file, tmp_final_output_file, fwidth, fheight, ffpsnum, ffpsden)

def calculate_metric() -> None:
    """
    Calculate SSIMULACRA2 or XPSNR metrics score.
    """
    try:
        source_clip = core.ffms2.Source(source=src_file, cachefile=f"{cache_file}")
    except:
        console.print("[red]Error indexing source file. Is it corrupted?")
        raise SystemExit(1)
    try:
        encoded_clip = core.ffms2.Source(source=fast_output_file, cache=False)
    except:
        console.print("[red]Error indexing fast pass file. Did you run stage 1?")
        raise SystemExit(1)

    if len(source_clip) != len(encoded_clip):
        console.print("[red]Source frame count and encode frame count are different. Did you successfully run stage 1?")
        raise SystemExit(1)

    if verbose:
        console.print(f"Source: {len(source_clip)} frames\nEncode: {len(encoded_clip)} frames")

    skip = 3
    if skip > 1:
        cut_source_clip = source_clip[::skip]
        cut_encoded_clip = encoded_clip[::skip]
    else:
        cut_source_clip = source_clip
        cut_encoded_clip = encoded_clip

    global ssimu2
    if ssimu2 == "":
        try:
            result = core.vszip.XPSNR(cut_source_clip, cut_encoded_clip, temporal=False, verbose=False)
        except:
            console.print(f"[red]vs-zip not found. Check your installation.")
            raise SystemExit(1)
    elif ssimu2 == "gpu":
        try:
            result = core.vship.SSIMULACRA2(cut_source_clip, cut_encoded_clip, numStream = 2)
        except:
            console.print(f"[red]Vship not found. Check your installation.")
            raise SystemExit(1)
    elif ssimu2 == "cpu":
        try:
            result = core.vszip.SSIMULACRA2(cut_source_clip, cut_encoded_clip)
        except:
            console.print(f"[red]vs-zip not found. Check your installation.")
            raise SystemExit(1)
    else:
        try:
            result = core.vship.SSIMULACRA2(cut_source_clip, cut_encoded_clip, numStream = 2)
            ssimu2 = "gpu"
        except:
            console.print(f"[yellow]Vship not found or available, defaulting to vs-zip.")
            try:
                result = core.vszip.SSIMULACRA2(cut_source_clip, cut_encoded_clip)
                ssimu2 = "cpu"
            except:
                console.print(f"[red]vs-zip not found either. Check your installation.")
                raise SystemExit(1)

    if ssimu2 == "":
        score_list = [[None] * cut_source_clip.num_frames for _ in range(3)]
    else:
        score_list = [None] * result.num_frames

    def get_xpsnrprops(n: int, f: vs.VideoFrame) -> None:
        for i, plane in enumerate(["Y", "U", "V"]):
            if str(f.props.get(f"XPSNR_{plane}")) == "inf":
                score_list[i][n] = "100.0"
            else:
                score_list[i][n] = float(f.props.get(f"XPSNR_{plane}"))

    def get_ssimu2props(n: int, f: vs.VideoFrame) -> None:
        score_list[n] = float(f.props.get('_SSIMULACRA2')) if ssimu2 == "gpu" else float(f.props.get('SSIMULACRA2'))

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            FPSColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:

        task = progress.add_task("[green]Calculating XPSNR scores" if ssimu2 == "" else "[green]Calculating SSIMULACRA2 scores", total=cut_source_clip.num_frames*skip)

        def progress_func(n: int, num_frames: int) -> None:
            progress.update(task, advance=skip)

        clip_async_render(
            result,
            outfile=None,
            progress=progress_func,
            callback=get_xpsnrprops if ssimu2 == "" else get_ssimu2props
        )

        progress.update(task, description="[cyan]Calculated XPSNR scores " if ssimu2 == "" else "[cyan]Calculated SSIMULACRA2 scores ", completed=cut_source_clip.num_frames*skip)

    if 'source_clip' in locals() and 'encoded_clip' in locals():
        del source_clip
        del encoded_clip
        gc.collect()

    skip_offset = 0
    if ssimu2 == "":
        with open(xpsnr_log_file, "w") as file:
            for index in range(len(score_list[0])):
                for i in range(skip):
                    file.write(f"{index+skip_offset+i}: {score_list[0][index]} {score_list[1][index]} {score_list[2][index]}\n")
                skip_offset += skip - 1
    else:
        with open(ssimu2_log_file, "w") as file:
            for index, score in enumerate(score_list):
                for i in range(skip):
                    file.write(f"{index+skip_offset+i}: {score}\n")
                skip_offset += skip - 1

def metrics_aggregation(score_list: list[float]) -> tuple[float, float]:
    """
    Takes a list of metrics scores and aggregatates them into the desired formats.

    :param score_list: list of metrics scores
    :type score_list: list[float]

    :return: average and 15th percentile scores
    :rtype: tuple[float, float]
    """
    filtered_score_list = [score if score >= 0 else 0.0 for score in score_list]
    sorted_score_list = sorted(filtered_score_list)
    average = sum(filtered_score_list)/len(filtered_score_list)
    percentile_15 = quantiles(sorted_score_list, n=100)[14]
    min_score = sorted_score_list[0]
    return (average, percentile_15, min_score)

def calculate_zones(ranges: list[float], hr: bool, nframe: int) -> None:
    """
    Retrieves SSIMULACRA2 or XPSNR scores, runs metrics aggregation and make CRF adjustement decisions.

    :param ranges: scene changes list
    :type ranges: list
    :param hr: switch for high resolution sources
    :type hr: bool
    :param nframe: source frame amount
    :type nframe: int

    :return: string containing zones information
    :rtype: str
    """
    metric_scores: list[int] = []

    if not xpsnr_log_file.exists() and ssimu2 == "":
        console.print("[red]Cannot find the metrics file. Did you run the previous stages?")
        raise SystemExit(1)
    elif not ssimu2_log_file.exists() and ssimu2 != "":
        console.print("[red]Cannot find the metrics file. Did you run the previous stages?")
        raise SystemExit(1)

    if ssimu2 == "":
        with open(xpsnr_log_file, "r") as file:
            for line in file:
                match = re.search(r"([0-9]+): ([0-9]+\.[0-9]+) ([0-9]+\.[0-9]+) ([0-9]+\.[0-9]+)", line)
                if match:
                    score_y = float(match.group(2))
                    score_u = float(match.group(3))
                    score_v = float(match.group(4))

                    maxval: int = 255
                    xpsnr_mse_y: float = (maxval**2) / (10 ** (score_y / 10)) # PSNR to MSE
                    xpsnr_mse_u: float = (maxval**2) / (10 ** (score_u / 10)) # PSNR to MSE
                    xpsnr_mse_v: float = (maxval**2) / (10 ** (score_v / 10)) # PSNR to MSE
                    w_xpsnr_mse: float = ((4.0 * xpsnr_mse_y) + xpsnr_mse_u + xpsnr_mse_v) / 6.0
                    score_weighted = 10.0 * log10((maxval**2) / w_xpsnr_mse)

                    metric_scores.append(score_weighted)
                else:
                    console.print("[red]Unexpected error with metric log file.\nTry re-running stage 2. Exiting.")
                    raise SystemExit(1)
    else:
        with open(ssimu2_log_file, "r") as file:
            for line in file:
                match = re.search(r"([0-9]+): (-?[0-9]+\.[0-9]+)", line)
                if match:
                    score = float(match.group(2))
                    metric_scores.append(score)
                else:
                    console.print("[red]Unexpected error with metric log file.\nTry re-running stage 2. Exiting.")
                    raise SystemExit(1)

    metric_total_scores = []
    metric_percentile_15_total = []
    metric_min_total = []

    for index in range(len(ranges)):
        metric_chunk_scores = []
        if index == len(ranges)-1:
            metric_frames = nframe - ranges[index]
        else:
            metric_frames = ranges[index+1] - ranges[index]
        for scene_index in range(metric_frames):
            metric_score = metric_scores[ranges[index]+scene_index]
            metric_chunk_scores.append(metric_score)
            metric_total_scores.append(metric_score)
        (metric_average, metric_percentile_15, metric_min) = metrics_aggregation(metric_chunk_scores)
        metric_percentile_15_total.append(metric_percentile_15)
        metric_min_total.append(metric_min)
    (metric_average, metric_percentile_15, metric_min) = metrics_aggregation(metric_total_scores)

    if verbose:
        index_min = min(range(len(metric_scores)), key=metric_scores.__getitem__)
        metric = "XPSNR" if ssimu2 == "" else "SSIMULACRA2"
        console.print(f'{metric}:\n'
                      f'Mean score: {metric_average:.4f}\n'
                      f'15th percentile: {metric_percentile_15:.4f}\n'
                      f'Worst scoring frame: {index_min} ({metric_scores[index_min]:.4f})')

    match quality:
        case "low":
            crf = 40 if hr else 35
        case "medium":
            crf = 35 if hr else 30
        case "high":
            crf = 30 if hr else 25
        case _:
            crf = quality

    with open(zones_file, "w") as file:
        for index in range(len(ranges)):
            
            # Calculate CRF adjustment using aggressive or normal multiplier
            multiplier = 40 if aggressive else 20
            adjustment = ceil((1.0 - (metric_percentile_15_total[index] / metric_average)) * multiplier * 4) / 4
            new_crf = crf - adjustment

            # Apply sane limits
            limit = 10 if unshackle else 5
            if adjustment < - limit: # Positive deviation (increasing CRF)
                new_crf = crf + limit
            elif adjustment > limit: # Negative deviation (decreasing CRF)
                new_crf = crf - limit

            if index == len(ranges)-1:
                end_range = nframe
            else:
                end_range = ranges[index+1]

            if verbose:
                console.print(f'Chunk: [{ranges[index]}:{end_range}] / 15th percentile: {metric_percentile_15_total[index]:.4f} / CRF adjustment: {-adjustment} / Final CRF: {new_crf}')

            if index == 0:
                file.write(f"Zones : {ranges[index]},{end_range-1},{new_crf};")
            else:
                file.write(f"{ranges[index]},{end_range-1},{new_crf};")
    
    console.print("[cyan]Successfully computed zones.")

console.print("[bold]Auto-boost start!\n")

if no_boosting:
    stage = 4

match stage:
    case 0:
        if stage_resume < 2:
            fast_pass()
            with open(stage_file, "w") as file:
                file.write("2")
            print('Stage 1 complete!')
        if stage_resume < 3:
            try:
                calculate_metric()
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted by user (Ctrl+C). Stopping...[/yellow]")
                raise SystemExit(1)
            with open(stage_file, "w") as file:
                file.write("3")
            print('Stage 2 complete!')
        if stage_resume < 4:
            try:
                ranges, hr, nframe, _, _, _, _ = get_file_info(fast_output_file, "")
                calculate_zones(ranges, hr, nframe)
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted by user (Ctrl+C). Stopping...[/yellow]")
                raise SystemExit(1)
            with open(stage_file, "w") as file:
                file.write("4")
            print('Stage 3 complete!')
        if stage_resume < 5:
            final_pass()
            shutil.move(tmp_final_output_file, final_output_file)
            with open(stage_file, "w") as file:
                file.write("5")
            print('Stage 4 complete!')
    case 1:
        fast_pass()
        with open(stage_file, "w") as file:
            file.write("2")
        print('Stage 1 complete!')
    case 2:
        try:
            calculate_metric()
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user (Ctrl+C). Stopping...[/yellow]")
            raise SystemExit(1)
        with open(stage_file, "w") as file:
            file.write("3")
        print('Stage 2 complete!')
    case 3:
        try:
            ranges, hr, nframe, _, _, _, _ = get_file_info(fast_output_file, "")
            calculate_zones(ranges, hr, nframe)
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user (Ctrl+C). Stopping...[/yellow]")
            raise SystemExit(1)
        with open(stage_file, "w") as file:
            file.write("4")
        print('Stage 3 complete!')
    case 4:
        final_pass()
        shutil.move(tmp_final_output_file, final_output_file)
        with open(stage_file, "w") as file:
            file.write("5")
        if not no_boosting:
            print('Stage 4 complete!')
    case _:
        console.print("[red]Stage argument invalid, exiting.")
        raise SystemExit(1)

console.print("\n[bold]Auto-boost complete!")
