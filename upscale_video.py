import subprocess
import os
import re
import sys
import time
import math

def get_video_duration(file_path: str) -> float | None:
    """
    Gets the duration of a video file using ffprobe.

    Args:
        file_path (str): Path to the video file.

    Returns:
        float | None: Duration of the video in seconds, or None if an error occurs.
    """
    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        duration_str = result.stdout.strip()
        if duration_str:
            return float(duration_str)
        else:
            print(f"Lỗi: ffprobe không trả về thời lượng cho file: {file_path}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Lỗi khi lấy thời lượng video bằng ffprobe cho file {file_path}: {e.stderr.strip()}")
        return None
    except FileNotFoundError:
        print("Lỗi: ffprobe không được tìm thấy. Hãy đảm bảo FFmpeg (bao gồm ffprobe) đã được cài đặt và nằm trong PATH của hệ thống.")
        return None
    except ValueError:
        print(f"Lỗi: Không thể phân tích thời lượng video từ output của ffprobe cho file: {file_path}")
        return None
    except Exception as e:
        print(f"Đã xảy ra lỗi không mong muốn khi lấy thời lượng video: {e}")
        return None

def get_human_readable_size(size_bytes: int) -> str:
    """Converts a size in bytes to a human-readable string (KB, MB, GB)."""
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

def upscale_video(input_path: str, output_path: str, resolution: str, crf: int = 23, preset: str = "medium", scaler: str = "bicubic"):
    """
    Upscales a video to the specified resolution using FFmpeg, with progress display.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the upscaled video file.
        resolution (str): Target resolution in WxH format (e.g., "1280x720", "3840x2160", "7680x4320").
        crf (int): Constant Rate Factor for H.264 encoding (0-51, lower is better quality, 18-28 is a good range).
                   Default is 23.
        preset (str): FFmpeg encoding speed preset (e.g., "ultrafast", "fast", "medium", "slow", "veryslow").
                      Slower presets provide better compression. Default is "medium".
        scaler (str): FFmpeg scaling algorithm (e.g., "bicubic", "lanczos", "spline").
                      Default is "bicubic". "lanczos" can sometimes offer sharper results.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    if not os.path.exists(input_path):
        print(f"Lỗi: Không tìm thấy file đầu vào: {input_path}")
        return False

    original_size_bytes = os.path.getsize(input_path)
    print(f"Dung lượng file gốc: {get_human_readable_size(original_size_bytes)}")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Đã tạo thư mục đầu ra: {output_dir}")

    total_duration = get_video_duration(input_path)
    can_show_progress = total_duration is not None and total_duration > 0

    if not can_show_progress:
        if total_duration is None:
            print("Không thể lấy thời lượng video. Sẽ upscale mà không hiển thị tiến độ chi tiết.")
        elif total_duration <= 0:
            print("Thời lượng video không hợp lệ (0 hoặc âm). Sẽ upscale mà không hiển thị tiến độ chi tiết.")

    # Construct the FFmpeg command
    # -vf scale=WxH:flags=lanczos (example of choosing a specific scaler flag)
    # Using -sws_flags for more control over scaler choice if needed.
    command = [
        "ffmpeg",
        "-i", input_path,
        "-vf", f"scale={resolution}:flags={scaler}", # Video filter for scaling
        "-c:v", "libx264",    # Video codec (H.264 is widely compatible)
        "-crf", str(crf),     # Constant Rate Factor for quality
        "-preset", preset,    # Encoding speed/compression preset
        "-c:a", "aac",        # Audio codec (AAC is widely compatible)
        "-b:a", "192k",       # Audio bitrate
        output_path,
        "-y"                  # Overwrite output file if it exists
    ]

    print(f"\nĐang thực thi lệnh FFmpeg:")
    print(" ".join(command))

    start_time = time.time()
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

        if can_show_progress:
            sys.stdout.write(f"Tiến độ: 0.00%")
            sys.stdout.flush()

        stderr_lines = []
        if process.stderr:
            for line in iter(process.stderr.readline, ''):
                stderr_lines.append(line)
                if can_show_progress:
                    # Example FFmpeg time format: time=00:00:10.54
                    match = re.search(r"time=(\d{2}):(\d{2}):(\d{2})\.(\d{2})", line)
                    if match:
                        hours = int(match.group(1))
                        minutes = int(match.group(2))
                        seconds = int(match.group(3))
                        hundredths = int(match.group(4))
                        current_time_seconds = hours * 3600 + minutes * 60 + seconds + hundredths / 100.0
                        
                        progress_percent = (current_time_seconds / total_duration) * 100
                        progress_percent = min(progress_percent, 100.0) # Cap at 100%
                        
                        sys.stdout.write(f"\rTiến độ: {progress_percent:.2f}%")
                        sys.stdout.flush()
        
        # Wait for process to terminate and get remaining outputs
        stdout_data, stderr_data_remaining = process.communicate()
        
        full_stderr_output = "".join(stderr_lines)
        if stderr_data_remaining:
            full_stderr_output += stderr_data_remaining

        if can_show_progress:
            sys.stdout.write("\n") # Move to the next line after progress is done
            sys.stdout.flush()

        if process.returncode == 0:
            end_time = time.time()
            processing_time = end_time - start_time
            output_size_bytes = os.path.getsize(output_path)
            print(f"\nThành công! Video đã được upscale và lưu tại: {output_path}")
            print(f"Dung lượng file mới: {get_human_readable_size(output_size_bytes)}")
            print(f"Thời gian xử lý: {time.strftime('%H:%M:%S', time.gmtime(processing_time))} (HH:MM:SS)")
            # if stdout_data: # ffmpeg usually doesn't output much to stdout for this command
            return True
        else:
            end_time = time.time()
            processing_time = end_time - start_time
            print("\nLỗi trong quá trình xử lý FFmpeg:")
            print(f"Thời gian xử lý trước khi lỗi: {time.strftime('%H:%M:%S', time.gmtime(processing_time))} (HH:MM:SS)")
            if stdout_data:
                 print("--- STDOUT ---")
                 print(stdout_data.strip())
            print("--- STDERR ---")
            print(full_stderr_output.strip())
            return False
    except FileNotFoundError:
        print("Lỗi: FFmpeg không được tìm thấy. Hãy đảm bảo nó đã được cài đặt và nằm trong PATH của hệ thống.")
        if can_show_progress: sys.stdout.write("\n") # Ensure newline if progress was being shown
        return False
    except Exception as e:
        print(f"Đã xảy ra lỗi không mong muốn: {e}")
        if can_show_progress: sys.stdout.write("\n") # Ensure newline
        return False

if __name__ == "__main__":
    INPUT_VIDEO = "video_data/700064993-1-208.mp4"  # THAY ĐỔI ĐƯỜNG DẪN NÀY
    OUTPUT_VIDEO_DIR = "output_videos"           # Thư mục lưu video đã upscale

    # --- Chọn độ phân giải mong muốn ---
    # target_resolution = "1280x720"  # HD
    target_resolution = "3840x2160" # 4K
    # target_resolution = "7680x4320" # 8K
    # ------------------------------------

    # --- Tùy chọn nâng cao (có thể giữ mặc định) ---
    quality_crf = 20  # Chất lượng video (0-51, thấp hơn = tốt hơn, 18-24 thường tốt)
    speed_preset = "medium" # Tốc độ mã hóa (ultrafast, fast, medium, slow, veryslow)
    scaling_algorithm = "lanczos" # Thuật toán upscale (bicubic, lanczos, spline, etc.)
    # -----------------------------------------------

    if INPUT_VIDEO == "path/to/your/input_video.mp4" or not os.path.exists(INPUT_VIDEO):
        print("Vui lòng cập nhật biến INPUT_VIDEO trong script với đường dẫn tới file video của bạn.")
    else:
        base_name, ext = os.path.splitext(os.path.basename(INPUT_VIDEO))
        output_filename = f"{base_name}_upscaled_{target_resolution.split('x')[0]}p{ext}"
        full_output_path = os.path.join(OUTPUT_VIDEO_DIR, output_filename)
        
        print(f"Chuẩn bị upscale video: {INPUT_VIDEO}")
        print(f"Độ phân giải mục tiêu: {target_resolution}")
        print(f"File đầu ra sẽ được lưu tại: {full_output_path}")
        
        success = upscale_video(
            input_path=INPUT_VIDEO,
            output_path=full_output_path,
            resolution=target_resolution,
            crf=quality_crf,
            preset=speed_preset,
            scaler=scaling_algorithm
        )
        if success:
            print("Hoàn thành upscale video.")
        else:
            print("Upscale video thất bại.")