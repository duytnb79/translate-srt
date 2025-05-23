import asyncio
from bilix.sites.bilibili import DownloaderBilibili
import os

# To install bilix, run: pip install bilix

async def download_bilibili_video(video_url: str, download_path: str = "."):
    """
    Downloads a Bilibili video to the specified path.

    Args:
        video_url: The URL of the Bilibili video.
        download_path: The directory where the video should be saved. Defaults to the current directory.
    """
    abs_download_path = os.path.abspath(download_path)
    print(f"Videos will be saved to: {abs_download_path}")
    
    try:
        # Ensure the bilix downloader is used within an async context
        # bilix uses the 'rich' library to display progress in the console during download.
        # This may include download percentage, speed, and size.
        # FFmpeg, when merging video/audio, may also print duration and other details.
        async with DownloaderBilibili(videos_dir=download_path) as d:
            print(f"Starting download for: {video_url}")
            # The get_video method handles fetching and saving the video.
            # It may download video and audio separately and then merge them.
            # It also handles different video parts if a video is a collection.
            await d.get_video(video_url)
            print(f"Video downloaded successfully to {abs_download_path}!")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure the video URL is correct and accessible.")
        print("For some videos, especially VIP-only or regionally restricted ones,")
        print("you might need to configure cookies. Check the bilix documentation for more details:")
        print("https://hfrost0.github.io/bilix/sites/bilibili/#%E4%B8%8B%E8%BD%BD%E9%85%8D%E7%BD%AE")

async def main():
    # Replace with the Bilibili video URL you want to download
    # Example: video_url = "https://www.bilibili.com/video/BV1GJ411x7h7/"
    video_url = input("Enter the Bilibili video URL: ")
    
    # Optional: Specify a different download directory.
    # If you want to save to a specific folder, uncomment and set the path below.
    # For example: download_directory = "my_bilibili_videos" 
    # By default, videos are saved in the current working directory.
    # await download_bilibili_video(video_url, download_directory)
    
    await download_bilibili_video(video_url)

if __name__ == "__main__":
    # Note: bilix uses asyncio, so we run the main function using asyncio.run()
    print("Bilibili Video Downloader using bilix")
    print("===================================")
    print("Make sure you have FFmpeg installed and in your system's PATH for merging video and audio files.")
    print("You can download FFmpeg from: https://ffmpeg.org/download.html")
    print("During the download, bilix will show progress. FFmpeg might show video duration/size when merging.")
    print("\\n")
    
    asyncio.run(main()) 