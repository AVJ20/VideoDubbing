import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)

try:
    from yt_dlp import YoutubeDL
except Exception:  # pragma: no cover - optional dependency
    YoutubeDL = None


class VideoDownloader:
    """Download video from a URL (YouTube or direct) using yt-dlp.

    Simple wrapper that returns the path to downloaded file.
    """

    def __init__(self, ytdl_opts: Optional[dict] = None):
        self.ytdl_opts = ytdl_opts or {"format": "best", "outtmpl": "%(title)s.%(ext)s"}
        if YoutubeDL is None:
            logger.warning("yt-dlp is not installed. VideoDownloader will raise if used.")

    def download(self, url: str, out_dir: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        if YoutubeDL is None:
            raise RuntimeError("yt-dlp not installed. Please install the requirements to use downloader.")

        opts = dict(self.ytdl_opts)
        opts.update({"outtmpl": os.path.join(out_dir, opts.get("outtmpl"))})

        logger.info("Downloading %s to %s", url, out_dir)
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # yt-dlp may return a playlist; handle first entry
            if "entries" in info:
                entry = info["entries"][0]
            else:
                entry = info

            filename = ydl.prepare_filename(entry)
        logger.info("Downloaded to %s", filename)
        return filename
