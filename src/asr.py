import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ASRResult:
    def __init__(self, text: str, segments: Optional[List[Dict]] = None):
        self.text = text
        self.segments = segments or []


class AbstractASR(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str) -> ASRResult:
        pass


class WhisperASR(AbstractASR):
    """Uses whisper (if installed) for offline transcription.

    This class is optional and will raise if whisper is not installed.
    """

    def __init__(self, model: str = "small"):
        try:
            import whisper
        except Exception:  # pragma: no cover - optional dependency
            raise RuntimeError("whisper is not installed; install it or use another ASR backend")
        self.whisper = whisper
        self.model_name = model
        self.model = whisper.load_model(model)

    def transcribe(self, audio_path: str) -> ASRResult:
        logger.info("Transcribing with whisper: %s", audio_path)
        result = self.model.transcribe(audio_path)
        text = result.get("text", "")
        segments = result.get("segments") or []
        return ASRResult(text=text, segments=segments)


class StubASR(AbstractASR):
    """Simple placeholder ASR for testing (returns empty transcript or file name).

    Useful when dependencies are missing.
    """

    def transcribe(self, audio_path: str) -> ASRResult:
        logger.warning("Using StubASR: returning filename as transcript.")
        text = os.path.basename(audio_path)
        return ASRResult(text=text, segments=[])


class AzureSpeechASR(AbstractASR):
    """Azure Cognitive Services Speech-to-Text adapter.

    Expects environment variables AZURE_SPEECH_KEY and AZURE_SPEECH_REGION
    or accepts them via constructor arguments.
    """

    def __init__(self, subscription_key: str | None = None, region: str | None = None):
        try:
            import azure.cognitiveservices.speech as speechsdk
        except Exception:  # pragma: no cover - optional dependency
            raise RuntimeError("azure.cognitiveservices.speech is not installed")

        self.speechsdk = speechsdk
        self.key = subscription_key or os.environ.get("AZURE_SPEECH_KEY")
        self.region = region or os.environ.get("AZURE_SPEECH_REGION")
        if not self.key or not self.region:
            raise RuntimeError(
                "Azure Speech subscription key and region are required. "
                "Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION or pass them in."
            )

        self.speech_config = speechsdk.SpeechConfig(subscription=self.key, region=self.region)

    def transcribe(self, audio_path: str) -> ASRResult:
        speechsdk = self.speechsdk
        audio_input = speechsdk.AudioConfig(filename=audio_path)
        recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=audio_input)

        logger.info("AzureSpeechASR: recognizing %s", audio_path)
        result = recognizer.recognize_once()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            text = result.text or ""
            # Azure SDK does not provide word-level segments by default here
            return ASRResult(text=text, segments=[])
        elif result.reason == speechsdk.ResultReason.NoMatch:
            logger.warning("AzureSpeechASR: no speech could be recognized for %s", audio_path)
            return ASRResult(text="", segments=[])
        else:
            # Other error
            err = getattr(result, "error_details", None)
            raise RuntimeError(f"Azure Speech recognition failed: {err}")


class AzureBatchASR(AbstractASR):
    """Azure Batch Transcription adapter using the Speech-to-Text REST API.

    This adapter submits a transcription job to the v3.0 Speech-to-Text REST API
    and polls until it finishes. It requests speaker diarization and word-level
    timestamps where available.

    Usage notes / requirements:
    - Requires `requests` and (optionally) `azure-storage-blob` for uploading local
      files and generating SAS URLs. Add these to requirements.
    - Caller should provide either publicly accessible URLs (SAS URLs) for audio
      files, or provide local file paths and set `storage_connection_string` and
      `storage_container` so the class can upload files and generate SAS links.
    - Must provide AZURE_SPEECH_KEY and AZURE_SPEECH_REGION via env or constructor.
    """

    def __init__(
        self,
        subscription_key: str | None = None,
        region: str | None = None,
        storage_connection_string: str | None = None,
        storage_container: str | None = None,
        polling_interval: int = 10,
    ):
        # lazy imports
        try:
            import requests
        except Exception:  # pragma: no cover - optional dependency
            raise RuntimeError("requests package is required for AzureBatchASR")

        self.requests = requests

        # optional blob helper
        try:
            from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
        except Exception:
            BlobServiceClient = None
            generate_blob_sas = None
            BlobSasPermissions = None

        self.BlobServiceClient = BlobServiceClient
        self.generate_blob_sas = generate_blob_sas
        self.BlobSasPermissions = BlobSasPermissions

        self.key = subscription_key or __import__("os").environ.get("AZURE_SPEECH_KEY")
        self.region = region or __import__("os").environ.get("AZURE_SPEECH_REGION")
        if not self.key or not self.region:
            raise RuntimeError(
                "Azure Speech subscription key and region are required. "
                "Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION or pass them in."
            )

        self.storage_connection_string = storage_connection_string
        self.storage_container = storage_container
        self.polling_interval = polling_interval

    def _upload_and_get_sas(self, local_path: str) -> str:
        """Upload local file to the storage container and return a read SAS URL.

        Requires `azure.storage.blob` and a connection string with account key.
        """
        if not self.BlobServiceClient:
            raise RuntimeError("azure-storage-blob is required to upload local files")

        if not self.storage_connection_string or not self.storage_container:
            raise RuntimeError(
                "To upload local files provide storage_connection_string and storage_container"
            )

        from datetime import datetime, timedelta
        from azure.storage.blob import BlobServiceClient as _BlobServiceClient

        bsc = _BlobServiceClient.from_connection_string(self.storage_connection_string)
        container_client = bsc.get_container_client(self.storage_container)
        try:
            container_client.create_container()
        except Exception:
            pass

        blob_name = os.path.basename(local_path)
        blob_client = container_client.get_blob_client(blob_name)
        with open(local_path, "rb") as fh:
            blob_client.upload_blob(fh, overwrite=True)

        # parse account name and key from connection string for SAS generation
        # connection string format contains AccountName and AccountKey
        conn = self.storage_connection_string
        account_name = None
        account_key = None
        for part in conn.split(";"):
            if part.startswith("AccountName="):
                account_name = part.split("=", 1)[1]
            if part.startswith("AccountKey="):
                account_key = part.split("=", 1)[1]

        if not account_name or not account_key:
            raise RuntimeError("Connection string must contain AccountName and AccountKey to generate SAS")

        sas_token = self.generate_blob_sas(
            account_name=account_name,
            container_name=self.storage_container,
            blob_name=blob_name,
            account_key=account_key,
            permission=self.BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(days=1),
        )
        return f"{blob_client.url}?{sas_token}"

    def _parse_time(self, timestr: str) -> float:
        """Parse Azure offset/duration string to seconds.

        Supports formats like 'PT1.234S' and '00:00:01.2340000'.
        """
        if not timestr:
            return 0.0
        if timestr.startswith("PT"):
            # PT#H#M#S or PT#.#S
            import re

            m = re.match(r"PT(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?(?:(?P<s>[0-9.]+)S)?", timestr)
            if not m:
                return 0.0
            h = float(m.group("h") or 0)
            mi = float(m.group("m") or 0)
            s = float(m.group("s") or 0)
            return h * 3600 + mi * 60 + s
        if ":" in timestr:
            # 00:00:01.2340000
            parts = timestr.split(":")
            parts = [float(p) for p in parts]
            while len(parts) < 3:
                parts.insert(0, 0.0)
            h, m, s = parts
            return h * 3600 + m * 60 + s
        try:
            return float(timestr)
        except Exception:
            return 0.0

    def transcribe(self, audio_paths, locale: str = "en-US", display_name: str | None = None) -> ASRResult:
        # Accept single path or list
        if isinstance(audio_paths, (str,)):
            audio_paths = [audio_paths]

        content_urls = []
        for p in audio_paths:
            if p.startswith("http://") or p.startswith("https://"):
                content_urls.append(p)
            else:
                # local file; upload to container if configured
                if self.storage_connection_string and self.storage_container:
                    url = self._upload_and_get_sas(p)
                    content_urls.append(url)
                else:
                    raise RuntimeError(
                        "Local file provided but storage_connection_string/storage_container not set. "
                        "Provide a SAS URL or configure storage credentials to upload."
                    )

        body = {
            "contentUrls": content_urls,
            "locale": locale,
            "displayName": display_name or "VideoDubbing transcription",
            "properties": {
                "diarizationEnabled": True,
                "wordLevelTimestampsEnabled": True,
            },
        }

        endpoint = f"https://{self.region}.api.cognitive.microsoft.com/speechtotext/v3.0/transcriptions"
        headers = {"Ocp-Apim-Subscription-Key": self.key, "Content-Type": "application/json"}

        resp = self.requests.post(endpoint, json=body, headers=headers)
        if resp.status_code not in (201, 202):
            raise RuntimeError(f"Failed to create transcription job: {resp.status_code} {resp.text}")

        location = resp.headers.get("Location")
        if not location:
            # fallback: try to find self link in body
            location = resp.json().get("self")
        if not location:
            raise RuntimeError("Could not determine transcription resource location")

        # poll
        import time

        while True:
            r = self.requests.get(location, headers={"Ocp-Apim-Subscription-Key": self.key})
            if r.status_code != 200:
                raise RuntimeError(f"Failed to poll transcription job: {r.status_code} {r.text}")
            job = r.json()
            status = job.get("status")
            if status in ("Succeeded", "Failed"):
                break
            time.sleep(self.polling_interval)

        if status != "Succeeded":
            raise RuntimeError(f"Transcription job finished with status: {status}")

        # retrieve files list
        files_url = f"{location}/files"
        r = self.requests.get(files_url, headers={"Ocp-Apim-Subscription-Key": self.key})
        if r.status_code != 200:
            raise RuntimeError(f"Failed to fetch transcription files: {r.status_code} {r.text}")

        files = r.json().get("values", [])
        transcript_json_url = None
        for f in files:
            name = f.get("name", "")
            kind = f.get("kind", "")
            # the transcription result JSON often contains 'transcription' or 'recognized' or endswith .json
            links = f.get("links") or {}
            content_url = links.get("contentUrl") or f.get("contentUrl")
            if not content_url:
                continue
            if name.endswith(".json") or "transcription" in name.lower() or "recognized" in name.lower():
                transcript_json_url = content_url
                break

        if not transcript_json_url and files:
            # fallback to first file
            links = files[0].get("links") or {}
            transcript_json_url = links.get("contentUrl") or files[0].get("contentUrl")

        if not transcript_json_url:
            raise RuntimeError("Could not find a transcription result file URL")

        r = self.requests.get(transcript_json_url)
        if r.status_code != 200:
            raise RuntimeError(f"Failed to download transcription file: {r.status_code} {r.text}")

        content = r.json()

        # Try to extract recognized phrases and speaker info
        phrases = content.get("combinedRecognizedPhrases") or content.get("recognizedPhrases") or []
        segments = []
        for p in phrases:
            text = p.get("display") or p.get("lexical") or p.get("itn") or ""
            speaker = p.get("speaker") or p.get("speakerId") or p.get("channel")
            offset = self._parse_time(p.get("offset") or p.get("startTime") or "")
            duration = self._parse_time(p.get("duration") or p.get("endTime") or "")
            # some formats return endTime, so if duration seems large and endTime present, compute
            segments.append({"text": text, "speaker": speaker, "offset": offset, "duration": duration})

        full_text = " \n".join([s["text"] for s in segments])
        return ASRResult(text=full_text, segments=segments)

