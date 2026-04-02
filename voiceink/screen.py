"""Screen OCR and NER entity extraction."""

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

from voiceink.config import CONFIG_DIR, DEFAULTS, log
from voiceink.utils import Timer

try:
    import Quartz
    import Vision

    _HAS_VISION = True
except ImportError:
    _HAS_VISION = False

# OCR cache
_ocr_cache_text = ""
_ocr_cache_time = 0.0
_OCR_CACHE_TTL = 5.0


def capture_all_screens():
    if not _HAS_VISION:
        return []
    try:
        from AppKit import NSScreen

        images = []
        for screen in NSScreen.screens():
            frame = screen.frame()
            rect = Quartz.CGRectMake(
                frame.origin.x, frame.origin.y,
                frame.size.width, frame.size.height,
            )
            img = Quartz.CGWindowListCreateImage(
                rect,
                Quartz.kCGWindowListOptionOnScreenOnly,
                Quartz.kCGNullWindowID,
                Quartz.kCGWindowImageDefault,
            )
            if img:
                images.append(img)
        return images
    except Exception as e:
        log.warning("Screen capture failed: %s", e)
        return []


def ocr_cgimage(cg_image):
    if cg_image is None or not _HAS_VISION:
        return ""
    try:
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
            cg_image, None
        )
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelFast)
        request.setUsesLanguageCorrection_(True)
        request.setRecognitionLanguages_(DEFAULTS["ocr_languages"])
        success, error = handler.performRequests_error_([request], None)
        if not success:
            return ""
        lines = []
        for obs in request.results():
            candidates = obs.topCandidates_(1)
            if candidates:
                lines.append(candidates[0].string())
        return "\n".join(lines)
    except Exception as e:
        log.warning("OCR failed: %s", e)
        return ""


def get_screen_text():
    global _ocr_cache_text, _ocr_cache_time
    now = time.monotonic()
    if now - _ocr_cache_time < _OCR_CACHE_TTL and _ocr_cache_text:
        log.info("Screen OCR: using cached result (%d chars)", len(_ocr_cache_text))
        return _ocr_cache_text

    images = capture_all_screens()
    if not images:
        return ""

    with ThreadPoolExecutor(max_workers=len(images)) as pool:
        parts = list(pool.map(ocr_cgimage, images))

    result = "\n".join(p for p in parts if p)
    _ocr_cache_text = result
    _ocr_cache_time = now
    return result


# NER daemon
_NER_DAEMON_PATH = str(CONFIG_DIR / "ner_daemon")
_NER_TOOL_PATH = str(CONFIG_DIR / "ner_tool")


class NERDaemon:
    """Long-running NER daemon for entity extraction."""

    def __init__(self):
        self._proc = None

    def start(self):
        try:
            self._proc = subprocess.Popen(
                [_NER_DAEMON_PATH],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            ready = self._proc.stdout.readline().strip()
            if ready == "READY":
                log.info("NER daemon started (PID %d)", self._proc.pid)
                return True
            log.warning("NER daemon unexpected output: %s", ready)
        except Exception as e:
            log.warning("NER daemon start failed: %s", e)
        self._proc = None
        return False

    def extract(self, text):
        if self._proc and self._proc.poll() is None:
            try:
                with Timer("NER extraction (daemon)"):
                    self._proc.stdin.write(text.replace("\n", " ") + "\n")
                    self._proc.stdin.flush()
                    results = []
                    while True:
                        line = self._proc.stdout.readline().strip()
                        if not line:
                            break
                        results.append(line)
                    return results
            except Exception as e:
                log.warning("NER daemon call failed: %s", e)
                self._proc = None

        # Fallback to subprocess
        try:
            with Timer("NER extraction (subprocess)"):
                result = subprocess.run(
                    [_NER_TOOL_PATH],
                    input=text,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
            if result.returncode == 0 and result.stdout.strip():
                return [w.strip() for w in result.stdout.strip().split("\n") if w.strip()]
        except Exception as e:
            log.warning("NER extraction failed: %s", e)
        return []

    def stop(self):
        if self._proc:
            try:
                self._proc.stdin.close()
                self._proc.terminate()
            except Exception:
                pass
