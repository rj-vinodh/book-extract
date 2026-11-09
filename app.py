"""Book frame selector with local model extraction.

- Drag & drop one or more videos.
- Extract candidate frames (every transition, or only new books), skipping frames
  that show only the empty background.
- Click thumbnails to select / unselect them (selected ones get a green border and a tick),
  choose how many columns are shown in the scrollable gallery, then download the selection as a ZIP.
- Send selected frames to a local OpenAI-compatible model to extract book title/author and
  export the results as a table with image data embedded in base64.
"""

import base64
import html
import json
import os
import re
import tempfile
import time
import zipfile

import cv2
import gradio as gr
import numpy as np
import pandas as pd
from openai import OpenAI

SMALL_W = 320       # width used for comparisons
THUMB_W = 800       # width of the thumbnails shown in the galleries (ZIP keeps full resolution)
VIDEO_TYPES = [".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm", ".mpg", ".mpeg", ".3gp"]

MODE_ORIGINAL = "Every transition (original)"
MODE_NEW_BOOKS = "New books only (one sharp frame per book)"
DEFAULT_THRESHOLD = {MODE_ORIGINAL: 15.0, MODE_NEW_BOOKS: 6.0}
BG_FIRST = "First frame of the video"
BG_MEDIAN = "Median of the whole video"


# ----------------------------- helpers -----------------------------

def _file_path(f):
    return f if isinstance(f, str) else getattr(f, "name", str(f))


def _safe_stem(path):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", os.path.splitext(os.path.basename(path))[0])[:40] or "video"


def _prep_color(frame):
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (SMALL_W, max(1, int(h * SMALL_W / w))), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small, (7, 7), 0)


def _changed_fraction(a, b, pixel_thresh):
    """Share of pixels where any colour channel differs by more than pixel_thresh."""
    return float(np.mean(cv2.absdiff(a, b).max(axis=2) > pixel_thresh))


def _sharpness(frame):
    return float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def _estimate_background(cap, total, n=60):
    """Median of frames spread over the video ~ the scene without the changing books."""
    imgs = []
    for pos in np.linspace(0, max(total - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
        ok, frame = cap.read()
        if ok:
            imgs.append(_prep_color(frame))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return np.median(np.stack(imgs), axis=0).astype(np.uint8) if len(imgs) >= 5 else None


def _first_frame_background(cap):
    """The very first frame of the video, assumed to show the empty background."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return _prep_color(frame) if ok else None


def _thumbnail(frame):
    h, w = frame.shape[:2]
    if w <= THUMB_W:
        return frame.copy()
    return cv2.resize(frame, (THUMB_W, max(1, int(h * THUMB_W / w))), interpolation=cv2.INTER_AREA)


def _mark_selected(thumb):
    """Copy of a thumbnail with a gr en border and a tick badge."""
    img = thumb.copy()
    h, w = img.shape[:2]
    green, border = (60, 190, 60), max(6, w // 55)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), green, border * 2)
    r = max(20, w // 16)
    cx, cy = border * 2 + r, border * 2 + r
    cv2.circle(img, (cx, cy), r, green, -1)
    tick = np.array([[cx - 0.50 * r, cy + 0.05 * r], [cx - 0.12 * r, cy + 0.42 * r],
                     [cx + 0.55 * r, cy - 0.38 * r]], dtype=np.int32)
    cv2.polylines(img, [tick], False, (255, 255, 255), max(3, r // 4), cv2.LINE_AA)
    return img


# ----------------------------- detection modes -----------------------------

def detect_every_transition(cap, total, step, diff_threshold, is_empty, save, tick):
    """Original behaviour: save a frame whenever it differs from the previous sample
    (except frames that only show the empty background)."""
    prev_gray, idx, fps = None, 0, cap.get(cv2.CAP_PROP_FPS) or 25.0
    while cap.grab():
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            small = cv2.resize(frame, (320, 240))
            gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (21, 21), 0)
            if prev_gray is None:
                is_transition = True
            else:
                delta = cv2.absdiff(prev_gray, gray)
                thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
                is_transition = (np.sum(thresh == 255) / thresh.size) * 100 > diff_threshold
            if is_transition and not is_empty(_prep_color(frame)):
                save(frame, idx, idx / fps)
            prev_gray = gray
            tick(idx / total)
        idx += 1


def detect_new_books(cap, total, step, still_secs, change_pct, motion_tol, is_empty, save, tick):
    """Save one sharp frame per book: wait for a still period, then keep it only if it
    differs by at least change_pct % of the pixels from the last saved frame (and is not
    just the empty background)."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    change_thresh = change_pct / 100.0
    min_still = max(2, int(np.ceil(still_secs * fps / step)))  # at least 2 similar samples
    pixel_thresh = 30

    last = {"gray": None}
    st = {"ref": None, "still": 0, "best": None}

    def finalize():
        best, still = st["best"], st["still"]
        st["ref"], st["still"], st["best"] = None, 0, None
        if best is None or still < min_still:
            return
        if is_empty(best["gray"]):
            return
        if last["gray"] is not None and _changed_fraction(best["gray"], last["gray"], pixel_thresh) < change_thresh:
            return
        last["gray"] = best["gray"]
        save(best["frame"], best["idx"], best["idx"] / fps)

    idx = 0
    while cap.grab():
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            gray = _prep_color(frame)
            if st["ref"] is not None and _changed_fraction(gray, st["ref"], pixel_thresh) > motion_tol:
                finalize()
            if st["ref"] is None:
                st["ref"] = gray
            st["still"] += 1
            s = _sharpness(frame)
            if st["best"] is None or s > st["best"]["sharp"]:
                st["best"] = {"sharp": s, "frame": frame.copy(), "gray": gray, "idx": idx}
            tick(idx / total)
        idx += 1
    finalize()


# ----------------------------- gallery rendering -----------------------------
# State used by the callbacks:
#   items    : list of {"name", "full", "thumb", "thumb_sel"} for every extracted frame
#   selected : sorted list of indices into items


def _render(items, selected, cols):
    """Render the full thumbnails list as a scrollable gallery."""
    cols = max(1, int(cols))
    chosen = set(selected)
    page = [(items[i]["thumb_sel"] if i in chosen else items[i]["thumb"],
             ("✓ " if i in chosen else "") + items[i]["name"]) for i in range(len(items))]
    sel_page = [(items[i]["thumb"], items[i]["name"]) for i in selected]

    page_text = "Scroll to browse all extracted frames" if items else "No frames yet"
    count_text = f"**{len(selected)}** of {len(items)} frames selected"
    return (gr.Gallery(value=page, columns=cols, height="auto"),
            gr.Gallery(value=sel_page, columns=max(1, cols), height="auto"),
            page_text, count_text)


def toggle_selection(items, selected, cols, evt: gr.SelectData):
    """Click on a thumbnail: select it, or unselect it if it already is."""
    i = evt.index
    if 0 <= i < len(items):
        chosen = set(selected)
        chosen.symmetric_difference_update({i})
        selected = sorted(chosen)
    return (selected,) + _render(items, selected, cols)


def remove_from_selection(items, selected, cols, evt: gr.SelectData):
    """Click on a frame in the 'Selected frames' gallery: unselect it."""
    if 0 <= evt.index < len(selected):
        selected = [i for k, i in enumerate(selected) if k != evt.index]
    return (selected,) + _render(items, selected, cols)


def select_all(items, cols):
    selected = list(range(len(items)))
    return (selected,) + _render(items, selected, cols)


def clear_selection(items, cols):
    return ([],) + _render(items, [], cols)


def change_layout(items, selected, cols):
    """Only the column count changes; the gallery remains scrollable."""
    return _render(items, selected, cols)


def make_zip(items, selected):
    if not selected:
        raise gr.Error("Select at least one frame first (click thumbnails above).")
    zip_path = os.path.join(tempfile.mkdtemp(prefix="book_zip_"), "selected_book_frames.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:  # JPEGs are already compressed
        for i in selected:
            zf.write(items[i]["full"], os.path.basename(items[i]["full"]))
    return zip_path


def _images_from_zip(zip_path):
    if not zip_path:
        return []
    archive = zipfile.ZipFile(zip_path)
    file_names = sorted(
        name for name in archive.namelist()
        if os.path.splitext(name)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not file_names:
        archive.close()
        raise gr.Error("The ZIP does not contain image files (jpg/png/webp/bmp).")

    tmp_dir = tempfile.mkdtemp(prefix="zip_frames_")
    items = []
    for idx, name in enumerate(file_names):
        target = os.path.join(tmp_dir, os.path.basename(name))
        with archive.open(name, "r") as src, open(target, "wb") as dst:
            dst.write(src.read())
        frame = cv2.imread(target, cv2.IMREAD_COLOR)
        if frame is None:
            continue
        full = os.path.join(tmp_dir, f"zip_{idx:04d}_{os.path.basename(name)}")
        cv2.imwrite(full, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        thumb = _thumbnail(frame)
        thumb_path = os.path.join(tmp_dir, f"thumb_{idx:04d}_{os.path.basename(name)}")
        thumb_sel_path = os.path.join(tmp_dir, f"thumb_sel_{idx:04d}_{os.path.basename(name)}")
        cv2.imwrite(thumb_path, thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        cv2.imwrite(thumb_sel_path, _mark_selected(thumb), [cv2.IMWRITE_JPEG_QUALITY, 85])
        items.append({
            "name": os.path.basename(full),
            "full": full,
            "thumb": thumb_path,
            "thumb_sel": thumb_sel_path,
        })
    archive.close()
    return items


def _model_base_url():
    base = os.getenv("MODEL_BASE_URL")
    if base:
        return base.rstrip("/") + "/v1" if not base.endswith("/v1") else base.rstrip("/")
    if os.path.exists("/.dockerenv"):
        return "http://host.docker.internal:9931/v1"
    return "http://localhost:9931/v1"


def _image_to_data_url(path):
    with open(path, "rb") as fh:
        encoded = base64.b64encode(fh.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def _normalize_book_json(raw):
    data = {"title": None, "author": None, "publisher": None, "isbn": None, "notes": None}
    if isinstance(raw, dict):
        data.update({k: raw.get(k) for k in data.keys()})
        return data
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                data.update({k: payload.get(k) for k in data.keys()})
                return data
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            try:
                payload = json.loads(match.group(0))
                if isinstance(payload, dict):
                    data.update({k: payload.get(k) for k in data.keys()})
                    return data
            except json.JSONDecodeError:
                pass
    return data


def render_metadata_csv_preview(csv_file):
    if csv_file is None:
        return "<p>No CSV file selected.</p>"

    path = csv_file.name if hasattr(csv_file, "name") else str(csv_file)
    if not os.path.exists(path):
        return "<p>CSV file not found.</p>"

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - UI feedback path
        return f"<p>Could not read CSV: {html.escape(str(exc))}</p>"

    if df.empty:
        return "<p>CSV is empty.</p>"

    cols = ["image_name", "title", "author", "publisher", "isbn", "notes", "image_base64"]
    for col in cols:
        if col not in df.columns:
            return f"<p>CSV is missing the required <b>{html.escape(col)}</b> column.</p>"

    rows_html = []
    preview_rows = df.head(20)
    for _, row in preview_rows.iterrows():
        image_value = row.get("image_base64", "")
        image_html = (
            f'<img src="{html.escape(str(image_value))}" '
            'style="max-width:120px; max-height:120px; object-fit:contain; '
            'border:1px solid #d0d7de; border-radius:8px; background:#fff;" />'
            if isinstance(image_value, str) and image_value.startswith("data:image")
            else "<span>No image</span>"
        )
        fields = [
            html.escape(str(row.get("image_name", ""))),
            html.escape(str(row.get("title", ""))),
            html.escape(str(row.get("author", ""))),
            html.escape(str(row.get("publisher", ""))),
            html.escape(str(row.get("isbn", ""))),
            html.escape(str(row.get("notes", ""))),
        ]
        rows_html.append(
            "<tr>" +
            f"<td>{image_html}</td>" +
            "".join(f"<td>{value}</td>" for value in fields) +
            "</tr>"
        )

    table_html = (
        "<div style='overflow:auto; max-height:500px;'><table style='border-collapse:collapse; width:100%; font-size:12px;'>"
        "<thead><tr><th style='padding:8px; border:1px solid #ddd;'>Image</th>"
        "<th style='padding:8px; border:1px solid #ddd;'>Image name</th>"
        "<th style='padding:8px; border:1px solid #ddd;'>Title</th>"
        "<th style='padding:8px; border:1px solid #ddd;'>Author</th>"
        "<th style='padding:8px; border:1px solid #ddd;'>Publisher</th>"
        "<th style='padding:8px; border:1px solid #ddd;'>ISBN</th>"
        "<th style='padding:8px; border:1px solid #ddd;'>Notes</th></tr></thead>"
        "<tbody>" + "".join(rows_html) + "</tbody></table></div>"
    )
    return table_html


def extract_book_metadata(items, selected, progress=gr.Progress()):
    if not selected:
        raise gr.Error("Select at least one frame before extracting the book metadata.")

    model_name = os.getenv("MODEL_NAME", "ggml-org/gemma-4-12B-it-GGUF:Q8_0")
    client = OpenAI(base_url=_model_base_url(), api_key=os.getenv("OPENAI_API_KEY", "no-key"))

    rows = []
    total = len(selected)
    response_times = []
    start_time = time.time()
    cols = ["image_name", "title", "author", "publisher", "isbn", "notes", "image_base64"]

    progress(0.0, desc=f"Connecting to model: {model_name} | 0/{total} images processed | ETA: --")
    empty_df = pd.DataFrame(columns=cols)
    yield empty_df, None, f"Connecting to model: {model_name} | 0/{total} images processed | ETA: --"

    for i, idx in enumerate(selected, start=1):
        item = items[idx]
        data_url = _image_to_data_url(item["full"])
        request_start = time.time()

        progress(i / total, desc=f"Sending image {i}/{total}: {item['name']} | avg response: -- | ETA: --")
        yield pd.DataFrame(rows, columns=cols), None, f"Sending image {i}/{total}: {item['name']} | avg response: -- | ETA: --"

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Look at this book cover image and extract the book metadata. "
                                "Return ONLY valid JSON with keys: title, author, publisher, isbn, notes. "
                                "Use null for missing values. If the image is not a book cover, return nulls. "
                                "Do not add any extra commentary or Markdown fences."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "book_metadata",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "title": {"type": ["string", "null"]},
                            "author": {"type": ["string", "null"]},
                            "publisher": {"type": ["string", "null"]},
                            "isbn": {"type": ["string", "null"]},
                            "notes": {"type": ["string", "null"]},
                        },
                        "required": ["title", "author", "publisher", "isbn", "notes"],
                        "additionalProperties": False,
                    },
                },
            },
        )

        response_time = time.time() - request_start
        response_times.append(response_time)
        avg_secs = sum(response_times) / len(response_times)
        remaining = max(total - i, 0)
        eta = avg_secs * remaining
        elapsed = time.time() - start_time

        content = response.choices[0].message.content
        payload = _normalize_book_json(content)
        rows.append(
            {
                "image_name": item["name"],
                "image_base64": data_url,
                "title": payload.get("title") or "",
                "author": payload.get("author") or "",
                "publisher": payload.get("publisher") or "",
                "isbn": payload.get("isbn") or "",
                "notes": payload.get("notes") or "",
            }
        )

        partial_df = pd.DataFrame(rows, columns=cols)
        log_msg = (
            f"Received response for {item['name']} | {i}/{total} processed | "
            f"avg {avg_secs * 1000:.0f} ms/image | "
            f"elapsed {elapsed:.1f}s | ETA {eta:.1f}s"
        )
        progress(i / total, desc=log_msg)
        yield partial_df, None, log_msg

    elapsed = time.time() - start_time
    avg_secs = sum(response_times) / len(response_times) if response_times else 0.0
    final_log = (
        f"Processed {total} image(s) in {elapsed:.1f}s. "
        f"Average response time: {avg_secs * 1000:.0f} ms/image. "
        f"Estimated remaining time at this throughput: {max(0.0, avg_secs * max(total - len(response_times), 0)):.1f}s."
    )

    df = pd.DataFrame(rows, columns=cols)
    csv_path = os.path.join(tempfile.mkdtemp(prefix="book_meta_"), "book_metadata.csv")
    df.to_csv(csv_path, index=False)
    yield df, csv_path, final_log


# ----------------------------- extraction -----------------------------

def default_threshold_for(mode):
    """Suggest a sensible threshold when the extraction mode changes."""
    return gr.Slider(value=DEFAULT_THRESHOLD[mode])


def run_extraction(files, zip_input, mode, diff_threshold, frame_step,
                   skip_empty, bg_source, bg_tol,
                   still_secs, motion_tol, cols,
                   progress=gr.Progress()):
    items = []
    problems = []
    skipped = [0]

    if zip_input:
        try:
            items = _images_from_zip(zip_input.name if hasattr(zip_input, "name") else zip_input)
            msg = f"Loaded {len(items)} images from the ZIP archive."
            selected = []
            return (items, selected) + _render(items, selected, cols) + (msg, None)
        except Exception as exc:  # pragma: no cover - UI feedback path
            raise gr.Error(f"Could not read the ZIP input: {exc}") from exc

    if not files:
        raise gr.Error("Drop at least one video or a ZIP of images into the upload box first.")

    run_dir = tempfile.mkdtemp(prefix="book_frames_")
    thumb_dir = os.path.join(run_dir, "thumbs")
    thumb_sel_dir = os.path.join(run_dir, "thumbs_selected")
    os.makedirs(thumb_dir)
    os.makedirs(thumb_sel_dir)

    for v_i, f in enumerate(files):
        vpath = _file_path(f)
        name = os.path.basename(vpath)
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            problems.append(f"could not open {name}")
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        stem = f"v{v_i + 1}_{_safe_stem(vpath)}"

        def save(frame, idx, ts, stem=stem):
            fname = f"{stem}_t{ts:07.2f}s_f{idx:06d}.jpg"
            full = os.path.join(run_dir, fname)
            cv2.imwrite(full, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            thumb = _thumbnail(frame)
            thumb_path = os.path.join(thumb_dir, fname)
            thumb_sel_path = os.path.join(thumb_sel_dir, fname)
            cv2.imwrite(thumb_path, thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
            cv2.imwrite(thumb_sel_path, _mark_selected(thumb), [cv2.IMWRITE_JPEG_QUALITY, 85])
            items.append({"name": fname, "full": full, "thumb": thumb_path, "thumb_sel": thumb_sel_path})

        def tick(frac, v_i=v_i, name=name):
            progress((v_i + min(frac, 1.0)) / len(files),
                     desc=f"{name}: {len(items)} frames so far")

        background = None
        if skip_empty:
            background = (_first_frame_background(cap) if bg_source == BG_FIRST
                          else _estimate_background(cap, total))
            if background is None:
                problems.append(f"no background reference for {name} (filter off)")

        def is_empty(prepped, background=background):
            if background is not None and _changed_fraction(prepped, background, 30) < bg_tol / 100.0:
                skipped[0] += 1
                return True
            return False

        if mode == MODE_ORIGINAL:
            detect_every_transition(cap, total, int(frame_step), float(diff_threshold),
                                    is_empty, save, tick)
        else:
            detect_new_books(cap, total, int(frame_step), still_secs, float(diff_threshold),
                             motion_tol, is_empty, save, tick)
        cap.release()

    msg = f"Extracted {len(items)} frames from {len(files) - len(problems)} video(s)."
    if skip_empty:
        msg += f" Skipped {skipped[0]} empty-background frame(s)."
    if problems:
        msg += " Notes: " + "; ".join(problems) + "."
    if not items:
        msg += " Try a lower threshold."

    selected = []
    return (items, selected) + _render(items, selected, cols) + (msg, None)


# ----------------------------- UI -----------------------------

with gr.Blocks(title="Book Frame Selector") as app:
    gr.Markdown(
        "# 📚 Book Frame Selector\n"
        "1. Drag & drop your video(s)  2. Extract frames  "
        "3. **Click a thumbnail to select it, click it again to unselect it**  "
        "4. Download the selection as a ZIP  "
        "5. Send the selected covers to the local model and export structured metadata."
    )

    items_state = gr.State([])
    selected_state = gr.State([])

    video_input = gr.File(
        label="Drop videos here (or click to browse)",
        file_count="multiple",
        file_types=VIDEO_TYPES,
    )
    zip_input = gr.File(
        label="Or upload a ZIP of extracted book images",
        file_count="single",
        file_types=[".zip"],
    )

    mode = gr.Radio([MODE_ORIGINAL, MODE_NEW_BOOKS], value=MODE_ORIGINAL, label="Extraction mode")

    # These two settings are used by BOTH modes
    with gr.Row():
        diff_threshold = gr.Slider(
            1.0, 50.0, value=DEFAULT_THRESHOLD[MODE_ORIGINAL], step=0.5,
            label="Transition threshold (%)",
            info="Share of the picture that must change. Every-transition mode: vs. the previous "
                 "analysed frame. New-books mode: vs. the last saved frame. Lower = more frames.",
        )
        step_input = gr.Slider(
            1, 1000, value=60, step=1, label="Frame skip step",
            info="Analyse every Nth frame (5 = frames 0, 5, 10, ...). Higher = faster but may miss quick changes.",
        )

    # Background filter, used by BOTH modes
    skip_empty = gr.Checkbox(value=True, label="Skip frames that show only the empty background")
    with gr.Row():
        bg_source = gr.Radio(
            [BG_FIRST, BG_MEDIAN], value=BG_FIRST, label="Background reference",
            info="'First frame' assumes the video starts with the empty background.",
        )
        bg_tol = gr.Slider(
            0.5, 30.0, value=4.0, step=0.5, label="Background match tolerance (%)",
            info="A frame whose picture differs from the background by less than this is treated "
                 "as empty and skipped. Raise it if empty frames still get through; lower it if "
                 "real book frames are being skipped.",
        )
    with gr.Accordion("Extra settings: new books only", open=False):
        still_secs = gr.Slider(0.25, 5, value=0.75, step=0.25, label="Book must stay still for (seconds)")
        motion_tol = gr.Slider(0.002, 0.2, value=0.03, step=0.002,
                               label="Motion tolerance while a book is held")

    process_btn = gr.Button("🔍 Extract frames", variant="primary")
    status = gr.Textbox(label="Status", interactive=False)

    gr.Markdown("### Extracted frames: click to select / unselect")
    cols_slider = gr.Slider(1, 10, value=4, step=1, label="Columns",
                            info="Choose how many thumbnails appear across each row; scroll down to browse the rest.")
    page_md = gr.Markdown("Scroll to browse all extracted frames")

    gallery = gr.Gallery(label="Extracted frames", columns=4, height="auto",
                         allow_preview=False, interactive=False)

    with gr.Row():
        count_md = gr.Markdown("**0** of 0 frames selected")
        select_all_btn = gr.Button("Select all")
        clear_btn = gr.Button("Clear selection")

    selected_gallery = gr.Gallery(label="Selected frames (click a frame here to unselect it)",
                                  columns=4, height="auto", allow_preview=False, interactive=False)

    zip_btn = gr.Button("📦 Create ZIP of selected frames", variant="primary")
    zip_file = gr.File(label="Download ZIP")

    extract_btn = gr.Button("🧠 Extract book metadata from selected frames", variant="primary")
    extraction_log = gr.Textbox(label="Log", lines=8, interactive=False)
    extracted_df = gr.DataFrame(
        headers=["image_name", "title", "author", "publisher", "isbn", "notes", "image_base64"],
        interactive=False,
        label="Extracted book metadata",
    )
    table_download = gr.File(label="Download metadata CSV")

    csv_loader = gr.File(label="Load metadata CSV for verification", file_count="single", file_types=[".csv"])
    csv_preview = gr.HTML(label="CSV preview")

    view_outputs = [gallery, selected_gallery, page_md, count_md]
    layout_inputs = [items_state, selected_state, cols_slider]

    mode.change(default_threshold_for, inputs=[mode], outputs=[diff_threshold])
    process_btn.click(
        run_extraction,
        inputs=[video_input, zip_input, mode, diff_threshold, step_input,
                skip_empty, bg_source, bg_tol, still_secs, motion_tol, cols_slider],
        outputs=[items_state, selected_state] + view_outputs + [status, zip_file],
    )
    gallery.select(toggle_selection, inputs=layout_inputs, outputs=[selected_state] + view_outputs)
    selected_gallery.select(remove_from_selection, inputs=layout_inputs,
                            outputs=[selected_state] + view_outputs)
    select_all_btn.click(select_all, inputs=[items_state, cols_slider],
                         outputs=[selected_state] + view_outputs)
    clear_btn.click(clear_selection, inputs=[items_state, cols_slider],
                    outputs=[selected_state] + view_outputs)
    cols_slider.change(change_layout, inputs=layout_inputs, outputs=view_outputs,
                       trigger_mode="always_last")
    zip_btn.click(make_zip, inputs=[items_state, selected_state], outputs=[zip_file])
    extract_btn.click(extract_book_metadata, inputs=[items_state, selected_state],
                     outputs=[extracted_df, table_download, extraction_log])
    csv_loader.change(render_metadata_csv_preview, inputs=[csv_loader], outputs=[csv_preview])

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)