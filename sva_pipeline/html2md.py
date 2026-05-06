"""
html2md.py
----------
HTML-to-Markdown converter for hardware design specifications.

Hardware specs (NVDLA, OpenTitan, RISC-V) are often published as HTML pages.
This module converts them to clean Markdown so the RAG layer can index them.

Images (timing diagrams, block diagrams, waveforms) are extracted from the
HTML and saved alongside the Markdown output in an ``_images/`` subdirectory.
The Markdown references them with relative paths so they remain accessible.

Usage is config-driven: the user lists HTML file paths in the YAML config,
and the pipeline converts them at startup before building RAG indices.

Conversion is cached by file modification time -- if the .md output already
exists and is newer than the .html source, conversion is skipped.
"""

import base64
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, List, Tuple
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Tag
from markdownify import markdownify

logger = logging.getLogger(__name__)


def convert_html_to_markdown(html_path: str, output_path: str) -> str:
    """
    Convert a single HTML file to clean Markdown with image extraction.

    Steps:
      1. Parse HTML with BeautifulSoup.
      2. Strip non-content elements (scripts, styles, nav, footers).
      3. Extract images to ``<output_dir>/_images/<stem>/`` and rewrite
         ``<img>`` tags to use relative markdown image syntax.
      4. Convert remaining HTML to Markdown via markdownify.
      5. Clean up excessive whitespace.
      6. Write to output_path.

    Parameters
    ----------
    html_path : str
        Path to the source HTML file.
    output_path : str
        Path where the Markdown output will be written.

    Returns
    -------
    str
        The output_path (for convenience in chaining).
    """
    logger.info("Converting %s → %s", html_path, output_path)

    # Read the HTML source.
    with open(html_path, "r", encoding="utf-8", errors="ignore") as fh:
        html_content = fh.read()

    # Parse with BeautifulSoup.
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove non-content elements that add noise to the markdown.
    _strip_non_content(soup)

    # Extract images — save files and rewrite <img> tags to relative paths.
    output_dir = str(Path(output_path).parent)
    stem = Path(output_path).stem
    html_dir = str(Path(html_path).parent)
    image_count = _extract_images(soup, html_dir, output_dir, stem)

    # Convert the cleaned HTML to Markdown.
    md_content = markdownify(
        str(soup),
        heading_style="ATX",         # use # style headings
        bullets="-",                 # use - for unordered lists
    )

    # Clean up the markdown output.
    md_content = _clean_markdown(md_content)

    # Add a source header.
    source_name = Path(html_path).name
    header = f"<!-- Converted from {source_name} by sva_pipeline/html2md.py -->\n\n"
    md_content = header + md_content

    # Write the output.
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(md_content)

    logger.info(
        "  Converted: %d chars HTML → %d chars Markdown, %d image(s) extracted",
        len(html_content), len(md_content), image_count,
    )
    return output_path


def convert_all_html_docs(config: Any) -> List[str]:
    """
    Convert all HTML files listed in the pipeline config.

    Skips conversion if the output .md file already exists and is newer
    than the source .html file (mtime-based caching).

    Parameters
    ----------
    config : PipelineConfig
        Must have html_docs_enabled, html_docs_files, html_docs_output_dir.

    Returns
    -------
    list of str
        Paths to the converted (or cached) Markdown files.
    """
    if not config.html_docs_enabled:
        return []

    files = config.html_docs_files
    if not files:
        logger.info("html_docs enabled but no files listed — nothing to convert.")
        return []

    # Determine output directory.
    output_dir = config.html_docs_output_dir or config.docs_dir
    if not output_dir:
        output_dir = "."
        logger.warning(
            "No output_dir or docs_dir specified for HTML conversion — "
            "writing to current directory."
        )

    converted: List[str] = []

    for html_path in files:
        if not os.path.exists(html_path):
            logger.warning("HTML file not found: %s — skipping.", html_path)
            continue

        # Determine output path: same stem, .md extension.
        stem = Path(html_path).stem
        md_path = os.path.join(output_dir, f"{stem}.md")

        # Check cache: skip if .md exists and is newer than .html.
        if os.path.exists(md_path):
            html_mtime = os.path.getmtime(html_path)
            md_mtime = os.path.getmtime(md_path)
            if md_mtime >= html_mtime:
                logger.info("  Cached (up to date): %s", md_path)
                converted.append(md_path)
                continue

        # Convert.
        try:
            convert_html_to_markdown(html_path, md_path)
            converted.append(md_path)
        except Exception as exc:
            logger.error("Failed to convert %s: %s", html_path, exc)

    return converted


# ---------------------------------------------------------------------------
# Image extraction
# ---------------------------------------------------------------------------

def _extract_images(
    soup: BeautifulSoup,
    html_dir: str,
    output_dir: str,
    stem: str,
) -> int:
    """
    Extract images from the HTML and save them to disk.

    Handles three image source types:
      1. Local/relative paths  (``src="images/timing.png"``)
      2. Data URIs             (``src="data:image/png;base64,..."``)
      3. Remote URLs           (``src="https://..."``) — skipped with a
         warning, as downloading external resources is out of scope.

    Each extracted image is saved to ``<output_dir>/_images/<stem>/``.
    The ``<img>`` tag in the soup is replaced with a markdown-compatible
    ``<img>`` tag using the relative path, which markdownify will convert
    to ``![alt](path)`` syntax.

    Parameters
    ----------
    soup : BeautifulSoup
        Parsed HTML (modified in-place).
    html_dir : str
        Directory containing the source HTML file (for resolving relative paths).
    output_dir : str
        Directory where the .md file will be written.
    stem : str
        Stem of the output .md filename (used as image subdirectory name).

    Returns
    -------
    int
        Number of images successfully extracted.
    """
    images = soup.find_all("img")
    if not images:
        return 0

    # Create the image output directory.
    image_dir = os.path.join(output_dir, "_images", stem)
    Path(image_dir).mkdir(parents=True, exist_ok=True)

    extracted = 0

    for idx, img in enumerate(images):
        src = img.get("src", "")
        alt = img.get("alt", "")
        title = img.get("title", "")

        if not src:
            img.decompose()
            continue

        # Determine the image filename and extract/copy it.
        saved_path = None

        if src.startswith("data:"):
            # Data URI — decode and save.
            saved_path = _save_data_uri(src, image_dir, idx)

        elif src.startswith("http://") or src.startswith("https://"):
            # Remote URL — log a warning but try to download.
            saved_path = _download_image(src, image_dir, idx)

        else:
            # Local/relative path — resolve and copy.
            saved_path = _copy_local_image(src, html_dir, image_dir, idx)

        if saved_path:
            # Compute the relative path from the .md file to the image.
            rel_path = os.path.relpath(saved_path, output_dir)
            # Replace the <img> tag with a markdown-friendly version.
            # Use alt text if available, otherwise use "diagram" + index.
            alt_text = alt or title or f"diagram_{idx}"
            # Replace the img tag with a text node that markdownify will
            # pass through.  We use a custom marker that survives conversion.
            marker = f"\n\n![{alt_text}]({rel_path})\n\n"
            img.replace_with(marker)
            extracted += 1
            logger.debug("  Extracted image %d: %s", idx, rel_path)
        else:
            # Could not extract — replace with alt text or remove.
            if alt:
                img.replace_with(f"[Image: {alt}]")
            else:
                img.decompose()

    if extracted:
        logger.info("  Extracted %d image(s) to %s", extracted, image_dir)

    return extracted


def _save_data_uri(data_uri: str, image_dir: str, idx: int) -> str:
    """
    Decode a data URI and save the image to disk.

    Handles: ``data:image/png;base64,iVBOR...``
    Returns the saved file path, or empty string on failure.
    """
    try:
        # Parse the data URI.
        match = re.match(r"data:image/(\w+);base64,(.+)", data_uri, re.DOTALL)
        if not match:
            return ""

        ext = match.group(1)
        if ext == "svg+xml":
            ext = "svg"
        b64_data = match.group(2)
        image_bytes = base64.b64decode(b64_data)

        filename = f"image_{idx:03d}.{ext}"
        filepath = os.path.join(image_dir, filename)
        with open(filepath, "wb") as fh:
            fh.write(image_bytes)
        return filepath

    except Exception as exc:
        logger.warning("Failed to decode data URI for image %d: %s", idx, exc)
        return ""


def _copy_local_image(
    src: str, html_dir: str, image_dir: str, idx: int
) -> str:
    """
    Copy a locally referenced image to the image output directory.

    Resolves relative paths against the HTML file's directory.
    Returns the saved file path, or empty string on failure.
    """
    # Resolve the source path relative to the HTML file.
    if os.path.isabs(src):
        source_path = src
    else:
        source_path = os.path.normpath(os.path.join(html_dir, src))

    if not os.path.exists(source_path):
        logger.warning(
            "Image not found: %s (resolved from %s) — skipping.", source_path, src
        )
        return ""

    # Determine output filename — keep original name but prefix with index
    # to avoid collisions.
    original_name = Path(source_path).name
    filename = f"{idx:03d}_{original_name}"
    dest_path = os.path.join(image_dir, filename)

    try:
        shutil.copy2(source_path, dest_path)
        return dest_path
    except OSError as exc:
        logger.warning("Failed to copy image %s: %s", source_path, exc)
        return ""


def _download_image(url: str, image_dir: str, idx: int) -> str:
    """
    Download a remote image and save it locally.

    Uses urllib to avoid adding external dependencies.
    Returns the saved file path, or empty string on failure.
    """
    try:
        from urllib.request import urlretrieve

        # Determine file extension from URL.
        parsed = urlparse(url)
        path = parsed.path
        ext = Path(path).suffix or ".png"
        filename = f"image_{idx:03d}{ext}"
        dest_path = os.path.join(image_dir, filename)

        logger.info("  Downloading image: %s", url)
        urlretrieve(url, dest_path)
        return dest_path

    except Exception as exc:
        logger.warning("Failed to download image %s: %s", url, exc)
        return ""


# ---------------------------------------------------------------------------
# Content stripping
# ---------------------------------------------------------------------------

# Tags to remove entirely (they add noise, not content).
_REMOVE_TAGS = ["script", "style", "nav", "header", "footer", "aside", "noscript"]

# CSS classes/IDs that indicate non-content elements.
_REMOVE_PATTERNS = re.compile(
    r"menu|nav|sidebar|footer|breadcrumb|toc|search|cookie|banner|ads",
    re.IGNORECASE,
)


def _strip_non_content(soup: BeautifulSoup) -> None:
    """
    Remove non-content elements from the parsed HTML in-place.

    Strips: scripts, styles, navigation, headers, footers, sidebars,
    and any element whose class or id matches common non-content patterns.
    """
    # Remove specific tags entirely.
    for tag_name in _REMOVE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Remove elements with non-content class or id.
    for element in soup.find_all(True):
        classes = " ".join(element.get("class", []))
        elem_id = element.get("id", "")
        if _REMOVE_PATTERNS.search(classes) or _REMOVE_PATTERNS.search(elem_id):
            element.decompose()


def _clean_markdown(text: str) -> str:
    """
    Clean up markdownify output.

    - Collapse 3+ consecutive blank lines into 2.
    - Strip trailing whitespace from each line.
    - Remove leading/trailing blank lines.
    """
    # Strip trailing whitespace per line.
    lines = [line.rstrip() for line in text.splitlines()]
    text = "\n".join(lines)

    # Collapse excessive blank lines (3+ → 2).
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace.
    return text.strip() + "\n"
