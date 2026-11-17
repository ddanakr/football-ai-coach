"""The "Download the labels" block of the viewer (FR-9): the current events as a ground-truth JSON
file, optionally only those a person has reviewed. Drawn into the sidebar by `app.py`; nothing is
written to the database or to the match directory (the file goes to the browser)."""

import streamlit as st

from aicoach.errors import AicoachError
from aicoach.i18n import ru
from aicoach.labels_export import DEFAULT_VIDEO_NAME, export_labels, ground_truth_json
from aicoach.ui.launch import AppArgs

DOWNLOAD_KEY = "btn_labels_download"
REVIEWED_KEY = "labels_reviewed_only"


def render_labels(args: AppArgs) -> None:
    """Draw the checkbox and the download button (in the sidebar), with the number of events.

    Each rerun builds the file again from the current state, so the download always has the
    latest edits and review decisions.
    """
    with st.sidebar:
        st.header(ru.LABELS_HEADER)
        reviewed_only = st.checkbox(
            ru.LABELS_REVIEWED_ONLY, key=REVIEWED_KEY, help=ru.LABELS_REVIEWED_HELP
        )
        try:
            exported = export_labels(
                args.db,
                args.match_id,
                args.results_dir,
                video_name=args.video.name if args.video is not None else DEFAULT_VIDEO_NAME,
                reviewed_only=reviewed_only,
            )
        except AicoachError as err:
            st.error(ru.LABELS_FAILED.format(error=err))
            return
        st.caption(ru.LABELS_SUMMARY.format(count=len(exported.ground_truth.events)))
        if exported.skipped_no_half:
            st.caption(ru.LABELS_SKIPPED.format(count=exported.skipped_no_half))
        st.download_button(
            ru.LABELS_DOWNLOAD,
            data=ground_truth_json(exported.ground_truth),
            file_name=f"{args.match_id}_labels.json",
            mime="application/json",
            key=DOWNLOAD_KEY,
        )
