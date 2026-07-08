"""
src/capture/flow_writer.py
--------------------------
A custom cicflowmeter OutputWriter that, instead of writing flows to a CSV
file or HTTP endpoint, calls a user-supplied Python callback with the raw
flow data dictionary.

This is the key integration point: cicflowmeter's FlowSession calls
``self.output_writer.write(data)`` for every completed flow. By injecting
a CallbackFlowWriter we intercept each flow and route it directly into our
model pipeline without any disk I/O.
"""

from typing import Callable


class CallbackFlowWriter:
    """An OutputWriter-compatible class that routes completed flows to a callback.

    Parameters
    ----------
    callback : Callable[[dict], None]
        Function called with the raw flow feature dictionary produced by
        cicflowmeter every time a flow is completed or garbage-collected.
    """

    def __init__(self, callback: Callable[[dict], None]) -> None:
        self._callback = callback

    def write(self, data: dict) -> None:
        """Called by cicflowmeter's FlowSession for each completed flow."""
        try:
            self._callback(data)
        except Exception as exc:  # noqa: BLE001
            # Never let a callback error crash the sniffer thread.
            print(f"[CallbackFlowWriter] Callback error (flow dropped): {exc}")
