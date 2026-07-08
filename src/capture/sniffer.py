"""
src/capture/sniffer.py
----------------------
LiveSniffer: a high-level wrapper around cicflowmeter's FlowSession and
scapy's AsyncSniffer that captures live traffic on a network interface and
routes completed flows to a user-supplied callback.

Integration strategy
--------------------
cicflowmeter's ``FlowSession.__init__`` calls ``output_writer_factory(...)``
immediately, which requires a valid ``output_mode``.  To bypass this we
subclass ``FlowSession`` as ``CallbackFlowSession``, overriding ``__init__``
so it sets ``self.output_writer`` to our ``CallbackFlowWriter`` directly,
skipping the factory entirely.
"""

import time
import threading
from typing import Callable

from scapy.sendrecv import AsyncSniffer

from cicflowmeter.flow_session import FlowSession
from cicflowmeter.sniffer import _start_periodic_gc

from src.capture.flow_writer import CallbackFlowWriter


# ---------------------------------------------------------------------------
# Custom FlowSession that bypasses the output_writer_factory
# ---------------------------------------------------------------------------

class CallbackFlowSession(FlowSession):
    """FlowSession subclass that injects a CallbackFlowWriter instead of
    the default CSV / HTTP writers."""

    def __init__(self, callback: Callable[[dict], None], verbose: bool = False):
        # Call object.__init__ to skip FlowSession.__init__ which calls the
        # factory we want to bypass.
        object.__init__(self)

        # Re-implement the minimal state that FlowSession expects:
        from cicflowmeter.utils import get_logger
        self.flows: dict = {}
        self.verbose = verbose
        self.fields = None
        self.output_mode = "callback"
        self.output = None
        self.logger = get_logger(self.verbose)
        self.packets_count = 0
        self.output_writer = CallbackFlowWriter(callback)
        self._lock = threading.Lock()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class LiveSniffer:
    """Captures live traffic on a network interface and streams completed
    flows to ``callback``.

    Parameters
    ----------
    interface : str
        Network interface name (e.g. ``"Wi-Fi"`` or ``"eth0"``).
    callback : Callable[[dict], None]
        Called for each completed flow with its feature dictionary.
    verbose : bool
        Enable cicflowmeter debug logging.
    """

    def __init__(
        self,
        interface: str,
        callback: Callable[[dict], None],
        verbose: bool = False,
    ) -> None:
        self._interface = interface
        self._callback = callback
        self._verbose = verbose
        self._sniffer: AsyncSniffer | None = None
        self._session: CallbackFlowSession | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the async sniffer and background GC thread."""
        self._session = CallbackFlowSession(
            callback=self._callback,
            verbose=self._verbose,
        )
        _start_periodic_gc(self._session)

        self._sniffer = AsyncSniffer(
            iface=self._interface,
            filter="ip and (tcp or udp)",
            prn=self._session.process,
            store=False,
        )
        self._sniffer.start()
        print(f"[LiveSniffer] Capturing on interface: {self._interface!r}")

    def stop(self) -> None:
        """Stop capture and flush remaining flows."""
        if self._sniffer is not None:
            try:
                self._sniffer.stop()
            except Exception:
                pass

        if self._session is not None:
            # Stop the periodic GC thread
            if hasattr(self._session, "_gc_stop"):
                self._session._gc_stop.set()
                self._session._gc_thread.join(timeout=2.0)
            # Flush any remaining in-progress flows
            try:
                self._session.flush_flows()
            except Exception:
                pass

        print("[LiveSniffer] Stopped.")

    def join(self) -> None:
        """Block until the sniffer finishes (useful for offline PCAP replay)."""
        if self._sniffer is not None:
            self._sniffer.join()
