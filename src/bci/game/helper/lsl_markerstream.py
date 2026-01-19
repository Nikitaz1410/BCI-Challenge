from pylsl import StreamInfo, StreamOutlet
import logging

LOGGER: logging.Logger = logging.getLogger("LSLMarkerStream")


class LSLMarkerStream:
    """Class to send markers to an LSL stream."""

    def __init__(self: "LSLMarkerStream", name: str, marker_id: str) -> None:
        """
        Constructor for the LSLMarkerStream class.

        Parameters
        ----------
           name : str
              The name of the LSL stream.
           id : str
              The ID of the LSL stream.
        """
        self.name = name
        self.marker_id = marker_id
        self.streamInfo = StreamInfo(name, "Markers", 1, 0, "string", self.marker_id)
        self.outletLSL = StreamOutlet(self.streamInfo)

    def send_marker(self: "LSLMarkerStream", markerName: str) -> None:
        """
        Send a marker to the LSL stream.

        Parameters
        ----------
           markerName : str
              The name of the marker to be sent.
        """
        self.outletLSL.push_sample([markerName])
        LOGGER.debug("Sent marker: %s", markerName)
