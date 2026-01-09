import numpy as np
import scipy


class Filter:
    def __init__(self, config, online=False) -> None:
        self.config = config

        self.sos = scipy.signal.butter(
            self.config.order,
            Wn=np.array(self.config.frequencies),
            btype="bandpass",
            fs=self.config.fs,
            output="sos",
        )

        # In the case of an online filter, we need to maintain state between calls
        if online:
            # Make this for multiple channels
            self.zi = scipy.signal.sosfilt_zi(self.sos)
            self.zi = np.tile(self.zi, (len(self.config.channels), 1, 1))

    def apply_filter_offline(self, raw):
        # Use SOS for actual filtering (stability)
        filtered_signal = scipy.signal.sosfilt(self.sos, raw.get_data())

        # Create a new Raw object with the filtered data
        filtered_raw = raw.copy()
        filtered_raw._data = filtered_signal

        return filtered_raw

    def apply_filter_online(self, data_chunk):
        # Apply the filter to the incoming data chunk, maintaining state
        filtered_chunk = np.zeros_like(data_chunk)
        for ch_idx in range(data_chunk.shape[0]):
            filtered_chunk[ch_idx, :], self.zi[ch_idx] = scipy.signal.sosfilt(
                self.sos, data_chunk[ch_idx, :], zi=self.zi[ch_idx]
            )
        return filtered_chunk

    def apply_notch(self, raw, freqs):
        for freq_band in freqs:
            raw.notch_filter(freqs=freq_band)  # since Nyquist = 80 Hz  # europe = 50 Hz
        return raw

    def get_filter_latency(self):
        # Convert SOS to transfer function
        b, a = scipy.signal.sos2tf(self.sos)

        # Compute group delay for each frequency
        w, gd = scipy.signal.group_delay((b, a))
        freqs = np.linspace(self.config.frequencies[0], self.config.frequencies[1], 100)
        gd_interp = np.interp(freqs, w * self.config.fs / (2 * np.pi), gd)

        # Average latency in milliseconds
        avg_latency_ms = np.mean(gd_interp) / self.config.fs * 1000

        return avg_latency_ms
