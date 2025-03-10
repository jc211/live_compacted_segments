from live_compacted_segments.timeseries_sampler import TimeSeriesSampler
import torch
import warp as wp


class CompositeSampler:
    def __init__(self, samplers: dict[str, TimeSeriesSampler], lead_sampler_name: str):
        assert lead_sampler_name in samplers, (
            "Lead sampler must be in the provided samplers."
        )
        self.lead_sampler_name = lead_sampler_name
        lead_sampler = samplers[lead_sampler_name]

        num_series = lead_sampler.num_series
        batch_size = lead_sampler.batch_size
        device = lead_sampler.device

        for sampler in samplers.values():
            assert sampler.num_series == num_series, (
                "All samplers must have the same number of series."
            )
            assert sampler.batch_size == batch_size, (
                "All samplers must have the same batch size."
            )
            assert sampler.device == device, "All samplers must be on the same device."

        self.samplers = samplers
        self.lead_sampler = lead_sampler
        self.streams = {name: wp.Stream(device=device) for name in samplers}
        self.profile_active = False

    def sample(self) -> dict[str, torch.Tensor]:
        with wp.ScopedTimer(
            f"Lead Sampler {self.lead_sampler_name}",
            use_nvtx=True,
            active=self.profile_active,
        ):
            time_segments, signal_index = self.lead_sampler.sample_time_ranges()
        sampled_data = {}
        for name, sampler in self.samplers.items():
            with wp.ScopedTimer(
                f"Sampler {name}", use_nvtx=True, active=self.profile_active
            ):
                # with wp.ScopedStream(self.streams[name]):
                # wp.wait_event(self.lead_sampler.time_sampled_event)
                sampler.sample_from_time_segments(time_segments, signal_index)
                sampled_data[name] = wp.to_torch(sampler.warp_arrays["output"])

        # for stream in self.streams.values():
        #     wp.synchronize_stream(stream)

        return sampled_data
