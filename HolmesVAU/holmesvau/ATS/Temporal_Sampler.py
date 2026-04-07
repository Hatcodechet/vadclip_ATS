import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - allows score-only ATS without torch installed
    torch = None


class Temporal_Sampler():
    def __init__(self, ckpt_path=None, device="cpu", tau=0.1):
        if torch is None:
            self.device = device
        else:
            self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.tau = tau
        self.anomaly_scorer = None

        if ckpt_path is not None:
            if torch is None:
                raise ImportError("torch is required when loading the URDMU anomaly scorer.")
            from .anomaly_scorer import URDMU
            self.anomaly_scorer = URDMU().to(self.device)
            self.anomaly_scorer.load_state_dict(torch.load(ckpt_path, map_location=self.device))
            self.anomaly_scorer.eval()

    @staticmethod
    def _uniform_sample_indices(num_frames, select_frames):
        if num_frames <= 0:
            return []
        return [int(i) for i in np.rint(np.linspace(0, num_frames - 1, select_frames))]

    @staticmethod
    def _to_numpy_scores(scores):
        if torch is not None and isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        scores = np.asarray(scores, dtype=np.float32)
        if scores.ndim == 0:
            scores = scores.reshape(1)
        if scores.ndim > 1:
            scores = np.squeeze(scores)
        if scores.ndim != 1:
            raise ValueError(f"Expected scores with shape [T], but got {scores.shape}.")
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    def get_anomaly_scores(self, pixel_values, model):
        if torch is None:
            raise ImportError("torch is required when computing anomaly scores from raw frames.")
        if self.anomaly_scorer is None:
            raise RuntimeError(
                "Anomaly scorer is not loaded. Provide ckpt_path when using the raw-frame ATS path."
            )

        def batchify(pixel_values, batch_size=16):
            num_frames = pixel_values.shape[0]
            batches = []
            for i in range(0, num_frames, batch_size):
                batch_end = min(i + batch_size, num_frames)
                batches.append(pixel_values[i:batch_end, :, :, :])
            return batches

        with torch.no_grad():
            pixel_values = pixel_values.to(torch.bfloat16)
            pixel_values_batched = batchify(pixel_values)
            cls_tokens = []
            for batch_id, data in enumerate(pixel_values_batched):
                data = data.to(self.device)
                vit_embeds = model.vision_model(
                    pixel_values=data,
                    output_hidden_states=False,
                    return_dict=True,
                ).last_hidden_state
                cls_tokens.append(vit_embeds[:, 0, :])
                print("Extracted {}/{}".format(batch_id, len(pixel_values_batched)), end="\r")
            vid_feats = torch.cat(cls_tokens, dim=0).to(torch.float32).unsqueeze(0)
            anomaly_scores = self.anomaly_scorer(vid_feats)["anomaly_scores"]
            anomaly_scores = anomaly_scores[0].detach().cpu().numpy()
        return anomaly_scores

    def density_aware_sample_from_scores(self, scores, select_frames=16, tau=None):
        scores = self._to_numpy_scores(scores)
        num_frames = scores.shape[0]
        if num_frames == 0:
            return []

        if num_frames <= select_frames or float(scores.sum()) < 1.0:
            return self._uniform_sample_indices(num_frames, select_frames)

        tau = self.tau if tau is None else tau
        density_scores = scores + tau
        if np.any(density_scores <= 0):
            density_scores = density_scores - density_scores.min() + tau

        if float(density_scores.sum()) <= 0:
            return self._uniform_sample_indices(num_frames, select_frames)

        score_cumsum = np.concatenate(
            (np.zeros((1,), dtype=np.float32), np.cumsum(density_scores, dtype=np.float32)),
            axis=0,
        )

        if np.any(np.diff(score_cumsum) <= 0):
            return self._uniform_sample_indices(num_frames, select_frames)

        max_score_cumsum = float(score_cumsum[-1])
        scale_x = np.linspace(1, max_score_cumsum, select_frames)
        sampled_idxs = np.interp(scale_x, score_cumsum, np.arange(num_frames + 1))
        return [min(num_frames - 1, max(0, int(idx))) for idx in sampled_idxs]

    def sample_from_scores(self, scores, select_frames=16, tau=None, return_scores=False):
        scores = self._to_numpy_scores(scores)
        sampled_idxs = self.density_aware_sample_from_scores(
            scores,
            select_frames=select_frames,
            tau=tau,
        )
        if return_scores:
            return scores, sampled_idxs
        return sampled_idxs

    def density_aware_sample(self, pixel_values, model, select_frames=16):
        """
        Raw-frame ATS path:
            pixel_values [T, C, H, W] -> vision_model -> URDMU -> temporal scores -> ATS indices
        """
        anomaly_score = self.get_anomaly_scores(pixel_values, model)
        sampled_idxs = self.density_aware_sample_from_scores(
            anomaly_score,
            select_frames=select_frames,
        )
        return anomaly_score, sampled_idxs
