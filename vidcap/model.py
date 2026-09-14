"""Vanilla captioner: selected frame embeddings -> connector -> projector -> LoRA'd LLM."""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import LLM_MODEL, VISION_DIM
from .lora import apply_lora, lora_parameters

IGNORE = -100


# --- Connectors: (B, K, D) frame embeddings -> (B, M, D) video tokens -----------

class MeanPool(nn.Module):
    """Ablation baseline. M=1; the projector expands it into several prefix tokens."""
    M = 1

    def forward(self, f):
        return f.mean(dim=1, keepdim=True)


class Resampler(nn.Module):
    """Primary connector: learned queries cross-attend over the selected frames (Flamingo-style)."""

    def __init__(self, d, n_query=8, heads=8, layers=2):
        super().__init__()
        self.M = n_query
        self.query = nn.Parameter(torch.randn(n_query, d) * 0.02)
        self.blocks = nn.ModuleList(
            nn.ModuleDict({
                "attn": nn.MultiheadAttention(d, heads, batch_first=True),
                "n1": nn.LayerNorm(d), "n2": nn.LayerNorm(d),
                "ff": nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)),
            }) for _ in range(layers))

    def forward(self, f):
        q = self.query.unsqueeze(0).expand(f.size(0), -1, -1)
        for b in self.blocks:
            h = b["n1"](q)
            q = q + b["attn"](h, f, f, need_weights=False)[0]
            q = q + b["ff"](b["n2"](q))
        return q


class TemporalTransformer(nn.Module):
    """Ablation baseline: self-attention over frames, one video token out per frame."""

    def __init__(self, d, heads=8, layers=2, max_frames=64):
        super().__init__()
        self.M = None  # M == K, set by input
        self.pos = nn.Parameter(torch.randn(max_frames, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)

    def forward(self, f):
        return self.enc(f + self.pos[:f.size(1)])


CONNECTORS = {"meanpool": MeanPool, "resampler": Resampler, "temporal": TemporalTransformer}


# --- Projector: video tokens -> LLM embedding space -----------------------------

class Projector(nn.Module):
    """Each video token becomes `expand` prefix tokens in the LLM's space."""

    def __init__(self, d_in, d_llm, expand=1, hidden=None):
        super().__init__()
        self.expand, self.d_llm = expand, d_llm
        hidden = hidden or 4 * d_in
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                                 nn.Linear(hidden, expand * d_llm))

    def forward(self, v):
        return self.net(v).view(v.size(0), -1, self.d_llm)


# --- Captioner ------------------------------------------------------------------

class VideoCaptioner(nn.Module):
    """Video prefix + causally-masked caption tokens, cross-entropy on caption tokens only."""

    def __init__(self, d_vis=VISION_DIM, connector="resampler", n_prefix=8,
                 lora_r=8, llm_name=LLM_MODEL, blind=False, dtype=None):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(llm_name)
        self.tok.pad_token = self.tok.pad_token or self.tok.eos_token
        # The frozen decoder dominates GPU memory and Qwen2.5 ships bf16, so fp32 would
        # cost 6.2GB of weights instead of 3.1GB for a model that is never updated —
        # on a 16GB T4 that is the difference between fitting and OOM. CPU stays fp32
        # (bf16 is slow there, and the tests run on CPU).
        if dtype is None:
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name, dtype=dtype)
        d_llm = self.llm.config.hidden_size
        self.n_lora = apply_lora(self.llm, r=lora_r) if lora_r else 0

        if connector == "meanpool":
            self.connector, expand = MeanPool(), n_prefix
        elif connector == "resampler":
            self.connector, expand = Resampler(d_vis, n_query=n_prefix), 1
        else:
            self.connector, expand = TemporalTransformer(d_vis), 1
        self.projector = Projector(d_vis, d_llm, expand)
        # SigLIP embeddings are large-magnitude and anisotropic (a red and a blue frame sit at
        # 0.924 cosine). Feeding them raw into a cold connector is a known collapse recipe;
        # every comparable system normalises first.
        self.vis_norm = nn.LayerNorm(d_vis)
        self.blind = blind  # control: zeroes the visual prefix, keeps everything else identical

    def prefix(self, frames):
        """frames: (B, K, d_clip) -> (B, P, d_llm), in the LLM's dtype.

        Connector and projector stay fp32 — they are what AdamW actually updates, and
        bf16 optimizer states converge worse. Casting here rather than at each call site
        means forward, generate and decode all meet the LLM in its own dtype.
        """
        f = self.vis_norm(frames)
        if self.blind:
            f = torch.zeros_like(f)
        return self.projector(self.connector(f)).to(self.llm.dtype)

    def forward(self, frames, input_ids, attention_mask=None):
        """Returns (loss, logits). input_ids are the caption; prefix positions are not predicted."""
        pre = self.prefix(frames)
        emb = self.llm.get_input_embeddings()(input_ids)
        x = torch.cat([pre, emb], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        mask = torch.cat([torch.ones(pre.shape[:2], device=x.device, dtype=attention_mask.dtype),
                          attention_mask], dim=1)
        labels = torch.cat([torch.full(pre.shape[:2], IGNORE, device=x.device, dtype=torch.long),
                            input_ids.masked_fill(attention_mask == 0, IGNORE)], dim=1)
        # ponytail: plain causal mask over the whole sequence (ClipCap-style). Bidirectional
        # attention within the video prefix is a later ablation, not needed to train.
        out = self.llm(inputs_embeds=x, attention_mask=mask, labels=labels)
        return out.loss, out.logits

    @torch.no_grad()
    def generate(self, frames, max_new_tokens=20):
        """Greedy decode. Phase 7 replaces this with hand-written greedy + beam search."""
        pre = self.prefix(frames)
        mask = torch.ones(pre.shape[:2], device=pre.device, dtype=torch.long)
        out = self.llm.generate(inputs_embeds=pre, attention_mask=mask,
                                max_new_tokens=max_new_tokens, do_sample=False,
                                pad_token_id=self.tok.pad_token_id)
        return [self.tok.decode(o, skip_special_tokens=True).strip() for o in out]

    def trainable_parameters(self):
        return list(self.connector.parameters()) + list(self.projector.parameters()) \
            + lora_parameters(self.llm)


def uniform_select(emb, k):
    """The default this project exists to beat: evenly spaced indices over the candidate pool."""
    n = emb.shape[0]
    if n <= k:
        return emb, list(range(n))
    idx = torch.linspace(0, n - 1, k).round().long().tolist()
    return emb[idx], idx
