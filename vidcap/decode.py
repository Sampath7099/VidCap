"""Hand-written greedy and beam search over the LLM, given a video prefix. No HF generate().

ponytail: no KV cache — each step re-runs the full sequence. Sequences here are ~30 tokens, so
the cost is negligible, and it keeps greedy and beam numerically identical (a cached path and an
uncached path can disagree in the last bits and flip a near-tie argmax, which would make the
beam==greedy gate flaky). Add a cache if sequence lengths ever grow.
"""
import torch


@torch.no_grad()
def _next_logits(model, pre, tokens):
    """pre: (B,P,D) video prefix. tokens: (B,T) generated so far, or None. -> (B,V) next-token logits."""
    x = pre if tokens is None or tokens.size(1) == 0 else \
        torch.cat([pre, model.llm.get_input_embeddings()(tokens)], dim=1)
    return model.llm(inputs_embeds=x).logits[:, -1, :]


@torch.no_grad()
def greedy(model, frames, max_new_tokens=20):
    """Argmax at every step. Returns list[str]."""
    tok = model.tok
    pre = model.prefix(frames)
    B, dev = pre.size(0), pre.device
    toks = torch.zeros(B, 0, dtype=torch.long, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)

    for _ in range(max_new_tokens):
        nxt = _next_logits(model, pre, toks).argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, tok.eos_token_id), nxt)
        toks = torch.cat([toks, nxt.unsqueeze(1)], dim=1)
        done |= nxt == tok.eos_token_id
        if done.all():
            break
    return _decode(tok, toks)


@torch.no_grad()
def beam_search(model, frames, beam=4, max_new_tokens=20, length_penalty=1.0):
    """Returns list[str]. beam=1 must reproduce greedy exactly — that is the correctness gate."""
    tok, eos = model.tok, model.tok.eos_token_id
    outs = []

    for b in range(frames.size(0)):  # one clip at a time; eval batches are small
        pre1 = model.prefix(frames[b:b + 1])
        seqs = [([], 0.0)]  # (tokens, cumulative logprob), all still live
        finished = []

        for _ in range(max_new_tokens):
            live = [(t, s) for t, s in seqs if not t or t[-1] != eos]
            finished += [(t, s) for t, s in seqs if t and t[-1] == eos]
            if not live:
                break

            n = len(live)
            pre = pre1.expand(n, -1, -1)
            toks = torch.tensor([t for t, _ in live], dtype=torch.long,
                                device=pre1.device) if live[0][0] else None
            lp = torch.log_softmax(_next_logits(model, pre, toks).float(), dim=-1)
            top = lp.topk(beam, dim=-1)

            cands = [(live[i][0] + [top.indices[i, j].item()], live[i][1] + top.values[i, j].item())
                     for i in range(n) for j in range(beam)]
            cands.sort(key=lambda x: x[1] / (len(x[0]) ** length_penalty), reverse=True)
            seqs = cands[:beam]

        finished += seqs
        best = max(finished, key=lambda x: x[1] / (len(x[0]) ** length_penalty))[0]
        outs.append(_decode(tok, torch.tensor([best]))[0])
    return outs


def _decode(tok, toks):
    return [tok.decode(row, skip_special_tokens=True).strip() for row in toks]
