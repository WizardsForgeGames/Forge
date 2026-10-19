"""Forge embedding layer using custom gather/scatter kernels."""
import torch
import torch.nn as nn
from forge import _forge_C as _C


class _EmbeddingFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, context, embeddings, B, ctx_size, E):
        I = ctx_size * E
        output = torch.empty(I, B, dtype=torch.float32, device=embeddings.device)
        _C.embedding_gather(context, embeddings, output, B, ctx_size, E)
        ctx.save_for_backward(context)
        ctx.B = B
        ctx.ctx_size = ctx_size
        ctx.E = E
        ctx.V = embeddings.shape[0]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        context, = ctx.saved_tensors
        d_embeddings = torch.zeros(
            ctx.V, ctx.E, dtype=torch.float32, device=grad_output.device)
        _C.embedding_scatter(
            context, grad_output, d_embeddings,
            ctx.B, ctx.ctx_size, ctx.E)
        return None, d_embeddings, None, None, None


class Embedding(nn.Module):
    """Embedding layer using Forge gather/scatter kernels.

    Stores a [V, E] embedding table and gathers rows by index.

    Args:
        num_embeddings: Size of the vocabulary (V).
        embedding_dim: Dimension of each embedding vector (E).
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.randn(num_embeddings, embedding_dim) * 0.02)

    def forward(self, context: torch.Tensor, B: int, ctx_size: int) -> torch.Tensor:
        """Gather embeddings for a batch of context windows.

        Args:
            context: [B * ctx_size] int32 token indices.
            B: Batch size.
            ctx_size: Context window length.

        Returns:
            [ctx_size * E, B] col-major output (Forge convention).
        """
        return _EmbeddingFn.apply(
            context, self.weight, B, ctx_size, self.embedding_dim)

    def extra_repr(self) -> str:
        return (f"num_embeddings={self.num_embeddings}, "
                f"embedding_dim={self.embedding_dim}")
