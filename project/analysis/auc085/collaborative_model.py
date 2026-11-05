"""Protected TabM plus query-excluded, cross-domain collaborative attention."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from modern_architectures import TabMFieldModel
from bounded_id_models import bound_field_vectors
from collaborative_memory import QueryExcludedMemory


class NeighborAttention(nn.Module):
    def __init__(self, dimension, hidden=32):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dimension * 4, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, history, query, mask, pooling="attention"):
        values, expanded = history.float(), query.float()[:, None].expand_as(history)
        if pooling == "mean":
            return ((values * mask[..., None]).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)).to(history.dtype)
        if pooling != "attention":
            raise ValueError(f"Unknown neighbor pooling {pooling}")
        interaction = torch.cat([values, expanded, values - expanded, values * expanded], dim=-1)
        scores = self.score(interaction).squeeze(-1).float() + (values * expanded).sum(dim=-1)
        scores = scores.masked_fill(~mask, torch.finfo(torch.float32).min)
        weights = scores.softmax(dim=-1) * mask.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return (values * weights[..., None]).sum(dim=1).to(history.dtype)


class CollaborativeAttentionTabM(TabMFieldModel):
    def __init__(self, *args, memory_mode="neighbors", max_neighbors=16, positive_edge_count=0,
                 metadata_fields=None, source_dim=32, graph_dim=32, **kwargs):
        super().__init__(*args, **kwargs)
        if memory_mode not in ["none", "neighbors", "degrees", "mean"]:
            raise ValueError(f"Unknown collaborative mode {memory_mode}")
        self.memory_mode, self.graph_dim = memory_mode, graph_dim
        self.metadata_fields = list(metadata_fields or ["adv_prim_id", "spread_app_id"])
        self.user_context_fields = [name for name in ["age", "sex", "residence", "device_name", "u_newsCatInterestsST", "feed_long_interest"]
                                   if name in self.encoder.categorical_fields + self.encoder.sequence_fields]
        if any(name not in self.encoder.categorical_fields for name in self.metadata_fields):
            raise ValueError("All task prototype fields must be available categorical inputs")
        n_users = self.encoder.embeddings[self.encoder.group_for_field["user_id"]].num_embeddings
        n_tasks = self.encoder.embeddings[self.encoder.group_for_field["task_id"]].num_embeddings
        dimension = self.encoder.embeddings[self.encoder.group_for_field["user_id"]].embedding_dim
        self.memory = QueryExcludedMemory(n_users, n_tasks, max_neighbors, positive_edge_count, len(self.metadata_fields))
        self.register_buffer("source_user_vectors", torch.zeros(n_users, source_dim))
        self.register_buffer("source_positive_count", torch.zeros(n_users))
        self.collaborative = nn.ModuleDict({
            "source_projection": nn.Linear(source_dim, graph_dim, bias=False),
            "user_context": nn.Sequential(nn.Linear(source_dim + len(self.user_context_fields) * dimension, 64), nn.ReLU(), nn.Linear(64, graph_dim)),
            "task_projection": nn.Sequential(nn.Linear(len(self.metadata_fields) * dimension, 64), nn.ReLU(), nn.Linear(64, graph_dim)),
            "user_attention": NeighborAttention(graph_dim),
            "task_attention": NeighborAttention(graph_dim),
            "residual": nn.Sequential(nn.Linear(graph_dim * 8 + 4, 128), nn.ReLU(), nn.Dropout(.1), nn.Linear(128, self.k)),
            "gate": nn.Linear(graph_dim * 2 + 6, 1),
        })
        if source_dim == graph_dim:
            nn.init.eye_(self.collaborative["source_projection"].weight)
        for name in ["user_context", "residual"]:
            nn.init.zeros_(self.collaborative[name][-1].weight)
            nn.init.zeros_(self.collaborative[name][-1].bias)
        nn.init.zeros_(self.collaborative["gate"].weight)
        nn.init.constant_(self.collaborative["gate"].bias, -2.)
        edges = kwargs.get("numeric_bin_edges")
        self.register_buffer("numeric_lower", torch.tensor([edge[0] for edge in edges], dtype=torch.float32) if edges else None)
        self.register_buffer("numeric_upper", torch.tensor([edge[-1] for edge in edges], dtype=torch.float32) if edges else None)
        self.user_column = self.encoder.categorical_fields.index("user_id")
        self.task_column = self.encoder.categorical_fields.index("task_id")

    def train(self, mode=True):
        super().train(mode)
        # The new branch can use dropout; the protected original function cannot.
        for module in [self.encoder, self.backbone, self.head, self.numeric_embeddings]:
            if module is not None:
                module.eval()
        return self

    def _base_logits(self, fields, numeric):
        if self.numeric_lower is not None:
            numeric = numeric.clamp(min=self.numeric_lower, max=self.numeric_upper)
        dense = numeric if self.numeric_embeddings is None else self.numeric_embeddings(numeric).flatten(1)
        features = torch.cat([fields.flatten(1), dense], dim=-1)
        hidden = self.backbone(features[:, None].expand(-1, self.k, -1))
        return self.head(hidden).squeeze(-1)

    def _augment(self, categories, fields, base_logits):
        # Ensemble heads are column-strided while cached logits are row-major.
        # Canonicalize before reductions so AMP gate rounding cannot depend on
        # the inference route. Cached training inputs already have this layout.
        base_logits = base_logits.float().contiguous()
        names = self.encoder.categorical_fields + self.encoder.sequence_fields
        lookup = dict(zip(names, fields.unbind(dim=1)))
        users, tasks = categories[:, self.user_column], categories[:, self.task_column]
        neighborhood = self.memory.query(users, tasks)
        if self.memory_mode == "none":
            neighborhood = {name: torch.zeros_like(value) for name, value in neighborhood.items()}
        other_tasks, other_users = neighborhood["user_tasks"], neighborhood["task_users"]
        source = self.source_user_vectors[users]
        context = torch.cat([source] + [lookup[name] for name in self.user_context_fields], dim=-1)
        query_user = F.normalize((self.collaborative["source_projection"](source) + self.collaborative["user_context"](context)).float(), dim=-1, eps=1e-8)
        current_metadata = torch.cat([lookup[name] for name in self.metadata_fields], dim=-1)
        query_task = F.normalize(self.collaborative["task_projection"](current_metadata).float(), dim=-1, eps=1e-8)
        other_source = self.source_user_vectors[other_users]
        user_vectors = F.normalize(self.collaborative["source_projection"](other_source).float(), dim=-1, eps=1e-8)
        user_mask = other_users.ge(2) & self.source_positive_count[other_users].gt(0)
        prototype = self.memory.task_metadata[other_tasks]
        metadata_vectors = [self.encoder.embeddings[self.encoder.group_for_field[name]](prototype[..., index])
                            for index, name in enumerate(self.metadata_fields)]
        task_vectors = F.normalize(self.collaborative["task_projection"](torch.cat(metadata_vectors, dim=-1)).float(), dim=-1, eps=1e-8)
        task_mask = other_tasks.ge(2)
        if self.memory_mode == "degrees":
            pooled_users, pooled_tasks = torch.zeros_like(query_user), torch.zeros_like(query_task)
        else:
            pooling = "mean" if self.memory_mode == "mean" else "attention"
            pooled_users = self.collaborative["user_attention"](user_vectors, query_user, user_mask, pooling=pooling)
            pooled_tasks = self.collaborative["task_attention"](task_vectors, query_task, task_mask, pooling=pooling)
        support = torch.stack([(neighborhood["user_degree"].log1p() / math.log(self.memory.max_neighbors + 1.)).clamp_max(3.),
                               (neighborhood["task_degree"].log1p() / math.log(self.memory.max_neighbors + 1.)).clamp_max(3.),
                               task_mask.float().mean(dim=1), user_mask.float().mean(dim=1)], dim=-1)
        representations = torch.cat([query_user, query_task, pooled_users, pooled_tasks,
            query_user * pooled_users, query_task * pooled_tasks, query_user - pooled_users, query_task - pooled_tasks, support], dim=-1)
        residual = self.collaborative["residual"](representations)
        baseline_context = torch.stack([base_logits.float().mean(dim=-1), base_logits.float().std(dim=-1, unbiased=False)], dim=-1)
        gate = torch.sigmoid(self.collaborative["gate"](torch.cat([query_user, query_task, support, baseline_context], dim=-1)))
        return base_logits.float() + gate.float() * residual.float()

    def forward_with_base(self, categories, sequences, base_logits):
        fields, _, _ = self.encoder(categories, sequences)
        fields = bound_field_vectors(fields, self.field_groups, self.group_norm_caps)
        return self._augment(categories, fields, base_logits)

    def forward(self, categories, sequences, numeric):
        fields, _, _ = self.encoder(categories, sequences)
        fields = bound_field_vectors(fields, self.field_groups, self.group_norm_caps)
        return self._augment(categories, fields, self._base_logits(fields, numeric))


def initialize_collaborative_adapter(model, teacher_state):
    allowed = ("collaborative.", "memory.", "source_user_vectors", "source_positive_count")
    missing = set(model.state_dict()).difference(teacher_state)
    unexpected = set(teacher_state).difference(model.state_dict())
    if unexpected or any(not name.startswith(allowed) for name in missing):
        raise ValueError("Teacher does not exactly cover the protected backbone")
    model.load_state_dict(teacher_state, strict=False)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("collaborative."))
    model.eval()
    return {"loaded_base_tensors": len(teacher_state),
            "trainable_branch_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
