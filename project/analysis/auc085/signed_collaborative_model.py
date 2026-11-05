"""Add a clicked-versus-never-clicked expert to a protected collaborative model."""
import math
import copy

import torch
from torch import nn
from torch.nn import functional as F

from collaborative_model import CollaborativeAttentionTabM, NeighborAttention
from collaborative_memory import QueryExcludedMemory


class SignedCollaborativeTabM(CollaborativeAttentionTabM):
    def __init__(self, *args, negative_edge_count=0, signed_mode="contrast", learn_queries=False, **kwargs):
        super().__init__(*args, **kwargs)
        if signed_mode not in ["positive", "contrast"]:
            raise ValueError(f"Unknown comparative collaboration mode {signed_mode}")
        self.signed_mode = signed_mode
        self.learn_queries = learn_queries
        d = self.graph_dim
        self.negative_memory = QueryExcludedMemory(self.memory.user_tasks.shape[0], self.memory.n_tasks,
            self.memory.max_neighbors, negative_edge_count, len(self.metadata_fields))
        self.signed = nn.ModuleDict({
            "user_attention": NeighborAttention(d), "task_attention": NeighborAttention(d),
            "residual": nn.Sequential(nn.Linear(d * 10 + 8, 128), nn.ReLU(), nn.Dropout(.1), nn.Linear(128, self.k)),
            "gate": nn.Linear(d * 2 + 10, 1)})
        if learn_queries:
            for name in ["source_projection", "user_context", "task_projection"]:
                self.signed[name] = copy.deepcopy(self.collaborative[name])
            self.signed["positive_user_attention"] = copy.deepcopy(self.collaborative["user_attention"])
            self.signed["positive_task_attention"] = copy.deepcopy(self.collaborative["task_attention"])
        nn.init.zeros_(self.signed["residual"][-1].weight)
        nn.init.zeros_(self.signed["residual"][-1].bias)
        nn.init.zeros_(self.signed["gate"].weight)
        nn.init.constant_(self.signed["gate"].bias, -2.)

    def train(self, mode=True):
        super().train(mode)
        self.collaborative.eval()
        return self

    def _augment(self, categories, fields, base_logits):
        positive_logits = super()._augment(categories, fields, base_logits).float().contiguous()
        lookup = dict(zip(self.encoder.categorical_fields + self.encoder.sequence_fields, fields.unbind(dim=1)))
        users, tasks = categories[:, self.user_column], categories[:, self.task_column]
        source = self.source_user_vectors[users]
        query_network = self.signed if self.learn_queries else self.collaborative
        context = torch.cat([source] + [lookup[name] for name in self.user_context_fields], dim=-1)
        query_user = F.normalize((query_network["source_projection"](source) + query_network["user_context"](context)).float(), dim=-1, eps=1e-8)
        query_task = F.normalize(query_network["task_projection"](torch.cat([lookup[name] for name in self.metadata_fields], dim=-1)).float(), dim=-1, eps=1e-8)
        positive = self.memory.query(users, tasks)
        negative = self.negative_memory.query(users, tasks)
        if self.signed_mode == "positive":
            negative = {name: torch.zeros_like(value) for name, value in negative.items()}

        def represent(neighborhood, storage, learned_attention):
            others_u, others_t = neighborhood["task_users"], neighborhood["user_tasks"]
            vector_u = F.normalize(query_network["source_projection"](self.source_user_vectors[others_u]).float(), dim=-1, eps=1e-8)
            mask_u = others_u.ge(2) & self.source_positive_count[others_u].gt(0)
            prototypes = storage.task_metadata[others_t]
            encoded = [self.encoder.embeddings[self.encoder.group_for_field[name]](prototypes[..., index])
                       for index, name in enumerate(self.metadata_fields)]
            vector_t = F.normalize(query_network["task_projection"](torch.cat(encoded, dim=-1)).float(), dim=-1, eps=1e-8)
            mask_t = others_t.ge(2)
            pool_u = learned_attention["user_attention"](vector_u, query_user, mask_u)
            pool_t = learned_attention["task_attention"](vector_t, query_task, mask_t)
            support = torch.stack([(neighborhood["user_degree"].log1p() / math.log(storage.max_neighbors + 1.)).clamp_max(3.),
                                   (neighborhood["task_degree"].log1p() / math.log(storage.max_neighbors + 1.)).clamp_max(3.),
                                   mask_t.float().mean(1), mask_u.float().mean(1)], dim=-1)
            return pool_u, pool_t, support

        positive_attention = {"user_attention": self.signed["positive_user_attention"], "task_attention": self.signed["positive_task_attention"]} if self.learn_queries else self.collaborative
        pu, pt, ps = represent(positive, self.memory, positive_attention)
        nu, nt, ns = represent(negative, self.negative_memory, self.signed)
        difference_u, difference_t = pu - nu, pt - nt
        feature = torch.cat([query_user, query_task, pu, pt, nu, nt, difference_u, difference_t,
                             query_user * difference_u, query_task * difference_t, ps, ns], dim=-1)
        residual = self.signed["residual"](feature)
        baseline = torch.stack([positive_logits.mean(-1), positive_logits.std(-1, unbiased=False)], dim=-1)
        gate = torch.sigmoid(self.signed["gate"](torch.cat([query_user, query_task, ps, ns, baseline], dim=-1)))
        return positive_logits + gate.float() * residual.float()


def initialize_signed_adapter(model, teacher_state):
    missing = set(model.state_dict()).difference(teacher_state)
    unexpected = set(teacher_state).difference(model.state_dict())
    if unexpected or any(not name.startswith(("negative_memory.", "signed.")) for name in missing):
        raise ValueError("Teacher does not exactly cover the existing collaborative model")
    model.load_state_dict(teacher_state, strict=False)
    if model.learn_queries:
        for name in ["source_projection", "user_context", "task_projection"]:
            model.signed[name].load_state_dict(model.collaborative[name].state_dict())
        model.signed["positive_user_attention"].load_state_dict(model.collaborative["user_attention"].state_dict())
        model.signed["positive_task_attention"].load_state_dict(model.collaborative["task_attention"].state_dict())
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("signed."))
    model.eval()
    return {"protected_teacher_tensors": len(teacher_state),
            "new_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
