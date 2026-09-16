import torch

from recoflow.model import TwoTowerModel, in_batch_softmax_loss


def test_two_tower_shapes_and_finite_loss():
    model = TwoTowerModel(num_users=3, num_items=4, embedding_dim=8)
    logits = model(torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3]))
    assert logits.shape == (3, 3)
    assert torch.isfinite(in_batch_softmax_loss(logits))

