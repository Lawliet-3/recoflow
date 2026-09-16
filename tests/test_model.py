import torch

from recoflow.model import MatrixFactorization, TwoTowerModel, bpr_loss, in_batch_softmax_loss


def test_two_tower_combines_history_and_metadata():
    model = TwoTowerModel(
        num_users=3,
        num_items=4,
        metadata_cardinalities=[3, 4],
        embedding_dim=8,
    )
    users = torch.tensor([0, 1, 2])
    items = torch.tensor([1, 2, 3])
    histories = torch.tensor([[0, 0], [0, 1], [1, 2]])
    masks = histories != 0
    metadata = torch.tensor([[1, 2], [2, 1], [1, 3]])
    logits = model(users, items, histories, masks, metadata)
    assert logits.shape == (3, 3)
    assert torch.isfinite(in_batch_softmax_loss(logits))


def test_matrix_factorization_bpr_loss_is_finite():
    model = MatrixFactorization(num_users=2, num_items=3, embedding_dim=4)
    positive = model.score(torch.tensor([0, 1]), torch.tensor([1, 2]))
    negative = model.score(torch.tensor([0, 1]), torch.tensor([3, 1]))
    assert torch.isfinite(bpr_loss(positive, negative))
