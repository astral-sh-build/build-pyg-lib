import operator
from collections.abc import Callable
from importlib.metadata import version

import pyg_lib
import pytest
import torch


@pytest.fixture(scope="module")
def device() -> torch.device:
    assert torch.cuda.is_available(), "The tests must run on a CUDA GPU"
    return torch.device("cuda")


def test_published_cuda_wheel(device: torch.device) -> None:
    assert version("pyg-lib") == "0.9.0+cu.12.8.torch.2.10"
    assert torch.__version__ == "2.10.0+cu128"
    assert torch.version.cuda == "12.8"
    assert pyg_lib.cuda_version() == 12080
    assert torch.cuda.get_device_name(device)


@pytest.mark.parametrize("transposed", [False, True])
def test_grouped_matmul(device: torch.device, transposed: bool) -> None:
    torch.manual_seed(0)
    inputs = [
        torch.randn((5, 16), device=device, requires_grad=True),
        torch.randn((3, 9), device=device, requires_grad=True),
    ]
    if transposed:
        weight_storage = [
            torch.randn((8, 16), device=device, requires_grad=True),
            torch.randn((6, 9), device=device, requires_grad=True),
        ]
        weights = [weight.t() for weight in weight_storage]
    else:
        weight_storage = [
            torch.randn((16, 8), device=device, requires_grad=True),
            torch.randn((9, 6), device=device, requires_grad=True),
        ]
        weights = weight_storage

    biases = [
        torch.randn(8, device=device, requires_grad=True),
        torch.randn(6, device=device, requires_grad=True),
    ]

    actual = pyg_lib.ops.grouped_matmul(inputs, weights, biases)
    expected = [
        input @ weight + bias
        for input, weight, bias in zip(inputs, weights, biases, strict=True)
    ]

    assert len(actual) == len(expected)
    for actual_group, expected_group in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_group, expected_group)

    sum(output.sum() for output in actual).backward()

    for tensor in [*inputs, *weight_storage, *biases]:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_segment_matmul(device: torch.device) -> None:
    torch.manual_seed(0)
    inputs = torch.randn((8, 16), device=device, requires_grad=True)
    boundaries = torch.tensor([0, 5, 8], device=device)
    weights = torch.randn((2, 16, 8), device=device, requires_grad=True)
    biases = torch.randn((2, 8), device=device, requires_grad=True)

    actual = pyg_lib.ops.segment_matmul(inputs, boundaries, weights, biases)
    expected = torch.cat(
        [
            inputs[:5] @ weights[0] + biases[0],
            inputs[5:] @ weights[1] + biases[1],
        ]
    )

    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()

    for tensor in [inputs, weights, biases]:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


@pytest.mark.parametrize(
    ("name", "reference"),
    [
        ("sampled_add", operator.add),
        ("sampled_sub", operator.sub),
        ("sampled_mul", operator.mul),
        ("sampled_div", operator.truediv),
    ],
)
def test_sampled_operations(
    device: torch.device,
    name: str,
    reference: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> None:
    left = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        device=device,
        requires_grad=True,
    )
    right = torch.tensor(
        [[2.0, 4.0], [3.0, 5.0], [7.0, 11.0], [13.0, 17.0]],
        device=device,
        requires_grad=True,
    )
    left_index = torch.tensor([0, 2, 1, 0], device=device)
    right_index = torch.tensor([3, 1, 2, 0], device=device)

    actual = getattr(pyg_lib.ops, name)(left, right, left_index, right_index)
    expected = reference(left[left_index], right[right_index])

    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()

    assert left.grad is not None
    assert right.grad is not None
    assert torch.isfinite(left.grad).all()
    assert torch.isfinite(right.grad).all()


@pytest.mark.parametrize("format", ["coo", "csr"])
@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_segment_reductions(
    device: torch.device,
    format: str,
    reduction: str,
) -> None:
    values = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
        device=device,
        requires_grad=True,
    )

    if format == "coo":
        index = torch.tensor([0, 0, 2, 2, 2], device=device)
    else:
        index = torch.tensor([0, 2, 2, 5], device=device)

    actual = getattr(pyg_lib.ops, f"segment_{reduction}_{format}")(
        values,
        index,
    )

    if reduction == "sum":
        expected = torch.tensor(
            [[4.0, 6.0], [0.0, 0.0], [21.0, 24.0]],
            device=device,
        )
    else:
        expected = torch.tensor(
            [[2.0, 3.0], [0.0, 0.0], [7.0, 8.0]],
            device=device,
        )

    torch.testing.assert_close(actual, expected)
    actual.sum().backward()

    assert values.grad is not None
    assert torch.isfinite(values.grad).all()


@pytest.mark.parametrize("format", ["coo", "csr"])
def test_segment_gather(device: torch.device, format: str) -> None:
    values = torch.tensor(
        [[2.0, 3.0], [100.0, 200.0], [7.0, 8.0]],
        device=device,
    )

    if format == "coo":
        index = torch.tensor([0, 0, 2, 2, 2], device=device)
    else:
        index = torch.tensor([0, 2, 2, 5], device=device)

    actual = getattr(pyg_lib.ops, f"gather_{format}")(values, index)
    expected = torch.tensor(
        [[2.0, 3.0], [2.0, 3.0], [7.0, 8.0], [7.0, 8.0], [7.0, 8.0]],
        device=device,
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("walk_length", [1, 5])
def test_random_walk(device: torch.device, walk_length: int) -> None:
    boundaries = torch.tensor([0, 2, 4, 6, 8], device=device)
    neighbors = torch.tensor([1, 3, 0, 2, 1, 3, 0, 2], device=device)
    seeds = torch.arange(4, device=device)

    actual = pyg_lib.sampler.random_walk(
        boundaries,
        neighbors,
        seeds,
        walk_length,
    )

    assert actual.shape == (4, walk_length + 1)
    torch.testing.assert_close(actual[:, 0], seeds)

    distances = (actual[:, 1:] - actual[:, :-1]).abs()
    assert torch.all((distances == 1) | (distances == 3))
