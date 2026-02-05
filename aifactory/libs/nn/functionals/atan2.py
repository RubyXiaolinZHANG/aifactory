import torch

def robust_atan2(y, x, eps=1e-8):
    """
    A fully customized robust atan2 implementation including both forward and backward passes.
    Avoids any potential numerical instability.
    Forward: atan2(y, x) = arctan(y / x)
    Backward: ∂atan2/∂x = -y / (x² + y²); ∂atan2/∂y = x / (x² + y²)
    """
    # Forward pass: use native atan2 but handle special cases
    # For (0,0), define angle as 0 (or other suitable value)
    # angle = torch.atan2(y, x)

    # If both x and y are 0, set angle to 0
    # zero_mask = (x == 0) & (y == 0)
    # angle = torch.where(zero_mask, torch.tensor(0.0), angle)

    # Custom gradient implementation
    class RobustAtan2(torch.autograd.Function):
        @staticmethod
        def forward(ctx, y, x, eps):
            ctx.save_for_backward(y, x)
            ctx.eps = eps

            # Forward computation
            angle = torch.atan2(y, x)

            # Handle (0,0) case
            zero_mask = (x == 0) & (y == 0)
            angle = torch.where(zero_mask, torch.tensor(0.0), angle)

            return angle

        @staticmethod
        def backward(ctx, grad_output):
            y, x = ctx.saved_tensors
            eps = ctx.eps

            # Compute safe denominator
            denom = x ** 2 + y ** 2

            # Strategies to avoid division by zero
            # Strategy 1: return zero gradient for zero denominator cases
            mask = denom < eps
            safe_denom = torch.where(mask, torch.tensor(float('inf')), denom)

            # Compute gradients
            grad_x = -y / safe_denom * grad_output
            grad_y = x / safe_denom * grad_output

            # Set gradients at inf positions to 0
            grad_x = torch.where(torch.isinf(grad_x), torch.zeros_like(grad_x), grad_x)
            grad_y = torch.where(torch.isinf(grad_y), torch.zeros_like(grad_y), grad_y)

            return grad_y, grad_x, None

    return RobustAtan2.apply(y, x, eps)


# Test edge cases
def test_op():
    x = torch.tensor([0.0, 0.0, 1e-10, -1e-10], requires_grad=True)
    y = torch.tensor([0.0, 1e-10, 0.0, -1e-10], requires_grad=True)

    angle = robust_atan2(y, x, eps=1e-12)
    loss = angle.sum()
    loss.backward()

    print(f"Angles: {angle}")
    print(f"x gradients: {x.grad}")
    print(f"y gradients: {y.grad}")

if __name__ == "__main__":
    test_op()