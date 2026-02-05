import torch


class SafeGradientMagnitude:
    """Safe gradient magnitude calculator"""

    def __init__(self, eps=1e-8, clip_value=1e6):
        self.eps = eps
        self.clip_value = clip_value

    def __call__(self, grad_x, grad_y):
        """Calculate magnitude of gradient vector"""
        # 1. Clean input gradients
        grad_x_clean = self._clean_gradient(grad_x)
        grad_y_clean = self._clean_gradient(grad_y)

        # 2. Calculate squares
        x_sq = self._safe_square(grad_x_clean)
        y_sq = self._safe_square(grad_y_clean)

        # 3. Calculate sum
        sum_sq = x_sq + y_sq

        # 4. Square root
        magnitude = torch.sqrt(torch.clamp(sum_sq, min=self.eps))

        return magnitude

    def _clean_gradient(self, grad):
        """Clean abnormal values from gradients"""
        if grad is None:
            return torch.tensor(0.0)

        # Replace NaN
        if torch.isnan(grad).any():
            grad = torch.where(torch.isnan(grad), torch.zeros_like(grad), grad)

        # Clip extreme values
        grad = torch.clamp(grad, min=-self.clip_value, max=self.clip_value)

        return grad

    def _safe_square(self, x):
        """Safe squaring operation"""

        class SafeSquare(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, clip_value):
                ctx.save_for_backward(x)
                ctx.clip_value = clip_value

                # Forward pass: simple square
                result = x ** 2

                # Prevent overflow
                result = torch.clamp(result, max=clip_value ** 2)

                return result

            @staticmethod
            def backward(ctx, grad_output):
                x, = ctx.saved_tensors
                clip_value = ctx.clip_value

                # Gradient: 2 * x
                grad = 2 * x * grad_output

                # Clean abnormal gradients
                grad = torch.where(
                    torch.isnan(grad) | torch.isinf(grad),
                    torch.zeros_like(grad),
                    grad
                )

                # Gradient clipping
                grad = torch.clamp(grad, min=-clip_value, max=clip_value)

                return grad, None

        return SafeSquare.apply(x, self.clip_value)


def test_op():
    # Usage example
    magnitude_calculator = SafeGradientMagnitude(eps=1e-8, clip_value=1e6)

    # Test with problematic gradients
    grad_x = torch.tensor([1.0, float('nan'), float('inf'), 3.0], requires_grad=True)
    grad_y = torch.tensor([2.0, 3.0, 4.0, float('-inf')], requires_grad=True)

    magnitude = magnitude_calculator.compute(grad_x, grad_y)
    print(f"Gradient magnitude: {magnitude}")

    # If backward propagation is needed
    loss = magnitude.sum()
    try:
        loss.backward()
        print("✅ Backward propagation successful")
        print(f"grad_x.grad: {grad_x.grad}")
    except RuntimeError as e:
        print(f"❌ Backward propagation failed: {e}")


if __name__ == "__main__":
    test_op()