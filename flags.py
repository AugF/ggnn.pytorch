save_flag = False
bias_flag = False

updated_weight_flag = False
grad_flag = True
forward_flag = False

# GGNN n_step
n_steps_flag = True
n_steps_set = 4

n_lr_flag = True
n_lr_set = 0.01

# train times
sing_step_flag = True


def weight_print(net):
    linears = ["in_0", "in_1", "out_0", "out_1", "propogator", "out"]
    for i, lin in enumerate(linears):
        print("{} weight".format(lin))
        if i <= 3:
            print(getattr(net, lin).weight.detach().numpy().T)
        if i == 4:
            gates = ["update_gate", "reset_gate", "tansform"]
            for gate in gates:
                print(gate)
                print(getattr(getattr(net, lin), gate)[0].weight.detach().numpy().T)
        if i == 5:
            print("weight_0", getattr(net, lin)[0].weight.detach().numpy().T)
            print("weight_2", getattr(net, lin)[2].weight.detach().numpy().T)


def grad_print(net):
    linears = ["in_0", "in_1", "out_0", "out_1", "propogator", "out"]
    for i, lin in enumerate(linears):
        print("{} grad weight".format(lin))
        if i <= 3:
            print(getattr(net, lin).weight.grad.numpy().T)
        if i == 4:
            gates = ["update_gate", "reset_gate", "tansform"]
            for gate in gates:
                print(gate + " grad")
                print(getattr(getattr(net, lin), gate)[0].weight.grad.numpy().T)
        if i == 5:
            print("grad weight_0", getattr(net, lin)[0].weight.grad.numpy().T)
            print("grad weight_2", getattr(net, lin)[2].weight.grad.numpy().T)


def numerical_grad_2d(f, X, h=1e-5):
    """under the very small number, h should be large"""
    grad = np.zeros(X.shape)
    m, n = X.shape
    for i in range(m):
        for j in range(n):
            X[i, j] += h
            loss1 = f(X)
            X[i, j] -= (2.0*h)
            loss2 = f(X)
            grad[i, j] = (loss1 - loss2) / (2.0*h)
            X[i, j] += h
    return grad

def save_grad(name, grads):
    def hook(grad):
        grads[name] = grad
    return hook