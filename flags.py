save_flag = False
bias_flag = False

weight_flag = False
grad_flag = True

# GGNN n_step
n_steps_flag = True
n_steps_set = 1

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
        print("{} weight".format(lin))
        if i <= 3:
            print(getattr(net, lin).weight.grad.numpy().T)
        if i == 4:
            gates = ["update_gate", "reset_gate", "tansform"]
            for gate in gates:
                print(gate)
                print(getattr(getattr(net, lin), gate)[0].weight.grad.numpy().T)
        if i == 5:
            print("weight_0", getattr(net, lin)[0].weight.grad.numpy().T)
            print("weight_2", getattr(net, lin)[2].weight.grad.numpy().T)
