import torch
from torch.nn.utils import parameters_to_vector


def compute_hessian_rmse(dataloader, net, output_size):
    # keep track of running sum
    H_running_sum = torch.zeros_like(parameters_to_vector(net.parameters()))
    counter = 0

    feature_maps = []

    def fw_hook_get_latent(module, input, output):
        feature_maps.append(output.detach())

    for k in range(len(net)):
        net[k].register_forward_hook(fw_hook_get_latent)

    for x, y in dataloader:

        feature_maps = []
        yhat = net(x)

        bs = x.shape[0]
        feature_maps = [x] + feature_maps

        # Saves the product of the Jacobians wrt layer input
        tmp = torch.diag_embed(torch.ones(bs, output_size, device=x.device))

        H = []
        with torch.no_grad():
            for k in range(len(net) - 1, -1, -1):
                if isinstance(net[k], torch.nn.Linear):
                    # diag_elements = torch.diagonal(tmp, dim1=1, dim2=2)
                    # feature_map_k2 = (feature_maps[k] ** 2).unsqueeze(1)
                    # h_k = torch.bmm(diag_elements.unsqueeze(2), feature_map_k2)\
                    #     .view(bs, -1)
                    # if net[k].bias is not None:
                    #     h_k = torch.cat([h_k, diag_elements], dim=1)

                    jacobian_phi = feature_maps[k]
                    temp_diag = torch.diagonal(tmp, dim1=1, dim2=2)
                    h_k = torch.einsum("", jacobian_phi, temp_diag, jacobian_phi)

                    if net[k].bias is not None:
                        h_k = torch.cat([h_k, temp_diag], dim=1)

                    H = [h_k] + H

                if k == 0:
                    break

                # Calculate the Jacobian wrt to the inputs
                if isinstance(net[k], torch.nn.Linear):
                    jacobian_x = net[k].weight.expand((bs, *net[k].weight.shape))
                elif isinstance(net[k], torch.nn.Tanh):
                    jacobian_x = torch.diag_embed(
                        torch.ones(feature_maps[k + 1].shape, device=x.device)
                        - feature_maps[k + 1] ** 2
                    )
                elif isinstance(net[k], torch.nn.ReLU):
                    jacobian_x = torch.diag_embed(
                        (feature_maps[k] > 0).float()
                    )
                else:
                    raise NotImplementedError

                # TODO: make more efficent by using row vectors
                tmp = torch.einsum("bnm,bnj,bjk->bmk", jacobian_x, tmp, jacobian_x)

            counter += len(torch.cat(H, dim=1))
            H_running_sum += torch.cat(H, dim=1).sum(0)

    assert counter == dataloader.dataset.__len__()

    # compute mean over dataset
    # final_H = 1 / counter * H_running_sum
    final_H = H_running_sum

    return final_H


def compute_hessian_contrastive_batch(x1, x2, y, model, output_size, margin=0.1, hessian_structure="diag", agg="sum"):
    raise NotImplementedError
