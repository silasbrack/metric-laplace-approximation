import copy

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
                    # h_k = torch.bmm(diag_elements.unsqueeze(2), feature_map_k2)
                    # h_k = h_k.view(bs, -1)
                    # if net[k].bias is not None:
                    #     h_k = torch.cat([h_k, diag_elements], dim=1)

                    # h_k = torch.einsum("bii,bj,bj->bij", tmp, feature_maps[k], feature_maps[k])
                    diag_elements = torch.einsum("bii->bi", tmp)
                    h_k = torch.einsum("bi,bj,bj->bij", diag_elements, feature_maps[k], feature_maps[k])
                    # feature_map_k2 = torch.einsum("bi,bi->bi", feature_maps[k], feature_maps[k]).unsqueeze(1)
                    # h_k = torch.einsum("bij,bkl->bil", diag_elements.unsqueeze(2), feature_map_k2)
                    h_k = h_k.view(bs, -1)
                    if net[k].bias is not None:
                        h_k = torch.cat([h_k, diag_elements], dim=1)

                    # in_shape = feature_maps[k].shape[1]
                    # out_shape = feature_maps[k+1].shape[1]
                    # jacobian_phi = feature_maps[k].unsqueeze(1).expand((bs, out_shape, in_shape))
                    # temp_diag = torch.diagonal(tmp, dim1=1, dim2=2)
                    # h_k = torch.einsum("bij,bjk,bkl->bij", jacobian_phi, tmp, jacobian_phi)
                    # h_k = torch.flatten(h_k, start_dim=1)
                    # if net[k].bias is not None:
                    #     h_k = torch.cat([h_k, temp_diag], dim=1)

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
                        (feature_maps[k + 1] > 0).float()
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
    # keep track of running sum
    H_running_sum = torch.zeros_like(parameters_to_vector(model.parameters()))
    counter = 0

    feature_maps = []
    def fw_hook_get_latent(module, input, output):
        feature_maps.append(output.detach())
    for k in range(len(model)):
        model[k].register_forward_hook(fw_hook_get_latent)

    # LOOP
    # iter_loader = iter(dataloader)
    # x1, y1 = next(iter_loader)
    # x2, y2 = next(iter_loader)

    feature_maps = []
    model(x1)
    feature_maps1 = copy.copy(feature_maps)
    feature_maps = []
    model(x2)
    feature_maps2 = copy.copy(feature_maps)
    feature_maps = None

    bs = x1.shape[0]
    feature_maps1 = [x1] + feature_maps1
    feature_maps2 = [x2] + feature_maps2

    # Saves the product of the Jacobians wrt layer input
    tmp1 = torch.diag_embed(torch.ones(bs, output_size, device=x1.device))
    tmp2 = torch.diag_embed(torch.ones(bs, output_size, device=x1.device))
    tmp3 = torch.diag_embed(torch.ones(bs, output_size, device=x1.device))

    H = []
    with torch.no_grad():
        for k in range(len(model) - 1, -1, -1):
            # if isinstance(model[k], torch.nn.Linear):
            #     tmp1 = torch.zeros((bs, model[k].weight.shape[0], model[k].weight.shape[0]))
            #     tmp2 = torch.zeros((bs, model[k].weight.shape[0], model[k].weight.shape[0]))
            #     tmp3 = torch.zeros((bs, model[k].weight.shape[0], model[k].weight.shape[0]))
            # else:
            #     tmp1 = torch.zeros((bs, model[k-1].weight.shape[0], model[k-1].weight.shape[0]))
            #     tmp2 = torch.zeros((bs, model[k-1].weight.shape[0], model[k-1].weight.shape[0]))
            #     tmp3 = torch.zeros((bs, model[k-1].weight.shape[0], model[k-1].weight.shape[0]))

            if isinstance(model[k], torch.nn.Linear):
                h1 = torch.einsum("bii,bj,bj->bij", tmp1, feature_maps1[k], feature_maps1[k]).view(bs, -1)
                h2 = torch.einsum("bii,bj,bj->bij", tmp2, feature_maps2[k], feature_maps2[k]).view(bs, -1)
                h3 = torch.einsum("bii,bj,bj->bij", tmp3, feature_maps1[k], feature_maps2[k]).view(bs, -1)
                if model[k].bias is not None:
                    h1 = torch.cat([h1, torch.einsum("bii->bi", tmp1)], dim=1)
                    h2 = torch.cat([h2, torch.einsum("bii->bi", tmp2)], dim=1)
                    h3 = torch.cat([h3, torch.einsum("bii->bi", tmp3)], dim=1)

                h_k = h1 + h2 - 4 * h3
                # h_k = torch.zeros((bs, model[k].weight.shape[0], model[k].weight.shape[1] + 1)).view(bs, -1)

                H = [h_k] + H

            if k == 0:
                break

            # Calculate the Jacobian wrt to the inputs
            if isinstance(model[k], torch.nn.Linear):
                jacobian_x1 = model[k].weight.expand((bs, *model[k].weight.shape))
                jacobian_x2 = jacobian_x1
            elif isinstance(model[k], torch.nn.Tanh):
                jacobian_x1 = torch.diag_embed(torch.ones(feature_maps1[k + 1].shape, device=x1.device) - feature_maps1[k + 1] ** 2)
                jacobian_x2 = torch.diag_embed(torch.ones(feature_maps2[k + 1].shape, device=x1.device) - feature_maps2[k + 1] ** 2)
            elif isinstance(model[k], torch.nn.ReLU):
                jacobian_x1 = torch.diag_embed((feature_maps1[k + 1] > 0).float())
                jacobian_x2 = torch.diag_embed((feature_maps2[k + 1] > 0).float())
            else:
                raise NotImplementedError

            # TODO: make more efficent by using row vectors
            # Right now this is 96% of our runtime
            tmp1 = torch.einsum("bnm,bnj,bjk->bmk", jacobian_x1, tmp1, jacobian_x1)
            tmp2 = torch.einsum("bnm,bnj,bjk->bmk", jacobian_x2, tmp2, jacobian_x2)
            tmp3 = torch.einsum("bnm,bnj,bjk->bmk", jacobian_x1, tmp3, jacobian_x2)

        counter += len(torch.cat(H, dim=1))
        H_running_sum += torch.cat(H, dim=1).sum(0)

    return H_running_sum
