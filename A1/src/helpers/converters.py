import torch 


def output_to_label(method):

    def to_hm(h, m):
        return torch.stack([h.float(), m.float()], dim=1)

    def scal_to_min(x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return torch.clamp(torch.round(x.squeeze(-1) * 60), 0, 59)

    def scal_to_hour(x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return torch.clamp(torch.round(x.squeeze(-1) * 12), 0, 11)

    def cls_total(out, verbose=False):
        # TODO: fix RuntimeError: CUDA error: device-side assert triggered
        if verbose:
            print("out.shape:", out.shape)
            print("out.dtype:", out.dtype)
            print("out.device:", out.device)
            print("out.min():", out.min().item())
            print("out.max():", out.max().item())
            print("out.isnan():", out.isnan().any().item())
            print("out.isinf():", out.isinf().any().item())
        total = out.argmax(dim=1) * (720 // out.shape[1])
        return to_hm(total // 60, total % 60)

    def scalar_to_lab(out):
        if out.dim() == 1:
            out = out.unsqueeze(-1)
        total = out.squeeze(-1) * 720
        return to_hm(torch.floor(total / 60), total % 60)

    def cls_cls(out):
        return to_hm(out[0].argmax(dim=1), out[1].argmax(dim=1))

    def cls_reg(out):
        return to_hm(out[0].argmax(dim=1), scal_to_min(out[1]))

    def reg_cls(out):
        return to_hm(scal_to_hour(out[0]), out[1].argmax(dim=1))

    def reg_reg(out):
        return to_hm(scal_to_hour(out[0]), scal_to_min(out[1]))

    def trig_to_lab(out):
        two_pi = 2 * torch.pi
        angle_h = torch.atan2(out[0][:, 0], out[0][:, 1]) % two_pi
        angle_m = torch.atan2(out[1][:, 0], out[1][:, 1]) % two_pi
        m = 60 * angle_m / two_pi
        h = torch.round(12 * angle_h / two_pi - m / 60) % 12
        return to_hm(h, m)
    
    def return_same(out):
        return out
    
    return {
        1: cls_total,
        2: scalar_to_lab,
        3: cls_cls,
        4: cls_reg,
        5: reg_cls,
        6: reg_reg,
        7: trig_to_lab,
    }.get(method, return_same)


def get_label_converter_fn(method: int, out_feat):
    """
    method 1: Convert labels to discrete minute bins.
    method 2: Convert labels to a scalar with two decimal places.
    method 3: Return labels unchanged.
    method 4: Classification hour + regression minute.
    method 5: Regression hour + classification minute.
    method 6: Regression hour + regression minute.
    method 7: Convert hours and minutes to sine/cosine pairs.
    """

    def same_label(label):
        return label

    def to_minutes(label):
        h, m = label[:, 0], label[:, 1]
        return h * 60 + m

    def convert_single_to_scalar(label):
        scaled = torch.where(label > 11, label / 60.0, label / 12.0)
        return (torch.round(scaled * 100) / 100).unsqueeze(-1)

    def convert_to_bins(label):
        total_mins = 720
        assert out_feat <= total_mins, "maximum 720 minutes possible"
        mins_per_bin = total_mins / out_feat
        assert mins_per_bin % 1 == 0, "provide valid number of output features"

        return (
            (to_minutes(label) // mins_per_bin) * mins_per_bin
        ).to(torch.int64)

    def convert_to_two_decimals(label):
        return (torch.round(to_minutes(label) / 720 * 100) / 100).unsqueeze(-1)

    def cls_reg(label):
        h, m = label[:, 0], label[:, 1]
        return h.to(torch.int64), convert_single_to_scalar(m)

    def reg_cls(label):
        h, m = label[:, 0], label[:, 1]
        return convert_single_to_scalar(h), m.to(torch.int64)

    def reg_reg(label):
        h, m = label[:, 0], label[:, 1]
        return convert_single_to_scalar(h), convert_single_to_scalar(m)

    def trig(label):
        h, m = label[:, 0], label[:, 1]
        angle_h = 2 * torch.pi * (h + m / 60) / 12
        angle_m = 2 * torch.pi * m / 60

        return (
            torch.stack([torch.sin(angle_h), torch.cos(angle_h)], dim=1),
            torch.stack([torch.sin(angle_m), torch.cos(angle_m)], dim=1),
        )

    return {
        1: convert_to_bins,
        2: convert_to_two_decimals,
        3: same_label,
        4: cls_reg,
        5: reg_cls,
        6: reg_reg,
        7: trig,
    }.get(method, same_label)
