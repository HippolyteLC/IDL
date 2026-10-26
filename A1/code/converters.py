import torch 


def output_to_label(method):

    def to_hm(h, m):
        return torch.stack([h.float(), m.float()], dim=1)

    def scal_to_min(x):
        return torch.clamp(torch.round(x.squeeze(-1) * 60), 0, 59)

    def scal_to_hour(x):
        return torch.clamp(torch.round(x.squeeze(-1) * 12), 0, 11)

    def cls_total(out):
        total = out.argmax(dim=1) * (720 // out.shape[1])
        return to_hm(total // 60, total % 60)

    def scalar_to_lab(out):
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
        return torch.round(scaled * 100) / 100

    def convert_to_bins(label):
        total_mins = 720
        assert out_feat <= total_mins, "maximum 720 minutes possible"
        mins_per_bin = total_mins / out_feat
        assert mins_per_bin % 1 == 0, "provide valid number of output features"

        return (
            torch.round(to_minutes(label) / mins_per_bin) * mins_per_bin
        ).to(torch.int64)

    def convert_to_two_decimals(label):
        return torch.round(to_minutes(label) / 720 * 100) / 100

    def cls_reg(label):
        h, m = label[:, 0], label[:, 1]
        return h.to(torch.int64), convert_single_to_scalar(m)

    def reg_cls(label):
        h, m = label[:, 0], label[:, 1]
        return convert_single_to_scalar(h).to(torch.int64), m.to(torch.int64)

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


# def get_label_converter_fn(method: int, out_feat):
#     """
#     method=0: classification,
#         convert labels to distinct bins based on num of output_feat 
#     method=1: regression,
#         convert labels to scalars. Given that we have 60 mins max in the labels,
#         we simply need 10^n values, given by n decimal places. in this case n=2. 
#     """
#     def same_label(label):
#         return label
#     def convert_single_to_scalar(label):
#         return torch.where(label > 11, torch.round(label / 60.0 * 100) / 100, torch.round(label / 12.0 * 100) / 100)
#     if method == 1:
#         def convert_to_bins(label):
#             total_mins = 12*60
#             assert out_feat <= total_mins, "maximum 720 minutes possible"
#             mins_per_bin = total_mins/out_feat
#             assert mins_per_bin%1 == 0, "provide valid number of output features"
#             h, m = label[:,0], label[:,1] 
#             label_in_mins = h*60 + m 
#             return (torch.round(label_in_mins/mins_per_bin) * mins_per_bin).to(torch.int64) 
#         return convert_to_bins
#     elif method == 2: 
#         def convert_to_two_decimals(label):
#             h, m = label[:,0], label[:,1] 
#             label_in_mins = h*60 + m 
#             return torch.tensor(torch.round((label_in_mins/720)*100) / 100)
#         return convert_to_two_decimals
#     elif method == 3:
#         return same_label
#     elif method == 4: 
#         def cls_reg(label):
#             h, m = label[:,0], label[:,1] 
#             return torch.tensor(h).to(torch.int64) , torch.tensor(convert_single_to_scalar(m))
#         return cls_reg
#     elif method == 5: 
#         def reg_cls(label):
#             h, m = label[:,0], label[:,1] 
#             return torch.tensor(convert_single_to_scalar(h)).to(torch.int64), torch.tensor(m).to(torch.int64) 
#         return reg_cls
#     elif method == 6:
#         def reg_reg(label):
#             h, m = label[:,0], label[:,1] 
#             return torch.tensor(convert_single_to_scalar(h)), torch.tensor(convert_single_to_scalar(m))
#         return reg_reg
#     elif method == 7:
#         def trig(label):
#             hours = label[:,0]
#             mins = label[:,1]
#             angle_hour = 2 * torch.pi * (hours+mins/60)/12
#             angle_min = 2 * torch.pi * mins/60
#             return torch.stack([
#                 torch.sin(angle_hour), 
#                 torch.cos(angle_hour),
#                 ], dim=1), \
#                 torch.stack([ 
#                 torch.sin(angle_min),
#                 torch.cos(angle_min)
#                 ], dim=1)
#         return trig
#     return same_label

# def output_to_label(method, out_feat):

#     def scalar_to_lab(output):
#         r = output * 720
#         h = torch.floor(r)
#         m = ((r - h) * 60).to(torch.int32)
#         return torch.stack([h, m])

#     def scal_min_to_int(output):
#         m = (output * 60).to(torch.int32)
#         return m
    
#     def scal_h_to_int(output):
#         h = (output * 12).to(torch.int32)
#         return h
#     def bin_to_lab(output):
#         mins_per_bin = (720 / len(output)).to(torch.int64)
#         return torch.argmax(output) * mins_per_bin
    
#     if method == 1:
#         return bin_to_lab
#     elif method == 2:
#         return scalar_to_lab
#     elif method == 3:
#         def split_to_whole(output):
#             out_1, out_2 = output[:,0], output[:,1]
#             return torch.stack([out_1, out_2])
#         return split_to_whole
#     elif method == 4:
#         def cls_reg(output):
#             out_1, out_2 = output[:,0], output[:,1]
#             out = torch.stack([bin_to_lab(out_1), scal_min_to_int(out_2)])
#             return out
#         return cls_reg
#     elif method == 5:
#         def reg_cls(output):
#             out_1, out_2 = output[:,0], output[:,1]
#             out = torch.stack([scal_h_to_int(out_1), bin_to_lab(out_2)])
#             return out
#         return reg_cls
#     elif method == 6:
#         def reg_reg(output):
#             out_1, out_2 = output[:,0], output[:,1]
#             out = torch.stack([scal_h_to_int(out_1), scal_min_to_int(out_2)])
#             return out
#         return reg_reg
#     elif method == 7:
#         def trig_to_angle(output):
#             y, x = output[:,0], output[:,1]
#             return torch.atan2(y,x)
#         def trig_to_lab(output):
#             out_1, out_2 = output[:,0], output[:,1]
#             angle_h, angle_m = trig_to_angle(out_1), trig_to_angle(out_2)
#             m = 60 * angle_m / (2 * torch.pi)
#             h = 12 * angle_h / (2 * torch.pi) - (m/60)
#             return torch.stack([h, m])
#         return trig_to_lab
#     def return_same(output):
#         return output
#     return return_same