import torch.nn as nn
import torch


class Yolo_head(nn.Module):
    def __init__(self, nC, anchors, stride):
        super(Yolo_head, self).__init__()

        self.__anchors = anchors
        self.__nA = len(anchors)
        self.__nC = nC
        self.__stride = stride


    def forward(self, p): # p:4*75*52*52
        bs, nG = p.shape[0], p.shape[-1] # bs:batchsize=4,nG:特征图尺寸=52
        # p:4*52*52*3*(5+20)
        # __nA:Anchor锚框的数量
        p = p.view(bs, self.__nA, 5 + self.__nC, nG, nG).permute(0, 3, 4, 1, 2)

        p_de = self.__decode(p.clone())

        return (p, p_de)

    # 1.生成锚框 2.调整锚框
    def __decode(self, p):
        batch_size, output_size = p.shape[:2] #bs:4,out:54

        device = p.device
        stride = self.__stride # 8/16/32
        anchors = (1.0 * self.__anchors).to(device)

        # 提取75维中的前两维,做调整参数,即x,y
        conv_raw_dxdy = p[:, :, :, :, 0:2]
        # 提取锚框宽高hw参数
        conv_raw_dwdh = p[:, :, :, :, 2:4]
        # 提取是否有目标置信度参数
        conv_raw_conf = p[:, :, :, :, 4:5]
        # 提取20维分类参数
        conv_raw_prob = p[:, :, :, :, 5:]

        # 生成一个[0,1,...,52]的点集y
        y = torch.arange(0, output_size).unsqueeze(1).repeat(1, output_size)
        # 生成一个[0,1,...,52]的点集x
        x = torch.arange(0, output_size).unsqueeze(0).repeat(output_size, 1)
        # 合并xy生成[(0,0),(0,1),...,(52,52)]一个坐标图
        grid_xy = torch.stack([x, y], dim=-1)
        # 复制3个坐标图
        grid_xy = grid_xy.unsqueeze(0).unsqueeze(3).repeat(batch_size, 1, 1, 3, 1).float().to(device)

        # 我感觉网络预测的不是锚框,而是锚框的缩放比、偏移量
        # 我感觉是通过缩放比、偏移量去还原真实的锚框,一共有3个坐标图,每个坐标图的一个点生成一个锚框,拼接所有坐标图
        # 对是否有物体置信度、目标类别进行激活一下,映射到概率区间
        # 最后将真实的坐标区间、各种值的概率拼在一起,组成最终的预测结果
        pred_xy = (torch.sigmoid(conv_raw_dxdy) + grid_xy) * stride
        pred_wh = (torch.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)
        pred_conf = torch.sigmoid(conv_raw_conf)
        pred_prob = torch.sigmoid(conv_raw_prob)
        pred_bbox = torch.cat([pred_xywh, pred_conf, pred_prob], dim=-1)

        # 我感觉推理阶段不需要一个点生成3个锚框,只需要生成1个锚框即可
        return pred_bbox.view(-1, 5 + self.__nC) if not self.training else pred_bbox




