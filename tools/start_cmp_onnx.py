import onnx
import onnx_graphsurgeon as gs

onnx_file_1 = 'D:/Program/ToGit/xiaomi/aifactory/model_zoo/ainr/models/onnx/ainr_unet_backbone_org.onnx'
onnx_file_2 = 'D:/Program/ToGit/xiaomi/aifactory/model_zoo/ainr/models/onnx/model_ainr_unet_backbone_ex.onnx'

graph_1 = gs.import_onnx(onnx.load(onnx_file_1))
graph_2 = gs.import_onnx(onnx.load(onnx_file_2))

assert len(graph_1.nodes) == len(graph_2.nodes)

for node_1, node_2 in zip(graph_1.nodes, graph_2.nodes):
    assert node_1.op == node_2.op, "node1: {}, node2:{}".format(node_1.op, node_2.op)
    if node_1.op == "Conv":
        assert node_1.inputs[1].shape == node_2.inputs[1].shape, "node1 weight: {}, node2 weight:{}".format(
            node_1.inputs[1].shape, node_2.inputs[1].shape)
        assert node_1.inputs[2].shape == node_2.inputs[2].shape, "node1 weight: {}, node2 weight:{}".format(
            node_1.inputs[2].shape, node_2.inputs[2].shape)
