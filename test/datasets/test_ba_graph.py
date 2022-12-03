from torch_geometric.datasets.generators import BAGraph, Motif


def test_base_graph():
    generator = BAGraph()
    data = generator.generate_base_graph()
    assert data.x.size() == (300, 10)
    assert data.edge_index.size(1) >= 1120


def test_graph_with_motif():
    motif = Motif('house')
    generator = BAGraph(num_nodes=300, num_motifs=80, motif=motif)
    data = generator.generate_base_graph()

    assert data.edge_index.size(1) >= 1120
    assert data.x.size() == (700, 10)
    assert data.y.size() == (700, )
    assert data.expl_mask.sum() == 60
    assert data.edge_label.sum() == 960
