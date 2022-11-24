import json
import unittest
from tqdm import tqdm
from os import makedirs
from os.path import exists, join
from arekit.common.bound import Bound
from arekit.contrib.source.brat.entities.compound import BratCompoundEntity
from arekit.contrib.source.brat.sentence import BratSentence
from arekit.contrib.source.sentinerel.io_utils import SentiNerelVersions
from arekit.contrib.source.sentinerel.reader import SentiNerelDocReader
from SentiNEREL.folding.factory import FoldingFactory


class TestGraph(unittest.TestCase):

    @staticmethod
    def find_entity(requested_id, top):
        assert(isinstance(top, dict))
        for e_id, top_entity in top.items():
            if e_id == requested_id:
                return top_entity
            if isinstance(top_entity, BratCompoundEntity):
                for child in top_entity.iter_childs():
                    if child.ID == requested_id:
                        return top_entity

        return None

    def test(self, limit=None, output_dir="_out",
             rel_filter=lambda r_type: r_type in ["POSITIVE_TO", "NEGATIVE_TO"]):
        """ Serialize graph-based input JSONL for the SentiNEREL collection.
        """
        assert(callable(rel_filter))

        train_filenames, test_filenames = FoldingFactory._read_train_test("../../data/split_fixed.txt")

        if not exists(output_dir):
            makedirs(output_dir)

        with open(join(output_dir, "graph-sampled.jsonl"), "w") as json_out:

            # Keep all files of fixed amound by the pre-defined limit.
            files = (train_filenames + test_filenames)
            if limit is not None:
                files = files[:limit]

            for file_name in tqdm(files):
                news = SentiNerelDocReader.read_document(file_name, doc_id=0, entities_to_ignore=[],
                                                         version=SentiNerelVersions.V21)

                json_doc = {"filename": file_name, "input": "", "flavor": 0, "nodes": [], "edges": []}
                top = {}
                nested = {}

                for sentence in news.iter_sentences():
                    assert(isinstance(sentence, BratSentence))

                    json_doc["input"] += sentence.Text

                    for entity, root_bound in sentence.iter_entity_with_local_bounds():
                        assert(isinstance(root_bound, Bound))
                        top[entity.ID] = entity
                        if isinstance(entity, BratCompoundEntity):
                            for e in entity.iter_childs():
                                assert(isinstance(e.ID, int))
                                nested[e.ID] = e
                                json_doc["nodes"].append({
                                    "id": e.ID,
                                    "label": e.Type,
                                    "value": e.Value,
                                    "anchor": [],
                                })

                        json_doc["nodes"].append({
                            "id": entity.ID,
                            "label": entity.Type,
                            "value": json_doc["input"][entity.IndexBegin:entity.IndexEnd],
                            "anchor": [entity.IndexBegin, entity.IndexEnd],
                        })

                for brat_relation in news.Relations:
                    if not rel_filter(brat_relation.Type):
                        continue

                    e_source = TestGraph.find_entity(brat_relation.SourceID, top=top)
                    e_target = TestGraph.find_entity(brat_relation.TargetID, top=top)

                    if e_source is None or e_target is None:
                        continue

                    json_doc["edges"].append({
                        "source": e_source.ID,
                        "target": e_target.ID,
                        "label": brat_relation.Type
                    })

                # Dump information for the whole document.
                json.dump(json_doc, json_out, separators=(",", ":"), ensure_ascii=False, indent=4)
                json_out.write("\n")


if __name__ == '__main__':
    unittest.main()
