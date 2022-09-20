import sys
import unittest
from os.path import join, dirname, realpath

sys.path.append('../')

from arekit.common.entities.base import Entity
from arekit.contrib.source.brat.entities.compound import BratCompoundEntity
from arekit.contrib.source.sentinerel.reader import SentiNerelDocReader

from SentiNEREL.folding.factory import FoldingFactory


class TestNestedRelations(unittest.TestCase):

    current_dir = dirname(realpath(__file__))
    output_dir = join(current_dir, "_out")

    @staticmethod
    def print_entity(entity):
        assert(isinstance(entity, Entity) or entity is None)

        if entity is None:
            print("IGNORED")
            return

        print("{}: ['{}', {}] {}".format(
            entity.ID, entity.Value, entity.Type,
            "[COMPOUND]" if isinstance(entity, BratCompoundEntity) else ""))

        if not isinstance(entity, BratCompoundEntity):
            return

        for child in entity.iter_childs():
            print("\t{}: ['{}',{}]".format(child.ID, child.Value, child.Type))

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

    def test_reading(self):
        train_filenames, test_filenames = FoldingFactory._read_train_test("../data/split_fixed.txt")

        total = 0

        for file_name in train_filenames + test_filenames:
            news = SentiNerelDocReader.read_document(file_name, doc_id=0, entities_to_ignore=[])

            top = {}
            nested = {}
            for sentence in news.iter_sentences():
                for entity, bound in sentence.iter_entity_with_local_bounds():
                    top[entity.ID] = entity
                    if isinstance(entity, BratCompoundEntity):
                        for e in entity.iter_childs():
                            assert(isinstance(e.ID, int))
                            nested[e.ID] = e

            for brat_relation in news.Relations:
                if brat_relation.Type not in ["POSITIVE_TO", "NEGATIVE_TO"]:
                    continue

                if brat_relation.SourceID in nested or brat_relation.TargetID in nested:
                    e_source = TestNestedRelations.find_entity(brat_relation.SourceID, top=top)
                    e_target = TestNestedRelations.find_entity(brat_relation.TargetID, top=top)
                    print("--- {}".format(file_name))
                    total += 1
                    print("Relation: {}->{} [{}]".format(brat_relation.SourceID, brat_relation.TargetID, brat_relation.Type))
                    self.print_entity(e_source)
                    self.print_entity(e_target)
                    print("---")

        print("Total: {}".format(total))


if __name__ == '__main__':
    unittest.main()
