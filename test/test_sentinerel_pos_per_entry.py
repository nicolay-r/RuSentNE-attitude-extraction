import sys
import unittest
from os.path import join, dirname, realpath

sys.path.append('../')

from arekit.common.entities.base import Entity
from arekit.contrib.source.sentinerel.io_utils import SentiNerelVersions
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
            print("IGNORED OR NESTED")
            return

        print("{}: ['{}', {}] {}".format(
            entity.ID, entity.Value, entity.Type,
            "[COMPOUND]" if isinstance(entity, BratCompoundEntity) else ""))

        if not isinstance(entity, BratCompoundEntity):
            return

        for child in entity.iter_childs():
            print("\t{}: ['{}',{}]".format(child.ID, child.Value, child.Type))

    @staticmethod
    def contains_entity_in_pairs(entity_id, pairs, top_entities, consider_all=False):
        assert(isinstance(pairs, dict))
        for pair_id, e_list in pairs.items():
            if len(e_list) < 2:
                continue

            if pair_id == entity_id:
                return pair_id

            if not consider_all:
                continue

            for e in e_list:
                if e.ID == entity_id:
                    return pair_id
                entity = top_entities[e.ID]
                if isinstance(entity, BratCompoundEntity):
                    for e_child in entity.iter_childs():
                        if e_child.ID == entity_id:
                            return pair_id

        return None

    def test_reading(self):
        train_filenames, test_filenames = FoldingFactory._read_train_test("../data/split_fixed.txt")

        relation_types = ["POSITIVE_TO", "NEGATIVE_TO"]
        consider_all = True

        total = {
            "POSITIVE_TO": 0,
            "NEGATIVE_TO": 0
        }

        for file_name in train_filenames + test_filenames:
            news = SentiNerelDocReader.read_document(file_name, doc_id=0, entities_to_ignore=[],
                                                     version=SentiNerelVersions.V21)

            top_entities = {}
            entity_bounds = {}
            pairs = {}
            for sentence in news.iter_sentences():
                prior_entity = None
                for entity, bound in sentence.iter_entity_with_local_bounds():

                    top_entities[entity.ID] = entity
                    entity_bounds[entity.ID] = bound

                    if entity.Type == "PROFESSION":
                        # OK, just collect it as a potential pair.
                        pairs[entity.ID] = [entity]
                    elif prior_entity is not None and prior_entity.Type == "PROFESSION" and entity.Type == "PERSON":
                        # Potential second component of the pair.
                        prior_bound = entity_bounds[prior_entity.ID]
                        prior_bound_end = prior_bound.Position + prior_bound.Length
                        if prior_bound_end == bound.Position - 1:
                            # check whether it is attached to the present entity.
                            pairs[prior_entity.ID].append(entity)

                    prior_entity = entity

            for brat_relation in news.Relations:
                if brat_relation.Type not in relation_types:
                    continue

                pair_id = self.contains_entity_in_pairs(
                    entity_id=brat_relation.TargetID, pairs=pairs,
                    top_entities=top_entities, consider_all=consider_all)

                if pair_id is None:
                    continue

                total[brat_relation.Type] += 1
                print("--- {}".format(file_name))
                print("R: {}->{} [{}]".format(brat_relation.SourceID, brat_relation.TargetID, brat_relation.Type))
                self.print_entity(top_entities[brat_relation.SourceID]
                                  if brat_relation.SourceID in top_entities else None)
                print("->")
                self.print_entity(pairs[pair_id][0])
                self.print_entity(pairs[pair_id][1])
                print("---")

        print("Total: {}".format(total))


if __name__ == '__main__':
    unittest.main()
