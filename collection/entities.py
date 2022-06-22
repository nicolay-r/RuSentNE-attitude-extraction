from arekit.common.entities.collection import EntityCollection
from arekit.common.synonyms import SynonymsCollection
from arekit.contrib.source.brat.annot import BratAnnotationParser
from arekit.contrib.source.brat.entities.entity import BratEntity

from collection.io_utils import CollectionIOUtils, CollectionVersions


class CollectionEntityCollection(EntityCollection):

    entities_to_ignore = [
        "EFFECT_NEG",
        "EFFECT_POS",
        "ARGUMENT_NEG",
        "ARGUMENT_POS",
        "EVENT"
    ]

    def __init__(self, contents, value_to_group_id_func):
        assert(isinstance(contents, dict))
        assert("entities" in contents)

        self.__dicard_entities = set(self.entities_to_ignore)

        contents["entities"] = [e for e in contents["entities"] if self.__keep_entity(e)]

        super(CollectionEntityCollection, self).__init__(
            entities=contents["entities"],
            value_to_group_id_func=value_to_group_id_func)

        self._sort_entities(key=lambda entity: entity.CharIndexBegin)

    def __keep_entity(self, entity):
        assert(isinstance(entity, BratEntity))
        return entity.Type not in self.__dicard_entities

    @staticmethod
    def get_synonym_group_index_or_add(synonyms, value):
        """ Allows a synonyms collection extensioning
        """
        assert(isinstance(synonyms, SynonymsCollection))

        if not synonyms.contains_synonym_value(value):
            synonyms.add_synonym_value(value)
        return synonyms.get_synonym_group_index(value)

    @classmethod
    def read_collection(cls, doc_id, synonyms, version=CollectionVersions.NO):
        assert(isinstance(synonyms, SynonymsCollection))
        assert(isinstance(doc_id, int))

        return CollectionIOUtils.read_from_zip(
            inner_path=CollectionIOUtils.get_annotation_innerpath(doc_id),
            process_func=lambda input_file: cls(
                contents=BratAnnotationParser.parse_annotations(input_file=input_file, encoding='utf-8-sig'),
                value_to_group_id_func=lambda value: cls.get_synonym_group_index_or_add(synonyms, value)),
            version=version)
