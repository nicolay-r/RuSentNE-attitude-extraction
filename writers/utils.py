from arekit.common.data.input.writers.base import BaseWriter
from arekit.common.data.input.writers.tsv import TsvWriter

from writers.opennre_json import OpenNREJsonWriter


def create_writer_extension(writer):
    assert(isinstance(writer, BaseWriter))

    if isinstance(writer, OpenNREJsonWriter):
        return ".json"
    if isinstance(writer, TsvWriter):
        return ".tsv.gz"

    raise NotImplementedError()