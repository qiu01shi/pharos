"""用户自定义的 pharos 实体。

被 graphs/06_python_word_counter.yaml 这种 YAML 引用:
    type: python
    class: "my_entities:WordCounter"
"""
from pharos.core.entity import Entity, entity
from pharos.core.port import InputPort, OutputPort
from pharos.core.token import TypedValue


@entity
class WordCounter(Entity):
    """统计文本词数。"""
    ins = {"text": InputPort(name="text", accepted_types=["text"])}
    outs = {"count": OutputPort(name="count", accepted_types=["int"])}

    async def fire(self, ctx):
        for t in self.ins["text"].consume():
            n = len((t.value.payload or "").split())
            self.outs["count"].emit(TypedValue(type="int", payload=n))


@entity
class PrefixAdder(Entity):
    """给输入加前缀(prefix 通过 __init__ kwargs 注入)。"""
    ins = {"in": InputPort(name="in", accepted_types=["text"])}
    outs = {"out": OutputPort(name="out", accepted_types=["text"])}

    def __init__(self, node_id: str, prefix: str = ">>> "):
        super().__init__(node_id=node_id)
        self._prefix = prefix

    async def fire(self, ctx):
        for t in self.ins["in"].consume():
            self.outs["out"].emit(
                TypedValue(type="text", payload=self._prefix + t.value.payload)
            )