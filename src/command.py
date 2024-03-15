from pydantic import BaseModel
from typing import Union, Literal

Node = int
Plane = Union[Literal["XY"], Literal["YZ"], Literal["XZ"]]

class Command(BaseModel):
    pass


class N(Command):
    node: Node


class M(Command):
    node: Node
    plane: Plane
    angle: float
    s_domain: list[Node]
    t_domain: list[Node]


class E(Command):
    nodes: tuple[Node, Node]


class C(Command):
    node: Node
    domain: list[Node]


class X(C):
    pass


class Z(C):
    pass
