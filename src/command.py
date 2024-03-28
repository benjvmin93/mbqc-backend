from pydantic import BaseModel
from typing import Union, Literal

Node = int
Plane = Union[Literal["XY"], Literal["YZ"], Literal["XZ"]]


class Command(BaseModel):
    pass


class N(Command):
    node: Node

    def __str__(self) -> str:
        return f"N[{self.node}]"


class M(Command):
    node: Node
    plane: Plane
    angle: float
    s_domain: list[Node]
    t_domain: list[Node]
    vop: int

    def __str__(self) -> str:
        return f"M[{self.node}], plane={self.plane}, angle={self.angle}, s_domain={self.s_domain}, t_domain={self.t_domain}, vop={self.vop}"


class E(Command):
    nodes: tuple[Node, Node]

    def __str__(self) -> str:
        return f"E[{self.nodes}]"


class C(Command):
    node: Node
    domain: list[Node]


class X(C):
    pass

    def __str__(self) -> str:
        return f"X[{self.node}], domain={self.domain}"


class Z(C):
    pass

    def __str__(self) -> str:
        return f"Z[{self.node}], domain={self.domain}"
