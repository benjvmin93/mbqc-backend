from .command import N, M, E, X, Z
from graphix.graphix import Pattern


def get_cmd_list(pattern_cmd: list[list]):
    cmd_list = []
    for cmd in pattern_cmd:
        match cmd[0]:
            case "N":
                cmd = N(node=cmd[1])
            case "M":
                cmd = M(
                    node=cmd[1],
                    plane=cmd[2],
                    angle=cmd[3],
                    s_domain=cmd[4],
                    t_domain=cmd[5],
                )
            case "E":
                cmd = E(nodes=cmd[1])
            case "X":
                cmd = X(node=cmd[1], domain=cmd[2])
            case "Z":
                cmd = Z(node=cmd[1], domain=cmd[2])
            case _:
                raise NameError(f"{cmd[0]} command type not found.")
        """
        case 'C':
            cmd = C()
        """

        cmd_list.append(cmd)
    return cmd_list


class Pattern:
    def __init__(self, pattern: Pattern):
        self.cmd_list = get_cmd_list(list(pattern))
        self.Nnode = pattern.Nnode
        self.input_nodes = pattern.input_nodes
        self.output_nodes = pattern.output_nodes

    def __repr__(self) -> str:
        return f"Pattern of {len(self.cmd_list)} commands."

    def print_pattern(self):
        for cmd in self.cmd_list:
            print(cmd)
