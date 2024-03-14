from .command import N, M, E, X, Z

class Pattern:
    def __init__(self, pattern):  
        self.cmd_list = []
        for cmd in pattern:
            t = cmd[0]
            match t:
                case 'N':
                    cmd = N(node=cmd[1])
                case 'M':
                    cmd = M(node=cmd[1], plane=cmd[2], angle=cmd[3], s_domain=cmd[4], t_domain=cmd[5])
                case 'E':
                    cmd = E(nodes=cmd[1])
                case 'X':
                    cmd = X(node=cmd[1], domain=cmd[2])
                case 'Z':
                    cmd = Z(node=cmd[1], domain=cmd[2])
                case _:
                    raise NameError(f"{t} command type not found.")
            self.cmd_list.append(cmd)

    def __repr__(self) -> str:
        return f"Pattern of {len(self.cmd_list)} commands."

    def print_pattern(self):
        for cmd in self.cmd_list:
            print(cmd)