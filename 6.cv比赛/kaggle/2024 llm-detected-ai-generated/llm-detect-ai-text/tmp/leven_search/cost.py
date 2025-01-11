from dataclasses import dataclass
from typing import Optional, List, Dict


class EditOp:
    DELETE = '[-]'
    ADD = '[+]'


@dataclass
class Edit:
    l1: EditOp | str
    l2: str

    def __repr__(self):
        if self.l1 in [EditOp.DELETE, EditOp.ADD]:
            return f"{self.l1} {self.l2}"
        return f"{self.l1} -> {self.l2}"


@dataclass
class EditCost(Edit):
    cost: int

    @staticmethod
    def from_edit(edit: Edit, cost: int):
        return EditCost(edit.l1, edit.l2, cost)

    def __repr__(self):
        return f"{super().__repr__()} : {self.cost}"


class EditCostConfig:
    default_cost: int

    def __init__(self, default_cost: int = 1):
        self.default_cost = default_cost

    def get_cost(self, edit: Edit):
        if edit is None:
            return 0
        return self.default_cost


class GranularEditCostConfig(EditCostConfig):

    def __init__(self, default_cost: int = 1, edit_costs: Optional[List[EditCost]] = None):
        super().__init__(default_cost)
        self.letter_cost: Dict[str, Dict[str, int]] = {}
        if edit_costs is not None:
            for c in edit_costs:
                m = self.letter_cost.get(c.l1, {})
                m[c.l2] = c.cost
                self.letter_cost[c.l1] = m

    def get_cost(self, edit: Edit) -> int:
        if edit is not None:
            if edit.l1 in self.letter_cost:
                if edit.l2 in self.letter_cost[edit.l1]:
                    return self.letter_cost[edit.l1][edit.l2]
        return super().get_cost(edit)

    def __repr__(self):
        lines = [f"GranularEditCost:",
                 f"\tdefault_cost: {self.default_cost}",
                 f"\tletter cost: "]
        for l1, m in self.letter_cost.items():
            for l2, cost in m.items():
                lines.append(f"\t\t{EditCost(l1, l2, cost)}")
        return "\n".join(lines)
