from enum import Enum, auto


class AgentState(Enum):
    NAVIGATING = auto()   # cached world point (confirmed subgoal or best-guess), drive to it every step
    SEARCHING = auto()    # working through SEARCH_PLAN, one detect per heading, until found or exhausted
