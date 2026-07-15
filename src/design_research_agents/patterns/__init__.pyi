"""Static public interface for lazy pattern exports."""

from design_research_agents._implementations._patterns import BlackboardPattern as BlackboardPattern
from design_research_agents._implementations._patterns import DebatePattern as DebatePattern
from design_research_agents._implementations._patterns import NominalTeamPattern as NominalTeamPattern
from design_research_agents._implementations._patterns import PlanExecutePattern as PlanExecutePattern
from design_research_agents._implementations._patterns import ProposeCriticPattern as ProposeCriticPattern
from design_research_agents._implementations._patterns import ProposeCriticResult as ProposeCriticResult
from design_research_agents._implementations._patterns import RAGPattern as RAGPattern
from design_research_agents._implementations._patterns import RalphLoopPattern as RalphLoopPattern
from design_research_agents._implementations._patterns import (
    RoundBasedCoordinationPattern as RoundBasedCoordinationPattern,
)
from design_research_agents._implementations._patterns import (
    RouterDelegatePattern as RouterDelegatePattern,
)
from design_research_agents._implementations._patterns import (
    SimulatedAnnealingPattern as SimulatedAnnealingPattern,
)
from design_research_agents._implementations._patterns import TreeSearchPattern as TreeSearchPattern
from design_research_agents._implementations._patterns import (
    TwoSpeakerConversationPattern as TwoSpeakerConversationPattern,
)

__all__: list[str]
