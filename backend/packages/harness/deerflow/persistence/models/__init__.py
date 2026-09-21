"""ORM model registration entry point.

Importing this module ensures all ORM models are registered with
``Base.metadata`` so Alembic autogenerate detects every table.

The actual ORM classes have moved to entity-specific subpackages:
- ``deerflow.persistence.thread_meta``
- ``deerflow.persistence.run``
- ``deerflow.persistence.feedback``
- ``deerflow.persistence.user``

``RunEventRow`` remains in ``deerflow.persistence.models.run_event`` because
its storage implementation lives in ``deerflow.runtime.events.store.db`` and
there is no matching entity directory.
"""

from deerflow.knowledge.schema import (
    AgentRunRow as KnowledgeAgentRunRow,
)
from deerflow.knowledge.schema import (
    ArtifactRow as KnowledgeArtifactRow,
)
from deerflow.knowledge.schema import (
    AssumptionRow as KnowledgeAssumptionRow,
)
from deerflow.knowledge.schema import (
    DatasetVersionRow as KnowledgeDatasetVersionRow,
)
from deerflow.knowledge.schema import (
    ExperimentArtifactRow as KnowledgeExperimentArtifactRow,
)
from deerflow.knowledge.schema import (
    ExperimentDatasetRow as KnowledgeExperimentDatasetRow,
)
from deerflow.knowledge.schema import (
    ExperimentRow as KnowledgeExperimentRow,
)
from deerflow.knowledge.schema import (
    FindingRow as KnowledgeFindingRow,
)
from deerflow.knowledge.schema import (
    ResearchProjectRow as KnowledgeResearchProjectRow,
)
from deerflow.persistence.agents.model import AgentRow
from deerflow.persistence.channel_connections.model import (
    ChannelConnectionRow,
    ChannelConversationRow,
    ChannelCredentialRow,
    ChannelOAuthStateRow,
)
from deerflow.persistence.feedback.model import FeedbackRow
from deerflow.persistence.managed_subagents.model import ManagedSubagentRow
from deerflow.persistence.mcp_tasks.model import McpTaskRow
from deerflow.persistence.models.run_event import RunEventRow
from deerflow.persistence.personal_access_tokens.model import PersonalAccessTokenRow
from deerflow.persistence.projects.model import ProjectRow
from deerflow.persistence.run.model import RunRow
from deerflow.persistence.scheduled_task_runs.model import ScheduledTaskRunRow
from deerflow.persistence.scheduled_tasks.model import ScheduledTaskRow
from deerflow.persistence.subagent_batches.model import SubagentBatchItemRow, SubagentBatchRow
from deerflow.persistence.thread_meta.model import ThreadMetaRow
from deerflow.persistence.user.model import UserRow
from deerflow.persistence.webhook_delivery.model import WebhookDeliveryRow

__all__ = [
    "AgentRow",
    "ChannelConnectionRow",
    "ChannelConversationRow",
    "ChannelCredentialRow",
    "ChannelOAuthStateRow",
    "FeedbackRow",
    "KnowledgeAgentRunRow",
    "KnowledgeArtifactRow",
    "KnowledgeAssumptionRow",
    "KnowledgeDatasetVersionRow",
    "KnowledgeExperimentArtifactRow",
    "KnowledgeExperimentDatasetRow",
    "KnowledgeExperimentRow",
    "KnowledgeFindingRow",
    "KnowledgeResearchProjectRow",
    "McpTaskRow",
    "ManagedSubagentRow",
    "PersonalAccessTokenRow",
    "ProjectRow",
    "RunEventRow",
    "RunRow",
    "ScheduledTaskRow",
    "ScheduledTaskRunRow",
    "SubagentBatchRow",
    "SubagentBatchItemRow",
    "ThreadMetaRow",
    "UserRow",
    "WebhookDeliveryRow",
]
