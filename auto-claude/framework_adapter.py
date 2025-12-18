"""
Framework adapter pattern for Auto Claude.

Provides unified interface for BMAD and Native framework operations.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

from models import Task, WorkUnit


class FrameworkAdapter(ABC):
    """
    Abstract adapter for framework-specific artifact creation and management.
    """

    def __init__(self, project_path: str):
        """
        Initialize adapter.

        Args:
            project_path: Path to project directory
        """
        self.project_path = Path(project_path)

    @abstractmethod
    def create_task(self, task: Task) -> str:
        """
        Create a task/story in framework-specific format.

        Args:
            task: Unified task model

        Returns:
            Task identifier (file path or ID)
        """
        pass

    @abstractmethod
    def update_task(self, task_id: str, updates: Dict) -> bool:
        """
        Update existing task with new data.

        Args:
            task_id: Task identifier
            updates: Dict of fields to update

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def create_work_unit(self, work_unit: WorkUnit) -> str:
        """
        Create a work unit (epic/phase) in framework-specific format.

        Args:
            work_unit: Unified work unit model

        Returns:
            Work unit identifier
        """
        pass

    @abstractmethod
    def update_work_unit(self, work_unit_id: str, updates: Dict) -> bool:
        """
        Update existing work unit.

        Args:
            work_unit_id: Work unit identifier
            updates: Dict of fields to update

        Returns:
            True if successful, False otherwise
        """
        pass

    # Note: Artifact creation removed - not part of unified model
    # BMAD workflows create artifacts directly in _bmad-output/

    @abstractmethod
    def get_artifacts_directory(self) -> Path:
        """
        Get directory where artifacts are stored.

        Returns:
            Path to artifacts directory
        """
        pass


class BMADAdapter(FrameworkAdapter):
    """
    BMAD Method adapter implementation.

    Stores artifacts in _bmad-output/ directory.
    """

    def get_artifacts_directory(self) -> Path:
        """Get BMAD output directory."""
        return self.project_path / "_bmad-output"

    def create_task(self, task: Task) -> str:
        """
        Create task as story in epic file.

        BMAD stores stories within epic markdown files.

        Args:
            task: Task to create

        Returns:
            Story identifier
        """
        # In BMAD, stories are embedded in epic files
        # So this updates the parent epic file
        if not task.epic_id:
            raise ValueError("Task must have epic_id for BMAD adapter")

        epic_file = self.get_artifacts_directory() / f"{task.epic_id}.md"

        # Read epic file, append story
        # (Simplified - real implementation would parse and insert)

        story_id = f"{task.epic_id}-{task.title.lower().replace(' ', '-')}"
        return story_id

    def update_task(self, task_id: str, updates: Dict) -> bool:
        """Update task in epic file."""
        # Parse epic file, find story, update fields
        # (Simplified implementation)
        return True

    def create_work_unit(self, work_unit: WorkUnit) -> str:
        """
        Create work unit (epic) as markdown file.

        Args:
            work_unit: WorkUnit to create

        Returns:
            Work unit file path
        """
        output_dir = self.get_artifacts_directory()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate work unit filename
        work_unit_id = work_unit.title.lower().replace(" ", "-")
        work_unit_file = output_dir / f"epic-{work_unit_id}.md"

        # Generate markdown content
        content = self._generate_work_unit_markdown(work_unit)

        # Write file
        work_unit_file.write_text(content)

        return str(work_unit_file)

    def update_work_unit(self, work_unit_id: str, updates: Dict) -> bool:
        """Update work unit file."""
        work_unit_file = self.get_artifacts_directory() / f"{work_unit_id}.md"

        if not work_unit_file.exists():
            return False

        # Read, update, write
        # (Simplified implementation)
        return True

    def _generate_work_unit_markdown(self, work_unit: WorkUnit) -> str:
        """
        Generate work unit (epic) markdown content.

        Args:
            work_unit: WorkUnit model

        Returns:
            Markdown content
        """
        lines = [
            f"# Epic: {work_unit.title}",
            "",
            f"**Status:** {work_unit.status.value if hasattr(work_unit.status, 'value') else work_unit.status}",
            "",
            "## Description",
            "",
            work_unit.description,
            "",
        ]

        # Add metadata if present
        if work_unit.metadata:
            phase = work_unit.metadata.get("phase")
            priority = work_unit.metadata.get("priority")
            user_value = work_unit.metadata.get("user_value")
            dependencies = work_unit.metadata.get("dependencies", [])

            if phase:
                lines.insert(3, f"**Phase:** {phase}")
            if priority:
                lines.insert(3, f"**Priority:** {priority}")

            if user_value:
                lines.extend(["## User Value", "", user_value, ""])

            if dependencies:
                lines.extend(
                    [
                        "## Dependencies",
                        "",
                        *[f"- {dep}" for dep in dependencies],
                        "",
                    ]
                )

        lines.extend(["## Stories", ""])

        for i, task in enumerate(work_unit.tasks, 1):
            lines.extend(
                [
                    f"### Story {i}: {task.title}",
                    "",
                    task.description,
                    "",
                    "**Acceptance Criteria:**",
                    "",
                    *[f"- {ac}" for ac in task.acceptance_criteria],
                    "",
                ]
            )

        return "\n".join(lines)


class NativeAdapter(FrameworkAdapter):
    """
    Native Auto Claude adapter implementation.

    Stores artifacts in .auto-claude/specs/ directory.
    """

    def get_artifacts_directory(self) -> Path:
        """Get native specs directory."""
        return self.project_path / ".auto-claude" / "specs"

    def create_task(self, task: Task) -> str:
        """Create task as spec file."""
        specs_dir = self.get_artifacts_directory()
        specs_dir.mkdir(parents=True, exist_ok=True)

        # Generate spec number
        existing_specs = list(specs_dir.glob("*"))
        spec_num = len(existing_specs) + 1

        # Generate spec directory
        task_id = f"{spec_num:03d}-{task.title.lower().replace(' ', '-')}"
        spec_dir = specs_dir / task_id
        spec_dir.mkdir(exist_ok=True)

        # Create spec.md
        spec_file = spec_dir / "spec.md"
        content = self._generate_spec_markdown(task)
        spec_file.write_text(content)

        return task_id

    def update_task(self, task_id: str, updates: Dict) -> bool:
        """Update task spec file."""
        spec_file = self.get_artifacts_directory() / task_id / "spec.md"

        if not spec_file.exists():
            return False

        # Update spec file
        # (Simplified implementation)
        return True

    def create_work_unit(self, work_unit: WorkUnit) -> str:
        """
        Create work unit - in Native, work units are collections of specs.

        Args:
            work_unit: WorkUnit to create

        Returns:
            Work unit identifier
        """
        # In Native framework, work units don't have special files
        # Each task becomes a separate spec
        work_unit_id = work_unit.title.lower().replace(" ", "-")

        for task in work_unit.tasks:
            # Store work unit reference in metadata
            task.metadata["work_unit_id"] = work_unit_id
            self.create_task(task)

        return work_unit_id

    def update_work_unit(self, work_unit_id: str, updates: Dict) -> bool:
        """Update work unit (updates constituent specs)."""
        return True

    def _generate_spec_markdown(self, task: Task) -> str:
        """Generate spec.md content."""
        lines = [
            f"# {task.title}",
            "",
            task.description,
            "",
            "## Acceptance Criteria",
            "",
            *[f"- {ac}" for ac in task.acceptance_criteria],
            "",
        ]

        if task.technical_notes:
            lines.extend(["## Technical Notes", "", task.technical_notes, ""])

        return "\n".join(lines)


def get_adapter(project_path: str) -> FrameworkAdapter:
    """
    Get appropriate framework adapter for project.

    Args:
        project_path: Path to project directory

    Returns:
        Framework adapter instance
    """
    from bmad_detector import get_active_framework

    framework = get_active_framework(project_path)

    if framework == "bmad":
        return BMADAdapter(project_path)
    else:
        return NativeAdapter(project_path)
