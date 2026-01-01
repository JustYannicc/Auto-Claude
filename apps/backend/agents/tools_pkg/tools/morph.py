"""
Morph Fast Apply Tools
=======================

Tools for AI-powered code editing using Morph Fast Apply API.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

try:
    from claude_agent_sdk import tool

    SDK_TOOLS_AVAILABLE = True
except ImportError:
    SDK_TOOLS_AVAILABLE = False
    tool = None

from services.morph_client import (
    MorphAPIError,
    MorphClient,
    MorphConfig,
    MorphConnectionError,
    MorphTimeoutError,
)

logger = logging.getLogger(__name__)


def create_morph_tools(spec_dir: Path, project_dir: Path) -> list:
    """
    Create Morph Fast Apply tools.

    Args:
        spec_dir: Path to the spec directory
        project_dir: Path to the project root

    Returns:
        List of Morph tool functions
    """
    if not SDK_TOOLS_AVAILABLE:
        return []

    tools = []

    # -------------------------------------------------------------------------
    # Tool: MorphApply (edit_file)
    # -------------------------------------------------------------------------
    @tool(
        "MorphApply",
        "Apply AI-powered code edits to a file using Morph Fast Apply. "
        "Provide the file path, current content, and instructions for what changes to make. "
        "Morph will intelligently apply the edits and return the transformed code.",
        {
            "file_path": str,
            "content": str,
            "instruction": str,
            "language": str,  # Optional programming language hint
        },
    )
    async def morph_apply(args: dict[str, Any]) -> dict[str, Any]:
        """Apply code edits using Morph Fast Apply."""
        file_path = args.get("file_path", "")
        content = args.get("content", "")
        instruction = args.get("instruction", "")
        language = args.get("language")

        # Validate required arguments
        if not file_path:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Error: file_path is required",
                    }
                ]
            }

        if not content:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Error: content is required (current file content)",
                    }
                ]
            }

        if not instruction:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Error: instruction is required (describe what changes to make)",
                    }
                ]
            }

        # Create Morph client
        try:
            config = MorphConfig.from_env()
            if not config.has_api_key():
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": "Error: MORPH_API_KEY not configured. Cannot use Morph Fast Apply.",
                        }
                    ]
                }

            with MorphClient(config) as client:
                # Apply the edit
                result = client.apply(
                    file_path=file_path,
                    original_content=content,
                    instruction=instruction,
                    language=language,
                )

                if result.success:
                    return {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Successfully applied edits to {file_path}\n\n"
                                f"Transformed content:\n\n{result.new_content}",
                            }
                        ]
                    }
                else:
                    return {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Error: Morph apply failed for {file_path}",
                            }
                        ]
                    }

        except MorphAPIError as e:
            logger.error(f"Morph API error: {e}")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error: Morph API error - {e.message} (code: {e.code})",
                    }
                ]
            }
        except MorphConnectionError as e:
            logger.error(f"Morph connection error: {e}")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error: Cannot connect to Morph API - {str(e)}",
                    }
                ]
            }
        except MorphTimeoutError as e:
            logger.error(f"Morph timeout error: {e}")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error: Morph API request timed out - {str(e)}",
                    }
                ]
            }
        except Exception as e:
            logger.error(f"Unexpected error in MorphApply: {e}")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error: Unexpected error - {str(e)}",
                    }
                ]
            }

    tools.append(morph_apply)

    return tools
