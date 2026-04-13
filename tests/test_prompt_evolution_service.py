import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from AgentCrew.modules.agents.local_agent import LocalAgent
from AgentCrew.modules.agents.prompt_evolution_service import PromptEvolutionService

SAMPLE_MEMORY_XML = """<MEMORY>
    <HEAD>debugging async streaming issue</HEAD>
    <DATE>2025-06-15</DATE>
    <CONTEXT>User debugging a race condition in async generator streaming</CONTEXT>
    <INSIGHTS>
        <INSIGHT>Async generators need explicit cleanup in long-running services</INSIGHT>
    </INSIGHTS>
    <ENTITIES>
        <ENTITY>
            <NAME>task_manager.py</NAME>
            <DESC>Core module handling task lifecycle</DESC>
        </ENTITY>
    </ENTITIES>
    <DOMAINS>
        <DOMAIN>Software Development</DOMAIN>
        <DOMAIN>Async Programming</DOMAIN>
    </DOMAINS>
    <RESOURCES>
        <RESOURCE>AgentCrew/modules/a2a/task_manager.py</RESOURCE>
    </RESOURCES>
    <CONVERSATION_NOTES>
        <NOTE>Caveat: task state dict must not be mutated while an async generator is yielding from it</NOTE>
        <NOTE>User prefers concise implementation-focused answers</NOTE>
    </CONVERSATION_NOTES>
</MEMORY>"""

SAMPLE_MEMORY_XML_MINIMAL = """<MEMORY>
    <HEAD>quick question</HEAD>
    <DATE>2025-06-16</DATE>
    <CONTEXT></CONTEXT>
    <INSIGHTS/>
    <CONVERSATION_NOTES/>
</MEMORY>"""


class TestPromptEvolutionService(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_llm = MagicMock()
        self.mock_llm.process_message = AsyncMock()
        self.agent = LocalAgent(
            name="Engineer",
            description="Implementation specialist",
            llm_service=self.mock_llm,
            services={},
            tools=[],
        )
        self.agent.set_system_prompt(
            "You are Engineer. Keep placeholders {current_date} and {cwd}."
        )
        self.memory_service = MagicMock()
        self.persistence_service = MagicMock()
        self.service = PromptEvolutionService(
            memory_service=self.memory_service,
            persistence_service=self.persistence_service,
        )

    async def test_create_evolution_proposal_filters_project_specific_items(self):
        self.memory_service.get_agent_memory_corpus.return_value = [
            {"id": "m1", "document": SAMPLE_MEMORY_XML, "metadata": {}},
            {"id": "m2", "document": SAMPLE_MEMORY_XML, "metadata": {}},
        ]
        self.mock_llm.process_message.return_value = """{
          "durable_traits": [{"item": "Prefer concise implementation-focused answers", "evidence": "repeated", "strength": "high"}],
          "output_preferences": [{"item": "Avoid verbose explanations", "evidence": "multiple memories", "strength": "medium"}],
          "recurring_user_corrections": [{"item": "Confirm with evidence before root-cause claims", "evidence": "repeated", "strength": "medium"}],
          "workflow_patterns": [{"item": "Always check repo structure first", "evidence": "memories #1, #2", "strength": "high"}],
          "tool_usage_preferences": [{"item": "Use uv instead of pip for Python projects", "evidence": "memories #1, #2", "strength": "high"}],
          "excluded_as_project_specific": [{"item": "task_manager.py race condition", "reason": "file-specific"}],
          "confidence_notes": ["good evidence"]
        }"""

        proposal = await self.service.create_evolution_proposal(self.agent)

        self.assertEqual(proposal["agent_name"], "Engineer")
        self.assertIn(
            "Prefer concise implementation-focused answers",
            proposal["user_editable_summary"],
        )
        self.assertIn(
            "Always check repo structure first", proposal["user_editable_summary"]
        )
        self.assertIn("Use uv instead of pip", proposal["user_editable_summary"])
        self.assertIn("[high]", proposal["user_editable_summary"])
        summary = proposal["analysis_summary"]
        self.assertTrue(len(summary["workflow_patterns"]) > 0)
        self.assertTrue(len(summary["tool_usage_preferences"]) > 0)

    async def test_create_evolution_proposal_rejects_empty_memory(self):
        self.memory_service.get_agent_memory_corpus.return_value = []
        with self.assertRaises(ValueError):
            await self.service.create_evolution_proposal(self.agent)

    async def test_build_revised_prompt_preserves_placeholders(self):
        self.mock_llm.process_message.return_value = (
            "You are Engineer. Keep placeholders {current_date} and {cwd}. Be concise."
        )
        revised = await self.service.build_revised_prompt(
            self.agent, "Durable traits:\n- Be concise"
        )
        self.assertIn("{current_date}", revised)
        self.assertIn("{cwd}", revised)

    async def test_build_revised_prompt_rejects_missing_placeholder(self):
        self.mock_llm.process_message.return_value = "You are Engineer. Be concise."
        with self.assertRaises(ValueError):
            await self.service.build_revised_prompt(
                self.agent, "Durable traits:\n- Be concise"
            )

    @patch(
        "AgentCrew.modules.agents.prompt_evolution_service.AgentsConfig.update_agent_system_prompt"
    )
    def test_apply_prompt_revision_persists_and_audits(self, mock_update_prompt):
        mock_update_prompt.return_value = True
        result = self.service.apply_prompt_revision(
            self.agent,
            "You are Engineer. Keep placeholders {current_date} and {cwd}. Be concise.",
            "Durable traits:\n- Be concise",
            memory_ids=["m1"],
            edited_by_user=True,
        )
        self.assertEqual(result["agent_name"], "Engineer")
        self.persistence_service.store_prompt_evolution.assert_called_once()

    def test_apply_prompt_revision_records_generated_and_approved_summaries_separately(self):
        self.service.agents_config = MagicMock()
        self.service.agents_config.update_agent_system_prompt.return_value = True
        result = self.service.apply_prompt_revision(
            self.agent,
            "You are Engineer. Keep placeholders {current_date} and {cwd}. Be concise.",
            "Approved summary",
            generated_summary="Generated summary",
            memory_ids=["m1"],
            edited_by_user=True,
        )

        stored_record = self.persistence_service.store_prompt_evolution.call_args[0][1]
        self.assertEqual(stored_record["generated_summary"], "Generated summary")
        self.assertEqual(stored_record["approved_summary"], "Approved summary")
        self.assertTrue(stored_record["edited_by_user"])
        self.assertEqual(result["generated_summary"], "Generated summary")
        self.assertEqual(result["accepted_summary"], "Approved summary")

    def test_apply_prompt_revision_raises_when_audit_storage_fails(self):
        self.service.agents_config = MagicMock()
        self.service.agents_config.update_agent_system_prompt.return_value = True
        self.persistence_service.store_prompt_evolution.side_effect = RuntimeError(
            "audit failed"
        )

        with self.assertRaises(ValueError) as ctx:
            self.service.apply_prompt_revision(
                self.agent,
                "You are Engineer. Keep placeholders {current_date} and {cwd}. Be concise.",
                "Approved summary",
                generated_summary="Generated summary",
                memory_ids=["m1"],
                edited_by_user=True,
            )

        self.assertIn("Prompt was persisted for agent 'Engineer'", str(ctx.exception))
        self.assertIn("audit record failed", str(ctx.exception))

    def test_extract_evolution_fields_parses_xml_memory(self):
        result = self.service._extract_evolution_fields(SAMPLE_MEMORY_XML, 1)
        self.assertIn("Memory #1", result)
        self.assertIn("2025-06-15", result)
        self.assertIn("Topic: debugging async streaming issue", result)
        self.assertIn("Domains: Software Development, Async Programming", result)
        self.assertIn("Async generators need explicit cleanup", result)
        self.assertIn("task state dict must not be mutated", result)
        self.assertNotIn("task_manager.py", result)
        self.assertNotIn("RESOURCE", result)

    def test_extract_evolution_fields_skips_empty_memory(self):
        result = self.service._extract_evolution_fields(SAMPLE_MEMORY_XML_MINIMAL, 2)
        self.assertEqual(result, "")

    def test_extract_evolution_fields_handles_invalid_xml(self):
        result = self.service._extract_evolution_fields("not xml at all", 1)
        self.assertEqual(result, "")

    def test_prepare_corpus_for_analysis(self):
        corpus = [
            {"id": "m1", "document": SAMPLE_MEMORY_XML, "metadata": {}},
            {"id": "m2", "document": SAMPLE_MEMORY_XML_MINIMAL, "metadata": {}},
            {"id": "m3", "document": SAMPLE_MEMORY_XML, "metadata": {}},
        ]
        result = self.service._prepare_corpus_for_analysis(corpus)
        self.assertIn("Memory #1", result)
        self.assertIn("Memory #3", result)
        self.assertNotIn("Memory #2", result)

    def test_parse_json_response_strips_markdown_fences(self):
        wrapped = '```json\n{"durable_traits": []}\n```'
        result = self.service._parse_json_response(wrapped)
        self.assertEqual(result, {"durable_traits": []})

    def test_parse_json_response_handles_plain_json(self):
        plain = '{"durable_traits": [{"item": "test", "evidence": "e", "strength": "high"}]}'
        result = self.service._parse_json_response(plain)
        self.assertIn("durable_traits", result)

    def test_sanitize_analysis_includes_new_categories(self):
        analysis = {
            "durable_traits": [],
            "output_preferences": [],
            "recurring_user_corrections": [],
            "workflow_patterns": [
                {"item": "Check structure first", "evidence": "e", "strength": "high"}
            ],
            "tool_usage_preferences": [
                {"item": "Use uv for python", "evidence": "e", "strength": "medium"}
            ],
            "excluded_as_project_specific": [],
            "confidence_notes": [],
        }
        sanitized = self.service._sanitize_analysis(analysis)
        self.assertEqual(len(sanitized["workflow_patterns"]), 1)
        self.assertEqual(len(sanitized["tool_usage_preferences"]), 1)

    def test_format_user_summary_includes_new_categories_with_strength(self):
        analysis = {
            "durable_traits": [
                {"item": "Be concise", "evidence": "e", "strength": "high"}
            ],
            "output_preferences": [],
            "recurring_user_corrections": [],
            "workflow_patterns": [
                {"item": "Read repo first", "evidence": "e", "strength": "medium"}
            ],
            "tool_usage_preferences": [
                {"item": "Use uv", "evidence": "e", "strength": "high"}
            ],
        }
        summary = self.service._format_user_summary(analysis)
        self.assertIn("Workflow patterns:", summary)
        self.assertIn("Tool usage preferences:", summary)
        self.assertIn("[high]", summary)
        self.assertIn("[medium]", summary)
