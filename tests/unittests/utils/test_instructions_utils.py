# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from google.adk.agents.llm_agent import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.sessions.session import Session
from google.adk.utils import instructions_utils
import pytest

from .. import testing_utils


class MockArtifactService:

  def __init__(self, artifacts: dict):
    self.artifacts = artifacts

  async def load_artifact(self, app_name, user_id, session_id, filename):
    if filename in self.artifacts:
      return self.artifacts[filename]
    else:
      return None


async def _create_test_readonly_context(
    state: dict = None,
    artifact_service: MockArtifactService = None,
    app_name: str = "test_app",
    user_id: str = "test_user",
    session_id: str = "test_session_id",
) -> ReadonlyContext:
  agent = Agent(
      model="gemini-2.0-flash",
      name="agent",
      instruction="test",
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  invocation_context.session = Session(
      state=state if state else {},
      app_name=app_name,
      user_id=user_id,
      id=session_id,
  )

  invocation_context.artifact_service = artifact_service
  return ReadonlyContext(invocation_context)


@pytest.mark.parametrize("use_jinja2", [False, True])
@pytest.mark.asyncio
async def test_inject_session_state(use_jinja2):
  if use_jinja2:
    instruction_template = (
        "Hello {{ user_name }}, you are in {{ app_state }} state."
    )
  else:
    instruction_template = "Hello {user_name}, you are in {app_state} state."
  invocation_context = await _create_test_readonly_context(
      state={"user_name": "Foo", "app_state": "active"}
  )
  populated_instruction = await instructions_utils.inject_session_state(
      instruction_template, invocation_context, use_jinja2=use_jinja2
  )
  assert populated_instruction == "Hello Foo, you are in active state."


@pytest.mark.parametrize("use_jinja2", [False, True])
@pytest.mark.asyncio
async def test_inject_session_state_with_artifact(use_jinja2):
  if use_jinja2:
    instruction_template = "The artifact content is: {{ artifact('my_file') }}"
  else:
    instruction_template = "The artifact content is: {artifact.my_file}"
  mock_artifact_service = MockArtifactService(
      {"my_file": "This is my artifact content."}
  )
  invocation_context = await _create_test_readonly_context(
      artifact_service=mock_artifact_service
  )
  populated_instruction = await instructions_utils.inject_session_state(
      instruction_template, invocation_context, use_jinja2=use_jinja2
  )
  assert (
      populated_instruction
      == "The artifact content is: This is my artifact content."
  )


@pytest.mark.parametrize("use_jinja2", [False, True])
@pytest.mark.asyncio
async def test_inject_session_state_with_optional_state(use_jinja2):
  if use_jinja2:
    instruction_template = "Optional value: {{ optional_value|default('') }}"
  else:
    instruction_template = "Optional value: {optional_value?}"
  invocation_context = await _create_test_readonly_context()
  populated_instruction = await instructions_utils.inject_session_state(
      instruction_template, invocation_context, use_jinja2=use_jinja2
  )
  assert populated_instruction == "Optional value: "


@pytest.mark.parametrize("use_jinja2", [False, True])
@pytest.mark.asyncio
async def test_inject_session_state_with_missing_state_raises_key_error(
    use_jinja2,
):
  if use_jinja2:
    instruction_template = "Hello {{ missing_key }}!"
  else:
    instruction_template = "Hello {missing_key}!"
  invocation_context = await _create_test_readonly_context(
      state={"user_name": "Foo"}
  )
  with pytest.raises(
      KeyError, match="Context variable not found: `missing_key`."
  ):
    await instructions_utils.inject_session_state(
        instruction_template, invocation_context, use_jinja2=use_jinja2
    )


@pytest.mark.parametrize("use_jinja2", [False, True])
@pytest.mark.asyncio
async def test_inject_session_state_with_missing_artifact_raises_key_error(
    use_jinja2,
):
  if use_jinja2:
    instruction_template = (
        "The artifact content is: {{ artifact('missing_file') }}"
    )
  else:
    instruction_template = "The artifact content is: {artifact.missing_file}"
  mock_artifact_service = MockArtifactService(
      {"my_file": "This is my artifact content."}
  )
  invocation_context = await _create_test_readonly_context(
      artifact_service=mock_artifact_service
  )
  with pytest.raises(KeyError, match="Artifact missing_file not found."):
    await instructions_utils.inject_session_state(
        instruction_template, invocation_context, use_jinja2=use_jinja2
    )


@pytest.mark.parametrize("use_jinja2", [False, True])
@pytest.mark.asyncio
async def test_inject_session_state_with_invalid_state_name_returns_original(
    use_jinja2,
):
  if use_jinja2:
    instruction_template = "Hello {{ invalid-key }}!"
  else:
    instruction_template = "Hello {invalid-key}!"
  invocation_context = await _create_test_readonly_context(
      state={"user_name": "Foo"}
  )
  populated_instruction = await instructions_utils.inject_session_state(
      instruction_template, invocation_context, use_jinja2=use_jinja2
  )
  if use_jinja2:
    # For Jinja2, invalid variable names might be treated differently
    assert populated_instruction == "Hello !"
  else:
    assert populated_instruction == "Hello {invalid-key}!"


@pytest.mark.parametrize("use_jinja2", [False, True])
@pytest.mark.asyncio
async def test_inject_session_state_with_invalid_prefix_state_name_returns_original(
    use_jinja2,
):
  if use_jinja2:
    instruction_template = "Hello {{ invalid:key }}!"
  else:
    instruction_template = "Hello {invalid:key}!"
  invocation_context = await _create_test_readonly_context(
      state={"user_name": "Foo"}
  )
  populated_instruction = await instructions_utils.inject_session_state(
      instruction_template, invocation_context, use_jinja2=use_jinja2
  )
  if use_jinja2:
    # For Jinja2, invalid variable names might be treated differently
    assert populated_instruction == "Hello !"
  else:
    assert populated_instruction == "Hello {invalid:key}!"


@pytest.mark.parametrize("use_jinja2", [False, True])
@pytest.mark.asyncio
async def test_inject_session_state_with_valid_prefix_state(use_jinja2):
  if use_jinja2:
    instruction_template = "Hello {{ app:user_name }}!"
  else:
    instruction_template = "Hello {app:user_name}!"
  invocation_context = await _create_test_readonly_context(
      state={"app:user_name": "Foo"}
  )
  populated_instruction = await instructions_utils.inject_session_state(
      instruction_template, invocation_context, use_jinja2=use_jinja2
  )
  assert populated_instruction == "Hello Foo!"


@pytest.mark.parametrize("use_jinja2", [False, True])
@pytest.mark.asyncio
async def test_inject_session_state_with_multiple_variables_and_artifacts(
    use_jinja2,
):
  if use_jinja2:
    instruction_template = """
      Hello {{ user_name }},
      You are {{ user_age }} years old.
      Your favorite color is {{ favorite_color|default('') }}.
      The artifact says: {{ artifact('my_file') }}
      And another optional artifact: {{ artifact('other_file') }}
      """
  else:
    instruction_template = """
      Hello {user_name},
      You are {user_age} years old.
      Your favorite color is {favorite_color?}.
      The artifact says: {artifact.my_file}
      And another optional artifact: {artifact.other_file}
      """
  mock_artifact_service = MockArtifactService({
      "my_file": "This is my artifact content.",
      "other_file": "This is another artifact content.",
  })
  invocation_context = await _create_test_readonly_context(
      state={"user_name": "Foo", "user_age": 30, "favorite_color": "blue"},
      artifact_service=mock_artifact_service,
  )
  populated_instruction = await instructions_utils.inject_session_state(
      instruction_template, invocation_context, use_jinja2=use_jinja2
  )
  expected_instruction = """
      Hello Foo,
      You are 30 years old.
      Your favorite color is blue.
      The artifact says: This is my artifact content.
      And another optional artifact: This is another artifact content.
      """
  # Strip whitespace for comparison to avoid issues with formatting
  assert populated_instruction.strip() == expected_instruction.strip()


@pytest.mark.parametrize("use_jinja2", [False, True])
@pytest.mark.asyncio
async def test_inject_session_state_with_empty_artifact_name_raises_key_error(
    use_jinja2,
):
  if use_jinja2:
    instruction_template = "The artifact content is: {{ artifact('') }}"
  else:
    instruction_template = "The artifact content is: {artifact.}"
  mock_artifact_service = MockArtifactService(
      {"my_file": "This is my artifact content."}
  )
  invocation_context = await _create_test_readonly_context(
      artifact_service=mock_artifact_service
  )
  with pytest.raises(KeyError, match="Artifact  not found."):
    await instructions_utils.inject_session_state(
        instruction_template, invocation_context, use_jinja2=use_jinja2
    )


@pytest.mark.parametrize("use_jinja2", [False, True])
@pytest.mark.asyncio
async def test_inject_session_state_artifact_service_not_initialized_raises_value_error(
    use_jinja2,
):
  if use_jinja2:
    instruction_template = "The artifact content is: {{ artifact('my_file') }}"
  else:
    instruction_template = "The artifact content is: {artifact.my_file}"
  invocation_context = await _create_test_readonly_context()
  with pytest.raises(ValueError, match="Artifact service is not initialized."):
    await instructions_utils.inject_session_state(
        instruction_template, invocation_context, use_jinja2=use_jinja2
    )


@pytest.mark.parametrize("use_jinja2", [False, True])
@pytest.mark.asyncio
async def test_inject_session_state_with_optional_missing_artifact_returns_empty(
    use_jinja2,
):
  if use_jinja2:
    instruction_template = (
        "Optional artifact: {{ artifact('missing_file')|default('') }}"
    )
  else:
    instruction_template = "Optional artifact: {artifact.missing_file?}"
  mock_artifact_service = MockArtifactService(
      {"my_file": "This is my artifact content."}
  )
  invocation_context = await _create_test_readonly_context(
      artifact_service=mock_artifact_service
  )
  populated_instruction = await instructions_utils.inject_session_state(
      instruction_template, invocation_context, use_jinja2=use_jinja2
  )
  assert populated_instruction == "Optional artifact: "


@pytest.mark.parametrize("use_jinja2", [False, True])
@pytest.mark.asyncio
async def test_inject_session_state_with_none_state_value_returns_empty(
    use_jinja2,
):
  if use_jinja2:
    instruction_template = "Value: {{ test_key|default('') }}"
  else:
    instruction_template = "Value: {test_key}"
  invocation_context = await _create_test_readonly_context(
      state={"test_key": None}
  )
  populated_instruction = await instructions_utils.inject_session_state(
      instruction_template, invocation_context, use_jinja2=use_jinja2
  )
  assert populated_instruction == "Value: "


@pytest.mark.parametrize("use_jinja2", [False, True])
@pytest.mark.asyncio
async def test_inject_session_state_with_optional_missing_state_returns_empty(
    use_jinja2,
):
  if use_jinja2:
    instruction_template = "Optional value: {{ missing_key|default('') }}"
  else:
    instruction_template = "Optional value: {missing_key?}"
  invocation_context = await _create_test_readonly_context()
  populated_instruction = await instructions_utils.inject_session_state(
      instruction_template, invocation_context, use_jinja2=use_jinja2
  )
  assert populated_instruction == "Optional value: "


@pytest.mark.asyncio
async def test_render_with_jinja2_conditional_if():
  instruction_template = """
{% if is_admin %}
Hello admin user {{ user_name }}!
{% endif %}
"""
  invocation_context = await _create_test_readonly_context(
      state={"user_name": "Alice", "is_admin": True}
  )
  populated_instruction = await instructions_utils.inject_session_state(
      instruction_template, invocation_context, use_jinja2=True
  )
  assert "Hello admin user Alice!" in populated_instruction


@pytest.mark.asyncio
async def test_render_with_jinja2_conditional_if_else():
  instruction_template = """
{% if is_active %}
User {{ user_name }} is active.
{% else %}
User {{ user_name }} is inactive.
{% endif %}
"""
  # Test the 'if' case
  invocation_context_active = await _create_test_readonly_context(
      state={"user_name": "Bob", "is_active": True}
  )
  populated_instruction_active = await instructions_utils.inject_session_state(
      instruction_template, invocation_context_active, use_jinja2=True
  )
  assert "User Bob is active." in populated_instruction_active

  # Test the 'else' case
  invocation_context_inactive = await _create_test_readonly_context(
      state={"user_name": "Carol", "is_active": False}
  )
  populated_instruction_inactive = (
      await instructions_utils.inject_session_state(
          instruction_template, invocation_context_inactive, use_jinja2=True
      )
  )
  assert "User Carol is inactive." in populated_instruction_inactive


@pytest.mark.asyncio
async def test_render_with_jinja2_for_loop():
  instruction_template = """
My favorite fruits are:
{% for fruit in fruits %}
- {{ fruit }}
{% endfor %}
"""
  invocation_context = await _create_test_readonly_context(
      state={"fruits": ["apple", "banana", "cherry"]}
  )
  populated_instruction = await instructions_utils.inject_session_state(
      instruction_template, invocation_context, use_jinja2=True
  )
  expected_output = """
My favorite fruits are:
- apple
- banana
- cherry
"""
  assert populated_instruction.strip() == expected_output.strip()


@pytest.mark.asyncio
async def test_render_with_jinja2_asynchronous_artifact():
  instruction_template = (
      "The artifact content is: {{ await artifact('my_file') }}"
  )
  mock_artifact_service = MockArtifactService(
      {"my_file": "This is my artifact content from a new test."}
  )
  invocation_context = await _create_test_readonly_context(
      artifact_service=mock_artifact_service
  )
  populated_instruction = await instructions_utils.inject_session_state(
      instruction_template, invocation_context, use_jinja2=True
  )
  assert (
      populated_instruction
      == "The artifact content is: This is my artifact content from a new test."
  )
