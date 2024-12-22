from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from meta_buffer import MetaBuffer
from meta_buffer_utils import extract_and_execute_code
from test_templates import checkmate, game24, word_sorting
from prompts import (
    META_DISTILLER_PROMPT,
    BUFFER_PROMPT,
    INSPECTOR_PROMPT,
    REASONER_PROMPT,
)

class Pipeline:
    """Pipeline for handling model interactions."""

    def __init__(
        self,
        model_id: str,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1/",
    ) -> None:
        """
        Initialize Pipeline.

        Args:
            model_id: The model identifier
            api_key: Optional API key for authentication
            base_url: Base URL for API requests
        """
        self.model_id = model_id
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def get_respond(self, prompt_template, **kwargs) -> str:
        """
        Get model response for given prompts.

        Args:
            prompt_template: LangChain ChatPromptTemplate
            **kwargs: Keywords arguments for prompt template formatting

        Returns:
            Model generated response
        """
        messages = prompt_template.format_messages(**kwargs)
        formatted_messages = []
        for msg in messages:
            # Map LangChain message types to OpenAI roles
            role = {
                "human": "user",
                "system": "system",
                "ai": "assistant"
            }.get(msg.type, "user")  # Default to user if unknown type
            formatted_messages.append({
                "role": role,
                "content": msg.content
            })
        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=formatted_messages,
        )
        return completion.choices[0].message.content


# Constants for problem types
PROBLEM_TYPE_GAME24 = 0
PROBLEM_TYPE_CHECKMATE = 1
PROBLEM_TYPE_WORD_SORTING = 2

MAX_RETRY_COUNT = 3


@dataclass
class BotConfig:
    """Configuration for BoT class."""

    model_id: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-large"
    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1/"
    verification_enabled: bool = False
    rag_directory: str | None = None


class BoT:
    """Bot class for handling problem-solving pipeline."""

    def __init__(
        self,
        user_input: str,
        problem_id: int = PROBLEM_TYPE_GAME24,
        config: BotConfig | None = None,
    ) -> None:
        """
        Initialize BoT.

        Args:
            user_input: Input text from user
            problem_id: Type of problem to solve
            config: Optional configuration object

        """
        self.config = config or BotConfig()
        self.pipeline = Pipeline(
            self.config.model_id,
            self.config.api_key,
            self.config.base_url,
        )
        self.meta_buffer = MetaBuffer(
            self.config.model_id,
            self.config.embedding_model,
            self.config.api_key,
            base_url=self.config.base_url,
        )
        self.user_input = user_input
        # Only for test use, stay tuned for our update
        self.problem_id = problem_id
        self.need_check = self.config.verification_enabled
        with Path("./math.txt").open() as f:
            self.meta_buffer.rag.insert(f.read())

    def update_input(self, new_input: str) -> None:
        """
        Update the user input.

        Args:
            new_input: New input text

        """
        self.user_input = new_input

    def problem_distillation(self) -> None:
        self.distilled_information = self.pipeline.get_respond(
            META_DISTILLER_PROMPT,
            query=self.user_input
        )

    def buffer_retrieve(self) -> None:
        # For initial test use, we will later update the embedding retrieval version to support more, trail version
        if self.problem_id == 0:
            self.thought_template = game24
        elif self.problem_id == 1:
            self.thought_template = checkmate
        elif self.problem_id == PROBLEM_TYPE_WORD_SORTING:
            self.thought_template = word_sorting

    def buffer_instantiation(self) -> None:
        prompt = BUFFER_PROMPT.format_messages(
            task=self.distilled_information,
            meta_buffer=self.meta_buffer.get_content()
        )
        self.result = self.meta_buffer.retrieve_and_instantiate(prompt[1].content)

    def buffer_manager(self) -> None:
        self.problem_solution_pair = self.user_input + self.result
        self.thought_distillation()
        self.meta_buffer.dynamic_update(self.distilled_thought)

    def thought_distillation(self) -> None:
        thought_distillation_prompt = """You are an expert in problem analysis and generalization. Your task is to follow the format of thought template below and distill a high-level thought template to solve similar problems:
        Example thought template:
        ### Problem Type 20: Solution Concentration Problem

**Definition**: This type of problem involves the relationship between a solvent (water or another liquid), solute, solution, and concentration.

**Quantitative Relationships**:
- Solution = Solvent + Solute
- Concentration = Solute ÷ Solution × 100%

**Solution Strategy**: Use the formulas and their variations to analyze and calculate the problem.

**Example**: There is 50 grams of a 16% sugar solution. How much water needs to be added to dilute it to a 10% sugar solution?

**Solution**:
Using the formula:
50 × 16% ÷ 10% - 50 = 30 grams of water need to be added.

It should be noted that you should only return the thought template without any extra output.
        """
        self.distilled_thought = self.pipeline.get_respond(
            thought_distillation_prompt,
            self.problem_solution_pair,
        )

    def reasoner_instantiation(self) -> None:
        # Temporay using selection method to select answer extract method
        problem_id_list = [0, 1, 2]
        self.result = self.pipeline.get_respond(
            prompt_template=REASONER_PROMPT,
            query=f"""
{self.distilled_information}
User Input:
{self.user_input}
Thought template:
{self.thought_template}"""
        )
        if self.problem_id in problem_id_list:
            self.final_result, code_str = extract_and_execute_code(self.result)
            if self.need_check:
                self.count = 0
                prompt = INSPECTOR_PROMPT.format_messages(
                    user_input=self.user_input,
                    code=code_str,
                    result=self.final_result
                )
                self.inter_input = prompt[1].content
                self.inter_result = self.final_result
                while "An error occurred" in self.inter_result or self.inter_result in (
                    "",
                    "None",
                ):
                    self.inter_input = self.pipeline.get_respond(
                        self.inspector_prompt,
                        self.inter_input,
                    )
                    self.inter_result, inter_code_str = extract_and_execute_code(
                        self.inter_input,
                    )
                    self.inter_input = f"""
                User_input:{self.user_input}
                {inter_code_str}
                The result of code execution: {self.inter_result}
                """
                    self.count = self.count + 1
                    if self.count > MAX_RETRY_COUNT:
                        break
                self.final_result = self.inter_result
        else:
            self.final_result = self.result

    def bot_run(self) -> str:
        """
        Run the bot pipeline.

        Returns:
            Final result string

        """
        self.problem_distillation()
        self.buffer_retrieve()
        self.reasoner_instantiation()
        return self.final_result

    def bot_inference(self) -> None:
        self.problem_distillation()
        self.buffer_instantiation()
        self.buffer_manager()
