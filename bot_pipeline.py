from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import transformers
from openai import OpenAI

from meta_buffer import MetaBuffer
from meta_buffer_utilis import extract_and_execute_code, meta_distiller_prompt
from test_templates import checkmate, game24, word_sorting


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
        self.api = False
        self.local = False
        self.base_url = base_url
        self.model_id = model_id
        if api_key is None:
            self.local = True
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=self.model_id,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )
        else:
            self.api = True
            self.api_key = api_key

    def get_respond(self, meta_prompt: str, user_prompt: str) -> str:
        """
        Get model response for given prompts.

        Args:
            meta_prompt: System/meta prompt
            user_prompt: User input prompt

        Returns:
            Model generated response

        """
        if self.api:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            completion = client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": meta_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return completion.choices[0].message.content
        messages = [
            {"role": "system", "content": meta_prompt},
            {"role": "user", "content": user_prompt},
        ]

        prompt = self.pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        outputs = self.pipeline(
            prompt,
            max_new_tokens=2048,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
        )
        return outputs[0]["generated_text"][len(prompt) :]


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
            meta_distiller_prompt,
            self.user_input,
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
        self.buffer_prompt = """
        You are an expert in problem analysis and can apply previous problem-solving approaches to new issues. The user will provide a specific task description and a meta buffer that holds multiple thought templates that will help to solve the problem. Your goal is to first extract most relevant thought template from meta buffer, analyze the user's task and generate a specific solution based on the thought template. Give a final answer that is easy to extract from the text.
        """
        prompt_text = self.buffer_prompt + self.distilled_information
        self.result = self.meta_buffer.retrieve_and_instantiate(prompt_text)

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
        self.instantiation_instruct = """
You are an expert in problem analysis and can apply previous problem-solving approaches to new issues. The user will provide a specific task description and a thought template. Your goal is to analyze the user's task and generate a specific solution based on the thought template. If the instantiated solution involves Python code, only provide the code and let the compiler handle it. If the solution does not involve code, provide a final answer that is easy to extract from the text.
It should be noted that all the python code should be within one code block, the answer should not include more than one code block! And strictly follow the thought-template to instantiate the python code but you should also adjust the input parameter according to the user input!
        """

        self.formated_input = f"""
Distilled information:
{self.distilled_information}
User Input:
{self.user_input}
Thought template:
{self.thought_template}

Instantiated Solution:
Please analyze the above user task description and thought template, and generate a specific, detailed solution. If the solution involves Python code, only provide the code. If not, provide a clear and extractable final answer.
        """
        self.inspector_prompt = """
You are an excellent python programming master who are proficient in analyzing and editing python code, and you are also good at understanding the real-world problem. Your task is:
1. Analyze the given python code
2. Edit the input code to make sure the edited code is correct and could run and solve the problem correctly.
Your respond should follow the format below:
```python
## Edited code here
```
        """
        self.result = self.pipeline.get_respond(
            self.instantiation_instruct,
            self.formated_input,
        )
        if self.problem_id in problem_id_list:
            self.final_result, code_str = extract_and_execute_code(self.result)
            if self.need_check:
                self.count = 0
                self.inter_input = f"""
                User_input:{self.user_input}
                {code_str}
                {self.final_result}
                """
                self.inter_result = self.final_result
                while (
                    "An error occurred" in self.inter_result or self.inter_result in ("", "None")
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
