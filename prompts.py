from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Meta Distiller Prompt
META_DISTILLER_SYSTEM = """As a highly professional and intelligent expert in information distillation, you excel at extracting essential information to solve problems from user input queries. You adeptly transform this extracted information into a suitable format based on the respective type of the issue."""

META_DISTILLER_HUMAN = """Please analyze the following query and provide:

1. Key information: Extract all essential variables and data
2. Restrictions: Note any real-world rules (e.g. operator precedence, parentheses)
3. Distilled task: Core problem statement
4. Python transformation: Input parameter names and types (if applicable)
5. Expected answer format (if specific format required)

Query: {query}"""

META_DISTILLER_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(META_DISTILLER_SYSTEM),
    HumanMessagePromptTemplate.from_template(META_DISTILLER_HUMAN)
])

# Buffer Prompt
BUFFER_SYSTEM = """You are an expert in problem analysis and can apply previous problem-solving approaches to new issues."""

BUFFER_HUMAN = """Task Description: {task}
Meta Buffer Content: {meta_buffer}

Please:
1. Extract the most relevant thought template
2. Analyze the task
3. Generate a specific solution based on the template
4. Provide a clear, extractable final answer"""

BUFFER_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(BUFFER_SYSTEM),
    HumanMessagePromptTemplate.from_template(BUFFER_HUMAN)
])

# Reasoner Instantiation Prompt
REASONER_SYSTEM = """You are an expert in problem analysis and can apply previous problem-solving approaches to new issues. Your goal is to analyze tasks and generate specific solutions based on thought templates. If the solution involves Python code, provide only the code in a single code block. If not, provide a clear, extractable final answer."""

REASONER_HUMAN = """Distilled information:
{query}

Please analyze the above task description and thought template, and generate a specific, detailed solution. If the solution involves Python code, only provide the code. If not, provide a clear and extractable final answer."""

REASONER_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(REASONER_SYSTEM),
    HumanMessagePromptTemplate.from_template(REASONER_HUMAN)
])

# Inspector Prompt
INSPECTOR_SYSTEM = """You are an excellent Python programming master proficient in analyzing and editing code."""

INSPECTOR_HUMAN = """User Input: {user_input}
Code: {code}
Result: {result}

Please analyze and edit the code to ensure it:
1. Is syntactically correct
2. Solves the problem correctly
3. Follows best practices

Return only the edited code in a Python code block."""

INSPECTOR_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(INSPECTOR_SYSTEM),
    HumanMessagePromptTemplate.from_template(INSPECTOR_HUMAN)
])

# Similarity Check Prompt
SIMILARITY_CHECK_SYSTEM = """Analyze thought templates for fundamental differences in problem-solving approaches."""

SIMILARITY_CHECK_HUMAN = """Compare this thought template with the most similar one in MetaBuffer:

{thought_template}

Output "True" if there is a fundamental difference in approach.
Output "False" if the approaches are similar."""

SIMILARITY_CHECK_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SIMILARITY_CHECK_SYSTEM),
    HumanMessagePromptTemplate.from_template(SIMILARITY_CHECK_HUMAN)
])
