# agent/agent_executor.py

import os
import re
import sys
from typing import Union

from langchain.agents.agent import AgentOutputParser
from langchain.schema.agent import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from agent.tools.calculator import calculator

from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from llm.provider import GroqAgentLLM

tools = [
    Tool.from_function(
        func=calculator,
        name="Calculator",
        description="수학 계산을 처리합니다."
    )
]


class SafeFinalAnswerParser(AgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in text:
            return AgentFinish(
                return_values={"output": text.split("Final Answer:")[-1].strip()},
                log=text
            )

        # ✅ fallback: LLM이 Action: Final Answer 로 착각했을 경우도 처리
        if '"action": "Final Answer"' in text:
            match = re.search(r'"action_input":\s*"(.*?)"', text)
            if match:
                return AgentFinish(
                    return_values={"output": match.group(1)},
                    log=text
                )

        raise OutputParserException(f"파싱 실패: Final Answer 포맷이 아닙니다\n{text}")


# Groq 기반 LLM 사용
llm = GroqAgentLLM()

agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=False,
    agent_kwargs={
        "system_message": (
            "다음 규칙을 반드시 지키세요:\n"
            "1. Thought: 어떤 행동을 해야 하는지 작성\n"
            "2. Action: 반드시 'Calculator' 로만 작성\n"
            "3. Action Input: 반드시 수식만 (예: 3 * 400). '×', '곱하기' 금지\n"
            "4. Observation: 계산 결과\n"
            "5. Final Answer: 반드시 'Final Answer: 1200' 형식으로 마무리\n\n"
            "**절대 'Action: Final Answer' 형식은 쓰지 마세요!**\n"
            "**반드시 'Final Answer: ...' 형식으로 출력하세요!**"
        ),
        "output_parser": SafeFinalAnswerParser()
    }
)
