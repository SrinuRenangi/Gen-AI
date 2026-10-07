"""
Memory Management Lab: ConversationBufferMemory and Multi-Turn Context
=======================================================================
Zero to Hero Gen AI Course — Module 03: The LangChain Framework & Chaining

This standalone educational lab demonstrates the mechanics of conversational
memory in LLM applications:
1. The Stateless Amnesia Problem: Multi-turn failure without memory
2. ConversationBufferMemory: Preserving multi-turn context
3. The Critical Switch: return_messages=True vs return_messages=False
4. Quadratic Token Growth Simulation: Quantifying prompt costs across turns
5. Multi-Tenant Session Isolation: Simulating concurrent user sessions with LCEL

Usage:
    py memory_management_lab.py
"""

import sys
import os
import json
import time
from typing import Any, Callable, Dict, List, Optional

# Ensure UTF-8 output on Windows consoles to prevent cp1252 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# =====================================================================
# SECTION 1: STANDALONE MESSAGE REPRESENTATIONS
# =====================================================================

class BaseMessage:
    def __init__(self, content: str):
        self.content = content

    def __repr__(self):
        return f"{self.__class__.__name__}(content={self.content!r})"


class HumanMessage(BaseMessage):
    pass


class AIMessage(BaseMessage):
    pass


class SystemMessage(BaseMessage):
    pass


# =====================================================================
# SECTION 2: STANDALONE CONVERSATION BUFFER MEMORY
# =====================================================================

class ConversationBufferMemory:
    """
    Simulates LangChain's ConversationBufferMemory:
    Stores conversation history and injects it into prompt templates.
    """
    def __init__(self, memory_key: str = "history", return_messages: bool = False):
        self.memory_key = memory_key
        self.return_messages = return_messages
        self.messages: List[BaseMessage] = []

    def save_context(self, inputs: Dict[str, str], outputs: Dict[str, str]):
        """Saves a single human-AI interaction turn to memory."""
        # Find user input string
        user_input = next(iter(inputs.values())) if inputs else ""
        # Find AI output string
        ai_output = next(iter(outputs.values())) if outputs else ""

        self.messages.append(HumanMessage(content=str(user_input)))
        self.messages.append(AIMessage(content=str(ai_output)))

    def load_memory_variables(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Loads accumulated history in requested format."""
        if self.return_messages:
            return {self.memory_key: list(self.messages)}
        else:
            # Format as human-readable string transcript
            lines = []
            for msg in self.messages:
                prefix = "Human" if isinstance(msg, HumanMessage) else "AI"
                lines.append(f"{prefix}: {msg.content}")
            return {self.memory_key: "\n".join(lines)}

    def clear(self):
        """Clears all stored message history."""
        self.messages.clear()


# =====================================================================
# SECTION 3: INTELLIGENT SIMULATED CONVERSATIONAL LLM
# =====================================================================

class ConversationalLLM:
    """
    Intelligent simulated chat model with contextual recall capability.
    Supports live OpenAI API calls if OPENAI_API_KEY is configured.
    """
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.3):
        self.model = model
        self.temperature = temperature
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.live_client = None

        if self.api_key and not self.api_key.startswith("sk-mock"):
            try:
                from openai import OpenAI
                self.live_client = OpenAI(api_key=self.api_key)
            except Exception:
                self.live_client = None

    def invoke(self, messages_or_prompt: Any) -> str:
        if self.live_client:
            try:
                # Convert messages to OpenAI payload
                formatted = []
                if isinstance(messages_or_prompt, list):
                    for m in messages_or_prompt:
                        role = "user" if isinstance(m, HumanMessage) else ("assistant" if isinstance(m, AIMessage) else "system")
                        formatted.append({"role": role, "content": m.content})
                else:
                    formatted.append({"role": "user", "content": str(messages_or_prompt)})

                resp = self.live_client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=formatted
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"   [Notice: Falling back to simulated chat due to: {e}]")

        # Context-aware simulator
        text_context = ""
        if isinstance(messages_or_prompt, list):
            text_context = " ".join([m.content for m in messages_or_prompt]).lower()
        else:
            text_context = str(messages_or_prompt).lower()

        # Intelligent response generation based on presence of memory
        if "what is my name" in text_context or "who am i" in text_context:
            if "aris thorne" in text_context:
                return "You are Dr. Aris Thorne, the lead engineer on Project Odyssey."
            elif "alice" in text_context:
                return "You are Alice!"
            elif "bob" in text_context:
                return "You are Bob!"
            else:
                return "I do not have access to your name. You haven't mentioned it in our conversation."
        
        elif "what project do i lead" in text_context or "my project" in text_context:
            if "project odyssey" in text_context:
                return "You lead Project Odyssey, focused on deep space telemetry systems."
            else:
                return "I don't know what project you lead, as you haven't mentioned it yet."

        elif "favorite programming language" in text_context or "what language" in text_context:
            if "rust" in text_context:
                return "Your favorite programming language is Rust!"
            else:
                return "You haven't told me your favorite language yet."

        elif "dr. aris thorne" in text_context or "project odyssey" in text_context:
            return "Pleased to meet you, Dr. Thorne! Project Odyssey sounds fascinating. How can I assist you with telemetry today?"

        elif "rust" in text_context:
            return "Rust is fantastic! Its borrow checker provides memory safety without garbage collection overhead."

        else:
            return "Understood. How else can I assist your engineering workflow today?"


# =====================================================================
# LAB EXPERIMENTS & DEMONSTRATION SUITE
# =====================================================================

def banner(title: str):
    print("\n" + "#" * 72)
    print(f"##  {title}")
    print("#" * 72)


def experiment_1_stateless_vs_memory():
    banner("EXPERIMENT 1: The Stateless Amnesia Problem vs Memory-Enabled Chat")
    llm = ConversationalLLM()

    print("\n🔴 PART A: Without Memory (Stateless API Calls)")
    print("-" * 55)
    # Turn 1
    t1_input = "Hello! My name is Dr. Aris Thorne, and I lead Project Odyssey."
    t1_reply = llm.invoke([HumanMessage(content=t1_input)])
    print(f"User:  '{t1_input}'")
    print(f"Model: '{t1_reply}'")

    # Turn 2: Sent in isolation
    t2_input = "What is my name and what project do I lead?"
    t2_reply = llm.invoke([HumanMessage(content=t2_input)])
    print(f"\nUser:  '{t2_input}'")
    print(f"Model: '{t2_reply}'")
    print("❌ The model forgot everything because Turn 2 had zero historical context!")

    print("\n\n🟢 PART B: With ConversationBufferMemory")
    print("-" * 55)
    memory = ConversationBufferMemory(return_messages=True)

    # Turn 1
    memory.save_context({"input": t1_input}, {"output": t1_reply})

    # Turn 2: Compile prompt with memory
    history = memory.load_memory_variables({})["history"]
    full_prompt = history + [HumanMessage(content=t2_input)]
    t2_memory_reply = llm.invoke(full_prompt)

    print(f"User:  '{t2_input}'")
    print(f"Model: '{t2_memory_reply}'")
    print("✅ The model accurately answered because ConversationBufferMemory injected past context!")


def experiment_2_return_messages_comparison():
    banner("EXPERIMENT 2: Deep Inspection: return_messages=False vs return_messages=True")

    # Instance 1: Plain text string (for legacy completion models)
    mem_string = ConversationBufferMemory(return_messages=False)
    mem_string.save_context({"input": "Hello!"}, {"output": "Hi there!"})
    mem_string.save_context({"input": "How are you?"}, {"output": "I am an AI, doing great!"})

    res_str = mem_string.load_memory_variables({})
    print("1. When return_messages=False (Legacy Text Format):")
    print(f"   Python Type: {type(res_str['history'])}")
    print("   Formatted Value:\n" + "-" * 35)
    print(res_str["history"])
    print("-" * 35)

    # Instance 2: Typed Message objects (for modern ChatModels)
    mem_msgs = ConversationBufferMemory(return_messages=True)
    mem_msgs.save_context({"input": "Hello!"}, {"output": "Hi there!"})
    mem_msgs.save_context({"input": "How are you?"}, {"output": "I am an AI, doing great!"})

    res_msgs = mem_msgs.load_memory_variables({})
    print("\n2. When return_messages=True (Modern Chat Model Format):")
    print(f"   Python Type: {type(res_msgs['history'])}")
    print("   Message Objects List:")
    for m in res_msgs["history"]:
        print(f"   - {type(m).__name__}: {m.content!r}")
    print("\n💡 Key Insight: MessagesPlaceholder requires return_messages=True!")


def experiment_3_token_growth_simulation():
    banner("EXPERIMENT 3: Quadratic Token Growth Simulation (O(N^2))")

    avg_user_tokens = 50
    avg_ai_tokens = 150
    delta_turn = avg_user_tokens + avg_ai_tokens  # 200 tokens per turn
    cost_per_million = 0.15  # $0.15 per 1M prompt tokens (e.g. gpt-4o-mini)

    print(f"Simulation Parameters: {avg_user_tokens} user tokens + {avg_ai_tokens} AI tokens = {delta_turn} tokens/turn\n")
    print(f"{'Turn':<6} | {'Stored Memory':<15} | {'Prompt Tokens/Turn':<20} | {'Cumulative Billed':<18} | {'Est. Cost ($)':<12}")
    print("-" * 78)

    cumulative_prompt_tokens = 0
    for turn in range(1, 11):
        stored_tokens = (turn - 1) * delta_turn
        prompt_tokens_this_turn = stored_tokens + avg_user_tokens
        cumulative_prompt_tokens += prompt_tokens_this_turn
        cost = (cumulative_prompt_tokens / 1_000_000) * cost_per_million
        print(f"{turn:<6} | {stored_tokens:<15} | {prompt_tokens_this_turn:<20} | {cumulative_prompt_tokens:<18} | ${cost:<11.6f}")

    print("-" * 78)
    print("⚠️  Notice: Prompt tokens per turn grow linearly, causing CUMULATIVE costs to grow QUADRATICALLY (O(N^2))!")


def experiment_4_multi_tenant_sessions():
    banner("EXPERIMENT 4: Multi-Tenant Session Isolation (LCEL Pattern)")
    llm = ConversationalLLM()

    # Multi-tenant central session store dictionary
    central_session_store: Dict[str, ConversationBufferMemory] = {}

    def get_or_create_session(session_id: str) -> ConversationBufferMemory:
        if session_id not in central_session_store:
            central_session_store[session_id] = ConversationBufferMemory(return_messages=True)
        return central_session_store[session_id]

    def chat_in_session(session_id: str, user_text: str) -> str:
        mem = get_or_create_session(session_id)
        history = mem.load_memory_variables({})["history"]
        prompt = history + [HumanMessage(content=user_text)]
        reply = llm.invoke(prompt)
        mem.save_context({"input": user_text}, {"output": reply})
        return reply

    print("1. Alice starts session 'user_alice':")
    r1 = chat_in_session("user_alice", "Hello! My name is Alice and my favorite language is Rust.")
    print(f"   Alice: Hello! My name is Alice...")
    print(f"   Model: {r1}")

    print("\n2. Bob starts session 'user_bob' (isolated environment):")
    r2 = chat_in_session("user_bob", "Hello! My name is Bob and my favorite language is Go.")
    print(f"   Bob:   Hello! My name is Bob...")
    print(f"   Model: {r2}")

    print("\n3. Testing Cross-Session Isolation:")
    alice_check = chat_in_session("user_alice", "What is my favorite programming language?")
    print(f"   Alice Asks: 'What is my favorite programming language?'")
    print(f"   Model to Alice: '{alice_check}'")

    bob_check = chat_in_session("user_bob", "What is my favorite programming language?")
    print(f"   Bob Asks:   'What is my favorite programming language?'")
    print(f"   Model to Bob:   '{bob_check}'")

    print("\n✅ Perfect Isolation: Alice's memory and Bob's memory did not leak into each other!")


def experiment_5_manual_mutation_and_clear():
    banner("EXPERIMENT 5: Manual State Mutation, Pre-loading & Reset")
    mem = ConversationBufferMemory(return_messages=True)

    print("1. Pre-loading historical context manually:")
    mem.save_context(
        {"input": "System Notice: User subscribed to Enterprise Tier."},
        {"output": "Acknowledged. Enterprise privileges active."}
    )
    print(f"   Memory size: {len(mem.messages)} messages.")

    print("\n2. Resetting session via .clear():")
    mem.clear()
    print(f"   Memory size after clear(): {len(mem.messages)} messages.")
    print("   ✅ Session successfully wiped for privacy compliance.")


def main():
    print("""
========================================================================
   CONVERSATIONAL MEMORY LAB: ConversationBufferMemory & Multi-Turn State
========================================================================
    """)
    experiment_1_stateless_vs_memory()
    experiment_2_return_messages_comparison()
    experiment_3_token_growth_simulation()
    experiment_4_multi_tenant_sessions()
    experiment_5_manual_mutation_and_clear()
    print("\n✅ All 5 Memory Management experiments completed successfully!\n")


if __name__ == "__main__":
    main()
