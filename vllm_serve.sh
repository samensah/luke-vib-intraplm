#!/bin/bash
# Serve model with vLLM
export CUDA_VISIBLE_DEVICES=0

python -m vllm.entrypoints.openai.api_server \
    --model microsoft/Phi-4-mini-instruct \
    --port 8000 \
    --host 0.0.0.0


\section{Client Resolution Agent}
\label{sec:agent}

Given a user query $q$, we assume a client mention $m_0$ has been extracted upstream via standard named entity recognition or slot filling. The resolution task is to map $m_0$ to a client entity $c^* \in \mathcal{C}$, where $\mathcal{C} = \{c_1, c_2, \ldots, c_N\}$ denotes the client database containing $N$ entities. Each entity $c_i$ is associated with a canonical name and potentially multiple known aliases.

\subsection{Problem Formulation}

Client mention resolution is challenging because users refer to entities through diverse surface forms: abbreviations (\textit{JPMC} for JPMorgan Chase), acronyms, shorthand conventions (\textit{Mgmt.} for Management), and identifiers with minimal lexical overlap to canonical names. Mentions may also contain typographical errors, and multiple challenges frequently co-occur within a single mention. The task requires identifying:
\begin{equation}
    c^* = \argmax_{c \in \mathcal{C}} \, \text{sim}(m, c)
\end{equation}
where $\text{sim}(\cdot, \cdot)$ is a similarity function and $m$ is the (potentially refined) mention. In high-stakes domains, we require the top-ranked candidate to be correct, i.e., we optimize for hits@1.

\subsection{Fuzzy Matching Tool}

We employ a fuzzy matcher $\mathcal{F}$ as a retrieval tool that, given a mention string, returns a ranked list of candidate entities with associated similarity scores:
\begin{equation}
    \mathcal{F}(m) \rightarrow \{(c_{(1)}, s_{(1)}), (c_{(2)}, s_{(2)}), \ldots, (c_{(k)}, s_{(k)})\}
\end{equation}
where $s_{(i)} \in [0, 1]$ denotes the similarity score between mention $m$ and candidate $c_{(i)}$, and candidates are ordered such that $s_{(1)} \geq s_{(2)} \geq \cdots \geq s_{(k)}$. The fuzzy matcher combines multiple string similarity signals including token-based matching, edit distance, and phonetic similarity to handle the variety of surface form transformations encountered in practice.

\subsection{Two-Stage Resolution Pipeline}

Our system employs a two-stage approach to balance efficiency and accuracy:

\paragraph{Stage 1: Direct Matching.} We first query the fuzzy matcher with the original mention $m_0$. If an exact match is found ($s_{(1)} = 1.0$), we immediately return $c^* = c_{(1)}$, bypassing the agent. This handles the common case where users provide precise client names.

\paragraph{Stage 2: Agentic Refinement.} When no exact match exists ($s_{(1)} < 1.0$), we invoke a ReAct-based agent~\cite{yao2023react} that iteratively refines the mention and re-queries the fuzzy matcher until a satisfactory match is found or a termination condition is met.

\subsection{ReAct Agent Formulation}

The agent follows the ReAct paradigm, interleaving reasoning traces (\textit{Thought}) with actions (\textit{Action}) and observations (\textit{Observation}). Let $t \in \{0, 1, \ldots, T\}$ denote the iteration step. The agent maintains the following state:

\begin{itemize}
    \item $m_t$: the current mention string at step $t$
    \item $\mathcal{H}_t$: the trajectory history up to step $t$
    \item $o_t$: the observation (fuzzy matcher output) at step $t$
\end{itemize}

At each step $t$, the agent executes the following sequence:

\paragraph{Thought.} The agent reasons about the current state, analyzing the mention $m_t$, the candidate scores in $o_{t-1}$, and potential refinement strategies:
\begin{equation}
    \tau_t = \textsc{LLM}_\theta\left(\mathcal{P}, m_t, o_{t-1}, \mathcal{H}_{t-1}\right)
\end{equation}
where $\mathcal{P}$ is the system prompt encoding the task instructions and $\theta$ represents the LLM parameters.

\paragraph{Action.} Based on the reasoning trace, the agent decides on an action $a_t$. Actions fall into two categories:
\begin{equation}
    a_t = \begin{cases}
        \textsc{Refine}(m_t) \rightarrow m_{t+1} & \text{if continuing search} \\
        \textsc{Answer}(c_{(1)}) & \text{if terminating with result} \\
        \textsc{Clarify} & \text{if disambiguation required}
    \end{cases}
\end{equation}

The \textsc{Refine} action transforms the current mention into a refined version $m_{t+1}$. Crucially, the agent does not follow a predetermined sequence of refinement strategies. Instead, conditioned on the full context---including the original mention, current candidates, similarity scores, and prior refinements---the agent may:
\begin{itemize}
    \item Expand abbreviations (e.g., \textit{Mgmt} $\rightarrow$ \textit{Management})
    \item Correct apparent typographical errors
    \item Contract to acronyms (e.g., \textit{JPMorgan Chase} $\rightarrow$ \textit{JPMC})
    \item Try alternative surface forms or name orderings
    \item Combine multiple transformations simultaneously
\end{itemize}

\paragraph{Observation.} After a \textsc{Refine} action, the agent queries the fuzzy matcher with the refined mention:
\begin{equation}
    o_t = \mathcal{F}(m_{t+1})
\end{equation}

The observation provides the agent with updated candidate scores, enabling it to assess whether the refinement improved the match quality.

\subsection{Agent Trajectory}

The complete agent trajectory can be expressed as:
\begin{equation}
    \mathcal{H}_T = \left\{ (\tau_0, a_0, o_0), (\tau_1, a_1, o_1), \ldots, (\tau_T, a_T, o_T) \right\}
\end{equation}

At each step, the agent has access to the full history $\mathcal{H}_{t-1}$, allowing it to avoid repeating failed refinements and to reason about which transformation strategies remain unexplored.

The iterative process can be summarized as:
\begin{align}
    \tau_t &= \textsc{LLM}_\theta\left(\mathcal{P}, m_t, o_{t-1}, \mathcal{H}_{t-1}\right) \\
    a_t &= \textsc{LLM}_\theta\left(\tau_t\right) \\
    m_{t+1} &= \textsc{Apply}(a_t, m_t) \\
    o_t &= \mathcal{F}(m_{t+1})
\end{align}

\subsection{Termination Conditions}

The agent terminates under the following conditions:

\paragraph{High-Confidence Match.} If the top candidate achieves a similarity score above threshold $\theta_{\text{high}}$:
\begin{equation}
    s_{(1)} \geq \theta_{\text{high}}
\end{equation}
the agent returns $c^* = c_{(1)}$ as the resolved entity.

\paragraph{Maximum Iterations.} To bound computation, we enforce a maximum iteration limit:
\begin{equation}
    t \geq T_{\max}
\end{equation}
If reached without a high-confidence match, the agent returns the best available candidate $c_{(1)}$ from the final iteration, provided $s_{(1)} \geq \theta_{\text{min}}$ for a minimum acceptable threshold.

\paragraph{No Progress.} If successive refinements fail to improve the top score:
\begin{equation}
    s_{(1)}^{(t)} \leq s_{(1)}^{(t-1)} \quad \text{for } k \text{ consecutive steps}
\end{equation}
the agent may terminate early to avoid unproductive iterations.

\subsection{Disambiguation Protocol}

When the candidate set exhibits ambiguity, the agent defers to the user rather than risking incorrect resolution. Specifically, disambiguation is triggered when:

\paragraph{Multiple High-Scoring Candidates.} Two or more candidates exceed the confidence threshold:
\begin{equation}
    \left| \{ c_i : s_i \geq \theta_{\text{high}} \} \right| > 1
\end{equation}

\paragraph{Insufficient Score Gap.} The margin between top candidates is below a discrimination threshold:
\begin{equation}
    s_{(1)} - s_{(2)} < \delta
\end{equation}

In such cases, the agent presents the ambiguous candidates to the user with their associated metadata, requesting explicit selection. This ensures that resolution errors do not propagate to downstream analysis, which is critical in high-stakes applications where incorrect client identification carries regulatory and liability risks.

\subsection{Prompt Design}

The system prompt $\mathcal{P}$ encodes the task specification, available actions, and guidelines for effective refinement. Key components include:
\begin{itemize}
    \item Task description emphasizing the goal of maximizing similarity scores
    \item Instructions for interpreting fuzzy matcher output
    \item Examples of common surface form variations and effective refinements
    \item Guidelines for when to terminate versus continue refining
    \item Criteria for triggering user disambiguation
\end{itemize}

The prompt is designed to leverage the LLM's world knowledge about common abbreviations, corporate naming conventions, and typographical error patterns, while grounding decisions in the concrete similarity scores returned by the fuzzy matcher tool.
